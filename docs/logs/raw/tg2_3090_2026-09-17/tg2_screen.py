#!/usr/bin/env python3
"""TG2 cheap kernel screening for the pointer-table INT2 decode GEMV.

Benchmarks the ACTIVE kernel (`sglang.srt.layers.moe.expert_gemv`) on the real
checkpoint weights/scales for one MoE layer, at the production shapes:
  gate/up: N=1280 K=2560 top_k=10
  down:    N=2560 K=640  top_k=1, R=10 rows
Synthetic top-10 residency mixes put `cold` of the 10 experts on pinned host
memory and the rest on the device. Allocation/loading happen outside timing.

  python3 tg2_screen.py --out screen.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "/root/quant/sglang-main/python")
from sglang.srt.layers.moe.expert_gemv import moe_gemv_int2_tab  # noqa: E402

Q = os.environ.get("Q", "/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang")


def load_layer(layer: int, n_exp: int):
    idx = json.load(open(f"{Q}/model.safetensors.index.json"))["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts."
    hs = {}

    def get(k):
        f = idx[k]
        if f not in hs:
            hs[f] = safe_open(f"{Q}/{f}", "pt")
        return hs[f].get_tensor(k)

    w13, s13, w2, s2 = [], [], [], []
    for e in range(n_exp):
        w13.append(torch.cat([get(f"{pre}{e}.gate_proj.qweight"),
                              get(f"{pre}{e}.up_proj.qweight")], 1))
        s13.append(torch.cat([get(f"{pre}{e}.gate_proj.scales"),
                              get(f"{pre}{e}.up_proj.scales")], 1))
        w2.append(get(f"{pre}{e}.down_proj.qweight"))
        s2.append(get(f"{pre}{e}.down_proj.scales"))
    st = lambda L: torch.stack(L).contiguous()  # noqa: E731
    return st(w13), st(s13), st(w2), st(s2)


def mixed_tables(w, s, ids, cold):
    """ids: list of selected expert ids, length R. Last `cold` go to pinned host."""
    R = len(ids)
    hot = ids[: R - cold]
    cld = ids[R - cold:]
    wt = torch.empty(w.shape[0], dtype=torch.int64)
    st = torch.empty(s.shape[0], dtype=torch.int64)
    keep = []
    if hot:
        wh = w[torch.tensor(hot)].cuda().contiguous()
        sh = s[torch.tensor(hot)].cuda().contiguous()
        keep += [wh, sh]
        wt[torch.tensor(hot)] = wh.data_ptr() + torch.arange(len(hot)) * (wh.stride(0) * 4)
        st[torch.tensor(hot)] = sh.data_ptr() + torch.arange(len(hot)) * (sh.stride(0) * 2)
    if cld:
        wc = w[torch.tensor(cld)].pin_memory().contiguous()
        sc = s[torch.tensor(cld)].pin_memory().contiguous()
        keep += [wc, sc]
        wt[torch.tensor(cld)] = wc.data_ptr() + torch.arange(len(cld)) * (wc.stride(0) * 4)
        st[torch.tensor(cld)] = sc.data_ptr() + torch.arange(len(cld)) * (sc.stride(0) * 2)
    return wt.cuda(), st.cuda(), w.stride(1), s.stride(1), keep


def timeit(fn, warmup=5, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cold", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.manual_seed(0)

    w13, s13, w2, s2 = load_layer(a.layer, a.experts)
    E, KW13, N13 = w13.shape
    K13 = KW13 * 16
    _, KW2, N2 = w2.shape
    K2 = KW2 * 16
    inter = N13 // 2
    print(f"w13 [{E},{KW13},{N13}] K={K13}  w2 [{E},{KW2},{N2}] K={K2}  "
          f"scales {s13.dtype}  inter={inter}", flush=True)

    ids = list(range(a.topk))
    tw = torch.rand(a.topk)
    tw = (tw / tw.sum()).cuda()
    ids_d = torch.tensor(ids, dtype=torch.int32).cuda()
    x = torch.randn(1, K13, dtype=torch.bfloat16).cuda()
    h = torch.randn(a.topk, K2, dtype=torch.bfloat16).cuda()

    # warm the GPU
    for _ in range(50):
        _ = torch.mm(x, torch.randn(K13, N2, dtype=torch.bfloat16, device="cuda"))
    torch.cuda.synchronize()

    configs = [(bn, nw, 128) for bn, nw in itertools.product([32, 64, 128], [4, 8])]
    # best gate/up config gets an extra block_k=256 point (K13=2560 divides 256)
    results = []
    tables = {}
    for cold in a.cold:
        tables[cold] = {
            "gu": mixed_tables(w13, s13, ids, cold),
            "dn": mixed_tables(w2, s2, ids, cold),
        }
    for cold in a.cold:
        for proj in ("gu", "dn"):
            if proj == "gu":
                wt, st, sw, ss, keep = tables[cold]["gu"]
                N, K, top_k, mrw, av = N13, K13, a.topk, False, x
            else:
                wt, st, sw, ss, keep = tables[cold]["dn"]
                N, K, top_k, mrw, av = N2, K2, 1, True, h
            for bn, nw, bk in configs:
                if K % bk != 0:
                    continue
                c = torch.empty((a.topk, N), dtype=torch.bfloat16, device="cuda")

                def run():
                    return moe_gemv_int2_tab(
                        av, wt, st, ids_d, tw, N, K, sw, ss, top_k=top_k,
                        mul_routed_weight=mrw, block_n=bn, block_k=bk,
                        num_warps=nw, scale_bf16=False)

                try:
                    out = run()
                    torch.cuda.synchronize()
                    finite = bool(torch.isfinite(out.float()).all().item())
                    ms = timeit(run)
                    rec = {"cold": cold, "proj": proj, "block_n": bn,
                           "block_k": bk, "num_warps": nw, "ms": ms,
                           "us": ms * 1000, "finite": finite}
                except Exception as exc:  # noqa: BLE001
                    rec = {"cold": cold, "proj": proj, "block_n": bn,
                           "block_k": bk, "num_warps": nw,
                           "error": f"{type(exc).__name__}: {exc}"}
                results.append(rec)
                print(f"  cold={cold} {proj} bn={bn} bk={bk} warps={nw} "
                      f"{rec.get('us', rec.get('error'))}", flush=True)
        del tables[cold]

    payload = {"layer": a.layer, "experts": a.experts, "topk": a.topk,
               "cold": a.cold, "configs": configs, "results": results,
               "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    a.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
