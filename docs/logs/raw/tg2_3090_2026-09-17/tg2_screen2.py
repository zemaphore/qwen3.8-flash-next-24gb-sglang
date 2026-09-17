#!/usr/bin/env python3
"""TG2 focused re-bench: repeated, interleaved timings over the real cold-count
range, plus identical-input output comparison against the unchanged kernel.

Cold-count histogram from the offline code route dumps vs the exact top-184
presence asset (weight per (layer,token)):
  cold=0 8.4%, 1 13.9%, 2 15.5%, 3 15.1%, 4 13.8%, 5 11.4%, 6 8.8%, ...
  mean 3.56
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tg2_screen import load_layer, mixed_tables  # noqa: E402

sys.path.insert(0, "/root/quant/sglang-main/python")
from sglang.srt.layers.moe.expert_gemv import moe_gemv_int2_tab  # noqa: E402

COLD = [0, 2, 3, 4, 5, 6]
# fixed histogram over the measured cold counts (renormalized)
HIST = {0: 84, 2: 155, 3: 151, 4: 138, 5: 114, 6: 88}
UNCHANGED = (64, 4, 128)
CONFIGS = [(64, 4, 128), (128, 8, 128), (128, 4, 128), (32, 4, 128)]
GU_EXTRA = [(128, 8, 256), (64, 4, 256)]


def timed(fn, warmup=5, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    w13, s13, w2, s2 = load_layer(a.layer, a.experts)
    E, KW13, N13 = w13.shape
    K13 = KW13 * 16
    _, KW2, N2 = w2.shape
    K2 = KW2 * 16
    inter = N13 // 2
    ids = list(range(10))
    tw = torch.rand(10)
    tw = (tw / tw.sum()).cuda()
    ids_d = torch.tensor(ids, dtype=torch.int32).cuda()
    x = torch.randn(1, K13, dtype=torch.bfloat16).cuda()
    h = torch.randn(10, K2, dtype=torch.bfloat16).cuda()

    # inputs hold fixed across the whole run
    xt = x.clone()
    ht = h.clone()

    payload = {"cold": COLD, "hist": HIST, "configs": CONFIGS,
               "gu_extra": GU_EXTRA, "rounds": a.rounds, "cells": {}}
    for cold in COLD:
        wt13, st13, sw13, ss13, k13 = mixed_tables(w13, s13, ids, cold)
        wt2, st2, sw2, ss2, k2 = mixed_tables(w2, s2, ids, cold)
        for proj in ("gu", "dn"):
            if proj == "gu":
                wt, st, sw, ss = wt13, st13, sw13, ss13
                N, K, tk, mrw, av = N13, K13, 10, False, xt
            else:
                wt, st, sw, ss = wt2, st2, sw2, ss2
                N, K, tk, mrw, av = N2, K2, 1, True, ht
            cfgs = list(CONFIGS) + (GU_EXTRA if proj == "gu" else [])
            ref = None
            # numeric reference from the unchanged config
            def mk(bn, nw, bk):
                return lambda: moe_gemv_int2_tab(
                    av, wt, st, ids_d, tw, N, K, sw, ss, top_k=tk,
                    mul_routed_weight=mrw, block_n=bn, block_k=bk,
                    num_warps=nw, scale_bf16=False)
            ref = mk(*UNCHANGED)().float().cpu()
            for cfg in cfgs:
                if K % cfg[2] != 0:
                    continue
                fn = mk(*cfg)
                out = fn().float().cpu()
                diff = (out - ref).abs()
                rel = (diff.max().item() / max(ref.abs().max().item(), 1e-9))
                rounds = [timed(fn) for _ in range(a.rounds)]
                cell = {"cold": cold, "proj": proj, "cfg": list(cfg),
                        "us": rounds, "us_median": statistics.median(rounds),
                        "us_mean": sum(rounds) / len(rounds),
                        "max_abs_diff": diff.max().item(),
                        "mean_abs_diff": diff.mean().item(),
                        "rel_max": rel,
                        "finite": bool(torch.isfinite(out).all().item())}
                payload["cells"][f"{proj}-c{cold}-{cfg}"] = cell
                print(f"  c{cold} {proj} bn={cfg[0]} bk={cfg[2]} w={cfg[1]} "
                      f"med={statistics.median(rounds):.1f}us runs={[round(x,1) for x in rounds]} "
                      f"maxdiff={diff.max().item():.2e}", flush=True)
            del wt, st, sw, ss, ref
        del wt13, st13, sw13, ss13, wt2, st2, sw2, ss2

    # weighted means over the histogram
    print("\n== weighted-mean us (cold histogram) ==")
    for proj in ("gu", "dn"):
        cfgs = list(CONFIGS) + (GU_EXTRA if proj == "gu" else [])
        for cfg in cfgs:
            if any(f"{proj}-c{c}-{cfg}" not in payload["cells"] for c in COLD if c in HIST):
                continue
            wsum = sum(HIST[c] for c in HIST if f"{proj}-c{c}-{cfg}" in payload["cells"])
            val = sum(HIST[c] * payload["cells"][f"{proj}-c{c}-{cfg}"]["us_median"]
                      for c in HIST if f"{proj}-c{c}-{cfg}" in payload["cells"]) / wsum
            payload.setdefault("weighted_us", {})[f"{proj}-{cfg}"] = val
            print(f"  {proj} bn={cfg[0]} bk={cfg[2]} w={cfg[1]}: {val:.1f} us")
    a.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
