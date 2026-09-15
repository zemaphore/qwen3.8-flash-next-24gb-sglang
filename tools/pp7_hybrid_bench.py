#!/usr/bin/env python3
"""PP7 sizing: hybrid cold-expert execution versus full staging, offline.

Replays real prefill routing (patches/prefill_route_dump.py artifacts) against
synthetic elastic placement at S=184: hot rows in one contiguous device arena,
cold rows in pinned host memory, both addressed through the production int64
address table.  Variants per (layer, chunk):

  base    stage every distinct expert, one fused-experts call (today's path)
  hyb,T   stage only experts that are resident or carry > T tokens; the low
          cold experts run through the pointer-table GEMV in place, their
          staged-path weights zeroed, results merged by an fp32 add

Reports median kernel time per variant, the staging-only cost, and the maximum
absolute output difference versus base (exactness audit: same tensors, same
values, only the summation route differs).

Run on the 3090 host with the server idle:
  SGLANG_MOE_CONFIG_DIR=/root/quant/assets/moe_configs \
    /root/quant/venv-sglang/bin/python3 tools/pp7_hybrid_bench.py \
      --dump /root/quant/route_dump_prefill --tokens 2048 --layers 0,1,2 --T 1 2 3 4
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics

import torch
import triton

from sglang.srt.layers.moe.expert_gemv import moe_gemv_int2_tab
from sglang.srt.layers.moe.expert_stream import _gather_rows_tab_kernel
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    fused_experts_impl,
)

E = 512
TOP_K = 10
HIDDEN = 2560
INTER = 640
GROUP = 128
NAMES = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
SHAPES = {
    "w13_qweight": (HIDDEN // 16, 2 * INTER),      # int32 words: [K/16, N]
    "w2_qweight": (INTER // 16, HIDDEN),           # int32 words
    "w13_scales": (HIDDEN // GROUP, 2 * INTER),    # bf16
    "w2_scales": (INTER // GROUP, HIDDEN),         # bf16
}
DTYPES = {
    "w13_qweight": torch.int32,
    "w2_qweight": torch.int32,
    "w13_scales": torch.bfloat16,
    "w2_scales": torch.bfloat16,
}
BLOCK = int(os.environ.get("SGLANG_MOE_GATHER_BLOCK", "2048"))


def _rb(name: str) -> int:
    n = 1
    for s in SHAPES[name]:
        n *= s
    return n * torch.empty(0, dtype=DTYPES[name]).element_size()


class Placement:
    """One synthetic elastic layer: arena (S rows, device) + host slots (E-S, pinned)."""

    def __init__(self, S: int, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.S = S
        self.rb = {name: _rb(name) for name in NAMES}
        self.addr: dict[str, torch.Tensor] = {}
        self.base: dict[str, int] = {}
        self.keep: list = []
        for name in NAMES:
            shape, dt = SHAPES[name], DTYPES[name]
            if dt == torch.int32:
                hot = torch.randint(-2**31, 2**31, (S,) + shape,
                                    generator=g, dtype=torch.int32).cuda()
                cold = torch.randint(-2**31, 2**31, (E - S,) + shape,
                                     generator=g, dtype=torch.int32).pin_memory()
            else:
                hot = (torch.rand((S,) + shape, generator=g) * 0.02).to(dt).cuda()
                cold = (torch.rand((E - S,) + shape, generator=g) * 0.02).to(dt).pin_memory()
            tab = torch.empty(E, dtype=torch.int64)
            tab[:S] = hot.data_ptr() + torch.arange(S, dtype=torch.int64) * self.rb[name]
            tab[S:] = cold.data_ptr() + torch.arange(E - S, dtype=torch.int64) * self.rb[name]
            self.addr[name] = tab.cuda()
            self.base[name] = hot.data_ptr()
            self.keep.append((hot, cold))

    def resident_mask(self, ids: torch.Tensor) -> torch.Tensor:
        a = self.addr["w13_qweight"].index_select(0, ids)
        return (a >= self.base["w13_qweight"]) & (
            a < self.base["w13_qweight"] + self.S * self.rb["w13_qweight"])

    def gather(self, name: str, ids: torch.Tensor, k: int) -> torch.Tensor:
        buf = torch.empty((k,) + SHAPES[name], dtype=DTYPES[name], device="cuda")
        _gather_rows_tab_kernel[(k, triton.cdiv(self.rb[name], BLOCK))](
            self.addr[name], ids, buf.view(torch.uint8), self.rb[name], BLOCK=BLOCK)
        return buf


def split(ids2d: torch.Tensor, pl: Placement, T: int):
    flat = ids2d.reshape(-1)
    uniq, inverse = torch.unique(flat, return_inverse=True)
    counts = torch.bincount(inverse)
    low = (~pl.resident_mask(uniq)) & (counts <= T)
    return flat, uniq, inverse, counts, ~low, low


def fused_stage(pl: Placement, x, ids2d, tw, uniq, k):
    w1 = pl.gather("w13_qweight", uniq, k)
    w2 = pl.gather("w2_qweight", uniq, k)
    s1 = pl.gather("w13_scales", uniq, k)
    s2 = pl.gather("w2_scales", uniq, k)
    return fused_experts_impl(
        x, w1, w2, tw, ids2d, use_int2_w2a16=True,
        w1_scale=s1, w2_scale=s2, block_shape=[0, GROUP], activation="silu",
        filter_expert=False)


def base_variant(pl: Placement, x: torch.Tensor, ids2d: torch.Tensor,
                 tw: torch.Tensor) -> torch.Tensor:
    _, uniq, inverse, *_ = split(ids2d, pl, 0)
    k = int(uniq.numel())
    return fused_stage(pl, x, inverse.reshape(ids2d.shape).to(ids2d.dtype), tw, uniq, k)


def gather_only(pl: Placement, ids2d: torch.Tensor) -> None:
    _, uniq, *_ = split(ids2d, pl, 0)
    k = int(uniq.numel())
    for name in NAMES:
        pl.gather(name, uniq, k)


def hybrid_variant(pl: Placement, x: torch.Tensor, ids2d: torch.Tensor,
                   tw: torch.Tensor, T: int):
    M = ids2d.shape[0]
    flat, uniq, inverse, counts, hi, low = split(ids2d, pl, T)
    pos = inverse.reshape(-1)
    low_pair = low.index_select(0, pos)                 # [M*top_k] bool
    hi_uniq = uniq[hi]
    k2 = int(hi_uniq.numel())
    rank2 = (hi.cumsum(0) - 1).to(torch.int32).clamp(min=0)
    mapped = rank2.index_select(0, pos)
    new_ids = torch.where(low_pair, torch.zeros_like(mapped), mapped)
    new_ids = new_ids.reshape(M, TOP_K).to(torch.int32)
    new_w = torch.where(low_pair.reshape(M, TOP_K),
                        torch.zeros_like(tw), tw)
    staged = fused_stage(pl, x, new_ids, new_w, hi_uniq, k2)
    sel = low_pair.nonzero().reshape(-1)
    out = staged
    if sel.numel() > 0:
        rows = torch.div(sel, TOP_K, rounding_mode="floor")
        ids_sel = flat.index_select(0, sel)
        w_sel = tw.reshape(-1).index_select(0, sel).to(torch.float32)
        xs = x.index_select(0, rows)
        c13 = moe_gemv_int2_tab(
            xs, pl.addr["w13_qweight"], pl.addr["w13_scales"],
            ids_sel, w_sel, 2 * INTER, HIDDEN,
            SHAPES["w13_qweight"][1], SHAPES["w13_scales"][1],
            top_k=1, mul_routed_weight=False, scale_bf16=True)
        h = torch.nn.functional.silu(c13[:, :INTER]) * c13[:, INTER:]
        c2 = moe_gemv_int2_tab(
            h, pl.addr["w2_qweight"], pl.addr["w2_scales"],
            ids_sel, w_sel, HIDDEN, INTER,
            SHAPES["w2_qweight"][1], SHAPES["w2_scales"][1],
            top_k=1, mul_routed_weight=True, scale_bf16=True)
        acc = torch.zeros((M, HIDDEN), dtype=torch.float32, device=x.device)
        acc.index_add_(0, rows, c2.float())
        out = (staged.float() + acc).to(x.dtype)
    return out, k2, int(sel.numel())


def timeit(fn, reps: int) -> float:
    fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(reps):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        vals.append(s.elapsed_time(e))
    return statistics.median(vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="/root/quant/route_dump_prefill")
    ap.add_argument("--model",
                    default="/mnt/ai_models/"
                            "Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--layers", default="0,1,2")
    ap.add_argument("--T", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--S", type=int, default=184)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--max-chunks", type=int, default=2)
    args = ap.parse_args()

    from sglang.srt.server_args import (ServerArgs,
                                        set_global_server_args_for_scheduler)
    set_global_server_args_for_scheduler(ServerArgs(model_path=args.model))
    if not torch.distributed.is_initialized():
        from sglang.srt.distributed import (init_distributed_environment,
                                            initialize_model_parallel)
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29517", backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=1)
        torch.cuda.set_device(0)

    layers = [int(v) for v in args.layers.split(",")]
    chunks = []
    for f in sorted(glob.glob(os.path.join(args.dump, "prouting_*.pt"))):
        d = torch.load(f, map_location="cpu")
        if d[0][1].shape[0] == args.tokens:
            chunks.append({lid: ids.cuda() for lid, ids in d})
        if len(chunks) == args.max_chunks:
            break
    if not chunks:
        raise SystemExit(f"no chunk with M={args.tokens} in {args.dump}")
    print(f"chunks={len(chunks)} M={args.tokens} S={args.S} "
          f"gather BLOCK={BLOCK} device={torch.cuda.get_device_name()}")

    tot = {"base": 0.0, "gather": 0.0}
    for t in args.T:
        tot[f"hyb{t}"] = 0.0
    print(f"{'layer':>5} {'chunk':>5} {'rows':>5} {'base ms':>8} {'gath ms':>8}"
          + "".join(f" | hyb{T:>4} k2    pairs maxerr" for T in args.T))
    for lid in layers:
        for ci, chunk in enumerate(chunks):
            ids2d = chunk[lid].to(torch.int32)
            M = ids2d.shape[0]
            x = (torch.randn((M, HIDDEN), dtype=torch.float32,
                             device="cpu") * 0.5).to(torch.bfloat16).cuda()
            tw = torch.softmax(torch.randn((M, TOP_K), device="cuda"), dim=-1)
            pl = Placement(args.S, seed=1000 + lid)
            _, u, _, _, _, _ = split(ids2d, pl, 0)
            base_ms = timeit(lambda: base_variant(pl, x, ids2d, tw), args.reps)
            gath_ms = timeit(lambda: gather_only(pl, ids2d), args.reps)
            ref = base_variant(pl, x, ids2d, tw).float()
            tot["base"] += base_ms
            tot["gather"] += gath_ms
            cells = []
            for T in args.T:
                hyb_ms = timeit(
                    lambda: hybrid_variant(pl, x, ids2d, tw, T)[0], args.reps)
                out, k2, npairs = hybrid_variant(pl, x, ids2d, tw, T)
                err = (out.float() - ref).abs().max().item()
                tot[f"hyb{T}"] += hyb_ms
                cells.append(f" | {hyb_ms:7.1f} {k2:4d} {npairs:5d} {err:7.4f}")
            print(f"{lid:>5} {ci:>5} {int(u.numel()):>5} {base_ms:8.1f} "
                  f"{gath_ms:8.1f}" + "".join(cells))
            del pl, x, tw, ids2d, ref
            torch.cuda.empty_cache()

    n = len(layers) * len(chunks)
    print(f"\ntotals over {n} layer-chunks: base {tot['base']:.1f} ms "
          f"(gather {tot['gather']:.1f} ms)")
    for T in args.T:
        d = tot[f"hyb{T}"] - tot["base"]
        print(f"  T={T}: {tot[f'hyb{T}']:.1f} ms ({d:+.1f} ms, "
              f"{100 * d / tot['base']:+.1f}%)  projected per request "
              f"(48L x (2x2048 + tail)): {d / n * 48 * 2.23:.0f} ms")


if __name__ == "__main__":
    main()
