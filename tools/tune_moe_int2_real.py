#!/usr/bin/env python3
"""Replay captured prefill routing through the real INT2 fused-MoE path.

PP13 tuned the INT2 kernel on uniform random token-to-expert ids and its
isolated wins regressed on the server, because real routing is skewed and large
tiles waste padding.  This tool replays the top-k ids recorded by
``patches/prefill_route_dump.py`` (``prouting_*.pt``) so candidate configs are
scored on the true per-expert token counts.

For each requested layer it compacts the real ids exactly as the server does
(``torch.unique`` + inverse), builds the production INT2 tensors at the compact
``E``, and times the production config against coordinate-descent neighbours.
The comparison is data-shape only; weight values do not affect kernel timing.

Run with the model server stopped (it needs the GPU)::

    /root/quant/venv-sglang/bin/python tools/tune_moe_int2_real.py \
        --route-file /root/quant/route_dump_pp15/prouting_00001.pt \
        --layers 0 12 24 36 47 --rounds 2

It writes a JSON report and prints per-layer production/best timings.  It never
writes into ``assets/``; promotion is a separate, explicit step.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

from tune_moe_int2 import Int2MoeBench, candidate_pairs, config_key, normalized


PRODUCTION_UP = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}
PRODUCTION_DOWN = dict(PRODUCTION_UP)


def compact_real_ids(ids):
    import torch

    ids = ids.to(torch.int64)
    flat = ids.reshape(-1)
    uniq, inverse = torch.unique(flat, return_inverse=True)
    k = int(uniq.numel())
    compact = inverse.reshape(ids.shape).to(torch.int32)
    counts = torch.bincount(flat, minlength=512).float()
    return compact, k, counts


def measure_pair(torch, bench, up, down):
    bench.x.copy_(bench.x_reference)
    out = bench.run(up, down)
    del out
    torch.cuda.synchronize()
    for _ in range(bench.warmup):
        bench.x.copy_(bench.x_reference)
        out = bench.run(up, down)
    del out
    torch.cuda.synchronize()
    samples = []
    for _ in range(bench.trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        bench.x.copy_(bench.x_reference)
        start.record()
        out = bench.run(up, down)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    del out
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24, 36, 47])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--intermediate-size", type=int, default=640)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument("--report", type=Path, default=Path("/root/quant/logs/moe_int2_real_tuning.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path("/root/sglang/python")))
    venv = Path(sys.prefix)
    cuda_home = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages/nvidia/cu13"
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ["PATH"] = os.pathsep.join([str(cuda_home / "bin"), str(venv / "bin"), os.environ.get("PATH", "")])

    import torch

    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(
        ServerArgs(model_path="/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang")
    )
    fm = importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe")

    route = torch.load(args.route_file, map_location="cpu", weights_only=True)
    by_layer = {int(lid): ids for lid, ids in route}
    report = {"route_file": str(args.route_file), "rounds": args.rounds, "layers": []}
    print(f"  {'layer':>5} {'M':>6} {'k':>5} {'prod ms':>9} {'best ms':>9} {'gain':>7}  best config")
    for lid in args.layers:
        if lid not in by_layer:
            continue
        ids = by_layer[lid]
        compact, k, counts = compact_real_ids(ids)
        m = int(ids.shape[0])
        bench = Int2MoeBench(
            torch=torch, fused_moe_module=fm, m=m, e=k,
            hidden=args.hidden_size, intermediate=args.intermediate_size,
            topk=args.topk, group_size=args.group_size,
            warmup=args.warmup, trials=args.trials, seed=7 + lid,
        )
        bench.ids = compact.to("cuda")
        weights = torch.rand((m, args.topk), device="cuda", dtype=torch.float32)
        bench.weights = weights / weights.sum(dim=1, keepdim=True)

        prod_up, prod_down = dict(PRODUCTION_UP), dict(PRODUCTION_DOWN)
        prod_ms = measure_pair(torch, bench, prod_up, prod_down)

        best_up, best_down, best_ms = prod_up, prod_down, prod_ms
        seeds = [(dict(prod_up), dict(prod_down))]
        tested = {config_key(prod_up), config_key(prod_down)}
        frontier = [seeds[0]]
        for _ in range(args.rounds):
            candidates = []
            for pair in frontier:
                for cand in candidate_pairs(pair):
                    key = (config_key(cand[0]), config_key(cand[1]))
                    if key in tested:
                        continue
                    tested.add(key)
                    candidates.append(cand)
            improved = False
            for cand in candidates:
                try:
                    ms = measure_pair(torch, bench, normalized(cand[0]), normalized(cand[1]))
                except Exception as exc:  # invalid launch configs are expected
                    print(f"    reject {config_key(cand[0])}: {type(exc).__name__}")
                    continue
                if ms < best_ms * 0.999:
                    best_ms, best_up, best_down = ms, dict(cand[0]), dict(cand[1])
                    improved = True
            if not improved:
                break
            frontier = [(dict(best_up), dict(best_down))]
        gain = prod_ms / best_ms if best_ms else math.inf
        print(
            f"  {lid:5d} {m:6d} {k:5d} {prod_ms:9.3f} {best_ms:9.3f} "
            f"{gain:6.3f}x  up={config_key(best_up)} down={config_key(best_down)}"
        )
        report["layers"].append(
            {
                "layer": lid, "m": m, "k": k,
                "production_ms": prod_ms, "best_ms": best_ms, "gain": gain,
                "production_up": prod_up, "production_down": prod_down,
                "best_up": best_up, "best_down": best_down,
                "max_expert_tokens": int(counts.max()),
                "nonzero_experts": int((counts > 0).sum()),
            }
        )
        del bench
        torch.cuda.empty_cache()

    gains = [item["gain"] for item in report["layers"]]
    report["mean_gain"] = sum(gains) / len(gains) if gains else 0.0
    report["min_gain"] = min(gains) if gains else 0.0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n  mean gain {report['mean_gain']:.3f}x, min {report['min_gain']:.3f}x")
    if report["min_gain"] < 1.0 + args.min_improvement:
        print("  REJECT: no config clears the per-layer improvement gate")
    else:
        print("  candidate clears the gate; run a server A/B before promoting")
    print(f"  report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
