#!/usr/bin/env python3
"""Tune the patched INT2 MoE prefill kernel on one CUDA GPU.

The upstream SGLang tuner does not construct the repository's symmetric
``int2_w2a16`` N-contiguous weight layout and can only override one config for
both MoE projections.  This tuner constructs the serving shapes directly and
times independent gate/up and down configs through the real patched
``fused_experts_impl`` path.

Examples (run with the server stopped)::

    /root/quant/venv-sglang/bin/python tools/tune_moe_int2.py \
        --sglang /root/sglang --expert-counts 128 256 384 512 \
        --batch-sizes 128 512 1024 --rounds 2

    # Fast first pass over the dominant 1024-token prefill chunk
    /root/quant/venv-sglang/bin/python tools/tune_moe_int2.py \
        --sglang /root/sglang --expert-counts 256 --batch-sizes 1024 --rounds 1

Output files use SGLang's exact lookup names under
``assets/moe_configs/configs/triton_<version>/``. Existing files are merged so
separate runs can fill more (E, M) points.  The output JSON report records every
accepted config and timing.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterator


Config = dict[str, int]
ConfigPair = tuple[Config, Config]

STANDARD_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_EXPERT_COUNTS = [128, 256, 384, 512]
DEFAULT_BATCH_SIZES = [128, 512, 1024]


def _config(
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    warps: int,
    stages: int,
) -> Config:
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": warps,
        "num_stages": stages,
    }


def generic_config(m: int, e: int) -> Config:
    """Materialize the patched SGLang fallback, including Triton defaults."""
    if m <= e:
        return _config(16, 32, 64, 1, 4, 3)
    return _config(64, 64, 32, 8, 4, 3)


def normalized(config: Config) -> Config:
    out = dict(config)
    out.setdefault("num_warps", 4)
    out.setdefault("num_stages", 3)
    return out


def config_key(config: Config) -> tuple[int, ...]:
    return tuple(
        config[key]
        for key in (
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
        )
    )


def pair_key(pair: ConfigPair) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return config_key(pair[0]), config_key(pair[1])


def candidate_pairs(best: ConfigPair) -> list[ConfigPair]:
    """One coordinate-descent neighborhood around an up/down config pair."""
    up, down = best
    pairs: list[ConfigPair] = []

    # Both projections share one expert sort, hence one BLOCK_SIZE_M.
    for value in (16, 32, 64, 128):
        u, d = dict(up), dict(down)
        u["BLOCK_SIZE_M"] = d["BLOCK_SIZE_M"] = value
        pairs.append((u, d))

    values = {
        "BLOCK_SIZE_N": (16, 32, 64, 128),
        # K=640 in the down projection rules out values above 128.
        "BLOCK_SIZE_K": (32, 64, 128),
        "GROUP_SIZE_M": (1, 4, 8, 16, 32),
        "num_warps": (4, 8),
        "num_stages": (2, 3, 4),
    }
    for side in (0, 1):
        for field, choices in values.items():
            for value in choices:
                u, d = dict(up), dict(down)
                (u if side == 0 else d)[field] = value
                pairs.append((u, d))

    unique: dict[tuple[tuple[int, ...], tuple[int, ...]], ConfigPair] = {}
    for pair in pairs:
        unique[pair_key(pair)] = pair
    return list(unique.values())


class Int2MoeBench:
    def __init__(
        self,
        *,
        torch: Any,
        fused_moe_module: Any,
        m: int,
        e: int,
        hidden: int,
        intermediate: int,
        topk: int,
        group_size: int,
        warmup: int,
        trials: int,
        seed: int,
    ) -> None:
        self.torch = torch
        self.fm = fused_moe_module
        self.original_resolver = fused_moe_module.try_get_optimal_moe_config
        self.m = m
        self.e = e
        self.warmup = warmup
        self.trials = trials
        self.group_size = group_size

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.x = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16)
        self.x_reference = self.x.clone()

        # Actual serving layout after to_word_ncontig(): [E, K/16, N].
        self.w1 = torch.randint(
            -(2**31),
            2**31 - 1,
            (e, hidden // 16, 2 * intermediate),
            device="cuda",
            dtype=torch.int32,
        )
        self.w2 = torch.randint(
            -(2**31),
            2**31 - 1,
            (e, intermediate // 16, hidden),
            device="cuda",
            dtype=torch.int32,
        )
        self.s1 = (
            torch.rand(
                (e, hidden // group_size, 2 * intermediate),
                device="cuda",
                dtype=torch.float32,
            )
            * 0.02
        ).to(torch.bfloat16)
        self.s2 = (
            torch.rand(
                (e, intermediate // group_size, hidden),
                device="cuda",
                dtype=torch.float32,
            )
            * 0.02
        ).to(torch.bfloat16)

        ids = torch.randint(0, e, (m * topk,), device="cuda", dtype=torch.int32)
        if ids.numel() >= e:
            ids[:e] = torch.arange(e, device="cuda", dtype=torch.int32)
        self.ids = ids.reshape(m, topk)
        weights = torch.rand((m, topk), device="cuda", dtype=torch.float32)
        self.weights = weights / weights.sum(dim=1, keepdim=True)

    @contextlib.contextmanager
    def configs(self, up: Config, down: Config) -> Iterator[None]:
        up_copy = normalized(up)
        down_copy = normalized(down)
        down_copy["BLOCK_SIZE_M"] = up_copy["BLOCK_SIZE_M"]

        def resolve(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("return_down_config", False):
                return dict(up_copy), (dict(down_copy), down_copy["BLOCK_SIZE_M"])
            return dict(up_copy)

        self.fm.try_get_optimal_moe_config = resolve
        try:
            yield
        finally:
            self.fm.try_get_optimal_moe_config = self.original_resolver

    def run(self, up: Config, down: Config) -> Any:
        with self.configs(up, down):
            return self.fm.fused_experts_impl(
                self.x,
                self.w1,
                self.w2,
                self.weights,
                self.ids,
                # Avoid SGLang's distributed symmetric-allocation path; the
                # production runner is also in-place. Callers restore x before
                # every invocation so candidates see identical inputs.
                inplace=True,
                use_int2_w2a16=True,
                w1_scale=self.s1,
                w2_scale=self.s2,
                block_shape=[0, self.group_size],
            )

    def measure(self, up: Config, down: Config) -> float:
        torch = self.torch
        # First call compiles any new Triton specialization.
        self.x.copy_(self.x_reference)
        out = self.run(up, down)
        del out
        torch.cuda.synchronize()
        for _ in range(self.warmup):
            self.x.copy_(self.x_reference)
            out = self.run(up, down)
        del out
        torch.cuda.synchronize()

        samples: list[float] = []
        for _ in range(self.trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self.x.copy_(self.x_reference)
            start.record()
            out = self.run(up, down)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        del out
        return statistics.median(samples)

    def verify(self, reference: ConfigPair, candidate: ConfigPair) -> tuple[float, float]:
        torch = self.torch
        self.x.copy_(self.x_reference)
        ref = self.run(*reference).float()
        self.x.copy_(self.x_reference)
        got = self.run(*candidate).float()
        delta = (got - ref).abs()
        rel = (got - ref).norm() / ref.norm().clamp(min=1e-12)
        return float(rel.item()), float(delta.max().item())


def tune_one(
    bench: Int2MoeBench,
    seeds: list[ConfigPair],
    rounds: int,
) -> tuple[ConfigPair, float, list[dict[str, Any]]]:
    tested: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    trace: list[dict[str, Any]] = []

    def test(pair: ConfigPair) -> float:
        key = pair_key(pair)
        if key in tested:
            return tested[key]
        try:
            elapsed = bench.measure(*pair)
        except Exception as exc:  # invalid launch/resource combinations are expected
            print(f"  reject {key}: {type(exc).__name__}: {str(exc)[:180]}", flush=True)
            elapsed = math.inf
        tested[key] = elapsed
        trace.append({"up": pair[0], "down": pair[1], "milliseconds": elapsed})
        if math.isfinite(elapsed):
            print(f"  {elapsed:8.3f} ms  up={config_key(pair[0])} down={config_key(pair[1])}", flush=True)
        return elapsed

    best = min(seeds, key=test)
    best_time = test(best)
    for round_index in range(rounds):
        print(f"  coordinate round {round_index + 1}/{rounds}", flush=True)
        neighbors = candidate_pairs(best)
        candidate = min(neighbors, key=test)
        candidate_time = test(candidate)
        if candidate_time >= best_time * 0.999:
            break
        best, best_time = candidate, candidate_time
    return best, best_time, trace


def load_json(path: Path) -> dict[str, Config]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, sort_keys=False)
        handle.write("\n")
    temporary.replace(path)


def closest_config(configs: dict[str, Config], m: int, fallback: Config) -> Config:
    if not configs:
        return fallback
    key = min(configs, key=lambda value: abs(int(value) - m))
    return normalized(configs[key])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang", type=Path, default=Path("/root/sglang"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/mnt/ai_models/"
            "Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang"
        ),
        help="Local model config used to initialize SGLang runtime settings",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expert-counts", type=int, nargs="+", default=DEFAULT_EXPERT_COUNTS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--intermediate-size", type=int, default=640)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.01,
        help="Keep a tuned pair only when it beats the fallback by this factor",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=Path, default=Path("/root/quant/logs/m4_int2_tuning.json"))
    args = parser.parse_args()

    source = args.sglang / "python"
    if not source.is_dir():
        parser.error(f"SGLang Python source not found: {source}")
    sys.path.insert(0, str(source))

    # Match serve-3090.sh: the CUDA toolkit is supplied by the venv wheels,
    # and SGLang's activation JIT also needs the venv's ninja on PATH.
    venv = Path(sys.prefix)
    cuda_home = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cu13"
    )
    if not (cuda_home / "bin" / "nvcc").is_file():
        parser.error(f"Venv CUDA toolkit not found: {cuda_home}")
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["PATH"] = os.pathsep.join(
        [str(cuda_home / "bin"), str(venv / "bin"), os.environ.get("PATH", "")]
    )

    import torch
    import triton
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (8, 6):
        parser.error(f"Expected sm_86 RTX 3090, found {props.name} sm_{props.major}{props.minor}")

    if not (args.model / "config.json").is_file():
        parser.error(f"Local model config not found: {args.model / 'config.json'}")
    set_global_server_args_for_scheduler(ServerArgs(model_path=str(args.model)))
    fm = importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    )

    repo = Path(__file__).resolve().parents[1]
    version_dir = f"triton_{triton.__version__.replace('.', '_')}"
    output_dir = args.output_dir or repo / "assets" / "moe_configs" / "configs" / version_dir
    device_name = props.name.replace(" ", "_")
    blackwell_dir = repo / "assets" / "moe_configs" / "configs" / version_dir
    blackwell_up = load_json(
        blackwell_dir
        / "E=512,N=160,device_name=NVIDIA_RTX_PRO_4000_Blackwell,dtype=int2_w2a16.json"
    )
    blackwell_down = load_json(
        blackwell_dir
        / "E=512,N=160,device_name=NVIDIA_RTX_PRO_4000_Blackwell,dtype=int2_w2a16_down.json"
    )

    report: dict[str, Any] = {
        "device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "shape": {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "topk": args.topk,
            "group_size": args.group_size,
        },
        "runs": [],
    }

    for e in args.expert_counts:
        up_name = f"E={e},N=160,device_name={device_name},dtype=int2_w2a16.json"
        down_name = f"E={e},N=160,device_name={device_name},dtype=int2_w2a16_down.json"
        up_path, down_path = output_dir / up_name, output_dir / down_name
        up_map, down_map = load_json(up_path), load_json(down_path)

        for m in args.batch_sizes:
            print(f"E={e} M={m}", flush=True)
            fallback = generic_config(m, e)
            seed_up = closest_config(up_map or blackwell_up, m, fallback)
            seed_down = closest_config(down_map or blackwell_down, m, fallback)
            seeds = [
                (dict(fallback), dict(fallback)),
                (dict(seed_up), dict(seed_down)),
            ]
            bench = Int2MoeBench(
                torch=torch,
                fused_moe_module=fm,
                m=m,
                e=e,
                hidden=args.hidden_size,
                intermediate=args.intermediate_size,
                topk=args.topk,
                group_size=args.group_size,
                warmup=args.warmup,
                trials=args.trials,
                seed=args.seed + e + m,
            )
            baseline_time = bench.measure(fallback, fallback)
            best, best_time, trace = tune_one(bench, seeds, args.rounds)
            candidate_time = best_time
            accepted = (
                math.isfinite(candidate_time)
                and baseline_time / candidate_time >= args.min_speedup
            )
            if not accepted:
                best = (dict(fallback), dict(fallback))
                best_time = baseline_time
            rel, max_abs = bench.verify((fallback, fallback), best)
            improvement = baseline_time / best_time if best_time else math.inf
            print(
                f"  BEST {best_time:.3f} ms, fallback {baseline_time:.3f} ms, "
                f"{improvement:.3f}x, accepted={accepted}, "
                f"rel={rel:.3e}, max={max_abs:.3e}",
                flush=True,
            )
            if not math.isfinite(best_time) or rel > 2e-2:
                raise RuntimeError(
                    f"E={e} M={m}: no valid numerically equivalent config (rel={rel})"
                )
            up_map[str(m)], down_map[str(m)] = best
            save_json(up_path, {key: up_map[key] for key in sorted(up_map, key=int)})
            save_json(down_path, {key: down_map[key] for key in sorted(down_map, key=int)})
            report["runs"].append(
                {
                    "experts": e,
                    "tokens": m,
                    "fallback_ms": baseline_time,
                    "candidate_ms": candidate_time,
                    "best_ms": best_time,
                    "speedup": improvement,
                    "accepted_tuned_config": accepted,
                    "relative_error": rel,
                    "max_abs_error": max_abs,
                    "up": best[0],
                    "down": best[1],
                    "trace": trace,
                }
            )
            save_json(args.report, report)
            del bench
            torch.cuda.empty_cache()

    print(f"Wrote configs to {output_dir}")
    print(f"Wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
