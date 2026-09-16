#!/usr/bin/env python3
"""PP13: real-tensor format sweep for the dominant GDN projections.

The current 4,608-token profile attributes most non-MoE Marlin time to the 36
linear-attention layers.  This tool reconstructs layer 0's merged W8A16
``in_proj_qkvz`` and its W8A16 ``out_proj`` from the checkpoint, then compares:

* production BF16-activation GPTQ-Marlin;
* FP16-activation GPTQ-Marlin, both kernel-only and with BF16<->FP16 casts;
* dequantized BF16 and FP16 cuBLAS, with casts included for FP16; and
* W8A8 cuBLAS with activation quantization and output scaling included.

The one-time weight conversion time and incremental persistent bytes are also
reported.  Run only with the model server stopped:

  CU=/root/quant/venv-sglang/lib/python3.12/site-packages/nvidia/cu13
  PATH="$CU/bin:/root/quant/venv-sglang/bin:$PATH" CUDA_HOME="$CU" \
    PYTHONPATH=/root/sglang/python /root/quant/venv-sglang/bin/python \
    tools/pp13_gdn_format_bench.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "/root/sglang/python")

from sglang.kernels.ops.quantization.gptq_marlin import (  # noqa: E402
    gptq_marlin_gemm,
)
from sglang.kernels.ops.quantization.gptq_marlin_repack import (  # noqa: E402
    gptq_marlin_repack,
)
from sglang.srt.layers.quantization.marlin_utils import (  # noqa: E402
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.utils import scalar_types  # noqa: E402


MODEL = Path(
    "/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang"
)
LAYER = "model.language_model.layers.0.linear_attn"
BITS = 8
GROUP = 128
M_VALUES = (2048, 4565)
PROBE_M = 32


def load_tensor(index: dict[str, str], name: str, device: str) -> torch.Tensor:
    with safe_open(MODEL / index[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).to(device)


def load_projection(index: dict[str, str], name: str, device: str):
    if name == "in_proj_qkvz":
        prefixes = (f"{LAYER}.in_proj_qkv", f"{LAYER}.in_proj_z")
        packed = torch.cat(
            [load_tensor(index, prefix + ".qweight", device) for prefix in prefixes],
            dim=1,
        )
        scales = torch.cat(
            [load_tensor(index, prefix + ".scales", device) for prefix in prefixes],
            dim=1,
        )
    elif name == "out_proj":
        prefix = f"{LAYER}.out_proj"
        packed = load_tensor(index, prefix + ".qweight", device)
        scales = load_tensor(index, prefix + ".scales", device)
    else:
        raise ValueError(name)
    k = packed.shape[0] * (32 // BITS)
    n = packed.shape[1]
    return packed.contiguous(), scales.to(torch.bfloat16).contiguous(), k, n


def unpack_rows(packed: torch.Tensor, k: int, n: int) -> torch.Tensor:
    pack_factor = 32 // BITS
    raw = packed.view(torch.uint8).reshape(k // pack_factor, n, pack_factor)
    return raw.permute(0, 2, 1).reshape(k, n).to(torch.int16)


def dequant(packed: torch.Tensor, scales: torch.Tensor, k: int, n: int, dtype):
    q = unpack_rows(packed, k, n).to(torch.float32)
    expanded = scales.to(torch.float32).repeat_interleave(GROUP, dim=0)[:k]
    return ((q - 128) * expanded).to(dtype)


def marlin_pack(packed: torch.Tensor, scales: torch.Tensor, k: int, n: int):
    empty = marlin_make_empty_g_idx(packed.device)
    weight = gptq_marlin_repack(
        packed, perm=empty, size_k=k, size_n=n, num_bits=BITS
    )
    permuted_scales = marlin_permute_scales(
        scales, size_k=k, size_n=n, group_size=GROUP
    )
    return weight, permuted_scales


def timefn(fn, iters: int, warmup: int = 4) -> float:
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
    return start.elapsed_time(end) / iters


def quant_int8_weight(weight_nk: torch.Tensor):
    scale = weight_nk.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    quantized = (weight_nk.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.squeeze(1).contiguous()


def w8a8(x: torch.Tensor, weight_kn_int8: torch.Tensor, weight_scale: torch.Tensor):
    act_scale = x.float().abs().amax().clamp_min(1e-8) / 127
    act = (x.float() / act_scale).round().clamp(-127, 127).to(torch.int8)
    accum = torch._int_mm(act.contiguous(), weight_kn_int8)
    return (accum.float() * (act_scale * weight_scale).reshape(1, -1)).to(
        torch.bfloat16
    )


def error(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    delta = (candidate.float() - reference.float()).abs()
    rel = delta.norm() / reference.float().norm().clamp_min(1e-9)
    return float(rel), float(delta.max())


def mib(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / 2**20


def bench_projection(index, name: str, device: str, iters: int):
    packed, scales_bf16, k, n = load_projection(index, name, device)
    start = time.monotonic()
    dense_bf16 = dequant(packed, scales_bf16, k, n, torch.bfloat16)
    torch.cuda.synchronize()
    dequant_seconds = time.monotonic() - start
    dense_fp16 = dense_bf16.to(torch.float16)
    marlin_weight, marlin_scales_bf16 = marlin_pack(
        packed, scales_bf16, k, n
    )
    marlin_scales_fp16 = marlin_scales_bf16.to(torch.float16)
    workspace = marlin_make_workspace(torch.device(device))
    weight_n_int8, weight_scale = quant_int8_weight(dense_bf16.t())
    weight_kn_int8 = weight_n_int8.t().contiguous()

    persistent = mib(packed) + mib(scales_bf16)
    dense_extra = mib(dense_bf16) - persistent
    print(
        f"\n[{name}] K={k} N={n} packed={persistent:.1f} MiB "
        f"dense={mib(dense_bf16):.1f} MiB extra={dense_extra:.1f} MiB "
        f"dequant_once={dequant_seconds:.3f}s"
    )

    probe = torch.randn(PROBE_M, k, device=device, dtype=torch.bfloat16) / 8
    reference = probe @ dense_bf16
    marlin_probe = gptq_marlin_gemm(
        probe,
        None,
        marlin_weight,
        marlin_scales_bf16,
        None,
        None,
        None,
        None,
        workspace,
        scalar_types.uint8b128,
        PROBE_M,
        n,
        k,
    )
    fp16_marlin_probe = gptq_marlin_gemm(
        probe.to(torch.float16),
        None,
        marlin_weight,
        marlin_scales_fp16,
        None,
        None,
        None,
        None,
        workspace,
        scalar_types.uint8b128,
        PROBE_M,
        n,
        k,
    ).to(torch.bfloat16)
    w8a8_probe = w8a8(probe, weight_kn_int8, weight_scale)
    for label, candidate in (
        ("marlin_bf16", marlin_probe),
        ("marlin_fp16_cast", fp16_marlin_probe),
        ("w8a8", w8a8_probe),
    ):
        rel, maximum = error(candidate, reference)
        print(f"  probe {label:18s} rel_l2={rel:.5e} max_abs={maximum:.5e}")

    rows = []
    for m in M_VALUES:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16) / 8

        def marlin_bf16():
            return gptq_marlin_gemm(
                x,
                None,
                marlin_weight,
                marlin_scales_bf16,
                None,
                None,
                None,
                None,
                workspace,
                scalar_types.uint8b128,
                m,
                n,
                k,
            )

        x_fp16 = x.to(torch.float16)

        def marlin_fp16_kernel():
            return gptq_marlin_gemm(
                x_fp16,
                None,
                marlin_weight,
                marlin_scales_fp16,
                None,
                None,
                None,
                None,
                workspace,
                scalar_types.uint8b128,
                m,
                n,
                k,
            )

        def marlin_fp16_cast():
            cast = x.to(torch.float16)
            result = gptq_marlin_gemm(
                cast,
                None,
                marlin_weight,
                marlin_scales_fp16,
                None,
                None,
                None,
                None,
                workspace,
                scalar_types.uint8b128,
                m,
                n,
                k,
            )
            return result.to(torch.bfloat16)

        candidates = (
            ("marlin_bf16", marlin_bf16),
            ("marlin_fp16_kernel", marlin_fp16_kernel),
            ("marlin_fp16_cast", marlin_fp16_cast),
            ("dense_bf16", lambda: x @ dense_bf16),
            (
                "dense_fp16_cast",
                lambda: (x.to(torch.float16) @ dense_fp16).to(torch.bfloat16),
            ),
            ("w8a8_e2e", lambda: w8a8(x, weight_kn_int8, weight_scale)),
        )
        timings = {}
        for label, candidate in candidates:
            try:
                timings[label] = timefn(candidate, iters)
            except RuntimeError as exc:
                if label != "w8a8_e2e":
                    raise
                timings[label] = float("nan")
                print(f"  M={m} {label} unsupported: {str(exc).splitlines()[0]}")
        gflops = 2 * m * n * k / 1e9
        # Keep the persisted report strict JSON.  Python's encoder otherwise
        # emits a non-standard NaN token for unsupported kernel shapes.
        json_timings = {
            label: elapsed if math.isfinite(elapsed) else None
            for label, elapsed in timings.items()
        }
        row = {"projection": name, "m": m, **json_timings}
        rows.append(row)
        print(f"  M={m} ({gflops:.1f} GFLOP)")
        for label, elapsed_ms in timings.items():
            speedup = timings["marlin_bf16"] / elapsed_ms
            print(
                f"    {label:20s} {elapsed_ms * 1000:9.1f} us "
                f"{gflops / elapsed_ms:7.1f} TF/s  {speedup:5.2f}x"
            )
    return {
        "projection": name,
        "k": k,
        "n": n,
        "packed_mib": persistent,
        "dense_mib": mib(dense_bf16),
        "dense_extra_mib": dense_extra,
        "dequant_once_seconds": dequant_seconds,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if args.iters < 1:
        raise SystemExit("--iters must be positive")
    torch.manual_seed(0)
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    print("device:", torch.cuda.get_device_name(0))
    results = [
        bench_projection(index, name, args.device, args.iters)
        for name in ("in_proj_qkvz", "out_proj")
    ]
    payload = {
        "device": torch.cuda.get_device_name(0),
        "bits": BITS,
        "group_size": GROUP,
        "iters": args.iters,
        "projections": results,
    }
    if args.json_output:
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
