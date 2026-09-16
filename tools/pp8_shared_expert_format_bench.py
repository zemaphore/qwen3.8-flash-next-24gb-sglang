#!/usr/bin/env python3
"""PP8: shared-expert format sweep on real GPTQ tensors.

Compares the production W8A16 GPTQ-Marlin path against dequantized BF16 cuBLAS
and INT8xINT8 (W8A8) for the shared-expert gate/up/down projections at the
canonical M values. The Marlin packing pipeline mirrors
``GPTQMarlinLinearKernel.process_weights_after_loading`` (repack + permuted
scales, zero_points=False / implicit uint8b128 offset) and is validated against
a dequantized BF16 reference before timing.

  CU=/root/quant/venv-sglang/lib/python3.12/site-packages/nvidia/cu13
  PATH="$CU/bin:/root/quant/venv-sglang/bin:$PATH" CUDA_HOME="$CU" \
    PYTHONPATH=/root/sglang/python /root/quant/venv-sglang/bin/python \
    tools/pp8_shared_expert_format_bench.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "/root/sglang/python")

from sglang.kernels.ops.quantization.gptq_marlin import gptq_marlin_gemm  # noqa: E402
from sglang.kernels.ops.quantization.gptq_marlin_repack import (  # noqa: E402
    gptq_marlin_repack,
)
from sglang.srt.layers.quantization.marlin_utils import (  # noqa: E402
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.utils import scalar_types  # noqa: E402

MD = Path("/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang")
LAYER = "model.language_model.layers.0.mlp.shared_expert"
SHAPES = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
GROUP = 128
BITS = 8
PROBE_M = 256


def load_tensor(name: str, device: str) -> torch.Tensor:
    index = json.loads((MD / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(MD / index[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).to(device)


def unpack_rows(packed: torch.Tensor, num_bits: int, k: int, n: int) -> torch.Tensor:
    pack_factor = 32 // num_bits
    raw = packed.contiguous().view(torch.uint8).reshape(k // pack_factor, n, pack_factor)
    return raw.permute(0, 2, 1).reshape(k, n).to(torch.int32)


def dequant_bf16(packed, scales, n, k, group, offset):
    q = unpack_rows(packed, BITS, k, n).to(torch.float32)
    scale = scales.to(torch.float32).repeat_interleave(group, dim=0)[:k]
    return ((q - offset) * scale).to(torch.bfloat16)


def marlin_pack(packed, scales, k, n, group, device):
    empty = marlin_make_empty_g_idx(device)
    w_q = gptq_marlin_repack(
        packed.contiguous(), perm=empty, size_k=k, size_n=n, num_bits=BITS
    )
    w_s = marlin_permute_scales(
        scales.contiguous(), size_k=k, size_n=n, group_size=group
    )
    return w_q, w_s, None, scalar_types.uint8b128


def timefn(fn, iters=30, warmup=5):
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


def quant_int8_act(x):
    scale = x.abs().amax().clamp_min(1e-8) / 127.0
    return (x.float() / scale).round().clamp(-127, 127).to(torch.int8), scale


def quant_int8_weight(w):
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    return (w.float() / scale).round().clamp(-127, 127).to(torch.int8), scale


def bench_projection(name, device="cuda"):
    n, k = SHAPES[name]
    prefix = f"{LAYER}.{name}"
    packed = load_tensor(prefix + ".qweight", device)
    scales = load_tensor(prefix + ".scales", device).to(torch.bfloat16)
    bf16_w = dequant_bf16(packed, scales, n, k, GROUP, 128)
    w_q, w_s, w_zp, weight_type = marlin_pack(packed, scales, k, n, GROUP, device)
    workspace = marlin_make_workspace(device)

    probe = torch.randn(PROBE_M, k, device=device, dtype=torch.bfloat16) / 8
    ref = probe.float() @ bf16_w.float()
    out = gptq_marlin_gemm(
        probe, None, w_q, w_s, None, w_zp, None, None, workspace, weight_type,
        PROBE_M, n, k, is_k_full=True,
    )
    rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
    print(f"  [{name}] N={n} K={k} probe_rel_err={rel:.4e}")

    for m in (469, 1024, 2048):
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16) / 8
        t_marlin = timefn(
            lambda: gptq_marlin_gemm(
                a, None, w_q, w_s, None, w_zp, None, None, workspace, weight_type,
                m, n, k, is_k_full=True,
            )
        )
        t_bf16 = timefn(lambda: torch.matmul(a, bf16_w))
        aq, a_scale = quant_int8_act(a)
        wq8, w_scale = quant_int8_weight(bf16_w.t())
        try:
            t_w8a8 = timefn(
                lambda: torch._int_mm(aq.contiguous(), wq8.t().contiguous()).float()
                * (a_scale * w_scale.reshape(1, -1)).float()
            )
        except RuntimeError as exc:
            print(f"  [{name}] M={m} w8a8 unsupported: {str(exc)[:70]}")
            t_w8a8 = float("nan")
        gflops = 2.0 * m * n * k / 1e9  # GFLOP; GFLOP/ms == TFLOP/s
        print(
            f"  [{name}] M={m:5d}  "
            f"marlin={t_marlin*1000:8.1f}us ({gflops/t_marlin:6.1f} TF)  "
            f"bf16={t_bf16*1000:8.1f}us ({gflops/t_bf16:6.1f} TF)  "
            f"w8a8={t_w8a8*1000:8.1f}us ({gflops/t_w8a8 if t_w8a8 == t_w8a8 else float('nan'):6.1f} TF)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    print("device:", torch.cuda.get_device_name(0))
    for name in SHAPES:
        bench_projection(name, args.device)


if __name__ == "__main__":
    main()
