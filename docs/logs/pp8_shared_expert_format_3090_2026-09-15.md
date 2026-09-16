# PP8 — shared-expert format sweep: REJECTED on measured ceiling

Date: 2026-09-15. Host: RTX 3090, driver 580.178.04. Stack: accepted PP5c/PP11
state (bulk pread + 128 MiB recent-row cache, 2,048-byte gather tile, host-row
DMA staging + batching, 2,048-token chunks, S184, presence placement). No
production flag was changed; the server was only profiled and then stopped.

## Question

The shared expert is quantized to INT8 (`*shared_expert.` bits 8) and runs as a
separate `Qwen2MoeMLP` through W8A16 GPTQ-Marlin (CUDA shared-expert fusion is
off: `enable_cuda_shared_expert_fusion` defaults False for this model). PP8
asked whether Ampere would prefer BF16 cuBLAS or W8A8 at M = 469/1024/2048.

## Method

1. Fresh stage profile of the accepted stack via `/start_profile`
   (`profile_by_stage`, 3 activities, output
   `docs/logs/raw/pp8_3090_2026-09-15/profile/`). The 4,565-token prompt runs
   as two 2,048-token extends plus the 469-token tail.
2. Model-free format sweep on the real layer-0 shared-expert GPTQ tensors with
   `tools/pp8_shared_expert_format_bench.py`: the production W8A16 Marlin path
   (repack + permuted scales, `zero_points=False` / implicit uint8b128 offset,
   validated against a dequantized BF16 reference at 2.1-2.6e-3 relative error)
   versus dequantized BF16 cuBLAS and INT8xINT8 (`torch._int_mm`).

## Profile: the accepted stack is now GPU-engine-bound

Span union of GPU activity (kernel and DMA events, overlap-aware), vs PP6's
CPU-bound reading:

| Span | Wall | GPU union | SM kernels | DMA | Marlin |
|---|---:|---:|---:|---:|---:|
| chunk 1 (2,048) | 1,507 ms | 88.0% | 43.2% | 45.6% | 11.9% |
| chunk 2 (2,048) | 1,527 ms | 88.3% | 44.4% | 44.6% | 11.7% |
| tail (469) | 759 ms | 82.6% | 29.7% | 54.8% | 6.1% |

After PP11 the GPU is active ~88% of the span wall (SM ~44%, DMA ~45%, largely
serialized), not 37% SM-busy as PP6 measured. Marlin is ~179 ms/chunk = ~12% of
wall, but that includes the PLE and linear-attention GEMMs, not just the shared
expert.

## Format sweep (real layer-0 shared-expert tensors)

Times are isolated, warm, per GEMM. GFLOP/ms = TFLOP/s.

| Projection | M | Marlin W8A16 | BF16 cuBLAS | W8A8 |
|---|---:|---:|---:|---:|
| gate (N=640, K=2560) | 469 | 55.8 us | **29.6 us** | unsupported |
| gate | 1024 | **63.0 us** | 101.7 us | 83.2 us |
| gate | 2048 | **124.7 us** | 198.7 us | 154.3 us |
| up (N=640, K=2560) | 469 | 55.6 us | **29.5 us** | unsupported |
| up | 1024 | **62.8 us** | 101.4 us | 83.2 us |
| up | 2048 | **124.5 us** | 198.6 us | 154.2 us |
| down (N=2560, K=640) | 469 | 46.9 us | **30.3 us** | unsupported |
| down | 1024 | 66.9 us | **56.5 us** | 127.1 us |
| down | 2048 | 133.3 us | **105.8 us** | 232.1 us |

Findings: gate/up strongly prefer Marlin (~1.6x BF16, ~1.3x W8A8); down_proj
prefers BF16 (~1.2-1.3x Marlin) and W8A8 is worst everywhere. At the 469-token
tail, BF16 wins all three (~1.6-1.9x Marlin). `torch._int_mm` rejects the odd
M=469 shape, so the W8A8 tail cannot even be measured through cuBLAS.

## Ceiling

Per-layer shared-expert GPU time: 382.5 us at M=2048, 158.3 us at M=469.
Across 48 layers and the two full chunks plus tail:

- Current Marlin: **44.3 ms per prompt**.
- Best possible hybrid (Marlin gate/up + BF16 down at M=2048, all-BF16 tail):
  **38.4 ms per prompt**.
- Saving: **5.9 ms of a ~3,822 ms prompt = 0.16%**.

Even making the shared expert free removes only ~1.2% of the wall. The 5% gate is
unreachable, so PP8 is rejected without an A/B and without needing a W8A8 quality
gate.

## Verdict

**REJECTED.** The shared expert is not the lever. The ~172 ms/chunk of Marlin in
the PP6 profile is dominated by the other Marlin GEMMs (PLE and the 36
linear-attention `in_proj` layers), which PP8 does not target. The profile
correction is the useful by-product: post-PP11 the accepted stack is ~88%
GPU-engine-active, so GPU-side levers are more plausible than PP6 implied, but
they must target the large Marlin/PLE GEMMs or the tail-chunk fixed staging cost,
not the shared expert.

## Evidence

- Profile trace:
  `docs/logs/raw/pp8_3090_2026-09-15/profile/1789508360.5259962-TP-0-EXTEND.trace.json.gz`
- Span union summary:
  `docs/logs/raw/pp8_3090_2026-09-15/profile/span_gpu_union.txt`
- Format sweep output:
  `docs/logs/raw/pp8_3090_2026-09-15/format_bench.txt`
- Server log: `docs/logs/raw/pp8_3090_2026-09-15/server.log`
- Tool: `tools/pp8_shared_expert_format_bench.py`

## Caveats

The format numbers are isolated microbenchmarks; in-model contention and the
profiler add overhead. That uncertainty is far smaller than the two orders of
magnitude between the measured 0.16% best-case saving and the 5% gate. The
profile was taken through the live `/start_profile` endpoint with no code change
or restart; the server was stopped after capture.
