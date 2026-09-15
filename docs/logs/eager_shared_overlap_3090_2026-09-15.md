# Eager shared/router overlap — RTX 3090

Date: 2026-09-15

## Baseline profile

The accepted PP3 configuration (chunk 2,048, S184) was profiled with the
4,565-token code request. The trace is:

```text
/tmp/pp4-profile/code-pp3-control-1789478992.7174063-TP-0.trace.json.gz
```

Within only the two full 2,048-token extend spans:

| Operation family | Aggregate duration |
|---|---:|
| Expert row gathers | 4,409.6 ms |
| `aten::_unique2` wall time | 3,656.8 ms |
| INT2 fused MoE | 477.3 ms |
| Marlin kernels (PLE + shared expert) | 342.9 ms |
| BF16 GEMM kernels | 123.9 ms |
| QSA indexer kernels | about 3.0 ms |

The trace was analyzed with `tools/chrome_trace_summary.py --within`, added in
this experiment to isolate named scheduler spans. GDN projection and QSA
indexer overlap cannot meet PP4's 5% gate even under the unrealistic assumption
that all their work disappears. The shared-expert branch was the only candidate
large enough to justify an end-to-end A/B.

## Candidate

`patches/moe_eager_shared_overlap.py` extends the existing CUDA-graph
shared/router dual-stream path to ordinary extend batches, gated by
`SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS`. The source and launcher default
remain zero. The candidate used 2,048, so both full chunks and the tail could
take the path; chunk size, S184 residency, PLE cache, and gather tile were held
constant.

The first candidate launch exposed a wrapper issue: the launcher did not yet
forward the new variable. It was stopped without benchmark traffic, the
pass-through was added, and only the corrected launch was measured.

One population/JIT request was excluded at 446 tok/s. Warm code-only samples:

| Sample | Prefill seconds | PP tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 1 | 7.03 | 649 | 37.3 |
| 2 | 7.14 | 639 | 38.1 |
| 3 | 7.37 | 620 | 38.8 |

Mean PP was **636.0 tok/s**, sample standard deviation 14.7, versus the PP3
control's **636.3 tok/s** (SD 12.0). Mean prefill was 7.18 s versus 7.17 s.
The candidate therefore moved PP by effectively zero and did not qualify for a
correctness-oracle run.

Post-workload free VRAM was 3,073 MiB, above the 1.2 GiB floor. No workload,
CUDA, OOM, or retraction error occurred.

## Verdict

Reject eager shared/router overlap for this code workload. Keep the reusable
opt-in patch installed but disabled (`...MAX_TOKENS=0`) so the accepted PP3
behavior remains the default. The result agrees with the profile: expert
staging and its synchronization dominate, while overlapping the comparatively
small shared branch cannot materially move end-to-end PP.

PP4's GDN and QSA sub-branches are also closed for this workload on profile
upper bounds. Revisit only after expert staging is substantially reduced.
