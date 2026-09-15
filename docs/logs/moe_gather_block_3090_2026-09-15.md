# Expert row-gather tile tuning — RTX 3090

Date: 2026-09-15

## Profile diagnosis

A warmed 4,565-token code request was captured with the built-in CPU/CUDA
profiler.  The original 1,024-byte row-gather tile produced:

| Operation | Count | Aggregate duration |
|---|---:|---:|
| `_gather_rows_tab_kernel` | 960 | 7,954.1 ms |
| `aten::_unique2` | 501 | 5,920.5 ms |
| `fused_moe_kernel_gptq_awq_word` | 480 | 704.8 ms |
| MoE alignment CUDA kernel | 240 | 1.1 ms |

The 960 gathers are four expert tensors x 48 layers x five prefill chunks.
The large rows dominate: `w13_qweight` accounted for 5,093.5 ms and
`w2_qweight` for 2,496.2 ms.  `aten::_unique2` includes synchronization while
materializing a data-dependent expert count, so much of its wall duration is
waiting on preceding GPU work rather than the unique kernel itself.

This refutes fixed `E=512` as the immediate next implementation: it might remove
the dynamic-size barrier, but it would retain the dominant row copies and add
empty-expert alignment/work.

## Model-free tile sweep

`tools/bench_expert_gather.py` recreated the production 512-expert address table
at S=224 with representative mixed GPU/pinned-host rows.  At 352 selected rows:

| Tensor | 1,024-byte tile | 2,048-byte tile | Change |
|---|---:|---:|---:|
| `w13_qweight` | 17.060 ms | 13.771 ms | **-19.3%** |
| `w2_qweight` | 7.292 ms | 6.991 ms | **-4.1%** |
| `w13_scales` | 0.867 ms | 1.057 ms | +21.9% |
| `w2_scales` | 0.514 ms | 0.393 ms | **-23.5%** |

The scale rows are a small fraction of total time.  Across 288, 352, and 432
selected rows, 2,048 bytes was the best consistent compromise for both large
qweight tensors; 4,096 and 8,192 bytes did not improve consistently.

## Code-only e2e validation

The only throughput workload was:

```bash
SGLANG_URL=http://127.0.0.1:30001/generate \
  python3 tools/bench_agentic.py --tokens 4096 --decode-tokens 256
```

PP1b / 1,024-byte control: 481, 487, 487 tok/s; mean **485.0**, sample standard
deviation 3.5.

After applying `patches/moe_gather_block.py` and selecting 2,048 bytes, the first
population/JIT request was excluded (418 tok/s).  Warm samples were:

| Sample | Prefill seconds | PP tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 1 | 9.05 | 505 | 46.0 |
| 2 | 9.21 | 496 | 46.5 |
| 3 | 8.87 | 515 | 43.7 |

Mean PP was **505.3 tok/s**, sample standard deviation 9.5: **+4.2%** over PP1b.
Mean prefill latency fell from 9.41 to 9.04 seconds.

The tie-breaker trace reduced aggregate gather time from 7,954.1 to 7,198.2 ms
(**-9.5%**) and `aten::_unique2` wall time from 5,920.5 to 5,433.9 ms
(**-8.2%**).  The profiled e2e request improved from 463 to 501 tok/s.

The oracle remained exactly at `LOGPROB_MAX=0.168757` and
`LOGPROB_MEAN=0.011632`, identical to PP1b.  No runtime/OOM/memory-leak error
occurred.  The existing VMM FABRIC fallback and low-free-memory Triton preload
warnings remain operational concerns, not new failures.

## Verdict

Keep the 2,048-byte tile as the RTX 3090 default.  It is a byte-identical copy
geometry change with a profiler-confirmed 9.5% reduction in the dominant kernel
family and a 4.2% end-to-end code PP gain.

The larger opportunity is to reduce rows copied at all.  A future hybrid can
run resident experts directly and stage only cold experts, but it needs either
a pointer-table fused MoE kernel or a carefully validated split hot/cold
execution path.

