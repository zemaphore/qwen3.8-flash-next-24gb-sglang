# PP15: real-routing INT2 MoE retune and PP14 chunk/threshold retune (RTX 3090)

Date: 2026-09-16 UTC

Base commit: `adc57b8` (accepted PP14)

Decision: **DONE** for the routed-MoE config; the chunk/threshold retune and the
full-table gather are recorded as negative or below-gate results.

- Accepted: add an `M=4565` entry to the E=384 and E=432 sm_86 INT2 MoE config
  buckets. Canonical PP improves **+4.2%** steady-state (control 2585.0 +/- 19.7
  vs 2693.9 +/- 13.7 tok/s) with the established exactness oracles unchanged.
  This is deliberately recorded as a below the historical 5% screen gate.
- Rejected: chunk 6144/8192 and threshold 512 (memory / regression evidence
  below).
- Kept opt-in, default off: the full-table gather patch, a below-gate result.

## Accepted routed-MoE config

`tools/tune_moe_int2_real.py` replays the real prefill top-k ids captured by
`patches/prefill_route_dump.py` through `fused_experts_impl`, compacting the ids
exactly as the server does. Two canonical captures (48 layers each) show the
real compact `E` per layer is 406-502 and the per-expert token counts are highly
skewed (layer 24: max 2691, mean 89, 457 nonzero).

The production lookup resolves those to the sm_86 **E=432** bucket and, at
M=4565, its **M=1024** entry: `BLOCK_SIZE_M=32, BLOCK_SIZE_N=64,
BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_warps=4, num_stages=3`. Real-id
coordinate-descent tuning found a consistent better pair across all sampled
layers (gain 1.13-1.21x, mean 1.16x):

| Projection | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K | GROUP_SIZE_M | warps | stages |
|---|---:|---:|---:|---:|---:|---:|
| gate/up (`w13`) | 64 | 64 | 32 | 8 | 4 | 3 |
| down (`w2`) | 64 | 128 | 32 | 8 | 4 | 3 |

The down projection's `BLOCK_SIZE_M` is forced to match gate/up by the shared
`moe_align_block_size` sort, so only the other fields move. This differs from
PP13's rejected synthetic tuning, which used uniform random routing; on the real
skewed distribution the larger `BLOCK_SIZE_M=64` wins and `BLOCK_SIZE_M=128`
does not. The entry is added only at M=4565, so prompts with M <= ~2794 keep the
M=1024 config.

### End-to-end result

`tools/capture_pp5b_arm.py --skip-heldout --chunk-size 4608` on the accepted
stack (raw: [accept-m64](raw/pp15_real_routing_moe_3090_2026-09-16/accept-m64/)):

| Arm | Canonical PP tok/s | Mean +/- sample SD | Decode mean |
|---|---|---:|---:|
| PP14 accepted reference | 2470 / 2636 / 2595 / 2616 / 2600 | 2583.4 +/- 65.4 | 41.2 |
| PP15 accept (M=4565 config) | 2607 / 2725 / 2713 / 2714 / 2678 | **2687.4 +/- 48.3** | 41.1 |

That is **+4.03%** at the capture protocol. A same-session 10-sample control and
a 10-sample variant, excluding the documented PLE re-warm transient, give
control **2585.0 +/- 19.7** (n=8) versus variant **2693.9 +/- 13.7** (n=8),
**+4.21%**. Both exactness oracles are unchanged: `m4_3090_untuned`
0.000000/0.000000 and `lp2` 0.168757/0.011632.

### Characterization on the accepted stack

Required tools, no A/B attached:

`llama-benchy 0.4.0`, `pp 2048 --tg 256 --depth 2048 --runs 3`:
`pp2048 @ d2048` **1246.81 +/- 40.96 tok/s** (3368.5 -> 3291.1 ms e2e TTFT),
`tg256 @ d2048` **29.39 +/- 1.23 tok/s** (peak 30.00 +/- 1.41). PP14 recorded
1218.09 +/- 42.17 and 31.21 +/- 0.67.

Authors' `tools/bench_speed.py`, unmodified except the endpoint:

| Prompt tokens | Prefill seconds | PP tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 101 | 0.40 | 255 | 38.5 |
| 421 | 0.59 | 710 | 38.2 |
| 1,701 | 1.04 | 1,632 | 41.5 |
| 6,821 | 2.73 | 2,501 | 34.1 |
| 10,001 | 3.96 | **2,524** | 42.3 |

Versus PP14's same-tool 10,001-token figure of 2,468 tok/s this is +2.3%; versus
the authors' published 2,271 tok/s reference (different GPU/host) it is +11.1%.

## Chunk-size / threshold retune (rejected)

Each arm changed one launcher variable. All kept PP14 on with the 2,048-token
floor unless stated.

- **chunk 8192** improves the 14,446-token prompt **1650 -> 1942 tok/s
  (+17.7%)** but drove minimum free VRAM to **80 MiB** on the 34,661-token
  prompt (chunk 4608 control: 1,462 MiB). Unsafe, rejected.
- **chunk 9216** crashes during prefill: `int8ring_int4: 9216 tokens in one
  write exceed the ring of 8192 slots` (`SGLANG_KV_TIERS_W=8192`). Hard ceiling;
  chunk must stay <= 8192.
- **chunk 6144** improves the 14,446-token prompt to **1838-1846 tok/s
  (+11.9%)** and completes a 67,024-token prompt at 2,068 tok/s with no OOM, but
  minimum free VRAM falls to **784 MiB** and six low-memory lazy Triton kernel
  loads are logged. Combined with M64 it reached 2,078 tok/s at 34,661 tokens.
  Kept as an explicitly selectable medium-workload option, not the default.
- **threshold 512** (prefetch small tails) **regresses**: 14,446 tokens
  1650 -> 1617 and the authors' 10,001-token prompt 2,476 -> 2,302. The
  prefetch path adds a device-to-device compaction that a short tail's compute
  cannot hide. Keep 2,048.

Raw: [c8192](raw/pp15_prefill_retune_3090_2026-09-16/c8192-t2048.txt),
[c6144](raw/pp15_prefill_retune_3090_2026-09-16/c6144-t2048.txt),
[threshold 512](raw/pp15_prefill_retune_3090_2026-09-16/c4608-t512.txt).

## Full-table gather (below gate, opt-in)

A fresh PP14 profile of the canonical extend is **compute-bound**: of the
1743.3 ms span, compute union is 1595.9 ms, DMA union 942.8 ms, and 901.3 ms
(95.6% of DMA) already overlaps compute, leaving only ~106 ms without a GPU
engine event. The profile's `aten::_unique2` is 922.1 ms of CPU wall, but it
runs behind queued GPU work.

`patches/moe_prefetch_full_table.py` (PP15) removes the data-dependent
`torch.unique`/inverse renumbering for prefetched layers by gathering all `E`
rows through the existing prefetch address table and keeping the router ids. It
is exact (`m4_3090_untuned` 0/0, `lp2` 0.168757/0.011632) and canonical
steady-state is 2599.6 vs 2577.2 control (**+0.9%**, within noise); combined
with the MoE config it is indistinguishable from the MoE config alone
(2694.4 vs 2693.9). Default off; retained as a research path.

## First-layer prefetch (inconclusive, opt-in)

PP14 leaves layer 0 on the old serial selected-row path; its trace attributes
18.4 ms of pinned HtoD to layer 0 alone. `patches/moe_first_layer_prefetch.py`
starts layer 0's routing-independent cold-row copy at the top of the model
forward, before its attention, via the same shared cache (no extra memory).

A 10-sample bracketed A/B with the same assets and source, one env flag apart,
measured **2703.1 +/- 17.1** (on) versus **2679.8 +/- 23.5** (off), +0.87%
(t=2.26), matching the ~1.05% upper bound from the hidden 18.4 ms. However,
pooling the 5-sample capture with the 10-sample arms gives M64+L0 **2694.9**
versus M64-only **2687.0**, only **+0.3%**, inside the capture noise band. It is
therefore opt-in (`SGLANG_MOE_PREFETCH_FIRST_LAYER=1`) and not a default. Oracles
on the on-arm are exact (`m4` 0/0, `lp2` 0.168757/0.011632).

## Direction 4 — QSA prefill metadata (measured no-op)

After PP14 the extend span is GPU-compute-bound, so the long-context plan's
first step was profiling. The QSA indexer path is `_compute_qsa_topk_indices`
216.7 ms, `qsa_indexer.forward_cuda` 215.5 ms, `get_prefill_mqa_inputs`
195.9 ms of CPU span and `_sparse_gqa_prefill` 102.7 ms of GPU time. The
metadata gather is dominated by `sequence_lengths.tolist()` — 186.4 ms over the
12 QSA layers (15.5 ms each) — a device sync run once per layer on a value that
is fixed per forward.

`patches/qsa_prefill_host_lens.py` (default off) threads the already-available
host `forward_batch.seq_lens_cpu` into the metadata so the loop no longer syncs,
falling back automatically when the host copy is absent. It is exact
(`m4` 0/0, `lp2` 0.168757/0.011632) and removes the sync, but it does **not**
move end-to-end PP: a 10-sample arm measured 2704.9 +/- 15.7 versus M64-only
2679.8 +/- 23.5 (t~2.5), yet pooling every QSA sample (22 -> 2685.1) against
every M64-only sample (21 -> 2687.0) shows no effect. The sync was already
hidden behind queued GPU work, consistent with the span's 91.5% GPU-compute
occupancy. Kept opt-in for audit; the remaining QSA cost would need a genuine
sparse-kernel change, which was not attempted.

The largest remaining compute blocks after fused MoE (536.0 ms) and this are
Marlin (463.0 ms, `SMEM`-bound with a fixed template and no easy lever) and the
QSA indexer itself.

## Environment

All arms used presence placement, S184, PP7 DMA staging, PP11 batching, gather
block 2048, and the accepted PLE/KV stack. `SGLANG_PREFILL_ROUTE_DUMP` was set
only for the two capture servers and the feature is off by default. The raw
routing captures are in
[route_dump](raw/pp15_prefill_retune_3090_2026-09-16/route_dump/). No OOM,
retraction, CUDA fault or request failure occurred in any accepted-stack run.
