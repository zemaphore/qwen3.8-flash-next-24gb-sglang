# RTX 3090 prompt-processing performance plan

Last updated: 2026-09-15

This is the resumability and status document for improving Qwen3.8-Flash-Next
prompt-processing (PP) performance on the 24 GB RTX 3090 host.  Decode work has
its own plan in [DECODE_PERF_PLAN.md](DECODE_PERF_PLAN.md).

## Current conclusion

The tuned INT2 MoE kernel is not the dominant end-to-end PP opportunity.  M4
improved its standalone path by 1.38x geometric mean but moved end-to-end PP by
only 1.8-4.1% in the earlier measurements.  Work should now target PLE storage,
expert routing/gather preparation, chunk amortization, and the 36-layer GDN
pipeline.

The PLE work now has a bounded policy that covers both measured regimes:

| Code workload state | Prior control | Current PP1b | Result |
|---|---:|---:|---:|
| Matched cold PLE cache | bulk pread 403 tok/s | 396 / 400 tok/s | within **-1.7%** |
| Warmed code prompt, same-day mean | bulk pread 436.7 tok/s | 485.0 tok/s | **+11.1%** |
| Warmed code prompt, earlier mmap mean | mmap 471 tok/s | 485.0 tok/s | **+3.0%** |

Bulk pread plus a bounded 128 MiB exact recent-row cache is now the accepted
launcher default.  It retains only 6.2 MiB for this workload.  A 2,048-byte
expert-gather tile raised warmed code PP from 485.0 to 505.3 tok/s.  PP3 then
reduced residency from S224 to S184 and increased the chunk to 2,048 tokens,
raising warmed code PP to **636.3 tok/s**: +25.9% over PP2a while retaining
3.17 GiB post-workload free VRAM.  PP4 intra-layer overlap was then rejected:
the largest shared/router candidate averaged 636.0 tok/s, effectively unchanged.
PP5 static presence-ranked placement was then rejected too: a code chunk routes
to ~329 of 512 experts per layer, so every static S=184 set leaves ~210 distinct
cold rows and the best-ranked alternative moved that by only -0.2%.  PP7 found
the true wall: the ~46 ms/layer-chunk of expert staging was SM host-read bound
at ~7.5-10 GB/s in every geometry, so the DMA engine (~22 GB/s) was used to
stage the cold rows instead, giving **1111.8 tok/s** warmed, +74.8% over PP3,
bit-identical.  Prompt-processing time on the code workload is now ~4.1 s.  A full
torch-profiler capture of the PP7 stack (PP6) then measured the steady
2,048-token chunk: 875.2 ms pinned-HtoD DMA staging, 238.0 ms INT2 fused MoE,
172.4 ms Marlin, and only 40.2 ms across all 36 GDN layers; the SMs idle ~63%
of the span while the single-threaded CPU staging path (unique2, tolist syncs,
~50k per-row copies) consumes 78% of the CPU wall.  The next lever is
therefore PP11: eliminate the staging path's CPU serialization so DMA and
compute interleave, with a rough 2x ceiling.

## Status key

- **DONE**: implemented and validated; keep unless later evidence overturns it.
- **PARTIAL**: useful result, but acceptance criteria are not satisfied in every
  relevant regime.
- **NEXT**: the next experiment to implement.
- **QUEUED**: defined well enough to resume without redesign.
- **RESEARCH**: promising, but needs profiling or a design decision first.
- **REJECTED**: measured and not worth continuing in its tested form.
- **SUPERSEDED**: overtaken by a later result that removed the premise.

## Canonical PP validation protocol

The primary workload is the deterministic code/agentic benchmark only:

```bash
SGLANG_URL=http://127.0.0.1:30001/generate \
  python3 tools/bench_agentic.py --tokens 4096 --decode-tokens 256
```

It currently tokenizes to 4,565 prompt tokens.  The accepted 2,048-token setup
is scheduled as two full chunks plus a padded tail.  Do not substitute the repeated-sentence benchmark when
making an acceptance decision.  Do not run `llama-benchy` unless a later task
explicitly asks for it.

For a cold-PLE comparison:

1. Stop the SGLang scope.
2. While no process maps `ple.f8_e4m3.bin`, issue `POSIX_FADV_DONTNEED` for the
   file.
3. Start the requested variant and let the launcher's ordinary warmup finish.
4. Run exactly one code prompt.

For warmed performance, run the identical prompt three times after the first
request and report mean plus sample standard deviation.  Change one variable
per server restart.  Record actual prompt tokens, prefill seconds, PP tok/s,
decode tok/s, exact launcher/environment delta, server log path, errors, and
minimum free VRAM.

An exact optimization must also pass the machine-local greedy/logprob oracle
before acceptance.  Approximate changes require an explicit quality plan and
must never be silently mixed into an exact A/B.

## Experiment queue

| ID | Status | Experiment | Main hypothesis | Acceptance gate |
|---|---|---|---|---|
| PP0 | DONE | RTX 3090 INT2 M4 buckets | Better real-shape kernels help PP after weights arrive | Exact outputs; retained despite modest e2e gain |
| PP1 | PARTIAL | Hybrid mmap / parallel PLE pread | NVMe queue depth removes cold random mmap stalls | Cold gain confirmed; warm regression prevents default acceptance |
| PP1b | **DONE** | Bounded recent PLE row cache | Use pread for misses and exact retained FP8 rows for known-hot chunks | Cold within 1.7% of PP1; warm +11.1% vs same-day pread and +3.0% vs mmap |
| PP2a | **DONE** | Tune expert row-gather tile to 2,048 bytes | Reduce CTA count for 0.82/0.41 MB qweight rows | Exact; +4.2% warmed PP and -9.5% profiled gather time |
| PP2 | SUPERSEDED | Direct/hybrid expert execution | Stage only selected cold experts; avoid copying resident rows | No longer needed for PP: PP7 DMA staging made full-row transfer cheap; low-token direct GEMV was +3.4% slower |
| PP3 | **DONE** | Prefill residency/chunk trade | Fewer 1536/2048-token chunks amortize expert census and transfer | Exact; 2048/S184 is +25.9%, with 3.17 GiB free VRAM |
| PP4 | **REJECTED** | Eager intra-layer dual-stream overlap | Hide shared/projection compute behind routed-expert work | Shared/router A/B was flat; GDN/QSA profile upper bounds are below 5% |
| PP5 | **REJECTED** | Prefill-aware expert placement | Presence-per-chunk is a better transfer objective than routing mass | Static rankings tie within 0.2% of mass; placement objective is saturated |
| PP6 | REJECTED | RTX 3090 GDN pipeline tuning/fusion | 36 Triton GDN layers contain more untuned PP time than fused MoE | Profiled: GDN is 40.2 ms of a 1,773 ms chunk (2.3%) versus 238.0 ms fused MoE; 5% gate unreachable |
| PP7 | **DONE** | Hybrid cold-expert execution | Full-row transfer is wasteful for experts assigned few tokens | Low-token direct GEMV rejected (SM host reads ~7.5-10 GB/s either way); the host-row DMA staging form is bit-exact and +74.8% warmed PP |
| PP8 | RESEARCH | Dense/shared-expert format sweep | Ampere may prefer BF16 cuBLAS or W8A8 over W8A16 Marlin at M=1024 | Memory-neutral enough to retain S; quality gate if W8A8 |
| PP9 | RESEARCH | Fused QSA score/mask/top-k | Avoid full FP32 logits materialization at long prefixes | Prioritize only after 4.5k code PP work |
| PP10 | RESEARCH | Host/PCIe operational audit | Link downgrade, IOMMU, VM scheduling, or CPU affinity may cap transfers | Narrowed: 23 GB/s H2D confirmed (Gen4 x16); remaining item is whether a PCIe or memory topology change unlocks more, and the decode GEMV's 10 GB/s SM host reads |
| PP11 | **NEXT** | Staging-path CPU/sync elimination | The single-threaded staging submission (unique2 + tolist syncs + ~50k per-row copies = 78% of CPU wall) serializes a 63%-idle GPU; unblocking it lets DMA and SM work interleave | Exact staging bytes; warmed PP A/B per protocol; adopt at >= 5% e2e |

## PP1 — PLE storage experiment

### Completed

- `tools/nvme_probe.py` measured about 54-56k random 4 KiB IOPS at QD16-32.
  QD64 collapsed and should not be used.
- `tools/ple_io_bench.py` replays the actual 16-row/token layout without loading
  the model and checks every returned byte.
- Fully unique 1024-token replay: cold mmap 3661.5 ms versus deduplicated
  16-worker pread 330.4 ms.
- A 256-row repeated working set favored mmap: 23.8 ms cold / 2.2 ms warm versus
  41.0 ms pread.
- `patches/ple_bulk_pread.py` implements the opt-in unique-count policy.
- `patches/enable_ple_bulk_pread.py` configures the 3090 launcher.  It can be
  disabled with `SGLANG_QWEN4_PLE_BULK_PREAD=0`.
- Matched code-only e2e: 351 -> 403 tok/s cold, but warmed mean 471 -> 448.

Evidence:

- [Storage replay](logs/ple_io_3090_2026-09-15.md)
- [Code-only e2e A/B](logs/ple_bulk_pread_e2e_3090_2026-09-15.md)

### PP1b completed

`patches/ple_recent_row_cache.py` retains deduplicated FP8 rows from high-entropy
pread chunks under an environment-controlled byte cap.  Exact repeated chunks
reuse those bytes directly; decode and small-working-set mmap behavior are
unchanged.  The launcher defaults to 128 MiB with profiling disabled.

Measured code-only results:

- Cold: 396 and 400 tok/s versus PP1's 403 tok/s.
- Warm clean mean: 485.0 tok/s, sample standard deviation 3.5.
- Same-day bulk-pread warm control: 436.7 tok/s, sample standard deviation 4.7.
- All five warmed chunks hit; retained rows plus IDs were 6.2 MiB.
- Free GPU memory bottomed at 1,325 MiB.

The old `lp2` reference produced MEAN 0.011632, marginally beyond its 0.010
gate, but cache-miss and all-cache-hit oracle passes were identical.  This is a
current-baseline/reference audit item, not a cache-induced numerical change.

Evidence: [PP1b code-only e2e](logs/ple_recent_row_cache_e2e_3090_2026-09-15.md)

Possible later extension: partial-overlap row reuse or asynchronous miss
prefetch.  Do not expand PP1b before PP2 unless a new workload shows a concrete
miss problem.

## PP2 — expert staging diagnosis and tile tuning

The code profile showed that compact staging, rather than the fused INT2 MoE
kernel, dominates prefill:

- Four table gathers x 48 layers x five chunks consumed 7.954 s of GPU time.
- The two qweight gathers accounted for 7.590 s of that total.
- INT2 fused MoE consumed only 0.705 s.
- `aten::_unique2` showed 5.921 s of CPU wall time, largely synchronization on
  the preceding GPU queue rather than unique compute.

PP2a tuned `_gather_rows_tab_kernel` from 1,024 to 2,048-byte tiles.  Clean warm
code PP improved 485.0 -> 505.3 tok/s (+4.2%), while a matched trace reduced
aggregate gather time by 9.5%.  Keep it enabled through
`SGLANG_MOE_GATHER_BLOCK=2048`.

Evidence: [PP2a gather tile](logs/moe_gather_block_3090_2026-09-15.md)

Do not implement the original naïve fixed `E=512` proposal: it retains the
dominant copies and schedules extra empty-expert work.  The higher-value design
is a hybrid path that reads S resident experts directly and stages only selected
cold experts.  This likely requires a pointer-table fused MoE kernel or split
hot/cold execution; keep the compact path as fallback until exactness and PP are
proven.

## PP3 — larger chunks by reducing prefill residency

Completed under the current elastic/tiered-KV stack.  The S224 / chunk 1,024
PP2a control averaged 505.3 tok/s.  Holding S184 and increasing the chunk gave:

- 1,536: 608 / 614 / 593 tok/s; mean **605.0**, sample SD 10.8.
- 2,048: 648 / 637 / 624 tok/s; mean **636.3**, sample SD 12.0.

The 2,048 winner is **+25.9%** over PP2a and kept 3,241 MiB free VRAM after
the workload.  Its oracle result exactly matched the accepted stack.  No OOM,
retraction, CUDA fault, or workload exception occurred.

Keeping S184 through decode reduced the measured decode rate, although the
256-token end-to-end tail still improved overall request time.  Phase-aware
resident-set regrowth is not implemented and remains follow-up work; do not
conflate it with PP3's accepted fixed-chunk result.

Evidence: [PP3 chunk/residency sweep](logs/prefill_chunk_residency_3090_2026-09-15.md)

## PP4 — eager intra-layer overlap

Existing dual-stream code was gated such that eager prefill missed it.  The
accepted PP3 baseline profile found 4.410 s in expert gathers for the two full
chunks, versus only 0.343 s across all PLE/shared Marlin kernels, 0.124 s across
BF16 GEMMs, and about 3 ms in QSA indexer kernels.

The largest candidate extended the existing shared-expert/router dual-stream
path to eager batches through an opt-in 2,048-token threshold.  Its warmed code
samples were 649 / 639 / 620 tok/s: mean **636.0**, sample SD 14.7, versus the
PP3 control's 636.3.  Reject it.  GDN and QSA overlap cannot meet the 5% gate
even at their profile upper bound; revisit only if staging is first reduced.

Evidence: [PP4 eager shared/router overlap](logs/eager_shared_overlap_3090_2026-09-15.md)

The tested branch list was:

1. Shared expert versus router/routed-expert branch.
2. QSA indexer versus QKV projection; include M=1024 in the eligibility check.
3. GDN `in_proj_qkvz` versus `in_proj_ba`; include M=1024 if shape-safe.

This is distinct from the earlier rejected server overlap-schedule experiment.
The source patch remains installed but disabled by default for future research.

## PP5 — prefill-aware placement

Completed and rejected in the static form.  `patches/prefill_route_dump.py`
(opt-in `SGLANG_PREFILL_ROUTE_DUMP`, installed and left disabled) recorded
routing ids for nine 2,048-token code chunks; `tools/pp5_presence.py` compared
top-S=184 residency by routing mass versus per-chunk presence and token
counts.  A code chunk routes to ~329 distinct experts per layer, so every
static set stages ~210 cold rows per layer-chunk and the presence ranking
improved that by only 0.2% (210.09 -> 209.66) despite a 56% resident-set
disagreement with mass.  The gate was unreachable before any A/B: mass
placement is kept and nothing numerically changed.  The dump histogram shows
22% of staged cold rows carry <= 2 tokens for 1% of the tokens — that is
PP7's size estimate, and request-local promotion cannot beat it either since
presence was already measured per chunk.

Evidence: [PP5 presence-vs-mass placement](logs/pp5_presence_placement_3090_2026-09-15.md)

Restart caveat learned in this line of sessions: after repeated weight loads
the PLE page cache re-warms slowly and the first warmed triplets read ~5% low
(600 then 625.7 versus the 629.7 steady baseline; the PP7 DMA-off control
likewise read 597-645 before holding).  Run at least four warm samples before
accepting any delta under ~5%.

## PP7 — hybrid cold-expert execution

Completed in two halves.  The planned low-token direct-GEMV variant was
rejected with data: `tools/pp7_hybrid_bench.py` replays real 2,048-token
code-chunk routing against synthetic S184 placement with the production
kernels (resident experts staged normally, low-count cold experts executed
in place through the pointer-table GEMV, results merged), and layer 0 came in
at +3.4% *slower* - both paths read pinned host through the SMs at the same
~7.5-10 GB/s, so direct execution pays T row-reads to save one staged row.
The same measurement program found the real ceiling: `cudaMemcpyAsync` moves
the identical scattered pinned rows at 21.8-22.2 GB/s (400 MB contiguous
copy: 23.0 GB/s), ~3x the best SM geometry in any tile/vector configuration.

The accepted form keeps staging for every expert but splits the transport:
resident rows still go through one existing tab-kernel launch into the
staging prefix, while each host row is copied into the staging tail by DMA
from its pinned elastic slot (`_placed["home"]`, added by
`patches/moe_host_dma_gather.py` to the elastic placement; resize mutates the
same lists, so it stays authoritative).  Top-k ids are renumbered onto the
permuted staging order through a position lookup; staged bytes per expert are
identical and the fused kernel is untouched, so the change is bit-exact by
construction and needs no extra device sync (the `tolist()` replaces the
existing `numel()` one).

Warmed code A/B, one variable (`SGLANG_MOE_GATHER_DMA`): variant 1096/1122/
1111/1110/1120, mean **1111.8** tok/s, sample SD 10.6; same-session DMA-off
control 597-645, mean 623.7 inside the documented re-warm band; documented
steady baselines 629.7-636.3.  **Accepted at +74.8%.**  Prefill fell 9.0 ->
~4.1 s; decode 38.5-39.6 tok/s in both arms (unchanged S184 note from PP3);
3,239 MiB free VRAM after the workload.  Both machine oracles are identical
to the accepted stack: `m4_3090_untuned` 0.000000/0.000000 and `lp2`
0.168757/0.011632.  The `oa` greedy reference predates this machine and
diverges identically before and after.

Evidence: [PP7 host-row DMA staging](logs/pp7_host_dma_staging_3090_2026-09-15.md)

This retires the PP2 staging-transfer motivation (a pointer-table fused MoE
kernel is no longer needed for PP), narrows PP10 to the decode GEMV's SM host
reads (a DECODE_PERF_PLAN candidate), and leaves resident-row staging copy
skipping (~1 ms per layer-chunk) as the small remaining gather item.

## PP6 — GDN pipeline profile (closed)

Profiled and rejected without an A/B.  A stage-filtered torch-profiler capture
of the accepted PP7 stack (`/start_profile`, `profile_by_stage`, 3 extend
forwards) measured the steady 2,048-token chunk:

- GPU occupancy: pinned-HtoD DMA staging 875.2 ms (16.2 GiB, 18.5 GB/s
  effective, 258 cold rows/layer-chunk x 4 kinds), INT2 fused MoE 238.0 ms,
  Marlin 172.4 ms, BF16 GEMMs 68.4 ms, QSA 51.9 ms, resident-row tab gather
  24.9 ms; SMs idle ~63% of the 1,773 ms span.
- All 36 GDN layers together cost **40.2 ms per chunk (1.12 ms/layer)**, led by
  `chunk_gated_delta_rule_fwd_kernel_h` at 259.7 us/layer.  The PP6 premise
  (GDN contains more untuned PP time than fused MoE) is false: fused MoE is
  5.9x GDN.  Halving every GDN kernel moves PP ~1.1%, below the 5% gate.
- GDN/QSA overlap recheck: GDN+QSA = 92.1 ms = 5.2% of span wall nominally,
  but the span is CPU-bound (staging path holds 78% of CPU wall) and layer
  structure forbids overlap (layer L+1 staging needs layer L+1's router, which
  needs its attention).  Consistent with PP4's flat A/B; still rejected.
- CPU attribution per span: `_run_qwen4_exp_mlp` 1,382 ms (28.8 ms/layer, of
  which `_gather_dma` 829.6 ms), `_unique2` 391 ms, `aten::item` 382 ms,
  `cudaStreamSynchronize` 381 ms, per-row view/copy submission ~800 ms.
- The 469-token tail span pays 664 of 1,107 ms for staging: the per-layer-chunk
  staging cost is fixed, not token-proportional.

Evidence: [PP6 GDN profile](logs/pp6_gdn_profile_3090_2026-09-15.md), trace
`/tmp/pp6-profile/1789493729.4675837-TP-0-EXTEND.trace.json.gz`.

## PP11 — staging-path CPU/sync elimination (NEXT)

The profile shows the pipeline is CPU-serialized: the single scheduler thread
spends 78% of the span wall inside the MoE staging path while the SMs idle 63%
and the DMA engine does 875 ms of real work.  Sub-items, in expected value
order:

1. Remove the `tolist()`/`item()` syncs from the staging critical path (keep
   top-k ids on device into the fused MoE; largest structural change).
2. Batch the ~50k per-row pinned copies per chunk (12392 copies/kind/span)
   into one batched submission per kind; precompute row views at elastic
   resize to kill the 119.8k `aten::select` calls.
3. Overlap `_unique2` with the previous layer's compute instead of blocking.

Ceiling if fully successful: span wall approaches max(DMA ~875 ms, SM ~652 ms)
plus tail effects, roughly 2x current warmed PP.  Acceptance: staging bytes
identical (bit-exact by construction or machine-local oracle pass), warmed
code A/B per protocol, adopt at >= 5% e2e.

## Later research

- Benchmark shared-expert W8A16 Marlin against selectively materialized BF16.
  Only then consider calibrated W8A8.
- At long prefixes, profile QSA's full FP32 score tensor and consider fusing
  score, mask, and hierarchical top-k.  The previously rejected paged-prefix KV
  kernel should not be repeated unchanged.
- Tail-chunk staging amortization: the 469-token tail pays nearly a full
  chunk's fixed staging cost; consider merging or special-casing it (~10% of
  prompt tokens).
- Audit negotiated PCIe generation/width, BAR1, clocks/power/thermals, VM CPU
  affinity, NUMA locality, huge pages, and IOMMU mode before attributing a hard
  ceiling to the GPU.

## Resume checklist

1. Read this file and the two newest linked result logs.
2. Run `git status`; preserve all pre-existing uncommitted changes.
3. Identify the active systemd scope from the first line of
   `/root/quant/logs/server.log`; do not assume the scope recorded below exists.
4. Confirm patch states before a restart:

   ```bash
   SGLANG=/root/sglang python3 patches/ple_bulk_pread.py --check
   SGLANG=/root/sglang python3 patches/ple_recent_row_cache.py --check
   python3 patches/enable_ple_bulk_pread.py --check
   python3 patches/enable_ple_recent_row_cache.py --check
   SGLANG=/root/sglang python3 patches/moe_gather_block.py --check
   python3 patches/enable_moe_gather_block.py --check
   SGLANG=/root/sglang python3 patches/moe_eager_shared_overlap.py --check
   python3 patches/enable_moe_eager_shared_overlap.py --check
   SGLANG=/root/sglang python3 patches/moe_host_dma_gather.py --check
   python3 patches/enable_moe_host_dma_gather.py --check
   SGLANG=/root/sglang python3 patches/prefill_route_dump.py --check
   python3 patches/enable_prefill_chunk_residency.py --check
   SGLANG=/root/sglang python3 patches/moe_config_buckets.py --check
   ```

5. Continue with PP11 unless the owner reprioritizes.
6. After every experiment, update the status table, append a dated result under
   the relevant section, and link the raw log/artifact.

State at this update: bulk pread, the 128 MiB recent-row cache, the 2,048-byte
expert-gather tile, host-row DMA staging, a 2,048-token prefill chunk, and S184
residency are enabled; profiling is disabled.  The prefill route dump is applied
and pass-through in `/root/quant/serve-3090.sh` but off unless
`SGLANG_PREFILL_ROUTE_DUMP` names a directory.  The server is running on port
30001 in `sglang-1789491753.scope` with `--sleep-on-idle`.  This is transient
operational state, not a prerequisite for resuming.  The PP6 profile was taken
through the live `/start_profile` endpoint with no code change or restart.
