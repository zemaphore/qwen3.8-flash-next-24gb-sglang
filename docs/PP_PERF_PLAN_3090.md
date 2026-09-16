# RTX 3090 prompt-processing performance plan

Last updated: 2026-09-15

This is the resumability and status document for improving Qwen3.8-Flash-Next
prompt-processing (PP) performance on the 24 GB RTX 3090 host.  Decode work has
its own plan in [DECODE_PERF_PLAN.md](DECODE_PERF_PLAN.md).

## Current conclusion

The tuned INT2 MoE kernel is not the dominant end-to-end PP opportunity.  M4
improved its standalone path by 1.38x geometric mean but moved end-to-end PP by
only 1.8-4.1% in the earlier measurements.  The remaining measured PP target is
expert staging-path CPU/synchronization overhead. PP11 removed the per-row
Python copy-submission bottleneck with `cudaMemcpyBatchAsync`: bracketing code
controls pooled to 1110.9 tok/s while PP11 reached **1183.4 tok/s (+6.53%)**,
so it is accepted and enabled by default. GDN tuning missed its standalone gate.
Static prefill-aware placement missed its original gate on PP7, but the PP5b
interaction test on PP11 measured presence **1244.0** versus pooled mass controls
**1177.3 tok/s (+5.67%)** with unchanged exactness oracles. Review flagged that
the placement was derived from that same prompt and that raw per-arm output was
not retained, so PP5c reran the bracket with raw capture and held-out code.
PP5c measured presence **1247.2 +/- 26.6** versus pooled mass **1185.9 +/- 17.1
tok/s (+5.17%)** on the canonical corpus, with held-out code at **-0.91%** and
**+1.47%** (both non-material) and identical oracles. Presence is therefore the
general launcher default, and routing mass remains explicitly selectable. PP8
then measured the shared-expert format path at ~44 ms/prompt (~1.2% of wall) and
rejected it; that profile also corrected PP6's CPU-bound reading by showing the
accepted post-PP11 stack is ~88% GPU-engine-active, leaving the tail-chunk fixed
staging cost and the large PLE/linear-attention GEMMs as the remaining levers.
PP12 then removed the tail's repeated staging by raising the prefill chunk to
4,608 so the canonical prompt is a single extend: **1937.8 +/- 12.9** versus
pooled 2,048 controls **1286.2 +/- 10.4 tok/s (+50.66%)**, exact oracles and
unchanged decode. The 4,608 chunk is now the general launcher default.

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
PP5's first static-placement analysis was invalidated on review: it included
two launcher warmups and crossed the chunk and layer axes.  The repaired
analysis found 236.68 -> 211.16 cold rows/layer-chunk (-10.8%) for presence,
but a bracketing PP7 E2E A/B measured only **+3.1%** (1110.0 -> 1144.8 tok/s),
below the 5% gate, so routing mass remains the default.  PP7 found
the true wall: the ~46 ms/layer-chunk of expert staging was SM host-read bound
at ~7.5-10 GB/s in every geometry, so the DMA engine (~22 GB/s) was used to
stage the cold rows instead, giving **1111.8 tok/s** warmed, +74.8% over PP3,
bit-identical.  Prompt-processing time on the code workload is now ~4.1 s.  A full
torch-profiler capture of the PP7 stack (PP6) then measured the steady
2,048-token chunk: 875.2 ms pinned-HtoD DMA staging, 238.0 ms INT2 fused MoE,
172.4 ms Marlin, and only 40.2 ms across all 36 GDN layers; the SMs idle ~63%
of the span while the single-threaded CPU staging path (unique2, tolist syncs,
~50k per-row copies) consumes 78% of the CPU wall.  The next lever is
therefore motivated PP11. Batching the cold-row copies reduced CPU submission
and reached **1183.4 tok/s**, +6.53% over pooled pre/post controls, while keeping
the staging bytes and machine oracles unchanged. The authors' original
repeated-sentence speed tool reached 1,909 tok/s at 10,001 tokens, about 84% of
the repository's 2,271 tok/s reference.

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

It currently tokenizes to 4,565 prompt tokens.  Since PP12 the accepted setup
uses a 4,608-token prefill chunk, so this workload runs as a single extend
instead of two full 2,048-token chunks plus a padded tail.  Do not substitute the repeated-sentence benchmark when
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

For PP5b/PP5c, use `tools/capture_pp5b_arm.py` on each freshly started arm. It
retains the unparsed output of every request, the filtered live server
environment, placement hash/signature, elastic status, GPU samples, exactness
oracles, and before/after server logs. Commit the resulting raw directory; a
summary table or systemd scope ID is not a substitute for raw evidence.

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
| PP5 | **SUPERSEDED** | Prefill-aware expert placement | Presence-per-chunk is a better transfer objective than routing mass | Corrected analysis saves 10.8% cold rows; bracketing PP7 E2E was only +3.1%, below the original gate; PP5b/PP5c later established a canonical-prompt interaction with no held-out material regression |
| PP5b | **PARTIAL** | PP11 + presence-placement interaction | Fewer cold rows and cheaper batched submission may compound; the earlier +3.1% is still useful | Canonical prompt: presence 1244.0 +/- 25.3 vs pooled mass 1177.3 +/- 26.1 tok/s (**+5.67%**, +4.94% vs control A, +6.40% vs control C); aggregate arithmetic and oracles check out, but raw outputs were not retained and the placement was trained on the same prompt; PP5c added the raw capture and held-out validation and promoted presence |
| PP5c | **DONE** | Captured PP5b rerun plus held-out code | Establish auditability and determine whether presence placement generalizes beyond its training prompt | Three fresh-server captured arms via `tools/capture_pp5b_arm.py`; canonical presence 1247.2 +/- 26.6 vs pooled mass 1185.9 +/- 17.1 tok/s (**+5.17%**, t=4.68); held-out code -0.91% and +1.47% (non-material); oracles exact and error-free; presence promoted to the launcher default |
| PP6 | REJECTED | RTX 3090 GDN pipeline tuning/fusion | 36 Triton GDN layers contain more untuned PP time than fused MoE | Profiled: GDN is 40.2 ms of a 1,773 ms chunk (2.3%) versus 238.0 ms fused MoE; 5% gate unreachable |
| PP7 | **DONE** | Hybrid cold-expert execution | Full-row transfer is wasteful for experts assigned few tokens | Low-token direct GEMV rejected (SM host reads ~7.5-10 GB/s either way); the host-row DMA staging form is bit-exact and +74.8% warmed PP |
| PP8 | **REJECTED** | Dense/shared-expert format sweep | Ampere may prefer BF16 cuBLAS or W8A8 over W8A16 Marlin at M=1024 | Measured on real tensors: shared-expert GEMMs are ~44 ms/prompt (~1.2% of wall) and the best hybrid (Marlin gate/up + BF16 down) saves 0.16%, so the 5% gate is unreachable; the 172 ms/chunk Marlin is the other PLE/linear-attention GEMMs |
| PP9 | RESEARCH | Fused QSA score/mask/top-k | Avoid full FP32 logits materialization at long prefixes | Prioritize only after 4.5k code PP work |
| PP10 | RESEARCH | Host/PCIe operational audit | Link downgrade, IOMMU, VM scheduling, or CPU affinity may cap transfers | Narrowed: 23 GB/s H2D confirmed (Gen4 x16); remaining item is whether a PCIe or memory topology change unlocks more, and the decode GEMV's 10 GB/s SM host reads |
| PP11 | **DONE** | Batched host-row DMA submission | Replace ~50k Python/Tensor row-copy submissions per chunk with four CUDA batch calls while resident gathers overlap on the current stream | Exact; 1183.4 vs 1110.9 pooled control, +6.53%; default on |
| PP12 | **DONE** | Tail-chunk staging amortization | The 469-token tail pays a nearly full per-layer-chunk staging cost; fewer, larger chunks should cut total staging | Exact; chunk 4608 (one extend for the canonical prompt) 1937.8 +/- 12.9 vs pooled 2048 controls 1286.2 +/- 10.4 tok/s (**+50.66%**, t=98.3); decode unchanged; default promoted via `patches/enable_prefill_chunk_residency.py` |

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

Completed and rejected in the static form after a corrected E2E test.
`patches/prefill_route_dump.py`
(opt-in `SGLANG_PREFILL_ROUTE_DUMP`, installed and left disabled) recorded
routing ids for the canonical code prompt. Review found that the original
analyzer admitted M=3/M=6 launcher warmups and paired chunks with layer sets
on its final cold-row loop. The repaired tool admits nine real chunks (six
2,048 plus three 469), matches each layer to its own resident set, and breaks
presence ties by routing mass. Corrected cold rows are 236.68 mass versus
211.16 presence (-10.8%), with 71.5% resident-set overlap and 66.0% routing
mass coverage.

That result earned a real A/B on the accepted PP7 DMA stack. Five warmed
presence samples averaged **1144.8 tok/s**; two bracketing mass controls pooled
to **1110.0 tok/s**. The +3.1% gain is directionally consistent but below the
5% gate. Decode did not regress (39.94 versus 38.65 tok/s pooled), and both
machine oracles were unchanged. Keep mass placement as the default; retain
`assets/expert_presence_code.pt` only as a reproducibility artifact.

Evidence: [corrected PP5/PP7 placement recheck](logs/pp5_presence_placement_recheck_3090_2026-09-15.md).
The [original analysis](logs/pp5_presence_placement_3090_2026-09-15.md) is
retained but marked superseded.

Restart caveat learned in this line of sessions: after repeated weight loads
the PLE page cache re-warms slowly and the first warmed triplets read ~5% low
(600 then 625.7 versus the 629.7 steady baseline; the PP7 DMA-off control
likewise read 597-645 before holding).  Run at least four warm samples before
accepting any delta under ~5%.

### PP5b — PP11 interaction: canonical-only result

The earlier +3.1% presence gain was measured on PP7.  Re-tested on the accepted
PP11 batched-DMA stack, the only functional variable being
`SGLANG_MOE_PLACEMENT`, one excluded request plus five measured code samples per
restart, mass arms bracketing the candidate:

| Arm | Placement | Scope | Measured PP tok/s | Mean +/- sample SD | Decode mean |
|---|---|---|---:|---:|---:|
| A control | mass | `sglang-1789499399.scope` | 1183 / 1182 / 1190 / 1161 / 1211 | 1185.4 +/- 17.9 | 38.94 |
| B candidate | presence | `sglang-1789499716.scope` | 1254 / 1214 / 1263 / 1220 / 1269 | **1244.0 +/- 25.3** | 41.52 |
| C control | mass | `sglang-1789500049.scope` | 1203 / 1129 / 1200 / 1167 / 1147 | 1169.2 +/- 32.4 | 38.68 |
| Pooled controls | mass | A + C | ten samples | 1177.3 +/- 26.1 | 38.81 |

Presence is **+5.67%** over pooled mass (t = 4.76; smallest presence sample
1214 > largest mass sample 1211), but only +4.94% against the stronger
bracketing control A.  The predeclared pooled bracketing method clears 5%, so
the owner accepted it for the canonical benchmark; the boundary is recorded
rather than smoothed over.
Both oracles are unchanged (`m4_3090_untuned` 0/0, `lp2`
0.168757/0.011632), decode did not regress, and post-workload free VRAM was
2,941 MiB (presence) versus 3,239 MiB (mass).

Review found two validation gaps: the raw client/server output was not retained,
and presence was derived from the exact canonical prompt used for the E2E A/B.
The aggregate arithmetic is correct, but this alone does not establish
performance on unseen code; PP5c closes both gaps below.

Evidence: [PP5b PP11+presence interaction](logs/pp5b_presence_pp11_3090_2026-09-15.md).

### PP5c — captured rerun and held-out code: accepted

PP5c reran the bracket with `tools/capture_pp5b_arm.py` on three fresh servers
(mass-a, presence, mass-c) and added two held-out repository code corpora. Only
`SGLANG_MOE_PLACEMENT` varied; raw client/server output, per-arm environment,
placement hashes/signatures, GPU telemetry, elastic status, and both exactness
oracles were retained for every arm.

Measured-only (warmup excluded), canonical corpus:

| Arm | Placement | Measured PP tok/s | Mean +/- sample SD | Decode mean |
|---|---|---:|---:|---:|
| mass-a | mass | 1185 / 1203 / 1171 / 1177 / 1201 | 1187.4 +/- 14.2 | 42.4 |
| presence | presence | 1240 / 1271 / 1204 / 1260 / 1261 | **1247.2 +/- 26.6** | 40.9 |
| mass-c | mass | 1185 / 1211 / 1191 / 1152 / 1183 | 1184.4 +/- 21.2 | 39.1 |
| Pooled mass | mass | ten samples | 1185.9 +/- 17.1 | 40.7 |

Presence is **+5.17%** over pooled mass (t = 4.68), clearing the canonical gate.
Held-out code is **-0.91%** (`scripts/05_quantize.py`) and **+1.47%**
(`scripts/phase1.py`), i.e. **+0.29%** combined: within run-to-run noise, with no
material regression. Both oracles were identical to the accepted baseline in all
three arms (`m4_3090_untuned` 0/0; `lp2` 0.168757/0.011632), every arm was
error-free, minimum free VRAM was 2,941 MiB, and the elastic status stayed fixed
at S184.

Because the canonical gate clears and neither held-out corpus materially
regresses, presence placement is promoted to the general launcher default with
`patches/enable_prefill_presence_placement.py apply`. The gain is
canonical-specific while held-out code is neutral; re-derive the placement if a
different workload mix becomes primary.

Evidence: [PP5c result](logs/pp5c_presence_placement_3090_2026-09-15.md);
raw arms [mass-a](logs/raw/pp5c_3090_2026-09-15/mass-a/),
[presence](logs/raw/pp5c_3090_2026-09-15/presence/),
[mass-c](logs/raw/pp5c_3090_2026-09-15/mass-c/) (see also the
[raw-evidence checkpoint](logs/raw/pp5c_3090_2026-09-15/README.md)).

## PP8 — shared-expert format sweep (closed)

Rejected on a measured ceiling, without an A/B. The shared expert is INT8
(`*shared_expert.` bits 8) and runs as a separate `Qwen2MoeMLP` through W8A16
GPTQ-Marlin (CUDA shared-expert fusion is off for this model). A fresh profile
of the accepted PP5c stack plus a model-free sweep on the real layer-0 tensors
(`tools/pp8_shared_expert_format_bench.py`, production Marlin packing validated
to 2.1-2.6e-3 relative error) showed:

- The accepted stack is now ~88% GPU-engine-active (SM ~44%, DMA ~45%, largely
  serialized), not the CPU-bound 37% SM-busy PP6 measured; Marlin is ~179
  ms/chunk = ~12% of wall.
- Gate/up prefer Marlin (~1.6x BF16, ~1.3x W8A8); down_proj prefers BF16
  (~1.2-1.3x Marlin); at the 469-token tail BF16 wins all three. W8A8 is worst
  and is unsupported by `torch._int_mm` at the odd tail M.
- The shared-expert GEMMs total **44.3 ms/prompt**; the best format hybrid
  saves **5.9 ms = 0.16%**, and even a free shared expert is only ~1.2% of the
  ~3.8 s prompt. The 5% gate is unreachable.

The ~172 ms/chunk of Marlin is therefore the PLE and linear-attention `in_proj`
GEMMs, not the shared expert. PP8 does not target those; a future format sweep
there is a separate, unrequested item.

Evidence: [PP8 shared-expert format](logs/pp8_shared_expert_format_3090_2026-09-15.md);
raw artifact dir [pp8_3090_2026-09-15](logs/raw/pp8_3090_2026-09-15/).

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

## PP11 — batched host-row DMA submission (completed)

The profile showed the pipeline was CPU-serialized: the single scheduler thread
spent 78% of the span wall inside the MoE staging path while the SMs idled 63%
and the DMA engine did 875 ms of real work. The lowest-risk measured sub-item
was implemented first.

`patches/moe_host_dma_batch.py` caches stable pinned source addresses and
submits each tensor kind with one `cudaMemcpyBatchAsync` call on a dedicated
non-legacy stream. Resident table gathers remain on the current stream and a
single bridge at each side of the layer preserves dependencies. A model-free
258-row benchmark reduced CPU submission from 0.984 to 0.082 ms per tensor kind
and raised transfer from 20.6 to 22.3 GB/s.

Five measured code samples averaged **1183.4 +/- 18.3 tok/s**. The preceding
control was 1120.0 +/- 14.8 and the following control 1101.8 +/- 34.3; pooled
control **1110.9 +/- 26.7**, hence **+6.53%**. The m4 oracle remained exactly
0/0 and lp2 remained at its established 0.168757/0.011632. The launcher now
defaults `SGLANG_MOE_GATHER_DMA_BATCH=1`; set it to zero for the PP7 loop.

The authors' original repeated-sentence sweep was also measured as a secondary
comparison: PP11 produced 258/772/1453/1778/**1909** tok/s at
101/421/1701/6821/10001 prompt tokens versus the batch-off control's
231/694/1524/1678/**1834**. Do not use this easier-routing workload for the
acceptance decision.

Remaining structural ideas—removing the `tolist()`/`item()` synchronization and
overlapping `_unique2`—are no longer required to call PP11 successful. Reopen
them only with a fresh profile of the accepted batch path.

Evidence: [PP11 batched host-row DMA](logs/pp11_dma_batch_3090_2026-09-15.md).

## PP12 — tail-chunk staging amortization (completed)

Accepted. PP6 left the 469-token tail paying a nearly full per-layer-chunk
staging cost, and the later-research list proposed merging or special-casing it.
The canonical 4,565-token prompt ran as two 2,048-token chunks plus that tail;
raising `SGLANG_3090_CHUNKED_PREFILL_SIZE` to 4,608 makes it one extend, so each
layer's selected experts are staged once instead of three times.

Three fresh-server arms via `tools/capture_pp5b_arm.py --skip-heldout`, presence
placement, S184, one variable per restart:

| Arm | Chunk | Measured PP tok/s | Mean +/- sample SD | Decode mean |
|---|---:|---|---:|---:|
| control A | 2,048 | 1296 / 1296 / 1283 / 1289 / 1288 | 1290.4 +/- 5.6 | 42.3 |
| variant | 4,608 | 1943 / 1939 / 1916 / 1950 / 1941 | **1937.8 +/- 12.9** | 42.4 |
| control B | 2,048 | 1287 / 1275 / 1264 / 1286 / 1298 | 1282.0 +/- 12.9 | 42.9 |
| Pooled control | 2,048 | ten samples | 1286.2 +/- 10.4 | 42.6 |

The variant is **+50.66%** over pooled control (t = 98.3; the smallest variant
sample beats the largest control sample). Both oracles matched the accepted
baseline in all arms (`m4_3090_untuned` 0/0, `lp2` 0.168757/0.011632); decode did
not regress; capture minimum free VRAM was 2,433 MiB (variant) versus 2,941 MiB
(controls). A long-prompt pass on a 4,608 server ran 4.6k/9.5k/20.3k/29.3k-token
prompts with no OOM or fault and 2,211 MiB minimum free VRAM.

The launcher default is promoted with the existing helper, now a three-state
ladder (pre-PP3 1024 -> PP3 2048 -> PP12 4608):

```bash
python3 patches/enable_prefill_chunk_residency.py apply
```

`SGLANG_3090_CHUNKED_PREFILL_SIZE` still overrides per start. Caveat: a larger
chunk raises the per-request logit/activation footprint about 0.5 GiB and would
cost request interleaving on a multi-request server; this box runs
`--max-running-requests 1`, so that trade does not apply.

Evidence: [PP12 tail-chunk](logs/pp12_tail_chunk_3090_2026-09-16.md); raw arms
[ctrl-a-2048](logs/raw/pp_tail_chunk_3090_2026-09-16/ctrl-a-2048/),
[var-4608](logs/raw/pp_tail_chunk_3090_2026-09-16/var-4608/),
[ctrl-b-2048](logs/raw/pp_tail_chunk_3090_2026-09-16/ctrl-b-2048/).

## Later research

- Shared-expert format sweep: rejected in PP8 (measured ceiling ~1.2% of wall).
  A format sweep for the larger PLE/linear-attention Marlin GEMMs is a separate
  item.
- At long prefixes, profile QSA's full FP32 score tensor and consider fusing
  score, mask, and hierarchical top-k.  The previously rejected paged-prefix KV
  kernel should not be repeated unchanged.
- Tail-chunk staging amortization: addressed by PP12 (chunk 4,608 makes the
  canonical prompt one extend, +50.66%).
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
   SGLANG=/root/sglang python3 patches/moe_host_dma_batch.py --check
   python3 patches/enable_moe_host_dma_batch.py --check
   python3 patches/enable_prefill_presence_placement.py --check
   SGLANG=/root/sglang python3 patches/prefill_route_dump.py --check
   python3 patches/enable_prefill_chunk_residency.py --check
   SGLANG=/root/sglang python3 patches/moe_config_buckets.py --check
   ```

5. PP11, PP5c and PP12 are accepted; the launcher defaults are PP5c
   prefill-presence placement (`assets/expert_presence_code.pt`) and a PP12
   4,608-token prefill chunk, both selectable back to routing mass and 2,048
   with `SGLANG_MOE_PLACEMENT` / `SGLANG_3090_CHUNKED_PREFILL_SIZE`. PP5b
   remains the canonical-only result; PP8 was rejected on a measured ceiling.
   The remaining defined experiments are the research items PP9, PP10 and the
   later-research list; start one only when the owner continues the PP campaign.
6. After every experiment, update the status table, append a dated result under
   the relevant section, and link the raw log/artifact.

Optional regression checks before future PP work:

```bash
/root/quant/venv-sglang/bin/python -m unittest \
  tools/test_pp5_presence.py tools/test_pp_patch_helpers.py
# With the model server stopped and the GPU free:
PYTHONPATH=/root/sglang/python /root/quant/venv-sglang/bin/python -m unittest \
  gemv/test_moe_host_dma_gather.py
```

State at this update: bulk pread, the 128 MiB recent-row cache, the 2,048-byte
expert-gather tile, host-row DMA staging, PP11 DMA batching, a 4,608-token
prefill chunk (PP12), S184 residency and PP5c prefill-presence placement are
enabled; profiling is disabled. The prefill route dump is applied
and pass-through in `/root/quant/serve-3090.sh` but off unless
`SGLANG_PREFILL_ROUTE_DUMP` names a directory. No server is currently running;
every PP5c arm was stopped cleanly after capture. The RTX 3090, `/dev/nvidia*`,
and driver 580.178.04 were available to the execution context that captured
PP5c, so the earlier sandbox device-node limitation did not apply here. The PP6
profile was taken through the live `/start_profile`
endpoint with no code change or restart.
