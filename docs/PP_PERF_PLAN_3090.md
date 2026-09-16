# RTX 3090 prompt-processing performance plan

Last updated: 2026-09-16

This is the resumability and status document for improving Qwen3.8-Flash-Next
prompt-processing (PP) performance on the 24 GB RTX 3090 host.  Decode work has
its own plan in [DECODE_PERF_PLAN.md](DECODE_PERF_PLAN.md).

## Current conclusion

The accepted post-PP14 stack measures **2583.4 +/- 65.4 tok/s** on the canonical
4,565-token code prompt. PP14 overlaps next-layer cold-row HtoD with current
MoE compute and is **+36.32%** in its fresh adjacent A/B (2601.2 versus 1908.2
tok/s). The confirming trace shows 901.3 ms / 51.70% compute-DMA overlap versus
10.9 ms / 0.46% on PP13. Exactness is unchanged, a 68,905-token prompt retains
1,221 MiB minimum free VRAM, and the launcher defaults the optimization on for
chunks of at least 2,048 tokens so short interactive requests keep the prior
selected-row path.

Earlier in the sequence, M4 improved its standalone INT2 MoE path by 1.38x
geometric mean but moved end-to-end PP by only 1.8-4.1%. PP11 removed the
per-row Python copy-submission bottleneck with `cudaMemcpyBatchAsync`:
bracketing code controls pooled to 1110.9 tok/s while PP11 reached **1183.4
tok/s (+6.53%)**, so it is accepted and enabled by default. GDN tuning missed
its standalone gate.
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
unchanged decode. The 4,608 chunk is now the general launcher default. PP13
then refreshed the accepted stack at **1968.2 +/- 7.6 tok/s**. Its one-extend
trace is 90.3% GPU-engine-active but contains 738.7 ms of pinned HtoD and
1426.4 ms of compute with only 10.9 ms overlap. A real-tensor GDN format sweep
was too small and memory-heavy, while synthetic high-M MoE configs regressed
the fresh server by 4.94% versus the adjacent control. Both are rejected and
no default changed. PP14 then implemented a reusable full cold-row device
cache and moved each next layer's HtoD behind current-layer compute. It reached
**2601.2 +/- 52.4 tok/s** versus adjacent control **1908.2 +/- 21.8**
(**+36.32%**), preserved both exactness oracles, and passed the long-prompt
memory check. It is accepted and enabled by default above a 2,048-token floor.
PP15 then retuned the routed INT2 MoE on captured real routing and added an
M=4565 entry to the E=384/E=432 sm_86 buckets: the captured canonical reaches
**2687.4 +/- 48.3 tok/s** versus the 2583.4 reference (**+4.0%**, below the
historical 5% gate, exact oracles). The PP14 chunk/threshold retune found no
safe improvement (larger chunks win on medium prompts but exhaust VRAM), so the
4,608-token chunk and 2,048-token prefetch floor remain.

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
below the 5% gate, so routing mass remained the default at that stage. PP7 found
the true wall: the ~46 ms/layer-chunk of expert staging was SM host-read bound
at ~7.5-10 GB/s in every geometry, so the DMA engine (~22 GB/s) was used to
stage the cold rows instead, giving **1111.8 tok/s** warmed, +74.8% over PP3,
bit-identical. Prompt-processing time on the code workload was then ~4.1 s. A full
torch-profiler capture of the PP7 stack (PP6) then measured the steady
2,048-token chunk: 875.2 ms pinned-HtoD DMA staging, 238.0 ms INT2 fused MoE,
172.4 ms Marlin, and only 40.2 ms across all 36 GDN layers; the SMs idle ~63%
of the span while the single-threaded CPU staging path (unique2, tolist syncs,
~50k per-row copies) consumed 78% of the CPU wall and motivated PP11. Batching
the cold-row copies reduced CPU submission
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

For captured PP experiments, use `tools/capture_pp5b_arm.py` on each freshly
started arm. It retains the unparsed output of every request, all live
`SGLANG_*` settings, placement hash/signature, elastic status bound to the live
control path, GPU samples, exactness oracles, and before/after server logs. Its
checksums cover the frozen log snapshot, not a live log that the server may
append during shutdown. Commit the resulting raw directory; a summary table or
systemd scope ID is not a substitute for raw evidence.

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
| PP13 | **REJECTED** | Accepted-stack profile plus GDN/MoE kernel screens | Retuning the now-dominant compute kernels may improve one-chunk PP | Fresh current 1968.2 +/- 7.6 tok/s; dense GDN ceiling ~1.8% for +1.42 GiB; synthetic high-M MoE tiles looked faster in isolation but measured 1830.8 +/- 14.0 vs adjacent current 1926.0 +/- 5.3 tok/s (-4.94%); no default change |
| PP14 | **DONE** | Cross-layer cold-row H2D prefetch | Prefetch fixed-order cold rows for layer L+1 during layer L compute, then GPU-compact after routing | Exact; 2601.2 vs 1908.2 adjacent control (+36.32%); 68.9k prompt passed with 1,221 MiB free; trace overlap 0.46% -> 51.70%; default on at >=2,048 tokens |
| PP15 | **DONE** | Real-routing MoE retune + chunk/threshold retune | Tune the fused INT2 MoE on captured real top-k, and retune chunk/threshold now that prefetch is on | MoE M=4565 entry: canonical 2687.4 +/- 48.3 vs 2583.4 reference (**+4.0%**, below the historical 5% gate), oracles exact, +4.2% steady-state; chunk 6144 medium +11.9% but only 784 MiB free, chunk 8192 80 MiB, threshold 512 regresses; all kept off |

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

## PP13 — accepted-stack profile and kernel screens (closed)

A fresh PP12-stack run reached **1968.2 +/- 7.6 tok/s**. The profiled one-extend
span was 2,390.1 ms with GPU engines active for 90.3% of wall: 1,426.4 ms
compute and 742.7 ms DMA, but only 10.9 ms of overlap. Named routed-MoE and
Marlin kernels contributed 481.6 and 421.5 ms respectively.

Two low-risk compute paths were screened and rejected:

- Dequantized BF16 improves the GDN `in_proj_qkvz` microbenchmark by 1.21 ms
  per layer, only ~43.5 ms / 1.8% per prompt, while adding ~1.42 GiB. The GDN
  `out_proj` does not improve; FP16 casts and W8A8 do not produce a viable
  alternative.
- Synthetic uniform-route tuning found 11-23% fused-MoE kernel reductions at
  E432/E480/E512, but the isolated configs measured **1830.8 +/- 14.0 tok/s**
  on the real server versus current-C **1926.0 +/- 5.3** (-4.94%) and pooled
  current **1947.1 +/- 23.1** (-5.97%). Exactness was unchanged. The synthetic
  route distribution did not predict the padding cost of real skewed routing.

No tuned config entered `assets/` and the launcher remains unchanged.

Evidence: [PP13 current-stack profile](logs/pp13_current_stack_profile_3090_2026-09-16.md);
[raw profile, microbenchmarks and fresh-server arms](logs/raw/pp13_current_profile_3090_2026-09-16/).

## PP14 — cross-layer cold-row prefetch (done)

`patches/moe_cross_layer_prefetch.py` adds one reusable full-expert device cache
per tensor kind. After layer L queues its fused MoE, the dedicated stream
copies every fixed cold row for L+1. The next layer's existing table-gather
kernel compacts resident and cached cold rows into the unchanged staging
layout. A release event makes one cache sufficient; the planned alternating
buffer was unnecessary after compaction.

The fresh adjacent control measured **1908.2 +/- 21.8 tok/s** and PP14 measured
**2601.2 +/- 52.4** (**+36.32%**). A second fresh server using the final guarded
launcher default reached **2583.4 +/- 65.4**. Both exactness oracles are
unchanged. The trace reduced the extend span 2390.1 -> 1743.3 ms while
compute/DMA overlap rose 10.9 -> 901.3 ms. A 68,905-token pass completed at
2,473 tok/s with 1,221 MiB minimum free VRAM. The 637 MiB cache is allocated
when a large prefill first initializes streamers.

The feature defaults on via `patches/enable_moe_cross_layer_prefetch.py`, but
only for chunks at or above `SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS=2048`.
`SGLANG_MOE_COLD_PREFETCH=0` restores the prior selected-row path. Requested
current-best secondary results are also retained: llama-benchy `pp2048 @
d2048` **1218.09 +/- 42.17 tok/s**, and the authors' default repeated-sentence
tool reached **2468 tok/s** at 10,001 prompt tokens.

Evidence: [PP14 result](logs/pp14_cross_layer_prefetch_3090_2026-09-16.md);
[raw arms, long-context pass, trace and secondary benchmarks](logs/raw/pp14_cross_layer_prefetch_3090_2026-09-16/).

## PP15 — real-routing MoE retune plus chunk/threshold retune (done)

After PP14 a fresh extend profile is compute-bound: 1595.9 ms compute union in a
1743.3 ms span with 901.3 ms (95.6%) of the 942.8 ms DMA union already overlapped
and only ~106 ms without a GPU engine event. The largest kernel is the routed
INT2 MoE at 536.0 ms, so PP15 tuned it on **real** routing rather than PP13's
uniform synthetic distribution.

`patches/prefill_route_dump.py` captured two canonical 48-layer top-k dumps.
Real compact `E` is 406-502 per layer and the token distribution is skewed
(layer 24: max 2691, mean 89, 457 nonzero). `tools/tune_moe_int2_real.py`
replays those ids through `fused_experts_impl` and found a stable better pair for
the M=4565 lookup (gate/up `BLOCK_SIZE_M=64, N=64, K=32, GROUP=8`; down
`BLOCK_SIZE_M=64, N=128, K=32, GROUP=8`), 1.13-1.21x (mean 1.16x) across layers.
The entries were added only to the E=384/E=432 sm_86 buckets at M=4565.

The captured accepted arm measured canonical **2687.4 +/- 48.3** vs the PP14
accepted reference **2583.4 +/- 65.4** (**+4.03%**); a same-session 10-sample
control/variant pair excluding the documented re-warm transient gives 2585.0 +/-
19.7 vs 2693.9 +/- 13.7 (**+4.21%**). Both oracles are unchanged. This is below
the historical 5% screen gate and is recorded as such. Required characterization:
llama-benchy `pp2048 @ d2048` **1246.81 +/- 40.96**, `tg256 @ d2048`
**29.39 +/- 1.23**; the authors' unmodified tool at 10,001 tokens **2,524 tok/s**
(PP14: 2,468; authors' published: 2,271).

The chunk/threshold retune is closed negative: chunk 8192 made the 14,446-token
prompt +17.7% but left **80 MiB** free on 34,661 tokens, chunk 9216 exceeds the
`int8ring_int4` 8,192-slot ring and crashes, chunk 6144 gives +11.9% medium but
leaves 784 MiB and low-memory lazy kernel loads, and a 512-token prefetch
threshold regresses small tails. Chunk 4,608 / threshold 2,048 stay the defaults.
The full-table gather (remove the `torch.unique` sync on prefetched layers,
`patches/moe_prefetch_full_table.py`) is exact but only +0.9% alone and
indistinguishable from the MoE config when stacked, so it is opt-in default-off.
A first-layer prefetch for layer 0's 18.4 ms serial HtoD
(`patches/moe_first_layer_prefetch.py`) measured +0.87% (t=2.26) in a 10-sample
bracket but only +0.3% pooled, inside the noise band; it is opt-in default-off.

Evidence: [PP15 result](logs/pp15_real_routing_moe_3090_2026-09-16.md); raw arms
and captures in `logs/raw/pp15_prefill_retune_3090_2026-09-16/` and
`logs/raw/pp15_real_routing_moe_3090_2026-09-16/`.

## Later research

- Shared-expert format sweep was rejected in PP8 (measured ceiling ~1.2% of
  wall); the larger GDN format path was rejected in PP13 (~1.8% ceiling for
  +1.42 GiB).
- If fused-MoE config tuning is revisited, replay retained real top-k routing
  distributions. PP13 showed that uniform random routing selects harmful large
  tiles even when the isolated kernel timing looks convincing.
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
   SGLANG=/root/sglang python3 patches/moe_cross_layer_prefetch.py --check
   SGLANG=/root/sglang python3 patches/moe_prefetch_full_table.py --check
   SGLANG=/root/sglang python3 patches/moe_first_layer_prefetch.py --check
   python3 patches/enable_moe_cross_layer_prefetch.py --check
   python3 patches/enable_prefill_presence_placement.py --check
   SGLANG=/root/sglang python3 patches/prefill_route_dump.py --check
   python3 patches/enable_prefill_chunk_residency.py --check
   SGLANG=/root/sglang python3 patches/moe_config_buckets.py --check
   ```

5. PP11, PP5c, PP12 and PP14 are accepted; the launcher defaults are PP5c
   prefill-presence placement (`assets/expert_presence_code.pt`) and a PP12
   4,608-token prefill chunk, plus PP14 prefetch above a 2,048-token floor.
   They remain selectable with `SGLANG_MOE_PLACEMENT`,
   `SGLANG_3090_CHUNKED_PREFILL_SIZE`, `SGLANG_MOE_COLD_PREFETCH`, and
   `SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS`. PP5b remains the canonical-only
   result; PP8 and PP13 were rejected. PP9 and PP10 remain research items.
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
prefill chunk (PP12), S184 residency, PP5c prefill-presence placement, PP14
cross-layer cold-row prefetch above 2,048 tokens, and the PP15 M=4565 INT2 MoE
config entries for the E=384/E=432 buckets are enabled; profiling is disabled.
`patches/moe_prefetch_full_table.py` is installed but default-off. The prefill route dump is applied
and pass-through in `/root/quant/serve-3090.sh` but off unless
`SGLANG_PREFILL_ROUTE_DUMP` names a directory. No server is currently running;
the PP14 prototype, adjacent control, and final-default scopes were stopped
cleanly.
The RTX 3090, `/dev/nvidia*`, and driver 580.178.04 were available to this
execution context. The PP14 profile was taken through the live
`/start_profile` endpoint with no production code or launcher change.
