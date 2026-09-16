# RAM-backed hybrid prefix caching for agentic workflows

Date: 2026-09-16
Status: RC0 and RC1 complete (see [RC0](logs/rc0_feasibility_3090_2026-09-16.md)
and [RC1](logs/rc1_gpu_hybrid_3090_2026-09-16.md) reports). Serving source is
unchanged; RC1 enabled the built-in hybrid tree via a separate launcher.
**Re-scoped 2026-09-16:** the target workload is one consumer driving one
linear conversation with growing context and no forks. For that workload RC1
§5.2 shows a 4-mamba-slot GPU-only profile keeps the full 262,144 pool and
hits on every turn of a 70-turn session to 241,666 tokens (warm turns
1.8–2.5 s vs ~157 s cold re-prefill), so the RAM tier (RC2/RC3) is **parked**
as out of scope.
RC4′ (done) measured the profile against the control.
**Promoted 2026-09-16:** `/root/quant/serve-3090.sh` is now the radix
profile — 4 mamba slots, no `--disable-radix-cache`, pool and context
235,520 (230K), served model name `qwen38-flash-230K` (SHA-256
`e09c4561...`). The previous accepted no-cache 256K launcher is kept verbatim
as the rollback at `/root/quant/serve-3090-nocache-256K.sh` (`11013fb6...`).
Copies and checksums: `logs/raw/promotion_3090_2026-09-16/`. The context was
set to 230K deliberately: RC1-e/RC4′ reached 247–249K with 30–100 MiB free,
so 230K leaves ~15K tokens of margin for generation and activations. The
profile is validated for one linear session per server; stage RB (R1–R5)
is required before a second client or a forking agent uses it. The
multi-conversation design below is retained for reference.
Starting point: accepted RTX 3090 stack; TG0 complete.

## Objective

Avoid recomputing conversation prefixes between agent turns by enabling a
hybrid radix cache with a small GPU hot tier and a bounded RAM tier. Preserve
weights, state precision, and correct branch isolation.

**The optimized metric is complete multi-turn agentic wall-clock time, not
isolated PP or TG throughput.** A cache change is judged on the end-to-end time
of realistic multi-turn traces (shared system/repository prefix, appended tool
results, forks, interleaved conversations), including deferred spill/checkpoint
cost. Warm follow-up TTFT is a diagnostic within that, not the target.

### Metric correction and tradeoff authority

This plan supersedes its earlier PP/TG framing. To make room for prefix caching
and active context, it is authorized to reduce the prefill chunk size and the
GPU expert residency (S) below the accepted PP defaults, and to use additional
host RAM for displaced experts and inactive conversation state. The shared
constraint is the whole trace, not any single stage. Specifically:

- A standalone TG or cold-prefill regression is **not** an automatic rejection.
  Report it explicitly, but accept it when measured agentic wall-clock time
  improves substantially and trace-level correctness holds.
- The old "no reproducible >5% TG or cold-miss PP regression" veto is retired
  for the agentic profile; it was an isolated-throughput gate. A regression is
  only disqualifying when the total trace does not improve.
- The decisive quantity is the crossover: saved prefix recomputation must exceed
  restore/checkpoint/spill overhead plus any additional suffix-prefill and
  generation time. Measure it; do not infer it from link specifications.
- The accepted configuration remains the frozen control and rollback. Any
  tradeoff is validated as an explicitly selectable agentic profile, never
  silently promoted to the default.

Keep one active inference request initially. Multiple conversations can remain
cached without concurrent execution. Restore inactive state to VRAM before
resuming computation; direct attention reads from RAM, NVMe persistence,
distributed prefill/decode, speculation, and new quantization are out of scope.

Preserve the 262,144 context setting as the target, and report any experimentally
necessary capacity tradeoff before promotion.

## Existing machinery and gaps

The inspected serving tree under `/root/sglang/python/sglang/srt` already has:

- `mem_cache/mamba_radix_cache.py`: joint KV/recurrent prefix matching, copying,
  tracking and eviction. Compressed QSA requires page size 64; the radix path
  must use a compatible extra-buffer strategy rather than `no_buffer`.
- `models/qwen4_exp.py`: `_ple_track_targets` snapshots PLE state at the same
  boundaries used by recurrent-state tracking.
- `mem_cache/ple_state_pool.py`: slot lifecycle hooks for PLE copying, reset,
  and host save/restore.
- `mem_cache/allocator/paged.py` and `kv_vmm_backing.py`: lazy physical backing.
  Present reclamation resets backing only when every page is free, a poor fit
  for retained prefixes and fragmented allocations.
- `mem_cache/int4_kv_pool.py`: CPU save/restore explicitly raises
  `NotImplementedError`. `tiered_kv_pool.py` inherits that gap and adds an INT8
  ring whose ownership depends on physical slot IDs.

These are source observations, not an assertion that the combination works.
Reinspect the actual serving revision after TG0 and retain its diff/hash before
implementation. Prefer extending existing hybrid-cache abstractions and host
transfer hooks over creating a second cache manager. Audit hierarchical-cache
support for this exact pool/state combination; do not assume a flag enables it.

## Intended architecture

```text
Request token IDs + model/cache namespace
                  |
           hybrid radix lookup
          /                   \
 GPU prefix hit          RAM prefix hit
          |              allocate + restore
          +---------+---------+
                    |
      copy checkpoint into mutable working slot
                    |
          prefill unmatched suffix → decode
                    |
     publish immutable boundary checkpoint
                    |
  GPU hot cache → RAM cache → evict/recompute
```

Use the longest prefix for which all required state exists at one valid
boundary. Matching KV alone is insufficient: recurrent state cannot generally
be reconstructed at an arbitrary interior token without replay. Share immutable
prefix pages and copy mutable state on a branch. The HTTP API can continue to
accept full conversation histories; reuse is based on exact token prefixes.
Record template/tokenization stability and expose actual matched token counts.

### Snapshot contents and identity

| Component | Required contents |
|---|---|
| Full attention KV | Packed INT4 K/V rows, scales and layout/shape metadata |
| Higher-precision tier | Valid INT8 K/V rows and scales associated with logical tokens, not old device pointers |
| QSA | Compressed keys/indexer data, valid lengths and any incomplete-group/boundary state needed for continuation |
| GDN | All recurrent and convolution states at the same token boundary, unchanged dtype |
| PLE | N-gram token history and short-convolution state at that boundary |
| Identity | Exact prefix token IDs or collision-verified digest, model/weight revision, tokenizer/template and state-format versions, cache namespace |

Inventory each tensor and its lifecycle in the actual source. Include all
runtime configuration that changes stored state or its interpretation in the
namespace. Persist lengths and logical block associations; device pointers and
allocator slots are rebuilt on restore. Never reuse a stale snapshot after
weights, cache format, or relevant model configuration change.

### Tiered KV precision contract: a design gate

The global INT8 ring can be overwritten by other conversations while INT4
backing remains valid. A cache hit can therefore read a different mix of INT8
and INT4 than an uninterrupted request. Storing only INT4 silently changes the
precision history and is not an exact restore.

First preserve the packed representation of every valid higher-precision row
at snapshot time. Map those rows to logical tokens and restore their ownership
only after all layer data is ready. Audit whether arbitrary new physical slots
introduce ring collisions that prevent preserving the snapshot's precision map.
If so, choose and validate a request-local ring or equivalent remapping before
claiming exactness. Never reconstruct INT8 from INT4 and call it preservation.

Compare both byte-level snapshot round trips and cached/uncached incremental
continuation. Distinguish restore fidelity from the baseline's own context- and
allocation-dependent tier behavior. A different cache precision policy would
require a separate, explicit quality decision; it is not this plan's default.

## Memory and transfer policy

Measure RAM available under load, process/cgroup limits, pinned expert memory,
PLE file-cache working set, swap/reclaim pressure, and other workloads. Assign
explicit byte budgets for host snapshots, pinned staging, GPU checkpoints, and
active execution. Do not allocate all `MemAvailable` or pin the whole cache.

**Measured (RC0).** The accepted server already runs at its 56 GiB cgroup cap
(55.77 GiB used, 833 max-events) with ~25.25 GiB non-reclaimable pinned host
expert memory and ~25.3 GiB reclaimable PLE page cache. VRAM has 1.6 GiB idle
free and 40–80 MiB at the accepted 257,456-token smoke. A RAM tier therefore
does not fit on top of the accepted settings: it must be provisioned from the
pinned-expert block, the PLE page cache, or a raised cap. Per-conversation cost
is 7,680 B/token + one ~110 MiB GDN checkpoint + up to 99 MiB ring sidecar
(≈1.06 GiB per 120K prefix). See the [RC0 report](logs/rc0_feasibility_3090_2026-09-16.md).

Use pageable RAM for retained snapshots and a bounded reusable pinned staging
pool for transfers. Account for temporary duplicate copies during spill/restore.
Record achieved bandwidth and CPU packing time on the actual host. Coalesce
transfers by layout where possible; do not perform thousands of tiny transfers
without measurement. PLE remains a competing user of host memory and I/O.

Estimate cache size from measured tensor bytes:

`host bytes = unique retained KV/QSA blocks + checkpoint states + precision
sidecars + metadata + staging/temporary reserves`.

Share common prefix blocks across branches instead of duplicating whole
conversations. GDN state is roughly 108 MiB per FP32 slot from the recorded
dimensions, before smaller side states; verify actual allocation. Current
non-overlap extra-buffer sizing budgets four slots per running request versus
one with radix disabled. Checkpoint capacity needs its own measured reserve.

Track physical VRAM backing and high-water slot usage, not just logical free
token counts. Eviction must enable useful physical reclamation. Start with a
correct bounded pool; add page/granule-aware unmapping or safe compaction only
if needed. Never unmap a granule containing a live or in-flight page. Keep graph
addresses stable and synchronize reclamation with completion events.

## Snapshot lifecycle and failure handling

1. At a supported aligned boundary, capture all state after producer kernels
   complete. A checkpoint length must describe exactly the KV and recurrent/PLE
   state being saved, including whether the last sampled token was processed.
2. Reserve host capacity before spilling. Pin source pages/slots against reuse,
   copy through staging, and publish the RAM entry only after completion.
3. Release GPU references only when the host snapshot is complete and no active
   request or transfer still owns them. Keep common-prefix reference counts.
4. On restore, reserve active working memory and destination pages before any
   destructive eviction. Copy all components, remap locations, and publish the
   usable cache hit atomically after completion.
5. On cancellation, allocation failure, partial copy, or incompatible metadata,
   discard the incomplete destination and retain the valid source. Fall back to
   a shorter complete prefix or recomputation; never serve a partial snapshot.

Initially use bounded LRU eviction with protection for active/in-flight data.
Prefer useful turn/fork boundaries and limit redundant interior checkpoints.
Add cost-aware retention only after measuring restore cost and hit patterns.
Memory pressure should evict cached work before failing an otherwise admissible
request. Host spill must not create an unbounded queue while the user is idle.

## Staged execution

| Stage | Scope | Deliverable / exit gate |
|---|---|---|
| RC0 | DONE | Read-only inventory after TG0; measure memory budgets and state layouts | [RC0 report](logs/rc0_feasibility_3090_2026-09-16.md): compatibility matrix, snapshot schema, precision contract, explicit budgets and test protocol |
| RC1 | DONE | Enable GPU-only hybrid cache on an isolated branch with enough state slots | [RC1 report](logs/rc1_gpu_hybrid_3090_2026-09-16.md): `UnifiedRadixCache` + Mamba extra-buffer; page-aligned hits for repeated/append/branch/A→B→A; GPU hit indistinguishable from recompute under the stack's own drift (teacher-forced suffix; same-slot hits only, not the RC2/RC3 bound); capacity capped 131,072 with 8 mamba slots (16 slots boots at 187,200 with ~46 MiB headroom; original OOM not reproduced); hot tier = 6 conversations, whole-prefix miss on slot eviction; re-hit after eviction works; free VRAM → 4 MiB under retention (lazy backing never released); no RAM transfer yet |
| RC2 | PARKED | Implement quantized KV/QSA and recurrent/PLE host round trip | Only needed for forks/interleaved sessions (out of target workload) |
| RC3 | PARKED | Integrate RAM entries, restore, eviction and physical-memory accounting | As RC2; if ever resumed, must first fix the RC1-d crash modes: allocator must evict on physical pressure and lazy backing must be reclaimable |
| RC4′ | DONE | Linear-session benchmark on the 4-slot profile vs the frozen no-cache control | [RC4′ report](logs/rc4_linear_3090_2026-09-16.md): warm TTFT 0.5–1.0 s at 35K–249K vs 15–114 s cold; TG 34–36 vs 30–32 tok/s (replicate); 12-turn trace 24.1 s vs 123.7 s; 70/70 hits to 247K; new session after a near-limit one 20/20; recommendation: promote for the single-consumer linear workload with RB scheduled |
| RB | R1+R3 VALIDATED | Robustness: fail requests, not the server, under physical VRAM pressure (R1–R5 below) | [RB report](logs/rb_robustness_3090_2026-09-16.md): the 16-slot chunk-2048 churn that crashed the server replays with 38 evict-and-retry events and 0 errors; R2 implemented but unexercised; flags still off in the promoted launcher pending a no-regression run |

## Robustness stage (RB): R1–R5

Measured in RC1-d: with retained prefixes, the server does not degrade, it
dies. Two mechanisms, both in the serving tree, both default-off patches:

- Lazy KV backing is prefix/high-water-mark based (`allocator/paged.py:149-158`
  backs `(max page + 1) × page_size` via `kv_vmm_backing.py:360 ensure_prefix`).
  `allocation.py:188` evicts only for the *logical* shortfall, so logically
  free but never-backed high-index pages are handed out, the commit fails
  against the driver, the hook returns `None`, and `allocation.py:214-227`
  raises `RuntimeError` inside the scheduler loop → SIGQUIT → whole process
  tree exits (2048 arm: 136K evictable, backed pages were sitting in the tree).
- Backing is released only when the pool is fully idle (`paged.py:163`), so
  under retention free VRAM drifts to ~0 and any activation can OOM (3072 arm).

Crash consequences: VRAM returns to 0 MiB and the port frees (no host reboot),
but the server is gone; recovery is the launcher (~3.5–4 min) with an empty
cache. It is manual today because the server is a transient `systemd-run
--scope`.

| Item | Change | Where | Exit test |
|---|---|---|---|
| **R1** evict on physical pressure | When `_lazy_hook` fails: evict `num_new_pages` from the tree, `merge_and_sort_free`, retry with the lowest-index (already backed) pages once, then return `None` | `allocator/paged.py` `alloc_extend`/`alloc_decode` | replay `c2048` churn: 30/30 without crash; hits after eviction |
| **R2** fail the request, not the server | Catch the prefill allocation failure in the scheduler and abort only that request with an error response (decode already retracts, `scheduler.py:3495-3527`; prefill has no equivalent) | `managers/scheduler.py` prefill path, `allocation.py` | oversized request → HTTP error, server healthy, next request served |
| **R3** backing headroom watermark | Refuse a lazy commit that would leave less than a configured free-VRAM floor (e.g. 512 MiB) and evict instead; env `SGLANG_KV_LAZY_MIN_FREE_MB`, default off | `kv_vmm_backing.py` / `_lazy_hook` | replay `c3072` churn: no activation OOM; free VRAM never below floor |
| **R4** release backing on trim | Wire `release_beyond`/`uncommit_beyond` (`kv_vmm_backing.py:367`) to tree eviction so backing above the live+retained high-water mark is unmapped; never unmap a granule with a live/in-flight page; keep graph addresses stable | `kv_vmm_backing.py`, tree eviction hook | free VRAM recovers after a large session ends; no CUDA-graph faults across 100 mixed requests |
| **R5** operational restart | Proper systemd unit with `Restart=on-failure` and a health probe instead of `systemd-run --scope`; alert on restart | launcher / unit file | kill -9 the scheduler → service back within one boot, alert emitted |

**Implementation status (2026-09-16):** R1, R2 and R3 are implemented as
default-off, env-gated patch scripts (`patches/kv_evict_on_physical_pressure.py`,
`patches/prefill_alloc_abort.py`, `patches/kv_lazy_strict_headroom.py`;
flags `SGLANG_KV_EVICT_ON_PRESSURE`, `SGLANG_PREFILL_ALLOC_ABORT`,
`SGLANG_KV_LAZY_STRICT_HEADROOM`) and applied to the serving tree
(modified-tree diff SHA-256 now `57cd72f9...`; with all flags at their
default `0` the code paths are inert). The experimental launcher passes the
three flags through. R5 is drafted as `tools/sglang-3090.service` (not
installed; needs a foreground mode in the launcher). R4 is not started.
Investigation note for R3: the existing `lazy_ensure` watermark
(`SGLANG_KV_LAZY_HEADROOM_MB`, default 1536) is skipped by its own 30 s
rate limit once the expert cache sits at its floor — that is why the 3072
arm committed at 0.03 GB free; R3 re-checks the half-headroom refusal
outside the rate limit.

First validation attempt (16 slots, chunk 2048, flags on,
`raw/rb_3090_2026-09-16/r123_slots16_c2048/`) failed on the first request
and taught two things, both fixed before the second attempt: (a) R3's
half-headroom threshold (768 MB) is above this box's normal ~750 MB idle
free at 16 slots, so it refused the very first commit — R3 now uses an
absolute floor `SGLANG_KV_LAZY_MIN_FREE_MB` (default 256); (b) the failure
landed on a *continuing 2048-token chunk* of a 10K prompt, i.e. a request
with committed KV, which R2 v1 deliberately did not roll back — R2 v2 now
releases such requests through `release_kv_cache(..., is_insert=False)`.
R1 correctly did nothing there (empty tree). Serving diff after the rework:
`b8b80a1a...`.

Order: R1 → R2 → R3, each validated against the retained crash logs as
regression cases and against the RC4′ linear trace for no-regression. R4 only
if R1–R3 leave measurable pressure (it is the riskiest: address stability and
in-flight pages). R5 is independent and can go first. With R1–R3, forks and
extra clients degrade to cache misses (hot tier stays `slots − 2`), which is
the intended behaviour for the linear-session profile.

Each stage should produce a scoped commit and raw evidence before proceeding.
RC0 can conclude a design is blocked; do not build on an unresolved snapshot
completeness or precision contract. Coordinate exclusive serving-tree/GPU use
with TG0; do not restart or patch its server. Keep the frozen no-cache launcher
as rollback. Default-off prototype flags must not alter unrelated runs.

## Validation matrix

### Correctness and isolation

- Identical request twice; append tool results; diverge at a cached fork; return
  to an earlier branch; alternate unrelated conversations A→B→A.
- Boundaries around QSA compression/page alignment, recurrent tracking interval,
  prefill chunk 4,608, and INT8-ring size 8,192; include partial final chunks.
- Partial prefix hits lacking a recurrent checkpoint must fall back to the last
  complete boundary. Verify outputs and actual matched/uncached token counts.
- GPU→RAM→GPU restores to different slot IDs; source slot reuse; shared-prefix
  eviction with another branch alive; metadata/format mismatch.
- Aborted request, interrupted spill/restore, host/GPU allocation failures,
  cache-full churn, and repeated long runs with no leaks or stale-state reuse.
- Require byte identity for packed values/scales and exact state copies before
  evaluating numerical continuation. Use actual incremental decode with fixed
  histories and all recurrent/KV/PLE updates. Inherit TG0's hardened comparison
  rules and record unresolved baseline variability; bulk suffix logprobs and
  greedy text alone are insufficient.

### Agentic performance

Freeze exact-token multi-turn workloads: shared system/repository prefix,
sequential tool-result appends, fork-and-return, and interleaved conversations
whose retained state exceeds the GPU hot tier but fits RAM. Include held-out
prose/reasoning and code. Test useful sizes around 8K/32K/128K and a near-limit
active request with an explicit generation reserve.

Compare three arms: frozen no-cache control, GPU-only hybrid cache, and RAM-backed
cache. Separate cold misses, GPU hits, RAM hits, and eviction misses. Predeclare
warmups/order and use at least five measured traces per main scenario, with
confirmation across boots. Do not count token intervals as independent runs.

Retain matched tokens, recomputed suffix tokens, checkpoint boundary, TTFT,
restore/spill and queue times, TG inter-token latency, total task wall time,
bytes moved, peak/sample memory, PLE cache effects, and failures. Include spill
cost in end-to-end task accounting even if it occurs after the response.

Model the benefit as:

`saved prefix recomputation - lookup/restore/checkpoint/spill overhead`.

Measure it; do not substitute hardware link specifications. A RAM hit is useful
only if it beats recomputation on the target workload. Short-prefix misses
should remain cheap and bypass unprofitable restoration where justified.

## Promotion and stopping criteria

Acceptance is driven by reproducible **end-to-end multi-turn agentic wall-clock
improvement** (>=20% lower full-trace time proposed), correct state reuse,
bounded memory, and a supported context capacity. Warm follow-up TTFT (>=2x on
long-prefix RAM hits proposed) is a diagnostic, not the gate. Report all
scenarios and confidence, not just the best hit.

TG and cold-prefill costs **must be reported explicitly**, but the plan's earlier
automatic >5% veto does not apply to the agentic profile. Quantify the crossover
for every scenario: saved prefix recomputation versus restore/checkpoint/spill
overhead plus extra suffix-prefill and generation time. A RAM hit is accepted
only where it wins on total trace time; short-prefix misses should bypass
unprofitable restoration.

Before RC4, fix scenario weights and regression tolerances with the measured
baseline. Keep repeated runs and confirmation across boots with predeclared
exclusions.

Correctness/state integrity, bounded memory, safe eviction and continued progress
are mandatory. No silent reduction of context, state precision, or workload
support. If near-256K plus caching does not fit, report measured alternatives
(more aggressive eviction, smaller hot tier, smaller chunks, lower expert
residency, or an explicit agentic context profile) and obtain a workload tradeoff
decision before making it the default.

Stop and reconsider if complete snapshots cannot be restored faithfully, RAM
pressure harms PLE or triggers reclaim, restore costs erase the multi-turn gain,
or active-context capacity needs an unacceptable compromise. Do not respond by
adding NVMe/distributed serving to the same project without a new scoped plan.

## Final deliverables

- Updated architecture and exact support matrix; snapshot format and ownership
  invariants; measured RAM/VRAM/cache capacity tables.
- Default-off implementation, focused unit/integration tests, reproducible agentic
  benchmark, complete manifests and retained raw results.
- A validated launcher/profile only after promotion; rollback procedure and
  explicit server/cache state at handoff.
- Updated docs linking the frozen [PP baseline](PP_PERF_PLAN_3090.md) and
  [TG plan](TG_PERF_PLAN_3090.md). Prefix reuse becomes the next agentic milestone
  after TG0; this document does not interrupt or change its active assignment.
