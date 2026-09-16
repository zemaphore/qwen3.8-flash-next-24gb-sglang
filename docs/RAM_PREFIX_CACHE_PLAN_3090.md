# RAM-backed hybrid prefix caching for agentic workflows

Date: 2026-09-16
Status: RC0 and RC1 complete (see [RC0](logs/rc0_feasibility_3090_2026-09-16.md)
and [RC1](logs/rc1_gpu_hybrid_3090_2026-09-16.md) reports). Serving source is
unchanged; RC1 enabled the built-in hybrid tree via a separate launcher at a
131,072-token capacity (reported tradeoff, not a promoted default). No RAM tier
or agentic benchmark yet. Objective corrected before implementation (below).
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
| RC1 | DONE | Enable GPU-only hybrid cache on an isolated branch with enough state slots | [RC1 report](logs/rc1_gpu_hybrid_3090_2026-09-16.md): `UnifiedRadixCache` + Mamba extra-buffer; page-aligned hits for repeated/append/branch/A→B→A; reuse Δlogprob below the server's own cold-vs-cold drift (same-slot hits only; not the RC2/RC3 bound); capacity capped 131,072 with 8 mamba slots; RC1-a OOM log not retained; eviction/re-hit untested; no RAM transfer yet |
| RC2 | Implement quantized KV/QSA and recurrent/PLE host round trip | Bit-preserving component tests, remapping/ring tests and incremental continuation comparisons |
| RC3 | Integrate RAM entries, restore, eviction and physical-memory accounting | A→B→A reuse works after GPU eviction; bounded RAM/VRAM and safe abort/failure behavior |
| RC4 | Agentic workload benchmark and long-context validation | Evidence-based latency/capacity tradeoff and accept/reject decision |

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
