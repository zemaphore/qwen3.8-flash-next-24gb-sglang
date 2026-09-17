# RTX 3090 TG research backlog after TG3b

Date: 2026-09-17. Status: research plan, not execution or promotion.
Scope: one RTX 3090 (SM86), 24 GB VRAM, single-consumer linear conversations.
TG3b's bounded agentic acceptance check is owned by another agent. This plan
does not assume its result or change the running baseline.

## Objective and working rules

Improve complete agentic conversation time, with sustained token generation
as a diagnostic and a useful optimization target. Preserve the current model,
KV precision, context capacity and single-request behavior unless a later task
explicitly changes those constraints. No other GPU architecture is a target.

The expected serving profile is main `b02e16a895` plus our patches, radix on,
four Mamba slots, 235,520-token context/pool, INT8/INT4 KV, R1–R3 enabled and
breakable decode graphs. At execution time record the actual promoted source,
launcher and asset hashes: the placement may change after TG3b acceptance.

Research authorization is not permission to run this backlog automatically.
Each selected task should be one 20–30-minute exploration with a stop rule.
No full verification campaign is implicit. Previously waived migration checks
stay waived; **PP14 long-prompt validation remains open and deferred**.
Proposed modifications to PP14 itself must be a separately selected task.

## What the experiments have established

| Evidence | Consequence for research |
|---|---|
| [TG1](logs/tg1_promoted_3090_2026-09-17.md): presence beat the pooled-mass asset across the tested promoted-profile workloads. | Do not switch back to pooled mass or use gate-weight coverage as a speed predictor. |
| [TG2](logs/tg2_gemv_exploration_3090_2026-09-17.md): the larger GEMV launch configuration scored +3.17% and +2.60% against the two confirmation baselines, below its 5% gate. | Park that exact candidate. Another parameter sweep needs a changed regime and new evidence, not another attempt to average away noise. |
| [TG3](logs/tg3_static_decode_placement_3090_2026-09-17.md): short-horizon placement improved code but regressed reasoning; its later reasoning cold-read advantage disappeared. | Train and evaluate across the generation horizon; score each trace under both placements to separate residency from different generated histories. |
| TG3b: 512-token calibration, fresh held-out prompts; estimated cold selections fell 49.9% overall and 53.1% later; measured TG +32.1% code, +12.1% reasoning, balanced +21.7%. | Strong candidate, not proof of a universal improvement. Freeze the asset while the separate 32K agentic acceptance check runs. |
| [TG0 profile](logs/tg0_baseline_3090_2026-09-16.md#4-decode-cost-breakdown): expert GEMV 12.96–17.13 ms/token, dense family 12.29–13.48, QSA/KV about 0.52, trace gaps 4.17–5.22. | These are older-profile clues, not additive savings or the post-TG3b cost table. Refresh only enough profiling to select the next target. |

TG0's family durations overlap. Its PLE/graph-boundary category includes GPU
hyperconnection kernels; it is not a measurement of CPU `pread` latency.
TG3b's held-out data are two excerpts from the same broad repository corpora
as calibration, not an independent sample of real user conversations. Treat
claimed throughput improvements as specific to those measured workloads.

## Cheap scoring protocol

1. Freeze the candidate, baseline and a small prompt set before timing. For
   future calibration work, use fresh acceptance prompts; old held-out prompts
   become development data once they inform changes.
2. Read source or replay existing traces before requesting a GPU window.
   Never interfere with the acceptance agent or a user workload.
3. Use actual shapes, storage dtypes and graph behavior. Record the imported
   source path; a different working directory does not override an editable
   install. Match warmup and prefix-cache state across arms.
4. Microbenchmarks screen candidates. They do not establish server speed.
   Use graph replay when that is the serving path, interleave configurations,
   and keep allocation/JIT out of kernel timing. FP16 checkpoint scales are not
   a substitute for BF16 serving scales without explicitly testing that path.
5. Take only one candidate to a timing boot. A typical screen is two workloads,
   256–512 generated tokens, one or two excluded warmups and three measurements.
   Use a closing baseline only when a promising result needs a bounded tie-break.
6. Predeclare a materiality threshold, normally 5% end-to-end for a code change.
   Primary acceptance is conversation wall time; report TG, cold/warm TTFT and
   individual regressions separately. Numerical checks are focused on the
   changed operation and are not described as full verification.
7. Restore baseline and stop at the time cap. Record promising, rejected,
   inconclusive or blocked; do not automatically extend or promote.

For a baseline token time T and target throughput gain g, the required saving
is T*g/(1+g). At 25 ms/token, +5% needs about 1.19 ms and +10% about 2.27 ms.
These are arithmetic screening thresholds, not forecasts. Estimate savings
from exposed critical-path time, not summed overlapping kernel durations.

## Ranked avenues

Ranks are provisional hypotheses. R0 can reorder them after the accepted
placement changes. Pilot budgets include analysis; larger implementations
require a new task rather than an automatic extension.

| Rank / ID | Avenue | Why investigate | First pilot | Resource/implementation cost |
|---|---|---|---|---|
| Prerequisite R0 | Small post-placement cost refresh | Faster experts may expose dense or host bottlenecks. | 15–20 min; two short decode windows and unprofiled comparisons. | No persistent allocation or serving change. |
| 1 / R1 | Dense projection and LM-head batch-one execution | Dense kernels were comparable in duration to experts before TG3b. | 20–30 min; identify and screen one dominant operation. | No new weight precision; bound scratch explicitly. |
| 2 / R2 | Fuse MoE activation/reduction intermediates | Launch tuning was weak; removing actual work is a different hypothesis. | 20–30 min feasibility/microbenchmark for one epilogue. | Medium code effort; aim to remove buffers. |
| 3 / R3 | PLE CPU lookup and graph-boundary critical path | Existing `pread`/cache work may leave synchronization or dispatch latency. | 20 min attribution; one tiny implementation only if exposed cost is material. | Bounded CPU buffers; no larger cache by default. |
| 4 / R4 | Host launch, scheduler and token-return overhead | Host gaps become proportionally larger after GPU speedups. | 20 min trace/source inspection; one needless synchronization or copy. | Small code change if found; no scheduler flag sweep. |
| 5 / R5 | Cost-aware static placement / per-layer budgets | Equal selection counts need not have equal latency value. | 20–30 min offline scoring only. | No live asset change; nonuniform residency needs allocator work. |
| 6 / R6 | Expert memory ordering and locality | Same resident set can have different cache/transaction behavior. | 20–30 min replay with identical IDs and residency. | Layout-only hypothesis; all pointer consumers need audit. |
| 7 / R7 | Host topology and PLE I/O tail latency | Remote NUMA placement or host pressure could add avoidable waits. | 15–20 min read-only topology/latency diagnosis. | Host-specific; settings unchanged during diagnosis. |
| Conditional R8 | Whole-trace PP/TG compromise | Decode-oriented residency may cost cold prefill. | Offline objective first, only if acceptance identifies that tradeoff. | At most one static compromise asset, no sweep. |
| Longer-term R9 | Fixed-budget temporal expert caching | Routing locality might permit reuse beyond a global static asset. | 30 min trace simulation, no runtime implementation. | High: migration, graph pointers, pinned slots and PP interactions. |
| Longer-term R10 | Reclaim unused backing for opportunistic residency | Reusable physical VRAM could avoid cold reads between growth events. | 20–30 min accounting/design only. | High: allocator and cache-lifecycle changes. |
| Parked R11 | Speculation | Can amortize model work if acceptance beats verification overhead. | Offline acceptance/cost feasibility only after a separate task. | High: recurrent/PLE/KV state and draft memory. |

## R0 — establish the next exposed bottleneck

After the TG3b acceptance owner finishes, capture short warmed decode windows
at roughly 2K and 32K on the actual accepted asset. Reuse `tools/tg_profile.py`;
disable shape/stack recording. Keep profiler measurements separate from client
TG because profiling previously distorted streamed timings.

Attribute expert GEMV, individual dense projections/LM head, hyperconnection,
QSA, CPU PLE, graph replay and scheduler gaps. Separate duration sums, interval
unions and exposed waits. Exit with one candidate and a plausible ms/token
ceiling. Stop if the data cannot justify a material experiment; do not redo
TG0's full matrix. Profiling around representative workloads and re-evaluating
the bottleneck after an optimization follows NVIDIA's
[incremental optimization guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#assess-parallelize-optimize-deploy).

## R1 — dense and LM-head execution without changing weights

Split the broad Marlin/cuBLAS family by operation and actual decode shape.
Select one large contributor: shared-expert projection, linear-attention
projection, hyperconnection, or output head. Check existing batch-one dispatch,
temporary repacking and materialization before implementing a new kernel.

Screen one existing alternative launch/implementation at the same quantized
weights, scales and accumulation contract. A specialized INT8 GEMV is a
possible second-stage project, not a mandate to write one in the pilot.
Fused bias/residual or dequantization epilogues are candidates if intermediates
are exposed on the critical path. Exclude vocabulary pruning, requantization,
skipping logits and precision changes: those change functionality or quality.

Stop if the operation is already near its measured memory limit, savings are
hidden by overlap, or a win requires a persistent expanded-weight copy that
does not fit the current context budget. Existing dense tensors were already
partly repacked to INT8; do not repeat the historical BF16-to-INT8 proposal.

## R2 — eliminate MoE intermediate work

`MoeWNA16Method._apply_gemv` currently produces gate/up output, applies SiLU
and multiplication, invokes down GEMV, then reduces the ten expert outputs.
Two bounded hypotheses: fuse gate/up with activation/gating, or fuse the final
weighted reduction. Pick one after measuring its exposed cost.

Preserve intermediate BF16 rounding where needed. Avoid unordered atomic sums
as a shortcut to an "exact" change. Combining gate and up halves may alter
coalescing and register pressure; a fusion is not automatically faster.
Benchmark current and candidate with actual mixed host/device pointers and
serving scale dtypes under graph replay. Retain identical-input differences.

The TG2 (128,8,128) launch candidate stays parked. Reopen it only if a measured
change in resident mix gives a new reason; use a very small comparison rather
than the old sweep. Configuration parameters describe execution choices, not
guaranteed improvements; see [Triton Config](https://triton-lang.org/main/python-api/generated/triton.Config.html).

## R3 — PLE and the host/GPU handoff

Existing parallel `pread`, bulk lookup, recent-row cache and eager graph break
are already implemented. First measure decode cache hit rate and p50/p95 time
for ID availability, CPU lookup, transfer and wait at consumption. Separate
cold filesystem behavior from warm-cache dispatch overhead.

If lookup is critical, investigate preparing the current token's required PLE
rows earlier while preceding GPU work runs. Prove that IDs and history are
available then; do not predict future generated tokens or reuse stale state.
Audit existing model prefetch helpers before adding duplicate machinery.
Fixed reusable buffers and a persistent worker may reduce dispatch overhead.
Do not enlarge the 128 MiB cache without a measured reuse-distance benefit.

Stop if lookup is already hidden or too small. Any graph restructuring must
preserve fixed-address and stream-dependency requirements; consult
[PyTorch CUDA graph semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
This investigation does not activate PP14 validation.

## R4 — host overhead and streaming

Look for redundant synchronization, repeated metadata conversion, unnecessary
CPU tensor work or excess graph replay dispatch on the actual single-request
path. Verify whether existing shared/router overlap already covers the proposed
work. The old eager shared-overlap experiment was a prefill experiment, not a
missing decode feature ([report](logs/eager_shared_overlap_3090_2026-09-15.md)).

Changing stream response batching may reduce HTTP overhead but changes visible
token latency. If considered, report server decode time, client TG, event
coalescing and first-token latency separately. Faster delivery accounting is
not faster model execution. Do not blindly re-enable overlap scheduling or
continuous decode steps: those have historical negative results.

## R5 — placement value beyond raw frequency

Now that graph-compatible decode capture works, replay accepted placement and
alternatives on the same histories. Train on calibration only; reserve fresh
prompts and later-generation windows for evaluation. Quantify both cold-row
count and which layers carry those misses. If measured miss costs differ,
weight avoided reads by exposed time instead of gate magnitude.

An ambitious offline upper bound redistributes a fixed global row-byte budget
across layers rather than assigning every layer S184. This is not supported
by merely writing a new asset: current elastic control, host-slot floors,
allocation granularity and PP prefetch assumptions constrain per-layer S.
Model those constraints or label the bound unattainable with today's runtime.
Do not assume a constant sum of rows means a constant physical memory footprint.

Stop if held-out benefit is small, concentrated in one early window, or requires
more memory. Do not continuously retrain TG3b while its acceptance is pending.

## R6 — improve locality without changing the resident expert set

Test whether the order of resident/cold expert storage or route processing
changes address locality, cache reuse or warp stalls. Keep the same selected
experts, weights and resident membership. This is distinct from TG3's residency
selection and TG2's launch grid. Begin with saved traces and real tensors.

Remapping expert IDs in execution must preserve output association and reduction
semantics. Loader changes must update every pointer table and elastic/prefetch
consumer. Prefer an offline replay before a source change; drop the idea if
ordering does not move a realistic mixed-residency benchmark.

## R7 — host topology and storage diagnosis

Read NUMA topology, PCIe link state, CPU placement and the location of pinned
allocations. If the host has one NUMA node or already-local placement, discard
the NUMA hypothesis. Check PLE major faults and host memory pressure only when
they coincide with token-latency tails.

Never infer transfer performance from another machine's historical bandwidth.
An affinity/first-touch experiment should restart with fixed placement and
compare identical traces; it is not permission to migrate live pinned pages,
change system-wide settings or drop filesystem caches. Mapped/pinned memory
has tradeoffs and should be budgeted explicitly; see NVIDIA's
[host/device memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device).

## Conditional and longer-term avenues

**R8 — optimize the whole conversation objective.** If the TG3b acceptance
check exposes a cold-prefill penalty, evaluate one static blend of prefill
presence and sustained-decode selection counts. Choose its weight from an
explicit expected conversation shape and calibration data, then freeze it.
If accepted TG3b already improves both full conversations, do not invent this
problem or start a blend sweep. PP14 implementation remains unchanged.

**R9 — temporal expert cache within a fixed budget.** Replay routing traces
with an adaptive subset of the current row budget, retaining a static core.
Simulate compulsory loads, hits, evictions, per-layer locality and migration
bytes. An oracle cache is only an upper bound; compare a causal policy.
Require predicted saved read time to exceed copy, bookkeeping and synchronization
cost by a material margin. Do not infer cross-layer correlation from numeric
expert IDs. Runtime implementation requires graph-safe pointers, startup-pinned
source capacity and compatibility with PP prefetch and RC; no speculative
implementation or new permanent staging cache is authorized here.

**R10 — reclaim backing and temporarily grow residency.** The outstanding RB
backing-trim idea may help between sessions. Audit retained radix pages, live
requests, quantized scale buffers, ring owners and VMM mappings before defining
reclaimable bytes. Extra experts must yield memory before context growth;
sampled free VRAM is not a guaranteed reserve. This is a cache/allocator project,
not a quick TG flag adjustment. No loss of the 230K capacity contract.

**R11 — speculation with explicit break-even.** Keep parked unless a separate
task establishes acceptance and memory feasibility. Historical own-history
NGRAM accepted only 1.11–1.27 tokens/step and was slower; the optional main
patch is untested. A faster target after TG3b raises the bar further.
For accepted output count A per iteration, compare (draft + verify + commit
time)/A with current single-token time. Include PLE/recurrent/KV rollback and
graph costs, not just acceptance. External-context drafts or a quantized MTP
head are distinct larger projects; do not assume a draft fits by reducing
context or expert residency. See the [historical speculation record](SPEC_NGRAM_PLAN.md).

## Ideas not queued without new evidence

- QSA/KV decode optimization: the older measured contribution was small; a
  new post-placement profile must first show a material exposed cost.
- The rejected paged-prefix kernel, eager shared-overlap patch and old
  continuous-decode/overlap-scheduler flag experiments: retain outcomes in
  [HISTORY.md](HISTORY.md); a changed bottleneck is required to revisit them.
- Lower weight/KV precision, vocabulary truncation, expert skipping or a
  smaller model: quality-changing tradeoffs, outside current research scope.
- Larger S or permanent new GPU buffers financed by reduced context: outside
  the current capacity constraint. Static placement wins are memory-neutral;
  new execution paths need explicit peak-memory accounting.
- PP14 validation: still open and deferred, not a prerequisite silently added
  to unrelated experiments and not claimed resolved by their success.

## Next dispatch and record format

Let the TG3b owner finish acceptance and record the decision first. Freeze
whichever asset is actually accepted. Dispatch R0, then choose R1, R2, R3 or
R4 from the measured exposed cost; do not run them all by default. R5–R11 are
a backlog, not an automatically authorized sequence.

For each pilot retain a short dated report: hypothesis, baseline hash, exact
change, time budget, small raw table, finite/numerical spot-check coverage,
memory delta, result and stop decision. Record whether the baseline was restored.
Separate observed speed, modeled ceilings and untested ideas. Update this
backlog only with the outcome; do not rewrite historical logs to imply broader
validation than was performed.
