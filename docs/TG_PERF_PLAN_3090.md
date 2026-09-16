# RTX 3090 token-generation performance plan

Date: 2026-09-16
Status: planned; TG0 has not run. Baseline evidence reviewed through `52b2238`.

## Objective and constraints

Improve single-stream token generation (TG) on the 24 GB RTX 3090 while
preserving model weights, KV precision, 262,144-token context configuration,
and the accepted PP behavior. Keep `max-running-requests=1`. Aggregate batching,
new quantization, and speculation are separate projects, not initial TG scope.

PP optimization is [closed](PP_PERF_PLAN_3090.md). Its defaults are the control:
S184 presence placement, chunk 4,608, PP14 prefetch at >=2,048, PP15 MoE configs,
accepted PLE cache/DMA settings, breakable decode graphs, and tiered INT8/INT4
KV. Full-table gathering, first-layer prefetch, and QSA host lengths remain off.
Capture effective settings rather than relying on defaults or historical logs.

The [old decode plan](DECODE_PERF_PLAN.md) describes another baseline and largely
completed work. N-contiguous pointer-table GEMV for M<=16, breakable CUDA graphs,
host fixes, and expert placement already exist. PP's M=4565 tuning and cold-row
staging do not optimize the normal decode GEMV path, which reads device or
pinned-host expert rows in place. Do not reuse historical GPU bandwidth or
56 tok/s figures as current 3090 measurements.

## Starting evidence and limitations

- Final PP capture: canonical 2666.8 +/- 12.5 tok/s; held-out 2590.0 +/- 6.9 and
  2581.3 +/- 13.6. Corresponding TG means were 41.4, 38.0, and 33.8 tok/s.
- The depth sweep accepted 257,456 actual input tokens but generated only 16
  tokens. Sampled free VRAM reached 40 MiB at one depth and 80 MiB at the
  largest accepted depth. These are not robust TG baselines or guaranteed
  memory reserves.
- Presence placement retains 184/512 experts per layer (35.9%). Its calibrated
  routing-mass coverage is 66.0% versus 84.3% for mass placement. The status
  value `mass_covered=0.4665` reflects an encoded placement score, not routing
  mass. Transfer bytes depend on selection counts and row sizes, not gate mass.
- Long-prompt oracle drift is unresolved: baseline max 1.781 versus one
  prefetch-off comparison at 2.797. The oracle scores an appended input suffix,
  not sustained decode. See the [closure review](logs/pp_wrapup_3090_2026-09-16.md).

## Stages

| Stage | Status | Work | Exit condition |
|---|---|---|---|
| TG0 | TODO | Baseline, decode profile, validation harness audit | Reproducible measurements, attributed costs, explicit validation status, ranked next experiment |
| TG1 | BLOCKED on TG0 | Presence versus mass placement at fixed S184 | Controlled TG/PP tradeoff and measured selection/traffic coverage |
| TG2 | BLOCKED on profile | One bottleneck-specific engineering experiment | Repeated end-to-end result and path-specific correctness evidence |
| TG3 | BLOCKED on candidate | Combined regression and capacity checks | Accept or reject candidate with retained evidence |

## TG0: establish the baseline

Use the [standalone assignment](prompts/TG0_AGENT_PROMPT.md). No production
optimization or default changes in TG0. Reversible instrumentation and tooling
fixes are allowed; remove instrumentation from the final served state.

### Measurement protocol

Freeze and hash prose, reasoning, and code prompts plus token IDs before timing.
Build actual input lengths by tokenization: approximately 2K, 32K, and 128K for
each class. Use natural held-out material; identify any repetition explicitly.
Generate exactly 512 tokens per request with EOS ignored, fixed sampling
settings, one request at a time. Preserve outputs and actual counts.

For each of the nine cells: one explicitly excluded warmup, then five measured
requests. Predeclare a balanced order across classes/depths and do not change
exclusions based on results. Repeat the short-context cells on a second boot
with one warmup and three measured requests each to expose restart variability.
Keep boot summaries separate. Run one capacity smoke at 257,456 actual input
tokens plus 512 generated tokens; label it a smoke, not a statistical result.
Do not flush caches or alter clocks between cells; record their state.

Report per-request TG as `(last_count-first_count)/(last_time-first_time)` with
a monotonic clock, excluding TTFT. Retain timestamps/counts, TTFT, median/p95
inter-token latency, throughput mean/SD across requests, and minimum sampled
free VRAM. Validate event counts: if streaming coalesces tokens, report event
latency separately and use server-side token timings for true token percentiles.
Do not treat individual token intervals as independent experimental replicates.
Record clocks, power, thermals, host pressure, failures, and retractions.

### Profile and attribution

Capture short warmed decode windows on representative short and long contexts,
separately from timing samples, without stack/shape tracing. Verify actual graph
replay and absence/presence of fallback. Compare profiled/unprofiled step times
to quantify perturbation. Attribute wall ms/token to critical-path contributions:
expert GEMV and host reads, dense/Marlin, QSA/KV, GDN, PLE/graph breaks, scheduler
and sampling. Distinguish summed durations from interval unions and overlaps;
do not turn engine-active time into hardware efficiency.

In separate diagnostic runs, measure decode selections per layer/expert and
resident versus host selections, routing mass, and estimated bytes. Label
estimates and compare with traffic counters if available. Do not instrument
every token during headline timing or add synchronizations to that path.

### Validation readiness

Harden the existing long oracle to require identical prompt and continuation
IDs, exactly the expected number of finite logprobs, and matching metadata.
Missing/truncated/NaN results must fail; add focused CPU tests for these cases.
Preserve raw per-token data from every arm. Separate measurement/reporting from
pass/fail: the existing fixed 0.05 maximum threshold contradicts baseline drift.

Characterize same-configuration variability and the unresolved PP14 outlier with
matched repetitions (three accepted and three prefetch-off observations per
existing long prompt, balanced boot order). This is a diagnostic control only;
restore prefetch-on and do not benchmark it as a new optimization. Compare
per-prompt/per-position distributions, not only a global maximum across prompts.
Do not discard early suffix positions or enlarge a gate simply to obtain a pass.
Stop after this bounded set if inconclusive and record a validation blocker.

Establish a decode-path numerical check with fixed token histories advanced one
decode step at a time through the actual recurrent/KV/PLE state and graph path,
or equivalent targeted kernel/state comparisons with clearly stated coverage.
A bulk teacher-forced input suffix is not that check. Greedy output comparisons
are diagnostics only. Include a KV ring-boundary crossing around 8,192 tokens.
If full decode checking requires substantial serving changes, document that gap
and block relevant promotions rather than implementing an unbounded harness.

### Deliverables and stop rule

Commit tooling/tests, raw timing and profile evidence, manifests, and a dated
TG0 report. Manifest repository/serving revisions and diffs, untracked serving
module hashes, launcher, effective environment, assets, software, GPU settings,
commands, prompt hashes, and boot identity. Record instrumentation state.

End with a cost table in ms/token, uncertainty/coverage limits, TG0 validation
status, and one ranked next experiment with a plausible end-to-end saving.
Do not start TG1/TG2 automatically. If the profile cannot support a material
hypothesis, recommend stopping rather than a blind sweep.

## TG1: placement tradeoff

Compare presence and mass assets with identical S184, graphs, PP settings,
workloads, and memory budgets. Use bracketed controls across boots; keep raw
and normalized per-workload results. Measure PP on canonical and held-out
corpora as well as TG. Earlier PP-era TG readings did not establish a winner.
Do not infer a speedup from 66% versus 84.3% calibration mass alone.

Do not implement dynamic PP/TG residency switching unless a measured static
benefit justifies migration cost, pinned-memory requirements, and graph-safe
pointer updates. Do not increase residency at the expense of long context.

## TG2: choose from measured costs

- Host/graph gaps dominant: inspect PLE boundary, graph segmentation, and
  scheduler overhead. Check source compatibility before any flag experiment.
- Expert cost dominant: tune the existing GEMV on real decode routing and mixed
  residency. Separate host-read limitations from device compute. No all-device
  synthetic result may stand in for the production mix.
- Depth-dependent QSA/KV cost dominant: target the measured decode kernel;
  preserve KV precision and context capacity.

Estimate ceilings against the current trace, accounting for overlaps. Select
one variable or bounded implementation at a time. NGRAM's historical low
acceptance/eager verify and MTP's memory cost make speculation deferred work.

## TG3: promotion policy

Predeclare acceptance before the candidate runs. Target >=5% repeatable
single-stream TG improvement on the balanced workload suite, supported across
boots and outside observed noise. Smaller improvements need an explicit
justification; do not silently waive the gate. Report every workload separately.
Require no reproducible >2% PP regression or TG class regression; inconclusive
measurements are not a pass. A workload-specific tradeoff remains opt-in unless
the user explicitly accepts it. Preserve context/precision and validate near-limit
operation with no added failures/retractions. Establish path-specific numerical
and state correctness before promotion; unresolved relevant baseline anomalies
block a correctness claim. Keep the frozen PP configuration as the rollback.
