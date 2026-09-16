# Agent assignment: TG0 baseline and profiling

Execute TG0 from `docs/TG_PERF_PLAN_3090.md` and stop at its exit condition.
Read applicable AGENTS.md instructions, inspect the current Git state, and
preserve unrelated work. Read the PP wrap-up closure review and the historical
decode plan's status note. Baseline evidence was reviewed through `52b2238`;
record the actual revision and serving tree you use.

The objective is a reliable single-stream RTX 3090 TG baseline, a measured
decode cost breakdown, and a validation harness suitable for future changes.
PP optimization is closed. Do not start placement tuning, kernel optimization,
speculation, quantization, or a flag sweep. Keep accepted defaults unchanged.
Tooling fixes, focused tests, and reversible diagnostic instrumentation are in
scope. Verify ownership of the GPU/server before disruptive operations; never
stop unrelated workloads or patch serving source while another agent uses it.

1. Snapshot the accepted server configuration, patched source, asset hashes,
   software/GPU settings, and commands. Verify breakable graph replay and the
   in-place expert GEMV path. Historical Blackwell results are not 3090 evidence.
2. Follow the TG0 matrix and predeclared warmup/order policy in the plan:
   prose/reasoning/code at actual ~2K/32K/128K input lengths, 512 output tokens,
   one warmup plus five measured requests per cell; second-boot short cells
   with one warmup plus three measured requests; one 257,456-input/512-output
   capacity smoke. Freeze prompts/token IDs first, retain every sample and output,
   and keep separate boot summaries. Do not reuse the 16-token PP depth sweep as
   the TG baseline.
3. Retain monotonic stream timestamps and cumulative token counts. Report TG
   excluding TTFT, per-request throughput, median/p95 token latency where actual
   token timing is available, TTFT, memory, clocks/thermals, and errors. Coalesced
   stream events are not individual token latency observations.
4. Profile representative warmed short/long decode windows separately, without
   stack/shape tracing. Quantify profiling overhead. Attribute critical-path
   ms/token to expert reads/GEMV, dense kernels, attention/KV, GDN, PLE/graph
   boundaries, and scheduler/sampling. Separate overlapped work and idle gaps.
   Collect routing/residency diagnostics outside headline timing; report actual
   selection coverage and estimated traffic separately from gate-weight mass.
5. Harden `tools/long_logprob_oracle.py`: require identical inputs/continuations,
   complete finite outputs and consistent metadata, retain per-token data, and
   add focused tests for invalid/truncated/mismatched results. Its fixed 0.05
   threshold fails the baseline itself; separate reporting from a justified
   acceptance policy, without choosing a threshold to hide observed outliers.
6. Perform the bounded PP14 diagnostic described in TG0: three accepted and
   three prefetch-off observations per existing long prompt in balanced boot
   order. Explain the unresolved 2.797 delta versus 1.781 baseline observations
   using matched evidence. A quieter repeat alone does not prove equivalence.
   Restore prefetch-on. If still inconclusive, report a validation blocker and
   stop this diagnostic rather than expanding it indefinitely.
7. Establish sustained decode-path numerical/state checking using fixed histories
   and actual incremental decode, or targeted comparisons with explicit coverage
   limits. Include recurrent/PLE/KV state and an 8,192-token ring crossing. Bulk
   input-suffix logprobs and greedy text similarity cannot substitute for this.
   If a full harness is too invasive, document the gap and block relevant future
   promotions; do not turn TG0 into a serving-engine rewrite.
8. Commit the tooling, tests, raw evidence/manifests, and a dated TG0 report.
   Update the plan with completion or blocking evidence. Run only appropriate
   checks, remove temporary instrumentation, and leave the accepted configuration
   restored. Record whether the server is running or stopped.

Finish with baseline results by workload/context, the ms/token cost table,
validation limitations, commit IDs, and one recommended next experiment with a
measured rationale and estimated ceiling. Do not execute TG1 or TG2 in this task.
