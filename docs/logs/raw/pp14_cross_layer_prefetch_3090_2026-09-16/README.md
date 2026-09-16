# PP14 raw evidence

This directory contains the 2026-09-16 cross-layer cold-row prefetch
experiment, its adjacent flag-off control, the final guarded-default
validation, a long-context check, a profiler trace, and requested secondary
benchmarks.

- `prefetch-arm/` is the first full canonical capture with
  `SGLANG_MOE_COLD_PREFETCH=1`: one excluded request, five measured requests,
  both exactness oracles, telemetry, environment, launcher and status
  snapshots, and frozen server logs.
- `control-arm/` is the fresh adjacent flag-off control with the same evidence.
- `default-validation-arm/` is a fresh capture after adding the 2,048-token
  activation floor and promoting both PP14 variables into the launcher.
- `profile/` contains the stage-filtered CPU+GPU trace. `span-summary.txt`
  reports interval unions clipped to the CPU extend span; `top80.txt` is the
  general event summary.
- `longctx-60000.txt` is a 68,905-token functional pass; its telemetry is in
  `longctx-nvidia-smi.csv`.
- `control-2048-*`, `default-2048-*`, and `default-1024-*` are short threshold
  probes. They are characterization, not the canonical acceptance statistic.
- `final-threshold-smoke/` validates the final allocation guard on a fresh
  server: the first 1,044-token request produced no cache-allocation log and
  left 3,639 MiB free; the first large request allocated exactly four cache
  tensors, and the following 4,565-token request reached 2,621 tok/s.
- `benchy-current-best.txt` is the requested three-run natural-corpus
  `llama-benchy 0.4.0` result on the final current-best stack.
- `author-default-bench-current-best.txt` is the requested unmodified
  `tools/bench_speed.py` default run on that same stack.
- `prefetch-server-final.log` and `default-validation-server-final.log` cover
  activity performed after each capture tool froze `server.log.after.txt`.

Each arm manifest intentionally excludes its live `server.log`. The recursive
root manifest was created after all three scopes stopped and covers every
retained file, including final live logs and traces.
