# PP13 raw evidence

This directory contains the 2026-09-16 refresh of the accepted PP12 stack and
two follow-up screens. No production asset or launcher setting was changed.

- The root `bench-canonical-*` files are one excluded warmup and five measured
  samples from a fresh accepted-stack server. `bench-profiled.txt` is excluded
  from performance statistics because profiler collection was active.
- `profile/` contains the CPU+GPU stage-filtered trace and text summaries.
- `gdn-format-bench.*` compares real layer-0 GDN tensors across Marlin, dense,
  FP16 and W8A8 paths.
- `moe-config-screen/` contains synthetic-route tuning reports and isolated
  config files. The original E432 M1024 config was used as the production
  reference; none of these files was copied into `assets/`.
- `experimental-assets/` is the complete isolated asset tree used by the tuned
  server. It is retained so the negative result is reproducible.
- `tuned-arm/` and `current-c-arm/` are fresh-server canonical captures in that
  order. Their per-arm manifests intentionally exclude the live `server.log`
  and cover the frozen `server.log.after.txt`; the root manifest was generated
  after both scopes stopped and covers the final live logs too.

The tuned arm was exact but slower: 1830.8 tok/s versus current-c at 1926.0
tok/s and pooled current controls at 1947.1 tok/s. The experimental configs are
therefore evidence only, not proposed defaults.
