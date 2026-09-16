# Step 5 — candidate validation (RTX 3090)

Date: 2026-09-16 (21:15–21:30 UTC). Execution-plan step 5 of
[`docs/SGLANG_MAIN_MIGRATION_PLAN.md`](../../../SGLANG_MAIN_MIGRATION_PLAN.md).
Candidate = `/root/quant/sglang-main` at `port/main-3090` (`b114264c01`), served
by `/root/quant/venv-sglang-main` through
[`/root/quant/serve-3090-main-candidate.sh`](/root/quant/serve-3090-main-candidate.sh)
(port 30011, separate elastic control and `SGLANG_CACHE_DIR`, log
`/root/quant/logs/server-main-candidate.log`). The frozen control
`/root/quant/serve-3090.sh` was stopped; no default was changed.

## Bring-up

Candidate booted and passed `/health`. Effective configuration matches the
frozen control (values from the candidate boot log):

| Setting | Candidate | Control |
|---|---|---|
| Load weight elapsed | 190.62 s | 170.48 s (control) |
| Loaded memory | 18.41 GB | 18.41 GB |
| Page size | 64 (QSA) | 64 |
| KV capacity | 423,488 → 235,520 @ safety 0.77 | same |
| KV backing | floor 4096, margin 2048, 6.8 KB/token | same |
| KV store | uint8, 235,520 tokens, K/V 0.76 GB | same |
| KV tiers | ring R=8192 int8_g64 over int4_g32, 99 MB | same |
| Mamba | 4 slots (0.01 + 0.53 GB) | same |
| Radix | UnifiedRadixCache, hybrid_ssm | same |
| Elastic | 48 layers S=184 | same |
| Graphs | decode breakable 1.60 s; prefill disabled | decode breakable; prefill disabled |

Launcher amendment for main: `--cuda-graph-backend-prefill disabled` is passed
explicitly. Main no longer auto-disables prefill graphs for a multimodal model
(the control relied on that rule), and capturing them is off-profile.

## Numerical behaviour — matched A/B

`tools/logprob_diff.py save cand3090` on the candidate, then the frozen control
checked against that reference on the same box:

```
MAX 0.0000   MEAN 0.00000   over 450 forced tokens (3 prompts x 150)
```

Bit-exact. Reference kept at `tools/logprob/cand3090.json`.
Against the historical `tools/logprob/lp2.json` the candidate reads
MAX 0.1688 / MEAN 0.01163, i.e. cross-config noise well under the 0.5 / 0.05
kernel-class caps (that reference is not a same-box control).

## Performance and capacity probes

`tools/rc1_reuse_checks.py --url http://127.0.0.1:30011`:

| Probe | Candidate | Frozen control (RB attempt 4 flags-on) |
|---|---|---|
| TG 28920, 256 gen (first, cold JIT) | 29.01 tok/s, TTFT 2.351 s | — |
| TG 28920, 256 gen (warmed) | 29.77 tok/s, TTFT 0.664 s | 31.3 tok/s, TTFT 0.58 s |
| 12-turn trace | 12/12, warm median 1.987 s | 12/12, median 1.95 s |
| Growing session 12 turns | 12/12 hits, final 42,799 | — |
| Two sessions A (40) then B (20) | A 40/40, B 20/20 | A 64/64 to 231,310; B 20/20 |

The 12-turn trace matches within ~2%. Single TG samples differ by ~5%; the
control's own boot-to-boot TG spread is larger (RC4′ 35.7/35.0/34.3 vs promoted
boot 30.8), so repetition across boots is still required before a tolerance can
be fixed. `free_vram` fell to 130 MiB during session B with no crash;
`KV lazy commit` fired 61 times; 0 `OutOfMemory`/`Traceback` lines.

## Environment and port fixes made during bring-up

- Venv: created `nvidia/cu13/lib64 -> lib` and
  `lib/libcudart.so -> libcudart.so.13` in `venv-sglang-main` (the JIT linker
  needs both; see `step2-venv.md`).
- `qsa_indexer._qsa_ensure_rope`: use `get_model().context_length` (main retired
  `get_global_server_args()`).
- paged allocator `_lazy_idle_check`: idle test is
  `len(free_pages) + num_staged_pages >= num_pages` (main has no
  `release_pages`).

The two source fixes are commit `b114264c01`; the series and flattened patch were
regenerated and re-verified (all three trees `9f505c06…f1d`).

## Gate status

| Gate | Status |
|---|---|
| Clean reproduction | passed (flat + series + tree identity) |
| Static/API compatibility | passed |
| Checkpoint bring-up | passed |
| Numerical behaviour | passed (bit-exact vs control) |
| Prefix cache | passed (repeat/append, next session, eviction) |
| Pressure handling | partial (no crash near-limit; churn reproducer across boots not repeated) |
| Performance/capacity | partial (trace matched ~2%; TG repeat across boots and near-limit capacity pending) |

## Remaining before promotion

1. Pressure: the RC1-d churn reproducer with R1 v3 / floor 64 across boots.
2. Performance: repeat TG/trace across boots and fix tolerances; near-limit
   (≈231K) session.
3. Then step 6: install the candidate launcher as a separately named default
   and keep `serve-3090.sh` as rollback. The frozen control is currently running
   as the default server (port 30001).
