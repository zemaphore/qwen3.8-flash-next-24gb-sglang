# series-main — Qwen3.8-Flash-Next serving series for upstream main

Target base: **`b02e16a895add01a0cfe24bb74922de92ab4d895`** (upstream `main`;
model support landed as `52fecfdf0908dca24f4c6799ff5967125cc4110e`, PR #37500).
Hardware scope: one RTX 3090, SM86, 24 GB (as in the migration plan).

This is the main-based port of the frozen 3090 control
(`docs/logs/raw/migration_3090_2026-09-16/control-freeze.md`). It is not the
historical `series-base/` / `series-q4head/`, which stay unchanged.

## Ordered patches (default series)

Apply with `git am` in file-name order on a clean checkout of the base.

| # | Patch | Commit subject | sha256 |
|---|---|---|---|
| 1 | `0001-fix-qwen4-CPU-offload-correctness-breakable-graphs-m.patch` | `fix(qwen4): CPU-offload correctness, breakable graphs, mmap PLE table` | `670b9d65…b1c785` |
| 2 | `0002-feat-moe_wna16-2-bit-experts-expert-streaming-N-cont.patch` | `feat(moe_wna16): 2-bit experts, expert streaming, N-contiguous GEMV` | `a6579494…e833b1` |
| 3 | `0003-feat-mem_cache-elastic-VMM-expert-row-arenas-lazy-KV.patch` | `feat(mem_cache): elastic VMM expert row arenas, lazy KV, quantized QSA pools` | `ba701590…1bdf2b9` |
| 4 | `0004-fix-mem_cache-R1-evict-on-pressure-and-R2-prefill-al.patch` | `fix(mem_cache): R1 evict-on-pressure and R2 prefill-alloc abort` | `12407576…74603a` |
| 5 | `0005-feat-moe-prefix-cache-PP-model-wiring-optional-share.patch` | `feat(moe): prefix-cache PP model wiring, optional shared-expert overlap` | `35ac9d3d…042882` |
| 6 | `0006-fix-main-runtime-API-adjustments-found-at-candidate-.patch` | `fix(main): runtime API adjustments found at candidate bring-up` | `abc32def…3089a7` |

Flattened form: [`../qwen4exp-serving-b02e16a8.patch`](../qwen4exp-serving-b02e16a8.patch)
(35 files, +4,528 / −94), sha256 `9743b441d58a1f0097ea5a25b98a2ce2e13de8b2e89d6bce83f97e8b2ae71acd`.
Base + flattened patch and base + the six patches both reproduce the identical
tree (`9f505c06…f1d`), verified in scratch worktrees.

## Optional patch (outside default acceptance)

`optional-spec/0001-feat-spec-optional-NGRAM-speculation-on-Qwen4-Exp-no.patch`
(sha256 `677549b8…11fe5`). Applies on top of the default series. NGRAM is
inert unless `--speculative-algorithm NGRAM` is passed and is not part of the
promoted profile. It also contains a genuine ReplaySSM commit bug fix that
affects MTP topk = 1.

## Reproduce

```bash
git clone https://github.com/sgl-project/sglang.git sglang-main
cd sglang-main
git checkout b02e16a895add01a0cfe24bb74922de92ab4d895
git am /path/to/series-main/000*.patch                 # or:
git apply /path/to/qwen4exp-serving-b02e16a8.patch
```

Revert: `git apply -R` the flattened patch, or `git reset --hard <base>` after
`git am`. The optional spec patch reverts the same way.

Environment: the port needs a venv built from this tree (main pins
`flashinfer_python==0.6.18`; `torch==2.13.0+cu130`). See
`docs/logs/raw/migration_3090_2026-09-16/step2-venv.md`.

## Assets (unchanged from the frozen control)

`assets/moe_configs/configs/triton_3_7_1/` (RTX 3090 nearest-E buckets),
`assets/expert_presence_code.pt`, `assets/expert_freq.pt`, and a writable
`SGLANG_MOE_ELASTIC_CTL` control file. Hashes in
`docs/logs/raw/migration_3090_2026-09-16/control-freeze.sha256`.

## Port notes and deviations

- Measurement-only hooks from the frozen tree are not ported: `kv_stats`,
  `kv_fakeq`, `SGLANG_NAN_TRACE`, the prefill route dump, and the dead
  `_process_weights_after_loading_disabled`.
- Expert placement/elastic loader plumbing lands with patch 2 because it lives
  inside `moe_wna16.py`'s loader; the `row_arena.py` / `expert_elastic.py`
  modules and lazy KV land in patch 3.
- The quantized QSA gather branches write compact rows and do not zero the
  padded tails of main's `zero_fill_cols=stride` path. That path is gated to
  SM100/SM120 by `_resolve_trtllm_sparse_decode()` and is therefore off the
  3090 profile; it remains an open item for other hardware.
- Validation status: static/API gate passed (compileall, `launch_server --help`,
  promoted-flag `ServerArgs` parse, no removed-VMM or old-field references).
  GPU gates are not run yet.
