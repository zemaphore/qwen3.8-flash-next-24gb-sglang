# Step 3 — main-based port status

Date: 2026-09-16. Execution-plan step 3 of
[`docs/SGLANG_MAIN_MIGRATION_PLAN.md`](../../../SGLANG_MAIN_MIGRATION_PLAN.md).
Companion to `control-freeze.md`, `port-compat-audit.txt`, `port-trial-apply.txt`,
`port-plan.md` and `step2-venv.md`.

## Result

The frozen control is ported onto pinned main `b02e16a895` as five reviewable
commits plus one optional commit, in `/root/quant/sglang-main`:

| Branch | Commits |
|---|---|
| `port/main-3090` | 5 (default series, base `b02e16a8`) |
| `port/main-3090-spec` | 1 optional NGRAM commit on top (exported separately) |

| # | Commit | Scope |
|---|---|---|
| 1 | `7fd7d0401a` `fix(qwen4): CPU-offload correctness, breakable graphs, mmap PLE table` | offloader, GemmaRMSNorm buffer, Conv1d view, QSA rope cache, BCG `LogitsProcessorOutput`, PLE mmap + pread + recent cache, language-model-only |
| 2 | `26aa2abdc3` `feat(moe_wna16): 2-bit experts, expert streaming, N-contiguous GEMV` | int2 kernels, `expert_stream.py`, `expert_gemv.py`, loader + placement plumbing |
| 3 | `b2ddd96474` `feat(mem_cache): elastic VMM expert row arenas, lazy KV, quantized QSA pools` | `row_arena.py`, `expert_elastic.py`, lazy VMM KV, `int8_g64`/`int4_g32`/`int8ring_int4` pools, QSA readers, KV choices |
| 4 | `d6c8cdbbdb` `fix(mem_cache): R1 evict-on-pressure and R2 prefill-alloc abort` | RB robustness (R3 was in commit 3) |
| 5 | `4ac0d3b35a` `feat(moe): prefix-cache PP model wiring, optional shared-expert overlap` | `prefetch_first_layer` call, eager-overlap switch |
| opt | `4aed18a006` `feat(spec): optional NGRAM speculation on Qwen4-Exp (not default)` | NGRAM chain, PLE commit fix (branch `port/main-3090-spec`) |

## Static / API gate (passed, CPU only)

- `python -m compileall python/sglang` — clean.
- Every ported module imports under the main venv with `CUDA_HOME` set.
- `python -m sglang.launch_server --help` — exit 0; the three custom KV choices
  are registered in `arg_groups/fields/model.py`.
- `ServerArgs(**<promoted flags>)` parses; `kv_cache_dtype=int8ring_int4`,
  `language_model_only=True`, `chunked_prefill_size=4608` resolve.
- No `sglang.srt.cuda_vmm_utils` references remain (moved to
  `sglang.srt.utils.cuda_vmm_utils`); no conflict markers; the ported files do
  not reference removed forward-batch fields.

## Reproduced artifacts (step 4 partial)

- `sglang/upstream/series-main/` — 5 `git am` patches + `MANIFEST.md`.
- `sglang/upstream/series-main/optional-spec/` — the optional NGRAM patch.
- `sglang/qwen4exp-serving-b02e16a8.patch` — flattened port (35 files,
  +4,526 / −94).
- Verified in scratch worktrees: base + flattened patch and base + the five
  patches both reproduce the port tree byte for byte (`f29efc9b…2370`).

## Open items carried forward

1. **Zero-fill gap (other hardware only).** Main's trtllm paged decode passes
   `zero_fill_cols=stride`; the quantized gather branches do not zero the padded
   tails. `_resolve_trtllm_sparse_decode()` gates that path to SM100/SM120, so
   it is off the 3090 profile but must be closed before any broader claim.
2. **Runtime behaviour not yet validated.** No server has been started from the
   ported tree; all GPU gates (checkpoint bring-up, numerical, prefix cache,
   pressure, performance/capacity) remain.
3. **Speculation is on its own branch** and must stay outside default
   acceptance.
4. Documentation (`sglang/README.md`, `PATCH_NOTES.md`, `UPSTREAM.md`,
   `patches/README.md`) still describes the historical base and has not been
   rewritten for the main anchors.
