# Step 3 — port plan against pinned main `b02e16a8`

Date: 2026-09-16. Execution-plan step 3 of
[`docs/SGLANG_MAIN_MIGRATION_PLAN.md`](../../../SGLANG_MAIN_MIGRATION_PLAN.md).
This is the bridge from the step-1 freeze and the step-2 checkout/venv to the
reviewable commits. It adds no source change.

Two mechanical audits were run against the pinned main checkout
(`/root/quant/sglang-main`, clean, detached at `b02e16a8`):

- `port-compat-audit.txt` — per target file, how far main moved from the branch
  base `73a255206f` (line-level), and whether the seven new modules collide.
- `port-trial-apply.txt` — a scratch `git apply --3way` of the frozen tracked
  diff (worktree removed afterwards; it is not the port). Result: **12 files
  conflict, 19 apply cleanly, all 7 new modules land without collision.**

"Applied cleanly" does **not** mean compatible: several clean files
(`scheduler.py`, `memory_pool.py`, `kv_cache_configurator.py`,
`pool_configurator.py`, `qwen3_5.py`, `qwen2_moe.py`, `kv_vmm_backing.py`) are
files that main rewrote heavily. Each still needs a line-by-line semantic
review against main's current code. The plan's patch-disposition table is the
specification; this document only orders the work.

## Ordered groups and targets

`C` = conflicted in the trial apply (rework by hand); `c` = applied cleanly
(review, do not trust); `n` = new module.

### G1 — model / offload / PLE / graphs

| Target | Trial | Main divergence | Reconcile |
|---|---|---|---|
| `srt/configs/qwen4_exp.py` | c | +16/-10; main has `qwen_sparse_attention` at line 88 | `packed_modules_mapping` propagation; drop any duplicate alias |
| `srt/utils/offloader.py` | c | +39/-37 | `functional_call` `tie_weights=False`, stale `param.data`, expert-only offload, `hook` |
| `srt/layers/layernorm.py` | c | +119/-42 | `gemma_weight` persistent buffer |
| `srt/models/qwen3_5.py` | c | +387/-52 | conv-module view (finding 8) |
| `srt/layers/radix_linear_attention.py` | **C** (1) | +118/-36 | conv-module view; drop `SGLANG_NAN_TRACE` |
| `srt/models/qwen4_exp.py` | **C** (1) | +102/-53; main has `Qwen4ExpMmapEmbedding`-era loader | PLE mmap + pread + bulk + recent-row cache; `bcg`; `ple_random`; forward-batch field names |
| `srt/model_executor/runner_backend/breakable_cuda_graph_backend.py` | c | +18/-8 | `LogitsProcessorOutput` structure helpers (`bcg2`) |
| `srt/layers/attention/linear/gdn_backend.py` | **C** (1) | +512/-48; main removed `ALL_DECODER_LAYER_TYPES` | decide whether the `qwen_sparse_attention` alias is still needed |
| `srt/server_args.py` (language-model-only part) | **C** (2) | +502/-9565; refactored into `srt/arg_groups/` | register `Qwen4ExpForConditionalGeneration` language-model-only in the new anchor, not `server_args.py` |
| `srt/models/qwen2_moe.py` | c | +195/-43 | shared-expert handling; keep `SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS` inert (0) |

### G2 — INT2 and expert streaming

| Target | Trial | Main divergence | Reconcile |
|---|---|---|---|
| `kernels/ops/moe/fused_moe_triton_kernels.py` | c | +131/-57 | 2-bit unpack + word-load int2 kernel |
| `srt/layers/moe/moe_runner/triton.py` | c | +0/-3 | `use_int2_w2a16` plumbing |
| `srt/layers/moe/moe_runner/triton_utils/fused_moe.py` | **C** (1) | +77/-50 | N-contiguous grid/stride |
| `srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py` | c | **identical** | `int2_w2a16` config dtype |
| `srt/layers/quantization/moe_wna16.py` | **C** (1) | +3/-4 | 2-bit loader, stream hook; main still restricts `moe_wna16` to 4/8-bit |
| `srt/layers/moe/expert_stream.py` | n | new | MoE-aware expert streamer |
| `test/manual/test_triton_moe_wna16.py` | c | **identical** | 2-bit test cases |

### G3 — placement / elastic / lazy KV

| Target | Trial | Main divergence | Reconcile |
|---|---|---|---|
| `srt/layers/moe/expert_gemv.py` | n | new | in-place int2 GEMV + address tables |
| `srt/layers/moe/row_arena.py` | n | new | CUDA VMM row arena |
| `srt/layers/moe/expert_elastic.py` | n | new | elastic expert cache |
| `srt/mem_cache/memory_pool.py` | c | +987/-438 | lazy VMM backing, pool APIs; **drop `kv_stats`/`kv_fakeq`** |
| `srt/mem_cache/kv_vmm_backing.py` | c | +12/-98; main has `BumpArenaStub` | adapt to `srt.utils.cuda_vmm_utils`; do not restore the removed allocator |
| `srt/mem_cache/kv_cache_configurator.py` | c | +605/-222 | lazy capacity + admission cap |
| `srt/mem_cache/allocator/paged.py` | **C** (1) | +130/-114 | `_lazy_hook`/`_lazy_idle_check` |
| `srt/mem_cache/allocator/token.py` | c | +20/-8 | alloc-time backing + idle reset |
| `srt/model_executor/pool_configurator.py` | c | +464/-156 | quantized-pool selection |
| `srt/mem_cache/allocation.py` | c | +114/-28 | (R1 lives here, G6) |

### G4 — quantized QSA and CLI

| Target | Trial | Main divergence | Reconcile |
|---|---|---|---|
| `srt/layers/attention/qsa/sparse_attn.py` | **C** (2) | +78/-13 | INT8/INT4/tiered read+write kernels; preserve main's 64-bit offsets, padded-row zero fill, query-dtype scratch, FP8 conversion |
| `srt/layers/attention/qwen_sparse_attn_backend.py` | **C** (4) | +152/-310 | bf16 scratch, prefix gather, pool dispatch |
| `srt/layers/attention/qsa/qsa_indexer.py` | **C** (2) | +31/-54 | rotary-cache hoist |
| `srt/layers/attention/qsa/metadata.py` | c | +18/-38 | keep `SGLANG_QSA_PREFILL_HOST_LENS` off |
| `srt/mem_cache/kv_cache_dtype.py` | c | +3/-0 | register `int8_g64`/`int4_g32`/`int8ring_int4` |
| `srt/mem_cache/int8_kv_pool.py` | n | new | INT8-G64 pool |
| `srt/mem_cache/int4_kv_pool.py` | n | new | INT4-G32 pool |
| `srt/mem_cache/tiered_kv_pool.py` | n | new | int8 ring over int4 |
| `srt/server_args.py` KV choices | **C** (2) | moved to `srt/arg_groups/fields/model.py` | register custom KV choices at the new anchor; reject incompatible backends |

### G5 — accepted PP

Within `expert_stream.py`, `expert_elastic.py`, `moe_wna16.py` and the 3090
launcher flags, in staging → batched DMA → cross-layer prefetch order, and the
new `enable_*`/`moe_*` layers: gather block 2048, host DMA gather, batched DMA,
presence placement, chunk 4608, cross-layer cold prefetch. Keep full-table
gather and first-layer prefetch off. (The 3090 PP layers are separate scripts in
`patches/`; the frozen diff already contains their merged result, so they must
be re-split, not applied as scripts.)

### G6 — RC / RB

| Target | Trial | Reconcile |
|---|---|---|
| `srt/mem_cache/allocation.py` | c | R1 evict-on-pressure (v3 re-sort → evict → sole-owner bypass) |
| `srt/managers/scheduler.py` | c | R2 prefill-alloc abort / HTTP 503; main diverged +1227/-382 — review carefully |
| `memory_pool.py` + `kv_vmm_backing.py` | c | R3 strict floor `SGLANG_KV_LAZY_MIN_FREE_MB=64` |
| launcher env | — | `SGLANG_KV_EVICT_ON_PRESSURE`, `SGLANG_KV_LAZY_STRICT_HEADROOM`, `SGLANG_PREFILL_ALLOC_ABORT` |

### Optional — speculation (export separately, outside default acceptance)

| Target | Trial | Reconcile |
|---|---|---|
| `srt/models/qwen4_exp.py` NGRAM hunks | **C** (part of G1) | keep separate |
| `srt/speculative/ngram_worker.py` | **C** (1) | linear draft chain; review against all current ReplaySSM early returns |
| `srt/speculative/spec_utils.py` | **C** (1) | PLE/recurrent commit after verify (genuine bug fix; affects MTP topk=1) |

## Rules carried from the plan

- Commit order G1 → G2 → G3 → G4 → G5 → G6; export the speculation diff
  separately.
- Drop the measurement-only hunks (`kv_stats`, `kv_fakeq`, `SGLANG_NAN_TRACE`,
  NGRAM debug) from the production series.
- Keep the disabled experiments off: full-table gather, first-layer prefetch,
  QSA host-length, eager shared overlap.
- Do not port the rejected paged-prefix kernel.
- Never trust a clean 3-way merge on a main-diverged file; every file gets a
  semantic review.
