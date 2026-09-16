# Control freeze — RTX 3090 promoted profile

Date: 2026-09-16. Execution-plan step 1 of
[`docs/SGLANG_MAIN_MIGRATION_PLAN.md`](../../../SGLANG_MAIN_MIGRATION_PLAN.md).
This records the complete live control that the main-based port must reproduce.
It changes no source, starts no server and stops no server. Weight files are not
hashed here (38.76 GB); only checkpoint metadata is recorded.

Companion files in this directory:

| File | Contents |
|---|---|
| `control-freeze.sha256` | Every hash named below |
| `serving-tracked.diff` | `git diff` of the live serving tree (tracked files only) |
| `effective-launch.txt` | Live PID 486994 command line and `/proc` environment |
| `effective-config.txt` | Resolved runtime configuration from the promoted boot log, and checkpoint metadata |
| `venv-pip-freeze.txt` | Full `pip freeze` of the control venv (205 packages) |

## A. Repository and serving revision

| Item | Value |
|---|---|
| Docs repo | `/root/qwen3.8-flash-next-24gb-sglang`, branch `pp-perf-3090`, HEAD `ab619907271d68f85e777a90c4f2e72466ecd474` ("Promote R1–R3 flags in serve-3090.sh …") |
| Docs repo working tree | 4 modified (`docs/README.md`, `sglang/README.md`, `sglang/UPSTREAM.md`, the promoted boot log) plus 2 untracked (the migration plan and this evidence directory); no code change |
| Serving checkout | `/root/sglang` |
| Serving HEAD | `73a255206f916366c8d26d4022f82ddfb0ab558d` ("Introduce Qwen 3.8 Flash Next", the first commit of `qwen4-main-squashed`) |
| Serving changes | 31 modified tracked files + 7 untracked modules; nothing committed |
| Serving venv | `/root/quant/venv-sglang` (Python 3.12.3) |
| Live server | PID 486994, scope `sglang-1789588603.scope`, `MemoryMax=56G`, port 30001, GPU 0 RTX 3090 (SM86), 21,184 MiB in use |

The serving base remains `73a255206f` plus the local changes; it is **not** on
upstream `main`. Upstream model support landed as
`52fecfdf0908dca24f4c6799ff5967125cc4110e` (PR #37500); the pinned main for the
first port is `b02e16a895add01a0cfe24bb74922de92ab4d895`, present locally in
`/root/sglang`'s object store and matching `origin/main`.

## B. Reconciliation with the final RB diff

The live `git diff` of the serving checkout is **byte-identical** to the final
RB reference [`serving-tree-with-R123-v4.diff`](../../rb_3090_2026-09-16/serving-tree-with-R123-v4.diff):

```
47ef80c2b8513319ea33033c6022890d99707c6d428158626d01fda10efd325a  serving-tracked.diff
47ef80c2b8513319ea33033c6022890d99707c6d428158626d01fda10efd325a  serving-tree-with-R123-v4.diff
```

That diff covers the 31 tracked files only; it does **not** contain the seven
untracked modules, which are recorded separately in section C. This satisfies
the plan's warning that a tracked-file diff alone is insufficient: the frozen
control is the tracked diff plus the seven module hashes plus the launcher and
assets.

Provenance checks that passed:

- Installed `/root/quant/serve-3090.sh` sha256 `d0866e7c…f57447` equals the repo
  promotion snapshot `serve-3090-flags.sh` (same bytes as the plan's manifest).
- Live PID 486994 command line and environment match that launcher exactly
  (`effective-launch.txt`).
- Resolved page size, KV tier ring, radix components and elastic placement in
  the promoted boot log match the launcher's intent (`effective-config.txt`).

## C. Seven added serving modules

Untracked in `/root/sglang`, therefore absent from the tracked diff:

| Module (`python/sglang/srt/…`) | Lines (PATCH_NOTES §3) | Introduced by | sha256 |
|---|---:|---|---|
| `layers/moe/expert_stream.py` | 182 | base 2-bit patch | `45fad94b…130eb5` |
| `layers/moe/expert_gemv.py` | 109 | `ncontig_gemv.py` | `f0850a1a…5cc694` |
| `layers/moe/row_arena.py` | 215 | `gemv/row_arena.py` | `8cdc7403…1ca52b` |
| `layers/moe/expert_elastic.py` | 376 | `gemv/expert_elastic.py` | `29c00e39…63a798` |
| `mem_cache/int8_kv_pool.py` | 159 | `kv_int8.py` | `83649aa0…1e93a4` |
| `mem_cache/int4_kv_pool.py` | 173 | `kv_int4.py` | `adec73d9…b1b09c` |
| `mem_cache/tiered_kv_pool.py` | 169 | `kv_tiers.py` | `cf0c96ed…d7a6d9` |

## D. Launcher and effective environment

Launcher: [`serve-3090-flags.sh`](../../promotion_3090_2026-09-16/serve-3090-flags.sh),
installed as `/root/quant/serve-3090.sh`, sha256 `d0866e7c…f57447`, run under
`systemd-run --user --scope … -p MemoryMax=56G`.

Effective flags (live):

```
--host 0.0.0.0 --port 30001 --tp-size 1 --cpu-offload-gb 19 --no-ple-offload-embedding
--mem-fraction-static 0.95 --language-model-only --page-size 1 --disable-overlap-schedule
--sleep-on-idle --weight-loader-drop-cache-after-load --chunked-prefill-size 4608
--max-prefill-tokens 32768 --cuda-graph-backend-decode breakable
--served-model-name qwen38-flash-230K
--model-path /mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang
--max-total-tokens 235520 --context-length 235520 --kv-cache-dtype int8ring_int4
--attention-backend triton --max-mamba-cache-size 4
--reasoning-parser qwen3 --tool-call-parser qwen3_coder --max-running-requests 1
```

Effective environment: see `effective-launch.txt` (32 `SGLANG_*` variables plus
`CUDA_HOME`, `PATH`). Resolved (not CLI-inferred) configuration from the boot log:

| Resolved setting | Value |
|---|---|
| Page size | **64**, not the requested `1` (compressed QSA forces 64) |
| KV capacity | profiled 423,488 → requested 235,520 at safety 0.77 |
| KV backing | 235,520 VA tokens, 4,096-token floor, margin 2,048, 6.8 KB/token |
| KV store | `torch.uint8`, 235,520 tokens, K 0.76 GB / V 0.76 GB |
| KV tiers | ring `R=8192` (int8_g64 over int4_g32), 99 MB + 32 KB owner |
| Mamba cache | 4 slots, conv 0.01 GB + ssm 0.53 GB |
| Radix | `UnifiedRadixCache`, components FULL + MAMBA, `hybrid_ssm=True`, streaming off |
| Elastic placement | 48 layers at S=184, VRAM free 5.31 GB |
| Decode graph | breakable, `bs=[1]`, capture 1.26 s / 0.08 GB |
| Prefill graph | disabled (Breakable graph incompatible with multimodal) |
| Load | 170.48 s, 18.41 GB device memory |

## E. Dependency versions (control venv)

| Package | Control |
|---|---|
| torch | 2.13.0+cu130 |
| triton | 3.7.1 |
| transformers | 5.12.1 |
| flashinfer-python | **0.6.17** |
| flash_attn | 2.8.3.post1 |
| flash-attn-4 | 4.0.0b19 |
| tilelang | 0.1.11 |
| sglang | editable, `73a255206f` |
| nvidia-cuda-nvcc | 13.3.73 |
| nvidia-cuda-runtime | 13.3.29 |
| numpy | 2.3.5 |

The plan notes main pins FlashInfer **0.6.18** against this control's 0.6.17; the
full 205-package list is `venv-pip-freeze.txt` and the delta must be recorded
when the independent venv is built (step 2).

## F. Checkpoint metadata

| Property | Value |
|---|---|
| Path | `/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang` |
| Architecture | `Qwen4ExpForConditionalGeneration` (`model_type=qwen4_exp`) |
| dtype / tied embeddings | bfloat16 / `tie_word_embeddings=False` |
| Quantization | AutoRound 0.14.2, bits 2, group_size 128, sym, `auto_round:auto_gptq`, 2,342 `extra_config` entries |
| Tensors | 222,856 weight-map entries, total 38,755,352,600 B |
| Shards | `model-0000{1..8}-of-00008` + `model-int8dense-000` (the INT8 dense re-pack) |
| PLE table | `ple/ple.f8_e4m3.bin`, 320,001,536 rows × 160 F8_E4M3, 128 shards, 51,200,245,760 B |
| Hashed metadata | `config.json`, `model.safetensors.index.json`, `generation_config.json`, `ple/ple.json`, `tokenizer.json`, `chat_template.jinja` (`control-freeze.sha256`) |

## G. Placement and kernel assets

All under `assets/`, hashed in `control-freeze.sha256`:

- `expert_presence_code.pt` sha256 `73d8f0df…ddfd8` (the launcher's placement
  histogram, `SGLANG_MOE_PLACEMENT`).
- `expert_freq.pt` sha256 `96697b3c…052cf` (the pooled routing-mass fallback).
- `moe_configs/configs/triton_3_7_1/`: 16 JSON files. The four
  `NVIDIA_RTX_PRO_4000_Blackwell` files (`E={10,512}`) are historical; the
  RTX 3090 buckets `E={128,192,288,352,384,432}` × `{up,down}` are the ones the
  running server selects by nearest-E. `SGLANG_MOE_CONFIG_DIR` points here.
- Writable elastic control file `/root/quant/elastic.ctl` (contents `S 184`);
  the server writes `/root/quant/elastic.ctl.status`.

## H. Delta classification

Classes: **accepted** (required by the promoted profile), **accepted-off**
(in the tree, required for the port but inert under the promoted launcher),
**diagnostic** (measurement/debug only; the plan says keep separate from
production), **rejected** (not in the tree, must not be ported).

### H.1 Tracked files

| # | Path (`python/sglang/`) | Feature(s) | Class |
|---|---|---|---|
| 1 | `kernels/ops/moe/fused_moe_triton_kernels.py` | 2-bit unpack; word-load int2 kernel | accepted |
| 2 | `srt/configs/qwen4_exp.py` | `packed_modules_mapping` propagation | accepted (drop the upstream `qwen_sparse_attention` alias) |
| 3 | `srt/layers/attention/linear/gdn_backend.py` | `qwen_sparse_attention` in decoder layer types | accepted |
| 4 | `srt/layers/attention/qsa/metadata.py` | `SGLANG_QSA_PREFILL_HOST_LENS` experiment | accepted-off |
| 5 | `srt/layers/attention/qsa/qsa_indexer.py` | rotary-cache/`positions.max()` hoist | accepted |
| 6 | `srt/layers/attention/qsa/sparse_attn.py` | fp8/int8/int4/tier quantize + dequant-on-gather readers | accepted |
| 7 | `srt/layers/attention/qwen_sparse_attn_backend.py` | bf16 scratch, prefix gather, pool dispatch | accepted |
| 8 | `srt/layers/layernorm.py` | `gemma_weight` persistent buffer | accepted |
| 9 | `srt/layers/moe/moe_runner/triton.py` | `use_int2_w2a16` plumbing | accepted |
| 10 | `srt/layers/moe/moe_runner/triton_utils/fused_moe.py` | int2 / N-contiguous grid handling | accepted |
| 11 | `srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py` | `int2_w2a16` config dtype | accepted |
| 12 | `srt/layers/quantization/moe_wna16.py` | 2-bit loader, stream hook, ncontig, placement, elastic | accepted |
| 13 | `srt/layers/radix_linear_attention.py` | conv-module view fix (+ `SGLANG_NAN_TRACE` print) | accepted (+ diagnostic sub-hunk) |
| 14 | `srt/managers/scheduler.py` | R2 `SGLANG_PREFILL_ALLOC_ABORT` | accepted |
| 15 | `srt/mem_cache/allocation.py` | R1 `SGLANG_KV_EVICT_ON_PRESSURE` | accepted |
| 16 | `srt/mem_cache/allocator/paged.py` | lazy-VMM alloc hooks | accepted |
| 17 | `srt/mem_cache/allocator/token.py` | lazy-VMM alloc hooks | accepted |
| 18 | `srt/mem_cache/kv_cache_configurator.py` | lazy capacity + admission cap | accepted |
| 19 | `srt/mem_cache/kv_cache_dtype.py` | `int8_g64`/`int4_g32`/`int8ring_int4` | accepted |
| 20 | `srt/mem_cache/kv_vmm_backing.py` | `ensure_prefix`/`uncommit_beyond`/`release_beyond` | accepted |
| 21 | `srt/mem_cache/memory_pool.py` | lazy VMM backing, pool APIs, **`kv_stats` + `kv_fakeq`** | accepted + **diagnostic sub-hunks** |
| 22 | `srt/model_executor/pool_configurator.py` | quantized KV pool selection | accepted |
| 23 | `srt/model_executor/runner_backend/breakable_cuda_graph_backend.py` | `LogitsProcessorOutput` structure helpers | accepted |
| 24 | `srt/models/qwen2_moe.py` | shared-expert handling; `SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS` | accepted + accepted-off hunk |
| 25 | `srt/models/qwen3_5.py` | conv-module view fix (silent NaN under offload) | accepted |
| 26 | `srt/models/qwen4_exp.py` | PLE mmap + pread + bulk + recent-row cache; `bcg`; `ple_random`; NGRAM guard/self-check | accepted + **optional (NGRAM)** |
| 27 | `srt/server_args.py` | language-model-only arch; KV dtype choices | accepted |
| 28 | `srt/speculative/ngram_worker.py` | linear draft chain for NGRAM | optional (NGRAM, inert) |
| 29 | `srt/speculative/spec_utils.py` | PLE/recurrent commit after ReplaySSM verify | optional (genuine bug-fix; affects MTP topk=1) |
| 30 | `srt/utils/offloader.py` | `functional_call` `tie_weights=False`, stale `param.data`, expert-only offload, hook | accepted |
| 31 | `test/manual/test_triton_moe_wna16.py` | 2-bit test cases | accepted (tests; not serving) |

### H.2 Seven modules

All seven are **accepted**: `expert_stream.py`, `expert_gemv.py`,
`row_arena.py`, `expert_elastic.py`, `int8_kv_pool.py`, `int4_kv_pool.py`,
`tiered_kv_pool.py`.

### H.3 Environment switches present in the frozen tree

Accepted and enabled by the launcher: `SGLANG_QWEN4_PLE_MMAP`,
`_PLE_BULK_PREAD`, `_PLE_BULK_PREAD_MIN_UNIQUE`, `_PLE_WORKERS`,
`_PLE_RECENT_CACHE_MB`, `SGLANG_VLM_CACHE_SIZE_MB`,
`SGLANG_MOE_EXPERT_STREAM`, `SGLANG_MOE_GATHER_BLOCK`, `_GATHER_DMA`,
`_GATHER_DMA_BATCH`, `SGLANG_MOE_COLD_PREFETCH`, `_MIN_TOKENS`,
`SGLANG_MOE_CONFIG_NEAREST_E`, `SGLANG_MOE_CONFIG_DIR`,
`SGLANG_MOE_PLACEMENT`, `_S`, `SGLANG_MOE_ELASTIC`, `_PIN_MB`, `_CTL`, `_FILL_MB`,
`_RESERVE_ROWS`, `SGLANG_KV_LAZY`, `_TOKENS`, `_SAFETY`, `SGLANG_KV_TIERS_W`,
`SGLANG_KV_EVICT_ON_PRESSURE`, `SGLANG_KV_LAZY_STRICT_HEADROOM`,
`SGLANG_KV_LAZY_MIN_FREE_MB`, `SGLANG_PREFILL_ALLOC_ABORT`.

Accepted defaults, not set by the launcher: `SGLANG_MOE_NCONTIG`,
`SGLANG_MOE_GEMV`, `SGLANG_KV_LAZY_FLOOR`, `_MARGIN`, `_HEADROOM_MB`.

Present but **disabled/experimental** (keep off in the port's default series):
`SGLANG_MOE_PREFETCH_FULL_TABLE` and `SGLANG_MOE_PREFETCH_FIRST_LAYER`
(`expert_stream.py`, both default `0`), `SGLANG_QSA_PREFILL_HOST_LENS`
(`metadata.py`, default `0`), `SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS`
(`qwen2_moe.py`, launcher forces `0`).

**Diagnostic/measurement-only**, to be dropped from the ported production
series: `SGLANG_KV_STATS` / `SGLANG_KV_STATS_FILE`, `SGLANG_KV_FAKEQ`
(`memory_pool.py` and the three pool modules), `SGLANG_NAN_TRACE`
(`radix_linear_attention.py`), `SGLANG_NGRAM_CHECK` /
`SGLANG_NGRAM_FORCE_REJECT` (optional NGRAM debug). Their exact hunk locations
are listed in `sglang/PATCH_NOTES.md` §5.

### H.4 Rejected / not in the tree

- `kv_paged_prefix.py` — rejected on timing (6.1 ms vs 1.4 ms per head-layer at
  prefix 60k); `--check` reads clean. Must not enter the default series.
- `host_fixes.py` `dump`/`moedump` — measurement routing dumps, not applied.

## I. Evidence gaps carried into the port

The plan's own open items, restated as control facts:

- The final R1 v3 / floor-64 combination has not been repeated across boots for
  the older churn reproducer (RB attempt 2 used R1 v1 / floor 256).
- R4 (release backing on trim) and R5 (systemd unit installation) are unfinished
  and are not promoted features.
- The `SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS` hunk is present but inert;
  eager shared overlap was not accepted.
- `SGLANG_MOE_CONFIG_NEAREST_E=1` selects compact expert-count buckets during
  prefill; the historical `E={10,512}` Blackwell configs remain in `assets/`
  only as provenance.

## J. Re-verify the freeze

```bash
# tracked diff identical to the RB v4 reference
cd /root/sglang && git diff | sha256sum          # 47ef80c2…f325a
# seven module hashes
sha256sum python/sglang/srt/layers/moe/{expert_stream,expert_gemv,row_arena,expert_elastic}.py \
          python/sglang/srt/mem_cache/{int8_kv_pool,int4_kv_pool,tiered_kv_pool}.py
# launcher identity
sha256sum /root/quant/serve-3090.sh              # d0866e7c…f57447
```

Any divergence means the control moved and the freeze must be redone before the
port continues.
