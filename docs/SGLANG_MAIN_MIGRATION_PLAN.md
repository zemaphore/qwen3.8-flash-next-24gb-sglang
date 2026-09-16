# SGLang main migration — RTX 3090

Date: 2026-09-16. Status: planned; implementation and serving migration have
not started. Repository baseline: `ab61990` (R1–R3 launcher promotion).
This document supersedes the future-facing migration assumptions in
[`sglang/UPSTREAM.md`](../sglang/UPSTREAM.md). Historical patches and logs remain
reproduction artifacts.

## Objective and scope

Reproduce the promoted PP/TG/RC serving behavior on a pinned upstream-main
commit, with an ordered patch series that applies cleanly from a fresh checkout.
The only hardware target is **one RTX 3090, SM86, 24 GB VRAM**. Other GPU
architectures have no migration claims, validation gates or compatibility
requirements. Leave unrelated upstream hardware code in place; it is outside
the acceptance scope.

Preserve the checkpoint, quantization, single-request operation and promoted
linear-conversation profile. This is a base migration, not a new optimization
campaign. RAM prefix tiers, new quantization, speculation enablement, concurrent
clients and forking workflows are outside scope.

## Verified upstream position

- Original model PR [#36497](https://github.com/sgl-project/sglang/pull/36497)
  closed unmerged on September 8.
- Replacement PR [#37500](https://github.com/sgl-project/sglang/pull/37500)
  landed model support as `52fecfdf0908dca24f4c6799ff5967125cc4110e` that day.
- Main inspected for this plan:
  [`b02e16a895add01a0cfe24bb74922de92ab4d895`](https://github.com/sgl-project/sglang/commit/b02e16a895add01a0cfe24bb74922de92ab4d895).
  Local `origin/main` matched the remotely inspected SHA. Use this exact commit
  for the first port; advancing it requires another delta review.
- Serving base remains `73a255206f916366c8d26d4022f82ddfb0ab558d` plus local
  changes. The September 3 flat patch and five-part series omit later PP/RC
  work and cannot alone reproduce the promoted baseline.

The initial audit compared original flat-patch hunk contexts with pristine
main blobs in memory. It identified conflicts but did not apply patches,
perform a rebase, or establish runtime compatibility.

## Promoted control and evidence

Use the final launcher snapshot
[`serve-3090-flags.sh`](logs/raw/promotion_3090_2026-09-16/serve-3090-flags.sh),
installed as `/root/quant/serve-3090.sh`. Its SHA-256 is
`d0866e7ca18ac6eac29290b429c147ac6fc7863d3d6571a8c4a4a80539f57447`;
see the [launcher manifest](logs/raw/promotion_3090_2026-09-16/launchers-flags.sha256).

| Setting | Migration control |
|---|---|
| Context and token pool | 235,520 tokens (230K) |
| Model name | `qwen38-flash-230K` |
| Request/state slots | One running request, four Mamba slots, radix cache enabled |
| KV | `int8ring_int4`, ring 8,192, lazy backing |
| Experts | S184 presence placement, elastic residency, INT2 GEMV/streaming |
| Prefill | Chunk 4,608; PP14 prefetch floor 2,048; accepted PP15 configs |
| PLE | CPU mmap/pread, bulk reads, 128 MiB recent-row cache |
| Graphs | Breakable decode graphs |
| Robustness | Pressure eviction, strict headroom and prefill abort enabled; floor 64 MiB; R1 v3 sole-owner bypass |
| Host limit | `MemoryMax=56G` |

Sources: [PP closure](logs/pp_wrapup_3090_2026-09-16.md),
[TG0](logs/tg0_baseline_3090_2026-09-16.md),
[RC4 linear profile](logs/rc4_linear_3090_2026-09-16.md),
[RB final validation and promotion](logs/rb_robustness_3090_2026-09-16.md).
The TG plan retains unexecuted candidate stages; those are not prerequisites
or implied features of this migration.

RB attempt 4 passed the flags-on linear checks, including 64/64 hits through
231,310 tokens and the next-session checks. R2 was exercised in earlier
flags-on testing: allocation refusal returned HTTP 503 and the server survived.
The final R1 v3/floor-64 combination has not had the older churn reproducer
repeated across boots. R4 backing release and R5 service installation remain
unfinished; they are not promoted features to invent during the port.

Rollback snapshots are `serve-3090-radix-noflags.sh` and
`serve-3090-nocache-256K.sh` in the same promotion directory. The old no-cache
256K profile is a fallback, not this migration's performance or capacity target.

## Patch disposition

Paths below are relative to SGLang's `python/sglang/` unless stated otherwise.
These are implementation requirements, not claims that the port is complete.

| Area | Required amendment |
|---|---|
| Model config and loading | Drop the already-upstream `qwen_sparse_attention` alias. Reconcile `packed_modules_mapping`, language-model-only registration and checkpoint weight loading with current `srt/models/qwen4_exp.py`. Preserve current forward-batch field names. |
| Offload correctness | Retain needed device-placement, expert-only offload and `functional_call` fixes. Main still creates `conv_weights` during initialization; port the dynamic-view fix and test storage replacement. |
| PLE and graph break | Retain CPU `pread`, bulk reads, recent-row cache, bounded host-memory behavior and eager graph break. Main's file-backed host-table implementation requires GPU access to pageable host memory and does not replace the 3090 path. Integrate around current loader logic. |
| Graph outputs | Port `LogitsProcessorOutput` support to the breakable graph backend and verify capture/replay. |
| INT2 MoE | Port 2-bit loading, unpacking, word-contiguous kernels, pointer-table GEMV and expert streaming. Main still restricts relevant `moe_wna16` paths to 4/8-bit weights. |
| Placement and PP | Carry accepted presence placement, expert-count config buckets, gather block, host DMA, batched DMA, cross-layer prefetch and PP15 assets. Preserve stream/event dependencies and stable pinned sources. |
| VMM/lazy KV | Change `srt.cuda_vmm_utils` imports to `srt.utils.cuda_vmm_utils`, including `row_arena.py`. Adapt to main's `BumpArenaStub`; do not restore its removed allocator implementation. Audit commit/decommit ownership, granularity and graph-stable pointers. |
| Quantized QSA | Port INT8/INT4/tiered pools and every active read path. Preserve main's 64-bit offsets, padded-row zero fill, query-dtype scratch and FP8 conversion. Drop duplicate FP8 hunks only after checking write/read semantics. |
| CLI and backend guards | Register custom KV choices in `srt/arg_groups/fields/model.py` and reconcile argument hooks, dtype resolution and pool selection. Old `server_args.py` anchors no longer apply. Reject incompatible backends rather than interpreting packed bytes as ordinary KV. |
| RC state and allocation | Preserve radix reuse of KV, QSA, recurrent and PLE state. Port R1 re-sort/retry, eviction and one-retry bypass; R2 cleanup/503 handling; R3 floor. Check bypass reset on success and exceptions and preserve existing non-strict allocation protections. |
| Optional speculation | Review PLE commit handling against all current ReplaySSM early returns. Keep NGRAM optional and outside default acceptance. Do not assume the old two-hunk fix covers new branches. |

Relevant upstream implementations:
[QSA gathers](https://github.com/sgl-project/sglang/blob/b02e16a895add01a0cfe24bb74922de92ab4d895/python/sglang/srt/layers/attention/qsa/sparse_attn.py),
[PLE host storage](https://github.com/sgl-project/sglang/blob/b02e16a895add01a0cfe24bb74922de92ab4d895/python/sglang/srt/models/qwen4_exp_ple_table.py),
[VMM backing](https://github.com/sgl-project/sglang/blob/b02e16a895add01a0cfe24bb74922de92ab4d895/python/sglang/srt/mem_cache/kv_vmm_backing.py).

Keep full-table gather, first-layer prefetch, QSA host-length experiments and
eager shared overlap disabled as in the accepted launcher. Keep diagnostic
hooks separate from production changes. Do not port the rejected paged-prefix
kernel into the default series.

## Execution plan

1. **Freeze the complete control.** Record repo revision, serving HEAD and diff,
   all seven added serving modules, launcher/effective environment, dependency
   versions, checkpoint metadata and placement/kernel asset hashes. Reconcile
   with the [final RB diff](logs/raw/rb_3090_2026-09-16/serving-tree-with-R123-v4.diff).
   A tracked-file diff alone is insufficient. Classify each delta as accepted,
   optional, diagnostic or rejected and account for every live-source delta.
2. **Create an independent checkout and venv.** Pin the inspected main SHA.
   Keep `/root/sglang`, the serving venv and rollback launchers intact. Main
   pins FlashInfer 0.6.18 versus the documented control's 0.6.17; record the
   full dependency delta and validate the supported SM86 toolchain separately.
   Use separate control files, logs and caches where sharing could interfere.
3. **Port in reviewable commits.** Order: model/offload/PLE/graphs; INT2 and
   streaming; placement/elastic/lazy KV; quantized QSA and CLI; accepted PP;
   RC/RB. Within PP preserve staging → batched DMA → cross-layer prefetch.
   Export optional speculation/research changes separately.
4. **Regenerate artifacts.** Produce `sglang/upstream/series-main/`, a new
   flattened serving patch named for the target SHA, and a manifest of base,
   ordered patch hashes and assets. Retain the historical exports unchanged.
   Amend or version helper scripts for the new anchors; retain symmetric
   `apply`/`revert`/`--check` and explicit prerequisites. Never run old scripts
   blindly against main.
5. **Run the validation gates below.** CPU checks can precede GPU scheduling.
   GPU tests and candidate server runs need exclusive access to the 3090;
   perform them in a deliberate maintenance window with rollback ready.
6. **Promote only after validation.** Install a separately named candidate
   launcher first, verify its effective profile, then switch the default.
   Roll back source/environment/launcher together if a gate fails. Retain the
   current flags-on launcher as the immediate migration rollback.

## Acceptance gates

| Gate | Required evidence |
|---|---|
| Clean reproduction | Fresh pinned checkout accepts the ordered series; flattened patch gives the identical tree; added modules/assets are complete; helper checks and reverse/reapply work on scratch trees. |
| Static/API compatibility | Imports, syntax, registered tests and launcher argument parsing succeed; no references to removed VMM utilities or old forward-batch fields remain. |
| Kernel/state correctness | INT2 packing/GEMV; offload storage swaps; PLE reads/cache eviction; graph outputs; VMM shrink/regrow/replay; KV scale/index/ring ownership; padded QSA gathers, short prompts and chunk boundaries. |
| Checkpoint bring-up | Same checkpoint loads with expected names/dtypes; CPU PLE and streamed experts activate; effective radix/page/state configuration is recorded rather than inferred from CLI spelling. |
| Numerical behavior | Matched repeated short and multi-chunk teacher-forced comparisons with complete token IDs, finite values and per-token results. Test sustained decode separately. Carry unresolved PP14 long-prompt drift forward; do not loosen thresholds after observing an outlier. |
| Prefix cache | Cold/repeat/append, eviction/rehit, next session after near-limit retention, and consistency of KV/QSA/recurrent/PLE state. Preserve single-consumer linear-session scope. |
| Pressure handling | Force failure with and without evictable prefixes; verify bounded retries, bypass lifecycle, HTTP 503 cleanup and a successful subsequent request. Repeat churn across boots with final R1 v3/floor-64 behavior. |
| Performance/capacity | Matched multi-turn trace wall time is primary; report cold PP, sustained TG, warm TTFT, peak host memory and near-limit capacity separately. Repeat across boots and compare to the frozen flags-on control. Preserve the promoted 235,520-token configuration and test actual token counts. |

Predeclare the repetition counts and non-regression tolerances from repeated
control measurements before candidate evaluation. Historical isolated PP/TG
thresholds do not override the RC decision to judge complete agentic traces.
A migration without an intentional workload tradeoff should reproduce that
trace performance within the predeclared tolerance. Report regressions openly;
do not silently reduce context or change precision to pass.

## Documentation and completion

On implementation, update `sglang/README.md`, `PATCH_NOTES.md`, `UPSTREAM.md`,
`patches/README.md` and launcher reproduction instructions with the final
manifest, dependencies, feature disposition, evidence and rollback procedure.
Review the existing contribution PRs against the new main-based split; posting
or retargeting them is separate from this local migration.

Completion requires a clean main-based reproduction of the promoted 3090
profile, passing correctness, pressure, trace-performance and capacity gates.
Clean patch application alone is not completion. This planning change does
not modify patches, install dependencies or switch the running server.
