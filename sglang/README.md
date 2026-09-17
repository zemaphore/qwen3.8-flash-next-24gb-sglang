# The SGLang changes

**Current status (2026-09-17):** the **main-based implementation** is the current
RTX 3090 serving stack. The promoted 3090 profile has been ported onto SGLang
main `b02e16a895add01a0cfe24bb74922de92ab4d895` (model support landed as
`52fecfdf0908dca24f4c6799ff5967125cc4110e`, PR #37500) and exported as the
six-commit series [`upstream/series-main/`](upstream/series-main/) plus the
flattened [`qwen4exp-serving-b02e16a8.patch`](qwen4exp-serving-b02e16a8.patch);
see [Main-based 3090 series (current)](#main-based-3090-series-current) below.
The historical `73a255206f` artifacts in this directory remain **unchanged** as
provenance and as the reproduction of the published sm_120 measurements. Plan:
[RTX 3090 main migration plan](../docs/SGLANG_MAIN_MIGRATION_PLAN.md).

Two forms of the same change to SGLang: the flat serving patch, which is the verbatim diff of
the tree that served the published numbers, and the five-commit series under `upstream/`, which
is the reviewable form for `sgl-project/sglang`. The weights they serve are on the Hub:
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang

| File | What it is |
|---|---|
| `upstream/series-main/` | **Current 3090 implementation.** Six `git am` patches porting the frozen control onto main `b02e16a895`, with [`MANIFEST.md`](upstream/series-main/MANIFEST.md) (ordered patch hashes, flattened-patch identity, dependency and port notes). Base + either form reproduces the identical tree `9f505c06…f1d`. |
| `qwen4exp-serving-b02e16a8.patch` | Flattened main-based port: 35 files, +4,528 / -94, SHA-256 `9743b441d58a1f0097ea5a25b98a2ce2e13de8b2e89d6bce83f97e8b2ae71acd`. `git apply` on the base, or use the series above. |
| `upstream/series-main/optional-spec/` | Optional NGRAM speculation patch on top of the main series (SHA-256 `677549b8…11fe5`). Outside default acceptance; not part of the promoted profile; **not tested** on the 3090. |
| `qwen4exp-serving-73a255206f.patch` | **Historical provenance.** The difference between SGLang commit `73a255206f916366c8d26d4022f82ddfb0ab558d` ("Introduce Qwen 3.8 Flash Next", the first commit of the branch `qwen4-main-squashed` of PR #36497) and the served tree. 34 files, +4,155 / -89, 5,439 lines, 251,847 bytes, SHA-256 `10a3ad54f9688099848b3a6985145050ed35327116bf7c0ffa8a951b338b69c9` (the 2026-09-03 form cited in older notes was `92f669b2…744bb5`, 251,796 bytes; the PLE workers line changed during the PP campaign). Plain `git diff` output: `git apply` or `patch -p1`. Base + patch reproduces all 34 patched files of the served tree byte for byte (`PATCH_NOTES.md` section 2). |
| `PATCH_NOTES.md` | Per-file map of the flat patch grouped by feature, the measurement-only and debug hunks that the series drops (`kv_stats` / `kv_fakeq`, `SGLANG_NAN_TRACE`, NGRAM debug switches), the server flags and environment, the validation evidence per feature, known issues. |
| `UPSTREAM.md` | Status of the upstream contribution: review target and why it is not `main`, the five parts and what each contains, what remains, related PRs, how the series was verified. |
| `upstream/series-q4head/0001..0005` | The series rebased onto the head of `qwen4-main-squashed` (`78c5024e9d`): 39 files, +8,042 / -94. `git am` format. This is what the PRs contain. |
| `upstream/series-base/0001..0005` | The same five commits on the served base `73a255206f`: 40 files, +8,082 / -175. Reference for the rebase. |
| `upstream/RFC.md` | The issue text that introduces the series to the maintainers (feature-request template). |
| `upstream/PR-1.md` .. `PR-5.md` | The PR descriptions (PR template: Motivation, Modifications, Accuracy Tests, Speed Tests and Profiling, Checklist, suggested reviewers, reproduction commands). |

Apply the flat patch:

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 73a255206f916366c8d26d4022f82ddfb0ab558d
git apply --check /path/to/qwen4exp-serving-73a255206f.patch && git apply /path/to/qwen4exp-serving-73a255206f.patch
```

Apply the series instead (review form; drops the measurement hooks, adds the registered tests):

```
git fetch origin qwen4-main-squashed && git checkout 78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
git am /path/to/upstream/series-q4head/000*.patch
```

## Main-based 3090 series (current)

The current RTX 3090 implementation ports the frozen control onto upstream main.
See [`upstream/series-main/MANIFEST.md`](upstream/series-main/MANIFEST.md) for the
authoritative manifest; the summary here is for orientation.

| Item | Value |
|---|---|
| Target base | SGLang main `b02e16a895add01a0cfe24bb74922de92ab4d895`; model support landed as `52fecfdf0908dca24f4c6799ff5967125cc4110e` (PR #37500) |
| Series | six patches in `upstream/series-main/0001..0006` (`git am`, in order) |
| Flattened | `qwen4exp-serving-b02e16a8.patch`, 35 files, +4,528 / −94, SHA-256 `9743b441…71acd` |
| Tree identity | base + flattened patch and base + the six patches both reproduce the identical tree `9f505c06…f1d` |
| Optional | NGRAM commit under `upstream/series-main/optional-spec/`, outside default acceptance |
| Tree / venv | `/root/quant/sglang-main` (branch `port/main-3090`, HEAD `b114264c01`) and `/root/quant/venv-sglang-main` |

Reproduce:

```bash
git clone https://github.com/sgl-project/sglang.git sglang-main
cd sglang-main
git checkout b02e16a895add01a0cfe24bb74922de92ab4d895
git am /path/to/series-main/000*.patch        # or:
git apply /path/to/qwen4exp-serving-b02e16a8.patch
```

The main venv is built from this tree; see `../docs/logs/raw/migration_3090_2026-09-16/step2-venv.md`
and `dependency-delta.txt`. Material dependencies: `flashinfer_python==0.6.18`
(control 0.6.17), `sglang-kernel 0.4.7` (control 0.4.6.post1), `tilelang 0.1.12`
(control 0.1.11); the SM86 toolchain is matched to the control
(`nvidia-cuda-nvcc 13.3.73`, `nvidia-cuda-runtime 13.3.29`). A rebuilt main venv
also needs the two `nvidia/cu13` JIT linker symlinks recorded in `step2-venv.md`.

Feature disposition versus the frozen tree (details in `MANIFEST.md`):

- **Not ported** (measurement/debug only): `kv_stats`, `kv_fakeq`,
  `SGLANG_NAN_TRACE`, the prefill route dump, and the dead
  `_process_weights_after_loading_disabled`.
- **Kept disabled** in the default series: full-table gather, first-layer
  prefetch, QSA host-length metadata, eager shared overlap.
- **Not ported** (rejected): the paged-prefix kernel.
- The `qwen_sparse_attention` alias is already upstream and is not re-added.

Validation to 2026-09-17: the static/API gate passed; the candidate booted with
the control's effective profile; matched teacher-forced logprob A/B against the
frozen control was bit-exact (MAX 0.0000 / MEAN 0.00000 over 450 forced tokens);
prefix-cache repeat/append/next-session and the R1/R3 pressure churn reproducer
passed across two boots. The TG0 redo and TG1 comparison at
[`../docs/logs/tg0_tg1_candidate_3090_2026-09-17.md`](../docs/logs/tg0_tg1_candidate_3090_2026-09-17.md)
are a **comparability arm** (pre-promotion 256K, radix-off profile), not the
current baseline. Near-limit capacity on the promoted radix profile passed
(232,000 input + 512 generated, status ok, 27.79 tok/s, 18 MiB minimum free
VRAM, with the R3 refusal → R1 re-sort → sole-owner bypass lifecycle exercised;
`raw/migration_3090_2026-09-16/capacity-promoted.json`). Open: the reduced set
of GPU kernel/state micro-gates (INT2 packing/GEMV, offload storage swaps, PLE
read/eviction, graph outputs, VMM shrink/regrow/replay) were not run in this
pass; the unresolved PP14 long-prompt drift is carried forward; NGRAM is
untested. Evidence: `../docs/logs/raw/migration_3090_2026-09-16/` and
`../docs/logs/tg0_tg1_candidate_3090_2026-09-17.md`.

### Promotion and rollback (2026-09-17)

The default 3090 launcher `/root/quant/serve-3090.sh` now points at the
main-based implementation with the promoted radix profile: tree
`/root/quant/sglang-main`, venv `/root/quant/venv-sglang-main`, radix enabled,
4 Mamba slots, 235,520-token pool, `qwen38-flash-230K`, R1–R3 on, and
`--cuda-graph-backend-prefill disabled` (main no longer auto-disables prefill
graphs for a multimodal model). The installed launcher is
`serve-3090-main.sh` (sha256 `fb72fc95…49eccd`, port 30001); the distinct
candidate form is `serve-3090-main-candidate.sh` (port 30011, separate
`SGLANG_CACHE_DIR` and elastic control). The promoted boot verified the
effective profile (`max_total_num_tokens=235520`, page 64, ring 8192, S184).

Rollback, source, environment and launcher together:

```bash
# install and start the frozen flags-on control launcher (sha256 d0866e7c…f57447)
install -m 755 /path/to/repo/docs/logs/raw/promotion_3090_2026-09-16/serve-3090-flags.sh \
  /root/quant/serve-3090.sh
sha256sum /root/quant/serve-3090.sh     # must be d0866e7c…f57447
# stop the running scope, then start the frozen launcher
```

A local copy of the frozen launcher also sits at
`/root/quant/serve-3090-frozen-rollback.sh`. The frozen tree `/root/sglang` and
venv `/root/quant/venv-sglang` remain in place, so no rebuild is needed to roll
back.

Relationship to [`../patches/`](../patches/): the scripts there are the exact-string edit layers
that were applied one after another to the served tree; the flat patch is their flattened result
plus the base 2-bit patch. Layer order (only needed to peel layers off with the scripts): base <
host_fixes items < ncontig_gemv < placement < elastic < kv_lazy < ple_random < kv_fp8 < kv_int8 <
kv_int4 < kv_tiers.

The published launch line for the patched tree is [`../scripts/serve.sh`](../scripts/serve.sh)
(`--max-running-requests 1 --max-mamba-cache-size 1`); `PATCH_NOTES.md` section 6 lists its flags
and environment. Every headline log in `../docs/logs/` was recorded with that flag set, except
`night4.log` and `elastic.ctl.status`, which were recorded on the concurrent-benchmark restart #23
(`--max-running-requests 4 --max-mamba-cache-size 8`, not the published configuration).
