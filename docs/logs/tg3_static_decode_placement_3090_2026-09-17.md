# TG3: decode-selection static expert placement vs presence at S184 (RTX 3090)

Date: 2026-09-17 06:09–06:27 UTC. Profile: promoted main-based serving stack
(`/root/quant/serve-3090.sh` sha256 `fb72fc95…49eccd`, tree
`/root/quant/sglang-main`, venv `venv-sglang-main`, port 30001) unchanged:
presence S184, radix, 235520 pool, `int8ring_int4` KV, 4 Mamba slots,
`--cuda-graph-backend-prefill disabled`, breakable decode graph. No promotion.

## 1. Decode selection capture (graph-compatible)

Used SGLang main's built-in capturer
(`python/sglang/srt/state_capturer/routed_experts.py`), the graph-compatible
in-graph mechanism, not the unusable Python-in-`apply` patch
(`patches/decode_route_dump.py`, which records only at graph *capture*). One
diagnostic boot with `--enable-return-routed-experts` (extra server arg only;
serving profile otherwise identical). Log:

```
DeviceCache[routed_experts] allocated: shape=(4608, 48, 10), size=8.44 MB
HostCache[routed_experts]  allocated: shape=(235584, 48, 10), size=0.42 GB
```

Requests set `return_routed_experts=true` and
`routed_experts_start_len=<prompt_len>`, so returned rows are the decode-step
forward positions only (prefill rows excluded).

**Capture verification (generated-token steps, not capture/prefill):** four
prompts, 2048 input tokens, 96 generated tokens → 95 returned rows each, and
**95/95 rows distinct** per prompt (`rows_equal_capture_constant=false`). A
capture-time or constant-routing artifact would give one repeated row.

## 2. Prompt split and candidate construction

Split frozen **before** inspecting routing (`tg3_make_prompts.py`), 2048 tokens
each, disjoint:

| Set | Source | Use |
|---|---|---|
| `code_cal`, `reasoning_cal` | tokens [0:2048] of the code / reasoning corpora | calibration |
| `code_held`, `reasoning_held` | tokens [4096:6144] of the same corpora | held-out |

One static candidate: per layer, rank the 512 experts by **decode selection
count** (not gate-weight mass) summed over the two calibration sets, keep the
top 184. Saved as `tg3_candidate_selection.pt` (sha256 `109c1be1…c844b1`,
`mass` = count). 96 decode tokens per calibration prompt (192 total).

## 3. Offline held-out scoring (presence vs candidate)

Cold = selected expert outside that layer's top-184. Totals over the two
held-out sets (190 decode-token steps, 48 layers × 10 selections/token).

| Placement | cold sel/token | resident sel. fraction | est. host traffic (MB/token) |
|---|---:|---:|---:|
| presence | 197.31 | 0.589 | 257.6 |
| candidate | 121.62 | 0.747 | 158.8 |

Per workload:

| Held-out | presence cold/tok | candidate cold/tok | ratio |
|---|---:|---:|---:|
| code_held | 194.74 | 101.55 | 0.521 (−47.9 %) |
| reasoning_held | 199.87 | 141.68 | 0.709 (−29.1 %) |

Cold selections fall **38.4 % overall, with no workload worsening**, clearing
the ≥15 % gate. (Bytes are **estimates**: selection count × per-expert
qweight+scales 1,305,600 B; not measured DMA. Per-layer arrays in `score.json`.)

## 4. Candidate timing boot (one candidate boot + baseline restore)

Same procedure both arms: the two held-out prompts, 256 greedy tokens, EOS
ignored, one excluded warmup + three measured. Candidate = unchanged server
with `SGLANG_MOE_PLACEMENT=tg3_candidate_selection.pt` (S184); baseline =
unchanged launcher.

| Prompt | baseline samples (tok/s) | mean | candidate samples | mean | Δ |
|---|---|---:|---|---:|---:|
| code_held | 33.94, 31.70, 33.81 | 33.152 | 46.92, 47.58, 47.36 | 47.287 | **+42.6 %** |
| reasoning_held | 36.34, 37.13, 36.25 | 36.573 | 31.53, 34.61, 34.33 | 33.490 | **−8.4 %** |

Balanced score (geometric mean of the two speedup ratios): **+14.29 %**.

## Verdict

**Promising, workload-asymmetric — not promoted.** The static decode-selection
placement clears the offline gate and is a large, repeatable win on the
code-style held-out prompt (+42.6 %, all three samples well above baseline),
but it **regresses the reasoning-style held-out prompt (−8.4 %, all three
samples below baseline)**. It therefore does not cleanly beat presence across
workloads at fixed S184, and the count-based cold reduction (−29 % reasoning)
did not translate into reasoning speed. The candidate asset is retained; no
promotion and no further extension.

## Coverage limits

- n=3 measured per prompt, one prompt per class; not statistical confidence.
- One candidate boot and one diagnostic-capture boot plus baseline restore.
- Calibration = 192 decode tokens; held-out = 190; prompt excerpts are
  repo-corpus text, not the served workload mix.
- Host traffic is an estimate, not measured DMA.
- No PP, NLL, needle, pressure, long-context or full-matrix work; PP14 remains
  deferred and untouched.

## Restoration

Diagnostic launcher removed; original `/root/quant/serve-3090.sh` re-run with
no overrides. Confirmed: port 30001 `/health` 200; scheduler cwd
`/root/quant/sglang-main`; launcher sha256 `fb72fc95…49eccd`; log
`max_total_num_tokens=235520`, `S=184`, Mamba 4. Promoted tree clean.

## Raw artifacts

`docs/logs/raw/tg3_3090_2026-09-17/`: `tg3_make_prompts.py`, `tg3_capture.py`,
`tg3_score.py`, `tg3_timing.py`, `prompts.json`, `cap/` (raw `*.npy` +
`manifest.json`), `score.json`, `tg3_candidate_selection.pt`, `t_candidate/`
and `t_baseline/` sample JSON + `summary.json`, `tg3_capture.server.log`,
`tg3_candidate.server.log`, `tg3_baseline.server.log`, launch outputs.

---

# Addendum: reasoning-regression diagnostic (2026-09-17)

Bounded diagnostic of the reasoning_held regression. No promotion, no asset
retraining, no new workloads, no timing matrix. **Sections 3–4 cold-selection
numbers above are SUPERSEDED by the corrected scoring below** (same overall
conclusion; per-workload sets shifted at the tie boundary).

## A1. Corrected offline scoring (server-exact `torch.argsort`)

The original candidate ranking used `np.argsort(-count)`, whose tie-breaking
differs from the server's resident selection
(`moe_wna16.py:466`: `torch.argsort(freq[layer], descending=True)[:S]`).
Recomputed with `torch.argsort` for both placements (`tg3_diag.py`,
`diag.json`):

| Held-out | presence cold/tok | candidate cold/tok | ratio |
|---|---:|---:|---:|
| code_held | 194.74 | 100.40 | 0.516 (−48.4 %) |
| reasoning_held | 199.87 | 142.69 | 0.714 (−28.6 %) |
| overall | — | — | **−38.4 %** |

Original (numpy): code −47.9 %, reasoning −29.1 %, overall −38.4 %. Overall
unchanged; per-workload numbers move by ~1 pt. The verdict is unaffected.

## A2. 256-token reasoning trace under both placements

One request per arm on the same `reasoning_held` prompt ids, greedy, EOS
ignored, via the same graph-compatible capturer; generated token IDs retained.

| Arm boot | prompt | generated | rows | rows_ok | distinct rows |
|---|---:|---:|---:|---|---:|
| presence | 2048 | 256 | 255 | True | 255/255 |
| candidate | 2048 | 256 | 255 | True | 255/255 |

Row/token alignment: `rows = generated − 1` (the prefill-produced first token
has no decode row); row *j* is the forward that produced `output_ids[j+1]`.
Verified programmatically (`row_token_alignment_ok=True`) in `cap256_*/manifest.json`.

Each trace scored under **both** placements (cold selections/token; standard
deviation small, single trace per arm), first 95 steps vs remaining 160:

| Trace (history) | scored under | first 95 | rest 160 | overall | est. MB/tok |
|---|---|---:|---:|---:|---:|
| presence-generated | presence | 168.2 | 182.0 | 176.9 | 230.9 |
| presence-generated | candidate | 163.6 | 182.8 | 175.6 | 229.3 |
| candidate-generated | presence | 203.3 | 178.4 | 187.6 | 245.0 |
| candidate-generated | candidate | 157.4 | 181.4 | 172.4 | 225.1 |

Within-trace placement effect (candidate − presence): presence-generated
first95 −2.7 %, rest **+0.4 %**, overall −0.7 %; candidate-generated first95
−22.6 %, rest **+1.7 %**, overall −8.1 %. Per layer (overall, candidate
better): 25/48 layers on the presence trace, 34/48 on the candidate trace; in
the rest window the candidate is worse more often than better.

The two arms' presence-scored cold counts differ (176.9 vs 187.6) because the
generated histories themselves diverge between boots (SGLang is not bitwise
reproducible), which is exactly why each trace is scored under both placements.

## A3. Answer

**The estimated host-read reduction does not persist in later reasoning — it
is concentrated in the first ~95 decode steps and disappears (slightly
reverses) afterward.** The original 96-token offline probe measured only that
early window, so its −29 % reasoning figure was an early-horizon artifact; the
later 256-token timing regression is consistent with there being no sustained
cold-read benefit. Inconclusive beyond that: one trace per arm, one prompt.

## A4. Recommended next action

Rank the static asset over a longer decode horizon (≥256–512 tokens) or use a
recency-weighted selection count, and re-score held-out with a longer window
before any further timing; do not rerun the TG timing on the current asset.

## A5. Restoration (addendum)

`serve-3090-tg3cap.sh` removed; normal `/root/quant/serve-3090.sh` re-run with
no overrides: health 200, scheduler cwd `/root/quant/sglang-main`, launcher
sha256 `fb72fc95…49eccd`, `S=184`, `max_total_num_tokens=235520`; promoted tree
clean. New raw artifacts: `tg3_diag.py`, `diag.json`, `diag_step1.json`,
`cap256_pres/`, `cap256_cand/`, `tg3_cap_pres.*`, `tg3_cap_cand.*`,
`tg3_restore2.*`.

---

# TG3b: replacement static-placement asset (2026-09-17)

The earlier TG3 asset is **parked** (its later-window benefit disappeared).
One replacement asset, one capture boot, one candidate timing boot. Presence
S184 remains the default; no promotion.

## B1. Frozen split (before capture)

Same code/reasoning corpora; disjoint from the old slices:

| Role | Key | Source slice (tokens) |
|---|---|---|
| calibration | `code_cal1`, `code_cal2` | code [0:2048], [8192:10240] |
| calibration | `reasoning_cal1`, `reasoning_cal2` | reasoning [0:2048], [8192:10240] |
| held-out (fresh) | `code_held_fresh` | code [16384:18432] |
| held-out (fresh) | `reasoning_held_fresh` | reasoning [16384:18432] |

The old held-out pair (`code_held`/`reasoning_held`) was **not** reused.
Prompt IDs + sha256 in `prompts.json`.

## B2. Capture

One presence-profile boot with `--enable-return-routed-experts` (no other
change). Six requests, 512 greedy tokens each, same graph-compatible capturer.
Every prompt: **rows=511, out_ids=512, alignment ok, 511/511 distinct rows**
(row *j* ↔ `output_ids[j+1]`; the prefill-produced first token has no decode
row). Raw traces in `cap/`.

## B3. Asset

One static top-184-per-layer asset from decode selection counts summed over the
four calibration traces (equal weight per calibration prompt; both classes have
two prompts of equal length). Server-exact ordering
`torch.argsort(mass, dim=1, descending=True)[:, :184]`.

`tg3b_candidate_selection.pt` sha256 `e68dd563…84f1c4`
(`mass` = calibration selection count).

## B4. Offline scoring on the fresh held-out traces

Cold = selected expert outside that layer's top-184; both placements scored on
the **same** held traces. Windows: first 128 vs remaining 383 decode steps.

| Held-out | placement | cold/tok overall | first 128 | rest 383 |
|---|---|---:|---:|---:|
| code_held_fresh | presence | 167.9 | — | 169.0 |
| code_held_fresh | candidate | 63.5 | — | 60.1 |
| reasoning_held_fresh | presence | 155.1 | — | 152.8 |
| reasoning_held_fresh | candidate | 98.3 | — | 90.9 |

Ratios (candidate/presence):

| Held-out | overall | first 128 | remaining |
|---|---:|---:|---:|
| code | 0.378 (−62.2 %) | 0.445 | 0.356 (−64.4 %) |
| reasoning | 0.634 (−36.6 %) | 0.745 | 0.595 (−40.5 %) |
| both (summed) | **0.501 (−49.9 %)** | — | **0.469 (−53.1 %)** |

Host-byte **estimates** (cold count × 1,305,600 B/expert): presence ≈ 211,
candidate ≈ 106 MB/token overall. Gate: overall and later-window reductions
≥15 % and no workload worsening overall or in either window →
**qualifies for timing**.

## B5. Candidate timing (one boot, no capture)

Same fresh held-out prompts, 512 greedy tokens, EOS ignored, one excluded
warmup + three measured each; baseline measured on the unchanged running
server with the identical procedure.

| Prompt | baseline samples (tok/s) | mean | candidate samples | mean | Δ |
|---|---|---:|---|---:|---:|
| code_held_fresh | 37.66, 37.20, 38.45 | 37.772 | 50.32, 46.84, 52.54 | 49.899 | **+32.1 %** |
| reasoning_held_fresh | 37.87, 37.40, 40.21 | 38.493 | 43.77, 44.06, 41.66 | 43.164 | **+12.1 %** |

Balanced score (geometric mean): **+21.71 %**. Both workloads improve; the
reasoning regression of the parked asset is gone.

## Verdict

**Qualifying candidate — PROMOTED 2026-09-17 (bounded acceptance).** The
replacement asset reduces estimated cold selections by ~50 % overall and
~53 % in the later window, with neither workload worse, and delivers
+32 % / +12 % TG (balanced +21.7 %) on fresh held-out prompts. It is a genuine
512-token-horizon improvement over presence at fixed S184. After the
acceptance check below (both conversations ≥5 % total-wall reduction, lower
aggregate decode), the asset was installed as
`assets/expert_decode_selection.pt` (sha256 `e68dd563…84f1c4`) and made the
launcher `SGLANG_MOE_PLACEMENT` default (launcher sha256 `0d1cd1fb…a8d2d`;
presence remains the rollback/override default). No further asset iteration.

## Coverage limits

- One held-out prompt per class; n=3 measured per prompt; no statistical
  confidence.
- Calibration = 4 × 511 decode steps; held-out = 2 × 511; host bytes estimated.
- One capture boot + one candidate boot + baseline restore; prompt text is
  repo-corpus, not the served workload mix.
- No PP, NLL, needle, pressure, long-context, profiling or full verification;
  PP14 remains deferred and untouched.

## Restoration (TG3b)

`serve-3090-tg3bcap.sh` removed; `/root/quant/serve-3090.sh` re-run with no
overrides: health 200, scheduler cwd `/root/quant/sglang-main`, launcher
sha256 `fb72fc95…49eccd`, `S=184`, `max_total_num_tokens=235520`.

## Raw artifacts (TG3b)

`docs/logs/raw/tg3b_3090_2026-09-17/`: `tg3b_make_prompts.py`,
`tg3b_build_score.py`, `tg3b_timing.py`, `prompts.json`, `cap/` (`*.npy` +
`manifest.json`), `report.json`, `tg3b_candidate_selection.pt`, `t_baseline/`,
`t_candidate/` sample JSON + `summary.json`, `tg3b_cap.server.log`,
`tg3b_candidate.server.log`, `tg3b_restore.server.log`, launch outputs.

---

# TG3b acceptance check of the frozen asset (2026-09-17)

Bounded operational check of the frozen candidate
(`tg3b_candidate_selection.pt` sha256 `e68dd563…84f1c4`) versus presence at
fixed S184; otherwise identical promoted profile. No calibration/tuning.

## Conversations (frozen before running)

Two fixed conversations, each base 32768 tokens with two appended fixed
follow-up turns; input histories identical across arms and the generated
answers are never fed into later prompts. Turn inputs grow by prepending the
same frozen base + follow-ups (prefix reuse).

| Conversation | base | follow1 | follow2 | turn prompt tokens |
|---|---:|---:|---:|---|
| code | 32768 | 22 | 16 | 32768, 32790, 32806 |
| reasoning | 32768 | 23 | 16 | 32768, 32791, 32807 |

Per arm: one short excluded warmup, `POST /flush_cache` before each
conversation (same reset procedure), then the three turns sequentially on the
same server; 256 greedy tokens/turn, EOS ignored, temp 0. Effective asset
verified from the scheduler/launcher environment: candidate
`SGLANG_MOE_PLACEMENT=…/tg3b_candidate_selection.pt` (hash matches), S184;
baseline `…/assets/expert_presence_code.pt`, S184.

## Results

| Conversation | arm | total wall (s) | total decode (s) |
|---|---|---:|---:|
| code | presence | 42.29 | 20.86 |
| code | candidate | 38.39 | 18.21 |
| reasoning | presence | 45.66 | 21.13 |
| reasoning | candidate | 41.79 | 18.82 |

| Conversation | total wall Δ | total decode Δ |
|---|---:|---:|
| code | **−9.22 %** | **−12.70 %** |
| reasoning | **−8.47 %** | **−10.90 %** |

Per-turn (wall s; baseline → candidate):

| Turn | code | reasoning |
|---|---|---|
| 1 (cold prefill) | 28.22 → 25.02 (−11.3 %), TTFT 20.08 → 19.46 | 30.64 → 28.49 (−7.0 %), TTFT 23.88 → 22.37 |
| 2 | 7.95 → 6.90 (−13.2 %) | 7.54 → 6.63 (−12.1 %) |
| 3 | 6.12 → 6.47 (**+5.8 %**), TG 43.7 → 41.4 | 7.48 → 6.67 (−10.9 %) |

Cold-prefill TTFT did not regress in either conversation. One individual-turn
regression is reported separately: **code turn 3 wall +5.8 %** (TTFT +0.03 s
and decode slightly slower), which is not part of the acceptance criterion and
is offset within the conversation total. All other five turns improved.

## Recommendation

**ACCEPT → PROMOTED (2026-09-17).** The exact frozen TG3b candidate lowers
total conversation wall time by ≥5 % for **each** conversation (code −9.2 %,
reasoning −8.5 %) with aggregate decode time also lower for each (code
−12.7 %, reasoning −10.9 %). The single code-turn-3 wall regression is
reported separately and does not change the per-conversation totals. The asset
was installed as `assets/expert_decode_selection.pt` and set as the launcher
`SGLANG_MOE_PLACEMENT` default (presence retained as the rollback override).
Promotion is bounded by the coverage limits below, not full verification.

## Coverage limits

- One conversation per class, one boot per arm; this is a bounded operational
  check, not statistical proof or full correctness verification.
- Per-turn decode metric excludes TTFT; wall includes it. Cold prefill is the
  turn-1 prefill only.
- No full matrix, capacity sweep, NLL, pressure, profiling or PP14 work.

## Restoration (acceptance)

Normal `/root/quant/serve-3090.sh` re-run with no overrides: health 200,
launcher cwd `/root/quant/sglang-main`, launcher sha256 `fb72fc95…49eccd`,
`SGLANG_MOE_PLACEMENT=assets/expert_presence_code.pt`, S184,
`max_total_num_tokens=235520`. Candidate asset retained, not promoted.

## Raw artifacts (acceptance)

`docs/logs/raw/tg3b_3090_2026-09-17/`: `tg3c_make_conv.py`, `tg3c_accept.py`,
`conv.json`, `acc_baseline/` and `acc_candidate/` `summary.json`,
`tg3c_candidate.server.log`, `tg3c_restore.server.log`, launch outputs.
