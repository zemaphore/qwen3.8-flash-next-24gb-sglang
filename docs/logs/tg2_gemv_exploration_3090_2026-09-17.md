# TG2: pointer-table INT2 decode GEMV launch-config exploration (RTX 3090)

Date: 2026-09-17 05:20–05:40 UTC. Scope: quick candidate scoring of the
existing INT2 decode GEMV launch configuration, not full verification or
promotion.

Profile under test (promoted main-based): `/root/quant/serve-3090.sh`
(sha256 `fb72fc95…49eccd`), tree `/root/quant/sglang-main`, venv
`/root/quant/venv-sglang-main`, port 30001. Effective per boot (baseline,
candidate, restore all identical): `max_total_num_tokens=235520`, radix,
4 Mamba slots, S184 presence placement
(`assets/expert_presence_code.pt`), `int8ring_int4` KV, R1–R3 on,
`--cuda-graph-backend-prefill disabled`. RTX 3090 SM86 only. TG1's
pooled-mass rejection was respected; presence placement was not touched.

## Baseline recorded

| Item | Value |
|---|---|
| SGLang tree revision | `/root/quant/sglang-main` HEAD `b114264c01`, clean working tree |
| GEMV kernel | `python/sglang/srt/layers/moe/expert_gemv.py` sha256 `f0850a1a…cc694` |
| Call site | `python/sglang/srt/layers/quantization/moe_wna16.py:569,572` (`_apply_gemv`) |
| Defaults in use | `block_n=64, block_k=128, num_warps=4` for both gate/up and down |
| Launcher | `/root/quant/serve-3090.sh` sha256 `fb72fc95…49eccd` |

Shapes (layer 5, checkpoint): gate/up `N=1280, K=2560, top_k=10`; down
`N=2560, K=640, top_k=1` with `R=10` rows.

## Experiment A — cheap kernel screening

Real checkpoint weights/scales, GPU-resident hot rows plus pinned-host cold
rows, allocation outside timing, active kernel imported from the checkout.
Grid `block_n∈{32,64,128} × num_warps∈{4,8}`, `block_k=128`; optional
`block_k=256` only for gate/up. The unchanged config was included in every
comparison; three interleaved rounds of 100 iterations, median used.

Residency mix: besides synthetic top-10 mixes (2, 4, 6 cold), an offline
check of the existing code route dumps against the exact top-184 `mass`
placement gives a real per-(layer,token) cold histogram:
`cold=0 8.4%, 1 13.9%, 2 15.5%, 3 15.1%, 4 13.8%, 5 11.4%, 6 8.8%, …`,
mean **3.56** cold experts/token. (Offline use of existing dumps; no new
route-capture pipeline.)

Weighted mean over that histogram (us per GEMV call; lower is better):

| Projection | Config (bn, warps, bk) | Weighted us | vs unchanged |
|---|---|---:|---:|
| gate/up | 64, 4, 128 (unchanged) | 323.2 | — |
| gate/up | **128, 8, 128** | **262.5** | **−18.8 %** |
| gate/up | 128, 4, 128 | 304.2 | −5.9 % |
| gate/up | 32, 4, 128 | 400.0 | +23.8 % |
| gate/up | 128, 8, 256 | 358.5 | +10.9 % |
| gate/up | 64, 4, 256 | 382.4 | +18.3 % |
| down | 64, 4, 128 (unchanged) | 192.5 | — |
| down | **128, 8, 128** | **178.8** | **−7.1 %** |
| down | 128, 4, 128 | 200.1 | +3.9 % |
| down | 32, 4, 128 | 214.7 | +11.5 % |

All-device-only (`cold=0`, 8.4 % of real cases) favours the unchanged/`bn=32`
configs (gate/up 48.2 us vs 52.3; down 15.0 vs 17.3), so the win is specific
to the mixed-residency regime that dominates decode. The mixed win is stable
from `cold=3` upward; at `cold=2` gate/up is a wash (150.3 vs 158.1).

Numerical spot-check on identical inputs/ids/weights/scales: every
`block_k=128` variant (including the candidate) produced **bit-identical**
outputs to the unchanged kernel (`max|Δ| = 0.0`). `block_k=256` differs by
`max|Δ| = 2.44e-4` (finite). No NaNs, no crashes.

## Experiment B — zero/one candidate end to end

Selected candidate: `block_n=128, num_warps=8, block_k=128` for **both**
gate/up and down (the clear mixed-regime winner; no config beat the unchanged
default at `cold=0` for either projection, but the real mix is dominated by
`cold≥2`). Wired only into the two `_apply_gemv` call sites of an isolated
copy `/root/quant/sglang-main-tg2cand` (diff retained as `candidate.diff`);
the promoted tree was never edited.

Results with the identical bounded request set (code 2048 and reasoning 32768
frozen TG0 token ids; 128 greedy tokens, EOS ignored; 1 excluded warmup + 2
measured per prompt; TG excludes TTFT):

| Cell | baseline | candidate | Δ | same-code noise control (independent re-run of baseline) |
|---|---:|---:|---:|---:|
| code 2048 (mean of 2) | 35.42 | 37.47 | **+5.8 %** | 35.72 (+0.8 %) |
| reasoning 32768 (mean of 2) | 38.34 | 36.98 | **−3.5 %** | 37.07 (−3.3 %) |

Individual measured requests — baseline code `[33.30, 37.54]`, candidate code
`[37.43, 37.51]`; baseline reasoning `[39.41, 37.27]`, candidate reasoning
`[35.54, 38.43]`. Generated counts were exactly 128 for every request; TTFT
and total wall times overlapped between arms; candidate boot had 0
`Traceback` / 0 `OutOfMemory`. Both candidate workloads are individually
inconclusive (−3.5 %/+5.8 % on n=2), and the same-code noise control already
spans ±3 %, so the two cells conflict in sign.

## Verdict

**Inconclusive.** The kernel-level candidate is credible and repeatable in the
mixed-residency regime (gate/up −18.8 %, down −7.1 % weighted; bit-identical
outputs), but the tiny end-to-end sample does not show a consistent ≥5 % TG
gain: code +5.8 % and reasoning −3.5 %, within the ±3 % same-code noise. Per
the task rules this is not promotable, and kernel timings alone do not imply a
production win.

### Recommended next action

Run a short **bracketed** end-to-end comparison (baseline, candidate, baseline
or candidate, baseline, candidate) with more measured requests per prompt and
both workload classes, using `block_n=128, num_warps=8, block_k=128` on both
projections. Only if the mixed-regime gain survives bracketing should this be
considered for a promotion-grade validation.

### Coverage limits / notes

- Single candidate boot; n=2 measured per workload; no statistical confidence.
- Synthetic mixed residency plus an offline cold histogram from existing
  prefill route dumps (not measured decode routing); not production routing.
- No PP, NLL, needle, pressure, long-context capacity or full-matrix work.
- PP14 remains deferred and was not touched.
- The first "candidate" boot actually re-ran the baseline code: the venv is an
  editable install pointing at `/root/quant/sglang-main/python`, so `cwd` does
  not select the source tree. It was detected via `/proc/<pid>/cwd` + import
  check, then re-run with `SGLANG=/root/quant/sglang-main-tg2cand/python`
  (candidate package root), verified by import. That first boot is retained as
  the same-code noise control above.

## Restoration

Stopped the candidate scope and re-ran `/root/quant/serve-3090.sh` with no
overrides. Confirmed: port 30001 `/health` = 200; scheduler cwd
`/root/quant/sglang-main`; launcher sha256 `fb72fc95…49eccd`; boot log
`max_total_num_tokens=235520`, `S=184`, Mamba 4. The candidate copy
`/root/quant/sglang-main-tg2cand` was removed after the run; the promoted tree
and launcher are unchanged.

## Raw artifacts

`docs/logs/raw/tg2_3090_2026-09-17/`: `tg2_probe.py`, `tg2_screen.py`,
`tg2_screen2.py`, `screen.json`, `screen2.json`, `candidate.diff`,
per-boot `candidate*/`, `baseline/` sample JSON + `summary.json`,
`tg2_candidate*.server.log`, `tg2_restore.launch.out`.
