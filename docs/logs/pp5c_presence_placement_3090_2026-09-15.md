# PP5c — captured PP5b rerun and held-out code validation: ACCEPTED

Date: 2026-09-15. Host: RTX 3090, driver 580.178.04. Stack: accepted PP11
batched host-row DMA staging on top of PP7, 2,048-token chunks, S184 residency,
bulk PLE pread plus the 128 MiB recent-row cache, 2,048-byte gather tile. No
production flag was changed except `SGLANG_MOE_PLACEMENT`, which is the single
A/B variable. No repeated-sentence benchmark or `llama-benchy` was run.

## Acceptance decision

**PROMOTE presence placement to the general launcher default.** The decision
proceeds from raw captured evidence, not the aggregate-only PP5b table.

- Canonical corpus gate: presence **1247.2 +/- 26.6** versus pooled mass
  **1185.9 +/- 17.1 PP tok/s** = **+5.17%** (t = 4.68), clearing the +5% gate.
- Held-out code: neither corpus materially regresses (>5%). `heldout-quantize`
  is **-0.91%** and `heldout-phase1` is **+1.47%**; combined **+0.29%**.
- Both exactness oracles match the accepted baseline exactly in all three arms.
- Every arm ran error-free with an unchanged environment and the same minimum
  free VRAM; capture is complete and checksummed.

Because the canonical gate clears and no held-out corpus materially regresses,
presence is promoted with `patches/enable_prefill_presence_placement.py apply`.
Routing mass remains selectable with
`SGLANG_MOE_PLACEMENT=/root/qwen3.8-flash-next-24gb-sglang/assets/expert_freq.pt`.

## Raw evidence

Three sequential fresh-server arms, never concurrent, each with its own server
log and output directory under
[`docs/logs/raw/pp5c_3090_2026-09-15/`](raw/pp5c_3090_2026-09-15/):

| Arm | Placement | Scope | Directory |
|---|---|---|---|
| mass-a | `assets/expert_freq.pt` | `sglang-1789504276.scope` | [mass-a](raw/pp5c_3090_2026-09-15/mass-a/) |
| presence | `assets/expert_presence_code.pt` | `sglang-1789504677.scope` | [presence](raw/pp5c_3090_2026-09-15/presence/) |
| mass-c | `assets/expert_freq.pt` | `sglang-1789505056.scope` | [mass-c](raw/pp5c_3090_2026-09-15/mass-c/) |

Each directory holds `metadata.json` (repo commit, live server argv and filtered
environment, placement hash/signature, endpoint health), `results.json`, the
unparsed `bench-*.txt` client output for every request, `nvidia-smi.csv` GPU
telemetry, `elastic-status.before/after.txt`, `server-launcher.snapshot.sh`,
`server.log.before.txt`, a frozen `server.log.after.txt`, the live `server.log`,
both oracle outputs, and `SHA256SUMS`.

`sha256sum -c SHA256SUMS` passes for every file except the live `server.log`,
which kept receiving launcher output after the capture-time checksum was
written; the frozen `server.log.after.txt` verifies and is a confirmed
append-only prefix of the live log. No other artifact is affected.

## Method

`tools/capture_pp5b_arm.py` was run against each freshly started server after
`/health` readiness, without `--skip-heldout`. The launcher stack is the
accepted one; only `SGLANG_MOE_PLACEMENT` and `SRVLOG` were set per arm. Between
arms only the exact active `sglang-*.scope` was stopped
(`systemctl --user stop`); no unrelated process was touched.

Each arm collects one excluded plus five measured canonical requests and one
excluded plus three measured requests for each held-out corpus
(`tools/bench_agentic.py --tokens 4096 --decode-tokens 256`). Warmup (excluded)
samples are dropped from every statistic below.

## Canonical corpus (`sglang/qwen4exp-serving-73a255206f.patch`)

Measured-only samples, excluded sample removed.

| Arm | Actual prompt tokens | Measured PP tok/s | Mean +/- sample SD | Decode mean |
|---|---:|---|---:|---:|
| mass-a | 4,565 | 1185 / 1203 / 1171 / 1177 / 1201 | 1187.4 +/- 14.2 | 42.4 |
| presence | 4,565 | 1240 / 1271 / 1204 / 1260 / 1261 | **1247.2 +/- 26.6** | 40.9 |
| mass-c | 4,565 | 1185 / 1211 / 1191 / 1152 / 1183 | 1184.4 +/- 21.2 | 39.1 |
| Pooled mass | 4,565 | ten samples | 1185.9 +/- 17.1 | 40.7 |

Presence versus pooled mass: **+61.3 tok/s, +5.17%** (t = 4.68). Excluded
warmup PP was 743 (mass-a), 873 (presence), 824 (mass-c).

## Held-out corpora

Measured-only samples, excluded sample removed.

| Corpus | Pooled mass | Presence | Gain vs pooled mass |
|---|---:|---:|---:|
| `scripts/05_quantize.py` (3,866 tok) | 1324.3 +/- 14.6 (n=6) | 1312.3 +/- 29.4 (n=3) | **-0.91%** (t = -0.67) |
| `scripts/phase1.py` (3,810 tok) | 1348.8 +/- 29.0 (n=6) | 1368.7 +/- 4.0 (n=3) | **+1.47%** (t = 1.64) |
| Combined held-out | 1336.6 +/- 25.3 (n=12) | 1340.5 +/- 36.1 (n=6) | **+0.29%** |

Per-arm held-out detail:

| Arm | quantize PP tok/s | phase1 PP tok/s |
|---|---|---|
| mass-a | 1338 / 1324 / 1314 (mean 1325.3 +/- 12.1) | 1315 / 1392 / 1364 (mean 1357.0 +/- 39.0) |
| presence | 1299 / 1346 / 1292 (mean 1312.3 +/- 29.4) | 1373 / 1365 / 1368 (mean 1368.7 +/- 4.0) |
| mass-c | 1325 / 1303 / 1342 (mean 1323.3 +/- 19.6) | 1332 / 1363 / 1327 (mean 1340.7 +/- 19.5) |

The canonical benefit does not transfer as a gain on held-out code, but it is
also not a material regression: presence is within run-to-run noise of pooled
mass on both held-out corpora and numerically positive on the combined average.

## Exactness

All three arms:

```text
m4_3090_untuned: LOGPROB_MAX=0.000000 LOGPROB_MEAN=0.000000
lp2:             LOGPROB_MAX=0.168757 LOGPROB_MEAN=0.011632
```

These are identical to the PP1b/PP2a/PP3/PP4/PP7/PP11/PP5b baseline. Placement
only changes which rows are resident versus staged on the host; PP7 stages
identical bytes per expert and the fused kernel is untouched, so the change is
exact by construction and the oracles confirm it.

## Telemetry, environment, and errors

- Live `SGLANG_MOE_PLACEMENT` resolved to the requested asset in each arm
  (`metadata.json`, `server_environment`); placement signatures matched the
  elastic status every time (mass 0.8430, presence 0.4665).
- Every other PP variable was identical across arms:
  `SGLANG_MOE_GATHER_BLOCK=2048`, `SGLANG_MOE_GATHER_DMA=1`,
  `SGLANG_MOE_GATHER_DMA_BATCH=1`, `SGLANG_MOE_PLACEMENT_S=184`,
  `SGLANG_MOE_EXPERT_STREAM=1`, `SGLANG_MOE_ELASTIC=1`, PLE pread + 128 MiB
  recent cache, chunked-prefill-size 2048. Elastic status stayed fixed at S184
  (`S_min 184 S_max 184 floor 184`) before and after every capture.
- Minimum free VRAM sampled during each capture: 2,941 MiB in all three arms.
- No OOM, retraction, CUDA fault, traceback, workload exception, or server
  error in any arm. The only `error` log matches are the `server_args` dump,
  the unrelated optional `sarashina2_vision` import, and the `CUDA VMM FABRIC`
  fallback probe, exactly as documented for earlier accepted runs.
- Placement hashes: mass
  `96697b3c5e4f847f42c4321b440b44a26da552e97a58fe9672549bb3ef5052cf`,
  presence
  `73d8f0df1875f6b61124b4835173dccf56d57792fa91bee8d270fc65b42ddfd8`.

## Reproduction

```bash
cd /root/qwen3.8-flash-next-24gb-sglang
for arm in mass-a presence mass-c; do
  case $arm in
    mass-a|mass-c) P=assets/expert_freq.pt ;;
    presence)      P=assets/expert_presence_code.pt ;;
  esac
  D=docs/logs/raw/pp5c_3090_2026-09-15/$arm
  mkdir -p "$D"
  SRVLOG="$PWD/$D/server.log" SGLANG_MOE_PLACEMENT="$PWD/$P" /root/quant/serve-3090.sh
  /root/quant/venv-sglang/bin/python tools/capture_pp5b_arm.py \
    --arm "$arm" --placement "$P" --output-dir "$D" --server-log "$D/server.log"
  systemctl --user stop "$(systemctl --user list-units 'sglang-*.scope' --plain --no-pager | awk '/sglang-/{print $1}')"
done
```

## Caveat

Presence is derived from the canonical code prompt's routing, so its positive
gain is canonical-specific. Held-out code shows no material regression and no
transferable gain, which is consistent with a placement tuned to a particular
routing distribution. If a materially different workload mix replaces the code
benchmark as the primary target, re-derive the placement before assuming the
canonical gain carries over.
