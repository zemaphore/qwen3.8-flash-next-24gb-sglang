# PP campaign wrap-up raw evidence (RTX 3090, 2026-09-16)

Accepted serving revision for every row below: repo `a50c32a`, serving source
`/root/sglang` at upstream `73a255206f` plus the applied patch set (diff SHA-256
in `manifest/serving-source.txt`). No server is running after this capture.

## Layout

| Path | Contents |
|---|---|
| `accepted/` | `tools/capture_pp5b_arm.py` run on the accepted defaults: canonical (one excluded warmup + five measured) and two held-out code corpora (one excluded + three measured each), both short exactness oracles, GPU samples, elastic status before/after, launcher snapshot, and before/after server logs. `metadata.json` binds the live server argv, all `SGLANG_*` settings, placement/config hashes, and the repo revision. |
| `long-oracle/` | Long-prompt teacher-forced logprob runs (`tools/long_logprob_oracle.py`): accepted repeats run1-3, prefetch-off repeats run1-2, the comparison `summary.json`, and a copy of the tool. |
| `pre-pp15-moe/` | One long-oracle check against the pre-PP15 INT2 MoE configs (M=4565 entry removed for E=384/E=432), live server snapshot, and config-key listing. |
| `prefetch-off/` | One long-oracle repeat against `SGLANG_MOE_COLD_PREFETCH=0`, live server snapshot. |
| `depth-sweep.json` | Prefill depth sweep on the accepted stack, one request per depth, actual server token counts and failures. |
| `activation-evidence.txt` | Proof the accepted stack took the intended paths: PP14 prefetch cache/active lines, PP15 E=432 nearest-bucket selection, and the config-key difference vs pre-PP15. |
| `manifest/` | Reproducibility bundle: repo/serving-source revisions, patch states, launcher and asset hashes, software versions, GPU settings, and the accepted live server snapshot. |

## Warmup and exclusion policy (fixed before measurement)

- `capture_pp5b_arm.py`: sample 0 of every corpus is a discarded warmup and is
  retained as `bench-<corpus>-excluded-00.txt`; measured means use samples 1..N.
- `depth_sweep.py`: exactly one request per depth, increasing order, no sample
  dropped; the launcher's own warmup is not part of the sweep.
- `long_logprob_oracle.py`: every run is retained; no run is dropped. Repeat
  runs are reported as run-to-run variability, not averaged away.

No two different server sessions are pooled as if they were a controlled A/B.
The accepted numbers here and the PP15 acceptance number (2687.4 +/- 48.3) come
from different sessions and are reported separately.

## Reproduction

```bash
# accepted stack
/root/quant/serve-3090.sh
# final canonical + held-out capture (server already up)
/root/quant/venv-sglang/bin/python tools/capture_pp5b_arm.py \
  --arm accepted --placement assets/expert_presence_code.pt \
  --output-dir docs/logs/raw/pp_wrapup_3090_2026-09-16/accepted \
  --server-log /root/quant/logs/server-wrapup-accepted.log --chunk-size 4608
# long-prompt oracle
SGLANG_URL=http://127.0.0.1:30001/generate \
  /root/quant/venv-sglang/bin/python tools/long_logprob_oracle.py save accepted-run1
# depth sweep
SGLANG_URL=http://127.0.0.1:30001/generate \
  /root/quant/venv-sglang/bin/python tools/depth_sweep.py \
  --depths 8192 32768 65536 131072 163840 180224 196608 204800 245760 262144 \
  --decode-tokens 16 --out docs/logs/raw/pp_wrapup_3090_2026-09-16/depth-sweep.json
```

The pre-PP15 arm used `ASSETS=/root/quant/assets_pre_pp15`, reconstructed by
copying `assets/` and restoring the four `E={384,432}` config files from
`adc57b8`. The prefetch-off arm used `SGLANG_MOE_COLD_PREFETCH=0`.
