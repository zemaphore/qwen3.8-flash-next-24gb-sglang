#!/usr/bin/env bash
set -euo pipefail
cd /root/qwen3.8-flash-next-24gb-sglang
RAW="$PWD/docs/logs/raw/tg0_3090_2026-09-16"
PY=/root/quant/venv-sglang/bin/python
$PY tools/tg_profile.py capture --prompts-json "$RAW/prompts/prompts.json" \
  --prompts-class reasoning --context 2048 --out-dir "$RAW/profile" --label short-2048 --num-steps 20
$PY tools/tg_profile.py capture --prompts-json "$RAW/prompts/prompts.json" \
  --prompts-class reasoning --context 32768 --out-dir "$RAW/profile" --label long-32768 --num-steps 20
$PY tools/tg_numcheck.py --prompts-json "$RAW/prompts/prompts.json" \
  --prompts-class code --prefix-tokens 8180 --generate 32 --out "$RAW/numcheck/ring8192.json"
$PY tools/tg_numcheck.py --prompts-json "$RAW/prompts/prompts.json" \
  --prompts-class code --prefix-tokens 2048 --generate 32 --out "$RAW/numcheck/prefix2048.json"
$PY tools/snapshot_server.py "$RAW/manifest/server-boot2-final.json"
echo BOOT2_CONTINUE_DONE
