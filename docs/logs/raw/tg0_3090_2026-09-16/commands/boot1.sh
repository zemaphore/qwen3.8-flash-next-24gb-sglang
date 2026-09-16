#!/usr/bin/env bash
set -euo pipefail
cd /root/qwen3.8-flash-next-24gb-sglang
export SGLANG_MOE_ELASTIC_CTL="/root/qwen3.8-flash-next-24gb-sglang/docs/logs/raw/tg0_3090_2026-09-16/elastic.ctl"
export TG0_SERVER_LOG="/root/qwen3.8-flash-next-24gb-sglang/docs/logs/raw/tg0_3090_2026-09-16/server.log"
exec /root/quant/venv-sglang/bin/python tools/tg_baseline.py \
  --prompts-json "/root/qwen3.8-flash-next-24gb-sglang/docs/logs/raw/tg0_3090_2026-09-16/prompts/prompts.json" \
  --out-dir "/root/qwen3.8-flash-next-24gb-sglang/docs/logs/raw/tg0_3090_2026-09-16/boot1" --boot-label boot1 --matrix full \
  --warmups 1 --measured 5 --decode-tokens 512 --capacity
