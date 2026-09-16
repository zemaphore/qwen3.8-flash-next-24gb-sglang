#!/usr/bin/env bash
set -euo pipefail
cd /root/qwen3.8-flash-next-24gb-sglang
RAW="$PWD/docs/logs/raw/tg0_3090_2026-09-16"
/root/quant/venv-sglang/bin/python tools/tg_pp14_diag.py --out "$RAW/pp14-diag"
echo PP14_DONE
