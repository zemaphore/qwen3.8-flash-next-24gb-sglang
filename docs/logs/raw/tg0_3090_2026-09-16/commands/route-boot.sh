#!/usr/bin/env bash
set -euo pipefail
cd /root/qwen3.8-flash-next-24gb-sglang
RAW="$PWD/docs/logs/raw/tg0_3090_2026-09-16"
PY=/root/quant/venv-sglang/bin/python
$PY tools/tg_server.py stop
SGLANG=/root/sglang $PY patches/decode_route_dump.py apply
$PY tools/tg_server.py start --log "$RAW/server-route.log" --ctl "$RAW/elastic-route.ctl" \
  --launch-out "$RAW/route.launch.out" --env "SGLANG_DECODE_ROUTE_DUMP=$RAW/route_dump"
$PY tools/snapshot_server.py "$RAW/manifest/server-route.json"
$PY tools/tg_route_collect.py --prompts-json "$RAW/prompts/prompts.json" \
  --prompts-class reasoning --context 2048 --decode 32
$PY tools/tg_route_diag.py --dump "$RAW/route_dump" \
  --placement assets/expert_presence_code.pt --out "$RAW/route_diag.json"
$PY tools/tg_server.py stop
SGLANG=/root/sglang $PY patches/decode_route_dump.py revert
SGLANG=/root/sglang $PY patches/decode_route_dump.py --check
echo ROUTE_DONE
