#!/usr/bin/env bash
# Candidate (main-based port) launcher for the TG0/TG1 profile: the TG0 control
# configuration = 256K no-radix accepted profile, ported to the main tree.
# Matches docs/logs/raw/promotion_3090_2026-09-16/serve-3090-nocache-256K.sh
# (SHA-256 11013fb6...), with main paths and prefill graphs explicitly disabled.
set -euo pipefail

SGLANG="${SGLANG:-/root/quant/sglang-main}"
VENV="${VENV:-/root/quant/venv-sglang-main}"
MODEL="${MODEL:-/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang}"
PLE="${PLE:-$MODEL/ple}"
REPO="${REPO:-/root/qwen3.8-flash-next-24gb-sglang}"
ASSETS="${ASSETS:-$REPO/assets}"
CTL="${CTL:-/root/quant/tg-main.ctl}"
SRVLOG="${SRVLOG:-/root/quant/logs/server-tg-main.log}"
CACHE="${CACHE:-/root/quant/cache-main}"
PORT="${PORT:-30011}"

CU="$VENV/lib/python3.12/site-packages/nvidia/cu13"
[ -f "$CTL" ] || echo "S 184" > "$CTL"
mkdir -p "$(dirname "$SRVLOG")" "$CACHE"

cd "$SGLANG"

setsid systemd-run --user --scope --unit="sglang-tg-$(date +%s)" -p MemoryMax=56G \
  env PATH="$CU/bin:$VENV/bin:$PATH" CUDA_HOME="$CU" SGLANG_CACHE_DIR="$CACHE" \
      SGLANG_QWEN4_PLE_MMAP="$PLE" \
      SGLANG_QWEN4_PLE_BULK_PREAD="${SGLANG_QWEN4_PLE_BULK_PREAD:-1}" \
      SGLANG_QWEN4_PLE_BULK_PREAD_MIN_UNIQUE="${SGLANG_QWEN4_PLE_BULK_PREAD_MIN_UNIQUE:-2048}" \
      SGLANG_QWEN4_PLE_WORKERS="${SGLANG_QWEN4_PLE_WORKERS:-16}" \
      SGLANG_QWEN4_PLE_RECENT_CACHE_MB="${SGLANG_QWEN4_PLE_RECENT_CACHE_MB:-128}" \
      SGLANG_QWEN4_PLE_PROFILE="${SGLANG_QWEN4_PLE_PROFILE:-0}" \
      SGLANG_VLM_CACHE_SIZE_MB=0 \
      SGLANG_MOE_EXPERT_STREAM=1 \
      SGLANG_MOE_GATHER_BLOCK="${SGLANG_MOE_GATHER_BLOCK:-2048}" \
      SGLANG_MOE_GATHER_DMA="${SGLANG_MOE_GATHER_DMA:-1}" \
      SGLANG_MOE_GATHER_DMA_BATCH="${SGLANG_MOE_GATHER_DMA_BATCH:-1}" \
      SGLANG_MOE_COLD_PREFETCH="${SGLANG_MOE_COLD_PREFETCH:-1}" \
      SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS="${SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS:-2048}" \
      SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS="${SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS:-0}" \
      SGLANG_MOE_CONFIG_NEAREST_E="${SGLANG_MOE_CONFIG_NEAREST_E:-1}" \
      SGLANG_MOE_CONFIG_DIR="$ASSETS/moe_configs" \
      SGLANG_MOE_PLACEMENT="${SGLANG_MOE_PLACEMENT:-$ASSETS/expert_presence_code.pt}" \
      SGLANG_MOE_PLACEMENT_S=184 \
      SGLANG_MOE_ELASTIC=1 \
      SGLANG_MOE_ELASTIC_PIN_MB=512 \
      SGLANG_MOE_ELASTIC_CTL="$CTL" \
      SGLANG_KV_LAZY=1 \
      SGLANG_MOE_ELASTIC_FILL_MB="${SGLANG_MOE_ELASTIC_FILL_MB:-99999}" \
      SGLANG_MOE_ELASTIC_RESERVE_ROWS=0 \
      SGLANG_KV_LAZY_TOKENS=262144 \
      SGLANG_KV_LAZY_SAFETY=0.77 \
      SGLANG_KV_TIERS_W=8192 \
      SGLANG_KV_EVICT_ON_PRESSURE="${SGLANG_KV_EVICT_ON_PRESSURE:-0}" \
      SGLANG_KV_LAZY_STRICT_HEADROOM="${SGLANG_KV_LAZY_STRICT_HEADROOM:-0}" \
      SGLANG_PREFILL_ALLOC_ABORT="${SGLANG_PREFILL_ALLOC_ABORT:-0}" \
  "$VENV/bin/python3" -m sglang.launch_server \
    --host "${HOST:-0.0.0.0}" --port "$PORT" --tp-size 1 --cpu-offload-gb 19 --no-ple-offload-embedding \
    --mem-fraction-static 0.95 --language-model-only --page-size 1 --disable-overlap-schedule --sleep-on-idle \
    --disable-radix-cache --weight-loader-drop-cache-after-load \
    --chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-4608}" --max-prefill-tokens 32768 --cuda-graph-backend-decode breakable \
    --cuda-graph-backend-prefill disabled \
    --served-model-name "${SERVED_MODEL_NAME:-qwen38-flash-256K}" \
    --model-path "$MODEL" --max-total-tokens 262144 --context-length 262144 \
    --kv-cache-dtype int8ring_int4 --attention-backend triton --max-mamba-cache-size 1 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --max-running-requests 1 \
  > "$SRVLOG" 2>&1 < /dev/null &

T0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  sleep 10
  if [ $(( $(date +%s) - T0 )) -gt 1800 ]; then echo "TIMEOUT (see $SRVLOG)"; exit 1; fi
done
echo "up after $(( $(date +%s) - T0 )) s"
sleep 5
curl -s -m 120 "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
  -d '{"text":"Warmup.","sampling_params":{"max_new_tokens":4,"temperature":0}}' >/dev/null
curl -s -m 30 -X POST "http://127.0.0.1:$PORT/freeze_gc" >/dev/null && echo "gc frozen"
