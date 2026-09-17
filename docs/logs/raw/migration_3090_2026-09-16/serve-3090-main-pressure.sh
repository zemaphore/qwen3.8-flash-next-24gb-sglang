#!/usr/bin/env bash
# Candidate pressure launcher for step 5d: the main-based port served with the
# RC1 pressure profile (16 mamba slots, chunk 2048, 262,144 pool, R1/R2/R3 on,
# floor 64). Same shape as serve-3090-rc1.sh but pointed at the main tree/venv,
# on port 30011, with prefill graphs disabled (main no longer auto-disables).
set -euo pipefail

SGLANG="${SGLANG:-/root/quant/sglang-main}"
VENV="${VENV:-/root/quant/venv-sglang-main}"
MODEL="${MODEL:-/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang}"
PLE="${PLE:-$MODEL/ple}"
REPO="${REPO:-/root/qwen3.8-flash-next-24gb-sglang}"
ASSETS="${ASSETS:-$REPO/assets}"
CTL="${CTL:-/root/quant/elastic-main-pressure.ctl}"
SRVLOG="${SRVLOG:-/root/quant/logs/server-main-pressure.log}"
CACHE="${CACHE:-/root/quant/cache-main}"
PORT="${PORT:-30011}"

CU="$VENV/lib/python3.12/site-packages/nvidia/cu13"
[ -f "$CTL" ] || echo "S 184" > "$CTL"
mkdir -p "$(dirname "$SRVLOG")" "$CACHE"

cd "$SGLANG"

setsid systemd-run --user --scope --unit="sglang-main-p-$(date +%s)" -p MemoryMax=56G \
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
      SGLANG_KV_LAZY_TOKENS="${SGLANG_3090_LAZY_TOKENS:-262144}" \
      SGLANG_KV_LAZY_SAFETY=0.77 \
      SGLANG_KV_TIERS_W=8192 \
      SGLANG_KV_EVICT_ON_PRESSURE="${SGLANG_KV_EVICT_ON_PRESSURE:-1}" \
      SGLANG_KV_LAZY_STRICT_HEADROOM="${SGLANG_KV_LAZY_STRICT_HEADROOM:-1}" \
      SGLANG_KV_LAZY_MIN_FREE_MB="${SGLANG_KV_LAZY_MIN_FREE_MB:-64}" \
      SGLANG_PREFILL_ALLOC_ABORT="${SGLANG_PREFILL_ALLOC_ABORT:-1}" \
  "$VENV/bin/python3" -m sglang.launch_server \
    --host "${HOST:-0.0.0.0}" --port "$PORT" --tp-size 1 --cpu-offload-gb 19 --no-ple-offload-embedding \
    --mem-fraction-static 0.95 --language-model-only --page-size 1 --disable-overlap-schedule --sleep-on-idle \
    --weight-loader-drop-cache-after-load \
    --chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-2048}" --max-prefill-tokens 32768 --cuda-graph-backend-decode breakable \
    --cuda-graph-backend-prefill disabled \
    --served-model-name "${SERVED_MODEL_NAME:-qwen38-flash-256K-rc1}" \
    --model-path "$MODEL" --max-total-tokens "${SGLANG_3090_MAX_TOKENS:-262144}" --context-length "${SGLANG_3090_CONTEXT:-262144}" \
    --kv-cache-dtype int8ring_int4 --attention-backend triton --max-mamba-cache-size "${SGLANG_3090_MAMBA_SIZE:-16}" \
    --mamba-radix-cache-strategy "${SGLANG_3090_MAMBA_STRATEGY:-extra_buffer}" \
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
