#!/usr/bin/env bash
# Qwen3.8-Flash-Next-int2 serving on RTX 3090 (sm_86), adapted from the author's scripts/serve.sh.
# Differences from the author's script:
#   - paths point at this box (SGLang /root/sglang, venv /root/quant/venv-sglang, model on /mnt/ai_models)
#   - MemoryMax raised to 56G (host has 62G; author was bound by 32G and used 30G)
#   - CUDA toolkit is the venv's cu13 (nvcc 13.3.73); SGLang JIT-compiles for compute_86 here
#   - M4's RTX 3090 configs are selected by nearest compact expert-count bucket during prefill
set -euo pipefail

SGLANG="${SGLANG:-/root/sglang}"
VENV="${VENV:-/root/quant/venv-sglang}"
MODEL="${MODEL:-/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang}"
PLE="${PLE:-$MODEL/ple}"
REPO="${REPO:-/root/qwen3.8-flash-next-24gb-sglang}"
ASSETS="${ASSETS:-$REPO/assets}"
CTL="${CTL:-/root/quant/elastic.ctl}"
SRVLOG="${SRVLOG:-/root/quant/logs/server.log}"

CU="$VENV/lib/python3.12/site-packages/nvidia/cu13"
[ -f "$CTL" ] || echo "S 184" > "$CTL"
mkdir -p "$(dirname "$SRVLOG")"

cd "$SGLANG"

setsid systemd-run --user --scope --unit="sglang-$(date +%s)" -p MemoryMax=56G \
  env PATH="$CU/bin:$VENV/bin:$PATH" CUDA_HOME="$CU" \
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
      SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS="${SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS:-0}" \
      SGLANG_MOE_CONFIG_NEAREST_E="${SGLANG_MOE_CONFIG_NEAREST_E:-1}" \
      SGLANG_MOE_CONFIG_DIR="$ASSETS/moe_configs" \
      SGLANG_MOE_PLACEMENT="${SGLANG_MOE_PLACEMENT:-$ASSETS/expert_presence_code.pt}" \
      SGLANG_MOE_PLACEMENT_S=184 \
      SGLANG_MOE_ELASTIC=1 \
      SGLANG_PREFILL_ROUTE_DUMP="${SGLANG_PREFILL_ROUTE_DUMP:-}" \
      SGLANG_MOE_ELASTIC_PIN_MB=512 \
      SGLANG_MOE_ELASTIC_CTL="$CTL" \
      SGLANG_KV_LAZY=1 \
      SGLANG_MOE_ELASTIC_FILL_MB="${SGLANG_MOE_ELASTIC_FILL_MB:-99999}" \
      SGLANG_MOE_ELASTIC_RESERVE_ROWS=0 \
      SGLANG_KV_LAZY_TOKENS=262144 \
      SGLANG_KV_LAZY_SAFETY=0.77 \
      SGLANG_KV_TIERS_W=8192 \
  "$VENV/bin/python3" -m sglang.launch_server \
    --host "${HOST:-0.0.0.0}" --port 30001 --tp-size 1 --cpu-offload-gb 19 --no-ple-offload-embedding \
    --mem-fraction-static 0.95 --language-model-only --page-size 1 --disable-overlap-schedule --sleep-on-idle \
    --disable-radix-cache --weight-loader-drop-cache-after-load \
    --chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-4608}" --max-prefill-tokens 32768 --cuda-graph-backend-decode breakable \
    --model-path "$MODEL" --max-total-tokens 262144 --context-length 262144 \
    --kv-cache-dtype int8ring_int4 --attention-backend triton --max-mamba-cache-size 1 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --max-running-requests 1 \
  > "$SRVLOG" 2>&1 < /dev/null &

T0=$(date +%s)
until curl -sf http://127.0.0.1:30001/health >/dev/null 2>&1; do
  sleep 10
  if [ $(( $(date +%s) - T0 )) -gt 1800 ]; then echo "TIMEOUT (see $SRVLOG)"; exit 1; fi
done
echo "up after $(( $(date +%s) - T0 )) s"
sleep 5
curl -s -m 120 http://127.0.0.1:30001/generate -H 'Content-Type: application/json' \
  -d '{"text":"Warmup.","sampling_params":{"max_new_tokens":4,"temperature":0}}' >/dev/null
curl -s -m 30 -X POST http://127.0.0.1:30001/freeze_gc >/dev/null && echo "gc frozen"
