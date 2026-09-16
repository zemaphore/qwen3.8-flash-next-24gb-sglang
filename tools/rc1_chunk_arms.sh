#!/usr/bin/env bash
# RC1 chunk-size arms: boot the RC1 launcher with 16 mamba slots and the full
# 262,144 pool at each prefill chunk size, run the reuse checks, then restore
# the frozen accepted control. Each arm writes under $OUT/c<chunk>/.
set -uo pipefail

REPO=/root/qwen3.8-flash-next-24gb-sglang
OUT="${OUT:-$REPO/docs/logs/raw/rc1_3090_2026-09-16/chunk_arms}"
ARMS="${ARMS:-2048 3072}"
SLOTS="${SLOTS:-16}"
POOL="${POOL:-262144}"
CHECKS="${CHECKS:-mamba_capacity,churn,evict_rehit}"
mkdir -p "$OUT"
cd "$REPO"

stop_server() {
  local u
  u=$(systemctl --user list-units --type=scope --no-legend | awk '/sglang-/{print $1}')
  [ -n "$u" ] && systemctl --user stop "$u"
  for _ in $(seq 1 45); do pgrep -f 'python3 -m sglang.launch_server' >/dev/null || break; sleep 2; done
  sleep 3
  echo "vram_used_after_stop $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
}

server_info() {
  curl -s -m 10 http://127.0.0.1:30001/get_server_info | python3 -c \
    'import sys,json;d=json.load(sys.stdin);print({k:d[k] for k in ["disable_radix_cache","max_total_tokens","max_mamba_cache_size","chunked_prefill_size"]})'
}

for C in $ARMS; do
  A="$OUT/c$C"; mkdir -p "$A"
  echo "=== arm chunk=$C slots=$SLOTS pool=$POOL $(date -u +%FT%TZ) ==="
  stop_server
  SRVLOG="$A/server.log" CTL="$A/elastic.ctl" \
    SGLANG_3090_CHUNKED_PREFILL_SIZE=$C SGLANG_3090_MAMBA_SIZE=$SLOTS \
    SGLANG_3090_MAX_TOKENS=$POOL SGLANG_3090_LAZY_TOKENS=$POOL \
    /root/quant/serve-3090-rc1.sh > "$A/launcher.out" 2>&1
  echo "launcher: $(tr '\n' ' ' < "$A/launcher.out")"
  grep -E 'Mamba Cache is allocated|Memory pool end|max_total_num_tokens=|Tree cache initialized' "$A/server.log" | cut -c1-220 | tee "$A/boot_lines.txt"
  server_info | tee "$A/server_info.txt"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee "$A/nvidia_smi_after_boot.txt"
  python3 - "$A" <<'EOF'
import json, sys, time
sys.path.insert(0, "/root/qwen3.8-flash-next-24gb-sglang/tools")
import rc1_cache_probe as p
out = sys.argv[1]
rows = []
for n in (1, 3, 5):
    text = "".join(p.repo_block(f"long{n}-{i}-{int(time.time())}") for i in range(n))
    r = p.call("http://127.0.0.1:30001/generate", text, 16)
    rows.append({"blocks": n, "prompt_tokens": r["prompt_tokens"], "cached": r["cached_tokens"], "wall_s": r["wall_s"],
                 "prefill_tok_s": round(r["prompt_tokens"] / r["wall_s"], 1)})
    print("long cold", rows[-1])
json.dump(rows, open(f"{out}/long_cold.json", "w"), indent=2)
EOF
  python3 tools/rc1_reuse_checks.py --out-dir "$A/reuse_checks" --pool-tokens "$POOL" \
    --capacity-max-k $((SLOTS + 2)) --session-turns "${SESSION_TURNS:-40}" --checks "$CHECKS" ${EXTRA_ARGS:-} 2>&1 | tail -80
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee "$A/nvidia_smi_after_checks.txt"
  echo "errors_in_log $(grep -ciE 'OutOfMemory|out of memory|Traceback' "$A/server.log")"
  echo "=== arm chunk=$C done $(date -u +%FT%TZ) ==="
done

echo "=== restoring control $(date -u +%FT%TZ) ==="
stop_server
/root/quant/serve-3090.sh > "$OUT/launcher-control-restore.out" 2>&1
echo "control: $(tr '\n' ' ' < "$OUT/launcher-control-restore.out")"
server_info | tee "$OUT/control_server_info.txt"
echo "ALL DONE $(date -u +%FT%TZ)"
