#!/usr/bin/env bash
# RC4': linear-session benchmark. Arm 1 = 4-slot radix profile (RC1 launcher),
# arm 2 = frozen accepted control (no radix). The control runs last and is
# left up as the default. Each arm writes under $OUT/<arm>/.
set -uo pipefail

REPO=/root/qwen3.8-flash-next-24gb-sglang
OUT="${OUT:-$REPO/docs/logs/raw/rc4_3090_2026-09-16}"
TRACE_TURNS="${TRACE_TURNS:-12}"
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

boot_lines() {
  grep -E 'Mamba Cache is allocated|Memory pool end|max_total_num_tokens=|Tree cache initialized' "$1" | cut -c1-220
}

# ---- arm 1: 4-slot radix profile ----
A="$OUT/profile_slots4"; mkdir -p "$A"
echo "=== arm profile_slots4 $(date -u +%FT%TZ) ==="
stop_server
SRVLOG="$A/server.log" CTL="$A/elastic.ctl" SGLANG_3090_MAMBA_SIZE=4 \
  SGLANG_3090_MAX_TOKENS=262144 SGLANG_3090_LAZY_TOKENS=262144 \
  /root/quant/serve-3090-rc1.sh > "$A/launcher.out" 2>&1
echo "launcher: $(tr '\n' ' ' < "$A/launcher.out")"
boot_lines "$A/server.log" | tee "$A/boot_lines.txt"
server_info | tee "$A/server_info.txt"
python3 tools/rc1_reuse_checks.py --out-dir "$A/checks" --checks tg_probe 2>&1 | tail -8
python3 tools/rc1_reuse_checks.py --out-dir "$A/checks" --checks growing_session --session-turns "$TRACE_TURNS" 2>&1 | tail -6
mv "$A/checks/growing_session.json" "$A/checks/trace12.json"; mv "$A/checks/growing_session.txt" "$A/checks/trace12.txt"
python3 tools/rc1_reuse_checks.py --out-dir "$A/checks" --checks two_sessions 2>&1 | tail -12
echo "errors_in_log $(grep -ciE 'OutOfMemory|out of memory|Traceback' "$A/server.log")"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee "$A/nvidia_smi_end.txt"
echo "=== arm profile_slots4 done $(date -u +%FT%TZ) ==="

# ---- arm 2: accepted control, left running ----
C="$OUT/control_nocache"; mkdir -p "$C"
echo "=== arm control_nocache $(date -u +%FT%TZ) ==="
stop_server
SRVLOG="$C/server.log" /root/quant/serve-3090.sh > "$C/launcher.out" 2>&1
echo "launcher: $(tr '\n' ' ' < "$C/launcher.out")"
boot_lines "$C/server.log" | tee "$C/boot_lines.txt"
server_info | tee "$C/server_info.txt"
python3 tools/rc1_reuse_checks.py --out-dir "$C/checks" --checks tg_probe --tg-cold 2>&1 | tail -8
python3 tools/rc1_reuse_checks.py --out-dir "$C/checks" --checks growing_session --session-turns "$TRACE_TURNS" 2>&1 | tail -6
mv "$C/checks/growing_session.json" "$C/checks/trace12.json"; mv "$C/checks/growing_session.txt" "$C/checks/trace12.txt"
echo "errors_in_log $(grep -ciE 'OutOfMemory|out of memory|Traceback' "$C/server.log")"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee "$C/nvidia_smi_end.txt"
server_info | tee "$OUT/final_server_info.txt"
echo "control left running (accepted default)"
echo "ALL DONE $(date -u +%FT%TZ)"
