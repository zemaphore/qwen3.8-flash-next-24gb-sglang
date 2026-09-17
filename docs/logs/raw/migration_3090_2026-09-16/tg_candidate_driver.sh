#!/usr/bin/env bash
# Unattended TG0-redo + TG1 driver for the main-based candidate (port 30011).
#
# Profile: the TG0 control configuration (256K no-radix accepted profile,
# serve-3090-nocache-256K.sh) ported to /root/quant/sglang-main.
#
# Sequence:
#   TG0 boot1   presence placement, full 9-cell matrix (1 warmup + 5), capacity smoke
#   TG0 boot2   presence placement, short 3-cell matrix (1 warmup + 3)
#   TG1 pres b1 presence placement, full matrix
#   TG1 mass b1 mass placement (expert_freq.pt), full matrix
#   TG1 mass b2 mass placement, short matrix
#   TG1 pres b2 presence placement, short matrix
#
# Robustness: every boot retries once; a failed boot or a failed matrix is
# logged and the driver moves on. Progress is appended to driver.log and the
# current step to state.txt. No interactive input.
set -uo pipefail

REPO=/root/qwen3.8-flash-next-24gb-sglang
OUT="$REPO/docs/logs/raw/migration_3090_2026-09-16/tg_candidate"
PROMPTS="$REPO/docs/logs/raw/tg0_3090_2026-09-16/prompts/prompts.json"
LAUNCHER=/root/quant/serve-3090-main-nocache-256K.sh
PORT=30011
URL="http://127.0.0.1:$PORT/generate"
CTL=/root/quant/tg-main.ctl
SRVLOG=/root/quant/logs/server-tg-main.log
PRESENCE="$REPO/assets/expert_presence_code.pt"
MASS="$REPO/assets/expert_freq.pt"
LOG="$OUT/driver.log"
STATE="$OUT/state.txt"
mkdir -p "$OUT"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
state() { echo "$*" > "$STATE"; }

server_pids() {
  ps -eo pid,cmd | awk -v port="$PORT" '/[s]glang\.launch_server/ && index($0, "--port " port) {print $1}'
}

stop_server() {
  local pids
  pids=$(server_pids)
  for p in $pids; do kill -TERM "$p" 2>/dev/null; done
  for _ in $(seq 1 60); do [ -z "$(server_pids)" ] && break; sleep 2; done
  for p in $(server_pids); do kill -KILL "$p" 2>/dev/null; done
  sleep 3
}

wait_health() {
  local i
  for i in $(seq 1 200); do
    if curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      sleep 5; return 0
    fi
    if [ -z "$(server_pids)" ]; then return 1; fi
    sleep 10
  done
  return 1
}

boot() {
  local label="$1" placement="$2"
  local attempt
  for attempt in 1 2; do
    log "boot $label (attempt $attempt) placement=$(basename "$placement")"
    stop_server
    : > "$SRVLOG"
    CTL="$CTL" SRVLOG="$SRVLOG" SGLANG_MOE_PLACEMENT="$placement" \
      setsid nohup "$LAUNCHER" > "$OUT/$label.launch.out" 2>&1 < /dev/null &
    disown
    if wait_health; then
      cp "$SRVLOG" "$OUT/$label.server.log" 2>/dev/null
      grep -aE 'max_total_num_tokens=|KV lazy backing: token capacity|Mamba Cache is allocated' "$SRVLOG" \
        | sed 's/^\[[0-9-]* [0-9:]*\] //' | sort -u > "$OUT/$label.effective.txt"
      log "boot $label healthy"
      return 0
    fi
    log "boot $label FAILED (attempt $attempt)"
    cp "$SRVLOG" "$OUT/$label.FAILED.server.log" 2>/dev/null
  done
  return 1
}

run_tg() {
  local label="$1" matrix="$2" measured="$3" capacity="$4"
  local caparg=""
  [ "$capacity" = "1" ] && caparg="--capacity"
  state "run_tg $label matrix=$matrix measured=$measured capacity=$capacity"
  SGLANG_MOE_ELASTIC_CTL="$CTL" TG0_SERVER_LOG="$SRVLOG" \
    python3 "$REPO/tools/tg_baseline.py" \
      --prompts-json "$PROMPTS" --out-dir "$OUT/$label" --boot-label "$label" \
      --matrix "$matrix" --measured "$measured" --url "$URL" $caparg 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  cp "$SRVLOG" "$OUT/$label.server.log" 2>/dev/null
  log "run_tg $label exit=$rc"
  return $rc
}

log "driver start; candidate profile = TG0 control (256K no-radix), port $PORT"
[ -f "$PROMPTS" ] || { log "FATAL: prompts json missing"; exit 1; }
sha256sum "$PROMPTS" | tee -a "$LOG"
stop_server

# ---- TG0 boot 1: full matrix + capacity smoke (presence) ----
if boot tg0_boot1 "$PRESENCE"; then
  run_tg tg0_boot1 full 5 1 || log "WARN: tg0_boot1 matrix failed"
else
  log "WARN: tg0_boot1 boot failed; skipping matrix"
fi

# ---- TG0 boot 2: short matrix (presence) ----
if boot tg0_boot2 "$PRESENCE"; then
  run_tg tg0_boot2 short 3 0 || log "WARN: tg0_boot2 matrix failed"
else
  log "WARN: tg0_boot2 boot failed; skipping matrix"
fi

# ---- TG1 bracketed: presence, mass | mass, presence ----
if boot tg1_pres_b1 "$PRESENCE"; then
  run_tg tg1_pres_b1 full 5 0 || log "WARN: tg1_pres_b1 matrix failed"
else
  log "WARN: tg1_pres_b1 boot failed; skipping matrix"
fi

if boot tg1_mass_b1 "$MASS"; then
  run_tg tg1_mass_b1 full 5 0 || log "WARN: tg1_mass_b1 matrix failed"
else
  log "WARN: tg1_mass_b1 boot failed; skipping matrix"
fi

if boot tg1_mass_b2 "$MASS"; then
  run_tg tg1_mass_b2 short 3 0 || log "WARN: tg1_mass_b2 matrix failed"
else
  log "WARN: tg1_mass_b2 boot failed; skipping matrix"
fi

if boot tg1_pres_b2 "$PRESENCE"; then
  run_tg tg1_pres_b2 short 3 0 || log "WARN: tg1_pres_b2 matrix failed"
else
  log "WARN: tg1_pres_b2 boot failed; skipping matrix"
fi

stop_server
state "done"
log "driver complete; server stopped"
