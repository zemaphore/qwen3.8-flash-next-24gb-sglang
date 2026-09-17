#!/usr/bin/env bash
# TG1 on the promoted main-based profile (port 30001): presence vs pooled-mass
# placement at fixed S184, bracketed across boots. Uses the promoted launcher
# /root/quant/serve-3090-main.sh with per-boot SRVLOG/CTL overrides; a final
# canonical boot restores the promoted default with the standard paths.
#
# TG1 pres b1 (full) -> mass b1 (full) -> mass b2 (short) -> pres b2 (short)
set -uo pipefail

REPO=/root/qwen3.8-flash-next-24gb-sglang
OUT="$REPO/docs/logs/raw/migration_3090_2026-09-16/tg_promoted"
PROMPTS="$REPO/docs/logs/raw/tg0_3090_2026-09-16/prompts/prompts.json"
LAUNCHER=/root/quant/serve-3090-main.sh
PORT=30001
URL="http://127.0.0.1:$PORT/generate"
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
    [ -z "$(server_pids)" ] && return 1
    sleep 10
  done
  return 1
}

boot() {
  local label="$1" placement="$2" srvlog="$3" ctl="$4"
  local attempt
  for attempt in 1 2; do
    log "boot $label (attempt $attempt) placement=$(basename "$placement")"
    stop_server
    : > "$srvlog"
    CTL="$ctl" SRVLOG="$srvlog" SGLANG_MOE_PLACEMENT="$placement" \
      setsid nohup "$LAUNCHER" > "$OUT/$label.launch.out" 2>&1 < /dev/null &
    disown
    if wait_health; then
      grep -aE 'max_total_num_tokens=|KV lazy backing: token capacity|KV tiers:' "$srvlog" \
        | sed 's/^\[[0-9-]* [0-9:]*\] //' | sort -u > "$OUT/$label.effective.txt"
      log "boot $label healthy"
      return 0
    fi
    log "boot $label FAILED (attempt $attempt)"
    cp "$srvlog" "$OUT/$label.FAILED.server.log" 2>/dev/null
  done
  return 1
}

run_tg() {
  local label="$1" matrix="$2" measured="$3" srvlog="$4" ctl="$5"
  state "run_tg $label matrix=$matrix measured=$measured"
  SGLANG_MOE_ELASTIC_CTL="$ctl" TG0_SERVER_LOG="$srvlog" \
    python3 "$REPO/tools/tg_baseline.py" \
      --prompts-json "$PROMPTS" --out-dir "$OUT/$label" --boot-label "$label" \
      --matrix "$matrix" --measured "$measured" --url "$URL" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  log "run_tg $label exit=$rc"
  return $rc
}

log "TG1 driver start on promoted profile (radix 230K), port $PORT"
sha256sum "$PROMPTS" | tee -a "$LOG"
stop_server

if boot tg1p_pres_b1 "$PRESENCE" "$OUT/tg1p_pres_b1.server.log" "$OUT/tg1p_pres_b1.ctl"; then
  run_tg tg1p_pres_b1 full 5 "$OUT/tg1p_pres_b1.server.log" "$OUT/tg1p_pres_b1.ctl" || log "WARN: tg1p_pres_b1 failed"
else
  log "WARN: tg1p_pres_b1 boot failed"
fi

if boot tg1p_mass_b1 "$MASS" "$OUT/tg1p_mass_b1.server.log" "$OUT/tg1p_mass_b1.ctl"; then
  run_tg tg1p_mass_b1 full 5 "$OUT/tg1p_mass_b1.server.log" "$OUT/tg1p_mass_b1.ctl" || log "WARN: tg1p_mass_b1 failed"
else
  log "WARN: tg1p_mass_b1 boot failed"
fi

if boot tg1p_mass_b2 "$MASS" "$OUT/tg1p_mass_b2.server.log" "$OUT/tg1p_mass_b2.ctl"; then
  run_tg tg1p_mass_b2 short 3 "$OUT/tg1p_mass_b2.server.log" "$OUT/tg1p_mass_b2.ctl" || log "WARN: tg1p_mass_b2 failed"
else
  log "WARN: tg1p_mass_b2 boot failed"
fi

if boot tg1p_pres_b2 "$PRESENCE" "$OUT/tg1p_pres_b2.server.log" "$OUT/tg1p_pres_b2.ctl"; then
  run_tg tg1p_pres_b2 short 3 "$OUT/tg1p_pres_b2.server.log" "$OUT/tg1p_pres_b2.ctl" || log "WARN: tg1p_pres_b2 failed"
else
  log "WARN: tg1p_pres_b2 boot failed"
fi

# Restore the canonical promoted default (standard CTL/SRVLOG) and leave it up.
log "restoring canonical promoted default"
stop_server
setsid nohup /root/quant/serve-3090.sh > "$OUT/restore.launch.out" 2>&1 < /dev/null &
disown
if wait_health; then
  log "canonical promoted default healthy on $PORT"
else
  log "WARN: canonical restore failed to become healthy"
fi
state "done"
log "TG1 driver complete"
