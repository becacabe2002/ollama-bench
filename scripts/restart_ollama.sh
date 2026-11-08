#!/bin/bash
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-60}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-0.5}"
SETTLE_S="${SETTLE_S:-2}"

log() { printf '[restart_ollama] %s\n' "$*" >&2; }

log "restarting ollama.service ..."
sudo -n systemctl restart ollama


log "waiting for ${OLLAMA_URL}/api/tags (timeout ${READY_TIMEOUT_S}s) ..."
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
while :; do
  code="$(curl -s -o /dev/null -w '%{http_code}' \
            --max-time 2 \
            "${OLLAMA_URL}/api/tags" || true)"
  if [[ "$code" == "200" ]]; then
    break
  fi
  if (( $(date +%s) >= deadline )); then
    log "ERROR: ollama did not become ready within ${READY_TIMEOUT_S}s (last HTTP ${code:-none})"
    log "       recent journal entries:"
    sudo -n journalctl -u ollama -n 20 --no-pager >&2 || true
    exit 1
  fi
  sleep "${POLL_INTERVAL_S}"
done

sleep "${SETTLE_S}"

log "ready."