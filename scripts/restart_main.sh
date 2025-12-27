#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8081}"
PID_FILE="${PID_FILE:-server.pid}"
LOG_FILE="${LOG_FILE:-server.log}"

# Stop existing PID if present
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_args="$(ps -p "$old_pid" -o args= 2>/dev/null || true)"
    if echo "$old_args" | grep -qE "((^|[[:space:]])(${ROOT_DIR}/)?app_main\.py([[:space:]]|$))|(uvicorn[[:space:]]+app_main:app)"; then
      echo "Stopping PID=$old_pid"
      kill "$old_pid" || true
    else
      echo "Warning: $PID_FILE points to PID=$old_pid but it does not look like app_main.py; skipping stop." >&2
      old_pid=""
    fi
    if [[ -n "${old_pid:-}" ]]; then
      # give it a moment
      for _ in {1..20}; do
        if kill -0 "$old_pid" 2>/dev/null; then
          sleep 0.1
        else
          break
        fi
      done
      if kill -0 "$old_pid" 2>/dev/null; then
        echo "PID $old_pid did not exit; sending SIGKILL"
        kill -9 "$old_pid" 2>/dev/null || true
      fi
    fi
  fi
fi

# Also stop any stray instances of this repo's app_main.py (e.g., if a previous
# start didn't write PID_FILE correctly).
strays="$(pgrep -f "${ROOT_DIR}/app_main\.py([[:space:]]|$)" 2>/dev/null || true)"
if [[ -n "${strays:-}" ]]; then
  echo "Stopping stray app_main.py PIDs: ${strays//$'\n'/ }"
  while read -r pid; do
    [[ -z "${pid:-}" ]] && continue
    kill "$pid" 2>/dev/null || true
  done <<< "$strays"
  # Give them a moment, then hard-kill survivors
  sleep 0.3
  while read -r pid; do
    [[ -z "${pid:-}" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<< "$strays"
fi

# Start
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

echo "Starting app_main.py (PORT=$PORT)"
nohup "$PY" app_main.py >"$LOG_FILE" 2>&1 &

# Wait for LISTEN on port
listener_pid=""
for _ in {1..80}; do
  listener_pid="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -n1 || true)"
  if [[ -n "$listener_pid" ]]; then
    break
  fi
  sleep 0.1
done

if [[ -z "$listener_pid" ]]; then
  echo "ERROR: app did not listen on :$PORT within timeout."
  echo "Tail of $LOG_FILE:"
  tail -n 50 "$LOG_FILE" || true
  exit 1
fi

echo "$listener_pid" > "$PID_FILE"
echo "Started. $PID_FILE=$listener_pid"
