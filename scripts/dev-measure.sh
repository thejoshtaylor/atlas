#!/usr/bin/env bash
# Boot the application for real, wait for it to say so, then drive
# scripts/measure_turns.py against it -- the one command plan 01.1-08 runs.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "no .env — copy .env.example to .env and fill it in" >&2
  exit 1
fi

# Delegate the readiness check to dev-preflight.sh rather than re-deriving
# it here: its output already confirms XAI_API_KEY/HA_URL/HA_TOKEN are
# non-blank AND that xAI and Home Assistant actually accept them, which
# "non-blank" alone cannot tell you. It never echoes a value -- see its own
# header comment -- and neither does anything below.
PREFLIGHT_OUTPUT="$(scripts/dev-preflight.sh)"
echo "$PREFLIGHT_OUTPUT"
if echo "$PREFLIGHT_OUTPUT" | grep -q ': BLANK$'; then
  echo "unset in .env — see dev-preflight output above" >&2
  exit 1
fi
if echo "$PREFLIGHT_OUTPUT" | grep -qE '(xai-auth|ha-auth): (REJECTED|UNREACHABLE)'; then
  echo "a credential was rejected, or a service was unreachable — see dev-preflight output above" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
. ./.env
set +a

TRANSPORT="${1:-both}"
TURNS="${2:-5}"

if [ "$TRANSPORT" != "websocket" ] && [ "$TRANSPORT" != "webrtc" ] && [ "$TRANSPORT" != "both" ]; then
  echo "unknown transport: $TRANSPORT (expected websocket, webrtc, or both)" >&2
  exit 1
fi

export PYTHONPATH=src:mcp
export SPIRE_CONFIG="${SPIRE_CONFIG:-config/config.example.yaml}"

LOG_FILE="$(mktemp -t spire-voice-dev-measure.XXXXXX)"
echo "application output: $LOG_FILE"

.venv/bin/python -m spire_voice.app >"$LOG_FILE" 2>&1 &
APP_PID=$!

# Kill the backgrounded application on every exit path this script takes —
# a clean finish, a failed startup wait, or an interrupted run — so a
# Ctrl-C here never leaves a process holding the configured bind port for
# the next attempt.
trap 'kill "$APP_PID" 2>/dev/null; wait "$APP_PID" 2>/dev/null' EXIT

# This wait is also the first thing in this project that boots the real
# application and asserts it came up, rather than importing a handler
# function directly the way every other test in this repository does.
# Phase 01's code review found three Critical defects on startup paths no
# test exercised, for exactly that reason. `lifespan` now also resolves
# every brain tier's model and synthesizes and caches every holding phrase
# and macro reply, so a cold first start legitimately takes a while — this
# loop is a check, not a sleep, and its timeout is sized generously on
# purpose so a slow-but-honest start is not mistaken for a hang.
STARTUP_TIMEOUT_S=60
waited=0
while [ "$waited" -lt "$STARTUP_TIMEOUT_S" ]; do
  if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "application exited before completing startup — last output:" >&2
    tail -n 40 "$LOG_FILE" >&2
    exit 1
  fi
  sleep 1
  waited=$((waited + 1))
done

if ! grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then
  echo "application did not report startup complete within ${STARTUP_TIMEOUT_S}s — last output:" >&2
  tail -n 40 "$LOG_FILE" >&2
  exit 1
fi

echo "application is up (pid $APP_PID) — running the measurement"

STATUS=0
if [ "$TRANSPORT" = "both" ]; then
  for one_transport in websocket webrtc; do
    echo "--- transport: $one_transport ---"
    .venv/bin/python scripts/measure_turns.py --transport "$one_transport" --turns "$TURNS" || STATUS=1
  done
else
  .venv/bin/python scripts/measure_turns.py --transport "$TRANSPORT" --turns "$TURNS" || STATUS=1
fi

echo "application output was captured at: $LOG_FILE"
exit "$STATUS"
