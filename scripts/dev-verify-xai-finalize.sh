#!/usr/bin/env bash
# Prove which finalize wire message the live xAI speech-to-text socket
# honours (10-05-PLAN.md Task 3, D-12).
#
# Secrets live in the gitignored `.env` and are loaded straight into this
# process, so they reach the application without passing through a
# terminal, a log, or an agent transcript -- the same pattern
# `scripts/dev-run.sh` already uses. Nothing here echoes a value.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "no .env — copy .env.example to .env and fill it in" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
. ./.env
set +a

if [ -z "${XAI_API_KEY:-}" ]; then
  echo "unset in .env: XAI_API_KEY" >&2
  exit 1
fi

# This script reads only config.stt/config.tts, but `atlas.config.
# load_config` expands every `${VAR}` placeholder in the whole config file
# before parsing it into sections -- so a camera/speaker section this
# script never touches still blocks loading if its own env var is unset.
# `.env.example`'s own documented "camera/speaker not in use" placeholders
# cover exactly that gap, applied only when the operator's `.env` leaves
# them unset (never overriding a real value).
: "${CAMERA_RTSP_URL:=rtsp://unused-in-phase-1:unused-in-phase-1@camera.invalid:554/stream1}"
: "${SPEAKER_BACKEND:=go2rtc}"
: "${CALIBRATION_ROUTE_ENABLED:=false}"
export CAMERA_RTSP_URL SPEAKER_BACKEND CALIBRATION_ROUTE_ENABLED

export PYTHONPATH=src:mcp
export ATLAS_CONFIG="${ATLAS_CONFIG:-config/config.example.yaml}"

exec .venv/bin/python scripts/verify_xai_finalize.py --config "$ATLAS_CONFIG"
