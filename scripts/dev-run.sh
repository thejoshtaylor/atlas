#!/usr/bin/env bash
# Start the Phase 1 dev harness with the real environment.
#
# Secrets live in the gitignored `.env` and are loaded straight into this
# process, so they reach the application without passing through a terminal,
# a log, or an agent transcript. Nothing here echoes a value.
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

missing=()
for v in XAI_API_KEY HA_URL HA_TOKEN; do
  [ -n "${!v:-}" ] || missing+=("$v")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "unset in .env: ${missing[*]}" >&2
  exit 1
fi

export PYTHONPATH=src:mcp
export SPIRE_CONFIG="${SPIRE_CONFIG:-config/config.example.yaml}"
exec .venv/bin/python -m spire_voice.app
