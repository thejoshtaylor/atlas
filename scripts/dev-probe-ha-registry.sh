#!/usr/bin/env bash
# Source .env, then hand off to probe_ha_registry.py -- credential handling
# stays here, the one place this project puts it (dev-run.sh,
# dev-calibrate-echo.sh). probe_ha_registry.py reads HA_URL/HA_TOKEN
# directly from the environment, the same way mcp/spire_mcp/ha.py's own
# _startup() does: this is a probe of the exact connection the MCP child
# makes, not a second way of reaching Home Assistant.
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

export PYTHONPATH=mcp
exec .venv/bin/python scripts/probe_ha_registry.py "$@"
