#!/usr/bin/env bash
# Source .env, then hand off to fetch_models.py -- the app's config file
# expands every ${VAR} it names in one pass (config.py::expand_env), so
# every variable the whole file needs must be set even though this script
# only reads stt.*/tts.* out of the result. Matches dev-capture-corpus.sh's
# and dev-run.sh's own convention: credential handling stays here, never
# inside the Python script it hands off to.
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

export PYTHONPATH=src:mcp
export ATLAS_CONFIG="${ATLAS_CONFIG:-config/config.example.yaml}"

exec .venv/bin/python scripts/fetch_models.py --config "$ATLAS_CONFIG" "$@"
