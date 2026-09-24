#!/usr/bin/env bash
# Source .env, then hand off to capture_wake_corpus.py -- credential
# handling stays here, the one place this project puts it (dev-run.sh,
# dev-measure.sh). capture_wake_corpus.py itself reads no environment
# variable directly; it reaches the camera only through
# atlas.config.load_config, which expands ${TAPO_USER}/${TAPO_PASSWORD}
# from what this script just sourced.
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

exec .venv/bin/python scripts/capture_wake_corpus.py --config "$ATLAS_CONFIG" "$@"
