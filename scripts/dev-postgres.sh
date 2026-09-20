#!/usr/bin/env bash
# Start (or reuse) a throwaway local Postgres, for development and for the
# one Postgres-backed integration test this phase adds (D-04's "the suite
# runs without a Postgres... one integration test runs against a real
# Postgres and skips when none is reachable").
#
# Ephemeral on purpose: `--rm` and no bind-mounted data directory means a
# `docker stop` throws the data away with it. This is a dev convenience,
# not the deployment path -- the Helm chart and the Docker Compose file
# (Phase 7) own a real, persistent Postgres.
#
# Usage: eval "$(scripts/dev-postgres.sh)" -- exports both connection
# strings into the current shell. Anything this script prints to stderr is
# status, not part of that export.
set -euo pipefail

_CONTAINER=spire-dev-postgres
_USER=spire
_PASSWORD=spire
_DB=spire
# WR-09 (code review): not 5432. That is the port every other Postgres on
# a development machine is already bound to, and this script reusing it
# means `.env.example`'s own DATABASE_URL reaches an unrelated container
# and fails with an authentication error that reads like a bug in this
# project. `.env.example` names this same port.
_PORT="${SPIRE_DEV_POSTGRES_PORT:-54329}"
_IMAGE=postgres:18

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not on PATH -- install Docker to run a local Postgres" >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx "$_CONTAINER"; then
  echo "# $_CONTAINER is already running" >&2
else
  echo "# starting $_CONTAINER ($_IMAGE) on port $_PORT..." >&2
  docker run -d --rm \
    --name "$_CONTAINER" \
    -e POSTGRES_USER="$_USER" \
    -e POSTGRES_PASSWORD="$_PASSWORD" \
    -e POSTGRES_DB="$_DB" \
    -p "${_PORT}:5432" \
    "$_IMAGE" >/dev/null

  echo "# waiting for $_CONTAINER to accept connections..." >&2
  tries=0
  until docker exec "$_CONTAINER" pg_isready -U "$_USER" >/dev/null 2>&1; do
    tries=$((tries + 1))
    if [ "$tries" -ge 60 ]; then
      echo "$_CONTAINER did not become ready within 30s" >&2
      exit 1
    fi
    sleep 0.5
  done
fi

# The same URL serves both roles here: a throwaway container has no reason
# to carry separate dev/test databases. Both DatabaseConfig.url (runtime,
# asyncpg) and the integration test's own connection string share this
# shape -- only the variable name differs, matching how the two consumers
# read it (config.yaml's ${DATABASE_URL} vs. the test's own env lookup).
echo "export DATABASE_URL=postgresql+asyncpg://${_USER}:${_PASSWORD}@127.0.0.1:${_PORT}/${_DB}"
echo "export SPIRE_TEST_DATABASE_URL=postgresql+asyncpg://${_USER}:${_PASSWORD}@127.0.0.1:${_PORT}/${_DB}"
