#!/usr/bin/env bash
# The proof behind docs/runbooks/deploy-compose.md (DEP-03): a fresh clone,
# with nothing on the machine that the clone did not bring, reaches a
# running assistant by following only the commands that runbook actually
# gives -- no extra step, no environment variable the runbook does not
# mention, and no file created that the runbook does not tell the reader
# to create.
#
# Runs entirely against a *local* clone (this script's own repository, by
# default) rather than the public GitHub remote: this is what lets the
# script prove commits that have not been pushed yet, and what lets it run
# with no network access to GitHub at all -- `git clone` of a local path
# is a real, independent clone (new working tree, no shared git config,
# no leftover .venv/node_modules/build output), which is exactly the
# "nothing the clone did not bring" property this script exists to test.
# Pass a real remote URL as $1 to prove the same thing against a pushed
# branch instead.
#
# Uses its own Compose project name and its own published port -- the
# override docs/runbooks/deploy-compose.md itself documents as the way to
# run a second instance -- so this can never collide with a stack the
# operator already has running. Everything this script creates (the
# temporary directory, the containers, the volumes) is removed in a trap
# that runs whether the script succeeds or fails.
#
# Usage:
#   scripts/verify-clean-clone.sh [SOURCE]
#     SOURCE: a git clone source (default: this repository's own working
#     copy, resolved from this script's own location).
set -uo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_SOURCE="${1:-$_REPO_ROOT}"

_RUN_ID="$$-${RANDOM}"
_PROJECT="spire-voice-verify-${_RUN_ID}"
_PORT="18080"
_HEALTH_TIMEOUT_S="180"

_TMP_DIR=""
_CLEANED_UP="false"

cleanup() {
  if [ "$_CLEANED_UP" = "true" ]; then
    return
  fi
  _CLEANED_UP="true"
  if [ -n "$_TMP_DIR" ] && [ -d "$_TMP_DIR" ]; then
    echo "# cleaning up: tearing down the ${_PROJECT} stack and its volumes" >&2
    (
      cd "$_TMP_DIR" 2>/dev/null \
        && COMPOSE_PROJECT_NAME="$_PROJECT" SPIRE_PORT="$_PORT" \
          docker compose down -v >/dev/null 2>&1
    ) || true
    echo "# cleaning up: removing $_TMP_DIR" >&2
    rm -rf "$_TMP_DIR"
  fi
}
trap cleanup EXIT

if ! command -v docker >/dev/null 2>&1; then
  echo "FATAL: docker is not on PATH" >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "FATAL: the docker compose plugin is not available (docker compose version failed)" >&2
  exit 1
fi

_TMP_DIR="$(mktemp -d -t spire-voice-verify-clean-clone.XXXXXX)"
echo "# cloning $_SOURCE into $_TMP_DIR" >&2
if ! git clone --quiet "$_SOURCE" "$_TMP_DIR"; then
  echo "FATAL: git clone failed" >&2
  exit 1
fi

cd "$_TMP_DIR"

# WR-10 (code review): the Dockerfile's `test` stage, built for real.
#
# That stage's own comment calls itself "the actual proof" that this
# project's whole dependency set installs and its suite passes on Python
# 3.12 (the version D-13 locks, against the 3.14 this project develops
# on). It had never run: `runtime` is `FROM python-base`, not `FROM
# test`, BuildKit builds only what the target needs, and nothing in the
# repository passed `--target test`. Building it here is what makes that
# comment true -- and this is the right place for it, because a clean
# clone that cannot pass its own tests on the Python it ships has not
# reached a running assistant in any sense DEP-03 means.
echo "# building the image's own test stage (Python 3.12, the version D-13 locks)" >&2
if ! docker build --target test -t "spire-voice-test:${_RUN_ID}" . >&2; then
  echo "FATAL: the image's own test stage failed -- this project's dependency set does not install and pass on the Python the image ships" >&2
  exit 1
fi
docker image rm "spire-voice-test:${_RUN_ID}" >/dev/null 2>&1 || true

# docs/runbooks/deploy-compose.md's own wake-word step -- run before the
# stack comes up, exactly as documented, with no .env file present.
echo "# provisioning the Vosk wake-word model (docs/runbooks/deploy-compose.md)" >&2
if ! COMPOSE_PROJECT_NAME="$_PROJECT" SPIRE_PORT="$_PORT" \
  docker compose run --rm app python scripts/fetch_wake_model.py --config config/config.example.yaml >&2
then
  echo "FATAL: scripts/fetch_wake_model.py failed inside the container -- the README's own documented step did not work from a clean clone" >&2
  exit 1
fi

# docs/runbooks/deploy-compose.md's own "Bring it up" command, unchanged
# apart from the project-name/port override the same document names as
# the sanctioned way to run a second instance.
echo "# bringing the stack up: docker compose up -d --build --wait" >&2
if ! COMPOSE_PROJECT_NAME="$_PROJECT" SPIRE_PORT="$_PORT" \
  docker compose up -d --build --wait >&2
then
  echo "FATAL: docker compose up did not reach a healthy stack from a clean clone" >&2
  echo "# app container logs:" >&2
  COMPOSE_PROJECT_NAME="$_PROJECT" SPIRE_PORT="$_PORT" docker compose logs app >&2 || true
  exit 1
fi

_HEALTH_URL="http://127.0.0.1:${_PORT}/health"
echo "# waiting for $_HEALTH_URL (up to ${_HEALTH_TIMEOUT_S}s)" >&2
_deadline=$((SECONDS + _HEALTH_TIMEOUT_S))
_health_body=""
while [ "$SECONDS" -lt "$_deadline" ]; do
  if _health_body="$(curl -fsS "$_HEALTH_URL" 2>/dev/null)"; then
    break
  fi
  sleep 2
done
if [ -z "$_health_body" ]; then
  echo "FATAL: $_HEALTH_URL never became reachable within ${_HEALTH_TIMEOUT_S}s" >&2
  exit 1
fi
echo "# health route: $_health_body" >&2

_ROOT_URL="http://127.0.0.1:${_PORT}/"
echo "# fetching the served page: $_ROOT_URL" >&2
_root_status_file="$(mktemp)"
_root_status="$(curl -sS -o "$_root_status_file" -w '%{http_code}' "$_ROOT_URL" 2>/dev/null || echo "000")"
_root_body="$(cat "$_root_status_file" 2>/dev/null || true)"
rm -f "$_root_status_file"
if [ "$_root_status" = "000" ]; then
  echo "FATAL: $_ROOT_URL was not reachable at all" >&2
  exit 1
fi
if printf '%s' "$_root_body" | grep -qi "<html"; then
  _served_page_status="confirmed: HTTP $_root_status served an HTML page"
else
  # deferred-items.md #1 (now resolved): the setup gate used to be an
  # application-level dependency that also refused the SPA shell itself,
  # so a fresh deployment's root page answered the same JSON refusal
  # instead of the first-run wizard. The gate now lives on the backend
  # routes only (never on `app` itself), so this is no longer a disclosed,
  # non-fatal gap -- a non-HTML root page here is a real regression of
  # that fix, and this script fails loudly on it rather than warning.
  echo "FATAL: $_ROOT_URL did not serve the first-run wizard -- NOT an HTML page (HTTP $_root_status): ${_root_body:0:200}" >&2
  exit 1
fi

echo ""
echo "RESULT: a clean clone, following only docs/runbooks/deploy-compose.md, reached a"
echo "running assistant -- health route confirmed. Served page: $_served_page_status"
exit 0
