#!/usr/bin/env bash
# Docker Compose's own D-16 mechanism: ATLAS_SECRET_KEY is generated once,
# on the first boot that finds neither the variable nor a saved copy of
# it, and persisted from then on. It is never regenerated.
#
# Why persistence, not generation, is the point: this one value derives
# both the JWT signing key and the credential-encryption key
# (src/atlas/auth/tokens.py). A key that changes across a restart
# makes every stored credential permanently unreadable and every issued
# session invalid, all at once. A version of this script that regenerated
# the key on every boot would look completely correct on the first boot
# and destroy the deployment on the second, with no error to show for it.
# This is the single highest-consequence line in this phase.
#
# The Helm chart takes a different path for the same problem (a
# `lookup`-guarded Kubernetes Secret, checked once at install/upgrade
# time) because Compose has no equivalent of `lookup` -- a file on the
# durable data volume is Compose's own form of that same persistence.
set -euo pipefail

DATA_DIR="${ATLAS_DATA_DIR:-/data}"
SECRET_FILE="${DATA_DIR}/secret_key"

if [ -z "${ATLAS_SECRET_KEY:-}" ]; then
  if [ ! -f "$SECRET_FILE" ]; then
    # The exact generation call src/atlas/config.py::read_secret_key
    # already tells an operator to run by hand on a missing-key refusal --
    # reused verbatim here rather than a second generation method.
    python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
  fi
  ATLAS_SECRET_KEY="$(cat "$SECRET_FILE")"
  export ATLAS_SECRET_KEY
fi

# Replaces this shell with the real process, so signals (SIGTERM on
# `docker compose down`, SIGHUP on a restart) reach the application
# directly, not a shell sitting in between it and the container runtime.
exec "$@"
