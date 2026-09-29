#!/usr/bin/env bash
# Moves this Pi to the commit the server was built from. The systemd timer
# atlas-edge-update.timer runs atlas-edge-update.service as root, and that
# unit runs this script.
#
# The script does these steps:
#   1. It reads the server address from the edge config and asks the
#      server's /health route for its build commit.
#   2. If that commit is the checkout's HEAD, it stops.
#   3. If the commit is not on origin/main, it refuses the commit.
#   4. Otherwise it checks out the commit, installs it as the runbook does
#      (docs/runbooks/edge-microphone.md, steps 3 and 11), and restarts
#      atlas-edge. If atlas-edge does not stay up for 30 seconds, it goes
#      back to the old commit.
#
# Env overrides (both optional):
#   ATLAS_EDGE_DIR          the checkout (default: /opt/atlas-edge)
#   ATLAS_EDGE_CONFIG_PATH  the edge config (default: /etc/atlas-edge/config.toml)
#
# Exit 0: nothing to do, or the update worked.
# Exit 1: the update was refused or failed.
set -euo pipefail

ATLAS_EDGE_DIR="${ATLAS_EDGE_DIR:-/opt/atlas-edge}"
# Mirrors DEFAULT_CONFIG_PATH in edge/src/atlas_edge/__main__.py.
ATLAS_EDGE_CONFIG_PATH="${ATLAS_EDGE_CONFIG_PATH:-/etc/atlas-edge/config.toml}"
VENV_PY="$ATLAS_EDGE_DIR/edge/.venv/bin/python"
UNITS_DIR="/etc/systemd/system"

log() {
  echo "atlas-edge-update: $*" >&2
}

git_c() {
  git -c safe.directory="$ATLAS_EDGE_DIR" -C "$ATLAS_EDGE_DIR" "$@"
}

# The server address comes from the edge config. Only server_url is read.
# It is a websocket URL with a path. The health URL replaces the scheme
# and the path, and keeps the host.
health_url() {
  "$VENV_PY" - "$ATLAS_EDGE_CONFIG_PATH" <<'PY'
import sys
import tomllib
from urllib.parse import urlsplit, urlunsplit

with open(sys.argv[1], "rb") as handle:
    server_url = tomllib.load(handle)["server_url"]
parts = urlsplit(server_url)
scheme = {"wss": "https", "ws": "http"}.get(parts.scheme)
if scheme is None or not parts.netloc:
    sys.exit(1)
print(urlunsplit((scheme, parts.netloc, "/health", "", "")))
PY
}

# Prints the "commit" string of a /health body. Prints nothing on any
# parse error or when the key is missing.
commit_of() {
  printf '%s' "$1" | "$VENV_PY" -c '
import json
import sys

try:
    value = json.load(sys.stdin).get("commit", "")
except Exception:
    value = ""
print(value if isinstance(value, str) else "")
'
}

# Install the current checkout and restart atlas-edge. Returns 0 only if
# atlas-edge stays up. Bash ignores set -e inside a function that an "if"
# calls, so each step ends with "|| return 1".
#
# The updater runs as root, so every file it writes is root-owned. This
# matches the runbook's sudo clone and sudo uv sync. Never change the
# owner to atlas-edge. That user faces the network, and it must not be
# able to write code that root runs later.
install_current() {
  local unit target restarts_before restarts_after

  # The same install as runbook step 3.
  (
    cd "$ATLAS_EDGE_DIR/edge" &&
      env UV_PYTHON_INSTALL_DIR="$ATLAS_EDGE_DIR/.python" uv sync --frozen
  ) || return 1

  # Refresh only the unit files that are already installed. Optional units
  # such as atlas-librespot stay opt-in. This also keeps the two
  # atlas-edge-update units current after the first install.
  for unit in "$ATLAS_EDGE_DIR"/edge/systemd/*.service "$ATLAS_EDGE_DIR"/edge/systemd/*.timer; do
    [ -e "$unit" ] || continue
    target="$UNITS_DIR/$(basename "$unit")"
    [ -e "$target" ] || continue
    cp "$unit" "$target" || return 1
  done
  systemctl daemon-reload || return 1

  # ponytail: no restart deferral during an active turn. This is the known
  # ceiling. A restart during a turn drops that turn. The upgrade path:
  # atlas-edge publishes its turn state (for example a file under
  # /run/atlas-edge), and the updater skips a run while a turn is live.
  systemctl restart atlas-edge || return 1
  restarts_before="$(systemctl show -p NRestarts --value atlas-edge)" || return 1
  sleep 30

  # Restart=always can make a crash loop read as active at any single
  # moment. The NRestarts comparison is what catches it. The client
  # reconnects in-process, so a restart in this window means the new
  # build is broken.
  systemctl is-active --quiet atlas-edge || return 1
  restarts_after="$(systemctl show -p NRestarts --value atlas-edge)" || return 1
  [ "$restarts_after" = "$restarts_before" ] || return 1
}

main() {
  local url body commit old_head

  url="$(health_url)" || {
    log "cannot read server_url from $ATLAS_EDGE_CONFIG_PATH"
    return 1
  }

  body="$(curl -fsS --max-time 10 "$url")" || {
    log "server unreachable"
    return 0
  }
  commit="$(commit_of "$body")"
  # Only a full lowercase hex commit ever reaches a git argument. A blank
  # value means the server image was built without the GIT_SHA arg.
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || return 0

  old_head="$(git_c rev-parse HEAD)"
  [ "$commit" != "$old_head" ] || return 0

  git_c fetch --quiet origin main || {
    log "cannot fetch origin main"
    return 1
  }
  # Only commits on main ever run as root.
  if ! git_c merge-base --is-ancestor "$commit" origin/main; then
    log "commit $commit is not on main. Refused."
    return 1
  fi

  # Nothing has changed yet, so a failed checkout needs no rollback.
  git_c checkout --quiet --detach "$commit" || {
    log "cannot check out $commit"
    return 1
  }

  if install_current; then
    log "now at $commit"
    return 0
  fi

  log "commit $commit did not stay up. Going back to $old_head."
  if git_c checkout --quiet --detach "$old_head" && install_current; then
    log "rolled back to $old_head"
  else
    log "ROLLBACK FAILED. atlas-edge needs a manual check."
  fi
  return 1
}

# Bash reads a script as it runs it, and the checkout above replaces this
# file. Everything is in functions, and this last line runs only after
# bash has parsed the whole file.
main "$@"
