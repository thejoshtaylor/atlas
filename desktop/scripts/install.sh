#!/usr/bin/env bash
# Builds, signs, and installs the ATLAS Mac app.
#
# Run this script from any directory:
#   desktop/scripts/install.sh            build, sign, install, and open the app
#   desktop/scripts/install.sh --verify   build and sign into desktop/build,
#                                         install nothing, and compare the
#                                         signing identity with the installed app
#   desktop/scripts/install.sh --help     show this text
#
# The app keeps its Accessibility grant and its Keychain access only when the
# signing identity stays the same. The script picks the identity in this order:
#   1. ATLAS_SIGN_IDENTITY (a SHA-1 hash or an exact identity name)
#   2. the saved identity in desktop/.signing-identity (not tracked by git)
#   3. the first "Developer ID Application" identity
#   4. the first "Apple Development" identity
#   5. "ATLAS Local Signing", a certificate this script creates once
# The script signs by SHA-1 hash. It never signs ad-hoc and never uses sudo.
#
# Env overrides (all optional):
#   ATLAS_SIGN_IDENTITY   the identity to sign with
#   ATLAS_IDENTITY_FILE   where the script saves the chosen identity
#
# Exit 0: done.
# Exit 1: --verify found a different identity, or no app is installed.
# Exit 2: this Mac does not meet the requirements, or the flag is unknown.
# Exit 3: ATLAS_SIGN_IDENTITY matches no valid identity.
# Exit 4: the local certificate is still not valid after the trust step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$DESKTOP_DIR/Resources/Info.plist.template"
BUILD_APP="$DESKTOP_DIR/build/ATLAS.app"
INSTALL_APP="/Applications/ATLAS.app"
IDENTITY_FILE="${ATLAS_IDENTITY_FILE:-$DESKTOP_DIR/.signing-identity}"
LOCAL_CERT_NAME="ATLAS Local Signing"
LOGIN_KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"
EXECUTABLE_NAME="AtlasDesktop"

log() {
  echo "install.sh: $*" >&2
}

usage() {
  cat <<'USAGE'
Usage:
  install.sh            Build, sign, install, and open ATLAS.app.
  install.sh --verify   Build and sign into desktop/build. Install nothing.
                        Exit 1 when the signing identity differs from the
                        installed app, or when no app is installed.
  install.sh --help     Show this text.
USAGE
}

preflight() {
  local failed=0 major xcode_path
  major="$(sw_vers -productVersion | cut -d. -f1)"
  if [ "${major:-0}" -lt 15 ]; then
    log "This Mac runs macOS $major. ATLAS needs macOS 15 or later."
    failed=1
  fi
  if [ "$(uname -m)" != "arm64" ]; then
    log "This Mac is not an Apple silicon Mac. ATLAS needs arm64."
    failed=1
  fi
  xcode_path="$(xcode-select -p 2>/dev/null || true)"
  case "$xcode_path" in
    *Xcode.app/Contents/Developer) ;;
    *)
      log "Full Xcode is not selected. Install Xcode from the App Store, then select it with xcode-select."
      failed=1
      ;;
  esac
  if ! xcodebuild -version >/dev/null 2>&1; then
    log "xcodebuild does not run. Open Xcode once and accept the license."
    failed=1
  fi
  if ! command -v swift >/dev/null 2>&1; then
    log "The swift command is not on PATH."
    failed=1
  fi
  if [ "$failed" -ne 0 ]; then
    exit 2
  fi
}

bundle_id_from_template() {
  /usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$TEMPLATE"
}

# One line per valid code-signing identity: "<SHA-1> <name>".
valid_identities() {
  { security find-identity -v -p codesigning || true; } |
    sed -n 's/^ *[0-9]*) \([0-9A-Fa-f]\{40\}\) "\(.*\)".*/\1 \2/p'
}

# Print the hash of the valid identity whose hash or exact name is $1.
identity_matching() {
  valid_identities | awk -v want="$1" '
    { name = substr($0, 42) }
    toupper($1) == toupper(want) || name == want { print toupper($1); exit }'
}

# Print the hash of the first valid identity whose name starts with $1.
first_identity_with_prefix() {
  valid_identities | awk -v prefix="$1" '
    index(substr($0, 42), prefix) == 1 { print toupper($1); exit }'
}

saved_identity() {
  [ -f "$IDENTITY_FILE" ] || return 0
  tr -d '[:space:]' <"$IDENTITY_FILE"
}

# Print a SHA-1 hash, or the word LOCAL_NEEDED. This function creates nothing.
pick_identity() {
  local found saved
  if [ -n "${ATLAS_SIGN_IDENTITY:-}" ]; then
    found="$(identity_matching "$ATLAS_SIGN_IDENTITY")"
    if [ -z "$found" ]; then
      log "ATLAS_SIGN_IDENTITY matches no valid code-signing identity."
      exit 3
    fi
    echo "$found"
    return 0
  fi
  saved="$(saved_identity)"
  if [ -n "$saved" ]; then
    found="$(identity_matching "$saved")"
    if [ -n "$found" ]; then
      echo "$found"
      return 0
    fi
    log "The saved signing identity is gone. macOS will reset the Accessibility grant and the Keychain access for ATLAS."
  fi
  found="$(first_identity_with_prefix 'Developer ID Application:')"
  if [ -z "$found" ]; then
    found="$(first_identity_with_prefix 'Apple Development:')"
  fi
  if [ -z "$found" ]; then
    found="$(identity_matching "$LOCAL_CERT_NAME")"
  fi
  echo "${found:-LOCAL_NEEDED}"
}

# Trust the local certificate for code signing only.
trust_local_certificate() {
  security add-trusted-cert -r trustRoot -p codeSign -k "$LOGIN_KEYCHAIN" "$1"
}

create_local_identity() {
  local work pass
  log "macOS will ask for your login password once. It lets this certificate sign code."
  work="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$work'" EXIT
  if security find-certificate -c "$LOCAL_CERT_NAME" -p "$LOGIN_KEYCHAIN" >"$work/cert.pem" 2>/dev/null &&
    [ -s "$work/cert.pem" ]; then
    log "The $LOCAL_CERT_NAME certificate exists but is not trusted. Trusting it now."
  else
    cat >"$work/openssl.cnf" <<CNF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $LOCAL_CERT_NAME
[ext]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
CNF
    /usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
      -config "$work/openssl.cnf" \
      -keyout "$work/key.pem" -out "$work/cert.pem" 2>/dev/null
    pass="$(/usr/bin/openssl rand -hex 16)"
    ATLAS_P12_PASS="$pass" /usr/bin/openssl pkcs12 -export \
      -inkey "$work/key.pem" -in "$work/cert.pem" -name "$LOCAL_CERT_NAME" \
      -passout env:ATLAS_P12_PASS -out "$work/atlas.p12"
    security import "$work/atlas.p12" -k "$LOGIN_KEYCHAIN" -P "$pass" \
      -T /usr/bin/codesign >/dev/null
  fi
  trust_local_certificate "$work/cert.pem"
  if [ -z "$(identity_matching "$LOCAL_CERT_NAME")" ]; then
    log "The $LOCAL_CERT_NAME identity is still not valid after the trust step."
    exit 4
  fi
}

save_identity() {
  (umask 077 && echo "$1" >"$IDENTITY_FILE")
}

build_app() {
  local bin build_number
  swift build -c release --arch arm64 --package-path "$DESKTOP_DIR" >&2
  bin="$(swift build -c release --arch arm64 --package-path "$DESKTOP_DIR" --show-bin-path)"
  rm -rf "$BUILD_APP"
  mkdir -p "$BUILD_APP/Contents/MacOS"
  cp "$bin/$EXECUTABLE_NAME" "$BUILD_APP/Contents/MacOS/$EXECUTABLE_NAME"
  build_number="$(date -u +%Y%m%d%H%M%S)"
  sed "s/__BUILD_NUMBER__/$build_number/" "$TEMPLATE" >"$BUILD_APP/Contents/Info.plist"
  plutil -lint "$BUILD_APP/Contents/Info.plist" >/dev/null
}

sign_app() {
  codesign --force --sign "$1" --identifier "$2" --timestamp=none "$BUILD_APP"
  codesign --verify --strict "$BUILD_APP"
}

# Print the designated requirement of the app at $1. Print nothing when $1 has none.
designated_requirement() {
  { codesign -d -r- "$1" 2>&1 || true; } | sed -n 's/^designated => //p'
}

quit_running_app() {
  local waited=0
  osascript -e "quit app id \"$1\"" >/dev/null 2>&1 || true
  while pgrep -x "$EXECUTABLE_NAME" >/dev/null 2>&1 && [ "$waited" -lt 5 ]; do
    sleep 1
    waited=$((waited + 1))
  done
  if pgrep -x "$EXECUTABLE_NAME" >/dev/null 2>&1; then
    pkill -x "$EXECUTABLE_NAME" || true
  fi
}

install_app() {
  if [ ! -w "$(dirname "$INSTALL_APP")" ]; then
    log "$(dirname "$INSTALL_APP") is not writable for this user. Ask an administrator account to run this script."
    exit 1
  fi
  quit_running_app "$1"
  rm -rf "$INSTALL_APP"
  ditto "$BUILD_APP" "$INSTALL_APP"
  open "$INSTALL_APP"
}

main() {
  local verify=0 sha bundle_id new_dr old_dr
  case "${1:-}" in
    "") ;;
    --verify) verify=1 ;;
    --help | -h)
      usage
      return 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  preflight
  bundle_id="$(bundle_id_from_template)"
  sha="$(pick_identity)"
  if [ "$sha" = "LOCAL_NEEDED" ]; then
    if [ "$verify" -eq 1 ]; then
      log "No signing identity yet. Run install.sh first."
      exit 1
    fi
    create_local_identity
    sha="$(pick_identity)"
  fi
  save_identity "$sha"
  build_app
  sign_app "$sha" "$bundle_id"
  new_dr="$(designated_requirement "$BUILD_APP")"
  old_dr=""
  if [ -d "$INSTALL_APP" ]; then
    old_dr="$(designated_requirement "$INSTALL_APP")"
  fi
  if [ -n "$old_dr" ] && [ "$old_dr" != "$new_dr" ]; then
    log "The signing identity changed. macOS will reset the Accessibility grant and the Keychain access for ATLAS."
  fi
  if [ "$verify" -eq 1 ]; then
    if [ -z "$old_dr" ]; then
      log "No app is installed at $INSTALL_APP."
      exit 1
    fi
    [ "$old_dr" = "$new_dr" ] || exit 1
    log "The signing identity matches the installed app."
    return 0
  fi
  install_app "$bundle_id"
  log "Installed $INSTALL_APP. Designated requirement: $new_dr"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
