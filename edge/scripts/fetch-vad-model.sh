#!/usr/bin/env bash
# Fetches the Silero VAD model this Pi's atlas_edge.vad.SileroGate reads
# from disk (config.py's DEFAULT_VAD_MODEL_PATH). Run this once, by hand,
# before starting the atlas-edge service -- the service itself never
# downloads a model (10-RESEARCH.md Pitfall 4, matching the server's own
# scripts/fetch_models.py discipline).
#
# Usage: edge/scripts/fetch-vad-model.sh [destination-directory]
#   (default destination directory: /var/lib/atlas-edge)
#
# Pinned sha256 computed 2026-09-25 against the published release file at:
#   https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
# A mismatch deletes the downloaded file and exits 1 -- a compromised
# mirror or a hijacked upstream account is the one risk this check
# guards against (T-10-31), and this project never hands an unverified
# file to a native extension.
set -euo pipefail

DEST_DIR="${1:-/var/lib/atlas-edge}"
DEST="${DEST_DIR}/silero_vad.onnx"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
EXPECTED_SHA256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"

if [ -f "$DEST" ]; then
  echo "already present: $DEST"
  exit 0
fi

mkdir -p "$DEST_DIR"
tmp="${DEST}.part"
curl -fsSL -o "$tmp" "$URL"

actual_sha256="$(sha256sum "$tmp" | awk '{print $1}')"
if [ "$actual_sha256" != "$EXPECTED_SHA256" ]; then
  rm -f "$tmp"
  echo "error: $tmp sha256 $actual_sha256 does not match the pinned digest $EXPECTED_SHA256 -- deleting rather than keeping an unverified model" >&2
  exit 1
fi

mv "$tmp" "$DEST"
echo "fetched: $DEST"
