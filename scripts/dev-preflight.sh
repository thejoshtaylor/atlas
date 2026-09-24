#!/usr/bin/env bash
# Report whether the live dev-harness run can proceed. Prints readiness only —
# never a secret value, never a length, never a prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then echo "env-file: MISSING"; exit 1; fi
set -a; . ./.env 2>/dev/null; set +a

for v in XAI_API_KEY HA_URL HA_TOKEN; do
  if [ -n "${!v:-}" ]; then echo "$v: set"; else echo "$v: BLANK"; fi
done

if [ -n "${XAI_API_KEY:-}" ]; then
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
    -H "Authorization: Bearer ${XAI_API_KEY}" https://api.x.ai/v1/models || echo "000")
  case "$code" in
    200) echo "xai-auth: OK (HTTP 200)";;
    401|403) echo "xai-auth: REJECTED (HTTP $code) — key not accepted";;
    000) echo "xai-auth: UNREACHABLE";;
    *) echo "xai-auth: unexpected HTTP $code";;
  esac
fi

if [ -n "${HA_URL:-}" ] && [ -n "${HA_TOKEN:-}" ]; then
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
    -H "Authorization: Bearer ${HA_TOKEN}" "${HA_URL%/}/api/" || echo "000")
  case "$code" in
    200) echo "ha-auth: OK (HTTP 200)";;
    401|403) echo "ha-auth: REJECTED (HTTP $code) — token not accepted";;
    000) echo "ha-auth: UNREACHABLE — check HA_URL host and port";;
    *) echo "ha-auth: unexpected HTTP $code";;
  esac
fi

# The speaker path (FfmpegSupervisor, plan 02-03/02-07) depends on the
# system `ffmpeg` binary -- not a Python package, not installed by anything
# in this repository. Without it the egress child can never start, and the
# failure surfaces as a respawn loop, not as a missing dependency. A
# version string is not secret material -- reported in full, unlike the
# credential lines above.
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg_version=$(ffmpeg -version 2>/dev/null | head -n 1 | awk '{print $3}')
  echo "ffmpeg: OK (${ffmpeg_version:-version unknown})"
else
  echo "ffmpeg: MISSING — the speaker egress child cannot start without it"
fi

# A missing speaker FIFO directory produces the exact same respawn-loop
# symptom as a missing `ffmpeg` binary, from an unrelated cause -- this
# line lets the operator tell the two apart before starting anything.
# Read from the configured `speaker.fifo_path` (`config.example.yaml`'s
# own default shown here as the fallback), never a secret, so printing the
# path itself is fine.
speaker_config_path="${ATLAS_CONFIG:-config/config.example.yaml}"
speaker_fifo_path=$(awk -F': *' '/^[[:space:]]*fifo_path:/ {gsub(/["'"'"']/, "", $2); print $2; exit}' "$speaker_config_path" 2>/dev/null)
speaker_fifo_path="${speaker_fifo_path:-/run/atlas/speaker.alaw}"
speaker_fifo_dir=$(dirname "$speaker_fifo_path")
if [ -d "$speaker_fifo_dir" ] && [ -w "$speaker_fifo_dir" ]; then
  echo "speaker-fifo-dir: OK ($speaker_fifo_dir exists and is writable)"
else
  echo "speaker-fifo-dir: MISSING or not writable ($speaker_fifo_dir)"
fi
