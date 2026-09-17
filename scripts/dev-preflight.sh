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
