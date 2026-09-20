#!/usr/bin/env bash
# The only real test of D-16's claim: that the guarded generation in
# charts/spire-voice/templates/secret.yaml actually behaves the way the
# template's own comments say across a REAL install and a REAL upgrade,
# against a real cluster. Nothing in this script is a rendered-template
# assertion -- tests/test_helm_chart.py already covers that ground. This
# creates a throwaway namespace, installs the chart into it, upgrades the
# release, and proves (or honestly disproves) three specific claims:
#
#   1. The generated SPIRE_SECRET_KEY survives the upgrade byte-identical
#      (T-07-37 -- the single highest-consequence failure this phase
#      guards against).
#   2. A row written to the bundled database before the upgrade is still
#      there after it (the deployment-level form of DEP-04).
#   3. The application pod itself becomes healthy (only attempted if
#      --wait-for-app is passed with a pullable image -- this cluster has
#      no registry path to the application image by default, and this
#      script must never claim to have proven something the pod that
#      would have proven it never ran).
#
# The namespace this script creates is always deleted afterwards, in a
# trap that runs whether the script succeeds or fails -- it never installs
# into, or deletes, a namespace it did not itself create.
#
# Usage:
#   scripts/verify-helm-deploy.sh [--image REPO:TAG] [--wait-for-app] [--timeout SECONDS]
set -uo pipefail

_CHART_DIR="charts/spire-voice"
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_RELEASE="spire-voice-verify-$$-${RANDOM}"
_NAMESPACE="$_RELEASE"

_IMAGE_REPO=""
_IMAGE_TAG=""
_WAIT_FOR_APP="false"
_APP_TIMEOUT="180"
_DB_TIMEOUT="180"

while [ $# -gt 0 ]; do
  case "$1" in
    --image)
      _IMAGE_REF="$2"
      _IMAGE_REPO="${_IMAGE_REF%%:*}"
      _IMAGE_TAG="${_IMAGE_REF##*:}"
      shift 2
      ;;
    --wait-for-app)
      _WAIT_FOR_APP="true"
      shift
      ;;
    --timeout)
      _APP_TIMEOUT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--image REPO:TAG] [--wait-for-app] [--timeout SECONDS]" >&2  # --timeout bounds the application rollout wait only; the database wait has its own fixed, generous budget
      exit 2
      ;;
  esac
done

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is not on PATH -- this script needs a reachable Kubernetes cluster" >&2
  exit 1
fi
if ! command -v helm >/dev/null 2>&1; then
  echo "helm is not on PATH" >&2
  exit 1
fi
if ! kubectl cluster-info --request-timeout=10s >/dev/null 2>&1; then
  echo "kubectl cluster-info failed -- no reachable cluster, or no current context" >&2
  exit 1
fi

_CLUSTER_SERVER="$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}' 2>/dev/null || echo "unknown")"
_CONTEXT="$(kubectl config current-context 2>/dev/null || echo "unknown")"

# Never installs into, or deletes, a namespace this script did not create
# itself -- this check is what makes that true rather than assumed.
if kubectl get namespace "$_NAMESPACE" >/dev/null 2>&1; then
  echo "FATAL: namespace $_NAMESPACE already exists -- refusing to touch a namespace this script did not create" >&2
  exit 1
fi

_CLEANED_UP="false"
cleanup() {
  if [ "$_CLEANED_UP" = "true" ]; then
    return
  fi
  _CLEANED_UP="true"
  echo "# cleaning up: uninstalling $_RELEASE and deleting namespace $_NAMESPACE" >&2
  helm uninstall "$_RELEASE" -n "$_NAMESPACE" >/dev/null 2>&1 || true
  kubectl delete namespace "$_NAMESPACE" --wait=true --timeout=120s >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "# creating throwaway namespace $_NAMESPACE" >&2
if ! kubectl create namespace "$_NAMESPACE" >/dev/null 2>&1; then
  echo "FATAL: could not create namespace $_NAMESPACE" >&2
  exit 1
fi

# Reused, unchanged, between install and upgrade -- an upgrade with
# different values would not test what D-16 claims.
_HELM_SET_ARGS=()
if [ -n "$_IMAGE_REPO" ]; then
  _HELM_SET_ARGS+=(--set "image.repository=$_IMAGE_REPO" --set "image.tag=$_IMAGE_TAG")
fi

_pg_selector="app.kubernetes.io/name=spire-voice-postgres,app.kubernetes.io/instance=$_RELEASE"
_secret_selector="app.kubernetes.io/instance=$_RELEASE"

_wait_for_postgres() {
  kubectl wait --for=condition=Ready pod \
    -l "$_pg_selector" -n "$_NAMESPACE" --timeout="${_DB_TIMEOUT}s" >/dev/null 2>&1
}

_postgres_pod_name() {
  kubectl get pods -n "$_NAMESPACE" -l "$_pg_selector" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

_secret_key_value() {
  kubectl get secret -n "$_NAMESPACE" -l "$_secret_selector" \
    -o jsonpath='{.items[0].data.SPIRE_SECRET_KEY}' 2>/dev/null
}

_run_psql() {
  # $1: the SQL. Runs inside the postgres pod itself -- no port-forward,
  # no client tooling needed on the host running this script.
  local pod
  pod="$(_postgres_pod_name)"
  kubectl exec -n "$_NAMESPACE" "$pod" -- \
    env PGPASSWORD="$(kubectl get secret -n "$_NAMESPACE" -l "$_secret_selector" -o jsonpath='{.items[0].data.POSTGRES_PASSWORD}' | base64 -d)" \
    psql -U spire -d spire -tAc "$1"
}

_MARKER_TOKEN="verify-$$-$(date +%s)"

_claim_key_status="not attempted"
_claim_db_status="not attempted"
_claim_app_status="not attempted"
_any_attempted_claim_failed="false"

echo "# installing $_RELEASE into $_NAMESPACE" >&2
if ! helm install "$_RELEASE" "$_CHART_DIR" -n "$_NAMESPACE" "${_HELM_SET_ARGS[@]}" >&2; then
  echo "FATAL: helm install failed" >&2
  exit 1
fi

echo "# waiting for the bundled database to become ready" >&2
if ! _wait_for_postgres; then
  echo "FATAL: the bundled database never became ready -- cannot write or read the marker row" >&2
  exit 1
fi

echo "# 1. reading the generated secret key before the upgrade" >&2
_secret_key_before="$(_secret_key_value)"
if [ -z "$_secret_key_before" ]; then
  echo "FATAL: could not read SPIRE_SECRET_KEY from the installed release's Secret" >&2
  exit 1
fi

echo "# 2. writing a marker row into the bundled database" >&2
if ! _run_psql "CREATE TABLE IF NOT EXISTS spire_voice_verify_marker (id serial primary key, note text); INSERT INTO spire_voice_verify_marker (note) VALUES ('${_MARKER_TOKEN}');" >/dev/null 2>&1; then
  echo "FATAL: could not write the marker row before the upgrade" >&2
  exit 1
fi

if [ "$_WAIT_FOR_APP" = "true" ]; then
  echo "# waiting for the application deployment to roll out, before the upgrade" >&2
  if kubectl rollout status "deployment/$_RELEASE" -n "$_NAMESPACE" --timeout="${_APP_TIMEOUT}s" >&2; then
    _app_ready_before="true"
  else
    _app_ready_before="false"
  fi
else
  _app_ready_before=""
fi

echo "# 3. upgrading $_RELEASE" >&2
if ! helm upgrade "$_RELEASE" "$_CHART_DIR" -n "$_NAMESPACE" "${_HELM_SET_ARGS[@]}" >&2; then
  echo "FATAL: helm upgrade failed" >&2
  exit 1
fi

echo "# waiting for the bundled database to be ready again after the upgrade" >&2
if ! _wait_for_postgres; then
  echo "FATAL: the bundled database never came back ready after the upgrade" >&2
  exit 1
fi

echo "# 4. reading the generated secret key again, after the upgrade" >&2
_secret_key_after="$(_secret_key_value)"
if [ -z "$_secret_key_after" ]; then
  _claim_key_status="attempted, failed (could not read the Secret after upgrade)"
  _any_attempted_claim_failed="true"
elif [ "$_secret_key_before" = "$_secret_key_after" ]; then
  _claim_key_status="proved: byte-identical before and after the upgrade"
else
  _claim_key_status="attempted, FAILED: the key changed across the upgrade"
  _any_attempted_claim_failed="true"
fi

echo "# 5. checking the marker row is still there after the upgrade" >&2
_marker_count="$(_run_psql "SELECT count(*) FROM spire_voice_verify_marker WHERE note = '${_MARKER_TOKEN}';" 2>/dev/null | tr -d '[:space:]')"
if [ "$_marker_count" = "1" ]; then
  _claim_db_status="proved: the marker row survived the upgrade"
else
  _claim_db_status="attempted, FAILED: expected 1 marker row, found '${_marker_count:-<unreadable>}'"
  _any_attempted_claim_failed="true"
fi

if [ "$_WAIT_FOR_APP" = "true" ]; then
  echo "# waiting for the application deployment to roll out, after the upgrade" >&2
  if kubectl rollout status "deployment/$_RELEASE" -n "$_NAMESPACE" --timeout="${_APP_TIMEOUT}s" >&2; then
    if [ "$_app_ready_before" = "true" ]; then
      _claim_app_status="proved: the application pod was healthy before and after the upgrade"
    else
      _claim_app_status="attempted, FAILED: healthy after the upgrade but was not healthy before it"
      _any_attempted_claim_failed="true"
    fi
  else
    _claim_app_status="attempted, FAILED: the application deployment never became ready"
    _any_attempted_claim_failed="true"
  fi
else
  _claim_app_status="not attempted -- pass --wait-for-app with a pullable --image to also prove this claim"
fi

echo ""
echo "==================== spire-voice Helm deploy verification ===================="
echo "cluster:    $_CLUSTER_SERVER (context: $_CONTEXT)"
echo "namespace:  $_NAMESPACE (deleted after this script exits)"
echo "release:    $_RELEASE"
echo ""
echo "claim 1 (SPIRE_SECRET_KEY survives an upgrade, byte-identical):"
echo "  -> $_claim_key_status"
echo "claim 2 (a row written before the upgrade is still there after it):"
echo "  -> $_claim_db_status"
echo "claim 3 (the application pod itself becomes healthy):"
echo "  -> $_claim_app_status"
echo "================================================================================"
if [ "$_WAIT_FOR_APP" != "true" ]; then
  echo "Owed: claim 3 was not attempted this run. Re-run with --image <repo you can"
  echo "pull from this cluster>:<tag> --wait-for-app to also prove it."
fi
echo ""

if [ "$_any_attempted_claim_failed" = "true" ]; then
  echo "RESULT: at least one attempted claim failed -- see above." >&2
  exit 1
fi

echo "RESULT: every attempted claim was proved."
exit 0
