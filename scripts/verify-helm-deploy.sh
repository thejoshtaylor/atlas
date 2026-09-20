#!/usr/bin/env bash
# The only real test of D-16's claim: that the guarded generation in
# charts/spire-voice/templates/secret.yaml actually behaves the way the
# template's own comments say across a REAL install and a REAL upgrade,
# against a real cluster. Nothing in this script is a rendered-template
# assertion -- tests/test_helm_chart.py already covers that ground. This
# creates a throwaway namespace, installs the chart into it, upgrades the
# release, and proves (or honestly disproves) five specific claims:
#
#   1. Every container in the application pod is accepted by the kubelet
#      and every init container runs to completion (CR-01, code review).
#      This claim is always attempted, because it needs nothing this
#      cluster may not have: it is satisfied, or refused, before the
#      application image is ever pulled. It exists because the version of
#      this script that lacked it printed "every attempted claim was
#      proved" against a chart whose pod could never start -- its
#      `wait-for-postgres` init container was refused outright
#      ("container has runAsNonRoot and image will run as root"), and
#      nothing here noticed, because `helm install` with no `--wait`
#      returns success the moment the API server has accepted the
#      manifests. A verification script that passes on a chart that
#      cannot deploy is worse than no script at all.
#   2. The generated SPIRE_SECRET_KEY survives the upgrade byte-identical
#      (T-07-37 -- the single highest-consequence failure this phase
#      guards against).
#   3. A row written to the bundled database before the upgrade is still
#      there after it (the deployment-level form of DEP-04).
#   4. The application pod itself becomes healthy (only attempted if
#      --wait-for-app is passed with a pullable image -- this cluster has
#      no registry path to the application image by default, and this
#      script must never claim to have proven something the pod that
#      would have proven it never ran). When it IS attempted, `helm
#      install`/`helm upgrade` are run with `--wait` as well, so a release
#      that never becomes ready fails the helm command itself rather than
#      being discovered (or missed) further down.
#   5. An uninstall followed by a reinstall keeps the generated key, and
#      the database that survived on its PVC still accepts the
#      reinstalled password (WR-01, code review). Always attempted, for
#      the same reason as claim 1: this is the failure a fresh operator
#      reaches by running two documented commands in order.
#
# Claim 1 and claim 4 are deliberately separate. Claim 4 cannot run on a
# cluster with no path to the application image; claim 1 can, and the one
# real deployment defect this phase shipped was inside exactly that gap.
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
# Claim 1's own budget. Generous on purpose: the init container it watches
# waits for a database whose very first start includes an initdb run on a
# freshly provisioned volume.
_INIT_TIMEOUT="240"

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
_app_selector="app.kubernetes.io/name=spire-voice,app.kubernetes.io/instance=$_RELEASE"
_secret_selector="app.kubernetes.io/instance=$_RELEASE"

# The kubelet's own words for "I refused this container before running it".
# Every one of these is a configuration the chart itself got wrong -- never
# a property of the cluster this happens to run on -- so each is a claim-1
# failure, by name. ImagePullBackOff/ErrImagePull are NOT in this set for
# the application container: that is the registry gap 07-08 disclosed, and
# claim 1 is scoped to end before the application image is needed.
_CONTAINER_REFUSAL_REASONS="CreateContainerConfigError CreateContainerError RunContainerError InvalidImageName CrashLoopBackOff"

_app_pod_name() {
  kubectl get pods -n "$_NAMESPACE" -l "$_app_selector" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

_init_container_states() {
  # One line per init container: "name|waitingReason|terminatedExitCode".
  # A field the API has not set yet renders empty (kubectl's jsonpath
  # writer tolerates a missing key by default), so "wait-for-postgres||"
  # means "created, still running" and "wait-for-postgres||0" means
  # "finished successfully".
  kubectl get pods -n "$_NAMESPACE" -l "$_app_selector" -o jsonpath=\
'{range .items[*].status.initContainerStatuses[*]}{.name}{"|"}{.state.waiting.reason}{"|"}{.state.terminated.exitCode}{"\n"}{end}' \
    2>/dev/null
}

_init_containers_settled() {
  # Echoes one of: "pending" (nothing to judge yet), "completed" (every
  # init container terminated with exit code 0), or "failed: <detail>".
  local states line name reason exit_code saw_any="false"
  states="$(_init_container_states)"
  [ -z "$states" ] && { echo "pending"; return; }
  while IFS='|' read -r name reason exit_code; do
    [ -z "$name" ] && continue
    saw_any="true"
    for refusal in $_CONTAINER_REFUSAL_REASONS; do
      if [ "$reason" = "$refusal" ]; then
        echo "failed: init container '$name' was refused by the kubelet ($reason)"
        return
      fi
    done
    case "$reason" in
      ErrImagePull | ImagePullBackOff | ErrImageNeverPull)
        echo "failed: init container '$name' could not pull its image ($reason)"
        return
        ;;
    esac
    if [ -n "$exit_code" ] && [ "$exit_code" != "0" ]; then
      echo "failed: init container '$name' exited $exit_code"
      return
    fi
    if [ -z "$exit_code" ]; then
      echo "pending"
      return
    fi
  done <<EOF
$states
EOF
  if [ "$saw_any" = "true" ]; then
    echo "completed"
  else
    echo "pending"
  fi
}

_wait_for_app_init_containers() {
  # Claim 1. Returns 0 when every init container has completed, 1 with a
  # named reason on stdout otherwise -- including the timeout case, which
  # is a failure and never a silent pass.
  local deadline outcome last="no application pod appeared"
  deadline=$(( $(date +%s) + _INIT_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    outcome="$(_init_containers_settled)"
    case "$outcome" in
      completed) return 0 ;;
      failed:*)
        echo "${outcome#failed: }"
        return 1
        ;;
    esac
    last="the init containers never finished within ${_INIT_TIMEOUT}s"
    sleep 3
  done
  echo "$last"
  return 1
}

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

_claim_init_status="not attempted"
_claim_key_status="not attempted"
_claim_db_status="not attempted"
_claim_app_status="not attempted"
_claim_reinstall_status="not attempted"
_any_attempted_claim_failed="false"

# `--wait` only when the application image is actually reachable: without
# it, helm would (correctly) fail on the disclosed registry gap and no
# claim below would ever be reached. With it, a release that never becomes
# ready is the helm command's own failure, which is what --wait-for-app
# asks for.
_HELM_WAIT_ARGS=()
if [ "$_WAIT_FOR_APP" = "true" ]; then
  _HELM_WAIT_ARGS+=(--wait --timeout "${_APP_TIMEOUT}s")
fi

echo "# installing $_RELEASE into $_NAMESPACE" >&2
if ! helm install "$_RELEASE" "$_CHART_DIR" -n "$_NAMESPACE" "${_HELM_SET_ARGS[@]}" "${_HELM_WAIT_ARGS[@]}" >&2; then
  echo "FATAL: helm install failed" >&2
  exit 1
fi

echo "# 0. waiting for the application pod's init containers to complete" >&2
_init_failure="$(_wait_for_app_init_containers)"
if [ -z "$_init_failure" ]; then
  _claim_init_status="proved: every init container in the application pod ran to completion"
else
  _claim_init_status="attempted, FAILED: $_init_failure"
  _any_attempted_claim_failed="true"
  # Nothing below this point can mean anything if the application pod
  # cannot get past its init containers -- the chart does not deploy.
  # Printing the kubelet's own event here is what turns "claim 1 failed"
  # into something an operator can act on without re-deriving it.
  kubectl get pods -n "$_NAMESPACE" -l "$_app_selector" >&2 || true
  kubectl describe pod -n "$_NAMESPACE" -l "$_app_selector" 2>/dev/null \
    | sed -n '/^Events:/,$p' >&2 || true
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
if ! helm upgrade "$_RELEASE" "$_CHART_DIR" -n "$_NAMESPACE" "${_HELM_SET_ARGS[@]}" "${_HELM_WAIT_ARGS[@]}" >&2; then
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

# --- claim 5: uninstall, then reinstall (WR-01, code review) --------------
#
# The path that destroyed a deployment in silence. `helm uninstall` deletes
# the Secret; Kubernetes never deletes the StatefulSet's PVC; a reinstall
# regenerated both POSTGRES_PASSWORD (which the surviving, already-initdb'd
# database rejects) and SPIRE_SECRET_KEY (which makes every credential row
# in that surviving database undecryptable). Two documented commands in
# order. This claim runs them in that order and checks all three of the
# things that have to still line up afterwards.
echo "# 6. uninstalling the release, then reinstalling it" >&2
if ! helm uninstall "$_RELEASE" -n "$_NAMESPACE" >&2; then
  _claim_reinstall_status="attempted, FAILED: helm uninstall failed"
  _any_attempted_claim_failed="true"
else
  _secret_key_after_uninstall="$(_secret_key_value)"
  if [ -z "$_secret_key_after_uninstall" ]; then
    _claim_reinstall_status="attempted, FAILED: the Secret was deleted by helm uninstall -- the surviving database PVC is now unreachable and every credential in it undecryptable"
    _any_attempted_claim_failed="true"
  elif ! helm install "$_RELEASE" "$_CHART_DIR" -n "$_NAMESPACE" "${_HELM_SET_ARGS[@]}" "${_HELM_WAIT_ARGS[@]}" >&2; then
    _claim_reinstall_status="attempted, FAILED: helm install after an uninstall failed"
    _any_attempted_claim_failed="true"
  elif ! _wait_for_postgres; then
    _claim_reinstall_status="attempted, FAILED: the database never became ready after the reinstall"
    _any_attempted_claim_failed="true"
  else
    _secret_key_reinstalled="$(_secret_key_value)"
    # The real question is not whether two base64 strings match -- it is
    # whether the reinstalled password still opens the database that
    # survived on its PVC. `_run_psql` reads the password out of the
    # reinstalled Secret, so this is exactly the authentication that
    # failed before.
    _reinstall_marker_count="$(_run_psql "SELECT count(*) FROM spire_voice_verify_marker WHERE note = '${_MARKER_TOKEN}';" 2>/dev/null | tr -d '[:space:]')"
    if [ "$_secret_key_reinstalled" != "$_secret_key_before" ]; then
      _claim_reinstall_status="attempted, FAILED: SPIRE_SECRET_KEY was regenerated by the reinstall -- every credential in the surviving database is now undecryptable"
      _any_attempted_claim_failed="true"
    elif [ "$_reinstall_marker_count" != "1" ]; then
      _claim_reinstall_status="attempted, FAILED: the surviving database refused the reinstalled password, or lost its data (marker rows found: '${_reinstall_marker_count:-<unreadable>}')"
      _any_attempted_claim_failed="true"
    else
      _claim_reinstall_status="proved: the key survived, and the surviving database still accepts the reinstalled password"
    fi
  fi
fi

echo ""
echo "==================== spire-voice Helm deploy verification ===================="
echo "cluster:    $_CLUSTER_SERVER (context: $_CONTEXT)"
echo "namespace:  $_NAMESPACE (deleted after this script exits)"
echo "release:    $_RELEASE"
echo ""
echo "claim 1 (every container in the application pod is accepted, and its"
echo "         init containers run to completion):"
echo "  -> $_claim_init_status"
echo "claim 2 (SPIRE_SECRET_KEY survives an upgrade, byte-identical):"
echo "  -> $_claim_key_status"
echo "claim 3 (a row written before the upgrade is still there after it):"
echo "  -> $_claim_db_status"
echo "claim 4 (the application pod itself becomes healthy):"
echo "  -> $_claim_app_status"
echo "claim 5 (an uninstall followed by a reinstall keeps the key, and the"
echo "         surviving database still accepts the reinstalled password):"
echo "  -> $_claim_reinstall_status"
echo "================================================================================"
if [ "$_WAIT_FOR_APP" != "true" ]; then
  echo "Owed: claim 4 was not attempted this run. Re-run with --image <repo you can"
  echo "pull from this cluster>:<tag> --wait-for-app to also prove it. Claim 1 was"
  echo "attempted regardless: it ends before the application image is needed."
fi
echo ""

if [ "$_any_attempted_claim_failed" = "true" ]; then
  echo "RESULT: at least one attempted claim failed -- see above." >&2
  exit 1
fi

echo "RESULT: every attempted claim was proved."
exit 0
