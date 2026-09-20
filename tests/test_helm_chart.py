"""Static proof for `charts/spire-voice` (DEP-01, D-14, D-15, D-16).

Renders the chart with the real `helm` binary and asserts what a human
reading the templates would otherwise have to check by eye: every
`${VAR}` the shipped configuration names has a key in the generated
Secret, the configuration shipped inside the chart is byte-identical to
the project's own file, the mount paths match what that configuration
references, and nothing in the chart depends on a third-party image
repository. Skipped by name, never vacuously passing, when the `helm`
binary itself is absent -- matching this codebase's own
`skip_without_postgres`/`skip_without_git_dir` precedent for "this check
has nothing to check in this environment."
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHART_DIR = _REPO_ROOT / "charts" / "spire-voice"
_EXAMPLE_CONFIG = _REPO_ROOT / "config" / "config.example.yaml"
_SHIPPED_CONFIG = _CHART_DIR / "files" / "config.yaml"

# Mirrors spire_voice.config._PLACEHOLDER_RE exactly -- this test does not
# import the application package (it does not need PYTHONPATH set up to
# run), so the pattern is duplicated rather than imported. If the two ever
# drift, test_config.py's own suite against the real loader is the
# authority; this test only needs "the same shape of ${NAME} placeholder."
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

skip_without_helm = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm is not on PATH -- install Helm to render/lint the chart "
    "instead of skipping these tests vacuously",
)


def _placeholder_names(raw_text: str) -> set[str]:
    """Every `${NAME}` config.example.yaml's own text names, on a
    non-comment line -- the same "skip whole-line comments" rule
    `spire_voice.config.expand_env` applies, so a placeholder mentioned only
    in prose (this file's own header) is never counted as a real variable."""
    names: set[str] = set()
    for line in raw_text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        names.update(_PLACEHOLDER_RE.findall(line))
    return names


def _helm_template(*extra_args: str, release: str = "spire-test") -> list[dict]:
    """Render the chart for real, and parse every YAML document it
    produces. Raises (via `check=True`) on a `helm template` failure --
    a bad template is a test failure, not a skip."""
    result = subprocess.run(
        ["helm", "template", release, str(_CHART_DIR), *extra_args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _helm_lint() -> subprocess.CompletedProcess:
    return subprocess.run(
        ["helm", "lint", str(_CHART_DIR)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _find_one(docs: list[dict], kind: str) -> dict:
    matches = [d for d in docs if d.get("kind") == kind]
    assert len(matches) == 1, f"expected exactly one {kind}, found {len(matches)}"
    return matches[0]


@skip_without_helm
def test_chart_lints_clean() -> None:
    result = _helm_lint()
    assert result.returncode == 0, result.stdout + result.stderr


@skip_without_helm
def test_chart_renders_to_valid_yaml_documents() -> None:
    docs = _helm_template()
    assert len(docs) >= 1
    kinds = {d.get("kind") for d in docs}
    assert "Secret" in kinds
    assert "ConfigMap" in kinds
    assert "Deployment" in kinds
    assert "Service" in kinds
    assert "PersistentVolumeClaim" in kinds


def test_shipped_config_is_byte_identical_to_the_projects_own_file() -> None:
    """The chart's `files/config.yaml` is a copy, not a fork -- if the
    project's own example config changes and this file is not updated to
    match, this test fails rather than silently drifting."""
    assert _SHIPPED_CONFIG.read_bytes() == _EXAMPLE_CONFIG.read_bytes()


@skip_without_helm
def test_secret_carries_a_key_for_every_variable_the_config_names() -> None:
    expected_vars = _placeholder_names(_EXAMPLE_CONFIG.read_text())
    assert expected_vars, "the example config named no ${VAR} placeholders -- test is broken"

    docs = _helm_template()
    secret = _find_one(docs, "Secret")
    secret_keys = set(secret.get("data", {}))

    missing = expected_vars - secret_keys
    assert not missing, f"Secret is missing keys for: {sorted(missing)}"


@skip_without_helm
def test_secret_values_default_to_empty_when_not_supplied() -> None:
    """An operator who supplies no credentials at all still gets a
    renderable Secret -- every value decodes, even to an empty string,
    rather than the template erroring on a missing key."""
    docs = _helm_template()
    secret = _find_one(docs, "Secret")
    for name, value in secret.get("data", {}).items():
        # Every value must be valid base64 (Kubernetes' own Secret.data
        # convention) -- a blank operator-supplied credential still
        # decodes to "" rather than failing to decode at all.
        base64.b64decode(value or "")


@skip_without_helm
def test_mount_paths_match_the_configs_own_references() -> None:
    """The paths the shipped config.yaml already names (/data, /models)
    must be exactly what the Deployment mounts -- this chart never
    templates the configuration's own contents, so a mismatch here would
    mean the pod cannot see what its own configuration expects."""
    config_text = _SHIPPED_CONFIG.read_text()
    assert '"/data' in config_text or "/data/" in config_text
    assert '"/models' in config_text or "/models/" in config_text

    docs = _helm_template()
    deployment = _find_one(docs, "Deployment")
    containers = deployment["spec"]["template"]["spec"]["containers"]
    mount_paths = {
        mount["mountPath"]
        for container in containers
        for mount in container.get("volumeMounts", [])
    }
    assert "/data" in mount_paths
    assert "/models" in mount_paths


@skip_without_helm
def test_no_template_names_a_third_party_image_repository() -> None:
    """No `bitnami` image anywhere -- Pitfall 5's finding that Bitnami's
    free image line is unmaintained as of 2025-08-28 is why this chart
    uses plain manifests against the official postgres image instead of a
    subchart."""
    docs = _helm_template()
    for doc in docs:
        rendered = yaml.safe_dump(doc)
        assert "bitnami" not in rendered.lower()


def test_chart_yaml_declares_no_dependencies() -> None:
    """No `dependencies:` block at all -- one chart, no subchart, per this
    plan's own objective."""
    chart_yaml = yaml.safe_load((_CHART_DIR / "Chart.yaml").read_text())
    assert "dependencies" not in chart_yaml


@skip_without_helm
@pytest.mark.parametrize("release", ["spire-key-check-a", "spire-key-check-b"])
def test_generated_secret_key_passes_the_real_application_validator(release: str) -> None:
    """Rendered twice (two independent renders, two independently
    generated keys -- `lookup` always returns nothing during
    `helm template`, so this cannot exercise the upgrade-preserves-the-key
    guarantee itself; that is scripts/verify-helm-deploy.sh's job against
    a real cluster). Both must satisfy auth/tokens.py's own
    validate_secret_key_strength -- not a reimplementation of its rules,
    the real function."""
    from spire_voice.auth.tokens import validate_secret_key_strength
    from spire_voice.config import SecurityConfig

    docs = _helm_template(release=release)
    secret = _find_one(docs, "Secret")
    decoded = base64.b64decode(secret["data"]["SPIRE_SECRET_KEY"]).decode("ascii")

    previous = os.environ.get("SPIRE_SECRET_KEY")
    os.environ["SPIRE_SECRET_KEY"] = decoded
    try:
        validate_secret_key_strength(SecurityConfig())
    finally:
        if previous is None:
            os.environ.pop("SPIRE_SECRET_KEY", None)
        else:
            os.environ["SPIRE_SECRET_KEY"] = previous


def test_generated_secret_key_is_backed_by_randbytes_not_randalphanum() -> None:
    """Pitfall 4: an alphanumeric string's decoded length is not its
    character count -- `randAlphaNum` must never back `SPIRE_SECRET_KEY`,
    only `randBytes` does."""
    template_text = (_CHART_DIR / "templates" / "secret.yaml").read_text()
    secret_key_line = next(
        line for line in template_text.splitlines() if "$secretKeyField = randBytes" in line
    )
    assert "randBytes 32" in secret_key_line
    assert "randAlphaNum" not in secret_key_line


@skip_without_helm
def test_database_manifests_name_the_official_image_and_publish_no_node_port() -> None:
    docs = _helm_template()
    stateful_sets = [d for d in docs if d.get("kind") == "StatefulSet"]
    assert len(stateful_sets) == 1
    postgres_sts = stateful_sets[0]
    images = [
        c["image"]
        for c in postgres_sts["spec"]["template"]["spec"]["containers"]
    ]
    assert any(image.startswith("postgres:") for image in images)

    services = [d for d in docs if d.get("kind") == "Service"]
    postgres_services = [
        s for s in services if "postgres" in s["metadata"]["name"]
    ]
    assert postgres_services, "no Postgres Service rendered"
    for svc in postgres_services:
        assert svc["spec"].get("type") not in {"NodePort", "LoadBalancer"}
        for port in svc["spec"].get("ports", []):
            assert "nodePort" not in port


@skip_without_helm
def test_no_default_database_password_ships_in_the_repository() -> None:
    """The bundled database's password is generated under the same guard
    as the application's own secret key -- nothing here is a literal
    default an operator could leave unchanged."""
    template_text = (_CHART_DIR / "templates" / "secret.yaml").read_text()
    assert "changeme" not in template_text.lower()
    assert "randAlpha 32" in template_text


# --- CR-01 (code review): the chart could not install at all --------------


@skip_without_helm
def test_every_container_the_pod_runs_can_satisfy_its_own_run_as_non_root() -> None:
    """The defect this test exists for: the pod-level `runAsNonRoot: true`
    applies to init containers too, and `postgres:18` (the
    `wait-for-postgres` init container's image) declares no `USER` at all
    -- it starts as root and drops privileges inside its own entrypoint.
    The kubelet refuses such a container before it runs
    (`Init:CreateContainerConfigError`), so the application pod never
    started on any cluster.

    `runAsNonRoot` can only refuse an image that would run as root; it
    cannot choose a uid. So the rule this asserts is: under a pod-level
    `runAsNonRoot`, any container whose image is not this project's own
    (the only image in the chart that declares a non-root `USER`) must
    name the uid it runs as, in its own securityContext.
    """
    docs = _helm_template()
    deployment = _find_one(docs, "Deployment")
    pod_spec = deployment["spec"]["template"]["spec"]

    assert pod_spec["securityContext"]["runAsNonRoot"] is True

    init_containers = pod_spec.get("initContainers", [])
    assert init_containers, "the wait-for-postgres init container disappeared"
    for container in init_containers:
        security_context = container.get("securityContext", {})
        run_as_user = security_context.get("runAsUser")
        assert isinstance(run_as_user, int), (
            f"init container {container['name']!r} runs image {container['image']!r} "
            "under the pod's runAsNonRoot but names no runAsUser of its own -- the "
            "kubelet refuses it if that image declares no non-root USER"
        )
        assert run_as_user != 0
        assert security_context.get("runAsNonRoot") is True


@skip_without_helm
def test_the_init_containers_declared_uid_is_the_one_its_image_really_has() -> None:
    """A uid pinned in a chart is a guess unless something checks it
    against the image. `postgres:18` creates its `postgres` user as
    uid/gid 999 (`id postgres` inside the real image). Pinning any other
    number would still satisfy the kubelet's root check while running
    `pg_isready` as a user that does not exist in the image's passwd
    database."""
    docs = _helm_template()
    deployment = _find_one(docs, "Deployment")
    init_containers = deployment["spec"]["template"]["spec"]["initContainers"]
    postgres_init = [c for c in init_containers if c["image"].startswith("postgres:")]
    assert postgres_init, "no init container runs the postgres image any more"
    for container in postgres_init:
        assert container["securityContext"]["runAsUser"] == 999
        assert container["securityContext"]["runAsGroup"] == 999


def test_the_deploy_verification_script_waits_for_the_pod_it_claims_to_verify() -> None:
    """The second half of CR-01. The script printed "every attempted claim
    was proved" against a release whose pod could never start, because its
    `helm install` carried no `--wait` and nothing else looked at the pod
    at all. This asserts the two properties that make that impossible:
    the script watches the application pod's init containers on every run
    (a claim that needs no registry access, unlike the pod-health claim),
    and it passes `--wait` to helm whenever it does have a pullable
    image."""
    script = (_REPO_ROOT / "scripts" / "verify-helm-deploy.sh").read_text()
    assert "_wait_for_app_init_containers" in script
    assert "initContainerStatuses" in script
    assert "CreateContainerConfigError" in script
    assert "_HELM_WAIT_ARGS+=(--wait" in script
    for command in ("helm install", "helm upgrade"):
        invocation = next(
            line for line in script.splitlines() if line.lstrip().startswith(f"if ! {command} ")
        )
        assert '"${_HELM_WAIT_ARGS[@]}"' in invocation, (
            f"{command} does not forward the --wait arguments, so a release that "
            "never becomes ready would be reported as a success"
        )
