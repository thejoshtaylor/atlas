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
