"""The deployment-time contracts DEP-02/DEP-03/D-16 stand on: a first-boot
secret key that persists across a restart (this file's Task 2 section),
and the Docker Compose deployment those persistence guarantees back (Task
3 adds to this file).

D-16 is the highest-stakes claim in this phase: `SPIRE_SECRET_KEY` derives
both the JWT signing key and the credential-encryption key
(`auth/tokens.py`). A key that changes across a restart makes every stored
credential permanently unreadable and every issued session invalid, all at
once -- and it would look like it worked on the first boot. The tests
below run the real script through a real subprocess rather than
reimplementing its logic in Python: the property under test is bash's own
filesystem and `exec` behaviour, and a Python mock of bash proves nothing
about that.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import yaml

from spire_voice.auth.tokens import validate_secret_key_strength
from spire_voice.config import SecurityConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENTRYPOINT = _REPO_ROOT / "deploy" / "docker-entrypoint.sh"
_COMPOSE_FILE = _REPO_ROOT / "docker-compose.yml"
_CONFIG_EXAMPLE = _REPO_ROOT / "config" / "config.example.yaml"

# The same ${NAME} shape config.py::expand_env matches -- kept here as an
# independent, deliberately duplicated regex (not an import) so this test
# does not silently stop meaning anything if that module's own pattern
# ever changes shape.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _run_entrypoint(
    data_dir: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Runs the real entrypoint, wrapping a command that prints
    `SPIRE_SECRET_KEY` back out -- the entrypoint's whole job is making
    sure that variable is correct and exported by the time the wrapped
    command runs, and this is the simplest real proof of that."""
    env = {"PATH": os.environ.get("PATH", ""), "SPIRE_DATA_DIR": str(data_dir)}
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(_ENTRYPOINT), "sh", "-c", 'printf %s "$SPIRE_SECRET_KEY"'],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_an_already_set_key_is_used_unchanged_and_writes_no_file(tmp_path):
    result = _run_entrypoint(tmp_path, {"SPIRE_SECRET_KEY": "already-set-value"})

    assert result.returncode == 0, result.stderr
    assert result.stdout == "already-set-value"
    assert not (tmp_path / "secret_key").exists()


def test_an_unset_key_with_no_file_present_generates_one_and_persists_it(tmp_path):
    result = _run_entrypoint(tmp_path)

    assert result.returncode == 0, result.stderr
    generated = result.stdout
    assert generated

    key_file = tmp_path / "secret_key"
    assert key_file.exists()
    assert key_file.read_text().strip() == generated

    mode = stat.S_IMODE(key_file.stat().st_mode)
    assert mode == 0o600, f"expected owner-only permissions, got {oct(mode)}"


def test_a_second_run_with_the_file_present_exports_the_same_value(tmp_path):
    """The assertion that matters most in the whole plan: a regenerated
    key on a later boot would invalidate every stored credential and every
    issued session at once, silently."""
    first = _run_entrypoint(tmp_path)
    assert first.returncode == 0, first.stderr

    second = _run_entrypoint(tmp_path)
    assert second.returncode == 0, second.stderr

    assert first.stdout == second.stdout
    assert first.stdout != ""


def test_the_generated_key_passes_the_applications_own_strength_validation(tmp_path, monkeypatch):
    result = _run_entrypoint(tmp_path)
    assert result.returncode == 0, result.stderr

    monkeypatch.setenv("SPIRE_SECRET_KEY", result.stdout)
    # Raises on anything short, malformed, or low-variety -- must not raise.
    validate_secret_key_strength(SecurityConfig())


def test_the_key_file_is_written_under_the_configured_data_root(tmp_path):
    nested = tmp_path / "a" / "nested" / "data" / "root"
    nested.mkdir(parents=True)

    result = _run_entrypoint(nested)

    assert result.returncode == 0, result.stderr
    assert (nested / "secret_key").exists()


def test_the_script_replaces_itself_rather_than_forking_a_child():
    """`exec "$@"` as the last line -- signals reach the wrapped process
    directly, rather than a shell sitting in between."""
    text = _ENTRYPOINT.read_text()
    assert text.rstrip().endswith('exec "$@"')


def test_the_script_fails_fast_on_an_unset_variable_or_a_failed_command():
    text = _ENTRYPOINT.read_text()
    assert "set -euo pipefail" in text


# --- Task 3: the Compose file's own variable contract -----------------------


def _referenced_config_vars() -> set[str]:
    """Every `${NAME}` config.example.yaml's own text names, outside a
    comment line -- the same rule config.py::expand_env applies, so this
    reads exactly what a real load would try to expand."""
    text = _CONFIG_EXAMPLE.read_text(encoding="utf-8")
    names: set[str] = set()
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        names.update(_PLACEHOLDER_RE.findall(line))
    return names


def test_compose_app_environment_names_every_variable_the_example_config_references():
    """A variable added to config.example.yaml later and forgotten here
    would silently break a clean-clone Compose start on the first missing
    variable (config.py::expand_env raises by name, never expands to
    an empty string) -- this test is what keeps that from being silent."""
    referenced = _referenced_config_vars()
    assert referenced, "expected config.example.yaml to reference at least one ${VAR}"

    compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
    app_environment = compose["services"]["app"]["environment"]

    missing = referenced - set(app_environment)
    assert not missing, (
        "docker-compose.yml's app service environment is missing variable(s) "
        f"config.example.yaml references: {sorted(missing)}"
    )


def test_compose_database_service_publishes_no_port():
    """D-14: the bundled database is reachable only on the Compose
    network -- its default password is bounded by that, not by secrecy of
    the password itself."""
    compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
    db_service = compose["services"]["db"]
    assert "ports" not in db_service, "the db service must not publish a port (D-14)"


def test_compose_app_port_is_bound_to_loopback_only():
    """The published port is bound to the host's own loopback address --
    the control this file's own comments say COOKIE_SECURE=false relies
    on."""
    compose = yaml.safe_load(_COMPOSE_FILE.read_text(encoding="utf-8"))
    app_ports = compose["services"]["app"]["ports"]
    assert all(str(p).startswith("127.0.0.1:") for p in app_ports), app_ports


# --- WR-07 (code review): the mounted volumes must be writable ----------


def test_the_image_pins_the_numeric_uid_and_gid_the_chart_names() -> None:
    """WR-07. The pod runs as the image's non-root `spire` user and
    mounts two ReadWriteOnce PVCs at /data and /models. Kubernetes knows
    nothing about an image's passwd file, so the chart has to name a
    numeric `fsGroup` to make those volumes writable -- and a number the
    chart states while the image lets `groupadd --system` pick its own is
    a drift waiting to happen. This asserts the two agree, from both
    sides, rather than asserting either one alone.
    """
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text()
    assert "--gid 1001 spire" in dockerfile
    assert "--uid 1001 --gid 1001" in dockerfile

    deployment = (
        _REPO_ROOT / "charts" / "spire-voice" / "templates" / "deployment.yaml"
    ).read_text()
    assert "fsGroup: 1001" in deployment
    assert "fsGroupChangePolicy: OnRootMismatch" in deployment


# --- WR-09 (code review): .env.example names what Compose reads ---------

_ENV_EXAMPLE = _REPO_ROOT / ".env.example"


def _env_example_names() -> "set[str]":
    names: set[str] = set()
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        names.add(stripped.split("=", 1)[0].strip())
    return names


def _compose_referenced_vars() -> "set[str]":
    """Every `${NAME}` docker-compose.yml substitutes, read off its own
    text -- comment lines skipped, the same rule
    `_referenced_config_vars` above applies to the configuration file, so
    a variable named only in prose is not counted as one Compose reads."""
    names: set[str] = set()
    for line in _COMPOSE_FILE.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        names.update(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", line))
    return names


def test_env_example_names_every_variable_docker_compose_reads() -> None:
    """WR-09. `docker-compose.yml` substitutes `${POSTGRES_PASSWORD:-changeme}`
    in two places and the runbook says to set it "in your `.env` before the
    first `docker compose up`" -- but `.env.example`, the file README step 1
    and the runbook both tell you to copy, never named it. An operator
    following the documented "copy it and fill in the values it names" path
    never learned the variable existed, and shipped `changeme` permanently,
    since Postgres reads that variable only while its data volume is empty.

    The mirror image of
    `test_compose_app_environment_names_every_variable_the_example_config_references`
    above, one layer out.
    """
    referenced = _compose_referenced_vars()
    assert referenced, "expected docker-compose.yml to substitute at least one ${VAR}"
    assert "POSTGRES_PASSWORD" in referenced

    missing = referenced - _env_example_names()
    assert not missing, (
        ".env.example does not name variable(s) docker-compose.yml reads: "
        f"{sorted(missing)} -- an operator who copies it and fills in what it names "
        "never learns these exist"
    )


def test_the_example_database_url_names_the_port_the_dev_script_actually_uses() -> None:
    """The second half of WR-09: `.env.example` pointed at 5432, the port
    every other Postgres on a development machine is already bound to, so
    a developer taking the example literally connected to an unrelated
    container and got an authentication error that reads like a bug in
    this project. The example and the script that starts the database
    must name the same port."""
    env_example = _ENV_EXAMPLE.read_text(encoding="utf-8")
    dev_script = (_REPO_ROOT / "scripts" / "dev-postgres.sh").read_text(encoding="utf-8")

    default_port = re.search(r'_PORT="\$\{SPIRE_DEV_POSTGRES_PORT:-(\d+)\}"', dev_script)
    assert default_port, "could not read the dev Postgres script's default port"
    port = default_port.group(1)

    assert port != "5432", "the dev database must not claim Postgres's own default port"
    assert f"127.0.0.1:{port}/spire" in env_example


# --- WR-10 (code review): the test stage is actually built -------------


def test_something_this_repository_ships_actually_builds_the_images_test_stage() -> None:
    """WR-10. The `test` stage calls itself "the actual proof" that this
    project's dependency set installs and its suite passes on the Python
    D-13 locks -- and nothing ever built it. `runtime` is `FROM
    python-base`, not `FROM test`; BuildKit builds only the stages the
    target needs; `docker build` and `docker compose build` default to
    the last stage; and no script, runbook or README line passed
    `--target test`. The first time it was ever built, it failed.

    `scripts/verify-clean-clone.sh` builds it now, and fails on it. This
    test is what keeps that from being quietly dropped again.
    """
    script = (_REPO_ROOT / "scripts" / "verify-clean-clone.sh").read_text(encoding="utf-8")
    assert "docker build --target test" in script

    build_line = next(
        line for line in script.splitlines() if "docker build --target test" in line
    )
    assert build_line.lstrip().startswith("if ! "), (
        "the test-stage build must be checked -- an unchecked `docker build` would "
        "repeat exactly the failure this finding is about"
    )


def test_the_test_stage_can_see_every_artifact_its_tests_read() -> None:
    """The reason that first build failed: four tests read deployment
    artifacts the stage did not copy in -- the chart's own templates and
    this Dockerfile. They are not skip-guarded (they need no `helm`
    binary), so their inputs being absent is a failure, not a skip."""
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    test_stage = dockerfile.split("AS test", 1)[1].split("AS runtime", 1)[0]
    for needed in (
        "charts/",
        "Dockerfile",
        "README.md",
        "docker-compose.yml",
        ".env.example",
        "deploy/",
    ):
        assert needed in test_stage, f"the test stage does not copy in {needed}"
