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
