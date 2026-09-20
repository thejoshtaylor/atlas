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
import stat
import subprocess
from pathlib import Path

from spire_voice.auth.tokens import validate_secret_key_strength
from spire_voice.config import SecurityConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENTRYPOINT = _REPO_ROOT / "deploy" / "docker-entrypoint.sh"


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
