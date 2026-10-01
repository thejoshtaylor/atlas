"""Tests for desktop/scripts/install.sh.

The script picks a signing identity by SHA-1 hash. These tests source the
script and call its functions with fake `security` and `codesign` programs
first on PATH, so no real keychain is read or changed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "desktop" / "scripts" / "install.sh"

_DEV_ID_HASH = "A1" * 20
_APPLE_DEV_HASH = "B2" * 20
_LOCAL_HASH = "C3" * 20
_GONE_HASH = "D4" * 20

_DEV_ID_LINE = f'  1) {_DEV_ID_HASH} "Developer ID Application: Test Org (TEAM000000)"'
_APPLE_DEV_LINE = f'  2) {_APPLE_DEV_HASH} "Apple Development: tester@example.test (TEAM000000)"'
_LOCAL_LINE = f'  3) {_LOCAL_HASH} "ATLAS Local Signing"'

_FAKE_SECURITY = """#!/bin/sh
if [ "$1" = "find-identity" ]; then
  printf '%s\\n' "$FAKE_IDENTITIES"
  exit 0
fi
exit 1
"""

_FAKE_CODESIGN = """#!/bin/sh
printf '%s\\n' "$FAKE_CODESIGN_OUTPUT"
"""


def _listing(*lines: str) -> str:
    count = len(lines)
    noun = "identity" if count == 1 else "identities"
    return "\n".join([*lines, f"     {count} valid {noun} found"])


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("security", _FAKE_SECURITY), ("codesign", _FAKE_CODESIGN)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    return bin_dir


def _run(
    function: str,
    fake_bin: Path,
    tmp_path: Path,
    *,
    identities: str = "",
    identity_file_text: str | None = None,
    sign_identity: str | None = None,
    codesign_output: str = "",
    args: str = "",
) -> subprocess.CompletedProcess[str]:
    identity_file = tmp_path / "signing-identity"
    if identity_file_text is not None:
        identity_file.write_text(identity_file_text)
    env = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "ATLAS_IDENTITY_FILE": str(identity_file),
        "FAKE_IDENTITIES": identities,
        "FAKE_CODESIGN_OUTPUT": codesign_output,
    }
    if sign_identity is not None:
        env["ATLAS_SIGN_IDENTITY"] = sign_identity
    return subprocess.run(
        ["/bin/bash", "-c", f'source "{_SCRIPT}"; {function} {args}'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _code_lines() -> list[str]:
    return [
        line
        for line in _SCRIPT.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]


def test_script_is_executable() -> None:
    assert os.access(_SCRIPT, os.X_OK)


def test_script_passes_bash_syntax_check() -> None:
    result = subprocess.run(
        ["/bin/bash", "-n", str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_script_passes_shellcheck() -> None:
    result = subprocess.run(
        ["shellcheck", str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout


def test_developer_id_wins_over_apple_development(fake_bin: Path, tmp_path: Path) -> None:
    result = _run(
        "pick_identity",
        fake_bin,
        tmp_path,
        identities=_listing(_APPLE_DEV_LINE, _DEV_ID_LINE),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _DEV_ID_HASH


def test_apple_development_when_it_is_the_only_apple_identity(
    fake_bin: Path, tmp_path: Path
) -> None:
    result = _run(
        "pick_identity",
        fake_bin,
        tmp_path,
        identities=_listing(_LOCAL_LINE, _APPLE_DEV_LINE),
    )
    assert result.stdout.strip() == _APPLE_DEV_HASH


def test_local_certificate_when_nothing_else_is_valid(fake_bin: Path, tmp_path: Path) -> None:
    result = _run("pick_identity", fake_bin, tmp_path, identities=_listing(_LOCAL_LINE))
    assert result.stdout.strip() == _LOCAL_HASH


def test_no_identity_prints_local_needed(fake_bin: Path, tmp_path: Path) -> None:
    result = _run("pick_identity", fake_bin, tmp_path, identities="     0 valid identities found")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LOCAL_NEEDED"


def test_env_identity_wins_over_developer_id(fake_bin: Path, tmp_path: Path) -> None:
    result = _run(
        "pick_identity",
        fake_bin,
        tmp_path,
        identities=_listing(_DEV_ID_LINE, _APPLE_DEV_LINE),
        sign_identity=_APPLE_DEV_HASH,
    )
    assert result.stdout.strip() == _APPLE_DEV_HASH


def test_env_identity_matches_an_exact_name(fake_bin: Path, tmp_path: Path) -> None:
    result = _run(
        "pick_identity",
        fake_bin,
        tmp_path,
        identities=_listing(_DEV_ID_LINE, _LOCAL_LINE),
        sign_identity="ATLAS Local Signing",
    )
    assert result.stdout.strip() == _LOCAL_HASH


def test_env_identity_that_matches_nothing_exits_non_zero(
    fake_bin: Path, tmp_path: Path
) -> None:
    result = _run(
        "pick_identity",
        fake_bin,
        tmp_path,
        identities=_listing(_DEV_ID_LINE),
        sign_identity=_GONE_HASH,
    )
    assert result.returncode == 3
    assert "ATLAS_SIGN_IDENTITY" in result.stderr


def test_identity_file_wins_over_developer_id(fake_bin: Path, tmp_path: Path) -> None:
    result = _run(
        "pick_identity",
        fake_bin,
        tmp_path,
        identities=_listing(_DEV_ID_LINE, _APPLE_DEV_LINE),
        identity_file_text=f"{_APPLE_DEV_HASH}\n",
    )
    assert result.stdout.strip() == _APPLE_DEV_HASH
    assert result.stderr == ""


def test_stale_identity_file_warns_and_falls_through(fake_bin: Path, tmp_path: Path) -> None:
    result = _run(
        "pick_identity",
        fake_bin,
        tmp_path,
        identities=_listing(_DEV_ID_LINE),
        identity_file_text=f"{_GONE_HASH}\n",
    )
    assert result.stdout.strip() == _DEV_ID_HASH
    assert "permissions" in result.stderr or "Accessibility" in result.stderr


def test_pick_identity_creates_nothing(fake_bin: Path, tmp_path: Path) -> None:
    _run("pick_identity", fake_bin, tmp_path, identities=_listing(_DEV_ID_LINE))
    assert not (tmp_path / "signing-identity").exists()


def test_bundle_id_comes_from_the_template(fake_bin: Path, tmp_path: Path) -> None:
    if not Path("/usr/libexec/PlistBuddy").exists():
        pytest.skip("PlistBuddy exists on macOS only")
    result = _run("bundle_id_from_template", fake_bin, tmp_path)
    assert result.stdout.strip() == "org.atlas-assistant.desktop"


def test_designated_requirement_strips_the_prefix(fake_bin: Path, tmp_path: Path) -> None:
    output = (
        "Executable=/Applications/ATLAS.app/Contents/MacOS/AtlasDesktop\n"
        'designated => identifier "x" and certificate leaf = H"ab"'
    )
    result = _run(
        "designated_requirement",
        fake_bin,
        tmp_path,
        codesign_output=output,
        args="/Applications/ATLAS.app",
    )
    assert result.stdout.strip() == 'identifier "x" and certificate leaf = H"ab"'


def test_script_uses_the_system_openssl_and_the_trust_step() -> None:
    text = "\n".join(_code_lines())
    assert "/usr/bin/openssl" in text
    assert "add-trusted-cert" in text
    assert "find-identity -v -p codesigning" in text


def test_script_never_signs_ad_hoc() -> None:
    sign_with_dash = re.compile(r"(--sign|\s-s)[ =]+(-|'-'|\"-\")(\s|$)")
    for line in _code_lines():
        assert not sign_with_dash.search(line), line


def test_script_has_no_sudo_and_no_transport_exception() -> None:
    text = "\n".join(_code_lines())
    assert "sudo" not in text
    assert "NSAllowsArbitraryLoads" not in text
