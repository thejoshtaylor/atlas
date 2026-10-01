"""Checks for docs/desktop.md, the guide to the ATLAS Mac app (D-26).

The guide is public and strangers copy its commands into a terminal. These
tests pin it to the real script, flags and bundle id. They also check that it
holds no identity material and gives no unsafe advice.
"""

from __future__ import annotations

import plistlib
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GUIDE = _REPO_ROOT / "docs" / "desktop.md"
_README = _REPO_ROOT / "README.md"
_TEMPLATE = _REPO_ROOT / "desktop" / "Resources" / "Info.plist.template"
_SCRIPT = _REPO_ROOT / "desktop" / "scripts" / "install.sh"

_REQUIRED_HEADINGS = (
    "Prerequisites",
    "Install",
    "Signing identity",
    "Permissions",
    "Pairing",
    "Troubleshooting",
    "Uninstall",
)


def _guide_text() -> str:
    assert _GUIDE.is_file(), "docs/desktop.md does not exist"
    return _GUIDE.read_text(encoding="utf-8")


def _bundle_id() -> str:
    # The template has a placeholder only in CFBundleVersion, so plistlib reads it.
    with _TEMPLATE.open("rb") as handle:
        return plistlib.load(handle)["CFBundleIdentifier"]


def _code_lines(text: str) -> list[str]:
    """Lines inside fenced blocks, plus every inline code span."""
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            lines.append(line)
        else:
            lines.extend(re.findall(r"`([^`]+)`", line))
    return lines


def _headings(text: str) -> list[str]:
    return [line.lstrip("#").strip() for line in text.splitlines() if re.match(r"^##\s", line)]


def test_guide_exists_and_readme_links_to_it() -> None:
    assert _GUIDE.is_file()
    readme = _README.read_text(encoding="utf-8")
    assert "(docs/desktop.md)" in readme


def test_guide_has_the_required_headings() -> None:
    headings = _headings(_guide_text())
    for wanted in _REQUIRED_HEADINGS:
        assert any(h == wanted for h in headings), f"missing heading: {wanted}"


def test_guide_names_the_real_script_flags_and_paths() -> None:
    text = _guide_text()
    script = _SCRIPT.read_text(encoding="utf-8")
    for name in (
        "desktop/scripts/install.sh",
        "install.sh --verify",
        "ATLAS_SIGN_IDENTITY",
        "desktop/.signing-identity",
        "/Applications/ATLAS.app",
    ):
        assert name in text, f"the guide does not name {name}"
    # Each name must also be real: the script has to know it.
    assert "--verify" in script
    assert "ATLAS_SIGN_IDENTITY" in script
    assert ".signing-identity" in script
    assert "/Applications/ATLAS.app" in script


def test_every_bundle_id_in_the_guide_matches_the_template() -> None:
    text = _guide_text()
    bundle_id = _bundle_id()
    assert bundle_id in text, "the guide never names the bundle id"
    prefix = bundle_id.rsplit(".", 1)[0] + "."
    for found in re.findall(re.escape(prefix) + r"[A-Za-z0-9.-]*", text):
        assert found.rstrip(".").startswith(bundle_id), f"wrong bundle id: {found}"
    # The uninstall commands use the real id for both Keychain services.
    assert f"-s {bundle_id}.pairing" in text
    assert f"-s {bundle_id}.diagnostics" in text
    assert f"tccutil reset Accessibility {bundle_id}" in text


def test_code_lines_give_no_unsafe_advice() -> None:
    code = "\n".join(_code_lines(_guide_text()))
    assert not re.search(r"(?:--sign|\s-s)\s+-(?:\s|$)", code), "ad-hoc signing in code"
    assert not re.search(r"\bsudo\b", code), "sudo in code"
    assert "NSAllowsArbitraryLoads" not in code
    assert "ws://" not in code


def test_guide_holds_no_identity_material() -> None:
    text = _guide_text()
    assert not re.search(r"[0-9a-fA-F]{40}", text), "a 40-hex identity hash"
    assert not re.search(
        r"""(?i)(token|secret|password)\s*[:=]\s*["'][^"'<>]{8,}["']""", text
    ), "a quoted credential literal"
    # A team id is ten uppercase letters or digits in parentheses after a name.
    assert not re.search(r"\([A-Z0-9]{10}\)", text), "a team id"
    # Every host in a URL or command is a placeholder.
    for host in re.findall(r"(?:https?|wss|atlas)://([^\s/?`)<>]+)", text):
        assert host.endswith("example.com") or host.endswith(".example") or host.startswith("<") or host in {
            "pair",
            "github.com",
        }, f"not a placeholder host: {host}"


def _section(text: str, heading: str) -> str:
    """The body of the `## heading` section, up to the next `## ` heading."""
    match = re.search(rf"^##\s+{re.escape(heading)}\s*$", text, re.MULTILINE)
    assert match, f"missing section: {heading}"
    rest = text[match.end() :]
    following = re.search(r"^##\s", rest, re.MULTILINE)
    return rest[: following.start()] if following else rest


def test_troubleshooting_names_the_recovery_tools() -> None:
    body = _section(_guide_text(), "Troubleshooting")
    for name in ("tccutil", "--diagnose", "launch.jsonl", "install.sh --verify"):
        assert name in body, f"Troubleshooting does not name {name}"


def test_troubleshooting_covers_the_signing_and_local_network_failures() -> None:
    body = _section(_guide_text(), "Troubleshooting")
    for phrase in (
        "Offline but the server is up",
        "Checking",
        "The certificate is lost",
        "codesign wants to sign using key",
        "internal certificate authority",
        "never skips certificate checks",
    ):
        assert phrase in body, f"Troubleshooting does not cover: {phrase}"


def test_rebuild_section_names_the_log_fields_and_the_verify_flag() -> None:
    body = _section(_guide_text(), "Check that permissions survive a rebuild")
    for name in ("ax_trusted", "keychain_pairing_status", "install.sh --verify", "three"):
        assert name in body, f"the rebuild section does not name {name}"


def test_log_fields_in_the_guide_exist_in_the_app() -> None:
    source = (_REPO_ROOT / "desktop" / "Sources" / "AtlasDesktop" / "LaunchDiagnostics.swift").read_text(
        encoding="utf-8"
    )
    for field in re.findall(r"`((?:ax_trusted|keychain_[a-z_]+|login_item_status))`", _guide_text()):
        assert f'"{field}"' in source, f"the app does not write {field}"
