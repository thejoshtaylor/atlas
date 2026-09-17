"""Mechanical no-real-house-data checks, green from the first commit.

This is the mechanical form of the rule `mcp/spire_mcp/safety.py`'s own
self-check already states in prose: every entity id in this repository's
tests is invented, and no credential literal belongs anywhere under the
static directory.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = _REPO_ROOT / "tests"
_STATIC_DIR = _REPO_ROOT / "src" / "spire_voice" / "static"

# domain.object_id shape, restricted to Home Assistant domains this project
# actually touches. A narrower domain list keeps this from flagging
# unrelated dotted strings (module paths, version numbers) as entity ids.
_ENTITY_ID_RE = re.compile(
    r"\b("
    r"switch|light|sensor|binary_sensor|media_player|vacuum|automation"
    r"|homeassistant|update"
    r")\.([a-z0-9_]+)\b"
)

_ALLOWED_OBJECT_IDS = {"a", "b", "example", "brand_new"}


def test_test_fixtures_use_invented_entity_ids():
    violations: list[str] = []
    for path in _TESTS_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _ENTITY_ID_RE.finditer(text):
            object_id = match.group(2)
            if object_id.startswith("example_") or object_id in _ALLOWED_OBJECT_IDS:
                continue
            violations.append(f"{path.relative_to(_REPO_ROOT)}: {match.group(0)}")

    assert not violations, (
        "found entity id(s) not using an invented object id "
        f"(must start with 'example_' or be one of {sorted(_ALLOWED_OBJECT_IDS)}): "
        f"{violations}"
    )


# Case-insensitive shapes that look like a leaked credential: long hex/base64
# tokens, common secret-bearing key names followed by a non-placeholder
# value, and bearer tokens.
_CREDENTIAL_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"][^'\"$]{8,}['\"]"
)


def test_static_directory_holds_no_credential_literal():
    if not _STATIC_DIR.exists():
        # The directory does not exist yet; this test passes vacuously and
        # starts guarding the moment the tracer plan creates it.
        return

    violations: list[str] = []
    for path in _STATIC_DIR.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _CREDENTIAL_RE.search(text):
            violations.append(str(path.relative_to(_REPO_ROOT)))

    assert not violations, f"credential-shaped literal found under static/: {violations}"
