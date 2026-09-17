"""Mechanical no-real-house-data checks, green from the first commit.

This is the mechanical form of the rule `mcp/spire_mcp/safety.py`'s own
self-check already states in prose: every entity id in this repository is
invented, and no credential literal belongs anywhere in the tree. Both
checks walk the whole repository (minus build/dependency/VCS directories
that never ship in the public repo either way) -- WR-01 (phase 01 code
review): scoping this to `tests/`/`static/` alone left every other file
(`mcp/`, the rest of `src/spire_voice/`, `config/*.yaml`) unguarded, which
is exactly where a real entity id or a hardcoded credential is most likely
to land by accident in a later phase.
"""

from __future__ import annotations

import os
import subprocess
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that never ship in the public repository (VCS internals, build
# output, dependency trees, caches, machine-local run state) or that are
# already covered by their own dedicated ignore rule -- walking into them
# wastes time at best and risks flagging a vendored dependency's own secrets
# fixture at worst. Kept as directory *names*, not paths, since none of them
# are expected to nest meaningfully deep.
_EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".vite",
    "models",
    "data",
    "sessions",
    ".planning",
    ".gsd",
    ".claude",
    ".idea",
    ".vscode",
}


def _iter_repo_files(suffixes: set[str] | None = None) -> list[Path]:
    """Every file under the repo root, pruning `_EXCLUDED_DIR_NAMES` as it
    walks rather than filtering after the fact -- `os.walk` lets a directory
    be skipped without ever descending into it, which matters for `.git/`'s
    size alone. `suffixes=None` yields every file regardless of extension.
    """
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIR_NAMES]
        for filename in filenames:
            if suffixes is None or Path(filename).suffix in suffixes:
                matches.append(Path(dirpath) / filename)
    return _drop_gitignored(matches)


def _drop_gitignored(paths: list[Path]) -> list[Path]:
    """Remove paths git will never track.

    The guarantee this module enforces is that no real house data reaches the
    REPOSITORY, which is public. A gitignored path cannot reach it. Several
    files exist precisely to hold that data locally -- `.env`,
    `config/*.local.yaml`, `.planning/` -- and scanning them reports the
    system working as designed as though it were a violation.

    Scope is narrowed to exactly what git would track, and no further. If
    `git check-ignore` cannot be run at all, every path is kept: a scan that
    cannot determine what is ignored must over-report, never under-report.
    """
    if not paths:
        return paths
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            input="\n".join(str(p) for p in paths),
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return paths
    # Exit 0 = some ignored, 1 = none ignored, anything else = it could not
    # tell, so keep everything rather than silently narrowing the scan.
    if proc.returncode not in (0, 1):
        return paths
    ignored = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return [p for p in paths if str(p) not in ignored]


# domain.object_id shape, restricted to Home Assistant domains this project
# actually touches. A narrower domain list keeps this from flagging
# unrelated dotted strings (module paths, version numbers) as entity ids.
_ENTITY_ID_RE = re.compile(
    r"\b("
    r"switch|light|sensor|binary_sensor|media_player|vacuum|automation"
    r"|homeassistant|update"
    r")\.([a-z0-9_]+)\b"
)

# `_example`-prefixed and a short allowlist of literal object ids used as
# minimal placeholders (`a`, `b`, `example`, `brand_new`), plus the Home
# Assistant service-call verbs that share the same `domain.name` shape as an
# entity id but name an action, not a house fixture (e.g. `light.turn_off`
# in a comment describing what an untargeted service call would do).
_ALLOWED_OBJECT_IDS = {"a", "b", "example", "brand_new", "turn_on", "turn_off", "toggle"}


def test_test_fixtures_use_invented_entity_ids():
    violations: list[str] = []
    for path in _iter_repo_files({".py", ".yaml", ".yml"}):
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
# value, and bearer tokens. The value itself is captured (group 2), so an
# invented test placeholder can be told apart from a real leaked value.
_CREDENTIAL_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]([^'\"$]{8,})['\"]"
)

# The one placeholder value this codebase's own test fixtures already use
# for every provider config's `api_key` (see `tests/test_providers.py`,
# `tests/conftest.py`) -- obviously invented, not a credential shape at all,
# and allowlisting it here (rather than narrowing which files get scanned)
# is what let this check widen to the whole repository without flagging its
# own test suite's existing, pre-phase convention as a finding.
_ALLOWED_CREDENTIAL_VALUES = {"test-key"}


def test_repository_holds_no_credential_literal():
    violations: list[str] = []
    for path in _iter_repo_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _CREDENTIAL_RE.finditer(text):
            if match.group(2) in _ALLOWED_CREDENTIAL_VALUES:
                continue
            violations.append(f"{path.relative_to(_REPO_ROOT)}: {match.group(0)}")
            break

    assert not violations, f"credential-shaped literal found in the repository: {violations}"
