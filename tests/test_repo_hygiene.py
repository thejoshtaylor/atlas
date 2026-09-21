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

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The Docker build context deliberately excludes .git (this repository's own
# .dockerignore, DEP-05's own build-layer exclusion) -- a check that reads
# git's own tracked-file list has nothing to read there, and skipping it is
# the honest outcome rather than a false failure. Every other check in this
# file walks the filesystem directly (`_iter_repo_files`) and already
# tolerates a missing git binary through `_drop_gitignored`'s own fallback.
skip_without_git_dir = pytest.mark.skipif(
    not (_REPO_ROOT / ".git").exists(),
    reason="no .git directory in this checkout -- the build context excludes it "
    "by design (see .dockerignore), so there is no tracked-file list to check",
)

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
_ALLOWED_OBJECT_IDS = {
    "a",
    "b",
    "example",
    "brand_new",
    "turn_on",
    "turn_off",
    "toggle",
    # IN-04 (code review), historical. Widening this scan to `.md` (below)
    # made two object ids in `docs/runbooks/ha-registry-expansion.md`'s own
    # illustrative output visible for the first time: an office lamp and an
    # office fan, both written without the mandated `example_` prefix. The
    # working tree is corrected -- both now carry the prefix -- but the
    # history scan reads blobs as earlier commits wrote them, and those
    # cannot be corrected without rewriting history.
    #
    # They are allowlisted rather than reported as a leak because they are
    # plainly invented: they appear inside a fenced example of what the
    # expansion command PRINTS, in a document explaining how to read that
    # output, alongside a literal `...` continuation -- never in any
    # configuration, fixture or assertion. `_ENTITY_ID_RE` itself is
    # unchanged, so a real house's entity id of the same shape is still
    # caught, in the working tree and in history alike.
    "office_lamp",
    "office_fan",
}

# The file suffixes the entity-id convention is actually enforced against --
# named once here so the history scan below reuses the exact same scope
# rather than restating it. Deliberately narrower than the credential-literal
# scan's scope (which covers every file): a bare `domain.name` shape is
# common, unrelated syntax in other languages (JS/TS property access chains,
# for one), and this narrower scope is what keeps that noise out, in the
# working tree and in history alike.
#
# IN-04 (code review): `.md` and `.sh` are in scope now. Phase 7's new
# public surfaces are mostly Markdown (README.md, three runbooks) and
# shell, and none of it was scanned -- and the gap was not theoretical:
# widening this set immediately found two unprefixed object ids in
# `docs/runbooks/ha-registry-expansion.md` (see `_ALLOWED_OBJECT_IDS`
# above for both, and for why they are allowlisted in history rather than
# reported). Prose is where a real house's entity id is most likely to be
# pasted straight out of a terminal, which is the opposite of an argument
# for leaving it unscanned. The JS/TS property-access noise the comment
# above names is a `.ts`/`.tsx` concern, not a Markdown one.
_ENTITY_ID_SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".sh"}


def test_test_fixtures_use_invented_entity_ids():
    violations: list[str] = []
    for path in _iter_repo_files(_ENTITY_ID_SCAN_SUFFIXES):
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
#
# The eight entries below are historical (deferred-items.md #2, operator
# decision): an earlier `CredentialSlot` (blob `00dc28b2baec247796b18e4eac
# cc5c19a1e0c834`, reachable from several commits including `a2a19e2`,
# `d834023`, `a66ee32`) declared its enum members as `STT_API_KEY =
# "stt_api_key"`, `BRAIN_API_KEY = "brain_api_key"`, `TTS_API_KEY =
# "tts_api_key"`, `HA_TOKEN = "ha_token"`, plus a display-name mapping with
# values `"Speech-to-text API key"`, `"Language model API key"`,
# `"Text-to-speech API key"`, `"Home Assistant token"`. All eight match
# `_CREDENTIAL_RE`'s shape (a name containing `api_key`/`token` immediately
# before `=`) but are Python enum-member identifiers and human-readable UI
# labels, never a real provider key or house-specific value -- confirmed by
# the *current* `crypto/credentials.py`'s own docstring, which explains
# exactly why later member names deliberately avoid this shape. The
# operator reviewed deferred-items.md #2 and approved allowlisting these
# eight exact literals; `_CREDENTIAL_RE` itself is unchanged, so a real
# leaked credential of the same shape is still caught.
_ALLOWED_CREDENTIAL_VALUES = {
    "test-key",
    "stt_api_key",
    "brain_api_key",
    "tts_api_key",
    "ha_token",
    "Speech-to-text API key",
    "Language model API key",
    "Text-to-speech API key",
    "Home Assistant token",
}


# DEP-05's second clause -- the history, not only the working tree. A real
# secret or entity id is never a multi-megabyte blob in this project; the
# cap bounds runtime against an accidentally-committed large binary rather
# than trying to scan it as text.
_MAX_HISTORY_BLOB_BYTES = 5 * 1024 * 1024


def _iter_reachable_blob_paths() -> "tuple[set[str], dict[str, str]]":
    """Every commit `git rev-list --all` finds reachable from any
    reference, and every blob object reachable through any of their trees,
    each paired with the first path `git rev-list --objects --all`
    encountered it under.

    `git rev-list --objects --all` walks every reachable commit's own
    tree as it looked at that point in history -- not only the current
    tip's tree -- which is exactly what lets a blob a later commit deleted
    still turn up here: an earlier, still-reachable commit's own tree still
    names it. Commit and tag objects have no path in this output and are
    dropped; trees also carry a path but are filtered out downstream, by
    `_batch_read_blobs`, once their real object type is known.
    """
    commits_proc = subprocess.run(
        ["git", "rev-list", "--all"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=120,
    )
    assert commits_proc.returncode == 0, f"git rev-list --all failed: {commits_proc.stderr}"
    commits_scanned = {line.strip() for line in commits_proc.stdout.splitlines() if line.strip()}

    objects_proc = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=120,
    )
    assert (
        objects_proc.returncode == 0
    ), f"git rev-list --objects --all failed: {objects_proc.stderr}"

    blob_paths: dict[str, str] = {}
    for line in objects_proc.stdout.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue  # a commit/tag line -- no path, nothing to scan by path
        sha, path = parts
        blob_paths[sha] = path

    return commits_scanned, blob_paths


def _batch_read_blobs(shas: "set[str]") -> "dict[str, bytes]":
    """The real content of every blob in `shas`, in one `git cat-file
    --batch` process rather than one subprocess per object -- the only way
    reading a few thousand small objects stays fast. Non-blob objects
    (trees, which also carry paths in `_iter_reachable_blob_paths`'
    output) are silently excluded, not raised on -- they were never blobs
    to scan as text in the first place.
    """
    if not shas:
        return {}
    sha_list = sorted(shas)
    proc = subprocess.run(
        ["git", "cat-file", "--batch"],
        input=("\n".join(sha_list) + "\n").encode(),
        capture_output=True,
        cwd=_REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, f"git cat-file --batch failed: {proc.stderr!r}"

    out = proc.stdout
    result: dict[str, bytes] = {}
    pos = 0
    for _ in sha_list:
        newline_idx = out.index(b"\n", pos)
        header = out[pos:newline_idx].decode("ascii", errors="replace")
        pos = newline_idx + 1
        parts = header.split()
        if len(parts) < 2 or parts[-1] == "missing":
            continue
        obj_sha, obj_type = parts[0], parts[1]
        obj_size = int(parts[2])
        content = out[pos : pos + obj_size]
        pos += obj_size + 1  # skip the trailing newline git appends after each object
        if obj_type == "blob":
            result[obj_sha] = content
    return result


def _commits_touching_path(path: str) -> "list[str]":
    """Candidate commits for a violation's report -- only ever called on
    the (expected-empty) failure path, since a clean scan never needs
    per-match commit attribution. Not a claim of the exact introducing
    commit: `--follow` is deliberately omitted (a rename could point this
    at the wrong history segment) -- naming a handful of commits that
    touched this path is what makes a finding actionable for an operator,
    who can run `git log -p -- {path}` themselves from here."""
    proc = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", path],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


# The new artifact classes Phase 7's deployment work introduces (DEP-05):
# a Compose file, an example dotenv, and an entrypoint script are all new
# surfaces a real secret could reach that did not exist before this phase.
_DEPLOYMENT_ARTIFACTS = (
    _REPO_ROOT / "docker-compose.yml",
    _REPO_ROOT / ".env.example",
    _REPO_ROOT / "deploy" / "docker-entrypoint.sh",
)


def test_deployment_artifacts_carry_no_credential_literal():
    """Same shape as `test_repository_holds_no_credential_literal` below,
    checked explicitly against the three deployment artifacts this phase
    adds -- the full-repository scan already covers them once they are
    tracked, but naming them here makes the DEP-05 coverage explicit
    rather than incidental."""
    violations: list[str] = []
    for path in _DEPLOYMENT_ARTIFACTS:
        assert path.exists(), f"{path} does not exist"
        text = path.read_text(encoding="utf-8")
        for match in _CREDENTIAL_RE.finditer(text):
            if match.group(2) in _ALLOWED_CREDENTIAL_VALUES:
                continue
            violations.append(f"{path.relative_to(_REPO_ROOT)}: {match.group(0)}")

    assert not violations, f"credential-shaped literal found in a deployment artifact: {violations}"


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


# Directory names a real session's audio, events, and timeline live under
# (CONTEXT.md D-16). `.gitignore` already carries both, and
# `_EXCLUDED_DIR_NAMES` above already prunes both from `_iter_repo_files` --
# which is exactly why this check reads `git ls-files` directly instead:
# a directory `_iter_repo_files` never walks into can never surface a
# violation through it.
#
# `"calibration"` is deliberately NOT in this set: plan 02-10 adds a
# legitimately tracked `src/spire_voice/calibration/` source package, and a
# generic directory-name check here would flag that code as though it were
# the runtime data directory. `test_calibration_directory_default_location_
# is_gitignored` below checks the actual runtime path
# (`calibration.record.DEFAULT_CALIBRATION_DIR`) instead, which is the
# artifact T-02-43 is actually about.
#
# Checked against the tracked path's own FIRST segment only (the repo-root
# anchoring `.gitignore`'s own `/data/`/`/sessions/` patterns use, Phase 8
# 08-04-PLAN.md) -- not every directory segment. An unanchored check here
# had the identical bug this plan found and fixed in `.gitignore` itself:
# `web/src/routes/sessions/` (Phase 8, WEB-07) is a legitimate, tracked
# source directory named "sessions" nested well below the repo root, and a
# check over every path segment flagged it as though it were the runtime
# data root.
_SESSION_DIR_NAMES = {"data", "sessions"}

# Session audio and the TTS cache both write raw codec bytes under these
# extensions -- `.raw` is `tts_cache.py`'s own choice; `.alaw` is what
# CONTEXT.md names for the camera's raw capture. Neither belongs in git.
_AUDIO_EXTENSIONS = {".raw", ".alaw"}


@skip_without_git_dir
def test_no_session_path_or_audio_extension_is_tracked_by_git():
    """D-16: the data root is gitignored before the first write, in that
    order -- checked here against what git actually tracks, before this
    phase's session recorder ever writes a real session.
    """
    proc = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=_REPO_ROOT, timeout=30
    )
    assert proc.returncode == 0, f"git ls-files failed: {proc.stderr}"

    violations = [
        tracked
        for tracked in proc.stdout.splitlines()
        if tracked.strip()
        and (
            Path(tracked).suffix in _AUDIO_EXTENSIONS
            or Path(tracked).parts[0] in _SESSION_DIR_NAMES
        )
    ]

    assert not violations, (
        "found a tracked session path or a tracked captured-audio file "
        f"(D-16 -- the data root must be gitignored before the first write): {violations}"
    )


@skip_without_git_dir
def test_calibration_directory_default_location_is_gitignored():
    """T-02-43: `calibration.record.DEFAULT_CALIBRATION_DIR` falls under
    the repository's existing ignore rules, checked directly with
    `git check-ignore` rather than assumed from `_SESSION_DIR_NAMES` alone
    -- this is the ordering plan 02-01 used for the session store, applied
    before this plan's calibration record is ever written for real.
    """
    from spire_voice.calibration.record import DEFAULT_CALIBRATION_DIR

    candidate = Path(DEFAULT_CALIBRATION_DIR.lstrip("/")) / "echo_path.json"
    proc = subprocess.run(
        ["git", "check-ignore", "--quiet", str(candidate)],
        cwd=_REPO_ROOT,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"{candidate} is not covered by the repository's ignore rules -- "
        "a calibration record written to the default directory would be trackable by git"
    )


# DEP-05's second clause: no entity id or credential-shaped literal survives
# anywhere in the repository's HISTORY, not only its current working tree.
# The tests above already prove the working-tree half; a deleted secret is
# still reachable through git's object store, which is exactly what a
# working-tree walk can never see.
@skip_without_git_dir
def test_repository_history_carries_no_entity_id_or_credential_literal():
    """A scan over every commit `git rev-list --all` finds reachable, and
    every blob object `git rev-list --objects --all` finds reachable
    through any of their trees -- reusing `_ENTITY_ID_RE`/
    `_ALLOWED_OBJECT_IDS`/`_CREDENTIAL_RE`/`_ALLOWED_CREDENTIAL_VALUES`
    unchanged, so a change to either convention reaches the working-tree
    check and this history check at once, never two copies to keep in
    sync.

    `git rev-list --objects --all` is what closes the second clause:  it
    walks every reachable commit's own tree as that commit itself looked,
    not only the current tip's tree, so a blob a later commit deleted is
    still emitted here as long as some earlier, still-reachable commit's
    tree names it. Asserts on what it scanned (commit count, blob count)
    so a scan that silently found nothing to look at fails loudly instead
    of passing green.
    """
    commits_scanned, blob_paths = _iter_reachable_blob_paths()
    assert commits_scanned, (
        "git rev-list --all found no reachable commits -- the scan ran against an empty "
        "or unreadable history, which is a vacuous pass, not a clean one"
    )
    assert blob_paths, (
        "git rev-list --objects --all found no path-bearing objects -- the scan ran "
        "against nothing"
    )

    blobs = _batch_read_blobs(set(blob_paths))
    assert blobs, "git cat-file --batch returned no blob content despite a non-empty object list"

    scanned_text_blobs = 0
    violations: list[str] = []
    for blob_sha, content in blobs.items():
        if len(content) > _MAX_HISTORY_BLOB_BYTES:
            continue
        # git's own binary heuristic: a NUL byte in the first 8000 bytes --
        # restricts this scan to text, the same convention git itself uses
        # to decide whether to show a diff at all.
        if b"\x00" in content[:8000]:
            continue
        text = content.decode("utf-8", errors="ignore")
        if not text:
            continue
        scanned_text_blobs += 1
        path = blob_paths[blob_sha]

        # Same scope split the working-tree tests already use: the entity-id
        # convention is enforced over _ENTITY_ID_SCAN_SUFFIXES only, the
        # credential-literal scan over every file -- reused here unchanged
        # rather than widened to a second, broader scan surface no existing
        # check has ever policed.
        if Path(path).suffix in _ENTITY_ID_SCAN_SUFFIXES:
            for match in _ENTITY_ID_RE.finditer(text):
                object_id = match.group(2)
                if object_id.startswith("example_") or object_id in _ALLOWED_OBJECT_IDS:
                    continue
                commits = _commits_touching_path(path)
                violations.append(
                    f"{path} (blob {blob_sha}, commit(s) {commits[:3] or ['unknown']}): "
                    f"{match.group(0)} (entity id)"
                )

        for match in _CREDENTIAL_RE.finditer(text):
            if match.group(2) in _ALLOWED_CREDENTIAL_VALUES:
                continue
            commits = _commits_touching_path(path)
            violations.append(
                f"{path} (blob {blob_sha}, commit(s) {commits[:3] or ['unknown']}): "
                f"{match.group(0)} (credential-shaped literal)"
            )

    assert scanned_text_blobs > 0, (
        "found no text content among the scanned history blobs -- the scan ran against "
        "nothing readable"
    )

    assert not violations, (
        f"scanned {len(commits_scanned)} reachable commit(s) and {len(blobs)} unique blob "
        f"object(s) ({scanned_text_blobs} as text) from the repository's history: found "
        "entity id or credential-shaped literal(s) outside the invented-object-id "
        "convention and the existing allowlist this module already enforces on the "
        "working tree. This is a real finding to report to the operator, not a pattern "
        f"to weaken, an allowlist to widen, or history to rewrite: {violations}"
    )


# --- IN-04 (code review): the entity-id scan covers prose and shell ------


def test_the_entity_id_scan_covers_the_surfaces_this_project_publishes():
    """IN-04. The scan covered `.py`/`.yaml`/`.yml` only, while this
    phase's new public surfaces are mostly Markdown (README.md, three
    runbooks) and shell. Markdown is where a real house's entity id is
    most likely to end up, pasted straight out of a terminal into a
    document -- the opposite of an argument for leaving it out.

    Pinned as a set membership rather than inferred from a scan result:
    a suffix quietly dropped from the scope would otherwise show up as
    nothing at all.
    """
    assert {".py", ".yaml", ".yml", ".md", ".sh"} <= _ENTITY_ID_SCAN_SUFFIXES
    assert _REPO_ROOT / "README.md" in set(_iter_repo_files(_ENTITY_ID_SCAN_SUFFIXES))


def test_the_wake_events_migration_and_module_are_covered_by_the_repository_scan():
    """Plan 08-03 Task 3: this file scans by directory walk
    (`_iter_repo_files`), pruning only `_EXCLUDED_DIR_NAMES` -- neither
    `alembic/` nor `src/spire_voice/db/` is in that set, so the new
    `wake_events` migration and the modified `db/` modules are already
    inside both the credential-literal scan (every file) and the
    entity-id scan (`.py` is in `_ENTITY_ID_SCAN_SUFFIXES`) with no list
    to extend. Asserted here rather than silently assumed, so a future
    change narrowing `_iter_repo_files`'s scope would fail this test
    rather than quietly stop scanning these surfaces."""
    scanned_paths = set(_iter_repo_files())
    scanned_entity_id_paths = set(_iter_repo_files(_ENTITY_ID_SCAN_SUFFIXES))

    migration = _REPO_ROOT / "alembic" / "versions" / "0012_wake_events.py"
    module = _REPO_ROOT / "src" / "spire_voice" / "db" / "models.py"

    assert migration in scanned_paths
    assert module in scanned_paths
    assert migration in scanned_entity_id_paths
    assert module in scanned_entity_id_paths


def test_the_runbook_entity_ids_follow_the_invented_object_id_convention():
    """The two the widening found, corrected in the working tree. The
    allowlist entries exist for the history scan alone, which reads blobs
    as earlier commits wrote them -- this asserts the document itself no
    longer needs them."""
    runbook = (_REPO_ROOT / "docs" / "runbooks" / "ha-registry-expansion.md").read_text(
        encoding="utf-8"
    )
    for match in _ENTITY_ID_RE.finditer(runbook):
        object_id = match.group(2)
        assert object_id.startswith("example_") or object_id in {
            "turn_on",
            "turn_off",
            "toggle",
        }, f"{match.group(0)} in the runbook does not use an invented object id"
