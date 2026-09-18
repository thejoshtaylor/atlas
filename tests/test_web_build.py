"""The built frontend is a build artifact, not a tracked one, and its
absence must not be the thing that stops the whole application from
booting (WEB-08's build seam, the deployment hazard plan 03-05 wires).

`web/dist` (or whatever directory the Vite build emits to) is exactly the
kind of generated output this repository's `.gitignore` already excludes
categorically (`dist/`, `build/`, `.vite/`) -- but a *new* build step is
also exactly the kind of change that can accidentally commit its own output
once, if a build runs before the ignore rule is confirmed to actually cover
the real emitted path. The second test guards the boot-time side of the
same seam: `FastAPI.frontend(...)` (03-RESEARCH.md Pattern 3) reads from a
directory that will not exist yet on a fresh clone before anyone runs `bun
run build` -- the application must still start (API routes still work; the
frontend route answers 404 or an equivalent, not an unhandled startup
exception) rather than crashing lifespan entirely because a directory is
missing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# `web/dist` is `vite.config.ts`'s own `build.outDir` (a comment there names
# `app.frontend()` in `src/spire_voice/app.py` as the consumer plan 03-05
# wires -- the two must name the same path). Walking `git ls-files` rather
# than trusting `.gitignore` to be correct is the same posture
# `test_repo_hygiene.py`'s `test_no_session_path_or_audio_extension_is_tracked_by_git`
# already takes for exactly this class of mistake: a *new* build step is
# exactly the kind of change that can accidentally commit its own output
# once, before the ignore rule is confirmed to actually cover the real
# emitted path.
_BUILD_OUTPUT_PREFIX = "web/dist/"


def test_the_built_frontend_is_not_tracked_by_git():
    """No file under the frontend's build output directory may be tracked
    by git, regardless of whether a build has run in this environment --
    `git ls-files` reports what git actually tracks, not what merely
    exists on disk right now."""
    proc = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=_REPO_ROOT, timeout=30
    )
    assert proc.returncode == 0, f"git ls-files failed: {proc.stderr}"

    tracked_build_output = [
        line for line in proc.stdout.splitlines() if line.startswith(_BUILD_OUTPUT_PREFIX)
    ]

    assert not tracked_build_output, (
        "found tracked file(s) under the frontend build output directory "
        f"(WEB-08: it is a build artifact, never tracked): {tracked_build_output}"
    )


def test_a_missing_build_directory_does_not_stop_the_application():
    """`lifespan` must start successfully, and every non-frontend route
    must keep working, even when the frontend build directory does not
    exist on disk -- plan 03-05 fills this in."""
    raise AssertionError(
        "plan 03-05 fills this in (a missing frontend build directory must not stop the application)"
    )
