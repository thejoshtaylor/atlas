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


def test_the_built_frontend_is_not_tracked_by_git():
    """No file under the frontend's build output directory may be tracked
    by git -- plan 03-03 fills this in, once that directory exists."""
    raise AssertionError(
        "plan 03-03 fills this in (WEB-08: the built frontend is a build artifact, never tracked)"
    )


def test_a_missing_build_directory_does_not_stop_the_application():
    """`lifespan` must start successfully, and every non-frontend route
    must keep working, even when the frontend build directory does not
    exist on disk -- plan 03-05 fills this in."""
    raise AssertionError(
        "plan 03-05 fills this in (a missing frontend build directory must not stop the application)"
    )
