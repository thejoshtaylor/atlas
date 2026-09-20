"""The built frontend is a build artifact, not a tracked one, and its
absence must not be the thing that stops the whole application from
booting (WEB-08's build seam, the deployment hazard plan 03-05 wires).

`web/dist` (or whatever directory the Vite build emits to) is exactly the
kind of generated output this repository's `.gitignore` already excludes
categorically (`dist/`, `build/`, `.vite/`) -- but a *new* build step is
also exactly the kind of change that can accidentally commit its own output
once, if a build runs before the ignore rule is confirmed to actually cover
the real emitted path. The remaining two tests guard the boot-time side of
the same seam, both booting the real `lifespan` (reusing `tests/
test_startup_smoke.py`'s fake builders): with the build directory absent,
the application must still start and every non-frontend route must keep
working, with a named warning logged rather than a silent blank page; with
it present, `GET /` must serve the built document while `GET /transport`
(an ordinary API route) still wins over the frontend's own fallback,
proving `app.frontend()`'s (03-RESEARCH.md Pattern 3) low-priority
ordering guarantee rather than assuming it.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest
import test_startup_smoke as smoke
from fastapi.testclient import TestClient

import spire_voice.app as app_module

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The Docker build context deliberately excludes .git (this repository's own
# .dockerignore, DEP-05's own build-layer exclusion) -- `git ls-files` has
# nothing to read there, so this check skips rather than reporting a false
# failure.
skip_without_git_dir = pytest.mark.skipif(
    not (_REPO_ROOT / ".git").exists(),
    reason="no .git directory in this checkout -- the build context excludes it "
    "by design (see .dockerignore), so there is no tracked-file list to check",
)


def _apply_smoke_monkeypatches(tmp_path: Path, monkeypatch) -> None:
    """The same fake-everything wiring `tests/test_auth_setup.py` and
    `tests/test_calibration_runner.py` already reuse from
    `test_startup_smoke.py`, so this file's own tests boot the real
    `lifespan`, not a unit call, matching this plan's own acceptance
    criteria for both tests below."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", smoke._TEST_SECRET_KEY)
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(smoke._write_fake_config(tmp_path)))
    monkeypatch.setattr(smoke.plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", smoke._fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", smoke._fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", smoke._FakeCameraSource)

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


@skip_without_git_dir
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


def test_a_missing_build_directory_does_not_stop_the_application(tmp_path, monkeypatch, caplog):
    """`lifespan` must start successfully, and every non-frontend route
    must keep working, even when the frontend build directory does not
    exist on disk -- and startup must log a warning naming the directory
    and the command that builds it, not stay silent about a blank page."""
    _apply_smoke_monkeypatches(tmp_path, monkeypatch)

    # The frontend route binds to FRONTEND_DIR's absolute path at import time
    # on the one module-level `app` singleton, so there is no per-test way to
    # redirect it -- the sibling test below says the same thing from the other
    # direction. So genuinely absent means genuinely absent: move a real build
    # aside for the duration and put it back afterwards, exactly as that
    # sibling does in reverse. Asserting the directory is missing instead would
    # make this test fail for any developer who has run `bun run build`, which
    # is every developer who has run the application.
    frontend_dir = app_module.FRONTEND_DIR
    stashed = frontend_dir.parent / f"{frontend_dir.name}.stashed-by-test"
    had_build = frontend_dir.exists()
    if had_build:
        if stashed.exists():
            shutil.rmtree(stashed)
        frontend_dir.rename(stashed)
    try:
        assert not frontend_dir.exists()

        with caplog.at_level(logging.WARNING, logger="spire_voice.app"):
            with TestClient(app_module.app) as client:
                response = client.get("/health")
                assert response.status_code == 200
    finally:
        if had_build:
            if frontend_dir.exists():
                shutil.rmtree(frontend_dir)
            stashed.rename(frontend_dir)

    warning_messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        str(app_module.FRONTEND_DIR) in message and "bun run build" in message
        for message in warning_messages
    ), (
        "no startup warning named both the missing frontend directory and the "
        f"build command -- warnings logged: {warning_messages!r}"
    )


def test_a_present_build_directory_serves_the_built_index_and_the_api_still_wins(
    tmp_path, monkeypatch
):
    """With the build output present, `GET /` returns the built index
    document and `GET /transport` still returns the transport -- in the
    same test, so the "an ordinary route is checked before the frontend
    fallback" guarantee (03-RESEARCH.md Pattern 3) is exercised, not
    assumed.

    Writes into the real `FRONTEND_DIR` (the frontend route is a low-priority
    route on the one module-level `app` singleton, bound to that exact
    absolute path at import time -- there is no per-test way to redirect it)
    and restores whatever was there beforehand, so a developer's own local
    `bun run build` output is never lost to this test run.
    """
    _apply_smoke_monkeypatches(tmp_path, monkeypatch)

    frontend_dir = app_module.FRONTEND_DIR
    backup_dir = frontend_dir.with_name(frontend_dir.name + ".test-backup")
    preexisting = frontend_dir.exists()
    if preexisting:
        frontend_dir.rename(backup_dir)
    frontend_dir.mkdir(parents=True)
    marker = "spire-voice-test-web-build-marker"
    (frontend_dir / "index.html").write_text(
        f"<!doctype html><title>{marker}</title>", encoding="utf-8"
    )

    try:
        with TestClient(app_module.app) as client:
            index_response = client.get("/")
            assert index_response.status_code == 200
            assert marker in index_response.text

            transport_response = client.get("/transport")
            assert transport_response.status_code == 200
            assert "transport" in transport_response.json()
    finally:
        import shutil

        shutil.rmtree(frontend_dir, ignore_errors=True)
        if preexisting:
            backup_dir.rename(frontend_dir)
