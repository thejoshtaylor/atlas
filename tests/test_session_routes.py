"""Session review over HTTP (D-01 .. D-04, WEB-07, 08-04-PLAN.md).

Same "primitives in isolation" shape `tests/test_provider_routes.py` already
uses: a throwaway `FastAPI()` app carrying only `routes/sessions.py`'s own
router, driven against a real filesystem (`tmp_path`) rather than a real
Postgres -- this module has no repository of its own to fake.

Every session directory this file needs is built through a real
`SessionRecorder`, constructed and closed exactly the way `turn/controller.py`
does, never hand-written -- so this test and the real recorder can never
drift about the on-disk format (this plan's own action text, Task 1).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig, SessionConfig
from spire_voice.routes.sessions import router as sessions_router
from spire_voice.session.recorder import SessionRecorder
from spire_voice.timing import TurnTimings

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


def _write_recorded_session(
    session_config: SessionConfig,
    *,
    turn_outcome: str = "completed",
    reply_text: "str | None" = "the reply",
    transcript_text: "str | None" = "the transcript",
    started_offset_days: float = 0.0,
    include_audio: bool = True,
) -> Path:
    """Build one real session directory through `SessionRecorder` --
    construct one and close it, exactly as `turn/controller.py` does,
    rather than hand-writing `events.jsonl`/`timing.json` ourselves."""
    timings = TurnTimings(turn_outcome=turn_outcome)
    timings.mark_turn_started()
    recorder = SessionRecorder(session_config, timings)
    recorder.set_audio_format("pcm", 8000)
    if transcript_text is not None:
        recorder.record_event({"type": "transcript.partial", "text": transcript_text})
    timings.mark_stt_final()
    if reply_text is not None:
        recorder.record_event({"type": "reply.text", "text": reply_text})
    timings.mark_first_audio()
    timings.mark_answer_audio()
    if include_audio:
        recorder.record_audio_chunk(b"\x00" * 16)
    recorder.close(timings)

    directory = recorder.directory
    if started_offset_days:
        # Rename to a deliberately older (or newer) timestamp, in the
        # exact `%Y%m%dT%H%M%S%fZ-{turn_id}` shape `recorder.py` itself
        # writes -- the only way to get deterministic ordering/retention
        # test fixtures out of a recorder whose own clock is real UTC
        # "now".
        target_at = datetime.now(timezone.utc) - timedelta(days=started_offset_days)
        stamp = target_at.strftime("%Y%m%dT%H%M%S%fZ")
        renamed = directory.parent / f"{stamp}-{timings.turn_id}"
        directory.rename(renamed)
        directory = renamed
    return directory


def _build_sessions_app(security: SecurityConfig, account_repo, session_config: SessionConfig) -> FastAPI:
    app = FastAPI()
    app.state.config = SimpleNamespace(security=security, session=session_config)
    app.state.account_repo = account_repo
    app.include_router(sessions_router)
    return app


def _client_with_role(app: FastAPI, security: SecurityConfig, account_repo, role: str) -> TestClient:
    user = asyncio.run(
        account_repo.create_user(
            email=f"{role}@example.invalid",
            display_name=role.title(),
            password_hash="not-checked-by-this-test",
            role=role,
        )
    )
    token = issue_access_token(user_id=user.id, role=role, security=security)
    return TestClient(app, cookies={security.cookie_name: token})


def test_one_recorded_directory_is_listed_to_an_operator(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get("/api/sessions")

    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert len(sessions) == 1
    session = sessions[0]
    assert session["id"] == directory.name
    assert session["turn_outcome"] == "completed"
    assert session["reply_text"] == "the reply"
    assert session["duration_ms"] is not None
    assert session["has_audio"] is True
    assert session["started_at"] is not None


def test_several_sessions_are_returned_newest_first(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    older = _write_recorded_session(session_config, started_offset_days=2)
    newer = _write_recorded_session(session_config, started_offset_days=0.5)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get("/api/sessions")

    ids = [entry["id"] for entry in response.json()["sessions"]]
    assert ids == [newer.name, older.name]


def test_a_directory_the_recorder_did_not_write_is_not_listed(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    _write_recorded_session(session_config)
    (tmp_path / "not-a-session-directory").mkdir()

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get("/api/sessions")

    assert len(response.json()["sessions"]) == 1


def test_a_viewer_is_refused_and_an_operator_succeeds(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    _write_recorded_session(session_config)

    app = _build_sessions_app(security, account_repo, session_config)
    viewer_client = _client_with_role(app, security, account_repo, "viewer")
    operator_client = _client_with_role(app, security, account_repo, "operator")

    assert viewer_client.get("/api/sessions").status_code == 403
    assert operator_client.get("/api/sessions").status_code == 200


def test_a_half_written_session_is_listed_with_a_named_outcome_not_dropped(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)
    (directory / "timing.json").unlink()

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get("/api/sessions")
    sessions = response.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["turn_outcome"] == "recording_incomplete"
    assert sessions[0]["duration_ms"] is None
    assert sessions[0]["has_audio"] is False
