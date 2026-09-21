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
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig, SessionConfig
from spire_voice.routes.sessions import _resolve_readable_session_directory
from spire_voice.routes.sessions import router as sessions_router
from spire_voice.session.recorder import EVENTS_FILENAME, SessionRecorder
from spire_voice.session.retention import parse_session_timestamp
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



def test_a_session_past_the_retention_window_is_not_listed_at_all(tmp_path, fake_account_repository):
    """The sweep only runs every `expiry_interval_s`, so a directory past
    `retain_days` sits on disk between two runs. Listing it advertises a
    row whose detail route says it was removed -- while it is still there
    (WR-02). One predicate, asked by both routes."""
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path), retain_days=7)
    inside = _write_recorded_session(session_config, started_offset_days=1)
    past = _write_recorded_session(session_config, started_offset_days=30)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    listed = client.get("/api/sessions").json()["sessions"]

    assert [session["id"] for session in listed] == [inside.name]
    # Still on disk -- the sweep has not reached it. The list is silent
    # about it because the detail route is about to refuse it, not because
    # anything deleted it.
    assert past.is_dir()
    assert client.get(f"/api/sessions/{past.name}").status_code == 404


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


def test_one_truncated_event_line_does_not_take_out_the_whole_list(tmp_path, fake_account_repository):
    """A turn killed mid-`record_event` leaves exactly one partial line in
    its own `events.jsonl`. That line must cost that session the events it
    could not read and nothing more -- not the other sessions, and not the
    list route (CR-01)."""
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    intact = _write_recorded_session(session_config, reply_text="the intact reply")
    damaged = _write_recorded_session(
        session_config, reply_text="the reply before the crash", started_offset_days=1
    )
    # Byte-for-byte what an interrupted append leaves behind: a line the
    # recorder started writing and never terminated.
    with (damaged / EVENTS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write('{"type": "reply.text", "text": "half')

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get("/api/sessions")

    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert [session["id"] for session in sessions] == [intact.name, damaged.name]
    assert sessions[0]["reply_text"] == "the intact reply"
    # The whole lines are still read; only the partial one is dropped.
    assert sessions[1]["reply_text"] == "the reply before the crash"
    assert sessions[1]["turn_outcome"] == "completed"


def test_a_directory_that_cannot_be_summarised_at_all_is_still_one_row(tmp_path, fake_account_repository):
    """Not the missing-`timing.json` case above: a `timing.json` that parses
    as JSON but is not the object the recorder writes. It must still be one
    `recording_incomplete` row beside the intact session, never a 500 that
    makes every recording unreachable (CR-01)."""
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    intact = _write_recorded_session(session_config)
    damaged = _write_recorded_session(session_config, started_offset_days=1)
    (damaged / "timing.json").write_text('["not", "an", "object"]', encoding="utf-8")

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get("/api/sessions")

    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert [session["id"] for session in sessions] == [intact.name, damaged.name]
    assert sessions[0]["turn_outcome"] == "completed"
    assert sessions[1]["turn_outcome"] == "recording_incomplete"
    assert sessions[1]["duration_ms"] is None



def test_the_session_the_list_calls_incomplete_is_refused_by_name_not_by_500(tmp_path, fake_account_repository):
    """One directory, all three routes. Whatever the list route advertises,
    the detail and audio routes have to honour -- they read the same
    `timing.json` through the same function now, so they cannot come to
    disagree about which recordings are openable (CR-02)."""
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)
    (directory / "timing.json").unlink()

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    listed = client.get("/api/sessions").json()["sessions"]
    assert [session["id"] for session in listed] == [directory.name]
    assert listed[0]["turn_outcome"] == "recording_incomplete"

    detail = client.get(f"/api/sessions/{directory.name}")
    assert detail.status_code == 409
    assert "was never finished being written" in detail.json()["detail"]

    audio = client.get(f"/api/sessions/{directory.name}/audio")
    assert audio.status_code == 409
    assert "was never finished being written" in audio.json()["detail"]


def test_a_detail_route_timeline_it_cannot_rebuild_is_the_same_named_refusal(tmp_path, fake_account_repository):
    """The mixed case: `timing.json` survived, `events.jsonl` did not, and
    the rendered timeline was never written. `regenerate_timeline` reads
    both with its own stricter reader, so this is the one path into the
    detail route that a guarded `timing.json` read does not already
    cover (CR-02)."""
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)
    (directory / "timeline.jsonl").unlink()
    with (directory / EVENTS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write('{"type": "reply.text", "text": "half')

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{directory.name}")

    assert response.status_code == 409
    assert "was never finished being written" in response.json()["detail"]


# --- Task 2 (TDD): GET /api/sessions/{id} -----------------------------------
#
# RED: every test below targets `GET /api/sessions/{session_id}`, a route
# `routes/sessions.py` does not register yet -- FastAPI's own generic 404
# ("Not Found", no `detail` naming a refusal) answers every one of these
# until Task 2's GREEN phase adds the route.


def test_a_real_session_is_returned_in_full_with_offsets_on_every_timeline_entry(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config, transcript_text="turn the kitchen lights on")

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{directory.name}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == directory.name
    assert body["turn_outcome"] == "completed"
    assert body["transcript"] == "turn the kitchen lights on"
    assert body["reply_text"] == "the reply"
    assert body["stage_durations_ms"]
    assert body["timeline"], "the merged timeline must not be empty for a real, closed session"
    for entry in body["timeline"]:
        assert entry["offset_s"] is not None
    # `_write_recorded_session` builds this fixture with no pre-roll, so
    # preroll_offset_s must fall back to 0.0 and the turn_started_at row
    # must acquire no shift at all -- this is what would catch a
    # preroll_offset_s returning a non-zero fallback (plan 08-11, DBG-03).
    turn_started_row = next(entry for entry in body["timeline"] if entry.get("stage") == "turn_started_at")
    assert turn_started_row["offset_s"] == 0.0


def test_no_partial_before_the_final_mark_means_a_null_transcript(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config, transcript_text=None)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{directory.name}")

    assert response.status_code == 200
    assert response.json()["transcript"] is None


def test_a_session_id_not_shaped_like_a_directory_name_is_refused_as_unrecognised(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    _write_recorded_session(session_config)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get("/api/sessions/not-a-session-id")

    assert response.status_code == 404
    assert "not a recognised session id" in response.json()["detail"]


def test_a_traversal_attempt_is_refused_before_any_path_is_built(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    _write_recorded_session(session_config)
    # A secret file this traversal attempt must never be able to reach.
    (tmp_path.parent / "secret.txt").write_text("do not read me", encoding="utf-8")

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    literal = client.get("/api/sessions/20260101T000000000000Z-../../secret.txt")
    assert literal.status_code == 404
    assert "not a recognised session id" in literal.json()["detail"]
    assert "do not read me" not in literal.text

    # `httpx.Client` (what `TestClient` wraps) normalizes a raw `../` in the
    # URL path itself before it ever reaches the server, so the meaningful
    # traversal probe against *this* server is the percent-encoded form,
    # which the transport does not touch. Starlette's own router never
    # matches `%2F` against a single-segment `{session_id}` path parameter
    # at all -- the request never reaches this module's handler, and the
    # 404 it gets is Starlette's own generic one, not this module's named
    # refusal. Either shape refuses the same traversal before any path is
    # built; only the message differs by which layer caught it.
    encoded = client.get("/api/sessions/20260101T000000000000Z-%2e%2e%2Fsecret.txt")
    assert encoded.status_code == 404
    assert "do not read me" not in encoded.text


def test_a_traversal_segment_is_refused_by_the_helper_itself_not_by_the_router(tmp_path, fake_account_repository):
    """Called directly, with no ASGI layer in front of it. Starlette's
    `[^/]+` is what stops the HTTP form of this today; the helper is the
    "one code path, one confinement check" both single-session routes lean
    on, and its own docstring claims the containment. It has to hold on its
    own (WR-01)."""
    session_config = SessionConfig(dir=str(tmp_path), retain_days=7)
    older = _write_recorded_session(session_config, started_offset_days=30)

    # A fresh-looking prefix in front of a segment that escapes back into
    # the root: the shape check passes (`.+` matches a separator) and the
    # resolved parent *is* the root, so the containment check alone let
    # this through -- and the retention gate then measured today's date,
    # parsed from the prefix, against a directory 30 days old.
    escaping_id = f"20260101T000000000000Z-x/../{older.name}"

    with pytest.raises(HTTPException) as refusal:
        _resolve_readable_session_directory(session_config, escaping_id)
    assert refusal.value.status_code == 404
    assert "not a recognised session id" in refusal.value.detail

    # And the directory it was reaching for is genuinely still there, so
    # the refusal is the helper's doing and not a missing fixture.
    assert older.is_dir()


def test_a_name_the_recorder_could_not_have_written_does_not_parse(tmp_path):
    """`re.match` accepted a trailing newline, because `$` does. The sweep
    and this module share this one parser, so a name it misreads is a name
    both misread (WR-01)."""
    assert parse_session_timestamp("20260101T000000000000Z-abc") is not None
    assert parse_session_timestamp("20260101T000000000000Z-abc\n") is None


def test_an_id_older_than_the_retention_window_is_refused_naming_the_sweep(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path), retain_days=7)
    directory = _write_recorded_session(session_config, started_offset_days=30)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{directory.name}")

    assert response.status_code == 404
    assert "removed by the retention sweep" in response.json()["detail"]


def test_an_id_inside_the_window_with_no_directory_is_refused_as_not_found(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path), retain_days=7)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    made_up_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-turn-never-existed"
    response = client.get(f"/api/sessions/{made_up_id}")

    assert response.status_code == 404
    assert "no session with id" in response.json()["detail"]


def test_a_viewer_is_refused_the_detail_route_and_an_operator_succeeds(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)

    app = _build_sessions_app(security, account_repo, session_config)
    viewer_client = _client_with_role(app, security, account_repo, "viewer")
    operator_client = _client_with_role(app, security, account_repo, "operator")

    assert viewer_client.get(f"/api/sessions/{directory.name}").status_code == 403
    assert operator_client.get(f"/api/sessions/{directory.name}").status_code == 200


# --- Task 1 of 08-05 (TDD): GET /api/sessions/{id}/audio --------------------
#
# RED: `routes/sessions.py` does not register this route yet -- FastAPI's
# own generic 404 answers every request below until Task 1's GREEN phase
# adds it.


def test_a_recorded_session_returns_a_valid_wav_body_to_an_operator(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{directory.name}/audio")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    body = response.content
    assert body[0:4] == b"RIFF"
    assert body[8:12] == b"WAVE"
    assert int(response.headers["content-length"]) == len(body)


def test_a_traversal_attempt_against_the_audio_route_is_refused(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    _write_recorded_session(session_config)
    (tmp_path.parent / "secret.txt").write_text("do not read me", encoding="utf-8")

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get("/api/sessions/20260101T000000000000Z-../../secret.txt/audio")
    assert response.status_code == 404
    assert "do not read me" not in response.text


def test_a_range_header_is_ignored_and_the_full_body_still_comes_back(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    full = client.get(f"/api/sessions/{directory.name}/audio")
    ranged = client.get(f"/api/sessions/{directory.name}/audio", headers={"Range": "bytes=0-3"})

    assert ranged.status_code == 200
    assert ranged.status_code != 206
    assert ranged.content == full.content


def test_a_session_with_no_recorded_audio_gets_its_own_named_refusal(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config, include_audio=False)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{directory.name}/audio")

    assert response.status_code == 404
    assert "no recorded audio" in response.json()["detail"]


def test_an_encoding_this_deployment_cannot_wrap_is_a_named_refusal(tmp_path, fake_account_repository):
    """`_has_audio` is satisfied by any `audio.{encoding}` that exists, and
    `wrap_session_audio` refuses anything outside its supported set --
    `AudioWrapError` exists precisely so that is a named refusal. The route
    dropped it on the floor as a 500 (WR-04)."""
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)
    timing = json.loads((directory / "timing.json").read_text(encoding="utf-8"))
    timing["audio_format"] = {"encoding": "opus", "sample_rate": 8000}
    (directory / "timing.json").write_text(json.dumps(timing), encoding="utf-8")
    (directory / "audio.opus").write_bytes(b"\x00" * 16)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{directory.name}/audio")

    assert response.status_code == 409
    assert "cannot play back" in response.json()["detail"]
    assert "opus" in response.json()["detail"]


def test_an_audio_format_with_no_sample_rate_is_the_same_named_refusal(tmp_path, fake_account_repository):
    """Same line, the other way it fails: `_has_audio` passes on `encoding`
    alone, so a format carrying no `sample_rate` reached `wrap_session_audio`
    as a `KeyError` (WR-04)."""
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)
    timing = json.loads((directory / "timing.json").read_text(encoding="utf-8"))
    timing["audio_format"] = {"encoding": "pcm"}
    (directory / "timing.json").write_text(json.dumps(timing), encoding="utf-8")

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{directory.name}/audio")

    assert response.status_code == 409
    assert "cannot play back" in response.json()["detail"]


def test_a_missing_session_gets_the_not_found_refusal_not_the_missing_audio_one(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path), retain_days=7)

    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    made_up_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-turn-never-existed"
    response = client.get(f"/api/sessions/{made_up_id}/audio")

    assert response.status_code == 404
    assert "no session with id" in response.json()["detail"]


def test_a_viewer_is_refused_the_audio_route_and_an_operator_succeeds(tmp_path, fake_account_repository):
    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = SessionConfig(dir=str(tmp_path))
    directory = _write_recorded_session(session_config)

    app = _build_sessions_app(security, account_repo, session_config)
    viewer_client = _client_with_role(app, security, account_repo, "viewer")
    operator_client = _client_with_role(app, security, account_repo, "operator")

    assert viewer_client.get(f"/api/sessions/{directory.name}/audio").status_code == 403
    assert operator_client.get(f"/api/sessions/{directory.name}/audio").status_code == 200
