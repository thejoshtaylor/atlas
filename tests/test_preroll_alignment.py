"""Ties one timeline offset to a byte position in the recorded audio
(DBG-03, WEB-07, plan 08-11).

Before this file, nothing in the suite proved that the camera path's
recorded audio begins with the replayed pre-roll window while every
timeline `offset_s` was measured from `turn_started_at` -- which sits at
the END of that window. Every timeline row therefore pointed about 1.5 s
earlier in the recording than the event it named. This file proves the
fix end to end: `PrerollReplayingSource.preroll_bytes` -> `SessionRecorder`
-> `timing.json`'s `preroll_bytes` -> `session/timeline.py::preroll_offset_s`
-> `routes/sessions.py`'s `offset_s`, checked against a byte position
located by CONTENT in the written `audio.alaw`, never by re-reading the
number the recorder itself wrote.

It also pins down everything the shift must NOT change: the two transports
that never built a pre-roll buffer, a `timing.json` written before this
fix, a malformed `preroll_bytes`, an unmodelled encoding, an empty
timeline, and two timeline entries tied at the same instant (plan 08-11
Task 2).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator, Sequence

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spire_voice.audio.ring import bytes_per_ms
from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig, SessionConfig
from spire_voice.providers.base import BrainReply, FinalTranscript
from spire_voice.routes.sessions import router as sessions_router
from spire_voice.session.recorder import SessionRecorder
from spire_voice.session.timeline import preroll_offset_s
from spire_voice.sources.runner import PrerollReplayingSource
from spire_voice.timing import TurnTimings
from spire_voice.transports.base import SourceFormat
from spire_voice.turn.controller import run_turn

# Distinguishable by content, never by re-reading the count the recorder
# wrote: the pre-roll and the live audio use different byte values so the
# boundary between them can be located in the written file directly.
_PREROLL_BYTE = b"\xd5"
_LIVE_BYTE = b"\x2a"
_PREROLL_CHUNK_SIZE = 4000
_PREROLL_CHUNK_COUNT = 3
_PREROLL_TOTAL_BYTES = _PREROLL_CHUNK_SIZE * _PREROLL_CHUNK_COUNT  # 12000


def _session_config(tmp_path: Path, *, record_audio: bool = True) -> SessionConfig:
    return SessionConfig(dir=str(tmp_path), record_audio=record_audio)


class FakeAlawSource:
    """A source declaring the camera's own `SourceFormat("alaw", 8000)` --
    `tests/conftest.py`'s `FakeAudioSource` declares 16 kHz PCM16, which
    would exercise the wrong byte rate for this test's whole point."""

    def __init__(self, frames: Sequence[bytes] = ()) -> None:
        self._frames = list(frames)
        self.sent_audio: list[bytes] = []

    async def frames(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    def source_format(self) -> SourceFormat:
        return SourceFormat("alaw", 8000)


def _build_sessions_app(security: SecurityConfig, account_repo, session_config: SessionConfig) -> FastAPI:
    """Same throwaway-app shape `tests/test_session_routes.py` uses -- a
    `FastAPI()` carrying only `routes/sessions.py`'s own router."""
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


async def _run_camera_turn(
    tmp_path: Path,
    recording_fake_stt,
    fake_brain,
    fake_tts,
    *,
    preroll_chunks: "list[bytes] | None" = None,
    record_audio: bool = True,
) -> SessionRecorder:
    """Drive one real `run_turn` through a real `SessionRecorder`, wrapping
    a `FakeAlawSource` in `PrerollReplayingSource` when `preroll_chunks` is
    given -- the real wrapper `sources/runner.py::_process_chunk` builds,
    never a hand-written stand-in.
    """
    config = _session_config(tmp_path, record_audio=record_audio)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)

    live_source = FakeAlawSource(frames=[_LIVE_BYTE * _PREROLL_CHUNK_SIZE])
    source = PrerollReplayingSource(live_source, preroll_chunks) if preroll_chunks else live_source

    stt = recording_fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = fake_brain(replies=[BrainReply(text="turned on the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host=None,
        tools_schema=[],
        system_prompt="",
        max_tool_rounds=3,
        timings=timings,
        session_recorder=recorder,
    )
    return recorder


def test_preroll_replaying_source_reports_total_preroll_bytes():
    """The property Task 1 adds: the sum of every held pre-roll chunk's
    length, over the live source's own frames -- computed once, from the
    chunks this wrapper already holds, never re-derived elsewhere."""
    live_source = FakeAlawSource(frames=[_LIVE_BYTE * _PREROLL_CHUNK_SIZE])
    preroll_chunks = [_PREROLL_BYTE * _PREROLL_CHUNK_SIZE for _ in range(_PREROLL_CHUNK_COUNT)]
    wrapped = PrerollReplayingSource(live_source, preroll_chunks)

    assert wrapped.preroll_bytes == _PREROLL_TOTAL_BYTES


def test_bytes_per_ms_is_public_and_matches_the_encodings_it_bounds():
    """`audio/ring.py`'s renamed, public conversion -- the same function
    `preroll_offset_s` inverts, asserted from both encodings the codebase
    carries end to end (`transports/base.py::SourceFormat`)."""
    assert bytes_per_ms(SourceFormat("alaw", 8000)) == 8.0
    assert bytes_per_ms(SourceFormat("pcm", 16000)) == 32.0


async def test_camera_turn_pre_roll_length_travels_to_the_browser_reported_offset(
    tmp_path, recording_fake_stt, fake_brain, fake_tts, fake_account_repository
):
    """The tracer's load-bearing assertion: the `turn_started_at` row's
    `offset_s` from `GET /api/sessions/{id}`, times the recording's own
    byte rate, equals the byte index at which the first live byte appears
    in `audio.alaw` -- located by CONTENT, never by reading back the
    `preroll_bytes` field this same test also checks separately.
    """
    preroll_chunks = [_PREROLL_BYTE * _PREROLL_CHUNK_SIZE for _ in range(_PREROLL_CHUNK_COUNT)]

    recorder = await _run_camera_turn(tmp_path, recording_fake_stt, fake_brain, fake_tts, preroll_chunks=preroll_chunks)

    timing_payload = json.loads((recorder.directory / "timing.json").read_text(encoding="utf-8"))
    assert timing_payload["preroll_bytes"] == _PREROLL_TOTAL_BYTES

    audio_bytes = (recorder.directory / "audio.alaw").read_bytes()
    live_byte_index = audio_bytes.index(_LIVE_BYTE)
    assert live_byte_index == _PREROLL_TOTAL_BYTES
    assert audio_bytes[:live_byte_index] == _PREROLL_BYTE * _PREROLL_TOTAL_BYTES

    assert preroll_offset_s(timing_payload) == 1.5

    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = _session_config(tmp_path)
    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{recorder.directory.name}")
    assert response.status_code == 200
    body = response.json()

    turn_started_row = next(entry for entry in body["timeline"] if entry.get("stage") == "turn_started_at")
    offset_s = turn_started_row["offset_s"]
    assert offset_s == 1.5
    byte_rate = bytes_per_ms(SourceFormat("alaw", 8000)) * 1000.0
    assert offset_s * byte_rate == live_byte_index == _PREROLL_TOTAL_BYTES


def test_declared_preroll_exceeding_bytes_received_serializes_the_smaller_count(tmp_path):
    """A turn that ends before draining its whole pre-roll -- the recorder
    must serialize what actually reached disk, never the declared count.
    Exercised directly against `SessionRecorder` (not through `run_turn`),
    the same shape `test_record_audio_false_serializes_zero_preroll_bytes`
    below uses, since the point here is `close()`'s own `min()` guard --
    not the wiring `run_turn` performs, which the tracer test above
    already proves end to end."""
    config = _session_config(tmp_path)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)
    recorder.set_audio_format("alaw", 8000)
    recorder.set_preroll_bytes(999999)
    recorder.record_audio_chunk(_PREROLL_BYTE * _PREROLL_CHUNK_SIZE)
    recorder.close(timings)

    timing_payload = json.loads((recorder.directory / "timing.json").read_text(encoding="utf-8"))
    assert timing_payload["preroll_bytes"] == _PREROLL_CHUNK_SIZE


def test_record_audio_false_serializes_zero_preroll_bytes(tmp_path):
    """A recorder configured with `record_audio=False` writes no audio
    file at all -- `preroll_bytes` must report `0`, never a count against
    bytes that were never written."""
    config = _session_config(tmp_path, record_audio=False)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)
    recorder.set_audio_format("alaw", 8000)
    recorder.set_preroll_bytes(12000)
    recorder.record_audio_chunk(_LIVE_BYTE * _PREROLL_CHUNK_SIZE)
    recorder.close(timings)

    timing_payload = json.loads((recorder.directory / "timing.json").read_text(encoding="utf-8"))
    assert timing_payload["preroll_bytes"] == 0
    assert not (recorder.directory / "audio.alaw").exists()


def test_preroll_offset_s_returns_zero_for_every_malformed_or_absent_payload():
    """T-08-23's mitigation, asserted directly against the pure function:
    a missing, negative, or non-numeric `preroll_bytes`, an unmodelled
    encoding, or an absent `audio_format` all fall back to `0.0` -- never
    a raised exception, never a guessed byte rate."""
    good_format = {"encoding": "alaw", "sample_rate": 8000}
    assert preroll_offset_s({}) == 0.0
    assert preroll_offset_s({"preroll_bytes": 12000, "audio_format": good_format}) == 1.5
    assert preroll_offset_s({"preroll_bytes": -1, "audio_format": good_format}) == 0.0
    assert preroll_offset_s({"preroll_bytes": "lots", "audio_format": good_format}) == 0.0
    assert preroll_offset_s({"preroll_bytes": 800, "audio_format": {"encoding": "opus", "sample_rate": 8000}}) == 0.0
    assert preroll_offset_s({"preroll_bytes": 800, "audio_format": None}) == 0.0


# --- Task 2: the paths that must not move --------------------------------


async def test_browser_and_webrtc_paths_report_no_preroll_and_no_shift(
    tmp_path, recording_fake_stt, fake_brain, fake_tts, fake_account_repository
):
    """Neither the browser-microphone nor the WebRTC transport wraps its
    source in `PrerollReplayingSource` -- both reach `run_turn` with a bare
    source, the exact shape this fixture stands for. A comment, not a
    second fixture, is what distinguishes the two: neither transport
    builds a `PrerollBuffer`, so both reach `run_turn` identically."""
    recorder = await _run_camera_turn(tmp_path, recording_fake_stt, fake_brain, fake_tts, preroll_chunks=None)

    timing_payload = json.loads((recorder.directory / "timing.json").read_text(encoding="utf-8"))
    assert timing_payload["preroll_bytes"] == 0

    audio_bytes = (recorder.directory / "audio.alaw").read_bytes()
    assert audio_bytes[:1] == _LIVE_BYTE

    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = _session_config(tmp_path)
    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{recorder.directory.name}")
    assert response.status_code == 200
    body = response.json()
    turn_started_row = next(entry for entry in body["timeline"] if entry.get("stage") == "turn_started_at")
    assert turn_started_row["offset_s"] == 0.0


async def test_a_timing_json_with_no_preroll_bytes_key_reads_the_same_as_an_explicit_zero(
    tmp_path, recording_fake_stt, fake_brain, fake_tts, fake_account_repository
):
    """The strongest form of the compatibility claim: a `timing.json`
    written before this fix -- carrying no `preroll_bytes` key at all --
    must answer with exactly the offsets an explicit `preroll_bytes: 0`
    produces. An absent key is read as zero, never as a reason to refuse
    the session."""
    recorder = await _run_camera_turn(tmp_path, recording_fake_stt, fake_brain, fake_tts, preroll_chunks=None)
    timing_path = recorder.directory / "timing.json"
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    assert timing_payload["preroll_bytes"] == 0
    del timing_payload["preroll_bytes"]
    timing_path.write_text(json.dumps(timing_payload, sort_keys=True, indent=2), encoding="utf-8")

    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = _session_config(tmp_path)
    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{recorder.directory.name}")
    assert response.status_code == 200
    body = response.json()
    turn_started_row = next(entry for entry in body["timeline"] if entry.get("stage") == "turn_started_at")
    assert turn_started_row["offset_s"] == 0.0


async def test_a_negative_preroll_bytes_answers_200_with_unshifted_offsets(
    tmp_path, recording_fake_stt, fake_brain, fake_tts, fake_account_repository
):
    """T-08-23: a tampered or corrupted `timing.json` carrying a negative
    `preroll_bytes` must never produce a negative `offset_s`, and must
    never turn into a 500."""
    recorder = await _run_camera_turn(tmp_path, recording_fake_stt, fake_brain, fake_tts, preroll_chunks=None)
    timing_path = recorder.directory / "timing.json"
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    timing_payload["preroll_bytes"] = -4000
    timing_path.write_text(json.dumps(timing_payload, sort_keys=True, indent=2), encoding="utf-8")

    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = _session_config(tmp_path)
    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{recorder.directory.name}")
    assert response.status_code == 200
    body = response.json()
    for entry in body["timeline"]:
        if entry["offset_s"] is not None:
            assert entry["offset_s"] >= 0.0


async def test_an_unmodelled_encoding_answers_200_with_unshifted_offsets(
    tmp_path, recording_fake_stt, fake_brain, fake_tts, fake_account_repository
):
    """T-08-23: `audio_format.encoding` naming something this codebase
    does not model exactly (`"opus"`) must fall back to unshifted offsets,
    never a guessed byte rate and never a 500."""
    recorder = await _run_camera_turn(tmp_path, recording_fake_stt, fake_brain, fake_tts, preroll_chunks=None)
    timing_path = recorder.directory / "timing.json"
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    timing_payload["preroll_bytes"] = 4000
    timing_payload["audio_format"]["encoding"] = "opus"
    timing_path.write_text(json.dumps(timing_payload, sort_keys=True, indent=2), encoding="utf-8")

    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = _session_config(tmp_path)
    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{recorder.directory.name}")
    assert response.status_code == 200
    body = response.json()
    for entry in body["timeline"]:
        if entry["offset_s"] is not None:
            assert entry["offset_s"] >= 0.0
    turn_started_row = next(entry for entry in body["timeline"] if entry.get("stage") == "turn_started_at")
    assert turn_started_row["offset_s"] == 0.0


async def test_an_empty_timeline_answers_200_with_an_empty_list(
    tmp_path, recording_fake_stt, fake_brain, fake_tts, fake_account_repository
):
    """WEB-07's empty-input edge, carried because this plan touches the
    arithmetic that runs over the timeline list."""
    recorder = await _run_camera_turn(tmp_path, recording_fake_stt, fake_brain, fake_tts, preroll_chunks=None)
    (recorder.directory / "timeline.jsonl").write_text("", encoding="utf-8")

    security = SecurityConfig()
    account_repo = fake_account_repository()
    session_config = _session_config(tmp_path)
    app = _build_sessions_app(security, account_repo, session_config)
    client = _client_with_role(app, security, account_repo, "operator")

    response = client.get(f"/api/sessions/{recorder.directory.name}")
    assert response.status_code == 200
    assert response.json()["timeline"] == []


def test_two_entries_tied_at_the_same_instant_keep_written_order_and_the_same_offset(tmp_path):
    """WEB-07's adjacency and ordering edges: adding one constant to every
    offset cannot change which of two equal offsets is later, and the
    frontend's own `activeTimelineIndexAt` (`deriveSessionDetailScreenState.ts`)
    already resolves a tie to the later of two equal offsets -- unaffected
    by a constant shift, which is why this fix needs no change there."""
    config = _session_config(tmp_path)
    timings = TurnTimings()
    timings.mark_turn_started()
    recorder = SessionRecorder(config, timings)
    recorder.set_audio_format("alaw", 8000)
    recorder.record_event({"type": "transcript.partial", "text": "first"})
    recorder.record_event({"type": "transcript.partial", "text": "second"})
    recorder.close(timings)

    from spire_voice.session import timeline as timeline_module

    events_path = recorder.directory / "events.jsonl"
    raw_events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tied_ts = raw_events[0]["recorded_at"]
    raw_events[1]["recorded_at"] = tied_ts
    events_path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in raw_events) + "\n", encoding="utf-8"
    )
    timeline_module.regenerate_timeline(recorder.directory)

    rendered = [
        json.loads(line)
        for line in (recorder.directory / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tied_entries = [entry for entry in rendered if entry.get("ts") == tied_ts]
    assert [entry["text"] for entry in tied_entries] == ["first", "second"]
