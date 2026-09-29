"""Voice inbox and retroactive clip routes (quick task 260929-j08).

A throwaway `FastAPI()` app carries the two speaker routers in the same
order as `routes/__init__.py`. Every session directory is built through a
real `SessionRecorder`, the same way `tests/test_session_routes.py` does.
"""

from __future__ import annotations

import asyncio
import io
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import test_startup_smoke as smoke
from atlas.auth.tokens import issue_access_token
from atlas.config import EdgeSourceConfig, SecurityConfig, SessionConfig, SpeakerIdConfig
from atlas.routes.speaker_inbox import router as speaker_inbox_router
from atlas.routes.speakers import router as speakers_router
from atlas.session.recorder import SessionRecorder
from atlas.speaker_id.enrollment import ClipStore
from atlas.speaker_id.matching import ReferenceSet
from atlas.timing import TurnTimings

from tests.edge_fakes import interleave
from tests.speaker_fakes import FakeEmbedder
from tests.speaker_repo_fakes import FakeSpeakerRepository

_MODEL_ID = "3dspeaker_speech_campplus_sv_en_voxceleb_16k"

_UNKNOWN_EVENT = {
    "type": "speaker.result",
    "status": "unknown",
    "speaker_id": None,
    "speaker_name": None,
    "score": 0.2,
    "blocked": True,
    "reason": "unknown_speaker",
    "detail": "below_threshold",
    "speech_ms": 2000.0,
}


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("ATLAS_SECRET_KEY", smoke._TEST_SECRET_KEY)


def _speaker_config(**overrides: object) -> SpeakerIdConfig:
    base = dict(
        mode="off",
        model="campplus",
        speech_rms_floor=None,
        min_enrollment_speech_ms=1000,
        enrollment_max_phrase_s=10.0,
    )
    base.update(overrides)
    return SpeakerIdConfig(**base)


def _write_edge_session(
    session_config: SessionConfig,
    *,
    ch0: int = 3000,
    ch1: int = 1000,
    samples: int = 32000,
    audio_format: "tuple[str, int, int]" = ("pcm", 16000, 2),
    preroll_bytes: int = 4096,
    turn_outcome: str = "unknown_speaker",
    transcript: str = "hey atlas turn on the lamp",
    speaker_event: "dict | None" = _UNKNOWN_EVENT,
    include_audio: bool = True,
    started_offset_days: float = 0.0,
) -> Path:
    timings = TurnTimings(turn_outcome=turn_outcome)
    timings.mark_turn_started()
    recorder = SessionRecorder(session_config, timings)
    recorder.set_audio_format(*audio_format)
    recorder.set_preroll_bytes(preroll_bytes)
    recorder.record_event({"type": "transcript.partial", "text": transcript})
    if speaker_event is not None:
        recorder.record_event(dict(speaker_event))
    timings.mark_stt_final()
    if include_audio:
        recorder.record_audio_chunk(interleave(ch0, ch1, samples))
    recorder.close(timings)

    directory = recorder.directory
    if started_offset_days:
        target_at = datetime.now(timezone.utc) - timedelta(days=started_offset_days)
        stamp = target_at.strftime("%Y%m%dT%H%M%S%fZ")
        renamed = directory.parent / f"{stamp}-{timings.turn_id}"
        directory.rename(renamed)
        directory = renamed
    return directory


class _Env:
    def __init__(self, tmp_path: Path, account_repo, **speaker_overrides: object) -> None:
        self.security = SecurityConfig()
        self.account_repo = account_repo
        self.session_config = SessionConfig(dir=str(tmp_path / "sessions"))
        Path(self.session_config.dir).mkdir()
        self.config = SimpleNamespace(
            security=self.security,
            session=self.session_config,
            speaker_id=_speaker_config(**speaker_overrides),
            edge=EdgeSourceConfig(),
        )
        self.repo = FakeSpeakerRepository()
        self.clip_store = ClipStore(tmp_path / "speakers")
        self.references = ReferenceSet()
        app = FastAPI()
        app.state.config = self.config
        app.state.account_repo = account_repo
        app.state.speaker_repo = self.repo
        app.state.speaker_clip_store = self.clip_store
        app.state.speaker_id_context = SimpleNamespace(references=self.references, worker=None)
        app.state.speaker_embedder_factory = lambda config: FakeEmbedder()
        app.include_router(speakers_router)
        app.include_router(speaker_inbox_router)
        self.app = app

    def client(self, role: str = "admin") -> TestClient:
        user = asyncio.run(
            self.account_repo.create_user(
                email=f"{role}@example.invalid",
                display_name=role.title(),
                password_hash="not-checked-by-this-test",
                role=role,
            )
        )
        token = issue_access_token(user_id=user.id, role=role, security=self.security)
        return TestClient(self.app, cookies={self.security.cookie_name: token})

    def member(self, name: str = "Ann"):
        return asyncio.run(
            self.repo.create_speaker(display_name=name, linked_user_id=None, created_at=datetime.now(timezone.utc))
        )

    def close(self) -> None:
        worker = getattr(self.app.state, "speaker_enrollment_worker", None)
        if worker is not None:
            worker.close()


@pytest.fixture
def env(tmp_path, fake_account_repository):
    environment = _Env(tmp_path, fake_account_repository())
    yield environment
    environment.close()


def _listed_ids(client: TestClient) -> "list[str]":
    response = client.get("/api/speakers/voice-inbox")
    assert response.status_code == 200, response.text
    return [item["session_id"] for item in response.json()["items"]]


# -- list, assign ---------------------------------------------------------


def test_an_unrecognized_edge_turn_is_listed_assigned_and_leaves_the_inbox(env):
    directory = _write_edge_session(env.session_config)
    ann = env.member("Ann")
    client = env.client()

    listed = client.get("/api/speakers/voice-inbox").json()["items"]
    assert [item["session_id"] for item in listed] == [directory.name]
    [item] = listed
    assert set(item) == {"session_id", "started_at", "transcript", "speech_ms", "score", "blocked"}
    assert item["blocked"] is True
    assert item["transcript"] == "hey atlas turn on the lamp"

    response = client.post(f"/api/speakers/voice-inbox/{directory.name}/assign", json={"speaker_id": ann.id})

    assert response.status_code == 200, response.text
    assert response.json()["phrase_index"] == 100
    assert response.json()["speech_ms"] == 2000.0
    assert env.clip_store.clip_path(ann.id, 100).is_file()
    assert (ann.id, 100, _MODEL_ID) in env.repo._embeddings
    assert env.references.name_for(ann.id) == "Ann"
    assert _listed_ids(client) == []


def test_assign_with_a_new_name_creates_the_member(env):
    directory = _write_edge_session(env.session_config)
    client = env.client()

    response = client.post(
        f"/api/speakers/voice-inbox/{directory.name}/assign", json={"new_speaker_name": "  Guest "}
    )

    assert response.status_code == 200, response.text
    new_id = response.json()["speaker_id"]
    assert asyncio.run(env.repo.get_speaker(new_id)).display_name == "Guest"
    assert env.clip_store.clip_path(new_id, 100).is_file()


def test_assign_refuses_a_bad_body_a_taken_name_and_a_too_short_turn(env):
    directory = _write_edge_session(env.session_config)
    short = _write_edge_session(env.session_config, samples=8000)
    ann = env.member("Ann")
    client = env.client()
    url = f"/api/speakers/voice-inbox/{directory.name}/assign"

    assert client.post(url, json={"new_speaker_name": "bad/name"}).status_code == 422
    assert client.post(url, json={"speaker_id": ann.id, "new_speaker_name": "Guest"}).status_code == 422
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"new_speaker_name": "Ann"}).status_code == 409
    assert env.clip_store.list_clips(ann.id) == []

    response = client.post(f"/api/speakers/voice-inbox/{short.name}/assign", json={"new_speaker_name": "Guest"})
    assert response.status_code == 422
    assert [s.display_name for s in asyncio.run(env.repo.list_speakers())] == ["Ann"]


def test_the_inbox_leaves_out_turns_that_cannot_be_assigned(env):
    listed = _write_edge_session(env.session_config)
    _write_edge_session(env.session_config, speaker_event={**_UNKNOWN_EVENT, "status": "identified"})
    _write_edge_session(env.session_config, audio_format=("alaw", 8000, 1))
    _write_edge_session(env.session_config, turn_outcome="wake_unverified")
    _write_edge_session(env.session_config, preroll_bytes=0)
    _write_edge_session(env.session_config, include_audio=False)
    _write_edge_session(env.session_config, started_offset_days=8)
    _write_edge_session(env.session_config, samples=8000)
    _write_edge_session(env.session_config, speaker_event={**_UNKNOWN_EVENT, "detail": "not_edge_source"})
    assigned = _write_edge_session(env.session_config)
    ann = env.member("Ann")
    client = env.client()
    assert client.post(f"/api/speakers/voice-inbox/{assigned.name}/assign", json={"speaker_id": ann.id}).status_code == 200

    assert _listed_ids(client) == [listed.name]


def test_assign_refusals(env, tmp_path):
    directory = _write_edge_session(env.session_config)
    camera = _write_edge_session(env.session_config, audio_format=("alaw", 8000, 1))
    old = _write_edge_session(env.session_config, started_offset_days=8)
    ann = env.member("Ann")
    client = env.client()

    def assign(session_id: str, **body):
        return client.post(f"/api/speakers/voice-inbox/{session_id}/assign", json=body or {"speaker_id": ann.id})

    assert assign(camera.name).status_code == 409
    assert assign(directory.name, speaker_id=999).status_code == 404
    assert assign("not-a-session").status_code == 404
    assert assign(old.name).status_code == 404
    assert assign(directory.name).status_code == 200
    assert assign(directory.name).status_code == 409

    fresh = _write_edge_session(env.session_config)
    env.config.edge = EdgeSourceConfig(asr_channel=None)
    assert assign(fresh.name).status_code == 409
    env.config.edge = EdgeSourceConfig()
    env.config.speaker_id = _speaker_config(model=None)
    response = assign(fresh.name)
    assert response.status_code == 409
    assert response.json()["detail"] == "speaker_id.model is not set"


def test_the_audio_route_returns_the_trimmed_mono_wav(env):
    directory = _write_edge_session(env.session_config)
    no_audio = _write_edge_session(env.session_config, include_audio=False)
    client = env.client()

    response = client.get(f"/api/speakers/voice-inbox/{directory.name}/audio")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert int(response.headers["content-length"]) == len(response.content)
    assert response.content.startswith(b"RIFF")
    with wave.open(io.BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 32000
    assert client.get(f"/api/speakers/voice-inbox/{no_audio.name}/audio").status_code == 404


def test_an_operator_gets_403_on_the_inbox_routes(env):
    directory = _write_edge_session(env.session_config)
    ann = env.member("Ann")
    client = env.client("operator")

    assert client.get("/api/speakers/voice-inbox").status_code == 403
    assert client.get(f"/api/speakers/voice-inbox/{directory.name}/audio").status_code == 403
    response = client.post(f"/api/speakers/voice-inbox/{directory.name}/assign", json={"speaker_id": ann.id})
    assert response.status_code == 403


# -- retroactive clips ------------------------------------------------------


def _assign(client, session_name: str, speaker_id: int):
    response = client.post(f"/api/speakers/voice-inbox/{session_name}/assign", json={"speaker_id": speaker_id})
    assert response.status_code == 200, response.text
    return response.json()


def test_the_assign_response_names_the_dropped_indices(env):
    directory = _write_edge_session(env.session_config)
    ann = env.member("Ann")
    client = env.client()

    assert _assign(client, directory.name, ann.id)["dropped_phrase_indices"] == []


def test_clip_routes_list_play_and_delete_a_clip_and_free_its_turn(env):
    directory = _write_edge_session(env.session_config)
    ann = env.member("Ann")
    client = env.client()
    _assign(client, directory.name, ann.id)
    assert _listed_ids(client) == []

    listed = client.get(f"/api/speakers/{ann.id}/retroactive-clips")
    assert listed.status_code == 200
    [clip] = listed.json()["clips"]
    assert clip["phrase_index"] == 100
    assert clip["session_id"] == directory.name
    assert clip["speech_ms"] == 2000.0
    assert clip["created_at"]

    audio = client.get(f"/api/speakers/{ann.id}/retroactive-clips/100/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert int(audio.headers["content-length"]) == len(audio.content)

    assert client.delete(f"/api/speakers/{ann.id}/retroactive-clips/100").status_code == 204
    assert client.get(f"/api/speakers/{ann.id}/retroactive-clips").json() == {"clips": []}
    assert env.references.name_for(ann.id) is None
    assert _listed_ids(client) == [directory.name]


def test_clip_routes_refuse_prompted_indices_unknown_members_and_missing_clips(env):
    ann = env.member("Ann")
    env.clip_store.write_clip(ann.id, 2, b"\x00\x00" * 16000)
    client = env.client()

    assert client.get(f"/api/speakers/{ann.id}/retroactive-clips/3/audio").status_code == 404
    assert client.get(f"/api/speakers/{ann.id}/retroactive-clips/100/audio").status_code == 404
    assert client.delete(f"/api/speakers/{ann.id}/retroactive-clips/2").status_code == 404
    assert env.clip_store.clip_path(ann.id, 2).is_file()
    assert client.delete(f"/api/speakers/{ann.id}/retroactive-clips/100").status_code == 404
    assert client.get("/api/speakers/999/retroactive-clips").status_code == 404
    assert client.get("/api/speakers/999/retroactive-clips/100/audio").status_code == 404
    assert client.delete("/api/speakers/999/retroactive-clips/100").status_code == 404


def test_an_operator_gets_403_on_the_clip_routes(env):
    ann = env.member("Ann")
    client = env.client("operator")

    assert client.get(f"/api/speakers/{ann.id}/retroactive-clips").status_code == 403
    assert client.get(f"/api/speakers/{ann.id}/retroactive-clips/100/audio").status_code == 403
    assert client.delete(f"/api/speakers/{ann.id}/retroactive-clips/100").status_code == 403


def test_member_counts_split_prompted_phrases_from_retroactive_clips(env):
    ann = env.member("Ann")
    for index in (0, 1, 2, 100, 101):
        asyncio.run(
            env.repo.upsert_embedding(
                speaker_id=ann.id,
                phrase_index=index,
                model_id=_MODEL_ID,
                vector=[1.0, 0.0],
                created_at=datetime.now(timezone.utc),
            )
        )
    client = env.client()

    [row] = client.get("/api/speakers").json()
    assert row["enrolled_phrases"] == 3
    assert row["retroactive_clips"] == 2

    env.config.speaker_id = _speaker_config(model=None)
    [row] = client.get("/api/speakers").json()
    assert (row["enrolled_phrases"], row["retroactive_clips"]) == (0, 0)
