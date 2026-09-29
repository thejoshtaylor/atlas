"""Retroactive clips from recorded edge turns (quick task 260929-j08).

Direct calls against `atlas.speaker_id.retroactive`: the trim, the format
check, and the store path. No app, no Postgres. The clip store is a real
directory under `tmp_path`, the repository and the embedder are fakes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from atlas.speaker_id.embedding import EmbeddingWorker, SpeakerAudioTooShort
from atlas.speaker_id.enrollment import ClipStore, EnrollmentError
from atlas.speaker_id.matching import ReferenceSet
from atlas.speaker_id.retroactive import (
    RETROACTIVE_CAP,
    RETROACTIVE_MIN_INDEX,
    assigned_session_ids,
    delete_retroactive_clip,
    is_edge_turn,
    store_retroactive_clip,
    trim_turn_speech,
)
from atlas.speaker_id.wiring import refresh_live_reference

from tests.edge_fakes import interleave
from tests.speaker_fakes import FakeEmbedder
from tests.speaker_repo_fakes import FakeSpeakerRepository, fake_speaker

_MODEL_ID = "3dspeaker_speech_campplus_sv_en_voxceleb_16k"
_TRIM = dict(channels=2, frame_samples=256, sample_rate=16000, speech_rms_floor=None, max_seconds=10.0)


def _pcm(value: int, samples: int) -> bytes:
    import struct

    return struct.pack(f"<{samples}h", *([value] * samples))


# -- trim_turn_speech -----------------------------------------------------


def test_trim_keeps_the_asr_channel_only():
    raw = interleave(3000, 1000, 32000)

    pcm, speech_ms = trim_turn_speech(raw, asr_channel=0, **_TRIM)

    assert len(pcm) == 64000
    assert pcm == _pcm(3000, 32000)
    assert speech_ms == 2000.0

    other, _ms = trim_turn_speech(raw, asr_channel=1, **_TRIM)
    assert other == _pcm(1000, 32000)


def test_trim_drops_frames_under_the_rms_floor():
    # 16384 samples is 64 whole 256-sample slices, so no slice mixes loud and quiet.
    raw = interleave(3000, 3000, 16384) + interleave(10, 10, 16384)

    pcm, speech_ms = trim_turn_speech(raw, asr_channel=0, **{**_TRIM, "speech_rms_floor": 0.01})

    assert speech_ms == 1024.0
    assert pcm == _pcm(3000, 16384)


def test_trim_cuts_the_head_to_the_length_cap():
    raw = interleave(3000, 1000, 32000)

    _pcm_out, speech_ms = trim_turn_speech(raw, asr_channel=0, **{**_TRIM, "max_seconds": 1.0})

    assert speech_ms == 1000.0


def test_trim_drops_a_partial_frame_and_an_odd_byte():
    raw = interleave(3000, 1000, 512) + b"\x00\x00\x00"

    pcm, _speech_ms = trim_turn_speech(raw, asr_channel=0, **_TRIM)

    assert pcm == _pcm(3000, 512)


# -- is_edge_turn ---------------------------------------------------------


def test_is_edge_turn_accepts_only_a_16k_pcm_edge_format():
    good = {"encoding": "pcm", "sample_rate": 16000, "channels": 2}

    assert is_edge_turn(good, None, edge_channels=2) is True
    assert is_edge_turn({"encoding": "alaw", "sample_rate": 8000, "channels": 1}, None, edge_channels=2) is False
    assert is_edge_turn({"encoding": "pcm", "sample_rate": 16000, "channels": 1}, None, edge_channels=2) is False
    assert is_edge_turn(None, None, edge_channels=2) is False
    assert is_edge_turn(good, {"detail": "not_edge_source"}, edge_channels=2) is False


# -- store_retroactive_clip ----------------------------------------------


async def _store(state, clip_store, repo, speaker, worker, *, session_id="s-1", pcm=None):
    return await store_retroactive_clip(
        state=state,
        clip_store=clip_store,
        speaker=speaker,
        session_id=session_id,
        pcm16_mono=pcm if pcm is not None else _pcm(3000, 32000),
        speech_ms=2000.0,
        worker=worker,
        speaker_repo=repo,
        model_id=_MODEL_ID,
    )


async def test_first_clip_lands_at_index_100_with_sidecar_row_and_live_reference(tmp_path):
    clip_store = ClipStore(tmp_path / "speakers")
    repo = FakeSpeakerRepository()
    speaker = await repo.create_speaker(display_name="Ann", linked_user_id=None, created_at=datetime.now(timezone.utc))
    references = ReferenceSet()
    state = SimpleNamespace(speaker_id_context=SimpleNamespace(references=references, worker=None))
    worker = EmbeddingWorker(FakeEmbedder())
    try:
        result = await _store(state, clip_store, repo, speaker, worker)
    finally:
        worker.close()

    assert result.phrase_index == RETROACTIVE_MIN_INDEX == 100
    assert clip_store.read_clip(speaker.id, 100) == _pcm(3000, 32000)
    import json

    sidecar = json.loads(clip_store.clip_path(speaker.id, 100).with_suffix(".json").read_text())
    assert sidecar["session_id"] == "s-1"
    assert sidecar["speech_ms"] == 2000.0
    assert (speaker.id, 100, _MODEL_ID) in repo._embeddings
    assert references.name_for(speaker.id) == "Ann"
    assert assigned_session_ids(clip_store) == {"s-1"}


async def test_mode_off_still_writes_the_clip_the_sidecar_and_the_row(tmp_path):
    clip_store = ClipStore(tmp_path / "speakers")
    repo = FakeSpeakerRepository()
    speaker = await repo.create_speaker(display_name="Ann", linked_user_id=None, created_at=datetime.now(timezone.utc))
    state = SimpleNamespace(speaker_id_context=None)
    worker = EmbeddingWorker(FakeEmbedder())
    try:
        await _store(state, clip_store, repo, speaker, worker)
    finally:
        worker.close()

    assert clip_store.clip_path(speaker.id, 100).is_file()
    assert clip_store.clip_path(speaker.id, 100).with_suffix(".json").is_file()
    assert (speaker.id, 100, _MODEL_ID) in repo._embeddings


async def test_second_clip_gets_101_and_prompted_clips_stay_untouched(tmp_path):
    clip_store = ClipStore(tmp_path / "speakers")
    repo = FakeSpeakerRepository()
    speaker = await repo.create_speaker(display_name="Ann", linked_user_id=None, created_at=datetime.now(timezone.utc))
    for index in range(5):
        clip_store.write_clip(speaker.id, index, _pcm(500 + index, 16000))
    state = SimpleNamespace(speaker_id_context=None)
    worker = EmbeddingWorker(FakeEmbedder())
    try:
        first = await _store(state, clip_store, repo, speaker, worker, session_id="s-1")
        second = await _store(state, clip_store, repo, speaker, worker, session_id="s-2")
    finally:
        worker.close()

    assert (first.phrase_index, second.phrase_index) == (100, 101)
    assert clip_store.list_clips(speaker.id) == [0, 1, 2, 3, 4, 100, 101]
    assert clip_store.read_clip(speaker.id, 2) == _pcm(502, 16000)


class _RaisingWorker:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def embed(self, pcm: bytes):
        raise self._exc


async def test_an_embed_failure_leaves_no_clip_and_no_sidecar(tmp_path):
    clip_store = ClipStore(tmp_path / "speakers")
    repo = FakeSpeakerRepository()
    speaker = await repo.create_speaker(display_name="Ann", linked_user_id=None, created_at=datetime.now(timezone.utc))
    state = SimpleNamespace(speaker_id_context=None)

    with pytest.raises(EnrollmentError) as too_short:
        await _store(state, clip_store, repo, speaker, _RaisingWorker(SpeakerAudioTooShort("short")))
    assert too_short.value.status_code == 422
    assert not clip_store.clip_path(speaker.id, 100).exists()
    assert not clip_store.clip_path(speaker.id, 100).with_suffix(".json").exists()

    with pytest.raises(RuntimeError):
        await _store(state, clip_store, repo, speaker, _RaisingWorker(RuntimeError("boom")))
    assert not clip_store.clip_path(speaker.id, 100).exists()
    assert not clip_store.clip_path(speaker.id, 100).with_suffix(".json").exists()


# -- refresh_live_reference ----------------------------------------------


async def test_refresh_upserts_when_vectors_exist_and_removes_when_none(tmp_path):
    repo = FakeSpeakerRepository()
    speaker = await repo.create_speaker(display_name="Ann", linked_user_id=None, created_at=datetime.now(timezone.utc))
    references = ReferenceSet()
    state = SimpleNamespace(speaker_id_context=SimpleNamespace(references=references, worker=None))

    await repo.upsert_embedding(
        speaker_id=speaker.id, phrase_index=100, model_id=_MODEL_ID, vector=[1.0, 0.0], created_at=datetime.now(timezone.utc)
    )
    await refresh_live_reference(state, repo, speaker_id=speaker.id, display_name="Ann", model_id=_MODEL_ID)
    assert references.name_for(speaker.id) == "Ann"

    repo._embeddings.clear()
    await refresh_live_reference(state, repo, speaker_id=speaker.id, display_name="Ann", model_id=_MODEL_ID)
    assert references.name_for(speaker.id) is None


async def test_refresh_does_nothing_without_a_context():
    repo = FakeSpeakerRepository()
    state = SimpleNamespace(speaker_id_context=None)

    await refresh_live_reference(state, repo, speaker_id=1, display_name="Ann", model_id=_MODEL_ID)


# -- cap, delete ------------------------------------------------------------


async def _seed_clip(clip_store, repo, speaker, index: int, *, session_id: str) -> None:
    from atlas.speaker_id.retroactive import write_sidecar

    clip_store.write_clip(speaker.id, index, _pcm(3000, 16000))
    write_sidecar(clip_store, speaker.id, index, session_id=session_id, speech_ms=1000.0)
    await repo.upsert_embedding(
        speaker_id=speaker.id,
        phrase_index=index,
        model_id=_MODEL_ID,
        vector=[1.0, 0.0],
        created_at=datetime.now(timezone.utc),
    )


async def test_delete_embedding_on_the_fake_removes_every_model_row():
    repo = FakeSpeakerRepository()
    speaker = await repo.create_speaker(display_name="Ann", linked_user_id=None, created_at=datetime.now(timezone.utc))
    for index, model_id in ((100, "a"), (100, "b"), (101, "a")):
        await repo.upsert_embedding(
            speaker_id=speaker.id,
            phrase_index=index,
            model_id=model_id,
            vector=[1.0],
            created_at=datetime.now(timezone.utc),
        )

    assert await repo.delete_embedding(speaker_id=speaker.id, phrase_index=100) == 2
    assert await repo.delete_embedding(speaker_id=speaker.id, phrase_index=100) == 0
    assert list(repo._embeddings) == [(speaker.id, 101, "a")]


async def test_the_21st_clip_drops_the_oldest_and_never_a_prompted_phrase(tmp_path):
    assert RETROACTIVE_CAP == 20
    clip_store = ClipStore(tmp_path / "speakers")
    repo = FakeSpeakerRepository()
    speaker = await repo.create_speaker(display_name="Ann", linked_user_id=None, created_at=datetime.now(timezone.utc))
    for index in [*range(5), *range(100, 120)]:
        await _seed_clip(clip_store, repo, speaker, index, session_id=f"s-{index}")
    references = ReferenceSet()
    state = SimpleNamespace(speaker_id_context=SimpleNamespace(references=references, worker=None))
    worker = EmbeddingWorker(FakeEmbedder())
    try:
        result = await _store(state, clip_store, repo, speaker, worker, session_id="s-new")
    finally:
        worker.close()

    assert result.phrase_index == 120
    assert result.dropped_phrase_indices == (100,)
    assert not clip_store.clip_path(speaker.id, 100).exists()
    assert not clip_store.clip_path(speaker.id, 100).with_suffix(".json").exists()
    kept = sorted(key[1] for key in repo._embeddings)
    assert kept == [*range(5), *range(101, 121)]
    assert clip_store.list_clips(speaker.id) == kept
    assert references.name_for(speaker.id) == "Ann"


async def test_delete_removes_the_row_the_files_and_an_only_reference(tmp_path):
    clip_store = ClipStore(tmp_path / "speakers")
    repo = FakeSpeakerRepository()
    speaker = await repo.create_speaker(display_name="Ann", linked_user_id=None, created_at=datetime.now(timezone.utc))
    references = ReferenceSet()
    state = SimpleNamespace(speaker_id_context=SimpleNamespace(references=references, worker=None))
    worker = EmbeddingWorker(FakeEmbedder())
    try:
        await _store(state, clip_store, repo, speaker, worker)
    finally:
        worker.close()
    assert references.name_for(speaker.id) == "Ann"

    deleted = await delete_retroactive_clip(
        state=state, clip_store=clip_store, speaker=speaker, phrase_index=100, speaker_repo=repo, model_id=_MODEL_ID
    )

    assert deleted is True
    assert repo._embeddings == {}
    assert not clip_store.clip_path(speaker.id, 100).exists()
    assert assigned_session_ids(clip_store) == set()
    assert references.name_for(speaker.id) is None

    again = await delete_retroactive_clip(
        state=state, clip_store=clip_store, speaker=speaker, phrase_index=100, speaker_repo=repo, model_id=_MODEL_ID
    )
    assert again is False
