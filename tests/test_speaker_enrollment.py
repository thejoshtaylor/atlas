"""Admin enrollment through the real edge microphone (D-02, D-03, D-04,
Phase 11, plan 11-07).

Most cases here drive `EnrollmentCapture`/`enroll_phrase` directly against
a small stub `edge_source` -- every `on_vad_start`/`on_frame`/`on_vad_end`
call and the coroutine awaiting `capture.wait()` share one event loop (the
test's own), so there is no cross-loop hazard to work around. The one
`type="tracer"` scenario below is the exception: it boots the real
application through `lifespan`, with a real `EdgeAudioSource` serving a
real `FakeEdgeSocket` for device 1, proving the whole stack once, for
real -- driven through `TestClient`'s own portal (`client.portal.
start_task_soon`/`call`) so the socket-driving task and the HTTP request
share the exact loop production code runs on, never two different ones
(a plain `asyncio.create_task` on the test's own loop would deadlock
against a blocking `client.post()` call, since `capture.wait()` would
then depend on a Future no other thread can safely resolve).
"""

from __future__ import annotations

import asyncio
import struct
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient as StarletteTestClient

import test_startup_smoke as smoke
from atlas.config import GateConfig, SpeakerIdConfig, WakeConfig
from atlas.sources.runner import SourceRunner
from atlas.speaker_id.enrollment import (
    ClipStore,
    EnrollmentCapture,
    EnrollmentError,
    EnrollmentTimeout,
    enroll_phrase,
)
from atlas.transports.base import SourceFormat
from atlas_edge.protocol import vad_end, vad_start

from tests.conftest import FakeAudioSource
from tests.edge_fakes import FakeEdgeDeviceRepository, FakeEdgeSocket, fake_edge_device, interleave
from tests.speaker_fakes import FakeEmbedder
from tests.speaker_repo_fakes import FakeSpeakerRepository, fake_speaker

_CAMPPLUS_MODEL_ID = "3dspeaker_speech_campplus_sv_en_voxceleb_16k"


def _reference_vector_for(value: int, num_samples: int = 250) -> "list[float]":
    """The exact vector a fresh `FakeEmbedder` produces for a constant
    PCM16 buffer at `value` -- deterministic across instances
    (`tests/speaker_fakes.py`), so this can be computed independently of
    whatever embedder instance a capture actually used."""
    buffer = struct.pack(f"<{num_samples}h", *([value] * num_samples))
    return list(FakeEmbedder().embed(buffer))


def _speaker_config(**overrides: object) -> SpeakerIdConfig:
    base = dict(
        mode="record",
        model="campplus",
        threshold=0.5,
        window_ms=500,
        min_window_ms=250,
        speech_rms_floor=0.01,
        change_similarity_floor=0.3,
        enrollment_gap_ms=300,
        enrollment_start_timeout_s=2.0,
        enrollment_max_phrase_s=2.0,
        min_enrollment_speech_ms=500,
    )
    base.update(overrides)
    return SpeakerIdConfig(**base)


class _StubEdgeSource:
    """What `enroll_phrase` needs from an `EdgeAudioSource` (D-02) -- a
    plain stub, not the real class, so most tests below never need a real
    `EdgeAudioSource`/`FakeEdgeSocket` at all. `listener` is captured for
    the test to drive directly (`on_vad_start`/`on_frame`/`on_vad_end`)."""

    def __init__(self, *, connected_device_id: "int | None" = 1, channels: int = 2, asr_channel: int = 1) -> None:
        self.connected_device_id = connected_device_id
        self.wake_suppressed = False
        self.listener: object = None
        self._channels = channels
        self._asr_channel = asr_channel

    def add_listener(self, listener: object):
        self.listener = listener

        def _unsubscribe() -> None:
            self.listener = None

        return _unsubscribe

    def source_format(self) -> SourceFormat:
        return SourceFormat("pcm", 16000, channels=self._channels, asr_channel=self._asr_channel)


def _state_for(
    *,
    edge_source: object,
    speaker_config: SpeakerIdConfig,
    clip_store: ClipStore,
    speaker_id_context: "object | None" = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        edge_source=edge_source,
        speaker_enrollment_in_progress=False,
        speaker_clip_store=clip_store,
        speaker_id_context=speaker_id_context,
        speaker_embedder_factory=lambda config: FakeEmbedder(),
    )


async def _push_frames(listener: object, *, value: int, num_frames: int, ch0: int = 1000) -> None:
    frame = interleave(ch0, value, 256)
    for index in range(num_frames):
        listener.on_frame(frame, index, 0.0)


# --- EnrollmentCapture / enroll_phrase, driven directly ------------------


async def test_a_captured_phrase_is_written_embedded_and_reaches_the_live_reference_set(tmp_path):
    from atlas.speaker_id.matching import ReferenceSet
    from atlas.speaker_id.turn_gate import SpeakerIdTurnContext

    stub = _StubEdgeSource()
    speaker_config = _speaker_config()
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1, display_name="Member A")
    speaker_repo._next_id = 2
    references = ReferenceSet()
    context = SpeakerIdTurnContext(
        tracker=None, references=references, mode="record", threshold=0.5,
        model_id=_CAMPPLUS_MODEL_ID, worker=None,
    )
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store, speaker_id_context=context)

    task = asyncio.create_task(
        enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    )
    await asyncio.sleep(0)
    listener = stub.listener
    assert listener is not None
    assert stub.wake_suppressed is True
    assert state.speaker_enrollment_in_progress is True

    listener.on_vad_start(1, 0.0)
    await _push_frames(listener, value=-2000, num_frames=40)  # 40 * 16ms = 640ms >= 500ms
    listener.on_vad_end(2, 0.0)

    result = await asyncio.wait_for(task, timeout=2.0)
    assert result.speaker_id == 1
    assert result.phrase_index == 0
    assert result.speech_ms >= 500

    assert stub.wake_suppressed is False
    assert state.speaker_enrollment_in_progress is False
    assert stub.listener is None  # detached in `finally`

    clip_path = clip_store.clip_path(1, 0)
    assert clip_path.is_file()
    with wave.open(str(clip_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000

    key = (1, 0, _CAMPPLUS_MODEL_ID)
    assert key in speaker_repo._embeddings

    reference_vector = _reference_vector_for(-2000)
    match = references.match(reference_vector)
    assert match.best_speaker_id == 1
    assert match.best_score is not None and match.best_score > 0.99


async def test_a_phrase_read_as_two_segments_within_the_gap_is_one_clip(tmp_path):
    stub = _StubEdgeSource()
    speaker_config = _speaker_config(enrollment_gap_ms=800)
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    task = asyncio.create_task(
        enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    )
    await asyncio.sleep(0)
    listener = stub.listener

    listener.on_vad_start(1, 0.0)
    await _push_frames(listener, value=-2000, num_frames=20)  # 320ms
    listener.on_vad_end(2, 0.0)
    await asyncio.sleep(0.4)  # 400ms < enrollment_gap_ms=800ms -- still the same phrase
    listener.on_vad_start(3, 0.0)
    await _push_frames(listener, value=-2000, num_frames=20)  # another 320ms
    listener.on_vad_end(4, 0.0)

    result = await asyncio.wait_for(task, timeout=2.0)
    # Both segments joined into one clip: ~640ms, not 320ms from either alone.
    assert result.speech_ms >= 600


async def test_a_segment_already_in_progress_when_attaching_is_ignored(tmp_path):
    stub = _StubEdgeSource()
    speaker_config = _speaker_config()
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    task = asyncio.create_task(
        enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    )
    await asyncio.sleep(0)
    listener = stub.listener

    # Frames from a segment already in progress when the capture attached
    # -- no `on_vad_start` for it was ever delivered to this listener.
    await _push_frames(listener, value=-9000, num_frames=10)
    listener.on_vad_end(1, 0.0)  # closes the ignored segment; still no target segment.

    # The real, targeted segment starts here.
    listener.on_vad_start(2, 0.0)
    await _push_frames(listener, value=-2000, num_frames=40)
    listener.on_vad_end(3, 0.0)

    result = await asyncio.wait_for(task, timeout=2.0)
    pcm = clip_store.read_clip(1, 0)
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    assert set(samples) == {-2000}, "the ignored in-progress segment's frames must not appear in the clip"
    assert result.speech_ms >= 500


async def test_no_speech_within_the_start_timeout_raises_enrollment_timeout(tmp_path):
    stub = _StubEdgeSource()
    speaker_config = _speaker_config(enrollment_start_timeout_s=0.05)
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    with pytest.raises(EnrollmentTimeout):
        await asyncio.wait_for(
            enroll_phrase(
                state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
                speaker_repo=speaker_repo, speaker_config=speaker_config,
            ),
            timeout=2.0,
        )
    assert stub.wake_suppressed is False
    assert state.speaker_enrollment_in_progress is False


async def test_speech_shorter_than_the_minimum_gives_too_short_error(tmp_path):
    stub = _StubEdgeSource()
    speaker_config = _speaker_config(min_enrollment_speech_ms=1000)
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    task = asyncio.create_task(
        enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    )
    await asyncio.sleep(0)
    listener = stub.listener
    listener.on_vad_start(1, 0.0)
    await _push_frames(listener, value=-2000, num_frames=18)  # 18*16ms = 288ms < 1000ms
    listener.on_vad_end(2, 0.0)

    with pytest.raises(EnrollmentError) as exc_info:
        await asyncio.wait_for(task, timeout=2.0)
    assert exc_info.value.status_code == 422
    assert "too short" in exc_info.value.detail

    # The flag and wake_suppressed clear after a 422 too (not only success).
    assert stub.wake_suppressed is False
    assert state.speaker_enrollment_in_progress is False


async def test_speech_longer_than_the_max_phrase_completes_at_that_length(tmp_path):
    stub = _StubEdgeSource()
    speaker_config = _speaker_config(enrollment_max_phrase_s=0.5)
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    task = asyncio.create_task(
        enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    )
    await asyncio.sleep(0)
    listener = stub.listener
    listener.on_vad_start(1, 0.0)
    # Well past 0.5s -- the capture must complete at the cap, without a
    # `vad.end` ever arriving.
    await _push_frames(listener, value=-2000, num_frames=200)

    result = await asyncio.wait_for(task, timeout=2.0)
    assert 480 <= result.speech_ms <= 520


async def test_a_second_enrollment_while_one_is_recording_gives_409(tmp_path):
    stub = _StubEdgeSource()
    speaker_config = _speaker_config()
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    first = asyncio.create_task(
        enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    )
    await asyncio.sleep(0)
    assert state.speaker_enrollment_in_progress is True

    with pytest.raises(EnrollmentError) as exc_info:
        await enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=1, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    assert exc_info.value.status_code == 409
    assert "already recording" in exc_info.value.detail

    # Finish the first one and confirm the flag clears after a genuine
    # success too (not only after the second request's 409).
    listener = stub.listener
    listener.on_vad_start(1, 0.0)
    await _push_frames(listener, value=-2000, num_frames=40)
    listener.on_vad_end(2, 0.0)
    await asyncio.wait_for(first, timeout=2.0)
    assert state.speaker_enrollment_in_progress is False


async def test_the_flag_clears_after_an_unhandled_exception_too(tmp_path):
    """The flag and `wake_suppressed` clear on every exit path, including
    one this function itself never raises -- `enroll_phrase`'s own
    `finally` around the capture, not a `try`/`except` naming specific
    error types."""
    stub = _StubEdgeSource()
    speaker_config = _speaker_config()
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2

    class _ExplodingClipStore(ClipStore):
        def write_clip(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("disk exploded")

    state = _state_for(
        edge_source=stub, speaker_config=speaker_config, clip_store=_ExplodingClipStore(tmp_path / "speakers")
    )

    task = asyncio.create_task(
        enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    )
    await asyncio.sleep(0)
    listener = stub.listener
    listener.on_vad_start(1, 0.0)
    await _push_frames(listener, value=-2000, num_frames=40)
    listener.on_vad_end(2, 0.0)

    with pytest.raises(RuntimeError, match="disk exploded"):
        await asyncio.wait_for(task, timeout=2.0)

    assert stub.wake_suppressed is False
    assert state.speaker_enrollment_in_progress is False


async def test_device_id_2_while_device_1_is_connected_gives_409(tmp_path):
    stub = _StubEdgeSource(connected_device_id=1)
    speaker_config = _speaker_config()
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    with pytest.raises(EnrollmentError) as exc_info:
        await enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=2,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    assert exc_info.value.status_code == 409
    assert "not connected" in exc_info.value.detail


async def test_no_edge_source_gives_409(tmp_path):
    speaker_config = _speaker_config()
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    state = _state_for(edge_source=None, speaker_config=speaker_config, clip_store=clip_store)

    with pytest.raises(EnrollmentError) as exc_info:
        await enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    assert exc_info.value.status_code == 409
    assert "not the configured audio source" in exc_info.value.detail


async def test_speaker_id_model_not_set_gives_409(tmp_path):
    stub = _StubEdgeSource()
    speaker_config = _speaker_config(mode="off", model=None)
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    with pytest.raises(EnrollmentError) as exc_info:
        await enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    assert exc_info.value.status_code == 409
    assert "speaker_id.model is not set" in exc_info.value.detail


async def test_the_model_file_missing_gives_409(tmp_path):
    from atlas.speaker_id.embedding import SpeakerModelError

    stub = _StubEdgeSource()
    speaker_config = _speaker_config()
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    def _raising_factory(config: object) -> object:
        raise SpeakerModelError("no speaker embedding model at /models/speaker-id/missing.onnx")

    state.speaker_embedder_factory = _raising_factory

    with pytest.raises(EnrollmentError) as exc_info:
        await enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    assert exc_info.value.status_code == 409
    assert "model file is missing" in exc_info.value.detail


async def test_enrollment_works_in_mode_off_using_a_lazily_built_cached_worker(tmp_path):
    """D-10: an operator enrolls before switching modes on. Mode `"off"`
    means `state.speaker_id_context` carries no worker at all (`wiring.py::
    build_speaker_context`'s own contract) -- `ensure_embedding_worker`
    must build and cache one itself, via `state.speaker_embedder_factory`."""
    stub = _StubEdgeSource()
    speaker_config = _speaker_config(mode="off")
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2

    built_embedders: "list[FakeEmbedder]" = []

    def _factory(config: object) -> FakeEmbedder:
        embedder = FakeEmbedder()
        built_embedders.append(embedder)
        return embedder

    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)
    state.speaker_embedder_factory = _factory

    task = asyncio.create_task(
        enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    )
    await asyncio.sleep(0)
    listener = stub.listener
    listener.on_vad_start(1, 0.0)
    await _push_frames(listener, value=-2000, num_frames=40)
    listener.on_vad_end(2, 0.0)
    await asyncio.wait_for(task, timeout=2.0)

    assert len(built_embedders) == 1

    # A second call reuses the same cached worker -- no second embedder built.
    task2 = asyncio.create_task(
        enroll_phrase(
            state=state, speaker=speaker_repo._speakers[1], phrase_index=1, device_id=1,
            speaker_repo=speaker_repo, speaker_config=speaker_config,
        )
    )
    await asyncio.sleep(0)
    listener2 = stub.listener
    listener2.on_vad_start(1, 0.0)
    await _push_frames(listener2, value=-2000, num_frames=40)
    listener2.on_vad_end(2, 0.0)
    await asyncio.wait_for(task2, timeout=2.0)

    assert len(built_embedders) == 1


async def test_recording_phrase_0_again_replaces_the_clip_and_the_row(tmp_path):
    stub = _StubEdgeSource()
    speaker_config = _speaker_config()
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2
    state = _state_for(edge_source=stub, speaker_config=speaker_config, clip_store=clip_store)

    async def _enroll(value: int) -> None:
        task = asyncio.create_task(
            enroll_phrase(
                state=state, speaker=speaker_repo._speakers[1], phrase_index=0, device_id=1,
                speaker_repo=speaker_repo, speaker_config=speaker_config,
            )
        )
        await asyncio.sleep(0)
        listener = stub.listener
        listener.on_vad_start(1, 0.0)
        await _push_frames(listener, value=value, num_frames=40)
        listener.on_vad_end(2, 0.0)
        await asyncio.wait_for(task, timeout=2.0)

    await _enroll(-2000)
    counts_after_first = await speaker_repo.count_embeddings(_CAMPPLUS_MODEL_ID)
    assert counts_after_first[1] == 1

    await _enroll(-4000)
    counts_after_second = await speaker_repo.count_embeddings(_CAMPPLUS_MODEL_ID)
    assert counts_after_second[1] == 1  # still one row -- replaced, not appended.

    pcm = clip_store.read_clip(1, 0)
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    assert set(samples) == {-4000}, "the clip must hold only the second recording's audio"


# --- ClipStore path safety ------------------------------------------------


def test_speaker_dir_refuses_a_negative_id(tmp_path):
    store = ClipStore(tmp_path / "speakers")
    with pytest.raises(ValueError):
        store.speaker_dir(-1)


def test_speaker_dir_refuses_a_path_that_would_escape_the_root(tmp_path):
    """A non-negative integer id can never itself produce a `..` segment
    -- the realistic escape is a symlink planted under the root, which
    `Path.resolve()` follows before this method's own containment check
    runs."""
    root = tmp_path / "speakers"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "1").symlink_to(outside)
    store = ClipStore(root)
    with pytest.raises(ValueError):
        store.speaker_dir(1)


def test_write_clip_then_read_clip_round_trips(tmp_path):
    store = ClipStore(tmp_path / "speakers")
    pcm = struct.pack("<4h", 1, 2, 3, 4)
    store.write_clip(1, 0, pcm, sample_rate=16000)
    assert store.read_clip(1, 0) == pcm


def test_list_clips_returns_every_recorded_phrase_index_sorted(tmp_path):
    store = ClipStore(tmp_path / "speakers")
    store.write_clip(1, 3, b"\x00\x00", sample_rate=16000)
    store.write_clip(1, 0, b"\x00\x00", sample_rate=16000)
    assert store.list_clips(1) == [0, 3]
    assert store.list_clips(2) == []


def test_delete_speaker_dir_removes_every_clip(tmp_path):
    store = ClipStore(tmp_path / "speakers")
    store.write_clip(1, 0, b"\x00\x00", sample_rate=16000)
    store.delete_speaker_dir(1)
    assert not store.speaker_dir(1).exists()
    store.delete_speaker_dir(1)  # idempotent -- no error on an already-gone directory.


# --- wake_suppressed at the SourceRunner level ----------------------------


class _AlwaysHitWakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, chunk: bytes) -> object:
        self.calls += 1
        raise AssertionError("the wake detector must never be called while wake_suppressed is true")


class _RecordingPreroll:
    def __init__(self) -> None:
        self.pushed: "list[bytes]" = []

    def push(self, chunk: bytes) -> None:
        self.pushed.append(chunk)


async def test_wake_suppressed_source_never_reaches_the_wake_detector_but_still_feeds_preroll():
    source = FakeAudioSource(frames=[b"\x00\x01", b"\x02\x03"], channels=2)
    source.wake_suppressed = True
    detector = _AlwaysHitWakeDetector()
    preroll = _RecordingPreroll()

    turns_started: "list[object]" = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    runner = SourceRunner(
        "edge",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=WakeConfig(engine="vosk", refractory_s=0.0),
        gate_config=GateConfig(),
        preroll=preroll,
    )
    await runner.run()

    assert detector.calls == 0
    assert turns_started == []
    assert preroll.pushed == [b"\x00\x01", b"\x02\x03"]


# --- The tracer: a real EdgeAudioSource serving a real FakeEdgeSocket ------


def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition never became true within the timeout")


class _NoOpEdgeWakeDetector:
    """Stands in for `VoskWakeDetector` -- never actually reached while
    `wake_suppressed` is true, but `lifespan` builds one unconditionally
    for the edge source regardless of whether an enrollment ever runs."""

    def process(self, chunk: bytes) -> None:
        return None

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


def _fake_build_edge_wake_detector(wake_config: object) -> _NoOpEdgeWakeDetector:
    return _NoOpEdgeWakeDetector()


def _create_admin(client: StarletteTestClient) -> None:
    response = client.post(
        "/api/auth/create-admin",
        json={
            "email": "enrollment-admin@example.invalid",
            "display_name": "Enrollment Admin",
            "password": "a-plainly-fictional-test-password",
        },
    )
    assert response.status_code == 201, response.text


def _boot_with_speaker_enrollment(tmp_path, monkeypatch):
    """`test_auth_setup._boot_with_empty_accounts`'s app (an empty account
    repository, so `_create_admin` below succeeds), plus `audio_source:
    edge`, an `edge_device_repo` holding device 1, a `speaker_repo`, and
    the `edge`/`speaker_id` config sections enrollment needs -- layered on
    top the same way `tests/test_speakers_route.py::_boot_with_speaker_repo`
    layers a `speaker_repo` onto that same base."""
    import atlas.app as app_module
    import test_auth_setup
    from atlas.db.repository import Setting
    from datetime import datetime, timezone

    client = test_auth_setup._boot_with_empty_accounts(tmp_path, monkeypatch)

    speaker_repo = FakeSpeakerRepository()
    edge_device_repo = FakeEdgeDeviceRepository()
    edge_device_repo.add("unused-hash-val", fake_edge_device(device_id=1))
    original_build_repositories = app_module._build_repositories

    def _repositories(config: object, engine: object) -> dict:
        repositories = original_build_repositories(config, engine)
        repositories["settings_repo"].settings["audio_source"] = Setting(
            id=1,
            key="audio_source",
            value="edge",
            updated_at=datetime.now(timezone.utc),
            updated_by_user_id=None,
        )
        repositories["edge_device_repo"] = edge_device_repo
        repositories["speaker_repo"] = speaker_repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _repositories)
    # `_boot_with_empty_accounts` already wrote a base config -- this
    # overwrite replaces `CONFIG_PATH` with one that also carries the
    # `edge`/`speaker_id` sections enrollment needs. `lifespan` reads
    # `CONFIG_PATH` fresh only once `with client:` actually starts it, so
    # re-pointing it here (before that happens) is not a race.
    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(
            smoke._write_fake_config(
                tmp_path,
                extra={
                    "edge": {"asr_channel": 1, "pre_roll_ms": 200, "tail_ms": 300},
                    "speaker_id": {
                        "mode": "record",
                        "model": "campplus",
                        "threshold": 0.5,
                        "window_ms": 500,
                        "min_window_ms": 250,
                        "speech_rms_floor": 0.01,
                        "change_similarity_floor": 0.3,
                        "enrollment_dir": str(tmp_path / "speakers"),
                        "enrollment_gap_ms": 300,
                        "enrollment_start_timeout_s": 5.0,
                        "min_enrollment_speech_ms": 1000,
                    },
                },
            )
        ),
    )
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_edge_wake_detector)

    created_embedders: "list[FakeEmbedder]" = []

    def _fake_build_speaker_embedder(speaker_config: object) -> FakeEmbedder:
        embedder = FakeEmbedder()
        created_embedders.append(embedder)
        return embedder

    monkeypatch.setattr(app_module, "_build_speaker_embedder", _fake_build_speaker_embedder)

    return client, speaker_repo, created_embedders


def test_tracer_an_admin_enrolls_a_phrase_through_a_real_edge_source(tmp_path, monkeypatch):
    client, speaker_repo, created_embedders = _boot_with_speaker_enrollment(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        speaker_id = client.post("/api/speakers", json={"display_name": "Member A"}).json()["id"]

        import atlas.app as app_module

        edge_source = app_module.app.state.edge_source
        assert edge_source is not None

        device = fake_edge_device(device_id=1)
        socket = FakeEdgeSocket()
        # Runs `edge_source.serve(...)` on the portal's own loop -- the
        # exact loop the enrollment route below also runs on, avoiding
        # the cross-loop Future hazard a bare `asyncio.create_task` on
        # this (calling) thread's own loop would create.
        serve_future = client.portal.start_task_soon(edge_source.serve, socket, device)
        try:
            _wait_until(lambda: socket.sent_text != [])
            assert edge_source.connected_device_id == 1

            response_holder: dict = {}

            def _do_post() -> None:
                response_holder["response"] = client.post(
                    f"/api/speakers/{speaker_id}/enrollment/0", json={"device_id": 1}
                )

            post_thread = threading.Thread(target=_do_post)
            post_thread.start()
            try:
                _wait_until(lambda: app_module.app.state.speaker_enrollment_in_progress)
                assert edge_source.wake_suppressed is True

                client.portal.call(socket.push_text, vad_start(1))
                frame = interleave(1000, -2000, 256)
                for _ in range(125):  # 125 * 16ms = 2000ms (D-09: "near 2000").
                    client.portal.call(socket.push_bytes, frame)
                client.portal.call(socket.push_text, vad_end(2))
            finally:
                post_thread.join(timeout=15.0)
                assert not post_thread.is_alive(), "the enrollment request never completed"

            response = response_holder["response"]
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["speaker_id"] == speaker_id
            assert body["phrase_index"] == 0
            assert 1900 <= body["speech_ms"] <= 2100
            assert body["enrolled_phrases"] == 1
            assert body["required_phrases"] == 5

            assert edge_source.wake_suppressed is False
            assert app_module.app.state.speaker_enrollment_in_progress is False

            clip_path = Path(tmp_path / "speakers" / str(speaker_id) / "phrase-0.wav")
            assert clip_path.is_file()
            with wave.open(str(clip_path), "rb") as wav_file:
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == 2
                assert wav_file.getframerate() == 16000

            assert (speaker_id, 0, _CAMPPLUS_MODEL_ID) in speaker_repo._embeddings
            assert created_embedders, "no speaker embedder was ever built"

            speaker_id_context = app_module.app.state.speaker_id_context
            assert speaker_id_context is not None
            reference_vector = _reference_vector_for(-2000)
            match = speaker_id_context.references.match(reference_vector)
            assert match.best_speaker_id == speaker_id
            assert match.best_score is not None and match.best_score > 0.99

            # An operator session gets 403 on both new routes.
            from atlas.auth.tokens import issue_access_token
            from atlas.config import SecurityConfig

            security = SecurityConfig()
            operator = asyncio.run(
                app_module.app.state.account_repo.create_user(
                    email="enrollment-operator@example.invalid",
                    display_name="Enrollment Operator",
                    password_hash="not-checked-by-this-test",
                    role="operator",
                )
            )
            operator_cookie = issue_access_token(user_id=operator.id, role="operator", security=security)
            client.cookies.set(security.cookie_name, operator_cookie)
            assert client.get("/api/speakers/enrollment-phrases").status_code == 403
            assert (
                client.post(f"/api/speakers/{speaker_id}/enrollment/1", json={"device_id": 1}).status_code
                == 403
            )
        finally:
            serve_future.cancel()
