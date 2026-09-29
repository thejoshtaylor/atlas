"""`SpeakerTracker`/`TurnSpeakerSpan` unit behavior (11-04-PLAN.md Task 3):
cadence, the flush-mid-speech and no-speech/decision-timeout paths,
bounded history, the synthetic `vad.end` on a dropped connection, and a
raising listener never stopping delivery to a second one.

`tests/test_speaker_tracer.py` already proves the whole pipeline end to
end against a real application; these tests isolate the tracker itself
with a real `SpeechWindowAccumulator` but a controllable embedding
worker, so a window's own cadence and the span's own timeout/flush
decisions can be asserted precisely and fast.
"""

from __future__ import annotations

import asyncio
import struct
import time

import numpy as np
import pytest

from atlas.config import EdgeSourceConfig
from atlas.speaker_id.embedding import EmbeddingWorker
from atlas.speaker_id.tracker import SpeakerTracker

from tests.edge_fakes import FakeEdgeSocket, fake_edge_device


def _frame(value: int, samples: int = 256) -> bytes:
    return struct.pack(f"<{samples}h", *([value] * samples))


class _CountingWorker:
    """Replaces `EmbeddingWorker`: records every submitted PCM buffer and
    resolves synchronously with a fixed vector -- no thread, no real
    embedding, so a test can assert the exact submission count right after
    the call that should have triggered it, with no race to poll for."""

    def __init__(self, dim: int = 4) -> None:
        self.pcm_calls: "list[bytes]" = []
        self._dim = dim

    def submit(self, pcm16_mono: bytes) -> "asyncio.Future[np.ndarray]":
        self.pcm_calls.append(pcm16_mono)
        future: "asyncio.Future[np.ndarray]" = asyncio.get_event_loop().create_future()
        future.set_result(np.ones(self._dim, dtype=np.float32) / np.sqrt(self._dim))
        return future


class _NeverFinishingWorker:
    """`submit()` returns a `Future` nobody ever resolves -- the wedged-
    worker case `decide()`'s own timeout exists for."""

    def submit(self, pcm16_mono: bytes) -> "asyncio.Future[np.ndarray]":
        return asyncio.get_event_loop().create_future()


class _SlowEmbedder:
    """A real, slow embedder: `embed()` blocks the calling thread for
    `delay_s` before returning -- run through a real `EmbeddingWorker` (its
    own dedicated thread), so `ready_at` is a real, later clock reading
    than the moment `decide()` was called, proving `speaker_id_ms` is
    computed from `ready_at`, not from when `decide()` happened to run."""

    dim = 4

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    def embed(self, pcm16_mono: bytes) -> "np.ndarray":
        time.sleep(self._delay_s)
        return np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)


def _tracker(worker, *, window_ms=750, min_window_ms=200, speech_rms_floor=0.01, clock=time.monotonic) -> SpeakerTracker:
    return SpeakerTracker(
        channels=1,
        asr_channel=0,
        sample_rate=16000,
        window_ms=window_ms,
        min_window_ms=min_window_ms,
        speech_rms_floor=speech_rms_floor,
        change_similarity_floor=0.3,
        worker=worker,
        clock=clock,
    )


# --- Cadence -----------------------------------------------------------


async def test_3_2s_of_speech_at_750ms_windows_produces_4_windows_then_a_5th_tail_at_vad_end():
    worker = _CountingWorker()
    tracker = _tracker(worker, window_ms=750, min_window_ms=200)
    tracker.on_vad_start(1, 0.0)

    # Pre-roll: 10 near-silent frames (RMS ~0, below the 0.01 floor) --
    # never enter the accumulator, never submitted.
    for index in range(10):
        tracker.on_frame(_frame(0), index, 0.0)

    # 3.2s of real speech: 200 frames * 16ms = 3200ms at 16kHz/256-sample
    # frames. window_ms=750 -> 12000-sample windows; 4 close during
    # streaming (48000 of 51200 samples), leaving a 3200-sample (200ms)
    # tail >= min_window_ms=200, flushed only at `vad.end`.
    for offset in range(200):
        tracker.on_frame(_frame(2000), 10 + offset, 0.0)

    assert len(worker.pcm_calls) == 4, "4 windows should have closed during the segment itself"

    tracker.on_vad_end(2, 0.0)
    assert len(worker.pcm_calls) == 5, "the 200ms tail (>= min_window_ms) should flush at vad.end"


async def test_a_tail_shorter_than_min_window_ms_is_discarded_not_flushed():
    worker = _CountingWorker()
    tracker = _tracker(worker, window_ms=750, min_window_ms=500)
    tracker.on_vad_start(1, 0.0)
    for offset in range(200):  # 3200ms -- same 4 full windows, 200ms tail.
        tracker.on_frame(_frame(2000), offset, 0.0)
    assert len(worker.pcm_calls) == 4
    tracker.on_vad_end(2, 0.0)
    # 200ms < min_window_ms=500 -- the tail is silently discarded.
    assert len(worker.pcm_calls) == 4


# --- decide(): flush mid-speech, no speech, decision timeout ------------


async def test_decide_while_still_in_speech_flushes_the_partial_window():
    worker = _CountingWorker()
    tracker = _tracker(worker, window_ms=750, min_window_ms=200)
    tracker.on_vad_start(1, 0.0)
    for offset in range(20):  # 320ms of speech -- above min_window_ms, well under one window.
        tracker.on_frame(_frame(2000), offset, 0.0)
    assert worker.pcm_calls == []  # nothing closed on its own yet.

    span = tracker.open_turn()
    measurement = await span.decide(end_of_speech_at=None, references=None)

    assert len(worker.pcm_calls) == 1, "decide() must flush the still-open partial window"
    assert measurement.window_count == 1
    assert measurement.detail is None
    assert measurement.speech_ms == pytest.approx(320.0, abs=1.0)


async def test_decide_with_no_speech_at_all_returns_no_speech_measured():
    worker = _CountingWorker()
    tracker = _tracker(worker, window_ms=750, min_window_ms=200)
    tracker.on_vad_start(1, 0.0)
    tracker.on_vad_end(2, 0.0)  # nothing was ever pushed.

    span = tracker.open_turn()
    measurement = await span.decide(end_of_speech_at=None, references=None)

    assert measurement.detail == "no_speech_measured"
    assert measurement.match is None
    assert measurement.window_count == 0


async def test_a_worker_that_never_finishes_times_out_within_budget():
    worker = _NeverFinishingWorker()
    tracker = _tracker(worker, window_ms=100, min_window_ms=50)
    tracker.on_vad_start(1, 0.0)
    for offset in range(10):  # 160ms -- one full 100ms window submitted.
        tracker.on_frame(_frame(2000), offset, 0.0)
    tracker.on_vad_end(2, 0.0)

    span = tracker.open_turn()
    started = time.monotonic()
    measurement = await span.decide(end_of_speech_at=None, references=None, timeout_s=0.05)
    elapsed = time.monotonic() - started

    assert measurement.detail == "decision_timeout"
    assert measurement.match is None
    assert elapsed < 0.5, "a wedged worker must not hang the turn past its own timeout budget"


async def test_speaker_id_ms_is_computed_from_ready_at_not_from_when_decide_was_called():
    embedder = _SlowEmbedder(delay_s=0.02)
    worker = EmbeddingWorker(embedder)
    try:
        tracker = _tracker(worker, window_ms=750, min_window_ms=100)
        tracker.on_vad_start(1, 0.0)
        for offset in range(20):  # 320ms -- one partial window, flushed at vad.end.
            tracker.on_frame(_frame(2000), offset, 0.0)
        tracker.on_vad_end(2, 0.0)

        span = tracker.open_turn()
        end_of_speech_at = time.monotonic()
        measurement = await span.decide(end_of_speech_at=end_of_speech_at, references=None)

        assert measurement.detail is None
        assert measurement.speaker_id_ms is not None
        assert 0.0 <= measurement.speaker_id_ms < 200.0
    finally:
        worker.close()


# --- Bounded history -----------------------------------------------------


async def test_history_stays_within_32_segments_under_a_long_synthetic_stream():
    worker = _CountingWorker()
    tracker = _tracker(worker, window_ms=750, min_window_ms=200)
    for seq in range(40):  # more than _MAX_SEGMENTS (32).
        tracker.on_vad_start(seq, 0.0)
        tracker.on_frame(_frame(2000), seq * 10, 0.0)
        tracker.on_vad_end(seq, 0.0)
    assert len(tracker._segments) <= 32


async def test_history_stays_within_240_windows_per_segment():
    worker = _CountingWorker()
    # A window closes on every single frame (window_ms == one frame's own
    # 16ms), so 300 frames closes far more than 240 windows fast.
    tracker = _tracker(worker, window_ms=16, min_window_ms=1)
    tracker.on_vad_start(1, 0.0)
    for offset in range(300):
        tracker.on_frame(_frame(2000), offset, 0.0)
    segment = tracker._segments[-1]
    assert len(segment.windows) <= 240


# --- A raising listener never stops a second one ------------------------


def _measured_edge_config(**overrides) -> EdgeSourceConfig:
    base = dict(sample_rate=16000, channels=1, asr_channel=0, pre_roll_ms=200, tail_ms=300)
    base.update(overrides)
    return EdgeSourceConfig(**base)


async def test_a_raising_on_frame_listener_never_stops_a_second_listener():
    from atlas.transports.edge import EdgeAudioSource

    config = _measured_edge_config()
    source = EdgeAudioSource(config)

    class _RaisingListener:
        def on_vad_start(self, seq, at):
            raise RuntimeError("boom")

        def on_frame(self, chunk, frame_index, at):
            raise RuntimeError("boom")

        def on_vad_end(self, seq, at):
            raise RuntimeError("boom")

        def on_wake_hit(self, frame_index):
            raise RuntimeError("boom")

    class _RecordingListener:
        def __init__(self):
            self.frames: "list[bytes]" = []
            self.vad_starts = 0
            self.vad_ends = 0

        def on_vad_start(self, seq, at):
            self.vad_starts += 1

        def on_frame(self, chunk, frame_index, at):
            self.frames.append(chunk)

        def on_vad_end(self, seq, at):
            self.vad_ends += 1

        def on_wake_hit(self, frame_index):
            pass

    source.add_listener(_RaisingListener())
    recorder = _RecordingListener()
    source.add_listener(recorder)

    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    serve_task = asyncio.create_task(source.serve(socket, device))
    await asyncio.sleep(0.01)  # let serve() reach its receive loop.

    import json

    socket.push_text(json.dumps({"type": "vad.start", "seq": 1}))
    socket.push_bytes(_frame(2000))
    socket.push_text(json.dumps({"type": "vad.end", "seq": 2}))
    await asyncio.sleep(0.01)

    assert recorder.vad_starts == 1
    assert recorder.frames == [_frame(2000)]
    assert recorder.vad_ends == 1

    socket.push_disconnect()
    await asyncio.wait_for(serve_task, timeout=2.0)


async def test_a_synthetic_vad_end_on_a_dropped_connection_reaches_listeners():
    from atlas.transports.edge import EdgeAudioSource

    config = _measured_edge_config()
    source = EdgeAudioSource(config)

    class _RecordingListener:
        def __init__(self):
            self.vad_ends: "list[int]" = []

        def on_vad_start(self, seq, at):
            pass

        def on_frame(self, chunk, frame_index, at):
            pass

        def on_vad_end(self, seq, at):
            self.vad_ends.append(seq)

        def on_wake_hit(self, frame_index):
            pass

    recorder = _RecordingListener()
    source.add_listener(recorder)

    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    serve_task = asyncio.create_task(source.serve(socket, device))
    await asyncio.sleep(0.01)

    import json

    socket.push_text(json.dumps({"type": "vad.start", "seq": 1}))
    socket.push_bytes(_frame(2000))
    await asyncio.sleep(0.01)
    # No `vad.end` at all -- the connection just drops.
    socket.push_disconnect()
    await asyncio.wait_for(serve_task, timeout=2.0)

    assert recorder.vad_ends == [1], "a dropped connection mid-segment must synthesize its own vad.end"
