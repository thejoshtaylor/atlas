"""RED for run_service (10-08-PLAN.md Task 3), with fakes only -- no test
opens a real audio device.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas_edge import protocol
from atlas_edge.client import LiveFrame
from atlas_edge.protocol import Hello
from atlas_edge.service import run_service

FRAME_SAMPLES = 256
CHANNELS = 2


def _make_frame(index: int, channels: int = CHANNELS, frame_samples: int = FRAME_SAMPLES) -> bytes:
    """Frame `index`'s channel 0 is always `10000 + index`, channel 1 is
    always `-(10000 + index)` -- distinct per-channel values so a test can
    prove the gate saw only one channel's samples."""
    interleaved = np.empty((frame_samples, channels), dtype=np.int16)
    interleaved[:, 0] = 10000 + index
    if channels > 1:
        interleaved[:, 1] = -(10000 + index)
    return interleaved.tobytes()


class FakeCapture:
    sample_rate = 16000
    channels = CHANNELS
    frame_samples = FRAME_SAMPLES

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self._batches: "list[list[tuple[bytes, float]]]" = []
        self.frames_calls = 0

    def add_batch(self, frames: "list[tuple[bytes, float]]") -> None:
        self._batches.append(frames)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    async def frames(self):
        batch = self._batches[self.frames_calls]
        self.frames_calls += 1
        for frame, captured_at in batch:
            yield frame, captured_at


class FakeGate:
    def __init__(self, pattern: "list[bool]") -> None:
        self._pattern = list(pattern)
        self.received: "list[bytes]" = []

    def push(self, samples: bytes) -> bool:
        self.received.append(samples)
        return self._pattern.pop(0)


def _hello(**overrides) -> Hello:
    defaults = dict(
        protocol=1,
        device_id=1,
        sample_rate=16000,
        channels=2,
        asr_channel=1,
        pre_roll_ms=48,  # 3 frames at 16ms/frame
        tail_ms=32,  # 2 frames
        frame_samples=256,
    )
    defaults.update(overrides)
    return Hello(**defaults)


def _classify(item, index_by_bytes):
    if isinstance(item, str):
        return ("event", item)
    if isinstance(item, LiveFrame):
        return ("live", index_by_bytes[item.data])
    return ("preroll", index_by_bytes[item])


@pytest.mark.asyncio
async def test_leading_silence_produces_nothing_then_start_preroll_live_end_tail():
    capture = FakeCapture()
    frames = [(_make_frame(i), float(i) * 0.016) for i in range(80)]
    capture.add_batch(frames)
    index_by_bytes = {_make_frame(i): i for i in range(80)}

    pattern = [False] * 30 + [True] * 20 + [False] * 30
    gate = FakeGate(pattern)

    sessions: "list[list]" = []

    async def fake_runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop):
        items = [item async for item in make_outbound(_hello())]
        sessions.append(items)

    await run_service(
        object(),
        capture=capture,
        gate_factory=lambda: gate,
        on_reply_audio=lambda data: None,
        runner=fake_runner,
    )

    assert capture.started
    assert capture.stopped

    items = sessions[0]
    classified = [_classify(item, index_by_bytes) for item in items]

    expected = (
        [("event", protocol.vad_start(0))]
        + [("preroll", 27), ("preroll", 28), ("preroll", 29)]  # pre_roll_frames=3
        + [("live", i) for i in range(30, 50)]  # the 20 speech frames, live
        + [("event", protocol.vad_end(0))]
        + [("live", 50), ("live", 51)]  # tail_frames=2
    )
    assert classified == expected


@pytest.mark.asyncio
async def test_gate_receives_only_the_hello_asr_channel_samples():
    capture = FakeCapture()
    frames = [(_make_frame(i), float(i) * 0.016) for i in range(5)]
    capture.add_batch(frames)
    gate = FakeGate([False] * 5)

    async def fake_runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop):
        [_ async for _ in make_outbound(_hello(asr_channel=1))]

    await run_service(
        object(),
        capture=capture,
        gate_factory=lambda: gate,
        on_reply_audio=lambda data: None,
        runner=fake_runner,
    )

    assert len(gate.received) == 5
    for index, received in enumerate(gate.received):
        values = np.frombuffer(received, dtype=np.int16)
        assert (values == -(10000 + index)).all()  # channel 1's value, never channel 0's


@pytest.mark.asyncio
async def test_hello_format_mismatch_ends_session_with_no_audio(caplog):
    import logging

    capture = FakeCapture()
    capture.add_batch([(_make_frame(0), 0.0)])
    gate = FakeGate([False])

    async def fake_runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop):
        items = [item async for item in make_outbound(_hello(sample_rate=8000))]
        assert items == []

    with caplog.at_level(logging.ERROR):
        await run_service(
            object(),
            capture=capture,
            gate_factory=lambda: gate,
            on_reply_audio=lambda data: None,
            runner=fake_runner,
        )

    assert any("protocol mismatch" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_two_consecutive_sessions_each_get_a_fresh_segmenter():
    capture = FakeCapture()
    # Session 1: silence then speech, but stops mid-speech (no vad.end) --
    # if the segmenter were reused, session 2 would still be "in speech".
    session1_frames = [(_make_frame(i), float(i) * 0.016) for i in range(5)]
    capture.add_batch(session1_frames)
    session2_frames = [(_make_frame(i), float(i) * 0.016) for i in range(5, 10)]
    capture.add_batch(session2_frames)

    gate1 = FakeGate([False, False, True, True, True])
    gate2 = FakeGate([False, False, True, True, True])
    gates = iter([gate1, gate2])

    sessions: "list[list]" = []

    async def fake_runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop):
        for _ in range(2):
            items = [item async for item in make_outbound(_hello())]
            sessions.append(items)

    await run_service(
        object(),
        capture=capture,
        gate_factory=lambda: next(gates),
        on_reply_audio=lambda data: None,
        runner=fake_runner,
    )

    assert sessions[0][0] == protocol.vad_start(0)
    assert sessions[1][0] == protocol.vad_start(0)  # seq resets -- a fresh Segmenter, not seq=1
