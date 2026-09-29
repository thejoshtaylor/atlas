"""Parallel turns on one edge source (Phase 12, plan 12-02).

A `SourceRunner` built with a `ParallelTurns` spec keeps reading frames and
detecting wake words while a turn runs, so a second wake hit starts a second
turn at once (D-01). These tests drive the real `run_turn` with the conftest
fakes and the real `SourceRunner`; only the wake detector is scripted.
"""

from __future__ import annotations

import asyncio
from typing import Any

from atlas.config import GateConfig, WakeConfig
from atlas.sources.runner import SourceRunner
from atlas.sources.turn_group import ParallelTurns
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn

from tests.conftest import BrainReply, FakeAudioSource, FakeBrain, FakeTts, FakeWakeHit, FinalTranscript

# 10 ms of 16 kHz mono PCM16 (`FakeAudioSource.source_format`): 32 bytes per ms.
CHUNK_BYTES = 320
FRAME_COUNT = 12
PREROLL_MS = 30  # 960 bytes: three whole chunks


def _frame(index: int) -> bytes:
    return bytes([index]) * CHUNK_BYTES


class _CountingSource(FakeAudioSource):
    """Counts `frames()` calls: the parallel path must call it exactly once."""

    def __init__(self, frames: list[bytes]) -> None:
        super().__init__(frames)
        self.frames_calls = 0

    async def frames(self):
        self.frames_calls += 1
        async for chunk in super().frames():
            yield chunk


class _HitOnCalls:
    """Hits on the listed 0-based `process()` call numbers."""

    def __init__(self, *hit_calls: int) -> None:
        self._hit_calls = set(hit_calls)
        self.calls = 0

    def process(self, chunk: bytes) -> FakeWakeHit | None:
        call = self.calls
        self.calls += 1
        return FakeWakeHit(score=1.0) if call in self._hit_calls else None

    def close(self) -> None:
        pass


class _UncalledToolHost:
    async def call_tool(self, name: str, arguments: dict) -> Any:
        raise AssertionError(f"tool_host.call_tool({name!r}) should never be reached")


class _Overlap:
    """Lets every turn's speech-to-text wait until `expected` turns are live."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.started = 0
        self.all_started = asyncio.Event()

    def enter(self) -> None:
        self.started += 1
        if self.started >= self.expected:
            self.all_started.set()


class _GatedStt:
    """Holds the transcript back until `overlap.expected` turns are live, then
    drains its own frames like `RecordingFakeStt`. Under a serial runner the
    first turn never sees a second one start, so `wait_for` times out."""

    def __init__(self, overlap: _Overlap) -> None:
        self._overlap = overlap
        self.received: list[bytes] = []

    async def stream(self, frames, source_format=None):
        self._overlap.enter()
        await asyncio.wait_for(self._overlap.all_started.wait(), timeout=1.0)
        async for chunk in frames:
            self.received.append(chunk)
        yield FinalTranscript(text="turn on the lamp")


def _runner(
    source: Any,
    run_turn_fn: Any,
    detector: Any,
    *,
    parallel: ParallelTurns | None = None,
    **kwargs: Any,
) -> SourceRunner:
    return SourceRunner(
        "edge",
        source,
        detector,
        lambda chunk: chunk,
        run_turn_fn,
        wake_config=WakeConfig(engine="vosk", refractory_s=0.0),
        gate_config=GateConfig(),
        parallel=parallel,
        **kwargs,
    )


def _real_turn_fn(stts: list[_GatedStt], overlap: _Overlap):
    async def run_turn_fn(source: Any) -> None:
        stt = _GatedStt(overlap)
        stts.append(stt)
        await run_turn(
            source,
            stt,
            FakeBrain(replies=[BrainReply(text="ok")]),
            FakeTts(chunks=[b"reply"]),
            _UncalledToolHost(),
            tools_schema=[],
            system_prompt="test system prompt",
            max_tool_rounds=3,
            timings=TurnTimings(),
            max_utterance_s=5.0,
        )

    return run_turn_fn


async def test_a_second_wake_hit_starts_a_second_turn_while_the_first_is_live():
    source = _CountingSource([_frame(i) for i in range(FRAME_COUNT)])
    overlap = _Overlap(expected=2)
    stts: list[_GatedStt] = []
    runner = _runner(
        source,
        _real_turn_fn(stts, overlap),
        _HitOnCalls(4, 8),
        parallel=ParallelTurns(max_concurrent=3, preroll_ms=PREROLL_MS),
    )

    await asyncio.wait_for(runner.run(), timeout=5.0)

    assert overlap.all_started.is_set(), "the second turn never started while the first was live"
    assert len(stts) == 2
    assert all(stt.received for stt in stts)


async def test_each_turn_reads_its_own_frames_from_its_replay_frame_with_no_gap_and_no_repeat():
    source = _CountingSource([_frame(i) for i in range(FRAME_COUNT)])
    overlap = _Overlap(expected=2)
    stts: list[_GatedStt] = []
    runner = _runner(
        source,
        _real_turn_fn(stts, overlap),
        _HitOnCalls(4, 8),
        parallel=ParallelTurns(max_concurrent=3, preroll_ms=PREROLL_MS),
    )

    await asyncio.wait_for(runner.run(), timeout=5.0)

    # Hit frame 4 minus three preroll frames, plus one: frames 2..11. The
    # second hit is frame 8: frames 6..11.
    assert stts[0].received == [_frame(i) for i in range(2, FRAME_COUNT)]
    assert stts[1].received == [_frame(i) for i in range(6, FRAME_COUNT)]


async def test_run_returns_only_after_both_turns_end_and_reads_the_source_once():
    source = _CountingSource([_frame(i) for i in range(FRAME_COUNT)])
    overlap = _Overlap(expected=2)
    stts: list[_GatedStt] = []
    ended: list[int] = []
    inner = _real_turn_fn(stts, overlap)

    async def run_turn_fn(turn_source: Any) -> None:
        await inner(turn_source)
        ended.append(1)

    runner = _runner(
        source,
        run_turn_fn,
        _HitOnCalls(4, 8),
        parallel=ParallelTurns(max_concurrent=3, preroll_ms=PREROLL_MS),
    )

    await asyncio.wait_for(runner.run(), timeout=5.0)

    assert ended == [1, 1]
    assert source.frames_calls == 1
    assert runner.turn_group is not None
    assert runner.turn_group.live_count == 0


async def test_a_runner_with_no_parallel_spec_awaits_each_turn_before_reading_the_next_chunk():
    source = FakeAudioSource([_frame(i) for i in range(FRAME_COUNT)])
    order: list[str] = []

    async def run_turn_fn(turn_source: Any) -> None:
        order.append("start")
        await asyncio.sleep(0.01)
        order.append("end")

    runner = _runner(source, run_turn_fn, _HitOnCalls(4, 8))

    await asyncio.wait_for(runner.run(), timeout=5.0)

    assert order == ["start", "end", "start", "end"]
    assert runner.turn_group is None
