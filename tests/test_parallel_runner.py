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
from atlas.sources.runner import BargeInMonitor, SourceRunner
from atlas.sources.turn_group import ParallelTurns, TurnGroup, TurnHooks
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.turn_context import TURN_GROUP_EVENT, TurnContext

from tests.conftest import (
    BrainReply,
    FakeAudioSource,
    FakeBrain,
    FakeTts,
    FakeWakeEventRepository,
    FakeWakeHit,
    FinalTranscript,
)

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
    refractory_s: float = 0.0,
    **kwargs: Any,
) -> SourceRunner:
    return SourceRunner(
        "edge",
        source,
        detector,
        lambda chunk: chunk,
        run_turn_fn,
        wake_config=WakeConfig(engine="vosk", refractory_s=refractory_s),
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


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true within the timeout")


async def test_a_hit_over_the_cap_starts_no_turn_and_is_recorded_as_turn_cap():
    source = FakeAudioSource([_frame(i) for i in range(FRAME_COUNT)])
    detector = _HitOnCalls(2, 4, 6)
    repo = FakeWakeEventRepository()
    release = asyncio.Event()
    started: list[str] = []

    async def run_turn_fn(turn_source: Any) -> None:
        started.append("turn")
        await release.wait()

    runner = _runner(
        source,
        run_turn_fn,
        detector,
        parallel=ParallelTurns(max_concurrent=2),
        wake_event_repo=repo,
    )
    run_task = asyncio.create_task(runner.run())
    await _wait_until(lambda: detector.calls == FRAME_COUNT)
    assert runner.turn_group is not None
    assert runner.turn_group.live_count == 2
    release.set()
    await asyncio.wait_for(run_task, timeout=5.0)
    await runner.drain_pending_wake_events()

    assert started == ["turn", "turn"]
    blocked = [event for event in repo.events if not event.allowed]
    assert [event.block_reason for event in blocked] == ["turn_cap"]
    assert len([event for event in repo.events if event.allowed]) == 2


async def test_a_hit_while_reply_audio_plays_is_recorded_as_reply_playing_and_a_later_hit_starts_a_turn():
    source = FakeAudioSource([_frame(i) for i in range(FRAME_COUNT)])
    repo = FakeWakeEventRepository()
    started: list[str] = []
    times = iter([100.5, 101.0])

    async def run_turn_fn(turn_source: Any) -> None:
        started.append("turn")

    # The refractory is longer than the gap between the two hits. A blocked
    # hit must not start the refractory clock, or the second hit is blocked.
    runner = _runner(
        source,
        run_turn_fn,
        _HitOnCalls(3, 7),
        parallel=ParallelTurns(max_concurrent=3, reply_echo_tail_s=0.8),
        refractory_s=0.6,
        clock=lambda: next(times),
        wake_event_repo=repo,
    )
    assert runner.turn_group is not None
    runner.turn_group.note_reply_playback(100.0)  # playback ends at 100.0, plus the 0.8 s tail

    await asyncio.wait_for(runner.run(), timeout=5.0)
    await runner.drain_pending_wake_events()

    assert started == ["turn"]
    assert [event.block_reason for event in repo.events if not event.allowed] == ["reply_playing"]


def _group(run_turn, *, spec: ParallelTurns | None = None, blocked: list | None = None) -> TurnGroup:
    async def no_listener(monitor: Any, frames: Any = None) -> None:
        return None

    hooks = TurnHooks(
        run_turn=run_turn,
        new_monitor=lambda: BargeInMonitor(floor=1.0, min_duration_s=0.1, guard_window_s=0.0, enabled=False),
        watch_barge_in=no_listener,
        record_blocked_hit=lambda score, reason, at: (blocked if blocked is not None else []).append(reason),
        clock=lambda: 0.0,
    )
    group = TurnGroup("edge", FakeAudioSource(), spec or ParallelTurns(max_concurrent=3), hooks)
    for index in range(4):
        group.push_frame(_frame(index), index)
    return group


class _Gate:
    """Holds turns open until `open()`, and records each turn's context."""

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.contexts: list[TurnContext] = []

    async def run_turn(self, turn_source: Any) -> None:
        self.contexts.append(turn_source.turn_context)
        await self.event.wait()

    def open(self) -> None:
        self.event.set()

    def reset(self) -> None:
        self.event = asyncio.Event()
        self.contexts.clear()


async def test_turn_keys_group_ids_and_order_frames_across_two_groups():
    gate = _Gate()
    group = _group(gate.run_turn, spec=ParallelTurns(max_concurrent=3, preroll_ms=30))

    group.start_wake_turn(3)  # preroll: three 320-byte frames, so the replay starts at frame 1
    group.start_wake_turn(3)
    await _wait_until(lambda: len(gate.contexts) == 2)
    first, second = gate.contexts
    assert first.turn_key == "edge:1"
    assert second.turn_key == "edge:2"
    assert first.group_id == second.group_id
    assert first.order_frame == 1
    assert first.replay_until is not None
    assert first.replay_until(3) == [_frame(1), _frame(2)]

    gate.open()
    await group.drain()
    assert group.live_count == 0

    gate.contexts.clear()
    group.start_wake_turn(3)
    await _wait_until(lambda: len(gate.contexts) == 1)
    third = gate.contexts[0]
    assert third.turn_key == "edge:3"
    assert third.group_id != first.group_id
    await group.drain()


async def test_claims_factory_runs_once_per_group_and_every_turn_in_the_group_shares_the_object():
    gate = _Gate()
    made: list[object] = []

    def claims_factory() -> object:
        made.append(object())
        return made[-1]

    group = _group(gate.run_turn, spec=ParallelTurns(max_concurrent=3, claims_factory=claims_factory))
    group.start_wake_turn(2)
    group.start_wake_turn(3)
    await _wait_until(lambda: len(gate.contexts) == 2)
    assert len(made) == 1
    assert gate.contexts[0].claims is made[0] and gate.contexts[1].claims is made[0]

    gate.open()
    await group.drain()

    gate.reset()
    gate.open()
    group.start_wake_turn(3)
    await group.drain()
    assert len(made) == 2
    assert gate.contexts[0].claims is made[1]


class _FakeReplyRegistry:
    def __init__(self, events: list[str]) -> None:
        self.registered: list[tuple[str, int, str]] = []
        self.handles: list["_FakeReplyHandle"] = []
        self._events = events

    def register(self, turn_key: str, order_frame: int, *, group_id: str) -> "_FakeReplyHandle":
        self.registered.append((turn_key, order_frame, group_id))
        handle = _FakeReplyHandle(self._events)
        self.handles.append(handle)
        return handle


class _FakeReplyHandle:
    def __init__(self, events: list[str]) -> None:
        self.finished = 0
        self.labels: list[str | None] = []
        self._events = events

    def finish(self) -> None:
        self.finished += 1
        self._events.append("finish")

    def set_label(self, label: str | None) -> None:
        self.labels.append(label)


async def test_each_turn_registers_a_reply_handle_once_and_finishes_it_after_the_turn_ends():
    events: list[str] = []
    registry = _FakeReplyRegistry(events)
    contexts: list[TurnContext] = []

    async def run_turn(turn_source: Any) -> None:
        contexts.append(turn_source.turn_context)
        events.append("turn_end")

    group = _group(run_turn, spec=ParallelTurns(max_concurrent=3, reply_registry=registry))
    group.start_wake_turn(2)
    await group.drain()

    assert len(registry.registered) == 1
    turn_key, order_frame, group_id = registry.registered[0]
    assert (turn_key, group_id) == (contexts[0].turn_key, contexts[0].group_id)
    assert order_frame == contexts[0].order_frame
    assert contexts[0].reply_group is registry.handles[0]
    assert registry.handles[0].finished == 1
    assert events == ["turn_end", "finish"]


def test_bind_recorder_records_one_turn_group_event_with_no_speaker_label():
    recorded: list[dict] = []
    reply_handle = _FakeReplyHandle([])
    context = TurnContext(turn_key="edge:1", group_id="abc-1", order_frame=7, reply_group=reply_handle)
    context.note_speaker("speaker-9", "Josh")
    context.record_event({"type": "before-binding"})  # a no-op before binding

    context.bind_recorder(recorded.append)
    context.record_event({"type": "later"})

    assert reply_handle.labels == ["Josh"]
    assert recorded[0] == {
        "type": TURN_GROUP_EVENT,
        "group_id": "abc-1",
        "turn_key": "edge:1",
        "order_frame": 7,
        "split_part": False,
        "follow_up": False,
    }
    assert "Josh" not in str(recorded[0])
    assert recorded[1] == {"type": "later"}
    assert len(recorded) == 2
