"""Parallel turns on one edge source (Phase 12, plan 12-02).

A `SourceRunner` built with a `ParallelTurns` spec keeps reading frames and
detecting wake words while a turn runs, so a second wake hit starts a second
turn at once (D-01). These tests drive the real `run_turn` with the conftest
fakes and the real `SourceRunner`; only the wake detector is scripted.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from atlas.config import BargeInConfig, EdgeSourceConfig, GateConfig, WakeConfig
from atlas.sources.runner import BargeInMonitor, SourceRunner
from atlas.sources.turn_group import ParallelTurns, TurnGroup, TurnHooks
from atlas.timing import TurnTimings
from atlas.transports.edge import EdgeAudioSource
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import FollowUpRequest
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
from tests.edge_fakes import FakeEdgeSocket, fake_edge_device

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


# --- LED ownership, one follow-up window at a time, barge-in on its own queue (Task 3) ---


def _edge_config() -> EdgeSourceConfig:
    return EdgeSourceConfig(sample_rate=16000, channels=2, asr_channel=1, pre_roll_ms=200, tail_ms=300)


def _led_states(socket: FakeEdgeSocket) -> list[str]:
    return [json.loads(text)["state"] for text in socket.sent_text if json.loads(text).get("type") == "led"]


async def _serve_edge(source: EdgeAudioSource, socket: FakeEdgeSocket) -> "asyncio.Task[None]":
    task = asyncio.create_task(source.serve(socket, fake_edge_device(device_id=1)))
    await _wait_until(lambda: socket.sent_text != [])
    return task


async def _disconnect_edge(socket: FakeEdgeSocket, task: "asyncio.Task[None]") -> None:
    socket.push_disconnect()
    await asyncio.wait_for(task, timeout=2.0)


def _edge_runner(source: EdgeAudioSource, run_turn_fn: Any) -> SourceRunner:
    runner = _runner(source, run_turn_fn, _HitOnCalls(), parallel=ParallelTurns(max_concurrent=3))
    assert runner.turn_group is not None
    runner.turn_group.push_frame(bytes(1024), 0)
    return runner


async def test_the_edge_led_state_property_and_no_thinking_while_replying():
    source = EdgeAudioSource(_edge_config())
    socket = FakeEdgeSocket()
    task = await _serve_edge(source, socket)
    assert source.led_state == "idle"

    await source.send_event({"type": "transcript.final"})
    assert source.led_state == "thinking"
    await source.send_audio(b"reply")
    assert source.led_state == "replying"
    await source.send_event({"type": "transcript.final"})  # a second turn's final transcript

    assert source.led_state == "replying"
    assert _led_states(socket) == ["thinking", "replying"]
    await _disconnect_edge(socket, task)


async def test_overlapping_turns_show_listening_once_then_thinking_then_replying_and_idle_after_the_last():
    source = EdgeAudioSource(_edge_config())
    socket = FakeEdgeSocket()
    task = await _serve_edge(source, socket)
    b_started, a_replying, release_a = asyncio.Event(), asyncio.Event(), asyncio.Event()
    turns = iter(["a", "b"])

    async def run_turn_fn(turn_source: Any) -> None:
        if next(turns) == "a":
            await b_started.wait()
            await source.send_event({"type": "transcript.final"})
            await source.send_audio(b"a-reply")
            a_replying.set()
            await release_a.wait()
        else:
            b_started.set()
            await a_replying.wait()
            await source.send_event({"type": "transcript.final"})
            await source.send_audio(b"b-reply")

    runner = _edge_runner(source, run_turn_fn)
    group = runner.turn_group
    assert group is not None
    group.start_wake_turn(0)
    group.start_wake_turn(0)
    await _wait_until(lambda: b_started.is_set() and group.live_count == 1)

    # The second turn ended while the first is still live: no idle yet.
    assert _led_states(socket) == ["listening", "thinking", "replying"]
    release_a.set()
    await group.drain()

    assert _led_states(socket) == ["listening", "thinking", "replying", "idle"]
    await _disconnect_edge(socket, task)


async def test_when_one_turn_raises_the_other_finishes_and_the_ring_ends_at_idle():
    source = EdgeAudioSource(_edge_config())
    socket = FakeEdgeSocket()
    task = await _serve_edge(source, socket)
    turns = iter(["a", "b"])
    finished: list[str] = []

    async def run_turn_fn(turn_source: Any) -> None:
        if next(turns) == "a":
            await source.send_event({"type": "transcript.final"})
            raise RuntimeError("brain blew up")
        await asyncio.sleep(0.02)
        finished.append("b")

    runner = _edge_runner(source, run_turn_fn)
    group = runner.turn_group
    assert group is not None
    group.start_wake_turn(0)
    group.start_wake_turn(0)
    await group.drain()

    assert finished == ["b"]
    assert _led_states(socket)[-1] == "idle"
    assert group.live_count == 0
    await _disconnect_edge(socket, task)


def _follow_up_request() -> FollowUpRequest:
    return FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist",
        question="add dentist?",
        pending_action_id=1,
        playback_ends_at=0.0,
    )


def _follow_up_runner(source: Any, run_turn_fn: Any, detector: Any, **kwargs: Any) -> SourceRunner:
    return _runner(
        source,
        run_turn_fn,
        detector,
        parallel=kwargs.pop("parallel", ParallelTurns(max_concurrent=3)),
        follow_up_window_s=lambda: 6.0,
        follow_up_echo_tail_s=0.0,
        clock=lambda: 0.0,
        **kwargs,
    )


async def test_two_turns_that_each_ask_a_question_open_their_follow_up_windows_one_at_a_time():
    source = FakeAudioSource([_frame(i) for i in range(FRAME_COUNT)])
    events: list[tuple[str, str]] = []
    release_window = asyncio.Event()

    async def run_turn_fn(turn_source: Any) -> None:
        key = turn_source.turn_context.turn_key
        if turn_source.follow_up.incoming is None:
            turn_source.follow_up.request(_follow_up_request())
            return
        events.append(("open", key))
        await release_window.wait()
        events.append(("end", key))

    runner = _follow_up_runner(source, run_turn_fn, _HitOnCalls(2, 4))
    run_task = asyncio.create_task(runner.run())
    await _wait_until(lambda: len(events) >= 1)
    await asyncio.sleep(0.05)
    assert [kind for kind, _ in events] == ["open"], "the second window opened while the first was open"

    release_window.set()
    await asyncio.wait_for(run_task, timeout=5.0)

    (first_open, first_key), (first_end, first_end_key), (second_open, second_key), (second_end, _) = events
    assert (first_open, first_end, second_open, second_end) == ("open", "end", "open", "end")
    assert first_key == first_end_key and first_key != second_key


async def test_a_follow_up_window_waits_until_every_other_live_turn_has_its_final_transcript():
    source = FakeAudioSource([_frame(i) for i in range(FRAME_COUNT)])
    window_opened = asyncio.Event()
    b_transcribed = asyncio.Event()
    turns = iter(["a", "b"])
    a_asked, b_started = asyncio.Event(), asyncio.Event()

    async def run_turn_fn(turn_source: Any) -> None:
        if turn_source.follow_up.incoming is not None:
            window_opened.set()
            return
        if next(turns) == "a":
            await b_started.wait()  # ask only once the other turn is live
            turn_source.follow_up.request(_follow_up_request())
            a_asked.set()
            return
        b_started.set()
        await a_asked.wait()
        await b_transcribed.wait()
        turn_source.barge_in.mark_transcript_done()
        await asyncio.sleep(0.02)

    runner = _follow_up_runner(source, run_turn_fn, _HitOnCalls(2, 4))
    run_task = asyncio.create_task(runner.run())
    await _wait_until(lambda: a_asked.is_set())
    await asyncio.sleep(0.05)
    assert not window_opened.is_set(), "the window opened before the other turn had its final transcript"

    b_transcribed.set()
    await asyncio.wait_for(run_task, timeout=5.0)
    assert window_opened.is_set()


async def test_a_follow_up_window_also_opens_once_the_other_turn_has_ended():
    source = FakeAudioSource([_frame(i) for i in range(FRAME_COUNT)])
    window_opened = asyncio.Event()
    turns = iter(["a", "b"])
    b_may_end, b_started = asyncio.Event(), asyncio.Event()

    async def run_turn_fn(turn_source: Any) -> None:
        if turn_source.follow_up.incoming is not None:
            window_opened.set()
            return
        if next(turns) == "a":
            await b_started.wait()  # ask only once the other turn is live
            turn_source.follow_up.request(_follow_up_request())
            return
        b_started.set()
        await b_may_end.wait()  # ends without ever marking its transcript done

    runner = _follow_up_runner(source, run_turn_fn, _HitOnCalls(2, 4))
    run_task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)
    assert not window_opened.is_set()

    b_may_end.set()
    await asyncio.wait_for(run_task, timeout=5.0)
    assert window_opened.is_set()


async def test_the_follow_up_turn_keeps_the_group_identity_and_answers_only_the_asking_speaker():
    source = FakeAudioSource([_frame(i) for i in range(FRAME_COUNT)])
    contexts: list[TurnContext] = []
    events: list[str] = []
    registry = _FakeReplyRegistry(events)
    claims = object()

    async def run_turn_fn(turn_source: Any) -> None:
        context = turn_source.turn_context
        contexts.append(context)
        if turn_source.follow_up.incoming is None:
            context.note_speaker("speaker-7", "Josh")
            turn_source.follow_up.request(_follow_up_request())

    runner = _follow_up_runner(
        source,
        run_turn_fn,
        _HitOnCalls(3),
        parallel=ParallelTurns(max_concurrent=3, reply_registry=registry, claims_factory=lambda: claims),
    )
    await asyncio.wait_for(runner.run(), timeout=5.0)

    asking, answering = contexts
    assert (answering.turn_key, answering.group_id) == (asking.turn_key, asking.group_id)
    assert answering.claims is claims and answering.reply_group is registry.handles[0]
    assert (asking.follow_up, answering.follow_up) == (False, True)
    assert asking.answer_only_from is None
    assert answering.answer_only_from == "speaker-7"
    assert len(registry.registered) == 1 and registry.handles[0].finished == 1


class _QueueSource(FakeAudioSource):
    """Frames come from a queue the test feeds. Counts `frames()` calls."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()
        self.frames_calls = 0

    async def frames(self):
        self.frames_calls += 1
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                return
            yield chunk


async def test_an_enabled_barge_in_listener_reads_its_own_subscription_and_the_source_is_read_once():
    source = _QueueSource()
    listened: list[bytes] = []
    transcript_done, release = asyncio.Event(), asyncio.Event()

    async def run_turn_fn(turn_source: Any) -> None:
        turn_source.barge_in.mark_transcript_done()
        transcript_done.set()
        await release.wait()

    runner = _runner(
        source,
        run_turn_fn,
        _HitOnCalls(2),
        parallel=ParallelTurns(max_concurrent=3),
        barge_in_config=BargeInConfig(enabled=True),
    )

    def record_energy(chunk: bytes) -> float:
        listened.append(chunk)
        return 0.0

    runner._barge_in_energy = record_energy  # noqa: SLF001
    run_task = asyncio.create_task(runner.run())
    for index in range(4):
        source.queue.put_nowait(_frame(index))
    await transcript_done.wait()
    await _wait_until(lambda: runner.turn_group is not None and runner.turn_group.fanout.latest_index == 3)
    await asyncio.sleep(0.05)  # the listener subscribes once it has seen transcript_done
    for index in range(4, 7):
        source.queue.put_nowait(_frame(index))
    await _wait_until(lambda: len(listened) >= 3)

    assert listened[-3:] == [_frame(4), _frame(5), _frame(6)]
    assert source.frames_calls == 1
    release.set()
    source.queue.put_nowait(None)
    await asyncio.wait_for(run_task, timeout=5.0)
    assert source.frames_calls == 1


def test_the_follow_up_window_opens_after_the_readback_plus_the_echo_tail():
    from atlas.sources.runner import follow_up_window_opens_at

    requested = SimpleNamespace(playback_ends_at=10.0)

    assert follow_up_window_opens_at(requested, echo_tail_s=0.8, calibration=None, now=99.0) == 10.8
    # No estimated playback end: the window is measured from `now`.
    assert follow_up_window_opens_at(SimpleNamespace(playback_ends_at=None), echo_tail_s=0.8, calibration=None, now=5.0) == 5.8
    # A calibrated echo delay plus the 0.2 s margin beats a smaller configured tail.
    calibration = SimpleNamespace(delay_s=1.5)
    assert follow_up_window_opens_at(requested, echo_tail_s=0.8, calibration=calibration, now=0.0) == 10.0 + 1.7
