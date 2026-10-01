"""Real assertions for D-16's edge-source barge-in path (10-10-PLAN.md
Task 2): a real `SourceRunner` over a real `EdgeAudioSource`, driven
through a scripted `FakeEdgeSocket` -- never a mock of either class.

`tests/test_barge_in.py` already covers `BargeInMonitor.process_speech_start`
as pure boundary logic (no asyncio). This file covers the asyncio wiring
around it: `SourceRunner._watch_barge_in`'s `speech_signals` branch, and
`app.py`'s `_resolve_edge_barge_in_config` -- the piece that decides
whether the edge runner is even built with barge-in on in the first place.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from atlas.config import EDGE_BARGE_IN_PROVEN, BargeInConfig, EdgeSourceConfig
from atlas.sources.runner import SourceRunner
from atlas.transports.edge import EdgeAudioSource
from atlas.turn.controller import _speak

from tests.conftest import FakeWakeHit
from tests.edge_fakes import FakeEdgeSocket, fake_edge_device


def _measured_config(**overrides) -> EdgeSourceConfig:
    base = dict(sample_rate=16000, channels=2, asr_channel=1, pre_roll_ms=200, tail_ms=300)
    base.update(overrides)
    return EdgeSourceConfig(**base)


class _NeverHitDetector:
    """`SourceRunner.__init__` requires a wake detector; this file never
    drives a real wake hit through `run()` -- every test here calls
    `_watch_barge_in` directly, the same private-method-under-test
    convention `tests/test_barge_in.py`'s own `_drive_one_hit` already
    uses."""

    def process(self, chunk: bytes):
        return None


class _ScriptedWakeDetector:
    """Hits on the scripted call numbers (1-based) and records every call and
    every reset. `reset()` counts, like a real detector's."""

    def __init__(self, hit_on_calls=()) -> None:
        self._hit_on_calls = set(hit_on_calls)
        self.calls = 0
        self.resets = 0

    def process(self, chunk: bytes):
        self.calls += 1
        if self.calls in self._hit_on_calls:
            return FakeWakeHit(score=1.0)
        return None

    def reset(self) -> None:
        self.resets += 1


def _edge_runner(
    source: EdgeAudioSource, *, barge_in_config: BargeInConfig, detector=None, wake_event_repo=None
) -> SourceRunner:
    async def run_turn_fn(turn_source) -> None:
        return None

    return SourceRunner(
        "edge",
        source,
        detector if detector is not None else _NeverHitDetector(),
        lambda chunk: chunk,
        run_turn_fn,
        barge_in_config=barge_in_config,
        wake_event_repo=wake_event_repo,
    )


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.005) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition never became true within the timeout")


class _PacedFakeTts:
    """Like `tests/conftest.py`'s `FakeTts`, but yields one real,
    measurable delay before every chunk -- `FakeTts` itself, and a bare
    `asyncio.sleep(0)`, both complete near-instantly with no window this
    test's own polling could ever observe mid-utterance, so every chunk
    would be sent before a single poll interval elapsed and this test
    would prove nothing about interrupting mid-stream."""

    def __init__(self, chunks, *, delay_s: float = 0.02) -> None:
        self._chunks = list(chunks)
        self._delay_s = delay_s
        self.received_sinks: list = []

    async def synthesize(self, text_deltas, sink=None):
        async for _ in text_deltas:
            pass
        self.received_sinks.append(sink)
        for chunk in self._chunks:
            await asyncio.sleep(self._delay_s)
            yield chunk


class _BurstFakeTts:
    """Yields every chunk with no sleep, which is how a real batch TTS
    reaches the Pi: the whole reply goes out in one burst."""

    def __init__(self, chunks) -> None:
        self._chunks = list(chunks)

    async def synthesize(self, text_deltas, sink=None):
        async for _ in text_deltas:
            pass
        for chunk in self._chunks:
            yield chunk


@pytest.fixture
def edge_source() -> EdgeAudioSource:
    return EdgeAudioSource(_measured_config())


async def test_vad_start_during_edge_reply_interrupts(edge_source):
    """A real `SourceRunner` over a real `EdgeAudioSource`, barge-in
    enabled: once the transcript is done and playback has started, a
    `vad.start` pushed through the fake socket sets `interrupt_requested`,
    and `_speak` stops writing further reply chunks to the socket."""
    socket = FakeEdgeSocket()
    device = fake_edge_device(device_id=1)
    serve_task = asyncio.create_task(edge_source.serve(socket, device))

    # guard_window=0: this test proves the vad.start path, not real-time
    # guard-window arithmetic (already covered, pure, by test_barge_in.py).
    runner = _edge_runner(
        edge_source, barge_in_config=BargeInConfig(enabled=True, post_playback_guard_ms=0)
    )
    monitor = runner._new_barge_in_monitor()
    monitor.mark_transcript_done()

    watch_task = asyncio.create_task(runner._watch_barge_in(monitor))
    try:
        # Let the listener actually subscribe to speech_signals and start
        # draining frames() before this test does anything else -- the
        # module docstring's "become the sole reader" contract.
        await asyncio.sleep(0)

        chunks = [f"chunk-{i}".encode() for i in range(10)]
        tts = _PacedFakeTts(chunks)
        from atlas.timing import TurnTimings

        timings = TurnTimings()
        speak_task = asyncio.create_task(
            _speak(edge_source, tts, timings, "reply text", kind="answer", barge_in=monitor)
        )

        # Wait for playback to actually start (the first chunk's own
        # mark_playback_started call) before the vad.start arrives --
        # matching this test's own docstring: "once ... playback has
        # started".
        await _wait_until(lambda: monitor.playback_started_at is not None)

        socket.push_text(json.dumps({"type": "vad.start", "seq": 1}))

        await _wait_until(lambda: monitor.interrupt_requested is True)

        await asyncio.wait_for(speak_task, timeout=2.0)

        assert len(socket.sent_bytes) < len(chunks), (
            "the interrupt must stop _speak before every chunk is written"
        )
        assert timings.turn_outcome == "barged_in"
    finally:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


async def test_edge_barge_in_off_ignores_vad_start(edge_source):
    """A disabled policy never even subscribes to speech_signals -- the
    same `vad.start` this file's other test proves interrupts on changes
    nothing here, and the listener returns at once rather than opening a
    second reader of frames() at all (D-12's per-source override)."""
    socket = FakeEdgeSocket()
    device = fake_edge_device(device_id=1)
    serve_task = asyncio.create_task(edge_source.serve(socket, device))

    runner = _edge_runner(edge_source, barge_in_config=BargeInConfig(enabled=False))
    monitor = runner._new_barge_in_monitor()
    monitor.mark_transcript_done()
    monitor.mark_playback_started(runner._clock())

    watch_task = asyncio.create_task(runner._watch_barge_in(monitor))
    try:
        # A disabled monitor's own `_watch_barge_in` returns immediately
        # (before ever awaiting transcript_done) -- proven by waiting for
        # the task to finish on its own, not by cancelling it.
        await asyncio.wait_for(watch_task, timeout=2.0)
        assert len(edge_source.speech_signals._subscribers) == 0, (
            "a disabled policy must never subscribe to speech_signals at all"
        )

        socket.push_text(json.dumps({"type": "vad.start", "seq": 1}))
        await asyncio.sleep(0.05)
        assert monitor.interrupt_requested is False
    finally:
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


async def test_process_energy_is_never_called_for_a_source_with_speech_signals(edge_source, monkeypatch):
    """The energy path is structurally unreachable for a source that
    carries `speech_signals` -- proven by making `process_energy` raise
    if it is ever called, then pushing loud raw audio frames (which would
    interrupt on the energy path, if it ran) with no `vad.start` at all."""
    socket = FakeEdgeSocket()
    device = fake_edge_device(device_id=1)
    serve_task = asyncio.create_task(edge_source.serve(socket, device))

    runner = _edge_runner(
        edge_source, barge_in_config=BargeInConfig(enabled=True, post_playback_guard_ms=0)
    )
    monitor = runner._new_barge_in_monitor()
    monitor.mark_transcript_done()
    monitor.mark_playback_started(runner._clock())

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("process_energy must never be called for a source with speech_signals")

    monkeypatch.setattr(monitor, "process_energy", _fail_if_called)

    watch_task = asyncio.create_task(runner._watch_barge_in(monitor))
    try:
        await asyncio.sleep(0)
        # Loud, full-scale PCM frames -- exactly the shape that would cross
        # the energy floor and interrupt, on the energy path this source
        # never takes.
        loud_frame = (b"\xff\x7f\x00\x80") * 40  # 2 channels, 16-bit, full scale
        for _ in range(10):
            socket.push_bytes(loud_frame)
        await asyncio.sleep(0.05)
        assert monitor.interrupt_requested is False
    finally:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


# --- Phase 13: the wake word during a held edge reply ---


def _stop_frames(socket: FakeEdgeSocket) -> list[dict]:
    parsed = [json.loads(text) for text in socket.sent_text]
    return [message for message in parsed if message.get("type") == "stop"]


async def test_wake_hit_during_edge_reply_hold_sends_stop_and_ends_barged_in(edge_source):
    """The tracer: "Atlas" said while an edge reply plays stops it. A real
    runner over a real source, a burst TTS (1.0 s of audio written at once),
    one whole stereo frame through the socket, and the detector hits."""
    from atlas.timing import TurnTimings

    socket = FakeEdgeSocket()
    device = fake_edge_device(device_id=1)
    serve_task = asyncio.create_task(edge_source.serve(socket, device))
    detector = _ScriptedWakeDetector(hit_on_calls={1})
    runner = _edge_runner(edge_source, barge_in_config=BargeInConfig(enabled=False, wake_word=True), detector=detector)
    monitor = runner._new_barge_in_monitor()
    assert monitor.wake_interrupts
    assert monitor.holds_for_playback
    monitor.mark_transcript_done()

    watch_task = asyncio.create_task(runner._watch_barge_in(monitor))
    try:
        await asyncio.sleep(0)
        loop = asyncio.get_running_loop()
        timings = TurnTimings()
        timings.turn_outcome = "completed"
        started = loop.time()
        speak_task = asyncio.create_task(
            _speak(
                edge_source,
                _BurstFakeTts([b"\x00\x00" * 1600] * 10),
                timings,
                "reply text",
                kind="answer",
                barge_in=monitor,
                sink=edge_source.sink_format(),
            )
        )
        await _wait_until(lambda: len(socket.sent_bytes) == 10)
        assert not speak_task.done(), "the answer must hold until the playback end"

        socket.push_bytes(b"\x00\x00\x00\x00" * 256)

        await asyncio.wait_for(speak_task, timeout=0.9)
        assert loop.time() - started < 0.9
        assert timings.turn_outcome == "barged_in"
        await _wait_until(lambda: len(_stop_frames(socket)) >= 1)
        await asyncio.sleep(0.02)
        assert _stop_frames(socket) == [{"type": "stop", "fade_ms": 120}]
        assert socket.binary_after_text_type("stop") == 0
        assert detector.resets == 1
    finally:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


async def _speak_with_one_hit(edge_source, repo, *, reply_text: str, atlas_margin_ms: int):
    """A burst reply of `reply_text` (1.0 s of audio) and a wake hit right
    after the burst. Returns the socket and the finished timings."""
    from atlas.timing import TurnTimings

    socket = FakeEdgeSocket()
    device = fake_edge_device(device_id=1)
    serve_task = asyncio.create_task(edge_source.serve(socket, device))
    detector = _ScriptedWakeDetector(hit_on_calls={1})
    runner = _edge_runner(
        edge_source,
        barge_in_config=BargeInConfig(enabled=False, wake_word=True, atlas_margin_ms=atlas_margin_ms),
        detector=detector,
        wake_event_repo=repo,
    )
    monitor = runner._new_barge_in_monitor()
    monitor.mark_transcript_done()
    watch_task = asyncio.create_task(runner._watch_barge_in(monitor))
    try:
        await asyncio.sleep(0)
        timings = TurnTimings()
        timings.turn_outcome = "completed"
        speak_task = asyncio.create_task(
            _speak(
                edge_source,
                _BurstFakeTts([b"\x00\x00" * 1600] * 10),
                timings,
                reply_text,
                kind="answer",
                barge_in=monitor,
                sink=edge_source.sink_format(),
            )
        )
        await _wait_until(lambda: len(socket.sent_bytes) == 10)
        socket.push_bytes(b"\x00\x00\x00\x00" * 256)
        await asyncio.wait_for(speak_task, timeout=3.0)
        await asyncio.sleep(0.05)
        return socket, timings
    finally:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


async def test_a_hit_lined_up_with_atlas_in_the_reply_does_not_interrupt(edge_source, fake_wake_event_repository):
    """D-02: the reply says "Atlas" at its start, and the hit lands right
    after the burst, inside that word's window. The reply is not cut."""
    repo = fake_wake_event_repository()
    socket, timings = await _speak_with_one_hit(
        edge_source, repo, reply_text="Atlas is here to help you", atlas_margin_ms=1000
    )
    assert timings.turn_outcome == "completed"
    assert _stop_frames(socket) == []
    assert [(e.allowed, e.block_reason) for e in repo.events] == [(False, "reply_wake_word")]


async def test_a_hit_outside_every_atlas_window_still_interrupts(edge_source, fake_wake_event_repository):
    """With no margin and "Atlas" as the last word, an early hit is far from
    the word, so the reply is cut as before."""
    repo = fake_wake_event_repository()
    socket, timings = await _speak_with_one_hit(
        edge_source, repo, reply_text="the answer is Atlas", atlas_margin_ms=0
    )
    assert timings.turn_outcome == "barged_in"
    assert _stop_frames(socket) == [{"type": "stop", "fade_ms": 120}]
    assert [(e.allowed, e.block_reason) for e in repo.events] == [(True, None)]


async def test_edge_reply_with_no_wake_hit_holds_until_playback_end(edge_source):
    """An edge reply nobody interrupts keeps its turn open until the
    estimated playback end, then ends completed with no stop frame."""
    from atlas.timing import TurnTimings

    socket = FakeEdgeSocket()
    device = fake_edge_device(device_id=1)
    serve_task = asyncio.create_task(edge_source.serve(socket, device))
    runner = _edge_runner(edge_source, barge_in_config=BargeInConfig(enabled=False, wake_word=True))
    monitor = runner._new_barge_in_monitor()
    monitor.mark_transcript_done()

    watch_task = asyncio.create_task(runner._watch_barge_in(monitor))
    try:
        await asyncio.sleep(0)
        loop = asyncio.get_running_loop()
        timings = TurnTimings()
        timings.turn_outcome = "completed"
        started = loop.time()
        await _speak(
            edge_source,
            _BurstFakeTts([b"\x00\x00" * 1600] * 2),
            timings,
            "reply text",
            kind="answer",
            barge_in=monitor,
            sink=edge_source.sink_format(),
        )
        assert loop.time() - started >= 0.15
        assert timings.turn_outcome == "completed"
        assert _stop_frames(socket) == []
    finally:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


# --- app.py's own resolution of the edge runner's barge-in policy (D-16) ---


def test_edge_runner_barge_in_uses_the_configured_override_when_present():
    from atlas.config import Config

    from tests.test_config import _minimal_raw_config

    raw = _minimal_raw_config()
    raw["barge_in"] = {"sources": {"edge": {"enabled": True}}}
    config = Config.from_config(raw)

    from atlas.app import _resolve_edge_barge_in_config

    resolved = _resolve_edge_barge_in_config(config).resolve("edge")
    assert resolved.enabled is True
    assert resolved.wake_word is True


def test_edge_runner_barge_in_falls_back_to_the_spikes_verdict_when_unconfigured():
    from atlas.config import Config

    from tests.test_config import _minimal_raw_config

    raw = _minimal_raw_config()
    config = Config.from_config(raw)
    assert "edge" not in config.barge_in.sources

    from atlas.app import _resolve_edge_barge_in_config

    resolved = _resolve_edge_barge_in_config(config).resolve("edge")
    assert resolved.enabled is EDGE_BARGE_IN_PROVEN
    assert resolved.wake_word is True


def test_edge_runner_barge_in_never_inherits_the_bare_global_enabled():
    """The global `barge_in.enabled` defaults `True` -- proving the edge
    runner's own resolved policy is `False` (matching
    `EDGE_BARGE_IN_PROVEN`) with no `edge` override configured proves the
    global default never leaked through by accident."""
    from atlas.config import Config

    from tests.test_config import _minimal_raw_config

    raw = _minimal_raw_config()
    config = Config.from_config(raw)
    assert config.barge_in.enabled is True

    from atlas.app import _resolve_edge_barge_in_config

    resolved = _resolve_edge_barge_in_config(config).resolve("edge")
    assert resolved.enabled is False


def test_edge_runner_gets_wake_word_when_the_operator_only_disabled_the_vad_path():
    """A cluster config that already says `barge_in.sources.edge.enabled:
    false` gets the wake path with no edit (Phase 13)."""
    from atlas.app import _resolve_edge_barge_in_config
    from atlas.config import Config

    from tests.test_config import _minimal_raw_config

    raw = _minimal_raw_config()
    raw["barge_in"] = {"sources": {"edge": {"enabled": False}}}
    resolved = _resolve_edge_barge_in_config(Config.from_config(raw)).resolve("edge")
    assert resolved.enabled is False
    assert resolved.wake_word is True


def test_the_operators_own_wake_word_key_wins_for_the_edge():
    from atlas.app import _resolve_edge_barge_in_config
    from atlas.config import Config

    from tests.test_config import _minimal_raw_config

    raw = _minimal_raw_config()
    raw["barge_in"] = {"sources": {"edge": {"wake_word": False}}}
    resolved = _resolve_edge_barge_in_config(Config.from_config(raw)).resolve("edge")
    assert resolved.wake_word is False


# --- Phase 13 (plan 13-05): the interrupting speech becomes the next turn ---


def _stereo(value: int) -> bytes:
    """One whole stereo frame run that no other value's run equals."""
    return bytes([value, 0, value, 0]) * 256


def _handover_runner(edge_source, run_turn_fn, detector, *, barge_in_config, follow_up_window_s=None):
    from atlas.audio.ring import PrerollBuffer

    return SourceRunner(
        "edge",
        edge_source,
        detector,
        lambda chunk: chunk,
        run_turn_fn,
        barge_in_config=barge_in_config,
        preroll=PrerollBuffer(edge_source.source_format(), 1500),
        follow_up_window_s=follow_up_window_s,
    )


def _interrupted_monitor(runner, kind):
    monitor = runner._new_barge_in_monitor()
    monitor._on_interrupt = None
    monitor.interrupt_requested = True
    monitor.interrupt_kind = kind
    return monitor


async def test_a_wake_interrupt_hands_its_frames_to_the_next_wake_turn_once_each(edge_source, monkeypatch):
    """The end-to-end hand-over: turn 1 speaks a held reply, the operator's
    speech (B, then C where the detector hits) arrives during the hold, and
    turn 2 reads B and C first, each once, then the live frames."""
    from atlas.timing import TurnTimings
    from atlas.turn.follow_up import FollowUpChannel

    socket = FakeEdgeSocket()
    device = fake_edge_device(device_id=1)
    serve_task = asyncio.create_task(edge_source.serve(socket, device))
    wake_marks: list[None] = []
    monkeypatch.setattr(edge_source, "mark_wake_hit", lambda: wake_marks.append(None))

    a_frame, b_frame, c_frame, d_frame, e_frame = (_stereo(n) for n in (1, 2, 3, 4, 5))
    turn_two: dict = {"started": asyncio.Event(), "chunks": [], "sources": []}
    turn_one_outcome: list[str] = []
    calls = 0

    async def run_turn_fn(turn_source) -> None:
        nonlocal calls
        calls += 1
        monitor = turn_source.barge_in
        if calls == 1:
            monitor.mark_transcript_done()
            timings = TurnTimings()
            timings.turn_outcome = "completed"
            await _speak(
                turn_source,
                _BurstFakeTts([b"\x00\x00" * 1600] * 10),
                timings,
                "reply text",
                kind="answer",
                barge_in=monitor,
                sink=edge_source.sink_format(),
            )
            turn_one_outcome.append(timings.turn_outcome)
            return
        turn_two["sources"].append(turn_source)
        turn_two["started"].set()
        async for chunk in turn_source.frames():
            turn_two["chunks"].append(chunk)
            if len(turn_two["chunks"]) == 4:
                return

    runner = _handover_runner(
        edge_source,
        run_turn_fn,
        _ScriptedWakeDetector(hit_on_calls={1, 3}),
        barge_in_config=BargeInConfig(enabled=False, wake_word=True),
        follow_up_window_s=lambda: 8.0,
    )
    run_task = asyncio.create_task(runner.run())
    try:
        await asyncio.sleep(0)
        socket.push_bytes(a_frame)
        await _wait_until(lambda: len(socket.sent_bytes) == 10)
        socket.push_bytes(b_frame)
        socket.push_bytes(c_frame)
        await asyncio.wait_for(turn_two["started"].wait(), timeout=2.0)
        socket.push_bytes(d_frame)
        socket.push_bytes(e_frame)
        await _wait_until(lambda: len(turn_two["chunks"]) == 4)
        await asyncio.sleep(0.02)

        assert turn_one_outcome == ["barged_in"]
        assert turn_two["chunks"] == [b_frame, c_frame, d_frame, e_frame]
        [second_source] = turn_two["sources"]
        assert second_source.barge_in.after_interrupt is True
        assert isinstance(second_source.follow_up, FollowUpChannel)
        assert second_source.follow_up.incoming is None
        assert len(wake_marks) == 2, "the playback hit marks its own frame for the speaker tracker"
        assert _stop_frames(socket) == [{"type": "stop", "fade_ms": 120}]
    finally:
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


async def test_the_listener_gives_every_chunk_it_holds_to_the_pre_roll_or_the_hand_over(edge_source):
    """RESEARCH Pitfall 2: chunks read before the hit are in the pre-roll the
    hit takes. Chunks read after it go to the hand-over, in order."""
    socket = FakeEdgeSocket()
    device = fake_edge_device(device_id=1)
    serve_task = asyncio.create_task(edge_source.serve(socket, device))

    async def never(turn_source) -> None:
        return None

    runner = _handover_runner(
        edge_source,
        never,
        _ScriptedWakeDetector(hit_on_calls={2}),
        barge_in_config=BargeInConfig(enabled=False, wake_word=True),
    )
    monitor = runner._new_barge_in_monitor()
    monitor.mark_transcript_done()
    watch_task = asyncio.create_task(runner._watch_barge_in(monitor))
    try:
        await asyncio.sleep(0)
        frames = [_stereo(n) for n in (1, 2, 3, 4)]
        for frame in frames:
            socket.push_bytes(frame)
        await _wait_until(lambda: len(monitor.handover) == 4)
        assert monitor.interrupt_kind == "wake"
        assert monitor.handover == frames
    finally:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


async def test_interrupt_turns_stop_after_the_cap_and_the_runner_goes_back_to_wake_listening(edge_source, monkeypatch):
    from atlas.sources.runner import MAX_CHAINED_INTERRUPT_TURNS
    from atlas.turn.follow_up import FollowUpChannel

    runner = _handover_runner(
        edge_source,
        lambda turn_source: None,
        _NeverHitDetector(),
        barge_in_config=BargeInConfig(enabled=False, wake_word=True),
        follow_up_window_s=lambda: 8.0,
    )
    ran = []

    async def interrupt_every_turn(turn_source, monitor) -> None:
        ran.append((turn_source, monitor))
        monitor._on_interrupt = None
        monitor.interrupt_requested = True
        monitor.interrupt_kind = "wake"

    monkeypatch.setattr(runner, "_run_one_turn", interrupt_every_turn)
    first = _interrupted_monitor(runner, "wake")
    first.handover = [_stereo(9)]

    last, channel = await runner._continue_after_interrupts(first, FollowUpChannel())

    assert len(ran) == MAX_CHAINED_INTERRUPT_TURNS == 3
    assert channel is None
    assert last is ran[-1][1]
    assert all(monitor.after_interrupt for _, monitor in ran)
    assert [chunk async for chunk in _first_chunks(ran[0][0], 1)] == [_stereo(9)]


async def _first_chunks(turn_source, count):
    seen = 0
    async for chunk in turn_source.frames():
        yield chunk
        seen += 1
        if seen == count:
            return


async def test_a_turn_that_is_not_interrupted_starts_no_extra_turn(edge_source, monkeypatch):
    from atlas.turn.follow_up import FollowUpChannel

    runner = _handover_runner(
        edge_source,
        lambda turn_source: None,
        _NeverHitDetector(),
        barge_in_config=BargeInConfig(enabled=False, wake_word=True),
    )
    ran = []

    async def record(turn_source, monitor) -> None:
        ran.append(monitor)

    monkeypatch.setattr(runner, "_run_one_turn", record)
    channel = FollowUpChannel()
    quiet = runner._new_barge_in_monitor()
    assert await runner._continue_after_interrupts(quiet, channel) == (quiet, channel)
    energy = _interrupted_monitor(runner, "energy")
    assert await runner._continue_after_interrupts(energy, channel) == (energy, channel)
    assert ran == []


def test_a_second_resume_note_replaces_the_first(edge_source):
    from atlas.turn.follow_up import AnswerScope

    runner = _handover_runner(
        edge_source, lambda turn_source: None, _NeverHitDetector(), barge_in_config=BargeInConfig(enabled=False)
    )
    monitor = runner._new_barge_in_monitor()
    assert monitor.resume_transcript is None and monitor.resume_scope is None
    monitor.note_resume_context("first", AnswerScope(tool_names=frozenset({"a"})))
    monitor.note_resume_context("second", AnswerScope(tool_names=frozenset()))
    assert monitor.resume_transcript == "second"
    assert monitor.resume_scope == AnswerScope(tool_names=frozenset())


async def _vad_interrupted_edge_turn(edge_source, *, noted_scope):
    """Turn 1 speaks a held reply and (optionally) notes its scope. A
    `vad.start` interrupts it, and frames B and C follow. Returns what turn 2
    saw: its incoming request and its first two chunks."""
    from atlas.timing import TurnTimings

    socket = FakeEdgeSocket()
    device = fake_edge_device(device_id=1)
    serve_task = asyncio.create_task(edge_source.serve(socket, device))
    b_frame, c_frame = _stereo(2), _stereo(3)
    seen: dict = {"incoming": None, "chunks": [], "after_interrupt": None, "done": asyncio.Event()}
    calls = 0

    async def run_turn_fn(turn_source) -> None:
        nonlocal calls
        calls += 1
        monitor = turn_source.barge_in
        if calls == 1:
            if noted_scope is not None:
                monitor.note_resume_context("what is the weather", noted_scope)
            monitor.mark_transcript_done()
            timings = TurnTimings()
            timings.turn_outcome = "completed"
            await _speak(
                turn_source,
                _BurstFakeTts([b"\x00\x00" * 1600] * 10),
                timings,
                "It is sunny today",
                kind="answer",
                barge_in=monitor,
                sink=edge_source.sink_format(),
            )
            return
        seen["incoming"] = turn_source.follow_up.incoming
        seen["after_interrupt"] = monitor.after_interrupt
        async for chunk in turn_source.frames():
            seen["chunks"].append(chunk)
            if len(seen["chunks"]) == 2:
                break
        seen["done"].set()

    runner = _handover_runner(
        edge_source,
        run_turn_fn,
        _ScriptedWakeDetector(hit_on_calls={1}),
        barge_in_config=BargeInConfig(enabled=True, wake_word=False, post_playback_guard_ms=0),
        follow_up_window_s=lambda: 8.0,
    )
    run_task = asyncio.create_task(runner.run())
    try:
        await asyncio.sleep(0)
        socket.push_bytes(_stereo(1))
        await _wait_until(lambda: len(socket.sent_bytes) == 10)
        socket.push_text(json.dumps({"type": "vad.start", "seq": 1}))
        await _wait_until(lambda: len(_stop_frames(socket)) == 1)
        socket.push_bytes(b_frame)
        socket.push_bytes(c_frame)
        await asyncio.wait_for(seen["done"].wait(), timeout=2.0)
        assert seen["chunks"] == [b_frame, c_frame]
        assert seen["after_interrupt"] is True
        return seen["incoming"]
    finally:
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


async def test_a_vad_interrupt_becomes_a_scoped_no_wake_word_answer_turn(edge_source):
    from atlas.turn.follow_up import AnswerScope

    scope = AnswerScope(tool_names=frozenset({"weather_now"}))
    incoming = await _vad_interrupted_edge_turn(edge_source, noted_scope=scope)
    assert incoming.kind == "answer"
    assert incoming.chain_depth == 1
    assert incoming.original_transcript == "what is the weather"
    assert incoming.question == "It is sunny today"
    assert incoming.answer_scope == scope


async def test_a_vad_interrupt_with_no_note_reaches_no_tool(edge_source):
    from atlas.turn.follow_up import AnswerScope

    incoming = await _vad_interrupted_edge_turn(edge_source, noted_scope=None)
    assert incoming.answer_scope == AnswerScope(tool_names=frozenset())
    assert incoming.original_transcript == ""
