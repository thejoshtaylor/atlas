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


def _edge_runner(source: EdgeAudioSource, *, barge_in_config: BargeInConfig) -> SourceRunner:
    async def run_turn_fn(turn_source) -> None:
        return None

    return SourceRunner(
        "edge",
        source,
        _NeverHitDetector(),
        lambda chunk: chunk,
        run_turn_fn,
        barge_in_config=barge_in_config,
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


def test_edge_runner_barge_in_falls_back_to_the_spikes_verdict_when_unconfigured():
    from atlas.config import Config

    from tests.test_config import _minimal_raw_config

    raw = _minimal_raw_config()
    config = Config.from_config(raw)
    assert "edge" not in config.barge_in.sources

    from atlas.app import _resolve_edge_barge_in_config

    resolved = _resolve_edge_barge_in_config(config).resolve("edge")
    assert resolved.enabled is EDGE_BARGE_IN_PROVEN


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
