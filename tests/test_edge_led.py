"""LED state messages from `EdgeAudioSource` (260928-m11): the turn order
the `SourceRunner` and `send_event`/`send_audio` hooks produce on the
socket, and the rule that no LED problem ever breaks a turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from atlas.config import EdgeSourceConfig, GateConfig, WakeConfig
from atlas.sources.runner import SourceRunner
from atlas.transports.edge import EdgeAudioSource
from atlas.turn.follow_up import FollowUpRequest

from tests.conftest import FakeAudioSource, FakeWakeHit
from tests.edge_fakes import FakeEdgeSocket, fake_edge_device


class _AlwaysHitWakeDetector:
    def process(self, chunk: bytes) -> Any:
        return FakeWakeHit(score=1.0)

    def close(self) -> None:
        pass


class _LedFailingSocket(FakeEdgeSocket):
    """Raises from `send_text` for `led` messages only."""

    async def send_text(self, data: str) -> None:
        if json.loads(data).get("type") == "led":
            raise RuntimeError("socket write failed")
        await super().send_text(data)


def _config() -> EdgeSourceConfig:
    return EdgeSourceConfig(sample_rate=16000, channels=2, asr_channel=1, pre_roll_ms=200, tail_ms=300)


def _led_states(socket: FakeEdgeSocket) -> list[str]:
    states = []
    for text in socket.sent_text:
        parsed = json.loads(text)
        if parsed.get("type") == "led":
            states.append(parsed["state"])
    return states


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true within the timeout")


async def _serve(source: EdgeAudioSource, socket: FakeEdgeSocket) -> "asyncio.Task[None]":
    task = asyncio.create_task(source.serve(socket, fake_edge_device(device_id=1)))
    await _wait_until(lambda: socket.sent_text != [])
    return task


async def _disconnect(socket: FakeEdgeSocket, task: "asyncio.Task[None]") -> None:
    socket.push_disconnect()
    await asyncio.wait_for(task, timeout=2.0)


def _runner(source: Any, run_turn_fn, **kwargs: Any) -> SourceRunner:
    return SourceRunner(
        "edge",
        source,
        _AlwaysHitWakeDetector(),
        lambda chunk: chunk,
        run_turn_fn,
        wake_config=WakeConfig(engine="vosk", refractory_s=0.0),
        gate_config=GateConfig(),
        **kwargs,
    )


async def test_a_turn_sends_listening_thinking_replying_idle_in_order():
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    replying_before_first_reply_chunk: list[bool] = []

    async def run_turn_fn(turn_source: Any) -> None:
        await source.send_audio(b"wake-cue")  # the cue plays while listening
        await source.send_event({"type": "transcript.final", "text": "hello"})
        socket_bytes_before = len(socket.sent_bytes)
        await source.send_audio(b"reply-1")
        replying_before_first_reply_chunk.append(
            _led_states(socket) == ["listening", "thinking", "replying"]
            and len(socket.sent_bytes) == socket_bytes_before + 1
        )
        await source.send_audio(b"reply-2")

    runner = _runner(source, run_turn_fn)
    await runner._process_chunk(bytes(1024))  # noqa: SLF001

    assert _led_states(socket) == ["listening", "thinking", "replying", "idle"]
    assert replying_before_first_reply_chunk == [True]
    assert socket.sent_bytes == [b"wake-cue", b"reply-1", b"reply-2"]
    await _disconnect(socket, task)


async def test_the_replying_message_goes_out_before_the_first_reply_bytes():
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    order: list[str] = []
    original_text, original_bytes = socket.send_text, socket.send_bytes

    async def record_text(data: str) -> None:
        parsed = json.loads(data)
        if parsed.get("type") == "led":
            order.append(f"led:{parsed['state']}")
        await original_text(data)

    async def record_bytes(data: bytes) -> None:
        order.append("bytes")
        await original_bytes(data)

    socket.send_text = record_text  # type: ignore[method-assign]
    socket.send_bytes = record_bytes  # type: ignore[method-assign]

    async def run_turn_fn(turn_source: Any) -> None:
        await source.send_audio(b"cue")
        await source.send_event({"type": "transcript.final"})
        await source.send_audio(b"reply")

    await _runner(source, run_turn_fn)._process_chunk(bytes(1024))  # noqa: SLF001

    assert order == ["led:listening", "bytes", "led:thinking", "led:replying", "bytes", "led:idle"]
    await _disconnect(socket, task)


async def test_a_turn_that_raises_still_ends_at_idle():
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)

    async def run_turn_fn(turn_source: Any) -> None:
        await source.send_event({"type": "transcript.final"})
        raise RuntimeError("brain blew up")

    runner = _runner(source, run_turn_fn)
    with pytest.raises(RuntimeError, match="brain blew up"):
        await runner._process_chunk(bytes(1024))  # noqa: SLF001

    assert _led_states(socket) == ["listening", "thinking", "idle"]
    await _disconnect(socket, task)


async def test_a_follow_up_window_shows_listening_again():
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    calls: list[int] = []

    async def run_turn_fn(turn_source: Any) -> None:
        calls.append(1)
        if len(calls) > 1:
            return
        await source.send_event({"type": "transcript.final"})
        await source.send_audio(b"question")
        turn_source.follow_up.request(
            FollowUpRequest(
                kind="confirmation",
                chain_depth=1,
                original_transcript="add dentist",
                question="add dentist?",
                pending_action_id=1,
                playback_ends_at=0.0,
            )
        )

    runner = _runner(
        source,
        run_turn_fn,
        follow_up_window_s=lambda: 6.0,
        follow_up_echo_tail_s=0.0,
        clock=lambda: 0.0,
    )
    await runner._process_chunk(bytes(1024))  # noqa: SLF001

    assert calls == [1, 1]
    assert _led_states(socket) == ["listening", "thinking", "replying", "listening", "idle"]
    await _disconnect(socket, task)


async def test_the_same_state_twice_sends_one_message():
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)

    await source.set_led_state("listening")
    await source.set_led_state("listening")

    assert _led_states(socket) == ["listening"]
    await _disconnect(socket, task)


async def test_no_connected_device_neither_raises_nor_sends():
    source = EdgeAudioSource(_config())

    await source.set_led_state("listening")
    await source.set_led_state("thinking")
    await source.send_audio(b"reply")  # drops with a warning, still no raise


async def test_an_unknown_state_is_ignored(caplog):
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)

    with caplog.at_level(logging.WARNING, logger="atlas.transports.edge"):
        await source.set_led_state("bogus")

    assert _led_states(socket) == []
    assert "bogus" in caplog.text
    await _disconnect(socket, task)


async def test_a_led_send_failure_never_breaks_the_turn_and_warns_once(caplog):
    source = EdgeAudioSource(_config())
    socket = _LedFailingSocket()
    task = await _serve(source, socket)

    with caplog.at_level(logging.WARNING, logger="atlas.transports.edge"):
        await source.set_led_state("listening")
        await source.set_led_state("thinking")
        await source.send_audio(b"reply")  # flips to replying, the send fails, the bytes still go

    assert socket.sent_bytes == [b"reply"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "led state" in r.getMessage()]
    assert len(warnings) == 1
    await _disconnect(socket, task)


async def test_a_new_connection_starts_at_idle_and_sends_on_the_new_socket():
    source = EdgeAudioSource(_config())
    socket_a = FakeEdgeSocket()
    task_a = await _serve(source, socket_a)
    await source.set_led_state("listening")
    assert _led_states(socket_a) == ["listening"]

    socket_b = FakeEdgeSocket()
    task_b = await _serve(source, socket_b)
    await source.set_led_state("listening")

    assert _led_states(socket_b) == ["listening"]
    await _disconnect(socket_b, task_b)
    await asyncio.gather(task_a, return_exceptions=True)


async def test_a_source_without_led_support_runs_a_wake_turn_unchanged():
    source = FakeAudioSource(frames=[b"\x00"])
    turns: list[int] = []

    async def run_turn_fn(turn_source: Any) -> None:
        turns.append(1)

    await _runner(source, run_turn_fn)._process_chunk(bytes(1024))  # noqa: SLF001

    assert turns == [1]


class _EdgeDelegatingSource(FakeAudioSource):
    """A turn source that hands audio and events to a served edge source.

    It declares no `sink_format`, so `run_turn` resolves `sink=None`, plays
    no wake cue, and looks the filler up under the key `None`."""

    def __init__(self, edge: EdgeAudioSource, frames: Any = ()) -> None:
        super().__init__(frames=frames)
        self._edge = edge

    async def send_audio(self, chunk: bytes) -> None:
        await self._edge.send_audio(chunk)

    async def send_event(self, event: dict[str, Any]) -> None:
        await self._edge.send_event(event)


async def test_a_filler_keeps_the_ring_spinning_until_the_answer_audio(
    fake_stt, fake_brain, fake_tts, fake_envelope_client
):
    from atlas.providers.base import BrainReply, FinalTranscript
    from atlas.providers.tier_reply import DEFAULT_FILLER, FILLER_TEXT, FillerPhrase, TierReply
    from atlas.timing import TurnTimings
    from atlas.turn import brain_race
    from atlas.turn.controller import run_turn

    edge = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(edge, socket)
    order: list[Any] = []
    original_text, original_bytes = socket.send_text, socket.send_bytes

    async def record_text(data: str) -> None:
        parsed = json.loads(data)
        if parsed.get("type") == "led":
            order.append(f"led:{parsed['state']}")
        await original_text(data)

    async def record_bytes(data: bytes) -> None:
        order.append(data)
        await original_bytes(data)

    socket.send_text = record_text  # type: ignore[method-assign]
    socket.send_bytes = record_bytes  # type: ignore[method-assign]

    answer_bytes = b"\x01\x02"
    filler_bytes = b"\xfe\xff"
    answer = "it is done"
    reply = TierReply(answer=answer, confident=True, needs_tool=False, filler=FillerPhrase.ONE_MOMENT)
    top_tier = brain_race.TierBrain(
        index=0,
        model="top-model",
        brain=fake_brain(replies=[BrainReply(text=answer)], delay_s=0.05),
        envelope_client=fake_envelope_client(reply=reply, delay_s=0.0),
        calls_tools=True,
    )
    fake_now = [0.0]

    def clock() -> float:
        fake_now[0] += 0.3
        return fake_now[0]

    await run_turn(
        _EdgeDelegatingSource(edge, frames=[b"\x00\x01"]),
        fake_stt(events=[FinalTranscript(text="what time is it")]),
        top_tier.brain,
        fake_tts(chunks=[answer_bytes]),
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=TurnTimings(),
        tiers=[top_tier],
        filler_after_ms=600,
        filler_cache={None: {FILLER_TEXT[DEFAULT_FILLER]: filler_bytes}},
        clock=clock,
        poll_interval_s=0.01,
    )

    assert order == ["led:thinking", filler_bytes, "led:replying", answer_bytes]
    await _disconnect(socket, task)


async def test_filler_audio_sends_no_led_message_while_thinking():
    from atlas.transports.base import speech_kind

    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    await source.set_led_state("thinking")

    token = speech_kind.set("filler")
    try:
        await source.send_audio(b"filler")
    finally:
        speech_kind.reset(token)
    assert _led_states(socket) == ["thinking"]
    assert socket.sent_bytes == [b"filler"]

    await source.send_audio(b"answer")
    assert _led_states(socket) == ["thinking", "replying"]
    assert socket.sent_bytes == [b"filler", b"answer"]
    await _disconnect(socket, task)


async def test_filler_audio_leaves_the_listening_state_alone():
    from atlas.transports.base import speech_kind

    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    await source.set_led_state("listening")

    token = speech_kind.set("filler")
    try:
        await source.send_audio(b"filler")
    finally:
        speech_kind.reset(token)

    assert _led_states(socket) == ["listening"]
    assert socket.sent_bytes == [b"filler"]
    await _disconnect(socket, task)


class _KindRecordingSource:
    def __init__(self) -> None:
        self.kinds: list[str | None] = []

    async def send_audio(self, chunk: bytes) -> None:
        from atlas.transports.base import speech_kind

        self.kinds.append(speech_kind.get())


@pytest.mark.parametrize("kind", ["filler", "answer"])
async def test_speak_shows_its_kind_to_the_source_and_resets_it(fake_tts, kind):
    from atlas.timing import TurnTimings
    from atlas.transports.base import speech_kind
    from atlas.turn.controller import _speak

    source = _KindRecordingSource()
    await _speak(source, fake_tts(chunks=[b"a", b"b"]), TurnTimings(), "text", kind=kind)

    assert source.kinds == [kind, kind]
    assert speech_kind.get() is None


class _RaisingTts:
    async def synthesize(self, text_deltas, sink: Any = None):
        async for _ in text_deltas:
            pass
        yield b"a"
        raise RuntimeError("synthesis failed")


async def test_speak_resets_the_kind_when_synthesis_raises():
    from atlas.timing import TurnTimings
    from atlas.transports.base import speech_kind
    from atlas.turn.controller import _speak

    source = _KindRecordingSource()
    with pytest.raises(RuntimeError):
        await _speak(source, _RaisingTts(), TurnTimings(), "text", kind="filler")

    assert source.kinds == ["filler"]
    assert speech_kind.get() is None
