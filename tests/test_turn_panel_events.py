"""Panel events from run_turn (Phase 15).

`run_turn` emits `wake.confirmed` once, when the server's own transcript
opens with the wake phrase (D-05). A recording source captures every event
`run_turn` sends, so each test reads the real emission order.
"""

from __future__ import annotations

from typing import Sequence

from atlas.providers.base import FinalTranscript, PartialTranscript
from atlas.timing import TurnTimings
from atlas.transports.base import SourceFormat
from atlas.turn.controller import run_turn
from tests.conftest import BrainReply


class _RecordingSource:
    """A source that replays one frame per `frames()` call and records every
    event `run_turn` sends."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.sent_audio: list[bytes] = []

    async def frames(self):
        yield b"\x2a" * 8

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    async def send_event(self, event: dict) -> None:
        self.events.append(event)

    def source_format(self) -> SourceFormat:
        return SourceFormat("pcm", 16000)


class _SequentialStt:
    """One scripted event list per `stream()` call, like the second drain of
    a wake-only first final needs."""

    def __init__(self, calls: Sequence[Sequence[object]]) -> None:
        self._calls = list(calls)
        self.opened = 0

    async def stream(self, frames, source_format=None):
        if self.opened >= len(self._calls):
            raise AssertionError("stream() called more times than scripted")
        events = self._calls[self.opened]
        self.opened += 1
        async for _ in frames:
            pass
        for event in events:
            yield event


class _Brain:
    async def chat(self, messages, tools=None, response_format=None):
        return BrainReply(text="done")


async def _run(source, calls, fake_tts, **kwargs) -> TurnTimings:
    timings = TurnTimings()
    await run_turn(
        source,
        _SequentialStt(calls),
        _Brain(),
        fake_tts(chunks=[b"\x01\x02"]),
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        **kwargs,
    )
    return timings


def _types(source: _RecordingSource) -> list[str]:
    return [event["type"] for event in source.events]


async def test_one_wake_confirmed_comes_before_the_first_partial_that_names_atlas(fake_tts) -> None:
    source = _RecordingSource()
    await _run(
        source,
        [
            [
                PartialTranscript(text="hey"),
                PartialTranscript(text="hey atlas"),
                PartialTranscript(text="hey atlas what"),
                FinalTranscript(text="hey atlas what time is it"),
            ]
        ],
        fake_tts,
        wake_phrase="hey atlas",
    )

    assert _types(source).count("wake.confirmed") == 1
    confirmed = _types(source).index("wake.confirmed")
    named = [
        index
        for index, event in enumerate(source.events)
        if event["type"] == "transcript.partial" and "atlas" in event["text"]
    ]
    assert named, "the script should forward a partial that names atlas"
    assert confirmed < named[0]


async def test_a_wake_only_first_final_then_a_command_confirms_one_time(fake_tts) -> None:
    source = _RecordingSource()
    await _run(
        source,
        [[FinalTranscript(text="hey atlas")], [FinalTranscript(text="turn on the lights")]],
        fake_tts,
        wake_phrase="hey atlas",
    )

    assert _types(source).count("wake.confirmed") == 1


async def test_a_turn_with_no_wake_phrase_confirms_nothing(fake_tts) -> None:
    source = _RecordingSource()
    await _run(source, [[FinalTranscript(text="hey atlas what time is it")]], fake_tts)

    assert "wake.confirmed" not in _types(source)


async def test_a_follow_up_turn_confirms_nothing(fake_tts) -> None:
    from atlas.turn.follow_up import FollowUpChannel, FollowUpRequest

    source = _RecordingSource()
    source.follow_up = FollowUpChannel(
        incoming=FollowUpRequest(
            kind="clarification",
            chain_depth=1,
            original_transcript="add dentist on friday at 3",
            question="home or work?",
        )
    )
    await _run(source, [[FinalTranscript(text="hey atlas home")]], fake_tts, wake_phrase="hey atlas")

    assert "wake.confirmed" not in _types(source)


async def test_a_transcript_that_never_opens_with_the_phrase_confirms_nothing(fake_tts) -> None:
    source = _RecordingSource()
    await _run(
        source,
        [[FinalTranscript(text="You always say AM in the morning.")]],
        fake_tts,
        wake_phrase="hey atlas",
        verify_wake=True,
    )

    assert "wake.confirmed" not in _types(source)
