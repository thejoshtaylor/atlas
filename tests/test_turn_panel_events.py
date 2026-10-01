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


# --- Pure builders ---------------------------------------------------------


def test_asks_question_is_false_without_a_request() -> None:
    from atlas.turn.panel_events import asks_question

    assert asks_question(None) is False


def test_a_confirmation_or_a_clarification_asks_a_question() -> None:
    from atlas.turn.follow_up import FollowUpRequest
    from atlas.turn.panel_events import asks_question

    for kind in ("confirmation", "clarification"):
        request = FollowUpRequest(kind=kind, chain_depth=1, original_transcript="x", question="y?")
        assert asks_question(request) is True


def test_an_answer_window_asks_a_question_only_when_the_reply_expects_one() -> None:
    from atlas.turn.follow_up import FollowUpRequest
    from atlas.turn.panel_events import asks_question

    plain = FollowUpRequest(kind="answer", chain_depth=1, original_transcript="x", question="", expects_reply=False)
    asked = FollowUpRequest(kind="answer", chain_depth=1, original_transcript="x", question="", expects_reply=True)

    assert asks_question(plain) is False
    assert asks_question(asked) is True


def test_a_request_left_from_an_earlier_turn_is_not_this_turns_request() -> None:
    from atlas.turn.panel_events import request_of_this_turn

    older = object()
    newer = object()

    assert request_of_this_turn(None, None) is None
    assert request_of_this_turn(older, older) is None
    assert request_of_this_turn(newer, older) is newer
    assert request_of_this_turn(newer, None) is newer


def test_a_completed_turn_without_a_request_ends_quiet() -> None:
    from atlas.turn.panel_events import turn_ended_event

    event = turn_ended_event(
        outcome="completed", failed=False, cancelled=False, request=None, playback_end_at=None, now=10.0
    )

    assert event == {
        "type": "turn.ended",
        "outcome": "completed",
        "follow_up": False,
        "asks_question": False,
        "playback_ms_left": 0,
    }


def test_failed_wins_over_cancelled_and_over_the_outcome_argument() -> None:
    from atlas.turn.panel_events import turn_ended_event

    kwargs = dict(request=None, playback_end_at=None, now=1.0)

    assert turn_ended_event(outcome="completed", failed=True, cancelled=False, **kwargs)["outcome"] == "failed"
    assert turn_ended_event(outcome="completed", failed=True, cancelled=True, **kwargs)["outcome"] == "failed"
    assert turn_ended_event(outcome="completed", failed=False, cancelled=True, **kwargs)["outcome"] == "cancelled"


def test_playback_ms_left_counts_to_the_end_of_the_reply_and_never_goes_negative() -> None:
    from atlas.turn.panel_events import turn_ended_event

    kwargs = dict(outcome="completed", failed=False, cancelled=False, request=None, now=10.0)

    ahead = turn_ended_event(playback_end_at=12.5, **kwargs)["playback_ms_left"]
    behind = turn_ended_event(playback_end_at=9.0, **kwargs)["playback_ms_left"]

    assert ahead == 2500
    assert isinstance(ahead, int)
    assert behind == 0


def test_an_answer_window_after_a_plain_answer_is_a_follow_up_that_asks_nothing() -> None:
    from atlas.turn.follow_up import FollowUpRequest
    from atlas.turn.panel_events import turn_ended_event

    request = FollowUpRequest(kind="answer", chain_depth=1, original_transcript="x", question="", expects_reply=False)

    event = turn_ended_event(
        outcome="completed", failed=False, cancelled=False, request=request, playback_end_at=None, now=0.0
    )

    assert event["follow_up"] is True
    assert event["asks_question"] is False


def test_reply_started_carries_the_answer_text() -> None:
    from atlas.turn.panel_events import reply_started_event

    assert reply_started_event("It is sunny.") == {"type": "reply.started", "text": "It is sunny."}
