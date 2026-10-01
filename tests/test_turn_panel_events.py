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


# --- run_turn exits ---------------------------------------------------------


class _SinkSource(_RecordingSource):
    """The camera shape: an 8 kHz A-law sink, so a reply has a playback length."""

    def sink_format(self):
        from atlas.providers.tts_xai import SinkFormat

        return SinkFormat(codec="alaw", sample_rate=8000)


class _RaisingBrain:
    def __init__(self, error: BaseException, delay_s: float = 0.0) -> None:
        self._error = error
        self._delay_s = delay_s

    async def chat(self, messages, tools=None, response_format=None):
        if self._delay_s:
            import asyncio

            await asyncio.sleep(self._delay_s)
        raise self._error


class _HangingStt:
    """A transcriber that never answers, so only a cancellation ends the turn."""

    async def stream(self, frames, source_format=None):
        import asyncio

        async for _ in frames:
            pass
        await asyncio.Event().wait()
        yield  # pragma: no cover


async def _run_with(source, stt, brain, tts, **kwargs) -> TurnTimings:
    timings = TurnTimings()
    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        **kwargs,
    )
    return timings


def _ended(source: _RecordingSource) -> list[dict]:
    return [event for event in source.events if event["type"] == "turn.ended"]


async def test_a_completed_turn_ends_with_one_turn_ended_after_turn_timing(fake_tts) -> None:
    source = _RecordingSource()
    await _run(source, [[FinalTranscript(text="what time is it")]], fake_tts)

    assert _types(source).count("turn.ended") == 1
    assert source.events[-1]["type"] == "turn.ended"
    assert source.events[-1]["outcome"] == "completed"
    assert _types(source).index("turn.timing") < _types(source).index("turn.ended")


async def test_an_empty_transcript_turn_ends_with_the_empty_transcript_outcome(fake_tts) -> None:
    source = _RecordingSource()
    await _run(source, [[FinalTranscript(text="")]], fake_tts)

    (ended,) = _ended(source)
    assert ended["outcome"] == "empty_transcript"
    assert source.events[-1] is ended


async def test_a_wake_unverified_turn_ends_with_that_outcome(fake_tts) -> None:
    source = _RecordingSource()
    await _run(
        source,
        [[FinalTranscript(text="You always say AM in the morning.")]],
        fake_tts,
        wake_phrase="hey atlas",
        verify_wake=True,
    )

    (ended,) = _ended(source)
    assert ended["outcome"] == "wake_unverified"


async def test_a_turn_that_raises_ends_failed_and_the_exception_still_propagates(fake_tts) -> None:
    import pytest

    source = _RecordingSource()
    with pytest.raises(RuntimeError, match="boom"):
        await _run_with(
            source,
            _SequentialStt([[FinalTranscript(text="what time is it")]]),
            _RaisingBrain(RuntimeError("boom")),
            fake_tts(chunks=[b"\x01\x02"]),
        )

    (ended,) = _ended(source)
    assert ended["outcome"] == "failed"


async def test_a_turn_cancelled_while_it_listens_ends_cancelled(fake_tts) -> None:
    import asyncio

    source = _RecordingSource()
    task = asyncio.ensure_future(_run_with(source, _HangingStt(), _Brain(), fake_tts(chunks=[b"\x01"])))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    (ended,) = _ended(source)
    assert ended["outcome"] == "cancelled"


async def test_an_observer_fault_while_turn_ended_is_sent_changes_nothing(fake_tts) -> None:
    import pytest

    class _FaultySource(_RecordingSource):
        async def send_event(self, event: dict) -> None:
            if event["type"] == "turn.ended":
                raise OSError("observer is gone")
            await super().send_event(event)

    completed = _FaultySource()
    timings = await _run(completed, [[FinalTranscript(text="what time is it")]], fake_tts)
    assert timings.turn_outcome == "completed"

    failing = _FaultySource()
    with pytest.raises(RuntimeError, match="boom"):
        await _run_with(
            failing,
            _SequentialStt([[FinalTranscript(text="what time is it")]]),
            _RaisingBrain(RuntimeError("boom")),
            fake_tts(chunks=[b"\x01\x02"]),
        )


async def test_a_reply_that_plays_for_a_second_reports_the_time_it_has_left(fake_tts) -> None:
    source = _SinkSource()
    await _run_with(
        source,
        _SequentialStt([[FinalTranscript(text="what time is it")]]),
        _Brain(),
        fake_tts(chunks=[b"\x01" * 8000]),
    )

    (ended,) = _ended(source)
    assert 500 <= ended["playback_ms_left"] <= 1000


async def test_one_reply_started_carries_the_answer_text_before_reply_text(fake_tts) -> None:
    source = _RecordingSource()
    await _run(source, [[FinalTranscript(text="what time is it")]], fake_tts)

    assert _types(source).count("reply.started") == 1
    started = _types(source).index("reply.started")
    assert source.events[started]["text"] == "done"
    assert started < _types(source).index("reply.text")


async def _filler_turn(fake_brain, fake_envelope_client, brain, fake_tts, source):
    from atlas.providers.tier_reply import DEFAULT_FILLER, FILLER_TEXT, FillerPhrase, TierReply
    from atlas.turn import brain_race

    top_reply = TierReply(answer="it is done", confident=True, needs_tool=False, filler=FillerPhrase.ONE_MOMENT)
    top_tier = brain_race.TierBrain(
        index=0,
        model="top-model",
        brain=brain,
        envelope_client=fake_envelope_client(reply=top_reply, delay_s=0.0),
        calls_tools=True,
    )
    fake_now = [0.0]

    def clock() -> float:
        fake_now[0] += 0.3
        return fake_now[0]

    return await _run_with(
        source,
        _SequentialStt([[FinalTranscript(text="what time is it")]]),
        top_tier.brain,
        fake_tts(chunks=[b"\x01\x02"]),
        tiers=[top_tier],
        filler_after_ms=600,
        filler_cache={None: {FILLER_TEXT[DEFAULT_FILLER]: b"\xfe\xff"}},
        clock=clock,
        poll_interval_s=0.01,
    )


async def test_a_filler_starts_no_reply_and_the_answer_after_it_starts_one(
    fake_brain, fake_envelope_client, fake_tts
) -> None:
    source = _RecordingSource()
    await _filler_turn(
        fake_brain,
        fake_envelope_client,
        fake_brain(replies=[BrainReply(text="it is done")], delay_s=0.05),
        fake_tts,
        source,
    )

    assert source.sent_audio[0] == b"\xfe\xff"
    started = [event for event in source.events if event["type"] == "reply.started"]
    assert [event["text"] for event in started] == ["it is done"]


async def test_a_filler_before_a_failure_starts_no_reply(fake_brain, fake_envelope_client, fake_tts) -> None:
    import pytest

    source = _RecordingSource()
    with pytest.raises(RuntimeError, match="boom"):
        await _filler_turn(
            fake_brain,
            fake_envelope_client,
            _RaisingBrain(RuntimeError("boom"), delay_s=0.05),
            fake_tts,
            source,
        )

    assert source.sent_audio[:1] == [b"\xfe\xff"]
    assert "reply.started" not in _types(source)
    (ended,) = _ended(source)
    assert ended["outcome"] == "failed"


async def test_a_request_left_on_the_channel_before_the_turn_is_not_this_turns(fake_tts) -> None:
    from atlas.turn.follow_up import FollowUpChannel, FollowUpRequest

    source = _RecordingSource()
    source.follow_up = FollowUpChannel(
        requested=FollowUpRequest(
            kind="confirmation", chain_depth=1, original_transcript="old", question="really?"
        )
    )
    await _run(source, [[FinalTranscript(text="what time is it")]], fake_tts)

    (ended,) = _ended(source)
    assert ended["follow_up"] is False
    assert ended["asks_question"] is False


async def test_an_answer_window_this_turn_opens_shows_as_a_follow_up_that_asks_nothing(fake_tts) -> None:
    from atlas.turn.follow_up import FollowUpChannel

    source = _RecordingSource()
    source.follow_up = FollowUpChannel()
    await _run(source, [[FinalTranscript(text="what time is it")]], fake_tts, answer_windows=True)

    (ended,) = _ended(source)
    assert source.follow_up.requested is not None
    assert ended["follow_up"] is True
    assert ended["asks_question"] is False


def test_the_browser_timing_event_does_not_carry_the_playback_estimate() -> None:
    timings = TurnTimings()
    timings.reply_playback_end_at = 5.0

    assert "reply_playback_end_at" not in timings.to_event()


async def test_both_turns_of_a_reply_group_start_a_reply_and_end_once(tmp_path, fake_stt, fake_brain, fake_tts) -> None:
    from test_turn_group_speech import _Turn, _of_type, _run_both
    from atlas.turn.reply_group import GroupSpeaker

    speaker = GroupSpeaker(merge_wait_s=0.2)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    common = dict(live_tts=live_tts, fake_stt=fake_stt, fake_brain=fake_brain)
    josh = _Turn(tmp_path, speaker, key="src:1", order_frame=100, answer="the fan is on", member=(1, "Josh"), **common)
    sam = _Turn(tmp_path, speaker, key="src:2", order_frame=300, answer="the door is locked", member=(2, "Sam"), **common)

    await _run_both(josh, sam)

    for turn, own_answer in ((josh, "the fan is on"), (sam, "the door is locked")):
        events = turn.events
        started = _of_type(events, "reply.started")
        assert [event["text"] for event in started] == [own_answer]
        assert len(_of_type(events, "turn.ended")) == 1
        assert events[-1]["type"] == "turn.ended"
