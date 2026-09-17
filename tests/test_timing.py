"""Real assertions for the per-turn stage-timing validation map.

Turned green by plan 01-05.
"""

import pytest


async def test_stage_timestamps_recorded(fake_audio_source, fake_stt, fake_brain, fake_tts):
    """A full turn driven by fakes records all seven stage timestamps, each
    at or after the one before it, with none left unset.

    The scripted STT events include a partial before the final, so
    `first_partial_at` has something real to record -- a fixture with only
    the final transcript would leave that one stage genuinely unreached,
    which is a different (and also correct) case this test does not cover.
    """
    from types import SimpleNamespace

    from spire_voice.providers.base import BrainReply, FinalTranscript, PartialTranscript, ToolCall
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    class _StubToolHost:
        async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])

    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(
        events=[
            PartialTranscript(text="turn"),
            PartialTranscript(text="turn on"),
            FinalTranscript(text="turn on the fan"),
        ]
    )
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="ha_call_service",
                        arguments={"domain": "switch", "service": "turn_on", "entity_id": "switch.example_fan"},
                    )
                ]
            ),
            BrainReply(text="turned on the fan"),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        _StubToolHost(),
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    stages = [
        timings.turn_started_at,
        timings.stt_socket_open_at,
        timings.first_partial_at,
        timings.stt_final_at,
        timings.brain_first_token_at,
        timings.tool_rounds_done_at,
        timings.first_audio_at,
    ]
    assert all(stage is not None for stage in stages), stages
    assert stages == sorted(stages)
    assert timings.turn_outcome == "completed"


def test_end_of_speech_to_first_audio_is_measured():
    """The budget number is derived from the two recorded timestamps it
    names, not measured a second, separate way."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()
    timings.stt_final_at = 10.0
    timings.first_audio_at = 10.35

    assert timings.end_of_speech_to_first_audio_ms == pytest.approx(350.0)


def test_mark_first_audio_is_idempotent():
    """A second call to `mark_first_audio()` leaves the first timestamp in
    place -- the regression this phase's guard exists to prevent."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()
    timings.mark_first_audio()
    first = timings.first_audio_at

    timings.mark_first_audio()

    assert timings.first_audio_at == first


def test_mark_answer_audio_is_idempotent():
    """A second call to `mark_answer_audio()` leaves the first timestamp in
    place, matching `mark_first_audio()`'s guard."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()
    timings.mark_answer_audio()
    first = timings.answer_audio_at

    timings.mark_answer_audio()

    assert timings.answer_audio_at == first


def test_end_of_speech_to_answer_audio_is_measured():
    """The answer's own budget number is derived from `stt_final_at` and
    `answer_audio_at`, mirroring the first-audio property exactly."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()
    timings.stt_final_at = 10.0
    timings.answer_audio_at = 10.35

    assert timings.end_of_speech_to_answer_audio_ms == pytest.approx(350.0)


def test_end_of_speech_to_answer_audio_is_none_until_both_marks_exist():
    """`None` until both `stt_final_at` and `answer_audio_at` are set --
    matching `end_of_speech_to_first_audio_ms`'s contract."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()
    assert timings.end_of_speech_to_answer_audio_ms is None

    timings.stt_final_at = 10.0
    assert timings.end_of_speech_to_answer_audio_ms is None

    timings.answer_audio_at = 10.35
    assert timings.end_of_speech_to_answer_audio_ms == pytest.approx(350.0)


def test_end_of_speech_to_answer_audio_keeps_fractional_milliseconds():
    """No precision is discarded before the budget comparison: a fractional
    millisecond difference must survive, not round away."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()
    timings.stt_final_at = 10.0
    timings.answer_audio_at = 10.0014

    assert timings.end_of_speech_to_answer_audio_ms == pytest.approx(1.4)


def test_filler_then_answer_leaves_first_audio_and_answer_audio_distinct():
    """RESEARCH.md Pitfall 1's named regression check: a turn on which a
    filler played and an answer followed must have `first_audio_at !=
    answer_audio_at`, with `first_audio_at` the earlier value. This would
    fail if the two marks ever shared one call site."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()

    # The filler's first chunk marks only `first_audio_at`.
    timings.mark_first_audio()
    filler_mark = timings.first_audio_at

    # The answer's first chunk marks only `answer_audio_at` -- it must not
    # also move `first_audio_at`, which `mark_first_audio`'s guard ensures
    # even if a caller mistakenly invoked both marks from the answer path.
    timings.mark_answer_audio()

    assert timings.first_audio_at == filler_mark
    assert timings.answer_audio_at is not None
    assert timings.first_audio_at != timings.answer_audio_at
    assert timings.first_audio_at < timings.answer_audio_at


def test_stage_durations_ms_has_answer_audio_at_after_first_audio_at():
    """`stage_durations_ms()` carries an `answer_audio_at` entry, positioned
    immediately after `first_audio_at` in `_STAGE_ORDER`."""
    from spire_voice.timing import _STAGE_ORDER, TurnTimings

    index = _STAGE_ORDER.index("first_audio_at")
    assert _STAGE_ORDER[index + 1] == "answer_audio_at"

    timings = TurnTimings()
    durations = timings.stage_durations_ms()
    assert "answer_audio_at" in durations


def test_stage_durations_ms_answer_audio_at_is_none_when_unreached():
    """A turn that never reached an answer (no filler, no answer utterance
    marked) reports `None` for the `answer_audio_at` stage duration, not a
    fabricated zero."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()
    timings.turn_started_at = 1.0
    timings.first_audio_at = 1.5

    durations = timings.stage_durations_ms()

    assert durations["answer_audio_at"] is None


def test_to_event_and_log_carry_end_of_speech_to_answer_audio_ms():
    """`to_event()` and `log()` both carry the new derived number alongside
    the existing first-audio number -- the panel has nothing to render for
    the answer number if either omits it."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()
    timings.stt_final_at = 10.0
    timings.mark_first_audio()
    timings.mark_answer_audio()

    event = timings.to_event()

    assert "end_of_speech_to_answer_audio_ms" in event
    assert "answer_audio_at" in event["stage_durations_ms"]

    # log() must not raise and must accept the same fields without a
    # KeyError/AttributeError -- it reads properties, computing nothing new.
    timings.log()
