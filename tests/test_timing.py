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

    from atlas.providers.base import BrainReply, FinalTranscript, PartialTranscript, ToolCall
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

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
        timings.speech_end_at,
        timings.stt_final_at,
        timings.brain_first_round_at,
        timings.tool_rounds_done_at,
        timings.first_audio_at,
    ]
    assert all(stage is not None for stage in stages), stages
    assert stages == sorted(stages)
    assert timings.turn_outcome == "completed"


def test_end_of_speech_to_first_audio_is_measured():
    """The budget number is derived from the two recorded timestamps it
    names, not measured a second, separate way."""
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.stt_final_at = 10.0
    timings.first_audio_at = 10.35

    assert timings.end_of_speech_to_first_audio_ms == pytest.approx(350.0)


def test_mark_first_audio_is_idempotent():
    """A second call to `mark_first_audio()` leaves the first timestamp in
    place -- the regression this phase's guard exists to prevent."""
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.mark_first_audio()
    first = timings.first_audio_at

    timings.mark_first_audio()

    assert timings.first_audio_at == first


def test_mark_answer_audio_is_idempotent():
    """A second call to `mark_answer_audio()` leaves the first timestamp in
    place, matching `mark_first_audio()`'s guard."""
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.mark_answer_audio()
    first = timings.answer_audio_at

    timings.mark_answer_audio()

    assert timings.answer_audio_at == first


def test_end_of_speech_to_answer_audio_is_measured():
    """The answer's own budget number is derived from `stt_final_at` and
    `answer_audio_at`, mirroring the first-audio property exactly."""
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.stt_final_at = 10.0
    timings.answer_audio_at = 10.35

    assert timings.end_of_speech_to_answer_audio_ms == pytest.approx(350.0)


def test_end_of_speech_to_answer_audio_is_none_until_both_marks_exist():
    """`None` until both `stt_final_at` and `answer_audio_at` are set --
    matching `end_of_speech_to_first_audio_ms`'s contract."""
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    assert timings.end_of_speech_to_answer_audio_ms is None

    timings.stt_final_at = 10.0
    assert timings.end_of_speech_to_answer_audio_ms is None

    timings.answer_audio_at = 10.35
    assert timings.end_of_speech_to_answer_audio_ms == pytest.approx(350.0)


def test_end_of_speech_to_answer_audio_keeps_fractional_milliseconds():
    """No precision is discarded before the budget comparison: a fractional
    millisecond difference must survive, not round away."""
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.stt_final_at = 10.0
    timings.answer_audio_at = 10.0014

    assert timings.end_of_speech_to_answer_audio_ms == pytest.approx(1.4)


def test_filler_then_answer_leaves_first_audio_and_answer_audio_distinct():
    """RESEARCH.md Pitfall 1's named regression check: a turn on which a
    filler played and an answer followed must have `first_audio_at !=
    answer_audio_at`, with `first_audio_at` the earlier value. This would
    fail if the two marks ever shared one call site."""
    from atlas.timing import TurnTimings

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
    from atlas.timing import _STAGE_ORDER, TurnTimings

    index = _STAGE_ORDER.index("first_audio_at")
    assert _STAGE_ORDER[index + 1] == "answer_audio_at"

    timings = TurnTimings()
    durations = timings.stage_durations_ms()
    assert "answer_audio_at" in durations


def test_stage_durations_ms_answer_audio_at_is_none_when_unreached():
    """A turn that never reached an answer (no filler, no answer utterance
    marked) reports `None` for the `answer_audio_at` stage duration, not a
    fabricated zero."""
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.turn_started_at = 1.0
    timings.first_audio_at = 1.5

    durations = timings.stage_durations_ms()

    assert durations["answer_audio_at"] is None


def test_to_event_and_log_carry_end_of_speech_to_answer_audio_ms():
    """`to_event()` and `log()` both carry the new derived number alongside
    the existing first-audio number -- the panel has nothing to render for
    the answer number if either omits it."""
    from atlas.timing import TurnTimings

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


# --- 260924-4iv (item e): speech_end_at, endpointing_delay_ms, and the
# renamed brain_first_round_at mark ----------------------------------------


def test_stage_order_places_speech_end_at_between_first_partial_and_stt_final():
    from atlas.timing import _STAGE_ORDER, TurnTimings

    assert _STAGE_ORDER.index("first_partial_at") + 1 == _STAGE_ORDER.index("speech_end_at")
    assert _STAGE_ORDER.index("speech_end_at") + 1 == _STAGE_ORDER.index("stt_final_at")
    assert "brain_first_round_at" in _STAGE_ORDER
    assert "brain_first_token_at" not in _STAGE_ORDER

    timings = TurnTimings()
    assert not hasattr(timings, "brain_first_token_at")
    assert not hasattr(timings, "mark_brain_first_token")


def test_mark_speech_end_assigns_unconditionally_the_last_call_wins():
    """Unlike every other mark, a later call must win -- the wake-phrase
    second drain's own speech-end time must be able to overwrite the first
    drain's, and a drain that reached no word-changing partial must be able
    to clear a stale value."""
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.mark_speech_end(5.0)
    assert timings.speech_end_at == 5.0
    timings.mark_speech_end(7.0)
    assert timings.speech_end_at == 7.0
    timings.mark_speech_end(None)
    assert timings.speech_end_at is None


def test_mark_first_partial_accepts_an_explicit_arrival_time_and_stays_idempotent():
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.mark_first_partial(at=3.0)
    assert timings.first_partial_at == 3.0
    timings.mark_first_partial(at=9.0)
    assert timings.first_partial_at == 3.0

    # `at=None` (the default) still reads the real clock, matching the
    # pre-4iv no-argument call.
    timings2 = TurnTimings()
    timings2.mark_first_partial()
    assert timings2.first_partial_at is not None


def test_endpointing_delay_and_speech_end_to_audio_properties_are_measured():
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.speech_end_at = 10.0
    timings.stt_final_at = 10.4
    timings.first_audio_at = 10.9
    timings.answer_audio_at = 11.2

    assert timings.endpointing_delay_ms == pytest.approx(400.0)
    assert timings.speech_end_to_first_audio_ms == pytest.approx(900.0)
    assert timings.speech_end_to_answer_audio_ms == pytest.approx(1200.0)


def test_endpointing_delay_and_speech_end_to_audio_are_none_until_both_marks_exist():
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    assert timings.endpointing_delay_ms is None
    assert timings.speech_end_to_first_audio_ms is None
    assert timings.speech_end_to_answer_audio_ms is None

    timings.speech_end_at = 10.0
    assert timings.endpointing_delay_ms is None
    assert timings.speech_end_to_first_audio_ms is None

    timings.stt_final_at = 10.4
    assert timings.endpointing_delay_ms == pytest.approx(400.0)


def test_to_event_and_log_carry_the_new_speech_end_keys():
    from atlas.timing import TurnTimings

    timings = TurnTimings()
    timings.speech_end_at = 10.0
    timings.stt_final_at = 10.4
    timings.mark_first_audio()
    timings.mark_answer_audio()

    event = timings.to_event()
    assert "endpointing_delay_ms" in event
    assert "speech_end_to_first_audio_ms" in event
    assert "speech_end_to_answer_audio_ms" in event
    assert "end_of_speech_to_first_audio_ms" in event
    assert "end_of_speech_to_answer_audio_ms" in event

    # log() must not raise against the new fields either.
    timings.log()


async def test_speech_end_at_lands_on_the_last_word_changing_partials_arrival(
    fake_audio_source, fake_brain, fake_tts
):
    """The tracer's own end-to-end proof: a scripted STT records its own
    arrival time (`time.monotonic()`) just before each event, with a short
    real sleep between them. `speech_end_at` must land on the arrival of
    the last partial whose normalized words changed -- "turn on the fan"
    -- not on the punctuation-only-different repeat, and not on the final
    transcript's own (later) endpointing decision.
    """
    import asyncio
    import time as _time

    from atlas.providers.base import BrainReply, FinalTranscript, PartialTranscript
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    class _TimedScriptedStt:
        def __init__(self, script):
            self._script = script
            self.arrivals: list[float] = []

        async def stream(self, frames, source_format=None):
            for text, is_final, delay in self._script:
                await asyncio.sleep(delay)
                self.arrivals.append(_time.monotonic())
                yield FinalTranscript(text=text) if is_final else PartialTranscript(text=text)

    script = [
        ("turn on", False, 0.02),
        ("turn on the fan", False, 0.02),
        ("Turn on the fan.", False, 0.02),
        ("turn on the fan", True, 0.02),
    ]
    stt = _TimedScriptedStt(script)
    source = fake_audio_source(frames=[b"\x00\x01"] * 4)
    brain = fake_brain(replies=[BrainReply(text="ok")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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
    )

    t0, t1, t2, t3 = stt.arrivals
    assert t0 <= timings.first_partial_at < t1
    assert t1 <= timings.speech_end_at < t2
    assert timings.stt_final_at >= t3
    assert timings.endpointing_delay_ms is not None and timings.endpointing_delay_ms > 0
    assert timings.endpointing_delay_ms == pytest.approx((timings.stt_final_at - timings.speech_end_at) * 1000)
    assert timings.first_partial_at <= timings.speech_end_at <= timings.stt_final_at


async def test_speech_end_at_stays_none_with_only_a_final_transcript(fake_audio_source, fake_brain, fake_tts):
    from atlas.providers.base import BrainReply, FinalTranscript
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    class _SttNoPartials:
        async def stream(self, frames, source_format=None):
            yield FinalTranscript(text="turn on the fan")

    source = fake_audio_source(frames=[b"\x00\x01"])
    brain = fake_brain(replies=[BrainReply(text="ok")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        _SttNoPartials(),
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert timings.speech_end_at is None
    assert timings.endpointing_delay_ms is None
    assert timings.speech_end_to_first_audio_ms is None
    assert timings.speech_end_to_answer_audio_ms is None
    assert timings.end_of_speech_to_first_audio_ms is not None


async def test_a_punctuation_only_partial_never_sets_speech_end_at(fake_audio_source, fake_brain, fake_tts):
    from atlas.providers.base import BrainReply, FinalTranscript, PartialTranscript
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    class _SttPunctuationOnlyPartial:
        async def stream(self, frames, source_format=None):
            yield PartialTranscript(text="...")
            yield FinalTranscript(text="turn on the fan")

    source = fake_audio_source(frames=[b"\x00\x01"])
    brain = fake_brain(replies=[BrainReply(text="ok")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        _SttPunctuationOnlyPartial(),
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert timings.speech_end_at is None


async def test_speech_end_at_reflects_the_second_drain_not_the_wake_only_first(
    fake_audio_source, fake_brain, fake_tts
):
    """A wake-phrase turn whose first drain is wake-only, and whose second
    drain has its own word-changing partial: `speech_end_at` must land in
    the second drain's own time range, never carried over from the first
    (which reached no word-changing partial at all -- only its own final
    "hey atlas")."""
    import asyncio
    import time as _time

    from atlas.providers.base import BrainReply, FinalTranscript, PartialTranscript
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    class _TwoCallStt:
        def __init__(self):
            self.calls = 0
            self.second_drain_partial_arrival: float | None = None

        async def stream(self, frames, source_format=None):
            self.calls += 1
            if self.calls == 1:
                yield FinalTranscript(text="hey atlas")
                return
            yield PartialTranscript(text="turn")
            await asyncio.sleep(0.02)
            self.second_drain_partial_arrival = _time.monotonic()
            yield PartialTranscript(text="turn on the fan")
            await asyncio.sleep(0.02)
            yield FinalTranscript(text="turn on the fan")

    stt = _TwoCallStt()
    source = fake_audio_source(frames=[b"\x00\x01"] * 6)
    brain = fake_brain(replies=[BrainReply(text="ok")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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
        wake_phrase="hey atlas",
    )

    assert stt.calls == 2
    assert timings.speech_end_at is not None
    # The mark can only have come from the second drain's own "turn on the
    # fan" partial -- the first drain never reached a word-changing
    # partial at all.
    assert timings.speech_end_at == pytest.approx(stt.second_drain_partial_arrival, abs=0.01)
