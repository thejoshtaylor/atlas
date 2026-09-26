"""D-09 through D-13 (10-05-PLAN.md): a Pi's own `vad.end` finalizes
speech-to-text at once, through `turn/early_finalize.py::wait_for_end_of_speech`
and its wiring into `turn/controller.py::_drain_to_final_transcript`.

Uses the real `atlas.transports.edge.SpeechSignals` throughout -- this
module's own subscribe/publish/`in_speech` contract is exactly what
`wait_for_end_of_speech` is duck-typed against, so there is no value in a
second fake reimplementing it.
"""

from __future__ import annotations

import asyncio

import pytest
from typing import AsyncIterator

from atlas.providers.base import BrainReply, FinalTranscript, PartialTranscript
from atlas.session.recorder import _serialize_timings
from atlas.transports.base import SourceFormat
from atlas.transports.edge import SpeechSignals
from atlas.turn.early_finalize import wait_for_end_of_speech


# ---------------------------------------------------------------------------
# wait_for_end_of_speech
# ---------------------------------------------------------------------------


async def test_wait_for_end_of_speech_returns_the_next_vad_end():
    signals = SpeechSignals(hangover_s=0.0)
    fake_now = [100.0]

    def clock() -> float:
        return fake_now[0]

    async def _publish() -> None:
        await asyncio.sleep(0.01)
        fake_now[0] = 105.0
        signals.publish({"type": "vad.end", "seq": 1})

    task = asyncio.ensure_future(_publish())
    at = await wait_for_end_of_speech(signals, hangover_s=0.0, already_ended_counts=False, clock=clock)
    await task

    assert at == 105.0


async def test_wait_for_end_of_speech_hangover_waits_past_a_vad_start():
    """A `vad.start` inside the hangover cancels that candidate `vad.end`;
    the following `vad.end` is what this call actually returns."""
    signals = SpeechSignals(hangover_s=0.2)
    fake_now = [0.0]

    def clock() -> float:
        return fake_now[0]

    async def _publish() -> None:
        fake_now[0] = 1.0
        signals.publish({"type": "vad.end", "seq": 1})  # candidate, cancelled below
        await asyncio.sleep(0.02)
        signals.publish({"type": "vad.start", "seq": 2})  # inside the 0.2s hangover
        await asyncio.sleep(0.02)
        fake_now[0] = 2.0
        signals.publish({"type": "vad.end", "seq": 3})  # the one that actually ends the wait

    task = asyncio.ensure_future(_publish())
    at = await wait_for_end_of_speech(signals, hangover_s=0.2, already_ended_counts=False, clock=clock)
    await task

    assert at == 2.0


async def test_wait_for_end_of_speech_already_ended_counts_true_returns_at_once():
    """`already_ended_counts=True` plus `in_speech` already `False` (the
    default -- nothing was ever published) satisfies the wait immediately,
    with no event required at all."""
    signals = SpeechSignals(hangover_s=0.0)
    fake_now = [42.0]

    at = await wait_for_end_of_speech(
        signals, hangover_s=0.0, already_ended_counts=True, clock=lambda: fake_now[0]
    )

    assert at == 42.0


async def test_wait_for_end_of_speech_already_ended_counts_false_waits_for_a_new_vad_end():
    """The identical starting state as the test above (`in_speech` already
    `False`) does NOT satisfy the wait when `already_ended_counts=False` --
    proven by the returned time being the later, published one, not the
    time the call started."""
    signals = SpeechSignals(hangover_s=0.0)
    fake_now = [0.0]

    def clock() -> float:
        return fake_now[0]

    async def _publish() -> None:
        await asyncio.sleep(0.01)
        fake_now[0] = 9.0
        signals.publish({"type": "vad.end", "seq": 1})

    task = asyncio.ensure_future(_publish())
    at = await wait_for_end_of_speech(signals, hangover_s=0.0, already_ended_counts=False, clock=clock)
    await task

    assert at == 9.0


# ---------------------------------------------------------------------------
# run_turn wiring
# ---------------------------------------------------------------------------


class _SpeechSignalsSource:
    """A `fake_audio_source`-shaped double that also exposes a real
    `SpeechSignals` -- the shape `run_turn` sees once plan 10-02's wrapper
    forwarding properties reach an `EdgeAudioSource`. `frames()` never ends
    on its own, mirroring a live mic (`stt_xai.py`'s own `sender()`
    docstring states the same fact)."""

    def __init__(self, speech_signals: SpeechSignals) -> None:
        self.speech_signals = speech_signals
        self.sent_audio: list[bytes] = []

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            yield b"\x00\x00"
            await asyncio.sleep(0.01)

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    def source_format(self) -> SourceFormat:
        return SourceFormat("pcm", 16000)


class _FinalizeGatedStt:
    """Yields its scripted final only once `finalize` is set -- the same
    shape `stt_xai.py`'s real finalizer drives, without a real socket. One
    scripted text per call, in call order (`test_wake_only_second_drain_
    waits_for_the_next_segment` calls this twice, once per drain)."""

    def __init__(self, texts: "list[str]", *, partials: bool = True) -> None:
        self._texts = list(texts)
        self._partials = partials
        self.call_count = 0

    async def stream(self, frames, source_format, *, finalize=None):
        text = self._texts[self.call_count]
        self.call_count += 1
        if self._partials:
            # Real xAI streams partials as it hears words; early finalize
            # only acts once one has carried words.
            yield PartialTranscript(text=text)
        if finalize is not None:
            await finalize.wait()
        yield FinalTranscript(text=text)


class _ScriptedStt:
    """Like `tests/conftest.py`'s own `FakeStt`, but accepts (and ignores)
    a `finalize` keyword -- required once a source carries `speech_signals`,
    since `_drain_to_final_transcript` then always passes `finalize=`."""

    def __init__(self, events=(), hang: bool = False) -> None:
        self._events = list(events)
        self._hang = hang

    async def stream(self, frames, source_format=None, *, finalize=None):
        for event in self._events:
            yield event
        if self._hang:
            await asyncio.Event().wait()


async def test_vad_end_finalizes_the_turn(fake_brain, fake_tts):
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    signals = SpeechSignals(hangover_s=0.0)
    # Published before the drain ever starts watching, so `in_speech` is
    # already True by the time the watch subscribes -- the ordinary case,
    # not the already-ended one the wake-only tests below exercise.
    signals.publish({"type": "vad.start", "seq": 1})

    source = _SpeechSignalsSource(signals)
    stt = _FinalizeGatedStt(["turn on the fan"])
    brain = fake_brain(replies=[BrainReply(text="the fan is on")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    async def _end_speech_soon() -> None:
        await asyncio.sleep(0.02)
        signals.publish({"type": "vad.end", "seq": 2})

    asyncio.ensure_future(_end_speech_soon())

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

    assert timings.turn_outcome == "completed"
    assert timings.vad_end_at is not None
    assert timings.vad_end_to_stt_final_ms is not None
    payload = _serialize_timings(timings, None, 0)
    assert "vad_end_to_stt_final_ms" in payload


async def test_lost_vad_end_never_hangs_the_turn(fake_brain, fake_tts):
    """D-13: xAI's own endpointing final, or `max_utterance_s`, still ends
    the turn when no `vad.end` is ever published -- the finalize event is
    never set in either case, proven by `timings.vad_end_at` staying
    `None`."""
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    # Case A: the provider's own endpointing final ends the turn. `vad.start`
    # is published (so `in_speech` is True, the real "a segment began and its
    # end never arrived" shape) but its `vad.end` never is -- without this,
    # `already_ended_counts=True`'s own "already ended" branch would return
    # at once and this would test the wake-only case instead of a lost event.
    signals_a = SpeechSignals(hangover_s=0.0)
    signals_a.publish({"type": "vad.start", "seq": 1})
    source_a = _SpeechSignalsSource(signals_a)
    stt_a = _ScriptedStt(events=[FinalTranscript(text="turn on the fan")])
    brain_a = fake_brain(replies=[BrainReply(text="done")])
    tts_a = fake_tts(chunks=[b"\x01\x02"])
    timings_a = TurnTimings()

    await run_turn(
        source_a,
        stt_a,
        brain_a,
        tts_a,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings_a,
    )

    assert timings_a.turn_outcome == "completed"
    assert timings_a.vad_end_at is None

    # Case B: no final at all -- ends at max_utterance_s. Same "vad.start
    # seen, vad.end lost" shape as case A above.
    signals_b = SpeechSignals(hangover_s=0.0)
    signals_b.publish({"type": "vad.start", "seq": 1})
    source_b = _SpeechSignalsSource(signals_b)
    stt_b = _ScriptedStt(hang=True)
    brain_b = fake_brain(replies=[])
    tts_b = fake_tts(chunks=[b"\x01\x02"])
    timings_b = TurnTimings()

    fake_now = [0.0]

    def clock() -> float:
        fake_now[0] += 5.0
        return fake_now[0]

    await run_turn(
        source_b,
        stt_b,
        brain_b,
        tts_b,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings_b,
        max_utterance_s=15,
        clock=clock,
        poll_interval_s=0.01,
    )

    assert timings_b.turn_outcome == "timeout"
    assert timings_b.vad_end_at is None


async def test_wake_only_second_drain_waits_for_the_next_segment(fake_brain, fake_tts):
    """260922-woc: the first drain ends on a wake-only transcript (the
    segment already ended before speech-to-text even opened, so the first
    drain's own watch finalizes at once); the second drain must NOT
    finalize on that same already-ended state -- only a genuinely new
    segment's own `vad.end`, published after it began, may finalize it."""
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    signals = SpeechSignals(hangover_s=0.0)
    source = _SpeechSignalsSource(signals)
    stt = _FinalizeGatedStt(["hey atlas", "turn on the fan"])
    brain = fake_brain(replies=[BrainReply(text="done")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    async def _second_segment() -> None:
        await asyncio.sleep(0.02)
        signals.publish({"type": "vad.start", "seq": 10})
        await asyncio.sleep(0.01)
        signals.publish({"type": "vad.end", "seq": 11})

    asyncio.ensure_future(_second_segment())

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

    assert stt.call_count == 2
    assert tts.received_text == ["done"]


async def test_source_without_speech_signals_is_unchanged(fake_audio_source, fake_stt, fake_brain, fake_tts):
    """A source with no `speech_signals` at all, and an STT double
    (`FakeStt`) whose `stream` takes only `(frames, source_format)`, runs a
    whole turn exactly as before -- proof: if `_drain_to_final_transcript`
    passed a `finalize` keyword here, `FakeStt.stream` would raise
    `TypeError` and this test would fail."""
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    assert getattr(source, "speech_signals", None) is None

    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = fake_brain(replies=[BrainReply(text="done")])
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

    assert timings.turn_outcome == "completed"
    assert timings.vad_end_at is None
    assert tts.received_text == ["done"]


def test_speech_signals_forwarding_properties():
    """`PrerollReplayingSource`, `FollowUpSource`, and
    `ObserverPublishingSource` return the wrapped source's own
    `speech_signals`, or `None` when it has none."""
    from atlas.session.observers import ObserverPublishingSource, ObserverRegistry
    from atlas.sources.runner import FollowUpSource, PrerollReplayingSource

    class _WithSignals:
        def __init__(self) -> None:
            self.speech_signals = object()

    class _WithoutSignals:
        pass

    with_signals = _WithSignals()
    without_signals = _WithoutSignals()
    registry = ObserverRegistry()

    assert PrerollReplayingSource(with_signals, []).speech_signals is with_signals.speech_signals
    assert PrerollReplayingSource(without_signals, []).speech_signals is None

    assert FollowUpSource(with_signals, 0.0, lambda: 0.0).speech_signals is with_signals.speech_signals
    assert FollowUpSource(without_signals, 0.0, lambda: 0.0).speech_signals is None

    assert (
        ObserverPublishingSource(with_signals, "edge", registry).speech_signals
        is with_signals.speech_signals
    )
    assert ObserverPublishingSource(without_signals, "edge", registry).speech_signals is None


async def test_vad_end_before_any_heard_word_does_not_finalize(fake_brain, fake_tts):
    """The real-Pi failure: the wake cue's echo (or a noise blip) opened and
    closed a segment before the operator spoke. That `vad.end` must not
    finalize a turn speech-to-text has not heard a word of."""
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    signals = SpeechSignals(hangover_s=0.0)
    signals.publish({"type": "vad.start", "seq": 1})
    source = _SpeechSignalsSource(signals)
    stt = _FinalizeGatedStt(["turn on the fan"], partials=False)
    timings = TurnTimings()

    async def _blip_then_nothing() -> None:
        await asyncio.sleep(0.02)
        signals.publish({"type": "vad.end", "seq": 1})

    asyncio.ensure_future(_blip_then_nothing())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            run_turn(
                source,
                stt,
                fake_brain(replies=[BrainReply(text="x")]),
                fake_tts(chunks=[b"\x01\x02"]),
                None,
                tools_schema=[],
                system_prompt="you control a home",
                max_tool_rounds=3,
                timings=timings,
            ),
            timeout=0.5,
        )
    assert timings.vad_end_at is None
