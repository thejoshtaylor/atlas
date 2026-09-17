"""Real assertions for VOICE-07's barge-in gate: the energy measure alone
(Task 1), then `BargeInMonitor`'s own boundary semantics (Task 2) -- the
energy floor, sustained duration, and the post-playback guard window, all
driven by an injected `now` rather than a real clock or real asyncio.

`sources/runner.py`'s `_speak`-facing pieces (the interrupt actually landing
on the emission loop, and the recorded turn outcome) are covered by
`tests/test_turn_controller.py`'s `-k barge_in` cases instead, per the
validation map: those need `run_turn`'s own fakes (`FakeTts`,
`FakeAudioSource`), not this file's.

Plan 02-12 Task 1 adds one more section at the bottom: `EmittedAudioTrace`
alone, indexed by playback offset and never by a clock, plus the tap that
appends to it from `turn/controller.py`'s `_speak`. Task 2 adds the
correlation gate built on top of it in a later commit.
"""

from __future__ import annotations

import struct

import pytest

from spire_voice.audio.alaw import pcm16_to_alaw
from spire_voice.audio.energy import rms_amplitude
from spire_voice.speaker.output_trace import EmittedAudioTrace
from spire_voice.sources.runner import BargeInMonitor

# --- Task 1: rms_amplitude alone, no policy, no duration --------------------


def _tone(amplitude: int, n_samples: int) -> bytes:
    """`n_samples` of a constant-amplitude signed-16-bit signal, alternating
    sign every sample -- a full-scale square wave is the simplest fixed
    signal whose RMS is exactly `amplitude`, with no partial-cycle averaging
    error to account for."""
    samples = [amplitude if i % 2 == 0 else -amplitude for i in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)


def _level_chunk(level: float, *, duration_s: float, sample_rate: int) -> bytes:
    """`duration_s` seconds of PCM16 at `sample_rate`, built the same
    alternating-sign way `_tone` is, whose measured `rms_amplitude` is
    exactly `level` -- a fraction of full scale, `energy.py`'s own unit.
    Plan 02-12's correlation tests build known-level chunks against this
    rather than reasoning backward from an arbitrary peak amplitude."""
    n_samples = max(1, round(duration_s * sample_rate))
    amplitude = round(level * 32768.0)
    return _tone(amplitude, n_samples)


def test_silence_measures_at_zero():
    assert rms_amplitude(_tone(0, 64)) == 0.0


def test_empty_chunk_measures_at_zero():
    assert rms_amplitude(b"") == 0.0


def test_full_scale_tone_measures_near_the_maximum():
    # 32767, not 32768: signed 16-bit's positive bound is one less than its
    # magnitude-defining full scale (`_INT16_FULL_SCALE` in energy.py).
    measured = rms_amplitude(_tone(32767, 256))
    assert 0.999 <= measured <= 1.0


def test_measure_is_independent_of_chunk_length():
    """A constant signal measures the same whether handed one long chunk or
    several short ones -- accumulating duration later must not secretly be
    accumulating amplitude too."""
    short = rms_amplitude(_tone(10000, 32))
    long = rms_amplitude(_tone(10000, 4096))
    assert short == long


def test_measure_is_independent_of_sample_rate():
    """`rms_amplitude` takes no rate parameter at all (module docstring):
    the same waveform, resampled to a different rate, is still the same
    number of *cycles* just spread across a different sample count. This
    proves rate never entered the calculation -- there is nothing here for
    it to have implicitly assumed."""
    at_8khz_equivalent = rms_amplitude(_tone(5000, 80))  # 10ms at 8kHz
    at_16khz_equivalent = rms_amplitude(_tone(5000, 160))  # 10ms at 16kHz
    assert at_8khz_equivalent == at_16khz_equivalent


def test_rejects_a_sample_width_other_than_16_bit():
    with pytest.raises(ValueError):
        rms_amplitude(_tone(100, 8), sample_width=1)


# --- Task 2: BargeInMonitor's boundary semantics ----------------------------

# Chosen to be exactly representable in binary floating point (halves and
# quarters), so a boundary test comparing `now` values built by simple
# addition never fails on float-rounding noise unrelated to what it proves.
_FLOOR = 0.5
_MIN_DURATION_S = 1.0
_GUARD_S = 0.5


def _monitor(enabled: bool = True) -> BargeInMonitor:
    return BargeInMonitor(
        floor=_FLOOR,
        min_duration_s=_MIN_DURATION_S,
        guard_window_s=_GUARD_S,
        enabled=enabled,
    )


def test_no_interrupt_before_any_playback_started():
    """Nothing is playing yet -- there is nothing to interrupt, however
    loud the room is."""
    monitor = _monitor()
    monitor.process_energy(1.0, now=100.0)
    assert monitor.interrupt_requested is False


def test_energy_exactly_at_the_floor_does_not_interrupt():
    monitor = _monitor()
    monitor.mark_playback_started(0.0)
    # Held at exactly the floor for well past the minimum duration -- still
    # never counts, because `<=` the floor is not "above" it.
    for step in range(8):
        monitor.process_energy(_FLOOR, now=_GUARD_S + step * 0.25)
    assert monitor.interrupt_requested is False


def test_energy_above_the_floor_for_exactly_the_minimum_duration_interrupts():
    monitor = _monitor()
    monitor.mark_playback_started(0.0)
    monitor.process_energy(_FLOOR + 0.25, now=_GUARD_S)
    monitor.process_energy(_FLOOR + 0.25, now=_GUARD_S + _MIN_DURATION_S)
    assert monitor.interrupt_requested is True


def test_energy_above_the_floor_for_one_step_less_does_not_interrupt():
    monitor = _monitor()
    monitor.mark_playback_started(0.0)
    monitor.process_energy(_FLOOR + 0.25, now=_GUARD_S)
    monitor.process_energy(_FLOOR + 0.25, now=_GUARD_S + _MIN_DURATION_S - 0.25)
    assert monitor.interrupt_requested is False


def test_energy_inside_the_guard_window_never_interrupts_however_loud():
    monitor = _monitor()
    monitor.mark_playback_started(0.0)
    for step in range(4):
        monitor.process_energy(1.0, now=step * (_GUARD_S / 4))  # as loud as a 16-bit signal gets
    assert monitor.interrupt_requested is False


def test_energy_exactly_at_the_end_of_the_guard_window_is_eligible():
    """At `now == guard_window_s` the guard no longer applies -- a run that
    starts exactly there and holds for the minimum duration interrupts."""
    monitor = _monitor()
    monitor.mark_playback_started(0.0)
    monitor.process_energy(_FLOOR + 0.25, now=_GUARD_S)
    monitor.process_energy(_FLOOR + 0.25, now=_GUARD_S + _MIN_DURATION_S)
    assert monitor.interrupt_requested is True


def test_an_above_floor_run_broken_by_one_at_or_below_frame_resets_the_duration():
    monitor = _monitor()
    monitor.mark_playback_started(0.0)
    monitor.process_energy(_FLOOR + 0.25, now=_GUARD_S)
    # A single at-floor frame partway through the run resets accumulation --
    # a door closing must not interrupt a reply.
    monitor.process_energy(_FLOOR, now=_GUARD_S + _MIN_DURATION_S - 0.25)
    monitor.process_energy(_FLOOR + 0.25, now=_GUARD_S + _MIN_DURATION_S)
    assert monitor.interrupt_requested is False
    # Held for the full duration measured from the *first* above-floor
    # reading after the reset (`_GUARD_S + _MIN_DURATION_S`, the previous
    # call), it does.
    monitor.process_energy(
        _FLOOR + 0.25, now=(_GUARD_S + _MIN_DURATION_S) + _MIN_DURATION_S
    )
    assert monitor.interrupt_requested is True


def test_a_disabled_policy_never_signals_an_interrupt():
    monitor = _monitor(enabled=False)
    monitor.mark_playback_started(0.0)
    now = _GUARD_S
    while now < _GUARD_S + _MIN_DURATION_S + 1.0:
        monitor.process_energy(1.0, now=now)
        now += 0.05
    assert monitor.interrupt_requested is False


def test_a_fresh_utterance_restarts_the_guard_window():
    """`mark_playback_started` is called again for the answer after the
    filler -- CONTEXT.md's guard window is anchored to *this* utterance
    starting, not to the turn as a whole."""
    monitor = _monitor()
    monitor.mark_playback_started(0.0)
    monitor.process_energy(_FLOOR + 0.25, now=_GUARD_S)
    # A new utterance starts right as the first one's guard window would
    # have ended -- the new guard window still applies from this point.
    monitor.mark_playback_started(_GUARD_S)
    monitor.process_energy(1.0, now=_GUARD_S + _GUARD_S / 2)
    assert monitor.interrupt_requested is False


def test_a_source_whose_policy_disables_barge_in_never_signals_an_interrupt():
    """Distinct from `_monitor(enabled=False)` above: this constructs a
    `BargeInMonitor` the way `SourceRunner` actually does, from a resolved
    per-source `BargeInConfig` (D-12) whose `enabled` field is `False`."""
    from spire_voice.config import BargeInConfig

    resolved = BargeInConfig(enabled=False).resolve("camera")
    monitor = BargeInMonitor(
        floor=resolved.energy_floor,
        min_duration_s=resolved.min_duration_ms / 1000.0,
        guard_window_s=resolved.post_playback_guard_ms / 1000.0,
        enabled=resolved.enabled,
    )
    monitor.mark_playback_started(0.0)
    monitor.process_energy(1.0, now=10.0)
    assert monitor.interrupt_requested is False


# --- Plan 02-12 Task 1: EmittedAudioTrace, indexed by playback offset -------


def test_trace_offsets_accumulate_from_byte_count_and_encoding_never_a_clock():
    trace = EmittedAudioTrace(encoding="pcm", sample_rate=1000)
    trace.append(_tone(1000, 100))  # 0.1s
    trace.append(_tone(1000, 200))  # 0.2s
    trace.append(_tone(1000, 50))  # 0.05s

    # Each chunk's offset is the sum of its predecessors' durations, derived
    # purely from sample count / sample_rate -- there is no clock anywhere
    # in `append` for this assertion to accidentally be exercising instead.
    assert trace.level_at(0.05) is not None  # inside chunk 1's span [0.0, 0.1)
    assert trace.level_at(0.2) is not None  # inside chunk 2's span [0.1, 0.3)
    assert trace.level_at(0.32) is not None  # inside chunk 3's span [0.3, 0.35)
    assert trace.level_at(0.5) is None  # well past the end -- nothing written there


def test_level_at_the_three_cases_containing_before_and_past_the_end():
    trace = EmittedAudioTrace(encoding="pcm", sample_rate=1000)
    chunk = _tone(5000, 100)
    trace.append(chunk)

    contained = trace.level_at(0.05)
    assert contained is not None
    assert contained == pytest.approx(rms_amplitude(chunk))
    assert trace.level_at(-0.01) is None  # before the first entry
    assert trace.level_at(0.1) is None  # past the end of what has been written


def test_alaw_and_pcm16_renderings_of_the_same_audio_measure_the_same_level():
    pcm = _tone(8000, 200)
    alaw = pcm16_to_alaw(pcm)

    pcm_trace = EmittedAudioTrace(encoding="pcm", sample_rate=1000)
    pcm_trace.append(pcm)
    alaw_trace = EmittedAudioTrace(encoding="alaw", sample_rate=1000)
    alaw_trace.append(alaw)

    pcm_level = pcm_trace.level_at(0.0)
    alaw_level = alaw_trace.level_at(0.0)
    assert pcm_level is not None
    assert alaw_level is not None
    # A-law reconstructs from 256 levels, not bit-exact -- "within the
    # conversion's own tolerance" (acceptance criteria), not equality.
    assert alaw_level == pytest.approx(pcm_level, abs=0.02)


def test_trace_is_bounded_and_stops_growing_past_its_configured_span():
    trace = EmittedAudioTrace(encoding="pcm", sample_rate=1000, span_s=0.5)
    for _ in range(1000):
        trace.append(_tone(1000, 10))  # 0.01s each -- 1000 of them is 10s
    assert len(trace) < 1000


def test_a_new_utterance_resets_the_trace():
    trace = EmittedAudioTrace(encoding="pcm", sample_rate=1000)
    trace.append(_tone(1000, 100))
    trace.append(_tone(1000, 100))
    assert len(trace) == 2

    trace.reset()

    assert len(trace) == 0
    assert trace.level_at(0.0) is None
    # The next utterance's own first chunk starts at offset zero again --
    # never continuing the previous utterance's cursor.
    trace.append(_tone(2000, 100))
    assert trace.level_at(0.05) is not None
    assert trace.level_at(0.2) is None


class _TraceTappingBargeIn:
    """The minimal `_speak`-facing shape Task 1 needs to prove the tap: an
    always-enabled, never-interrupting monitor carrying a real `trace` --
    interrupting mid-emission is `test_turn_controller.py`'s
    `_FakeBargeIn`'s own job, not this one's."""

    def __init__(self, trace: EmittedAudioTrace) -> None:
        self.enabled = True
        self.interrupt_requested = False
        self.trace = trace
        self.playback_started_at: float | None = None

    def mark_playback_started(self, now: float) -> None:
        self.playback_started_at = now


async def test_speak_with_no_monitor_is_byte_for_byte_unchanged(fake_audio_source, fake_tts):
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import _speak

    chunks = [b"1", b"2", b"3"]
    source = fake_audio_source()
    timings = TurnTimings()

    await _speak(source, fake_tts(chunks=chunks), timings, "reply", kind="answer")

    assert source.sent_audio == chunks


async def test_speak_with_a_monitor_appends_every_written_chunk_to_its_trace(fake_audio_source, fake_tts):
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import _speak

    chunks = [_tone(500, 80), _tone(500, 80), _tone(500, 80)]
    source = fake_audio_source()
    trace = EmittedAudioTrace(encoding="pcm", sample_rate=8000)
    barge_in = _TraceTappingBargeIn(trace)
    timings = TurnTimings()

    await _speak(source, fake_tts(chunks=chunks), timings, "reply", kind="answer", barge_in=barge_in)

    assert source.sent_audio == chunks
    assert len(trace) == 3


async def test_speak_appends_nothing_to_the_trace_after_an_interrupt_is_requested(fake_audio_source, fake_tts):
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import _speak

    class _InterruptingBargeIn:
        """Latches `interrupt_requested` true on its third read (one read
        per chunk, matching `test_turn_controller.py`'s own `_FakeBargeIn`
        convention) -- proving the trace stops exactly where emission does,
        not merely that emission stops."""

        def __init__(self, trace: EmittedAudioTrace) -> None:
            self.enabled = True
            self.trace = trace
            self._checks = 0

        @property
        def interrupt_requested(self) -> bool:
            self._checks += 1
            return self._checks > 2

        def mark_playback_started(self, now: float) -> None:
            pass

    chunks = [_tone(500, 80) for _ in range(5)]
    source = fake_audio_source()
    trace = EmittedAudioTrace(encoding="pcm", sample_rate=8000)
    barge_in = _InterruptingBargeIn(trace)
    timings = TurnTimings()

    await _speak(source, fake_tts(chunks=chunks), timings, "reply", kind="answer", barge_in=barge_in)

    assert source.sent_audio == chunks[:2]
    assert len(trace) == 2
