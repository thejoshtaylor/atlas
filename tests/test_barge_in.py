"""Real assertions for VOICE-07's barge-in gate: the energy measure alone
(Task 1), then `BargeInMonitor`'s own boundary semantics (Task 2) -- the
energy floor, sustained duration, and the post-playback guard window, all
driven by an injected `now` rather than a real clock or real asyncio.

`sources/runner.py`'s `_speak`-facing pieces (the interrupt actually landing
on the emission loop, and the recorded turn outcome) are covered by
`tests/test_turn_controller.py`'s `-k barge_in` cases instead, per the
validation map: those need `run_turn`'s own fakes (`FakeTts`,
`FakeAudioSource`), not this file's.
"""

from __future__ import annotations

import struct

import pytest

from spire_voice.audio.energy import rms_amplitude
from spire_voice.sources.runner import BargeInMonitor

# --- Task 1: rms_amplitude alone, no policy, no duration --------------------


def _tone(amplitude: int, n_samples: int) -> bytes:
    """`n_samples` of a constant-amplitude signed-16-bit signal, alternating
    sign every sample -- a full-scale square wave is the simplest fixed
    signal whose RMS is exactly `amplitude`, with no partial-cycle averaging
    error to account for."""
    samples = [amplitude if i % 2 == 0 else -amplitude for i in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)


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
