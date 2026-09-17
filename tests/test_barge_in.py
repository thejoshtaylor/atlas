"""Real assertions for VOICE-07's barge-in gate: the energy measure alone
(Task 1). Plan 02-06's Task 2 extends this file with the detector's own
boundary semantics (the energy floor, sustained duration, and the
post-playback guard window).

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
