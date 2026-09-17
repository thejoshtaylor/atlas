"""`audio/alaw.py` and `audio/probe.py` (plan 02-10, Task 1).

A-law conversion round-trips against its own 256-code table, and the probe
is deterministic and broadband by an asserted property rather than by
assertion in a comment -- a pure sine of equal duration is asserted to fail
the same single-peak property, which is the executable reason the probe is
broadband at all. Task 2 appends `measure_echo_path`'s own cases to this
same file.
"""

from __future__ import annotations

import numpy as np
import pytest

from spire_voice.audio.alaw import AlawError, alaw_to_pcm16, bytes_per_sample, pcm16_to_alaw
from spire_voice.audio.probe import DEFAULT_PROBE_SEED, MIN_DURATION_S, ProbeError, build_probe

SAMPLE_RATE = 8000


def _pcm16_bytes(samples: np.ndarray) -> bytes:
    return samples.astype(np.int16).tobytes()


def _pcm16_array(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.int64)


# ---------------------------------------------------------------------------
# alaw.py
# ---------------------------------------------------------------------------


def test_alaw_round_trips_all_256_codes_exactly():
    codes = bytes(range(256))
    pcm = alaw_to_pcm16(codes)
    back = pcm16_to_alaw(pcm)
    assert back == codes


def test_alaw_round_trip_of_arbitrary_pcm16_stays_within_one_quantization_step():
    rng = np.random.default_rng(7)
    samples = rng.integers(-32768, 32767, size=2000, dtype=np.int16)
    buf = samples.tobytes()
    decoded = _pcm16_array(alaw_to_pcm16(pcm16_to_alaw(buf)))
    original = samples.astype(np.int64)
    error = np.abs(decoded - original)
    # The A-law table's coarsest adjacent-codeword gap is 1024 (companded
    # encoding's widest step, near full scale); nearest-value quantization
    # error is at most half of the local step, so 512 with headroom bounds
    # every sample regardless of where in the range it falls.
    assert error.max() <= 600


def test_alaw_round_trip_error_grows_with_amplitude():
    rng = np.random.default_rng(11)
    quiet = (rng.standard_normal(2000) * 200).astype(np.int16)
    loud = (rng.standard_normal(2000) * 20000).astype(np.int16)
    quiet_decoded = _pcm16_array(alaw_to_pcm16(pcm16_to_alaw(quiet.tobytes())))
    loud_decoded = _pcm16_array(alaw_to_pcm16(pcm16_to_alaw(loud.tobytes())))
    quiet_error = np.abs(quiet_decoded - quiet.astype(np.int64)).mean()
    loud_error = np.abs(loud_decoded - loud.astype(np.int64)).mean()
    assert loud_error > quiet_error


def test_alaw_to_pcm16_of_empty_bytes_is_empty():
    assert alaw_to_pcm16(b"") == b""


def test_pcm16_to_alaw_of_empty_bytes_is_empty():
    assert pcm16_to_alaw(b"") == b""


def test_pcm16_to_alaw_rejects_odd_length_buffer_by_name():
    with pytest.raises(AlawError):
        pcm16_to_alaw(b"\x01")


def test_bytes_per_sample_for_the_two_declared_encodings():
    assert bytes_per_sample("pcm") == 2
    assert bytes_per_sample("alaw") == 1


def test_bytes_per_sample_rejects_unknown_encoding_by_name():
    with pytest.raises(AlawError):
        bytes_per_sample("mp3")


# ---------------------------------------------------------------------------
# probe.py
# ---------------------------------------------------------------------------


def test_build_probe_is_deterministic_for_a_fixed_seed():
    alaw1, pcm1 = build_probe(SAMPLE_RATE, 2.0, seed=42)
    alaw2, pcm2 = build_probe(SAMPLE_RATE, 2.0, seed=42)
    assert alaw1 == alaw2
    assert pcm1 == pcm2


def test_build_probe_default_seed_is_stable_and_named():
    # DEFAULT_PROBE_SEED is what a real calibration run reaches for, so two
    # calibrations weeks apart are comparable -- it must exist and behave
    # like any other seed value, not as a special case.
    alaw1, pcm1 = build_probe(SAMPLE_RATE, 1.5, seed=DEFAULT_PROBE_SEED)
    alaw2, pcm2 = build_probe(SAMPLE_RATE, 1.5, seed=DEFAULT_PROBE_SEED)
    assert alaw1 == alaw2
    assert pcm1 == pcm2


def test_build_probes_alaw_and_pcm16_describe_the_same_signal():
    alaw_bytes, pcm16_bytes = build_probe(SAMPLE_RATE, 2.0, seed=99)
    decoded = _pcm16_array(alaw_to_pcm16(alaw_bytes))
    reference = _pcm16_array(pcm16_bytes)
    assert len(decoded) == len(reference)
    error = np.abs(decoded - reference)
    assert error.max() <= 600


def _normalized_autocorrelation(samples: np.ndarray) -> np.ndarray:
    """Full symmetric-lag autocorrelation, normalized so zero lag is 1.0."""
    x = samples.astype(np.float64)
    n = len(x)
    spectrum = np.fft.rfft(x, n=2 * n)
    ac_full = np.fft.irfft(spectrum * np.conj(spectrum), n=2 * n)
    ac = np.concatenate([ac_full[-(n - 1):], ac_full[:n]])
    return ac / ac[n - 1]


def test_probe_autocorrelation_has_a_single_dominant_peak_near_zero_lag():
    _, pcm16_bytes = build_probe(SAMPLE_RATE, 2.0, seed=5)
    samples = _pcm16_array(pcm16_bytes).astype(np.float64)
    ac = _normalized_autocorrelation(samples)
    center = len(ac) // 2
    window = 5
    outside = np.concatenate([ac[: center - window], ac[center + window + 1 :]])
    assert np.max(np.abs(outside)) < 0.5


def test_a_pure_sine_of_equal_duration_fails_the_single_peak_property():
    # This is the executable reason the probe is broadband rather than a
    # tone: a periodic signal's autocorrelation is periodic too, so this
    # same assertion applied to a sine must fail.
    duration_s = 2.0
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    sine = np.sin(2 * np.pi * 1000.0 * t) * 10000.0
    ac = _normalized_autocorrelation(sine)
    center = len(ac) // 2
    window = 5
    outside = np.concatenate([ac[: center - window], ac[center + window + 1 :]])
    assert np.max(np.abs(outside)) >= 0.5


def test_build_probe_below_minimum_duration_raises_named_error():
    with pytest.raises(ProbeError):
        build_probe(SAMPLE_RATE, MIN_DURATION_S - 0.5, seed=1)
