"""`audio/alaw.py`, `audio/probe.py`, and `audio/echo_path.py` (plan 02-10).

Every case here is checkable with no camera: A-law conversion round-trips
against its own 256-code table, the probe is deterministic and broadband by
an asserted property, and `measure_echo_path` is driven entirely by
synthetic fixtures with a delay and a gain the test itself injects. That the
real camera's echo path is measurable at all is plan 02-11's live check, not
this file's.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas.audio.alaw import AlawError, alaw_to_pcm16, bytes_per_sample, pcm16_to_alaw
from atlas.audio.echo_path import (
    AGC_ABSENT,
    AGC_INDETERMINATE,
    AGC_PRESENT,
    CONFIDENCE_USABLE_THRESHOLD,
    measure_echo_path,
)
from atlas.audio.probe import DEFAULT_PROBE_SEED, MIN_DURATION_S, ProbeError, build_probe

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


# ---------------------------------------------------------------------------
# echo_path.py (Task 2)
# ---------------------------------------------------------------------------


def _build_recording(
    reference: np.ndarray,
    *,
    delay_samples: int,
    scale: float = 1.0,
    total_length: int | None = None,
    noise_rms: float = 0.0,
    noise_seed: int = 1,
) -> bytes:
    """A synthetic recording: silence, then `reference` delayed and scaled,
    optionally with independent noise added everywhere -- exactly the shape
    a real echo-path recording has, with the delay and gain the test itself
    injected rather than measured."""
    if total_length is None:
        total_length = delay_samples + len(reference) + delay_samples
    recording = np.zeros(total_length, dtype=np.float64)
    end = min(delay_samples + len(reference), total_length)
    span = end - delay_samples
    if span > 0:
        recording[delay_samples:end] = reference[:span].astype(np.float64) * scale
    if noise_rms > 0:
        rng = np.random.default_rng(noise_seed)
        recording += rng.standard_normal(total_length) * noise_rms
    recording = np.clip(np.round(recording), -32768, 32767)
    return recording.astype(np.int16).tobytes()


@pytest.fixture()
def probe_reference() -> np.ndarray:
    _, pcm16_bytes = build_probe(SAMPLE_RATE, 2.0, seed=DEFAULT_PROBE_SEED)
    return _pcm16_array(pcm16_bytes).astype(np.float64)


def test_measure_echo_path_recovers_injected_delay_and_scale(probe_reference):
    delay_samples = 400
    scale = 0.6
    recorded = _build_recording(probe_reference, delay_samples=delay_samples, scale=scale)
    result = measure_echo_path(
        _pcm16_bytes(probe_reference), recorded, SAMPLE_RATE
    )
    assert result.failure_reason is None
    assert result.delay_s is not None
    expected_delay_s = delay_samples / SAMPLE_RATE
    assert abs(result.delay_s - expected_delay_s) <= 1.0 / SAMPLE_RATE
    assert result.gain is not None
    assert abs(result.gain - scale) < 0.05


def test_measure_echo_path_with_moderate_noise_still_recovers_delay(probe_reference):
    delay_samples = 250
    recorded = _build_recording(
        probe_reference, delay_samples=delay_samples, scale=0.8, noise_rms=300.0
    )
    result = measure_echo_path(_pcm16_bytes(probe_reference), recorded, SAMPLE_RATE)
    assert result.failure_reason is None
    assert result.delay_s is not None
    expected_delay_s = delay_samples / SAMPLE_RATE
    assert abs(result.delay_s - expected_delay_s) <= 2.0 / SAMPLE_RATE
    assert result.confidence >= CONFIDENCE_USABLE_THRESHOLD


def test_measure_echo_path_of_silence_reports_failure_with_no_delay(probe_reference):
    silence = np.zeros(len(probe_reference) + 1000, dtype=np.int16).tobytes()
    result = measure_echo_path(_pcm16_bytes(probe_reference), silence, SAMPLE_RATE)
    assert result.delay_s is None
    assert result.failure_reason is not None
    assert 0.0 <= result.confidence <= 1.0


def test_measure_echo_path_of_uncorrelated_noise_reports_failure(probe_reference):
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(len(probe_reference) + 1000) * 500).astype(np.int16).tobytes()
    result = measure_echo_path(_pcm16_bytes(probe_reference), noise, SAMPLE_RATE)
    assert result.delay_s is None
    assert result.failure_reason is not None
    assert result.confidence < CONFIDENCE_USABLE_THRESHOLD


def test_measure_echo_path_agc_absent_for_constant_level(probe_reference):
    delay_samples = 200
    recorded = _build_recording(probe_reference, delay_samples=delay_samples, scale=0.7)
    result = measure_echo_path(_pcm16_bytes(probe_reference), recorded, SAMPLE_RATE)
    assert result.failure_reason is None
    assert result.agc_verdict == AGC_ABSENT


def test_measure_echo_path_agc_present_for_monotonic_ramp(probe_reference):
    delay_samples = 200
    n = len(probe_reference)
    ramp = np.linspace(0.3, 1.4, n)
    ramped_reference = probe_reference * ramp
    recorded = _build_recording(ramped_reference, delay_samples=delay_samples, scale=1.0)
    result = measure_echo_path(_pcm16_bytes(probe_reference), recorded, SAMPLE_RATE)
    assert result.failure_reason is None
    assert result.agc_verdict == AGC_PRESENT


def test_measure_echo_path_agc_indeterminate_for_non_monotonic_level(probe_reference):
    delay_samples = 200
    n = len(probe_reference)
    # Rises then falls -- a real trend reversal, not noise -- so no
    # monotonic verdict can honestly be read from it.
    half = n // 2
    wobble = np.concatenate([
        np.linspace(0.4, 1.6, half),
        np.linspace(1.6, 0.3, n - half),
    ])
    wobbled_reference = probe_reference * wobble
    recorded = _build_recording(wobbled_reference, delay_samples=delay_samples, scale=1.0)
    result = measure_echo_path(_pcm16_bytes(probe_reference), recorded, SAMPLE_RATE)
    assert result.failure_reason is None
    assert result.agc_verdict == AGC_INDETERMINATE


def test_measure_echo_path_reduced_confidence_for_a_shortened_overlap(probe_reference):
    # The operator stopped the recording before the probe finished playing
    # -- a real calibration run's "I cut it off too early" mistake. The
    # aligned overlap is shorter than the full reference, and the delay is
    # still recoverable from what overlap exists, but with visibly reduced
    # confidence rather than the full-overlap recording's confidence.
    delay_samples = 200
    full_recorded = _build_recording(probe_reference, delay_samples=delay_samples, scale=0.8)
    full_result = measure_echo_path(_pcm16_bytes(probe_reference), full_recorded, SAMPLE_RATE)

    half_overlap = len(probe_reference) // 2
    shortened_recorded = _build_recording(
        probe_reference,
        delay_samples=delay_samples,
        scale=0.8,
        total_length=delay_samples + half_overlap,
    )
    partial_result = measure_echo_path(
        _pcm16_bytes(probe_reference), shortened_recorded, SAMPLE_RATE
    )

    assert full_result.failure_reason is None
    assert partial_result.failure_reason is None
    assert partial_result.delay_s == pytest.approx(full_result.delay_s, abs=1.0 / SAMPLE_RATE)
    assert partial_result.confidence < full_result.confidence
