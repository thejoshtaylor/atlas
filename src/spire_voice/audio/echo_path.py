"""One measurement: how long the assistant's own voice takes to return to
its own microphone, how loud it is when it does, and whether the camera is
quietly changing that level underneath.

Mirroring `audio/energy.py`'s own governing rule, stated the same way here:
this module measures and decides nothing about barge-in policy. It never
reads a clock, never touches the filesystem, never logs, and takes every
input as an explicit parameter, so `tests/test_echo_path.py` can drive
every boundary -- an injected delay, an injected gain, silence, noise --
with no camera and no hardware at all.

`measure_echo_path` is the one public entry point. It returns
`EchoPathMeasurement`, a frozen record carrying a confidence alongside
every number it reports, because a correlation peak indistinguishable from
the noise floor is a failed measurement with a named reason, never a delay
of zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spire_voice.audio.energy import rms_amplitude

# The AGC verdict has three values, not two. `absent` and `present` are the
# two outcomes a clean measurement can read from a trend; `indeterminate`
# is a third, equally real answer -- too few segments, too much variance,
# or a trend that reverses sign -- and plan 02-12 treats it as
# conservatively as `present`. It must never be collapsed into `absent` for
# tidiness: an answer that could not be established is not evidence the
# echo path is stable.
AGC_ABSENT = "absent"
AGC_PRESENT = "present"
AGC_INDETERMINATE = "indeterminate"

# A normalized cross-correlation peak below this is indistinguishable from
# the noise floor -- a threshold on the peak value returned by
# `_cross_correlate`, itself already normalized by the two signals' own
# energies into a `[-1.0, 1.0]`-bounded quantity (bounded above by 1.0 for
# non-negative signal pairs by Cauchy-Schwarz). Chosen with margin above
# what uncorrelated noise of moderate level produces and below what even a
# heavily attenuated, moderately noisy real echo produces (see
# 02-10-SUMMARY.md for the values this was checked against).
CONFIDENCE_USABLE_THRESHOLD = 0.25

# Segments the aligned overlap is sliced into to read an AGC trend. 200ms is
# short enough that a several-second probe yields enough segments to trend
# across, and long enough that each segment's RMS is a stable estimate
# rather than noise itself.
_AGC_SEGMENT_DURATION_S = 0.2

# Fewer segments than this cannot honestly support a trend verdict --
# reported as indeterminate rather than guessed at.
_AGC_MIN_SEGMENTS = 4

# A segment-to-segment level step within this fraction of the running level
# is noise, not a trend reversal -- absorbing ordinary measurement jitter
# so a flat recording does not read as "non-monotonic" by chance.
_AGC_STEP_NOISE_TOLERANCE = 0.05

# The total change from the first segment to the last must exceed this
# fraction of the first segment's level before a monotonic trend is called
# `present` rather than `absent` -- a small drift is not what "the camera
# applies gain control" means.
_AGC_TREND_THRESHOLD = 0.20

# A segment level at or below this normalized RMS is treated as having no
# reliable baseline to compute a relative trend against, guarding the
# absent/present classification's division rather than the correlation
# path above.
_MIN_SEGMENT_LEVEL = 1e-6


@dataclass(frozen=True)
class EchoPathMeasurement:
    """One measurement's full result. `confidence` is always present, even
    on failure -- it is the number that explains *why* a measurement
    failed. `delay_s`, `echo_level`, and `gain` are `None` exactly when
    `failure_reason` is not `None`: a measurement that could not be made
    reports that it found nothing, never a number that looks measured and
    is not.
    """

    delay_s: float | None
    confidence: float
    echo_level: float | None
    gain: float | None
    agc_verdict: str
    segment_levels: tuple[float, ...]
    failure_reason: str | None
    # `True` exactly for the "no correlation peak" failure -- never for the
    # "reference or recording is empty" failure, and never on success. The
    # discriminator a caller checks instead of matching `failure_reason`'s
    # text (`calibration/runner.py`): a peak this low means the reference
    # was not found in the recording at all, which is what a camera that
    # cancels its own echo would produce, but it is equally what a dead
    # microphone recording nothing but silence would produce -- this field
    # says only "no correlation was found," not "the camera cancelled its
    # echo." Telling those apart needs the recording's own energy too,
    # which this module has no reason to read (module docstring).
    no_echo: bool


def _cross_correlate(reference: np.ndarray, recorded: np.ndarray) -> tuple[int | None, float]:
    """The lag, in samples, at which `recorded` best matches `reference`
    delayed by that lag, and a confidence in `[0.0, 1.0]` for non-negative
    signal pairs: the raw correlation peak normalized by the product of
    the two signals' own L2 energies, via `numpy.fft.rfft`/`irfft` -- the
    only path that stays fast over a probe of several seconds at 8 kHz.

    Returns `(None, 0.0)` if either signal carries no energy at all --
    there is no lag to report a peak at.
    """
    ref_energy = float(np.sqrt(np.sum(reference * reference)))
    rec_energy = float(np.sqrt(np.sum(recorded * recorded)))
    if ref_energy == 0.0 or rec_energy == 0.0:
        return None, 0.0

    n_ref = len(reference)
    n_rec = len(recorded)
    nfft = n_ref + n_rec
    rec_spectrum = np.fft.rfft(recorded, n=nfft)
    ref_spectrum = np.fft.rfft(reference, n=nfft)
    # sum_i recorded[l + i] * reference[i], for lag l -- linear
    # cross-correlation, read out of the circular result before it would
    # wrap (guaranteed by nfft = n_ref + n_rec).
    corr = np.fft.irfft(rec_spectrum * np.conj(ref_spectrum), n=nfft)
    valid = corr[:n_rec]
    normalized = valid / (ref_energy * rec_energy)
    peak_idx = int(np.argmax(normalized))
    return peak_idx, float(normalized[peak_idx])


def _segment_levels(overlap: np.ndarray, sample_rate: int) -> list[float]:
    """RMS level (`audio/energy.py`'s unit) of each fixed-duration segment
    across the aligned overlap, oldest first."""
    segment_len = max(int(round(_AGC_SEGMENT_DURATION_S * sample_rate)), 1)
    levels: list[float] = []
    for start in range(0, len(overlap), segment_len):
        segment = overlap[start : start + segment_len]
        if len(segment) < segment_len:
            break  # a short trailing remainder is not a full segment to trend against
        segment_bytes = np.clip(np.round(segment), -32768, 32767).astype(np.int16).tobytes()
        levels.append(rms_amplitude(segment_bytes))
    return levels


def _classify_agc(levels: list[float]) -> str:
    """`absent` for a flat trend, `present` for a monotonic trend past the
    stated threshold, `indeterminate` for everything else: too few
    segments, a trend that changes sign, or variance too large to call
    monotonic at all. `indeterminate` is a real, reachable answer -- see
    the module docstring's AGC paragraph."""
    if len(levels) < _AGC_MIN_SEGMENTS:
        return AGC_INDETERMINATE

    diffs = np.diff(np.asarray(levels, dtype=np.float64))
    running_scale = np.maximum(np.abs(np.asarray(levels[:-1], dtype=np.float64)), _MIN_SEGMENT_LEVEL)
    tolerance = running_scale * _AGC_STEP_NOISE_TOLERANCE

    non_decreasing = bool(np.all(diffs >= -tolerance))
    non_increasing = bool(np.all(diffs <= tolerance))
    if not (non_decreasing or non_increasing):
        return AGC_INDETERMINATE

    baseline = max(levels[0], _MIN_SEGMENT_LEVEL)
    relative_change = abs(levels[-1] - levels[0]) / baseline
    if relative_change > _AGC_TREND_THRESHOLD:
        return AGC_PRESENT
    return AGC_ABSENT


def measure_echo_path(
    reference_pcm16: bytes, recorded_pcm16: bytes, sample_rate: int
) -> EchoPathMeasurement:
    """Delay, level, gain, and an AGC verdict, all from one recording of
    `reference_pcm16` played back and re-recorded as `recorded_pcm16`.

    A recording with no usable correlation peak -- silence, or noise
    uncorrelated with the reference -- returns a failed measurement: no
    delay, no level, no gain, and a named reason. The AGC question is
    answered only over the aligned overlap of a *successful* measurement;
    there is no honest trend to read from a recording the reference was
    never found in, so a failure reports `AGC_INDETERMINATE` rather than
    guessing.
    """
    reference = np.frombuffer(reference_pcm16, dtype="<i2").astype(np.float64)
    recorded = np.frombuffer(recorded_pcm16, dtype="<i2").astype(np.float64)

    if len(reference) == 0 or len(recorded) == 0:
        return EchoPathMeasurement(
            delay_s=None,
            confidence=0.0,
            echo_level=None,
            gain=None,
            agc_verdict=AGC_INDETERMINATE,
            segment_levels=(),
            failure_reason="reference or recording is empty",
            no_echo=False,
        )

    peak_idx, confidence = _cross_correlate(reference, recorded)
    confidence = min(max(confidence, 0.0), 1.0)

    if peak_idx is None or confidence < CONFIDENCE_USABLE_THRESHOLD:
        return EchoPathMeasurement(
            delay_s=None,
            confidence=confidence,
            echo_level=None,
            gain=None,
            agc_verdict=AGC_INDETERMINATE,
            segment_levels=(),
            failure_reason=(
                "no correlation peak above the usable confidence threshold "
                f"({CONFIDENCE_USABLE_THRESHOLD}) -- the reference was not found in the recording"
            ),
            no_echo=True,
        )

    overlap_len = min(len(reference), len(recorded) - peak_idx)
    overlap_recorded = recorded[peak_idx : peak_idx + overlap_len]
    overlap_reference = reference[:overlap_len]

    echo_level = rms_amplitude(
        np.clip(np.round(overlap_recorded), -32768, 32767).astype(np.int16).tobytes()
    )
    reference_level = rms_amplitude(
        np.clip(np.round(overlap_reference), -32768, 32767).astype(np.int16).tobytes()
    )
    gain = (echo_level / reference_level) if reference_level > 0 else None

    segment_levels = _segment_levels(overlap_recorded, sample_rate)
    agc_verdict = _classify_agc(segment_levels)

    return EchoPathMeasurement(
        delay_s=peak_idx / sample_rate,
        confidence=confidence,
        echo_level=echo_level,
        gain=gain,
        agc_verdict=agc_verdict,
        segment_levels=tuple(segment_levels),
        failure_reason=None,
        no_echo=False,
    )
