"""Pure verdict functions for the five D-18 spike questions.

numpy only, no I/O, no hardware. Every function here is pure: same input,
same output, so `edge/tests/test_spike_analysis.py` can prove each verdict
against synthetic audio on the dev host, and `edge/spike/run_spike.py` can
call the same functions the tests already proved rather than improvising a
rule at run time (D-19).

Energy is measured in 10 ms frames. The noise floor is the median frame
energy of the first 300 ms of a buffer. An onset is the first frame at the
floor plus 12 dB that stays there for 3 frames. An offset is the last such
frame before a sustained (500 ms) drop back below that threshold.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

FRAME_MS = 10
_ONSET_THRESHOLD_DB = 12.0
_ONSET_RUN_FRAMES = 3
_SUSTAINED_SILENCE_MS = 500.0
_NOISE_FLOOR_WINDOW_MS = 300.0

PASS = "PASS"
FAIL = "FAIL"
NEEDS_LISTENING = "NEEDS_LISTENING"
PROVEN = "proven"
NOT_PROVEN = "not_proven"


def _frame_energies_db(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Split `samples` (mono) into 10 ms frames and return each frame's
    energy in dB (10*log10(mean square), with a floor to avoid log(0))."""
    frame_len = max(1, int(round(sample_rate * FRAME_MS / 1000.0)))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        raise ValueError("buffer shorter than one 10ms frame")
    trimmed = np.asarray(samples[: n_frames * frame_len], dtype=np.float64)
    frames = trimmed.reshape(n_frames, frame_len)
    power = np.mean(frames**2, axis=1) + 1e-12
    return 10.0 * np.log10(power)


def _noise_floor_db(energies_db: np.ndarray) -> float:
    n_floor_frames = max(1, int(round(_NOISE_FLOOR_WINDOW_MS / FRAME_MS)))
    n_floor_frames = min(n_floor_frames, len(energies_db))
    return float(np.median(energies_db[:n_floor_frames]))


def _merge_speech_regions(
    above: np.ndarray, min_silence_frames: int
) -> list[tuple[int, int]]:
    """Return (start, end_inclusive) frame-index pairs for merged speech
    regions: runs of frames at/above threshold, with silence gaps shorter
    than `min_silence_frames` merged into the surrounding region."""
    n = len(above)
    regions: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        start = i
        end = i
        j = i + 1
        while j < n:
            if above[j]:
                end = j
                j += 1
                continue
            k = j
            while k < n and not above[k]:
                k += 1
            gap_len = k - j
            if gap_len < min_silence_frames and k < n:
                j = k
                continue
            break
        regions.append((start, end))
        i = j
    return regions


def _speech_regions(samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    energies_db = _frame_energies_db(samples, sample_rate)
    floor = _noise_floor_db(energies_db)
    threshold = floor + _ONSET_THRESHOLD_DB
    above = energies_db >= threshold
    min_silence_frames = max(1, int(round(_SUSTAINED_SILENCE_MS / FRAME_MS)))
    regions = _merge_speech_regions(above, min_silence_frames)
    return [r for r in regions if (r[1] - r[0] + 1) >= _ONSET_RUN_FRAMES]


def energy_onset_ms(samples: np.ndarray, sample_rate: int) -> float:
    """The start, in ms, of the first sustained (>=3 frame) speech region
    at floor + 12 dB."""
    regions = _speech_regions(samples, sample_rate)
    if not regions:
        raise ValueError("no sustained onset found")
    return regions[0][0] * FRAME_MS


def energy_offset_ms(samples: np.ndarray, sample_rate: int) -> float:
    """The end, in ms, of the first sustained (>=3 frame) speech region,
    at the last frame before a sustained (500 ms) drop below threshold."""
    regions = _speech_regions(samples, sample_rate)
    if not regions:
        raise ValueError("no sustained offset found")
    return (regions[0][1] + 1) * FRAME_MS


def channel_snr_db(buffer: np.ndarray, sample_rate: int) -> tuple[float, ...]:
    """`buffer`: shape (n_samples, n_channels). Returns one SNR value per
    channel: that channel's speech-band energy (the median frame energy
    after the first 300 ms) minus that channel's own noise floor (the
    median frame energy of the first 300 ms)."""
    buffer = np.asarray(buffer)
    if buffer.ndim != 2:
        raise ValueError("buffer must be shape (n_samples, n_channels)")
    n_channels = buffer.shape[1]
    result = []
    for ch in range(n_channels):
        energies_db = _frame_energies_db(buffer[:, ch], sample_rate)
        floor = _noise_floor_db(energies_db)
        n_floor_frames = max(1, int(round(_NOISE_FLOOR_WINDOW_MS / FRAME_MS)))
        n_floor_frames = min(n_floor_frames, len(energies_db))
        speech = energies_db[n_floor_frames:]
        if len(speech) == 0:
            speech = energies_db
        speech_energy_db = float(np.median(speech))
        result.append(speech_energy_db - floor)
    return tuple(result)


def asr_channel_verdict(
    measurements: Sequence[tuple[float, float]], vendor_channel: int
) -> dict:
    """`measurements`: one `channel_snr_db` result (a 2-tuple) per measured
    position. `vendor_channel`: the channel index the vendor's routing
    claim (host_control/README.md) names as the ASR beam. PASS when the
    measured SNR ordering agrees with the vendor claim (margin >= 1 dB) at
    2 of 3 positions. FAIL when it contradicts (margin <= -1 dB) at 2 of 3
    positions. NEEDS_LISTENING otherwise -- including when margins sit
    under 1 dB either way."""
    other_channel = 1 - vendor_channel
    margins = [m[vendor_channel] - m[other_channel] for m in measurements]
    agree = sum(1 for m in margins if m >= 1.0)
    contradict = sum(1 for m in margins if m <= -1.0)
    if agree >= 2:
        verdict = PASS
    elif contradict >= 2:
        verdict = FAIL
    else:
        verdict = NEEDS_LISTENING
    return {
        "verdict": verdict,
        "vendor_channel": vendor_channel,
        "margins_db": margins,
    }


def _circular_mean_deg(angles_deg: Sequence[float]) -> float:
    radians = np.radians(np.asarray(angles_deg, dtype=np.float64))
    s = float(np.mean(np.sin(radians)))
    c = float(np.mean(np.cos(radians)))
    return float(np.degrees(math.atan2(s, c)) % 360.0)


def _circular_diff_deg(a: float, b: float) -> float:
    """Signed a - b, wrapped to [-180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def doa_verdict(positions: Sequence[tuple[float, Sequence[float]]]) -> dict:
    """`positions`: one (declared_angle_deg, [readings_deg...]) pair per
    measured position. PASS when every position's circular-mean reading is
    within 30 degrees of its declared angle after one constant offset
    (itself a circular mean of the per-position diffs) is removed. Handles
    the 359/1 degree wrap throughout."""
    diffs = []
    for declared, readings in positions:
        mean_reading = _circular_mean_deg(readings)
        diffs.append(_circular_diff_deg(mean_reading, declared))
    offset = _circular_mean_deg(diffs)
    adjusted = [_circular_diff_deg(d, offset) for d in diffs]
    verdict = PASS if all(abs(a) <= 30.0 for a in adjusted) else FAIL
    return {
        "verdict": verdict,
        "offset_deg": offset,
        "adjusted_diffs_deg": adjusted,
    }


def aec_verdict(playback_only_segments: int, doubletalk_segments: int) -> dict:
    """PASS ("proven") only when the playback-only run opened zero VAD
    segments and the double-talk run opened at least one. Otherwise
    "not_proven" -- a real AEC failure (playback alone triggers VAD) and an
    inconclusive run (double-talk triggers nothing) both land here, and the
    evidence numbers say which."""
    verdict = (
        PROVEN if playback_only_segments == 0 and doubletalk_segments >= 1 else NOT_PROVEN
    )
    return {
        "verdict": verdict,
        "playback_only_segments": playback_only_segments,
        "doubletalk_segments": doubletalk_segments,
    }


def multibeam_verdict(
    together_azimuths_deg: tuple[float, float, float, float],
    talker_angles_deg: tuple[float, float],
) -> dict:
    """D-18 Q4: does the multi-beam output give one stream per talker
    direction? `together_azimuths_deg` is one `AEC_AZIMUTH_VALUES` reading
    (focused beam 1, focused beam 2, free-running beam, auto-selected beam)
    taken while both talkers spoke at once. PASS when the two focused beams
    (indices 0 and 1) each land within 30 degrees of a distinct talker
    angle -- one beam per talker, not both beams converging on one."""
    beam1, beam2 = together_azimuths_deg[0], together_azimuths_deg[1]
    talker_a, talker_b = talker_angles_deg

    def _within(beam: float, talker: float) -> bool:
        return abs(_circular_diff_deg(beam, talker)) <= 30.0

    matches_a_b = _within(beam1, talker_a) and _within(beam2, talker_b)
    matches_b_a = _within(beam1, talker_b) and _within(beam2, talker_a)
    verdict = PASS if (matches_a_b or matches_b_a) else FAIL
    return {
        "verdict": verdict,
        "focused_beam_1_deg": beam1,
        "focused_beam_2_deg": beam2,
        "talker_angles_deg": list(talker_angles_deg),
    }


def derive_pre_roll_ms(onset_delays_ms: Sequence[float]) -> int:
    """The D-08 default `pre_roll_ms`: the worst (largest) measured onset
    delay, plus a 100 ms margin, rounded up to the nearest 50 ms."""
    worst = max(onset_delays_ms)
    return int(math.ceil((worst + 100.0) / 50.0) * 50)


def derive_tail_ms(offset_lags_ms: Sequence[float], endpointing_ms: float) -> int:
    """The D-08 default `tail_ms`: enough trailing silence that even the
    worst (smallest) measured offset lag still leaves `endpointing_ms` of
    silence for xAI's own endpointing fallback (D-13) to end the turn.
    Never below 0."""
    worst = min(offset_lags_ms)
    raw_frames = (endpointing_ms + 100.0 - worst) / 50.0
    frames = max(0, math.ceil(raw_frames))
    return int(frames * 50)
