"""Proves the D-18 verdict functions against synthetic audio -- no real
XVF3800 or microphone anywhere in this file."""

from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import (
    FAIL,
    NEEDS_LISTENING,
    NOT_PROVEN,
    PASS,
    PROVEN,
    aec_verdict,
    asr_channel_verdict,
    channel_snr_db,
    derive_pre_roll_ms,
    derive_tail_ms,
    doa_verdict,
    energy_offset_ms,
    energy_onset_ms,
    multibeam_verdict,
)

SAMPLE_RATE = 16000


def _noise(duration_ms: float, rng: np.random.Generator, amplitude: float = 50.0) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    return rng.normal(0.0, amplitude, n)


def _tone(duration_ms: float, freq_hz: float, amplitude: float) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.arange(n) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


class TestOnsetOffset:
    def test_onset_within_one_frame_of_tone_start(self) -> None:
        rng = np.random.default_rng(42)
        samples = np.concatenate(
            [_noise(300, rng), _tone(500, 1000.0, 1000.0), _noise(600, rng)]
        )
        onset = energy_onset_ms(samples, SAMPLE_RATE)
        assert abs(onset - 300.0) <= 10.0

    def test_offset_within_one_frame_of_tone_end(self) -> None:
        rng = np.random.default_rng(42)
        samples = np.concatenate(
            [_noise(300, rng), _tone(500, 1000.0, 1000.0), _noise(600, rng)]
        )
        offset = energy_offset_ms(samples, SAMPLE_RATE)
        assert abs(offset - 800.0) <= 10.0


class TestChannelSnr:
    def test_channel_1_reports_about_12db_higher(self) -> None:
        rng = np.random.default_rng(7)
        ch0 = np.concatenate([_noise(300, rng), _tone(700, 1000.0, 400.0)])
        scale = 10 ** (12.0 / 20.0)  # +12 dB in amplitude terms
        ch1 = np.concatenate([_noise(300, rng), _tone(700, 1000.0, 400.0 * scale)])
        buffer = np.stack([ch0, ch1], axis=1)

        snr0, snr1 = channel_snr_db(buffer, SAMPLE_RATE)

        assert abs((snr1 - snr0) - 12.0) < 2.0


class TestAsrChannelVerdict:
    def test_pass_when_two_of_three_positions_agree(self) -> None:
        measurements = [(5.0, 17.0), (6.0, 18.0), (10.0, 9.0)]
        result = asr_channel_verdict(measurements, vendor_channel=1)
        assert result["verdict"] == PASS
        assert result["vendor_channel"] == 1

    def test_needs_listening_when_margin_under_1db(self) -> None:
        measurements = [(10.0, 10.5), (10.0, 10.3), (10.0, 9.8)]
        result = asr_channel_verdict(measurements, vendor_channel=1)
        assert result["verdict"] == NEEDS_LISTENING

    def test_fail_when_contradicted_at_two_of_three_positions(self) -> None:
        measurements = [(17.0, 5.0), (18.0, 6.0), (9.0, 10.0)]
        result = asr_channel_verdict(measurements, vendor_channel=1)
        assert result["verdict"] == FAIL


class TestDoaVerdict:
    def test_pass_within_30_degrees_after_offset_removed(self) -> None:
        positions = [
            (0.0, [10.0, 12.0, 8.0]),
            (90.0, [100.0, 98.0, 102.0]),
            (180.0, [190.0, 188.0, 192.0]),
        ]
        result = doa_verdict(positions)
        assert result["verdict"] == PASS
        assert abs(result["offset_deg"] - 10.0) < 1.0

    def test_handles_the_359_1_degree_wrap(self) -> None:
        positions = [
            (359.0, [4.0, 3.0, 5.0]),
            (1.0, [6.0, 5.0, 7.0]),
        ]
        result = doa_verdict(positions)
        assert result["verdict"] == PASS

    def test_fail_when_beyond_30_degrees_after_offset_removed(self) -> None:
        positions = [
            (0.0, [10.0]),
            (90.0, [95.0]),
            (180.0, [270.0]),
        ]
        result = doa_verdict(positions)
        assert result["verdict"] == FAIL


class TestAecVerdict:
    def test_proven_when_playback_only_is_silent_and_doubletalk_opens(self) -> None:
        result = aec_verdict(playback_only_segments=0, doubletalk_segments=1)
        assert result["verdict"] == PROVEN

    def test_not_proven_when_playback_alone_triggers_vad(self) -> None:
        result = aec_verdict(playback_only_segments=2, doubletalk_segments=1)
        assert result["verdict"] == NOT_PROVEN

    def test_not_proven_when_doubletalk_triggers_nothing(self) -> None:
        result = aec_verdict(playback_only_segments=0, doubletalk_segments=0)
        assert result["verdict"] == NOT_PROVEN


class TestMultibeamVerdict:
    def test_pass_when_focused_beams_split_the_two_talkers(self) -> None:
        result = multibeam_verdict(
            together_azimuths_deg=(45.0, 200.0, 100.0, 45.0),
            talker_angles_deg=(50.0, 195.0),
        )
        assert result["verdict"] == PASS

    def test_fail_when_focused_beams_converge_on_one_talker(self) -> None:
        result = multibeam_verdict(
            together_azimuths_deg=(50.0, 55.0, 100.0, 50.0),
            talker_angles_deg=(50.0, 195.0),
        )
        assert result["verdict"] == FAIL


class TestDeriveTimings:
    def test_derive_pre_roll_ms_matches_worst_case_formula(self) -> None:
        assert derive_pre_roll_ms([40, 60, 90]) == 200

    def test_derive_tail_ms_matches_worst_case_formula(self) -> None:
        assert derive_tail_ms([280, 300], endpointing_ms=400) == 250

    def test_derive_tail_ms_never_below_zero(self) -> None:
        assert derive_tail_ms([1000], endpointing_ms=50) == 0
