"""The wake cue: encoded for the sink the source plays, and played first
in a camera turn so the operator knows when to speak."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.audio.alaw import alaw_to_pcm16
from atlas.audio.cue import silence, wake_cue
from atlas.providers.tts_xai import SinkFormat


def test_alaw_cue_is_about_180ms_of_audible_8khz_alaw():
    cue = wake_cue(SinkFormat(codec="alaw", sample_rate=8000))
    assert len(cue) == 2 * int(8000 * 0.09)  # one byte per A-law sample
    pcm = np.frombuffer(alaw_to_pcm16(cue), dtype="<i2").astype(float)
    assert np.sqrt(np.mean(pcm * pcm)) > 3000
    assert abs(pcm[0]) < 200 and abs(pcm[-1]) < 200  # faded, no click


def test_pcm_cue_is_little_endian_pcm16_and_unknown_codecs_get_nothing():
    assert len(wake_cue(SinkFormat(codec="pcm", sample_rate=24000))) == 2 * 2 * int(24000 * 0.09)
    assert wake_cue(SinkFormat(codec="mp3", sample_rate=24000)) == b""


# --- 260923-sfi (D1): silence(), the speaker tail-flush pad's payload ---


def test_silence_is_pcm_zero_bytes_at_two_bytes_per_sample():
    assert silence(SinkFormat("pcm", 24000), 0.2) == b"\x00" * 9600


def test_silence_is_the_alaw_idle_byte_one_byte_per_sample():
    assert silence(SinkFormat("alaw", 8000), 0.2) == b"\xd5" * 1600


def test_silence_alaw_decodes_to_near_zero_pcm():
    pcm = np.frombuffer(alaw_to_pcm16(silence(SinkFormat("alaw", 8000), 0.2)), dtype="<i2")
    assert np.max(np.abs(pcm.astype(int))) <= 8


def test_silence_of_zero_seconds_is_empty():
    assert silence(SinkFormat("pcm", 24000), 0) == b""
    assert silence(SinkFormat("alaw", 8000), 0) == b""


def test_silence_rejects_an_unsupported_codec_naming_it():
    with pytest.raises(ValueError) as exc:
        silence(SinkFormat("mp3", 8000), 0.2)
    assert "mp3" in str(exc.value)


def test_silence_rejects_a_negative_duration():
    with pytest.raises(ValueError):
        silence(SinkFormat("pcm", 24000), -0.1)
