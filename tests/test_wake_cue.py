"""The wake cue: encoded for the sink the source plays, and played first
in a camera turn so the operator knows when to speak."""

from __future__ import annotations

import numpy as np

from atlas.audio.alaw import alaw_to_pcm16
from atlas.audio.cue import wake_cue
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
