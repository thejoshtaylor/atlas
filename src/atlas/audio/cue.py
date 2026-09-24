"""The short chime a wake-word turn plays before it listens, so the operator
knows when to start the command.

Generated in code, not shipped as a file: two rising tones, each with a
short fade so the speaker does not click. Encoded for the sink the source
plays (`providers/tts_xai.py::SinkFormat`), so the camera gets 8 kHz A-law
like every other byte it plays.

This module also makes sink-format silence for the speaker tail flush
(`speaker/fifo_writer.py`'s idle tail pad, 260923-sfi).
"""

from __future__ import annotations

import numpy as np

from atlas.audio.alaw import pcm16_to_alaw
from atlas.providers.tts_xai import SinkFormat

_TONES_HZ = (660.0, 880.0)
_TONE_S = 0.09
_FADE_S = 0.01
_AMPLITUDE = 0.35 * 32767


def wake_cue(sink: SinkFormat) -> bytes:
    """The chime in `sink`'s codec, or `b""` for a codec this module cannot
    encode (the turn then just skips the cue)."""
    rate = sink.sample_rate
    n = int(rate * _TONE_S)
    t = np.arange(n) / rate
    fade = np.ones(n)
    k = int(rate * _FADE_S)
    fade[:k] = np.linspace(0.0, 1.0, k)
    fade[-k:] = np.linspace(1.0, 0.0, k)
    pcm = np.concatenate([np.sin(2 * np.pi * hz * t) * fade * _AMPLITUDE for hz in _TONES_HZ])
    pcm16 = np.round(pcm).astype("<i2").tobytes()
    if sink.codec == "alaw":
        return pcm16_to_alaw(pcm16)
    if sink.codec == "pcm":
        return pcm16
    return b""


def silence(sink: SinkFormat, seconds: float) -> bytes:
    """`seconds` of digital silence, in `sink`'s own codec.

    A-law's idle byte is `0xD5`, not `0x00` (`0x00` decodes to a loud
    value in A-law's companding). "pcm" is 16-bit signed little-endian, so
    its silence is a real zero, two bytes per sample. Raises `ValueError`
    for a codec this module cannot encode -- an empty result here would
    write nothing to the FIFO and bring the tail hold straight back, with
    no error to say why.
    """
    if seconds < 0:
        raise ValueError(f"silence() duration must not be negative, got {seconds!r}")
    n = round(sink.sample_rate * seconds)
    if sink.codec == "alaw":
        return b"\xd5" * n
    if sink.codec == "pcm":
        return b"\x00" * (2 * n)
    raise ValueError(f"silence() has no encoding for sink codec {sink.codec!r}")
