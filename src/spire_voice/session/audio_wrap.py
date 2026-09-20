"""Wrap raw session audio bytes in a WAV container a browser can play.

`session/recorder.py` writes `audio.{encoding}` as raw codec bytes -- no
header, no container, because the recorder's only job is to save exactly
what it captured (D-15). Nothing plays a bare byte stream. This module is
where those bytes gain a WAV header, built by hand: Python's stdlib `wave`
module hardcodes `WAVE_FORMAT_PCM` in its header writer and its
compression-type setter accepts only that one value, both read directly
from its source during research -- it cannot write an A-law WAV at all. A
future reader should not reach for `wave.open()` here; that is why this
paragraph exists.

A-law's `fmt ` subchunk is 18 bytes, not 16: the WAVEFORMATEX convention
requires a trailing `cbSize` field for any non-PCM format tag, set to zero
here because there is no extra format-specific data to describe. PCM16's
subchunk stays the classic 16 bytes with no trailing field. Conflating
these two shapes is exactly the arithmetic error CONTEXT.md's original
"44-byte header" claim made.

`DEFAULT_ALAW_WRAPPING` selects which of the two wrappings the audio route
serves for a session recorded in A-law. It starts pointed at the
decoded-PCM16 path -- this plan's browser checkpoint is what decides it for
real, and this one constant is the only place that answer lives.

`wrap_session_audio`'s `encoding` argument accepts `"pcm"` and `"pcm16"` as
the same PCM16 branch: `transports/base.py`'s `SourceFormat.encoding` names
16-bit PCM audio `"pcm"`, and that is the exact string `timing.json`'s
`audio_format.encoding` will carry for a websocket/WebRTC-recorded turn, so
this function must recognise it to be usable by the audio route this module
exists to serve; `"pcm16"` is accepted too because it is the more precise
name and the one the plan for this module used.
"""

from __future__ import annotations

import struct
from typing import Literal

from spire_voice.audio.alaw import alaw_to_pcm16

_ALAW_FORMAT_TAG = 6  # WAVE_FORMAT_ALAW
_PCM_FORMAT_TAG = 1  # WAVE_FORMAT_PCM

# Encodings `wrap_session_audio` recognises for its `encoding` argument --
# a closed set, mirroring `audio/alaw.py::_BYTES_PER_SAMPLE`'s own habit of
# rejecting anything outside it by name rather than guessing a format tag.
_PCM_ENCODINGS = frozenset({"pcm", "pcm16"})
_ALAW_ENCODINGS = frozenset({"alaw"})

AlawWrapping = Literal["alaw", "pcm16"]

# Set to the decoded-PCM16 value until this plan's browser checkpoint
# answers whether A-law-tagged WAV actually plays in the operator's real
# browsers (08-RESEARCH.md Assumption A1). Flip this one constant, nothing
# else, if the answer is "yes".
DEFAULT_ALAW_WRAPPING: AlawWrapping = "pcm16"


class AudioWrapError(ValueError):
    """Raised for an encoding this module cannot honestly wrap, rather
    than guessing a format tag and emitting a file that misdescribes its
    own contents (mirrors `audio/alaw.py`'s own `AlawError(ValueError)`)."""


def wrap_alaw_as_wav(alaw_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw G.711 A-law bytes in a minimal, valid non-PCM WAV
    container: 8-bit, 1 byte/sample, format tag 6, an 18-byte `fmt `
    subchunk. The `data` payload is `alaw_bytes` untouched."""
    return _build_wav(
        payload=alaw_bytes,
        sample_rate=sample_rate,
        format_tag=_ALAW_FORMAT_TAG,
        bits_per_sample=8,
        include_cb_size=True,
    )


def wrap_pcm16_as_wav(samples: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit signed PCM samples in a standard PCM WAV container:
    format tag 1, a 16-byte `fmt ` subchunk, no trailing `cbSize`."""
    return _build_wav(
        payload=samples,
        sample_rate=sample_rate,
        format_tag=_PCM_FORMAT_TAG,
        bits_per_sample=16,
        include_cb_size=False,
    )


def wrap_session_audio(encoding: str, sample_rate: int, raw: bytes) -> bytes:
    """The one function the audio route calls, taking `timing.json`'s
    recorded `audio_format.encoding`/`sample_rate` and a session's raw
    `audio.{encoding}` bytes straight off disk.

    For `"alaw"`, dispatches on `DEFAULT_ALAW_WRAPPING`: either the raw
    bytes wrapped as-is via `wrap_alaw_as_wav`, or decoded through
    `alaw_to_pcm16` -- never reimplemented here -- and wrapped via
    `wrap_pcm16_as_wav`. For `"pcm"`/`"pcm16"`, wraps `raw` directly as
    PCM16 without touching the A-law table at all. Anything else raises
    `AudioWrapError` naming the encoding it was given.
    """
    if encoding in _ALAW_ENCODINGS:
        if DEFAULT_ALAW_WRAPPING == "alaw":
            return wrap_alaw_as_wav(raw, sample_rate)
        return wrap_pcm16_as_wav(alaw_to_pcm16(raw), sample_rate)
    if encoding in _PCM_ENCODINGS:
        return wrap_pcm16_as_wav(raw, sample_rate)
    raise AudioWrapError(
        f"unknown encoding {encoding!r} -- supported values are "
        f"{sorted(_ALAW_ENCODINGS | _PCM_ENCODINGS)!r}"
    )


def _build_wav(
    *,
    payload: bytes,
    sample_rate: int,
    format_tag: int,
    bits_per_sample: int,
    include_cb_size: bool,
) -> bytes:
    """Hand-built RIFF/WAVE bytes, per WAVEFORMATEX: `RIFF` + size + `WAVE`,
    then `fmt ` + size + body, then `data` + size + payload, all
    little-endian. Every declared size is the real byte length that
    follows it -- computed from `payload`/`fmt_chunk`, not assumed --
    which holds for a payload of odd length exactly as it does for even.
    """
    num_channels = 1
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    if include_cb_size:
        fmt_chunk = struct.pack(
            "<HHIIHHH",
            format_tag,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            0,  # cbSize -- required for any non-PCM format tag
        )
    else:
        fmt_chunk = struct.pack(
            "<HHIIHH",
            format_tag,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )

    data_size = len(payload)
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + data_size)
    header = (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_chunk))
        + fmt_chunk
        + b"data"
        + struct.pack("<I", data_size)
    )
    return header + payload
