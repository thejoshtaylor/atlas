"""Tests for `session/audio_wrap.py` -- the WAV-wrapping module that turns
raw session audio bytes into a file a browser's `<audio>` element can play.

Every fixture here is synthesised in-process (`bytes(range(256))`
exercises A-law's full code range end to end) rather than read from or
written to a real session directory:
`test_repo_hygiene.py::test_no_session_path_or_audio_extension_is_tracked_by_git`
is what keeps a home's actual audio out of git, and proving a WAV header
is well-formed needs no real recording.
"""

from __future__ import annotations

import pytest

from spire_voice.audio.alaw import alaw_to_pcm16
from spire_voice.session.audio_wrap import (
    DEFAULT_ALAW_WRAPPING,
    AudioWrapError,
    wrap_alaw_as_wav,
    wrap_pcm16_as_wav,
    wrap_session_audio,
)


def _fmt_chunk_size(wav: bytes) -> int:
    return int.from_bytes(wav[16:20], "little")


def _fmt_tag(wav: bytes) -> int:
    return int.from_bytes(wav[20:22], "little")


def _bits_per_sample(wav: bytes) -> int:
    # wBitsPerSample is the last 2-byte field of the PCM-shaped 14 bytes
    # that precede any trailing cbSize -- at a fixed offset regardless of
    # whether the fmt subchunk is 16 or 18 bytes.
    return int.from_bytes(wav[34:36], "little")


def _data_chunk_offset(wav: bytes) -> int:
    return 20 + _fmt_chunk_size(wav)


def _declared_data_size(wav: bytes) -> int:
    offset = _data_chunk_offset(wav)
    assert wav[offset : offset + 4] == b"data"
    return int.from_bytes(wav[offset + 4 : offset + 8], "little")


def _payload(wav: bytes) -> bytes:
    offset = _data_chunk_offset(wav) + 8
    return wav[offset:]


def _declared_riff_size(wav: bytes) -> int:
    return int.from_bytes(wav[4:8], "little")


@pytest.mark.parametrize("length", [256, 255])
def test_wrap_alaw_as_wav_header_and_payload(length: int) -> None:
    raw = bytes(range(256))[:length]
    wav = wrap_alaw_as_wav(raw, 8000)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert _fmt_chunk_size(wav) == 18
    assert _fmt_tag(wav) == 6
    assert _bits_per_sample(wav) == 8
    assert _declared_data_size(wav) == length
    assert _payload(wav) == raw
    assert _declared_riff_size(wav) == len(wav) - 8


@pytest.mark.parametrize("length", [256, 255])
def test_wrap_pcm16_as_wav_header_and_payload(length: int) -> None:
    samples = bytes(range(256))[:length]
    wav = wrap_pcm16_as_wav(samples, 16000)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert _fmt_chunk_size(wav) == 16
    assert _fmt_tag(wav) == 1
    assert _bits_per_sample(wav) == 16
    assert _declared_data_size(wav) == length
    assert _payload(wav) == samples
    assert _declared_riff_size(wav) == len(wav) - 8


def test_wrap_session_audio_alaw_dispatches_on_default_wrapping() -> None:
    raw = bytes(range(256))
    wav = wrap_session_audio("alaw", 8000, raw)

    if DEFAULT_ALAW_WRAPPING == "alaw":
        assert _fmt_tag(wav) == 6
        assert _bits_per_sample(wav) == 8
        assert _payload(wav) == raw
    else:
        assert DEFAULT_ALAW_WRAPPING == "pcm16"
        assert _fmt_tag(wav) == 1
        assert _bits_per_sample(wav) == 16
        assert _payload(wav) == alaw_to_pcm16(raw)


def test_wrap_session_audio_pcm_variants_never_touch_alaw_table() -> None:
    samples = bytes(range(256))
    for encoding in ("pcm", "pcm16"):
        wav = wrap_session_audio(encoding, 16000, samples)
        assert _fmt_tag(wav) == 1
        assert _bits_per_sample(wav) == 16
        assert _payload(wav) == samples


def test_wrap_session_audio_rejects_unknown_encoding() -> None:
    with pytest.raises(AudioWrapError, match="opus"):
        wrap_session_audio("opus", 8000, b"\x00")
