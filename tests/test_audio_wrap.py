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

import subprocess
import sys
from pathlib import Path

import pytest

from atlas.audio.alaw import alaw_to_pcm16
from atlas.session.audio_wrap import (
    DEFAULT_ALAW_WRAPPING,
    AudioWrapError,
    wrap_alaw_as_wav,
    wrap_pcm16_as_wav,
    wrap_session_audio,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "dev-write-sample-wav.py"


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


def test_dev_write_sample_wav_script_writes_two_playable_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "samples"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    alaw_path = out_dir / "sample-alaw.wav"
    pcm16_path = out_dir / "sample-pcm16.wav"
    assert alaw_path.is_file()
    assert pcm16_path.is_file()

    alaw_bytes = alaw_path.read_bytes()
    pcm16_bytes = pcm16_path.read_bytes()
    assert alaw_bytes[:4] == b"RIFF"
    assert pcm16_bytes[:4] == b"RIFF"

    # The whole point of A-law: one byte per sample against PCM16's two,
    # so the A-law file should land at roughly half the PCM16 file's size
    # -- the cheapest available proof the two paths are not silently
    # producing identical bytes under two names.
    ratio = len(alaw_bytes) / len(pcm16_bytes)
    assert 0.4 < ratio < 0.6, (len(alaw_bytes), len(pcm16_bytes))

    printed_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(printed_lines) == 2
    for line in printed_lines:
        assert Path(line).is_absolute()


def test_dev_write_sample_wav_default_out_dir_is_outside_repo() -> None:
    # No --out-dir given: the script's own default must never resolve
    # inside the working tree, so a bare invocation cannot leave household
    # audio -- or, here, a synthesised tone -- anywhere git can see.
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    printed_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(printed_lines) == 2
    for line in printed_lines:
        assert not Path(line).is_relative_to(_REPO_ROOT), line
