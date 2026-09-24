#!/usr/bin/env python3
"""Writes two sample WAV files a person can double-click, for the plan
08-02 checkpoint that asks a real browser which A-law wrapping it actually
decodes (D-10, 08-RESEARCH.md Assumption A1).

Deliberately synthesises its own tone rather than reading a real session
directory -- this script never touches household audio, and the point of
the checkpoint is a header-format question, not a listening test of any
specific recording.
"""

from __future__ import annotations

import argparse
import math
import struct
import tempfile
from pathlib import Path

from atlas.audio.alaw import pcm16_to_alaw
from atlas.session.audio_wrap import wrap_alaw_as_wav, wrap_pcm16_as_wav

_SAMPLE_RATE = 8000
_DURATION_SECONDS = 1
_TONE_HZ = 440
_AMPLITUDE = 12000  # comfortably inside int16 range, audible without clipping


def _synthesize_tone() -> bytes:
    """One second of a 440 Hz sine at 8 kHz, as 16-bit signed PCM samples."""
    sample_count = _SAMPLE_RATE * _DURATION_SECONDS
    samples = (
        int(_AMPLITUDE * math.sin(2 * math.pi * _TONE_HZ * i / _SAMPLE_RATE))
        for i in range(sample_count)
    )
    return b"".join(struct.pack("<h", sample) for sample in samples)


def write_samples(out_dir: Path) -> tuple[Path, Path]:
    """Write the A-law-tagged and PCM16 sample WAV files under `out_dir`,
    returning their absolute paths. Writes nowhere else."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pcm16_samples = _synthesize_tone()
    alaw_bytes = pcm16_to_alaw(pcm16_samples)

    alaw_path = (out_dir / "sample-alaw.wav").resolve()
    pcm16_path = (out_dir / "sample-pcm16.wav").resolve()

    alaw_path.write_bytes(wrap_alaw_as_wav(alaw_bytes, _SAMPLE_RATE))
    pcm16_path.write_bytes(wrap_pcm16_as_wav(pcm16_samples, _SAMPLE_RATE))

    return alaw_path, pcm16_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "atlas-wav-check",
        help="Directory to write the two sample WAV files into "
        "(default: a directory under the system temp directory).",
    )
    args = parser.parse_args()

    alaw_path, pcm16_path = write_samples(args.out_dir)
    print(alaw_path)
    print(pcm16_path)


if __name__ == "__main__":
    main()
