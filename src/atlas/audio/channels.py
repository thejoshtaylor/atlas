"""PCM16 channel selection for the wake detector and speech-to-text (D-09).

Every source before Phase 10 was implicitly mono: `frames()` yielded bytes
the detector and speech-to-text could read directly, with no channel to
pick. The edge source is the first that is genuinely two-channel (the
XVF3800 sends both its capture channels), and D-09 requires the session
recorder to keep both while the wake detector and speech-to-text see only
the one channel the spike identifies as the ASR beam.

`select_channel` is the one place that de-interleaves a chunk. `stt_view`
is the one place `_drain_to_final_transcript` (`turn/controller.py`) reads
to decide whether a source needs this split at all -- a one-channel format
returns `(frames, fmt)` unchanged, so every existing source is
byte-identical to before this module existed. This split runs after the
recorder's own tap (`_RecordingAudioSource` wraps `source` before
`_drain_to_final_transcript` is ever called) and before speech-to-text --
the only point where "both channels to the recorder, one to speech-to-text"
holds without a second reader of the source's own queue.
"""

from __future__ import annotations

from typing import AsyncIterator

import numpy as np

from atlas.transports.base import SourceFormat

_BYTES_PER_SAMPLE = 2  # PCM16, little-endian -- the only encoding this module handles.


def select_channel(chunk: bytes, channels: int, index: int) -> bytes:
    """Return channel `index` of `chunk`, a little-endian PCM16 buffer
    interleaved across `channels` channels.

    Raises `ValueError` for a `chunk` whose length is not a whole number of
    interleaved frames (`channels` samples of 2 bytes each) -- a partial
    frame at the boundary would silently shift every following sample onto
    the wrong channel rather than erroring, so this never guesses.
    """
    frame_bytes = _BYTES_PER_SAMPLE * channels
    if frame_bytes == 0 or len(chunk) % frame_bytes != 0:
        raise ValueError(
            f"select_channel: chunk of {len(chunk)} bytes is not a whole number of "
            f"{channels}-channel PCM16 frames ({frame_bytes} bytes each)"
        )
    samples = np.frombuffer(chunk, dtype="<i2").reshape(-1, channels)
    return samples[:, index].tobytes()


async def asr_channel_frames(frames: AsyncIterator[bytes], fmt: SourceFormat) -> AsyncIterator[bytes]:
    """De-interleave every chunk of `frames`, yielding only `fmt.asr_channel`."""
    async for chunk in frames:
        yield select_channel(chunk, fmt.channels, fmt.asr_channel)


def stt_view(
    frames: AsyncIterator[bytes], fmt: SourceFormat
) -> "tuple[AsyncIterator[bytes], SourceFormat]":
    """What speech-to-text should read: `(frames, fmt)` unchanged for a
    one-channel format (every source before Phase 10, byte-identical), or
    the ASR-channel generator plus a one-channel `SourceFormat` for a
    multi-channel `"pcm"` format.

    Any other multi-channel encoding raises `ValueError` -- there is no
    de-interleave defined for A-law, and a silent pass-through would feed
    speech-to-text an interleaved buffer it would decode as mono garbage
    with no error anywhere (RESEARCH.md Pitfall 2).
    """
    if fmt.channels == 1:
        return frames, fmt
    if fmt.encoding != "pcm":
        raise ValueError(
            f"stt_view: a multi-channel {fmt.encoding!r} format has no defined channel "
            "selection -- only multi-channel 'pcm' is supported"
        )
    return asr_channel_frames(frames, fmt), SourceFormat(fmt.encoding, fmt.sample_rate)
