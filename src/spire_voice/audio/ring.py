"""Bounded pre-roll ring buffer (VOICE-06, T-02-17).

This module holds only the audio a wake hit will actually replay into the
transcriber -- never a continuous rolling capture (D-15). Bounded in bytes
at every push, not trimmed lazily at drain: a buffer trimmed only on drain
would hold an unbounded amount of room audio between two hits, which is
the exact failure CONTEXT.md rejects by name.

The byte bound is derived from the source's own declared `SourceFormat`
(`transports/base.py`) rather than a hardcoded rate -- a source at a
different sample rate or encoding must never be silently mis-sized.
"""

from __future__ import annotations

from collections import deque

from spire_voice.transports.base import SourceFormat

# Bytes per sample for the two encodings this codebase carries end to end
# (`transports/base.py`'s `SourceFormat` docstring). A-law is one byte per
# sample by definition of the codec; PCM16 is two. Neither is a rate this
# module assumes independently of `source_format`.
_PCM_BYTES_PER_SAMPLE = 2
_ALAW_BYTES_PER_SAMPLE = 1


def _bytes_per_ms(source_format: SourceFormat) -> float:
    """The byte rate implied by `source_format`'s own encoding and sample
    rate -- read from the source, never assumed."""
    bytes_per_sample = _PCM_BYTES_PER_SAMPLE if source_format.encoding == "pcm" else _ALAW_BYTES_PER_SAMPLE
    return source_format.sample_rate * bytes_per_sample / 1000.0


class PrerollBuffer:
    """The most recent `window_ms` of raw source chunks, oldest first,
    bounded in bytes at every `push()`."""

    def __init__(self, source_format: SourceFormat, window_ms: int) -> None:
        self._max_bytes = round(_bytes_per_ms(source_format) * window_ms)
        self._chunks: deque[bytes] = deque()
        self._held_bytes = 0

    @property
    def held_bytes(self) -> int:
        """How many bytes this buffer currently holds -- never more than
        the byte bound computed at construction."""
        return self._held_bytes

    def push(self, chunk: bytes) -> None:
        """Append `chunk`, then trim from the oldest end until back at or
        under the byte bound.

        Trimming happens here, at push time, so the buffer never grows
        past its configured window regardless of how long the gap between
        two wake hits runs -- the continuous-rolling-capture failure this
        module exists to refuse.
        """
        if not chunk:
            return
        self._chunks.append(chunk)
        self._held_bytes += len(chunk)
        while self._held_bytes > self._max_bytes and self._chunks:
            oldest = self._chunks.popleft()
            self._held_bytes -= len(oldest)

    def drain(self) -> list[bytes]:
        """Return every held chunk, oldest first, and empty the buffer.

        A drained buffer holds nothing: audio already replayed into one
        turn must never be replayed into the turn that follows it.
        """
        chunks = list(self._chunks)
        self.clear()
        return chunks

    def clear(self) -> None:
        """Discard everything held, with no return value."""
        self._chunks.clear()
        self._held_bytes = 0
