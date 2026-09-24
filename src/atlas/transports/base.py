"""The one seam both browser transports terminate in.

The WebSocket transport and the WebRTC transport (plan 01-04) are two
physically different ways audio reaches this process -- one is raw binary
frames on an already-open socket, the other is Opus decoded out of an
`aiortc` media track. Nothing downstream of this protocol may know which one
is live: the turn controller takes an `AudioSource` and never imports either
concrete transport module. That is what keeps two transports from becoming
two pipelines.

Phase 2 adds a third `AudioSource`: the camera, at 8 kHz mono G.711 A-law,
never resampled or transcoded on the path to speech-to-text (PROV-07). A
source's format was safe to assume while there was exactly one -- it no
longer is. `source_format()` is how a consumer asks instead of assuming: an
`AudioSource` states the encoding and sample rate it actually produces, and
nothing downstream may assume a format it was not told.

Four members and nothing else:

- `frames()` -- an async iterator of raw audio byte frames, in the encoding
  `source_format()` names, until the source is exhausted.
- `send_audio()` -- reply audio going back out over the same connection the
  frames arrived on.
- `send_event()` -- the partial transcript, the reply text, and the timing
  line the page renders live, as JSON-shaped events.
- `source_format()` -- the `SourceFormat` this source actually produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


@dataclass(frozen=True)
class SourceFormat:
    """The encoding and sample rate one `AudioSource` actually produces.

    Two encodings exist in this codebase today: `"pcm"` (16 kHz mono PCM16,
    both browser transports) and `"alaw"` (8 kHz mono G.711 A-law, the
    camera). A consumer reads this value rather than hardcoding one, the
    same move `tts_xai.py`'s `SinkFormat` already made for the output side.
    """

    encoding: str
    sample_rate: int


class AudioSource(Protocol):
    """A source of raw audio frames, transport-agnostic, format-declared."""

    def frames(self) -> AsyncIterator[bytes]:
        """Yield raw audio frames, in this source's own encoding, until exhausted."""
        ...

    async def send_audio(self, chunk: bytes) -> None:
        """Send one chunk of reply audio back to whatever produced the frames."""
        ...

    async def send_event(self, event: dict[str, Any]) -> None:
        """Send one JSON-shaped event: a partial transcript, reply text, or a timing line."""
        ...

    def source_format(self) -> "SourceFormat":
        """The encoding and sample rate this source actually produces."""
        ...
