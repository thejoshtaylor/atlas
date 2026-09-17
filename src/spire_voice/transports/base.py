"""The one seam both browser transports terminate in.

The WebSocket transport and the WebRTC transport (plan 01-04) are two
physically different ways audio reaches this process -- one is raw binary
frames on an already-open socket, the other is Opus decoded out of an
`aiortc` media track. Nothing downstream of this protocol may know which one
is live: the turn controller takes an `AudioSource` and never imports either
concrete transport module. That is what keeps two transports from becoming
two pipelines.

Three members and nothing else:

- `frames()` -- an async iterator of 16 kHz mono PCM16 byte frames, always,
  regardless of which transport produced them.
- `send_audio()` -- reply audio going back out over the same connection the
  frames arrived on.
- `send_event()` -- the partial transcript, the reply text, and the timing
  line the page renders live, as JSON-shaped events.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol


class AudioSource(Protocol):
    """A source of 16 kHz mono PCM16 audio frames, transport-agnostic."""

    def frames(self) -> AsyncIterator[bytes]:
        """Yield 16 kHz mono PCM16 frames until the source is exhausted."""
        ...

    async def send_audio(self, chunk: bytes) -> None:
        """Send one chunk of reply audio back to whatever produced the frames."""
        ...

    async def send_event(self, event: dict[str, Any]) -> None:
        """Send one JSON-shaped event: a partial transcript, reply text, or a timing line."""
        ...
