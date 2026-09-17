"""The passthrough WebSocket transport.

Raw 16 kHz PCM16 mono arrives as binary WebSocket frames and is yielded
unchanged -- there is no decode step because there was no encode step on the
browser side (see `static/pcm-worklet.js`). Reply audio goes back as binary
frames; events (partial transcript, reply text, the timing line) go back as
JSON text frames on the same socket, so the WebSocket path never needs a
second channel.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Protocol

from spire_voice.transports.base import SourceFormat


class _Sendable(Protocol):
    """The subset of `starlette.websockets.WebSocket` this transport uses."""

    async def receive_bytes(self) -> bytes: ...
    async def send_bytes(self, data: bytes) -> None: ...
    async def send_text(self, data: str) -> None: ...


class WebSocketAudioSource:
    """Satisfies `AudioSource` over one already-accepted WebSocket connection."""

    def __init__(self, websocket: _Sendable) -> None:
        self._ws = websocket

    async def frames(self) -> AsyncIterator[bytes]:
        # WebSocketDisconnect is imported lazily so this module works against
        # any object satisfying `_Sendable`, including a test double that
        # never imports fastapi/starlette itself.
        from fastapi import WebSocketDisconnect

        while True:
            try:
                data = await self._ws.receive_bytes()
            except WebSocketDisconnect:
                return
            yield data

    async def send_audio(self, chunk: bytes) -> None:
        await self._ws.send_bytes(chunk)

    async def send_event(self, event: dict[str, Any]) -> None:
        await self._ws.send_text(json.dumps(event))

    def source_format(self) -> SourceFormat:
        """Raw binary WebSocket frames are 16 kHz mono PCM16, unchanged."""
        return SourceFormat("pcm", 16000)
