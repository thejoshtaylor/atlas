"""The xAI streaming speech-to-text client.

Two correctness constraints carry this whole module, both stated here rather
than performed silently, because getting either one wrong produces a
connection that looks fine and quietly does the wrong thing:

1. Pitfall 1 -- `SttConfig` carries `endpointing_ms` and
   `smart_turn_timeout_ms`; xAI's wire parameters are `endpointing` and
   `smart_turn_timeout`, with no `_ms` suffix. An unrecognized query
   parameter name is not an error: xAI's server silently falls back to its
   own default, so a mismatched name produces a connection that never
   honours the tuned values and reports nothing wrong. This module
   translates explicitly rather than passing the config keys straight
   through.
2. Pitfall 2 -- session configuration is the URL query string, not a
   message sent after connecting. The full URL is built before the socket
   opens, and the client waits for `transcript.created` before it streams
   the first audio frame.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import websockets

from spire_voice.config import SttConfig
from spire_voice.providers.base import FinalTranscript, PartialTranscript, SttError


class XaiStt:
    """Streaming speech-to-text over xAI's WebSocket endpoint."""

    def __init__(self, config: SttConfig) -> None:
        self._config = config

    def build_url(self) -> str:
        """The full connect URL, wire parameter names only.

        `encoding`/`sample_rate` describe the browser microphone's own
        capture format (16 kHz mono PCM16, per `pcm-worklet.js`) -- Phase 1's
        only audio source, so this is not itself a config value.
        """
        params = {
            "encoding": "pcm",
            "sample_rate": 16000,
            "endpointing": self._config.endpointing_ms,
            "smart_turn": self._config.smart_turn,
            "smart_turn_timeout": self._config.smart_turn_timeout_ms,
            "vad_threshold": self._config.vad_threshold,
            "interim_results": str(self._config.interim_results).lower(),
            "language": self._config.language,
        }
        return f"{self._config.url}?{urlencode(params)}"

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[PartialTranscript | FinalTranscript]:
        """Open the socket, stream `frames`, and yield transcript events.

        Opened the instant the turn starts (mic toggle pressed), not at end
        of speech -- with no wake word in Phase 1, that is the whole
        definition of "turn starts."
        """
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        async with websockets.connect(self.build_url(), additional_headers=headers) as ws:
            ready = json.loads(await ws.recv())
            if ready.get("type") != "transcript.created":
                raise SttError(f"unexpected first event from xAI STT: {ready!r}")

            async def sender() -> None:
                async for chunk in frames:
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "audio.done"}))

            send_task = asyncio.create_task(sender())
            try:
                async for raw in ws:
                    event: dict[str, Any] = json.loads(raw)
                    event_type = event.get("type")
                    if event_type == "transcript.partial":
                        yield PartialTranscript(text=event.get("text", ""))
                    elif event_type == "transcript.done":
                        yield FinalTranscript(text=event.get("text", ""))
                        break
                    elif event_type == "error":
                        raise SttError(event.get("message", "xAI STT reported an error"))
            finally:
                await send_task
