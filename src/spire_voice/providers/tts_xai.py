"""The xAI streaming text-to-speech client, browser sink format.

Pitfall 4 is the correctness constraint this module exists to close: the
camera-facing `tts.codec`/`tts.sample_rate` pair is A-law at 8 kHz, and the
Web Audio API cannot decode that at all -- it plays as audible noise, not a
clean failure, which looks like a working-but-garbled pipeline in manual
testing. The sink format is a parameter, not a hardcoded assumption, so the
Phase 2 camera sink can request its own codec without either sink assuming
the other's values.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import websockets

from spire_voice.config import TtsConfig
from spire_voice.providers.base import TtsError


@dataclass(frozen=True)
class SinkFormat:
    """The codec and sample rate one playback sink actually wants."""

    codec: str
    sample_rate: int


class XaiTts:
    """Streaming text-to-speech over xAI's WebSocket endpoint."""

    def __init__(self, config: TtsConfig) -> None:
        self._config = config

    def browser_sink(self) -> SinkFormat:
        """The Phase 1 dev-harness sink: Web Audio-playable PCM, never A-law."""
        return SinkFormat(codec=self._config.browser_codec, sample_rate=self._config.browser_sample_rate)

    def build_session_update(self, sink: SinkFormat | None = None) -> dict[str, Any]:
        """The first message sent after connecting: voice, language, and output format.

        Defaults to the browser sink because that is Phase 1's only consumer;
        a future camera sink (Phase 2) passes its own `SinkFormat` explicitly
        rather than this module assuming one config value serves both.
        """
        sink = sink or self.browser_sink()
        return {
            "type": "session.update",
            "voice_id": self._config.voice_id,
            "language": self._config.language,
            "output_format": {"codec": sink.codec, "sample_rate": sink.sample_rate},
            "optimize_streaming_latency": self._config.optimize_streaming_latency,
        }

    async def synthesize(
        self, text_deltas: AsyncIterator[str], sink: SinkFormat | None = None
    ) -> AsyncIterator[bytes]:
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        async with websockets.connect(self._config.url, additional_headers=headers) as ws:
            await ws.send(json.dumps(self.build_session_update(sink)))

            async def sender() -> None:
                async for delta in text_deltas:
                    await ws.send(json.dumps({"type": "text.delta", "delta": delta}))
                await ws.send(json.dumps({"type": "text.done"}))

            send_task = asyncio.create_task(sender())
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    event_type = event.get("type")
                    if event_type == "audio.delta":
                        yield base64.b64decode(event["delta"])
                    elif event_type == "audio.done":
                        break
                    elif event_type == "error":
                        raise TtsError(event.get("message", "xAI TTS reported an error"))
            finally:
                await send_task
