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

from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from spire_voice.config import TtsConfig
from spire_voice.providers.base import TtsError


# 20 ms of 16 kHz mono PCM16. Matches the frame size the browser plays and the
# camera speaker will want, so no consumer has to re-chunk.
_CHUNK_BYTES = 640


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
        """Render `text_deltas` to audio and yield it.

        VERIFIED AGAINST THE LIVE xAI API, 2026-09-17. This endpoint is REST,
        not a WebSocket:

            POST https://api.x.ai/v1/tts
            {"text": ..., "voice_id": ..., "language": ...,
             "output_format": {"codec": "pcm", "sample_rate": 16000}}
            -> 200, Content-Type: audio/pcm, the whole utterance as raw bytes

        This module previously opened a WebSocket to that same `https://` URL
        and died on `InvalidURI: scheme isn't ws or wss` the first time a real
        turn reached speech. The configured URL was right; the transport was
        not. `wss://api.x.ai/v1/tts` exists but rejected every parameter shape
        tried against it with HTTP 400, so REST is the path that works today.

        **A real latency consequence, stated rather than hidden:** REST returns
        the complete utterance in one response, so there is no first-chunk
        streaming. Time-to-first-audio is therefore full synthesis time, not
        time-to-first-delta, and `optimize_streaming_latency` has nothing to
        act on over this transport. That cost lands squarely in the
        end-of-speech-to-first-audio budget and is the honest reason a reply
        cannot start before the whole sentence is rendered. If xAI documents a
        working streaming socket later, this is the one function to change.

        The audio is yielded in chunks rather than as one object so the
        consumer's contract (an async iterator of bytes) stays identical
        across a future switch back to a streaming transport.
        """
        text = "".join([delta async for delta in text_deltas])
        if not text.strip():
            return

        sink = sink or self.browser_sink()
        payload = {
            "text": text,
            "voice_id": self._config.voice_id,
            "language": self._config.language,
            "output_format": {"codec": sink.codec, "sample_rate": sink.sample_rate},
        }
        headers = {"Authorization": f"Bearer {self._config.api_key}"}

        async with httpx.AsyncClient(timeout=self._config.request_timeout_s) as client:
            response = await client.post(self._config.url, headers=headers, json=payload)
            if response.status_code != 200:
                # 422 is a schema rejection and names the offending field --
                # surface it verbatim rather than flattening it to "TTS failed".
                raise TtsError(
                    f"xAI TTS returned {response.status_code}: {response.text[:300]}"
                )
            audio = response.content

        for start in range(0, len(audio), _CHUNK_BYTES):
            yield audio[start : start + _CHUNK_BYTES]
