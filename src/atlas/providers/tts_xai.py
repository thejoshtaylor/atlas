"""The xAI text-to-speech client: one REST call, browser sink format.

Pitfall 4 is the correctness constraint this module exists to close: the
camera-facing `tts.codec`/`tts.sample_rate` pair is A-law at 8 kHz, and the
Web Audio API cannot decode that at all -- it plays as audible noise, not a
clean failure, which looks like a working-but-garbled pipeline in manual
testing. The sink format is a parameter, not a hardcoded assumption, so the
Phase 2 camera sink can request its own codec without either sink assuming
the other's values.

Plan 07-02 (D-05, D-07): this client knows nothing about chunking or
timing any more. `synthesize_once()` returns the whole response body;
`providers/batch_tts_adapter.py::BatchTtsAdapter` is what chunks that
buffer and satisfies the streaming `TtsProvider` protocol on top of it.
Two chunking loops -- one here, one in the adapter -- would be the exact
duplicate D-07 forbids.

Quick task 260924-4iu (a): `synthesize_once` used to open a fresh
`httpx.AsyncClient` for every call, so every reply paid a new TCP and TLS
handshake before the request itself even started. One `XaiTts` instance
now carries one lazily built client, reused across every call it makes,
and `aclose()` closes it. `app.py`'s lifespan shutdown calls `aclose()`
through `BatchTtsAdapter`'s forwarding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from atlas.config import TtsConfig
from atlas.providers.base import TtsError

logger = logging.getLogger("atlas.providers.tts_xai")


# 20 ms of 16 kHz mono PCM16. Matches the frame size the browser plays and
# the camera speaker will want. `BatchTtsAdapter` imports this rather than
# restating it -- one definition of the chunk size, not two.
CHUNK_BYTES = 640

# httpx's own default keepalive expiry is 5 s. A normal back-and-forth
# conversation leaves the connection idle for longer than that between
# turns, so the default would drop the socket between almost every pair of
# replies and the shared client would then save nothing over the old
# per-call client. 120 s keeps one TLS connection alive across a normal
# conversation's pauses. httpx checks an idle pooled socket before it
# reuses one and reconnects on its own when the server already closed it.
#
# 260924-4iv (item d): `providers/base.py::PROVIDER_KEEPALIVE_EXPIRY_S` is
# 60 s -- the floor a wake-time warm call needs to still be useful by the
# time a turn reaches this provider. 120 s already clears that floor
# (chosen for a different, longer-lived reason above), so this value is
# left exactly as 260924-4iu set it rather than swapped for
# `provider_http_limits()`.
_KEEPALIVE_EXPIRY_S = 120.0


@dataclass(frozen=True)
class SinkFormat:
    """The codec and sample rate one playback sink actually wants."""

    codec: str
    sample_rate: int


class XaiTts:
    """xAI's text-to-speech endpoint: one REST call, the whole utterance
    in the response body -- batch, not streaming (D-05)."""

    def __init__(self, config: TtsConfig, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        # An injected client is a caller's own -- this instance never
        # builds one and `aclose()` never closes it. `self._client` starts
        # `None` either way; the owned case builds its client lazily, on
        # first use, in `_get_client` below.
        self._client = http_client
        self._owns_client = http_client is None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the one client this instance sends every request
        through, building it on first use.

        Built lazily rather than in `__init__`: the provider object can be
        constructed before the event loop that will use it exists, and
        `httpx.AsyncClient` binds to whichever loop is running when it is
        built. `httpx.AsyncClient` is read off the module here, at call
        time, rather than captured at import time, so a test that
        monkeypatches `atlas.providers.tts_xai.httpx.AsyncClient` still
        reaches its replacement.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.request_timeout_s,
                limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY_S),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the client this instance built, if it built one.

        Safe before any call was ever made (there is no client yet, and
        this does nothing) and safe to call twice (the second call finds
        `self._client` already `None`). Never closes an injected client --
        that client belongs to whoever passed it in.
        """
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def warm(self) -> None:
        """Open (or reuse) this instance's pooled connection at wake, so a
        turn that starts moments later skips the TLS handshake (260924-4iv,
        item d).

        One `HEAD` to `self._config.url`, through `_get_client()` -- the
        same client accessor `synthesize_once` uses, so the pool this warms
        is the one the next synthesis call reuses. `tts.url` answers `POST`
        only, so a `405` here is expected and still counts as warmed: any
        status code proves the connection opened and the server answered.
        Every `Exception` is swallowed, logged at debug -- a warm call must
        never reach a turn or raise past the background task
        `app.py::_warm_providers` runs it in.
        """
        try:
            headers = {"Authorization": f"Bearer {self._config.api_key}"}
            await self._get_client().head(self._config.url, headers=headers, timeout=5.0)
        except Exception:
            logger.debug("xAI TTS warm call failed", exc_info=True)

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

    async def synthesize_once(self, text: str, sink: SinkFormat | None = None) -> bytes:
        """Render `text` to audio in one call and return the whole buffer.

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

        Plan 07-02: no empty-text short circuit and no chunking here any
        more -- `BatchTtsAdapter` owns both. This method is a plain,
        unconditional REST call from text to bytes.

        Quick task 260924-4iu (a): the request goes through this instance's
        one shared client (`_get_client`), not a fresh client per call, so
        repeated replies reuse one TCP and TLS connection instead of paying
        a new handshake each time. The API key stays in this one request's
        own `Authorization` header, never in the shared client's default
        headers, so it is never held anywhere longer than one call needs it.
        A pooled keep-alive connection the server closed at the moment of
        reuse raises `httpx.RemoteProtocolError`; a render has no side
        effect, so this retries the POST exactly once on that error and
        lets any other exception, and a second `RemoteProtocolError`,
        propagate unchanged.
        """
        sink = sink or self.browser_sink()
        payload = {
            "text": text,
            "voice_id": self._config.voice_id,
            "language": self._config.language,
            "output_format": {"codec": sink.codec, "sample_rate": sink.sample_rate},
        }
        headers = {"Authorization": f"Bearer {self._config.api_key}"}

        client = self._get_client()
        try:
            response = await client.post(self._config.url, headers=headers, json=payload)
        except httpx.RemoteProtocolError:
            response = await client.post(self._config.url, headers=headers, json=payload)
        if response.status_code != 200:
            # 422 is a schema rejection and names the offending field --
            # surface it verbatim rather than flattening it to "TTS failed".
            raise TtsError(
                f"xAI TTS returned {response.status_code}: {response.text[:300]}"
            )
        return response.content
