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
import contextlib
import json
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import websockets

from atlas.config import SttConfig
from atlas.providers.base import FinalTranscript, PartialTranscript, SttError
from atlas.transports.base import SourceFormat

# VERIFIED AGAINST THE LIVE API, 2026-09-25 (10-05-PLAN.md Task 3,
# scripts/verify_xai_finalize.py). xAI's own documentation spells this
# "Finalize"; D-12 wrote lowercase "finalize". Both spellings worked
# against the live socket, streaming one synthesized command's audio with
# no trailing `audio.done` and no other finalize message, timed from the
# moment each variant was sent to the first `speech_final` partial that
# followed it, against a control that sent neither and never finalized
# within the 5 s limit:
#
#     "Finalize" (xAI's documented spelling): speech_final in 289.3ms
#     "finalize" (D-12's spelling):           speech_final in 153.9ms
#     control (no finalize message sent):     no final within 5000ms
#
# Both spellings work, so D-12's own lowercase spelling is what this
# constant keeps (10-05-PLAN.md's own rule for this exact outcome) -- the
# decision's intent, a finalize message on the provider's own socket, holds
# either way.
FINALIZE_MESSAGE = {"type": "finalize"}


class XaiStt:
    """Streaming speech-to-text over xAI's WebSocket endpoint."""

    def __init__(self, config: SttConfig) -> None:
        self._config = config

    def build_url(self, source_format: SourceFormat) -> str:
        """The full connect URL, wire parameter names only.

        `encoding`/`sample_rate` name what `source_format` says the calling
        `AudioSource` actually produces. Phase 1 had one source, always 16
        kHz mono PCM16, so hardcoding the pair was safe; Phase 2 adds a
        second source at 8 kHz A-law, and assuming here would open the
        socket for one format while a different one arrives. xAI's own wire
        values for both encodings this repository produces are `"pcm"` and
        `"alaw"`, matching `SourceFormat.encoding` exactly -- no translation
        table entry exists for an encoding no source here produces.

        Phase 10 (D-09): a two-channel edge source must never reach this
        socket as if it were mono -- `turn/controller.py::_drain_to_final_
        transcript` always hands this provider a one-channel view via
        `stt_view`, so `channels != 1` here means a caller skipped that
        seam, not a legitimate multi-channel request.
        """
        if source_format.channels != 1:
            raise SttError(
                f"xAI speech-to-text requires a single-channel source, got "
                f"{source_format.channels} channels"
            )
        params = {
            "encoding": source_format.encoding,
            "sample_rate": source_format.sample_rate,
            "endpointing": self._config.endpointing_ms,
            "smart_turn": self._config.smart_turn,
            "smart_turn_timeout": self._config.smart_turn_timeout_ms,
            "vad_threshold": self._config.vad_threshold,
            "interim_results": str(self._config.interim_results).lower(),
            "language": self._config.language,
        }
        # `keyterm` repeats once per term: biases recognition toward the
        # words this house actually says ("Atlas", a device name).
        query = urlencode([*params.items(), *(("keyterm", t) for t in self._config.keyterms)])
        return f"{self._config.url}?{query}"

    async def stream(
        self,
        frames: AsyncIterator[bytes],
        source_format: SourceFormat,
        *,
        finalize: "asyncio.Event | None" = None,
    ) -> AsyncIterator[PartialTranscript | FinalTranscript]:
        """Open the socket, stream `frames`, and yield transcript events.

        Opened the instant the turn starts (mic toggle pressed, or the wake
        word firing on the camera path), not at end of speech.

        `finalize` (D-12), when given, ends the utterance at once: a
        `finalizer` task waits for the event and sends `FINALIZE_MESSAGE`
        once, cancelled in the same `finally` as `sender()`'s own
        `send_task`. xAI's own endpointing (`transcript.partial` carrying
        `speech_final`) keeps running exactly as it does today -- whichever
        of the two fires first ends the turn (D-13), so a lost VAD event on
        the caller's side never hangs this stream.
        """
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        async with websockets.connect(self.build_url(source_format), additional_headers=headers) as ws:
            ready = json.loads(await ws.recv())
            if ready.get("type") != "transcript.created":
                raise SttError(f"unexpected first event from xAI STT: {ready!r}")

            async def sender() -> None:
                async for chunk in frames:
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "audio.done"}))

            async def finalizer() -> None:
                # `finalize` is per stream call, never provider state
                # (base.py's `SttProvider.stream` docstring, SRC-03) -- this
                # closure captures the one event this call was given.
                await finalize.wait()
                await ws.send(json.dumps(FINALIZE_MESSAGE))

            send_task = asyncio.create_task(sender())
            finalize_task = asyncio.create_task(finalizer()) if finalize is not None else None
            try:
                saw_final = False
                async for raw in ws:
                    event: dict[str, Any] = json.loads(raw)
                    event_type = event.get("type")
                    if event_type == "transcript.partial":
                        # VERIFIED AGAINST THE LIVE API, 2026-09-17. The final
                        # transcript arrives as a `transcript.partial` carrying
                        # `speech_final: true` -- NOT on `transcript.done`,
                        # which is a terminator whose `text` is always "".
                        #
                        # Reading the text off `transcript.done` (what this
                        # code did, and what RESEARCH.md assumed) discards a
                        # perfect transcription on every real turn and answers
                        # "sorry, i didn't catch that". No fake caught it,
                        # because the fake encoded the same wrong assumption.
                        #
                        # Observed sequence for one spoken sentence:
                        #   partial is_final=False speech_final=False  interim
                        #   partial is_final=True  speech_final=False  segment
                        #   partial is_final=True  speech_final=True   FINAL
                        #   done                                       text=""
                        if event.get("speech_final"):
                            saw_final = True
                            yield FinalTranscript(text=event.get("text", ""))
                            break
                        yield PartialTranscript(text=event.get("text", ""))
                    elif event_type == "transcript.done":
                        # Reached without a `speech_final` partial: the
                        # utterance genuinely contained no speech. That is
                        # VOICE-08's empty-transcript path, and an empty final
                        # is the honest thing to report.
                        if not saw_final:
                            yield FinalTranscript(text=event.get("text", ""))
                        break
                    elif event_type == "error":
                        raise SttError(event.get("message", "xAI STT reported an error"))
            finally:
                # `sender()` drains `frames` -- the mic source -- into
                # `ws.send()`. It has no reason to finish just because the
                # transcript did: a live mic keeps streaming mid-turn (the
                # operator's own start/stop toggle is the only thing that
                # ends it), so once `transcript.done` has been yielded (or
                # an error breaks the loop), the outbound audio no longer
                # matters and `sender()` must be cancelled, not awaited to
                # completion. Awaiting it here instead is CR-02's bug: this
                # generator could never reach `StopAsyncIteration` while the
                # mic stayed open, which is always, mid-turn.
                send_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await send_task
                if finalize_task is not None:
                    finalize_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await finalize_task
