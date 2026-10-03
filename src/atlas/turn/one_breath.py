"""A wake phrase and a command in one breath, on a provider that gives no
word before it is finalized (debug session wake-command-needs-beep-wait).

On the Pi, "Hey Atlas, what time is it" said without a pause is one VAD
segment, and that segment is already in progress at the wake hit.
`_drain_to_final_transcript` lets a `vad.end` finalize only after a word, or
after a segment that began inside the drain. `ParakeetStt` and
`FasterWhisperStt` give no word before they are finalized, so the end of the
one-breath segment finalized nothing. The turn waited for new speech or for
`max_utterance_s`, and then ended with no transcript.

`EarlyTranscript` reads a copy of the frames that speech-to-text reads. When
the segment that was in progress at the start of the drain ends, it
transcribes those frames once, with `retranscribe`. The drain ends with that
text only when `accept` takes it and no new speech began meanwhile. In all
other cases nothing changes: the drain waits as before, and the final
transcript covers the whole turn. This is important in wait mode. Parakeet
hears "atlas" best with the whole turn as context. On recorded turns, a wake
segment alone decoded to "Yeah." or "" several times.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable

from atlas.turn.early_finalize import wait_for_end_of_speech
from atlas.turn.transcript_trim import retranscribe

logger = logging.getLogger(__name__)

# The early transcription is one decode of a few seconds of audio. Past this
# time, the drain continues as if there were no early transcription.
EARLY_TRANSCRIPT_TIMEOUT_S = 5.0


class EarlyTranscript:
    """One early transcription for the first drain of a wake turn.

    `tee` wraps the frames that speech-to-text reads and keeps a copy.
    `watch` waits for the end of the current segment, transcribes the copy,
    and sets `text` and `end_at` when `accept` takes the text and no new
    segment started. The caller runs `watch` as a task and reads `text`.
    """

    def __init__(
        self,
        stt: Any,
        source_format: Any,
        speech_signals: Any,
        accept: Callable[[str], bool],
        *,
        timeout_s: float = EARLY_TRANSCRIPT_TIMEOUT_S,
    ) -> None:
        self._stt = stt
        self._source_format = source_format
        self._signals = speech_signals
        self._accept = accept
        self._timeout_s = timeout_s
        self._frames: list[bytes] = []
        self.text: str | None = None
        self.end_at: float | None = None

    def tee(self, frames: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        async def _copy() -> AsyncIterator[bytes]:
            async for chunk in frames:
                self._frames.append(chunk)
                yield chunk

        return _copy()

    async def watch(self) -> None:
        try:
            await self._watch()
        except asyncio.CancelledError:
            raise
        except Exception:
            # The drain continues without an early transcription.
            logger.exception("one-breath early transcription failed")

    async def _watch(self) -> None:
        end_at = await wait_for_end_of_speech(
            self._signals,
            hangover_s=self._signals.hangover_s,
            already_ended_counts=True,
        )
        new_speech = False

        def _on_event(event: dict[str, Any]) -> None:
            nonlocal new_speech
            if event.get("type") == "vad.start":
                new_speech = True

        unsubscribe = self._signals.subscribe(_on_event, replay_segment=False)
        try:
            text = await retranscribe(
                self._stt, list(self._frames), self._source_format, timeout_s=self._timeout_s
            )
        finally:
            unsubscribe()
        if text is None or new_speech or self._signals.in_speech:
            return
        if not self._accept(text):
            return
        logger.info("one-breath wake: the wake segment holds the command")
        self.text = text
        self.end_at = end_at
