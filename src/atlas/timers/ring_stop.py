"""Stop a ringing timer or alarm with one bare spoken word.

While a ring plays, the room has no wake word. A person who says "stop"
expects the noise to end. This module holds three parts.

- `is_stop_command` decides whether a transcript is a stop command.
- `RingStopWindow` listens to the source only while a ring plays. It calls
  nothing except `stop_ringing()`. Speech that is not a stop command reaches
  no brain, no tool and no turn, and the ring continues.
- `make_stt_transcribe` and `QuietSource` let the window read one utterance
  through the deployment's own speech-to-text provider. `QuietSource` drops
  every event, so un-woken room speech is never published or recorded.

The TV can say "stop". At worst, it silences a ring. The owner accepts that
risk. The window exists only while a ring plays, and the word set is tight.

The camera has no echo cancellation, so its microphone hears the ring's own
announcement. The matcher removes the announcement text before it matches.
No announcement form holds only stop words, so an echo that is not removed
still does not match.

This module never imports `atlas.turn.controller` at module level. The
controller imports this module. `make_stt_transcribe` imports the controller
inside its function body, the same pattern `turn/macros.py` uses.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger("atlas.timers.ring_stop")

# How long one listen waits for a person to start speaking.
RING_LISTEN_S = 8.0
# A listen that ends sooner than this, with no stop word, counts as quick.
_QUICK_LISTEN_S = 0.5
# This many quick listens in a row mean the STT answers at once with nothing.
_MAX_QUICK_LISTENS = 3
_MAX_STOP_TOKENS = 6

_PUNCT_RE = re.compile(r"[^\w\s]")

_STOP_WORDS = frozenset({"stop", "cancel", "off", "okay", "ok", "enough", "thanks"})
_STOP_PHRASES = tuple(
    sorted(
        (
            tuple(phrase.split())
            for phrase in (
                "shut up",
                "thank you",
                "stop it",
                "stop that",
                "thats enough",
                "turn it off",
                "stop the timer",
                "stop the alarm",
                "cancel the timer",
                "cancel the alarm",
                "turn off the timer",
                "turn off the alarm",
                "turn the timer off",
                "turn the alarm off",
                "stop timer",
                "stop alarm",
                "timer off",
                "alarm off",
            )
        ),
        key=len,
        reverse=True,
    )
)


def _normalize(text: str) -> list[str]:
    return _PUNCT_RE.sub("", text.lower()).split()


def _remove_run(tokens: list[str], run: list[str]) -> list[str]:
    """Drop every whole-token occurrence of `run` from `tokens`."""
    if not run:
        return tokens
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index : index + len(run)] == run:
            index += len(run)
        else:
            kept.append(tokens[index])
            index += 1
    return kept


def _covered(tokens: list[str]) -> bool:
    """True when the tokens split into stop phrases and stop words."""
    if not tokens:
        return True
    for phrase in _STOP_PHRASES:
        if tuple(tokens[: len(phrase)]) == phrase and _covered(tokens[len(phrase) :]):
            return True
    return tokens[0] in _STOP_WORDS and _covered(tokens[1:])


def is_stop_command(text: str, *, ring_text: str | None = None) -> bool:
    """True when `text` is a bare command to stop the ring.

    `ring_text` is the announcement that plays now. Its words are removed
    first, so the microphone hearing the ring does not count.
    """
    tokens = _normalize(text)
    if ring_text:
        tokens = _remove_run(tokens, _normalize(ring_text))
    if tokens[:2] == ["hey", "atlas"]:
        tokens = tokens[2:]
    elif tokens[:1] == ["atlas"]:
        tokens = tokens[1:]
    tokens = [token for token in tokens if token != "please"]
    if not 1 <= len(tokens) <= _MAX_STOP_TOKENS:
        return False
    return _covered(tokens)


class RingControl(Protocol):
    """What the window and the turn need from a ring owner."""

    @property
    def ringing(self) -> bool: ...

    @property
    def ring_text(self) -> str | None: ...

    def stop_ringing(self) -> bool: ...

    async def wait_ring_over(self) -> None: ...


class QuietSource:
    """Wraps a source so the window can read it and publish nothing.

    `send_event` does nothing. The transcript partials, the observer feed and
    the edge LED events that a real turn sends never happen for un-woken
    speech.
    """

    def __init__(self, source: Any) -> None:
        self._source = source

    def frames(self) -> Any:
        return self._source.frames()

    def source_format(self) -> Any:
        return self._source.source_format()

    def sink_format(self) -> Any:
        sink_format = getattr(self._source, "sink_format", None)
        return sink_format() if sink_format is not None else None

    async def send_event(self, event: dict[str, Any]) -> None:
        return None


def make_stt_transcribe(
    stt: Callable[[], Any | None],
    *,
    max_utterance_s: float,
    listen_s: float = RING_LISTEN_S,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[Any], Awaitable[str | None]]:
    """Build the `transcribe(source)` the window calls for each listen.

    `stt` is read on every call, because the provider can change at run time.
    It returns None while speech-to-text is unavailable.
    """

    async def transcribe(source: Any) -> str | None:
        provider = stt()
        if provider is None:
            raise RuntimeError("speech-to-text is not available for the ring-stop window")
        from atlas.timing import TurnTimings
        from atlas.turn.controller import _DEFAULT_POLL_INTERVAL_S, _drain_to_final_transcript

        final = await _drain_to_final_transcript(
            QuietSource(source),
            provider,
            max_utterance_s,
            TurnTimings(),
            clock=clock,
            poll_interval_s=_DEFAULT_POLL_INTERVAL_S,
            onset_deadline=clock() + listen_s,
            speech_signals=getattr(source, "speech_signals", None),
            finalize_if_already_ended=False,
        )
        return getattr(final, "text", None) if final is not None else None

    return transcribe


class RingStopWindow:
    """Listens with no wake word while a ring plays. Its only effect is a stop."""

    def __init__(
        self,
        ring: Callable[[], RingControl | None],
        transcribe: Callable[[Any], Awaitable[str | None]],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ring = ring
        self._transcribe = transcribe
        self._clock = clock
        self._gave_up = False

    def active(self) -> bool:
        """True while a ring plays and the window has not given up on it."""
        ring = self._ring()
        if ring is None or not ring.ringing:
            self._gave_up = False
            return False
        return not self._gave_up

    def _give_up(self, reason: str) -> None:
        self._gave_up = True
        logger.warning("ring-stop window gave up for this ring: %s", reason)

    async def run(self, source: Any) -> None:
        """Listen until the ring ends, a stop word comes, or the window gives up."""
        quick_listens = 0
        while True:
            ring = self._ring()
            if ring is None or not ring.ringing:
                return
            ring_text = ring.ring_text
            listen = asyncio.ensure_future(self._transcribe(source))
            over = asyncio.ensure_future(ring.wait_ring_over())
            started = self._clock()
            try:
                await asyncio.wait({listen, over}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (listen, over):
                    if not task.done():
                        task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            if not listen.done() or listen.cancelled():
                return
            error = listen.exception()
            if error is not None:
                self._give_up(f"listening failed: {error}")
                return
            text = listen.result()
            if text and is_stop_command(text, ring_text=ring_text):
                ring.stop_ringing()
                logger.info("ring stopped by a spoken stop word")
                return
            logger.debug("ring-stop window heard speech that is not a stop word")
            quick_listens = quick_listens + 1 if self._clock() - started < _QUICK_LISTEN_S else 0
            if quick_listens >= _MAX_QUICK_LISTENS:
                self._give_up("speech-to-text keeps answering at once with no stop word")
                return
