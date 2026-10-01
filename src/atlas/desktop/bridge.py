"""`DesktopEventBridge`: the Mac panel's one reader of the turn event feed
(Phase 15, D-05, D-12, D-13).

It subscribes to the `ObserverRegistry` one time, at lifespan start. A task
reads that queue and calls `handle` for each event. `handle` and
`DesktopHub.broadcast` are plain functions that only append to a bounded
per-Mac outbox, so a slow or dead Mac never slows a turn.

Admission gate (T-15-01). A turn reaches a Mac only when the server confirmed
its wake: the turn's own transcript opened with the wake phrase and `run_turn`
emitted `wake.confirmed`. The bridge admits that turn id. Every other event
of a turn that was not admitted is dropped. A detector hit that the server
does not confirm (a television) therefore puts nothing on a Mac socket.
The speaker check runs after this gate. A voice that says the wake phrase and
that the speaker check then refuses is already admitted: its words reach every
Mac before `turn.ended` closes the turn with `no_speech`.

A follow-up turn has no wake phrase. It is admitted only when its
`turn.started` says `follow_up`, it comes from the source of the turn that
opened the window, and it starts before that window closes (D-05, T-15-17).

Events are attributed by `turn_id`, never by source name, because turn groups
run several turns at once on one source (RESEARCH Pitfall 1).

The translation of one admitted turn, in order: `wake.confirmed`; each
`transcript.partial` (sanitized, the newest 1000 characters); `transcript.final`
then state `thinking`; on `reply.started`, state `speaking` then one text card
built in code from the reply text (CARD-08); `turn.ended` with a wire outcome
(D-09), the follow-up window and the playback hint.

Log lines name the event type and the turn id only, never transcript or reply
text.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Callable

from atlas.desktop.cards import build_text_card
from atlas.desktop.display_text import MS_MAX, TRANSCRIPT_TEXT_MAX, sanitize_display_text
from atlas.desktop.hub import DesktopHub
from atlas.desktop.outcomes import wire_outcome
from atlas.desktop.protocol import (
    MSG_CARD,
    MSG_STATE,
    MSG_TRANSCRIPT_FINAL,
    MSG_TRANSCRIPT_PARTIAL,
    MSG_TURN_ENDED,
    MSG_WAKE_CONFIRMED,
    build_transcript_final,
    build_transcript_partial,
    build_turn_ended,
    build_turn_state,
    build_wake_confirmed,
)
from atlas.session.observers import ObserverRegistry

logger = logging.getLogger(__name__)

# One reply card per turn this phase. A later reply.started in the same turn
# replaces it in place on the Mac (UI-SPEC E4).
REPLY_CARD_ID = "reply"
# Extra seconds on top of playback plus the follow-up window, for the camera's
# own end-of-speech detection and the turn.started publish delay.
FOLLOW_UP_SLACK_S = 2.0


class DesktopEventBridge:
    def __init__(
        self,
        registry: ObserverRegistry,
        hub: DesktopHub,
        *,
        follow_up_window_ms: Callable[[], int],
        clock: Callable[[], float] = time.monotonic,
        max_admitted: int = 32,
    ) -> None:
        self._registry = registry
        self._hub = hub
        self._follow_up_window_ms = follow_up_window_ms
        self._clock = clock
        self._max_admitted = max_admitted
        # turn_id -> source name, oldest first. The source name is a label
        # only; it never attributes an event.
        self._admitted: dict[str, str] = {}
        # The window the last admitted turn opened for a follow-up (D-05).
        self._follow_source: str | None = None
        self._follow_until = 0.0
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Subscribe one time and start the reader task. Safe to call once."""
        if self._task is not None:
            return
        self._queue = self._registry.subscribe()
        self._task = asyncio.create_task(self._run(self._queue))

    async def stop(self) -> None:
        """Cancel the reader and unsubscribe."""
        task, queue = self._task, self._queue
        self._task = None
        self._queue = None
        try:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            if queue is not None:
                self._registry.unsubscribe(queue)

    async def _run(self, queue: "asyncio.Queue[dict[str, Any]]") -> None:
        while True:
            event = await queue.get()
            try:
                self.handle(event)
            except Exception:
                logger.exception("desktop bridge: could not handle a %s event", event.get("type"))

    def handle(self, event: dict[str, Any]) -> None:
        """Translate one turn event and broadcast it. Never awaits."""
        turn_id = event.get("turn_id")
        event_type = event.get("type")
        if not isinstance(turn_id, str) or not isinstance(event_type, str):
            return
        if event_type == MSG_WAKE_CONFIRMED:
            self._admit(turn_id, str(event.get("source", "")))
            self._hub.broadcast(build_wake_confirmed(turn_id), frame_type=MSG_WAKE_CONFIRMED)
            return
        if event_type == "turn.started":
            self._on_turn_started(turn_id, event)
            return
        if turn_id not in self._admitted:
            return
        try:
            if event_type == MSG_TRANSCRIPT_PARTIAL:
                self._on_partial(turn_id, event)
            elif event_type == MSG_TRANSCRIPT_FINAL:
                self._on_final(turn_id, event)
            elif event_type == "reply.started":
                self._on_reply_started(turn_id, event)
            elif event_type == MSG_TURN_ENDED:
                self._on_turn_ended(turn_id, event)
        except (TypeError, ValueError):
            # A field of the wrong type or range. The event is dropped; the
            # log names the type and turn id only.
            logger.warning("desktop bridge: dropped a malformed %s event for %s", event_type, turn_id)

    def _on_turn_started(self, turn_id: str, event: dict[str, Any]) -> None:
        if event.get("follow_up") is not True:
            return
        source = event.get("source")
        if (
            not isinstance(source, str)
            or source != self._follow_source
            or self._clock() >= self._follow_until
        ):
            return
        self._admit(turn_id, source)
        self._hub.broadcast(build_turn_state(turn_id, "listening"), frame_type=MSG_STATE)

    def _on_partial(self, turn_id: str, event: dict[str, Any]) -> None:
        text = _text(event)
        if not sanitize_display_text(text, TRANSCRIPT_TEXT_MAX, keep="end"):
            return
        self._hub.broadcast(
            build_transcript_partial(turn_id, text),
            frame_type=MSG_TRANSCRIPT_PARTIAL,
            coalesce_key=turn_id,
        )

    def _on_final(self, turn_id: str, event: dict[str, Any]) -> None:
        text = _text(event)
        self._hub.broadcast(build_transcript_final(turn_id, text), frame_type=MSG_TRANSCRIPT_FINAL)
        self._hub.broadcast(build_turn_state(turn_id, "thinking"), frame_type=MSG_STATE)

    def _on_reply_started(self, turn_id: str, event: dict[str, Any]) -> None:
        text = _text(event)
        self._hub.broadcast(build_turn_state(turn_id, "speaking"), frame_type=MSG_STATE)
        card = build_text_card(turn_id, REPLY_CARD_ID, text)
        if card is not None:
            self._hub.broadcast(card, frame_type=MSG_CARD)

    def _on_turn_ended(self, turn_id: str, event: dict[str, Any]) -> None:
        outcome = event.get("outcome")
        if not isinstance(outcome, str):
            raise TypeError("outcome")
        playback = event.get("playback_ms_left")
        if isinstance(playback, bool) or not isinstance(playback, int):
            raise TypeError("playback_ms_left")
        playback = max(0, min(MS_MAX, playback))
        window_ms = self._follow_up_window_ms()
        asks = event.get("asks_question") is True
        self._hub.broadcast(
            build_turn_ended(turn_id, wire_outcome(outcome), window_ms if asks else 0, playback),
            frame_type=MSG_TURN_ENDED,
        )
        source = self._admitted.pop(turn_id, None)
        if event.get("follow_up") is True:
            # The full window, also when the turn asked nothing: a Phase 13
            # answer window can still carry a reply turn.
            self._follow_until = self._clock() + (playback + window_ms) / 1000 + FOLLOW_UP_SLACK_S
            event_source = event.get("source")
            self._follow_source = event_source if isinstance(event_source, str) else source

    def _admit(self, turn_id: str, source: str) -> None:
        self._admitted.pop(turn_id, None)
        self._admitted[turn_id] = source
        while len(self._admitted) > self._max_admitted:
            self._admitted.pop(next(iter(self._admitted)))


def _text(event: dict[str, Any]) -> str:
    text = event.get("text")
    if not isinstance(text, str):
        raise TypeError("text")
    return text
