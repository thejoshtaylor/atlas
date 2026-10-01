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

Events are attributed by `turn_id`, never by source name, because turn groups
run several turns at once on one source (RESEARCH Pitfall 1).

This task translates `wake.confirmed` only. Plan 15-05 adds the other turn
events and the follow-up admission. `follow_up_window_ms` and `clock` are
constructor arguments now so that plan does not change the signature.

Log lines name the event type and the turn id only, never transcript text.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Callable

from atlas.desktop.hub import DesktopHub
from atlas.desktop.protocol import MSG_WAKE_CONFIRMED, build_wake_confirmed
from atlas.session.observers import ObserverRegistry

logger = logging.getLogger(__name__)


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
        if not isinstance(turn_id, str):
            return
        if event.get("type") == MSG_WAKE_CONFIRMED:
            self._admit(turn_id, str(event.get("source", "")))
            self._hub.broadcast(build_wake_confirmed(turn_id), frame_type=MSG_WAKE_CONFIRMED)
            return
        if turn_id not in self._admitted:
            return
        # Plan 15-05 translates the other turn events of an admitted turn.

    def _admit(self, turn_id: str, source: str) -> None:
        self._admitted.pop(turn_id, None)
        self._admitted[turn_id] = source
        while len(self._admitted) > self._max_admitted:
            self._admitted.pop(next(iter(self._admitted)))
