"""`DesktopRingRelay`: a ringing timer or alarm on every Mac (Phase 15, D-14,
D-15, D-17).

The scheduler owns the ring. This relay is the scheduler's `on_ring_event`
hook and the hub's `on_timer_stop` callback, and nothing else. It is apart
from the turn bridge on purpose: a ring needs no wake word and no turn.

`on_ring_event` only calls the synchronous `DesktopHub.broadcast`, so a slow
Mac never slows a ring. The ring frame is also kept as a sticky frame, so a Mac
that connects while the ring plays sees it at once. A Mac only asks to stop:
`stop_ring_for` acts only when that timer still rings. Log lines name the timer
id and the device id, never the label.
"""

from __future__ import annotations

import logging
from typing import Callable

from atlas.db.timer_repository import Timer
from atlas.desktop.hub import DesktopHub
from atlas.desktop.protocol import build_timer_ringing, build_timer_stopped

logger = logging.getLogger(__name__)

RING_STICKY_KEY = "ring"

# The wire model is a closed literal of these two kinds.
_KINDS = frozenset({"timer", "alarm"})


class DesktopRingRelay:
    def __init__(
        self, hub: DesktopHub, *, stop_ring_for: Callable[[int], bool] | None = None
    ) -> None:
        self._hub = hub
        self._stop_ring_for = stop_ring_for

    async def on_ring_event(self, ringing: bool, timer: Timer) -> None:
        """The scheduler hook. Never awaits a socket and never raises."""
        try:
            if timer.kind not in _KINDS:
                logger.warning("timer %s has an unknown kind; no ring frame sent", timer.id)
                return
            if ringing:
                self._hub.broadcast(
                    build_timer_ringing(timer.id, timer.kind, timer.label),
                    frame_type="timer.ringing",
                    sticky_key=RING_STICKY_KEY,
                )
            else:
                self._hub.clear_sticky(RING_STICKY_KEY)
                self._hub.broadcast(build_timer_stopped(timer.id), frame_type="timer.stopped")
        except Exception:
            logger.warning("timer %s: could not tell the Macs about the ring", timer.id, exc_info=True)

    def on_timer_stop(self, device_id: int, timer_id: int) -> None:
        """The hub callback for a `timer.stop` frame. Never raises."""
        try:
            stopped = self._stop_ring_for(timer_id) if self._stop_ring_for is not None else False
        except Exception:
            logger.warning(
                "desktop %s asked to stop timer %s: the stop failed", device_id, timer_id, exc_info=True
            )
            return
        logger.warning(
            "desktop %s asked to stop timer %s: %s",
            device_id,
            timer_id,
            "stopped" if stopped else "not ringing",
        )
