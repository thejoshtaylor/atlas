"""`LoopStallReporter`: logs the event loop's own stack when it stops
answering, from a thread that is not itself the loop -- the only vantage
point that can still observe a stall while the loop thread is the one
stalled (D1, quick task 260924-4is).

`start()` reads `sys._current_frames()` for the loop thread's own id, and
formats it with `traceback.format_stack` -- code locations and source
text only, never `f_locals` (T-4is-02): a transcript, a token, or an
entity id must never reach this log.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import traceback

logger = logging.getLogger("atlas.loop_stall")

_THREAD_NAME = "atlas-loop-stall"


class LoopStallReporter:
    """A daemon thread that watches one event loop's heartbeat and logs a
    WARNING, with the loop thread's own stack, when it goes quiet for
    `threshold_s` or more."""

    def __init__(self, threshold_s: float = 2.0, poll_s: float = 0.5) -> None:
        self._threshold_s = threshold_s
        self._poll_s = poll_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread_id: int | None = None
        self._last_beat = 0.0
        self._reported = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Call on the loop thread. Stores the loop and this thread's id,
        then starts the watcher thread."""
        self._loop = asyncio.get_running_loop()
        self._loop_thread_id = threading.get_ident()
        self._last_beat = time.monotonic()
        self._reported = False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch, name=_THREAD_NAME, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the watcher thread. Safe to call even if `start()` never
        ran, or if the loop is already closed."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_s * 4)

    def _beat(self) -> None:
        """Runs on the loop thread, posted by the watcher via
        `call_soon_threadsafe`. Proof the loop is still scheduling
        callbacks."""
        self._last_beat = time.monotonic()

    def _watch(self) -> None:
        loop = self._loop
        while not self._stop_event.is_set():
            self._stop_event.wait(self._poll_s)
            if self._stop_event.is_set():
                return
            try:
                loop.call_soon_threadsafe(self._beat)
            except RuntimeError:
                # The loop is closed -- nothing left to watch.
                return
            gap = time.monotonic() - self._last_beat
            if gap >= self._threshold_s and not self._reported:
                self._reported = True
                stack = self._format_loop_stack()
                logger.warning(
                    "event loop blocked for %.1fs (threshold %.1fs); loop thread stack:\n%s",
                    gap,
                    self._threshold_s,
                    stack,
                )
            elif gap < self._threshold_s and self._reported:
                self._reported = False
                logger.warning("event loop unblocked after %.1fs", gap)

    def _format_loop_stack(self) -> str:
        frame = sys._current_frames().get(self._loop_thread_id) if self._loop_thread_id is not None else None
        if frame is None:
            return "<loop thread frame unavailable>"
        return "".join(traceback.format_stack(frame))
