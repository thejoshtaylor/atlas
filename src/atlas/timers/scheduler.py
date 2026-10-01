"""`TimerScheduler`: the in-process poller that rings timers and alarms.

It has the `start()`/`stop()` shape of `WorkflowScheduler`. Each tick reads
every entry, keeps the list as `snapshot` for the per-turn context block, and
rings each entry that is due. Firing is at-most-once: the row is advanced in
the database first, and the ring plays only if that advance succeeded. An
entry more than `STALE_AFTER_S` late, for example after a restart, is
advanced without speaking.

A ring repeats until somebody stops it with `stop_ringing()`, or until
`max_ring_s` has passed. Each repetition calls `speak` with the
announcement text. In the app, `speak` plays a soft ring tone and ignores the
text, so a ring never speaks (261001-a6l). `speak` returns only after the
room has heard the tone, so the loop paces itself. A short gap follows each repetition. Then the ring goes
quiet by itself, and the timer row is gone, as it is after a stop. A second
entry that becomes due during a ring rings after the first ring ends. The
context `snapshot` does not refresh during a ring. The database advance comes
before any sound, so a ring still fires at most once.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from atlas.db.timer_repository import Timer, TimerRepository
from atlas.timers.core import announcement, next_alarm_at

logger = logging.getLogger("atlas.timers.scheduler")

POLL_INTERVAL_S = 1.0
STALE_AFTER_S = 600.0
RING_MAX_S = 120.0
RING_GAP_S = 1.0


class TimerScheduler:
    def __init__(
        self,
        repository: TimerRepository,
        speak: Callable[[str], Awaitable[None]],
        *,
        zone: Any,
        poll_interval_s: float = POLL_INTERVAL_S,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_ring_s: float = RING_MAX_S,
        ring_gap_s: float = RING_GAP_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._repository = repository
        self._speak = speak
        self._zone = zone
        self._poll_interval_s = poll_interval_s
        self._clock = clock
        self._sleep = sleep
        self._max_ring_s = max_ring_s
        self._ring_gap_s = ring_gap_s
        self._monotonic = monotonic
        self._stopping = False
        self._ring_task: asyncio.Task[None] | None = None
        self._ring_text: str | None = None
        # Set while no ring plays.
        self._ring_idle = asyncio.Event()
        self._ring_idle.set()
        self._task: asyncio.Task[None] | None = None
        # None until the first poll finishes: the context block says
        # "not available" rather than "none are set" before then.
        self.snapshot: "tuple[Timer, ...] | None" = None
        self.poll_count = 0
        self.fired_count = 0

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    @property
    def ringing(self) -> bool:
        task = self._ring_task
        return task is not None and not task.done() and not task.cancelling()

    @property
    def ring_text(self) -> str | None:
        return self._ring_text if self.ringing else None

    def stop_ringing(self) -> bool:
        """End the ring that plays now. True only when a ring was playing."""
        if not self.ringing:
            return False
        assert self._ring_task is not None
        self._ring_task.cancel()
        logger.warning("ring stopped on request")
        return True

    async def wait_ring_over(self) -> None:
        await self._ring_idle.wait()

    def _next_due_at(self, timer: Timer, now: datetime) -> "datetime | None":
        """The next ring of a repeating alarm. None for a timer, a one-time
        alarm, and a repeating alarm when no zone is set (it turns off)."""
        if timer.kind != "alarm" or not timer.repeat_days:
            return None
        if self._zone is None:
            logger.warning(
                "alarm %s repeats but server.timezone is not set; turning it off", timer.id
            )
            return None
        return next_alarm_at(timer.time_of_day or "00:00", timer.repeat_days, now, self._zone)

    async def _poll_once(self) -> None:
        now = self._clock()
        timers = await self._repository.list_timers()
        self.snapshot = tuple(timers)
        self.poll_count += 1
        for timer in timers:
            if not timer.enabled or timer.due_at is None or timer.due_at > now:
                continue
            claimed = await self._repository.advance_fired(
                timer.id, now=now, next_due_at=self._next_due_at(timer, now)
            )
            if not claimed:
                continue
            self.fired_count += 1
            late_s = (now - timer.due_at).total_seconds()
            if late_s > STALE_AFTER_S:
                logger.warning(
                    "timer %s was %.0f s late; cleared without ringing", timer.id, late_s
                )
                continue
            # Warning level, because the deployment shows no info logs.
            logger.warning("%s %s is ringing", timer.kind, timer.id)
            await self._ring(timer, announcement(timer))

    async def _ring_loop(self, text: str, started: float) -> None:
        while True:
            await self._speak(text)
            if self._monotonic() - started >= self._max_ring_s:
                return
            await self._sleep(self._ring_gap_s)

    async def _ring(self, timer: Timer, text: str) -> None:
        """Ring until the loop ends, `stop_ringing()` is called or this task is cancelled."""
        self._ring_idle.clear()
        self._ring_text = text
        task = asyncio.create_task(self._ring_loop(text, self._monotonic()))
        self._ring_task = task
        try:
            await asyncio.wait({task})
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            self._ring_task = None
            self._ring_text = None
            self._ring_idle.set()
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.exception("ringing timer %s failed", timer.id, exc_info=error)
        else:
            logger.warning("timer %s went quiet after max_ring_s=%.0f", timer.id, self._max_ring_s)

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("timer poll raised; the schedule continues")
            if self._stopping:
                return
            await self._sleep(self._poll_interval_s)
            if self._stopping:
                return
