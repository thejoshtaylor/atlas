"""`TimerScheduler`: the in-process poller that rings timers and alarms.

It has the `start()`/`stop()` shape of `WorkflowScheduler`. Each tick reads
every entry, keeps the list as `snapshot` for the per-turn context block, and
rings each entry that is due. Firing is at-most-once: the row is advanced in
the database first, and the ring plays only if that advance succeeded. An
entry more than `STALE_AFTER_S` late, for example after a restart, is
advanced without speaking.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from atlas.db.timer_repository import Timer, TimerRepository
from atlas.timers.core import announcement

logger = logging.getLogger("atlas.timers.scheduler")

POLL_INTERVAL_S = 1.0
STALE_AFTER_S = 600.0
RING_REPEATS = 3


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
    ) -> None:
        self._repository = repository
        self._speak = speak
        self._zone = zone
        self._poll_interval_s = poll_interval_s
        self._clock = clock
        self._sleep = sleep
        self._stopping = False
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

    def _next_due_at(self, timer: Timer, now: datetime) -> "datetime | None":
        return None

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
            text = announcement(timer)
            try:
                for _ in range(RING_REPEATS):
                    await self._speak(text)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ringing timer %s failed", timer.id)

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
