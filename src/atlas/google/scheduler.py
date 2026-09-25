"""`GoogleTokenRefreshScheduler`: calls its own refresh callable on a
fixed interval, for the life of the process -- the same `start()`/
`stop()` shape `RetentionScheduler` (`src/atlas/session/retention.py`)
already uses, copied here rather than re-derived: an internal `_stopping`
flag, checked both before and after the interval sleep, and `self._sleep`
injectable (matching `PluginManager`/`WorkflowScheduler`'s own
`sleep=`/`clock=` precedent) so a test drives the whole loop -- one call,
a raising call, the stop -- with no wall-clock wait.

1500 seconds (`interval_s`'s default) is below the 1800-second
`min_validity_s` `GoogleEnvBuilder` asks `GoogleTokenService` for, so a
child's own token always has at least five minutes left when the next
respawn replaces it (09-01-PLAN.md's own Task 3 action text).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Awaitable, Callable

logger = logging.getLogger("atlas.google.scheduler")


class GoogleTokenRefreshScheduler:
    """Calls `refresh()` once per `interval_s`, catching and logging
    whatever it raises so one bad interval never ends the schedule.
    Stops with no further call once `stop()` is awaited."""

    def __init__(
        self,
        refresh: Callable[[], Awaitable[None]],
        *,
        interval_s: float = 1500.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._refresh = refresh
        self._interval_s = interval_s
        self._sleep = sleep
        self._stopping = False
        self._task: "asyncio.Task[None] | None" = None

    def start(self) -> None:
        """Start the schedule -- never awaited by the caller, the same
        fire-and-forget shape `RetentionScheduler.start()` uses."""
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self._refresh()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("google token refresh raised; the schedule continues")
            if self._stopping:
                return
            await self._sleep(self._interval_s)
            if self._stopping:
                return
