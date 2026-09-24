"""`WorkflowScheduler`: the in-process poller, a fourth instance of the
`start()`/`stop()` shape this codebase has now run three times
(`speaker/ffmpeg_supervisor.py`'s `FfmpegSupervisor`,
`session/retention.py`'s `RetentionScheduler`, and this one) -- copied
from `RetentionScheduler`'s own shape exactly (05-CONTEXT.md D-01, D-03):
an explicit `_stopping` flag checked both before and after the interval
sleep, an injectable `clock` and `sleep`, `start()`/`stop()` called from
`lifespan`, and `asyncio.CancelledError` re-raised rather than swallowed.

Unlike `RetentionScheduler`'s `clock: Callable[[], datetime]` reading
wall-clock time directly, and unlike `turn/controller.py`'s silence-
timeout guard (`clock: Callable[[], float] = time.monotonic`), this
scheduler's own `clock` also returns a `datetime` -- `due_at` comparison
needs wall-clock time, not an elapsed-time base, matching
`RetentionScheduler`'s shape rather than `controller.py`'s.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from atlas.config import WorkflowConfig
from atlas.db.repository import WorkflowRepository

logger = logging.getLogger("atlas.workflow.scheduler")


class WorkflowScheduler:
    """Polls `repository.claim_and_execute_next_due_step` every
    `config.poll_interval_s`, draining up to `config.max_steps_per_poll`
    claims per tick before sleeping again.

    `executor` is `execute_step` (`workflow/steps.py`) already bound to a
    tool host, matching `execute_step`'s own `(step, tool_host, config,
    now)` signature minus `tool_host`/`config` -- `functools.partial(
    execute_step, tool_host=..., config=...)` is exactly what `app.py`'s
    `lifespan` hands this constructor, so this class itself only ever
    calls `executor(step, now)`.
    """

    def __init__(
        self,
        repository: WorkflowRepository,
        executor: "Callable[[Any, datetime], Awaitable[Any]]",
        config: WorkflowConfig,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._repository = repository
        self._executor = executor
        self._config = config
        self._clock = clock
        self._sleep = sleep
        self._stopping = False
        self._task: asyncio.Task[None] | None = None
        # Exposed for a test to prove the loop ran, and how many steps it
        # claimed, by counting -- never by timing it (RetentionScheduler's
        # own docstring, extended here).
        self.poll_count = 0
        self.claimed_count = 0

    def start(self) -> None:
        """Start the schedule. Never awaited by the caller -- the loop
        runs for the life of the application, the same fire-and-forget
        shape `RetentionScheduler.start()`/`FfmpegSupervisor.start()` both
        use."""
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _poll_once(self) -> None:
        """One tick: claim due steps until none remain or
        `max_steps_per_poll` have been claimed this tick, whichever comes
        first (T-05-05: bounds one tick so a large backlog cannot
        monopolise the loop). `now` is read once, at the top of the tick,
        matching `sweep_expired_sessions`'s own "one clock reading per
        run" discipline -- a step that comes due mid-drain waits for the
        next tick, never mid-drain."""
        self.poll_count += 1
        now = self._clock()

        async def _row_executor(step: Any) -> Any:
            return await self._executor(step, now)

        for _ in range(self._config.max_steps_per_poll):
            claimed = await self._repository.claim_and_execute_next_due_step(
                _row_executor, now
            )
            if not claimed:
                return
            self.claimed_count += 1

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("workflow poll raised; the schedule continues")
            if self._stopping:
                return
            await self._sleep(self._config.poll_interval_s)
            if self._stopping:
                return
