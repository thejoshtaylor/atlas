"""`TurnRun`: one turn on a parallel source, as an asyncio task (Phase 12, D-01).

`SourceRunner._run_one_turn` awaits a turn, so the runner reads no frames while
one runs. A `TurnRun` does the same work in its own task: it runs `run_turn_fn`
against its own `TurnFrameSource`, next to a barge-in listener that reads a
second subscription. The runner keeps reading frames and detecting wake words
the whole time.

A `TurnRun` holds itself in its group's live set from `start()` until its task
ends. Every exception is contained and logged, so one turn that fails never
stops the runner or another turn. Every subscription the run opened is closed
in `finally`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from atlas.sources.frame_fanout import FrameSubscription, TurnFrameSource
from atlas.turn.turn_context import TurnContext

if TYPE_CHECKING:
    from atlas.sources.turn_group import TurnGroup

logger = logging.getLogger("atlas.sources.turn_run")


class TurnRun:
    """One turn: its source, its barge-in monitor, and its task."""

    def __init__(
        self,
        group: "TurnGroup",
        turn_source: TurnFrameSource,
        monitor: Any,
        subscription: FrameSubscription,
        context: TurnContext,
        reply_handle: Any | None,
    ) -> None:
        self._group = group
        self._turn_source = turn_source
        self._monitor = monitor
        self._context = context
        self._reply_handle = reply_handle
        self._subscriptions: list[FrameSubscription] = [subscription]
        self.task: "asyncio.Task[None] | None" = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())
        self.task.add_done_callback(lambda _task: self._group.forget(self))

    async def _run(self) -> None:
        try:
            await self._run_stage(self._turn_source, self._monitor)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "source %r: a parallel turn ended in an error -- the other turns and the wake "
                "detector keep running",
                self._group.name,
            )
        finally:
            for subscription in self._subscriptions:
                subscription.close()
            self._finish_reply()
            await self._group.release(self)

    def _finish_reply(self) -> None:
        """Tell the reply handle this turn, follow-ups included, is over."""
        handle, self._reply_handle = self._reply_handle, None
        if handle is None:
            return
        try:
            handle.finish()
        except Exception:
            logger.exception("source %r: a reply handle failed to finish", self._group.name)

    async def _run_stage(self, turn_source: Any, monitor: Any) -> None:
        """One `run_turn_fn` call next to its own barge-in listener, the way
        `SourceRunner._run_one_turn` runs them. The listener is cancelled the
        moment the turn task finishes."""
        hooks = self._group.hooks
        turn_task = asyncio.ensure_future(hooks.run_turn(turn_source))
        listener_task = asyncio.ensure_future(hooks.watch_barge_in(monitor, self._listener_frames()))
        try:
            await turn_task
        finally:
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("source %r: a barge-in listener failed", self._group.name)

    async def _listener_frames(self) -> AsyncIterator[bytes]:
        """The barge-in listener's own subscription. It opens on the first
        read, which is after `transcript_done`, so it holds no backlog."""
        subscription = self._group.fanout.subscribe()
        self._subscriptions.append(subscription)
        try:
            async for chunk in subscription.frames():
                yield chunk
        finally:
            subscription.close()
