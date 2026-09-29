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
from atlas.turn.follow_up import MAX_CHAINED_FOLLOW_UPS, FollowUpChannel
from atlas.turn.turn_context import TurnContext

if TYPE_CHECKING:
    from atlas.sources.turn_group import TurnGroup

logger = logging.getLogger("atlas.sources.turn_run")


class _Stage:
    """One turn's barge-in monitor and its end: the two facts another turn's
    follow-up window waits on."""

    def __init__(self, monitor: Any) -> None:
        self.monitor = monitor
        self.over = asyncio.Event()


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
        # Set from construction, so no other turn sees this one as "not waiting
        # on a transcript" in the gap before its task starts.
        self._stage: _Stage | None = _Stage(monitor)

    @property
    def awaiting_transcript(self) -> bool:
        """True while a turn of this run is live and has no final transcript.
        A run between turns (waiting for its follow-up window) is not."""
        stage = self._stage
        return stage is not None and not stage.monitor.transcript_done.is_set()

    async def wait_transcribed(self) -> None:
        """Return once the current turn has its final transcript or has ended."""
        stage = self._stage
        if stage is None or stage.monitor.transcript_done.is_set():
            return
        waiters = [
            asyncio.ensure_future(stage.monitor.transcript_done.wait()),
            asyncio.ensure_future(stage.over.wait()),
        ]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())
        self.task.add_done_callback(lambda _task: self._group.forget(self))

    async def _run(self) -> None:
        try:
            await self._group.on_admitted()
            await self._run_stage(self._turn_source, self._stage)
            channel = getattr(self._turn_source, "follow_up", None)
            if channel is not None:
                await self._run_follow_ups(channel)
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

    async def _run_follow_ups(self, channel: FollowUpChannel) -> None:
        """After a turn left a request on `channel`, open one no-wake-word
        window per request in the chain, the way `SourceRunner._run_follow_ups`
        does, with three changes for parallel turns (D-10). One window is open
        on the source at a time (the group's lock). A window opens only after
        every other live turn has its final transcript. Each follow-up turn
        gets a context that names the asking turn's speaker."""
        group = self._group
        hooks = group.hooks
        assert hooks.follow_up_window_s is not None and hooks.follow_up_window_opens_at is not None
        assert hooks.follow_up_source is not None
        while channel.requested is not None and channel.requested.chain_depth <= MAX_CHAINED_FOLLOW_UPS:
            requested = channel.requested
            async with group.follow_up_lock:
                waited = await group.wait_for_transcripts(self)
                opens_at = hooks.follow_up_window_opens_at(requested)
                if waited:
                    # The wait used up part of the window. It opens now, not in the past.
                    opens_at = max(opens_at, hooks.clock())
                subscription = group.fanout.subscribe()
                self._subscriptions.append(subscription)
                follow_up_source = hooks.follow_up_source(TurnFrameSource(subscription, group.source), opens_at)
                monitor = hooks.new_monitor()
                follow_up_source.barge_in = monitor
                new_channel = FollowUpChannel(
                    incoming=requested, window_opens_at=opens_at, window_s=hooks.follow_up_window_s()
                )
                follow_up_source.follow_up = new_channel
                self._context = _follow_up_context(self._context)
                follow_up_source.turn_context = self._context
                await group.on_follow_up_window()
                await self._run_stage(follow_up_source, _Stage(monitor))
            channel = new_channel

    async def _run_stage(self, turn_source: Any, stage: _Stage | None) -> None:
        """One `run_turn_fn` call next to its own barge-in listener, the way
        `SourceRunner._run_one_turn` runs them. The listener is cancelled the
        moment the turn task finishes."""
        assert stage is not None
        self._stage = stage
        hooks = self._group.hooks
        turn_task = asyncio.ensure_future(hooks.run_turn(turn_source))
        listener_task = asyncio.ensure_future(hooks.watch_barge_in(stage.monitor, self._listener_frames()))
        try:
            await turn_task
        finally:
            stage.over.set()
            self._stage = None
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


def _follow_up_context(asking: TurnContext) -> TurnContext:
    """The context of the follow-up turn that answers `asking`'s question. It
    keeps the turn key, the group, the claims, and the reply handle. Only the
    speaker who asked may answer (D-10). A speaker the asking turn did not
    note carries the earlier limit forward."""
    return TurnContext(
        turn_key=asking.turn_key,
        group_id=asking.group_id,
        order_frame=asking.order_frame,
        follow_up=True,
        reply_group=asking.reply_group,
        claims=asking.claims,
        answer_only_from=asking.speaker_id or asking.answer_only_from,
        replay_until=asking.replay_until,
    )
