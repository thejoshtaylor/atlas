"""`ParallelTurns` and `TurnGroup`: several live turns on one edge source (Phase 12).

A source built with a `ParallelTurns` spec runs each wake turn as a `TurnRun`
task (`sources/turn_run.py`) while `SourceRunner.run()` keeps reading frames.
`SourceRunner.run()` is still the only caller of `source.frames()`: it pushes
every chunk into the group's `FrameFanout`, and each turn reads its own queue
that starts at its own replay frame.

A source without the spec (the camera, the browser listener) never builds a
`TurnGroup` and runs the serial path in `sources/runner.py` unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from atlas.audio.ring import bytes_per_ms
from atlas.sources.frame_fanout import FrameFanout, TurnFrameSource
from atlas.sources.turn_run import TurnRun
from atlas.transports.edge import MAX_QUEUED_FRAMES

logger = logging.getLogger("atlas.sources.turn_group")


@dataclass(frozen=True)
class ParallelTurns:
    """Turns a source may run at once, and what each one needs.

    `max_concurrent` is the cap (D-05). `preroll_ms` is how much audio before
    the wake hit a turn replays from the fan-out history. `history_frames`
    and `queue_frames` bound the fan-out (T-12-05).
    """

    max_concurrent: int
    preroll_ms: int = 0
    reply_registry: Any | None = None
    claims_factory: Callable[[], Any] | None = None
    speaker_context: Callable[[], Any] | None = None
    history_frames: int = MAX_QUEUED_FRAMES
    queue_frames: int = MAX_QUEUED_FRAMES
    reply_echo_tail_s: float = 0.8


@dataclass(frozen=True)
class TurnHooks:
    """The runner's own methods a group calls back into. `SourceRunner` builds
    this, so `turn_group.py` never imports `sources/runner.py`."""

    run_turn: Callable[[Any], Awaitable[None]]
    new_monitor: Callable[[], Any]
    watch_barge_in: Callable[..., Awaitable[None]]
    clock: Callable[[], float]


class TurnGroup:
    """The live turns on one source, and the one fan-out they read from."""

    def __init__(self, name: str, source: Any, spec: ParallelTurns, hooks: TurnHooks) -> None:
        self.name = name
        self.hooks = hooks
        self.fanout = FrameFanout(history_frames=spec.history_frames, queue_frames=spec.queue_frames)
        self._source = source
        self._spec = spec
        self._runs: set[TurnRun] = set()

    @property
    def live_count(self) -> int:
        return len(self._runs)

    def push_frame(self, chunk: bytes, frame_index: int | None) -> None:
        """`SourceRunner.run()` calls this for every chunk, before detection."""
        self.fanout.push(chunk, frame_index)

    def start_wake_turn(self, hit_frame_index: int) -> TurnRun:
        """Start a turn for a wake hit on frame `hit_frame_index`. Its queue
        starts at the hit frame minus the preroll frames, so the turn hears the
        wake phrase and what came just before it, with no gap and no repeat."""
        replay_from = hit_frame_index - self._preroll_frames() + 1
        monitor = self.hooks.new_monitor()
        subscription = self.fanout.subscribe(replay_from=replay_from)
        turn_source = TurnFrameSource(subscription, self._source, preroll_bytes=subscription.replayed_bytes)
        turn_source.barge_in = monitor
        run = TurnRun(self, turn_source, monitor, subscription)
        self._runs.add(run)
        run.start()
        return run

    def _preroll_frames(self) -> int:
        """How many whole chunks fit the configured preroll, the same byte
        bound `PrerollBuffer` applies on the serial path."""
        chunk_bytes = self.fanout.last_chunk_bytes
        if self._spec.preroll_ms <= 0 or chunk_bytes <= 0:
            return 0
        window_bytes = round(bytes_per_ms(self._source.source_format()) * self._spec.preroll_ms)
        return window_bytes // chunk_bytes

    def close_frames(self) -> None:
        """The source has no more frames: every turn's queue ends after what
        it already holds."""
        self.fanout.close()

    async def drain(self, *, cancel: bool = False) -> None:
        """Wait for every live turn. With `cancel`, cancel them first (the
        runner's own task was cancelled)."""
        tasks = [run.task for run in self._runs if run.task is not None]
        if cancel:
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def release(self, run: TurnRun) -> None:
        """A turn ended (`TurnRun._run`'s `finally`)."""
        self._runs.discard(run)

    def forget(self, run: TurnRun) -> None:
        """Backstop for a task cancelled before it ever started."""
        self._runs.discard(run)
