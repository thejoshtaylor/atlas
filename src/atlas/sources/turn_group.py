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
import secrets
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from atlas.audio.ring import bytes_per_ms
from atlas.sources.frame_fanout import FrameFanout, TurnFrameSource
from atlas.sources.turn_run import TurnRun
from atlas.transports.edge import MAX_QUEUED_FRAMES
from atlas.turn.turn_context import TurnContext

logger = logging.getLogger("atlas.sources.turn_group")

# `wake_events.block_reason` is free text, so these need no migration.
BLOCK_TURN_CAP = "turn_cap"
BLOCK_REPLY_PLAYING = "reply_playing"


@dataclass(frozen=True)
class ParallelTurns:
    """Turns a source may run at once, and what each one needs.

    `max_concurrent` is the cap (D-05). `preroll_ms` is how much audio before
    the wake hit a turn replays from the fan-out history. `history_frames`
    and `queue_frames` bound the fan-out (T-12-05). `reply_echo_tail_s` is how
    long after a reply's estimated playback end a wake hit is still taken as
    the reply heard back through the microphone.

    `reply_registry.register(turn_key, order_frame, *, group_id)` returns a
    handle with `finish()` and `set_label(label)` (plan 12-05).
    `claims_factory()` returns one registry per group (plan 12-03).
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
    record_blocked_hit: Callable[[float, str | None, float], None]
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
        # A short random token, so a `group_id` stays unique across restarts.
        self._token = secrets.token_hex(3)
        self._group_seq = 0
        self._run_seq = 0
        self._group_id: str | None = None
        self._claims: Any | None = None
        self._reply_quiet_until: float | None = None

    @property
    def source(self) -> Any:
        return self._source

    @property
    def live_count(self) -> int:
        return len(self._runs)

    @property
    def group_id(self) -> str | None:
        """The current group's id: set when the live count goes from 0 to 1,
        and kept after the last turn ends until the next group starts."""
        return self._group_id

    def at_capacity(self) -> bool:
        return len(self._runs) >= self._spec.max_concurrent

    def reply_playing(self, now: float) -> bool:
        """True until the latest reply's estimated playback end plus the echo tail."""
        return self._reply_quiet_until is not None and now < self._reply_quiet_until

    def note_reply_playback(self, ends_at: float | None) -> None:
        """A reply will finish playing at `ends_at` (the runner's clock)."""
        if ends_at is None:
            return
        quiet_until = ends_at + self._spec.reply_echo_tail_s
        if self._reply_quiet_until is None or quiet_until > self._reply_quiet_until:
            self._reply_quiet_until = quiet_until

    def check_admission(self, score: float, now: float) -> bool:
        """Whether an allowed wake hit may start a turn. A hit while reply audio
        plays, or with `max_concurrent` turns live, starts nothing and is
        recorded through the runner with its own block reason (T-12-04, T-12-06)."""
        reason = BLOCK_REPLY_PLAYING if self.reply_playing(now) else BLOCK_TURN_CAP if self.at_capacity() else None
        if reason is None:
            return True
        self.hooks.record_blocked_hit(score, reason, now)
        return False

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
        if not self._runs:
            self._open_group()
        self._run_seq += 1
        turn_key = f"{self.name}:{self._run_seq}"
        first_frame = subscription.first_frame_index
        order_frame = first_frame if first_frame is not None else replay_from
        assert self._group_id is not None
        reply_handle = self._register_reply(turn_key, order_frame)
        context = TurnContext(
            turn_key=turn_key,
            group_id=self._group_id,
            order_frame=order_frame,
            reply_group=reply_handle,
            claims=self._claims,
            replay_until=lambda end, start=order_frame: self.fanout.slice(start, end),
        )
        turn_source = TurnFrameSource(subscription, self._source, preroll_bytes=subscription.replayed_bytes)
        turn_source.barge_in = monitor
        turn_source.turn_context = context
        run = TurnRun(self, turn_source, monitor, subscription, context, reply_handle)
        self._runs.add(run)
        run.start()
        return run

    def _open_group(self) -> None:
        """The live count went from 0 to 1: a new group, and a new claims
        registry that lives until the count returns to 0 (D-14)."""
        self._group_seq += 1
        self._group_id = f"{self._token}-{self._group_seq}"
        factory = self._spec.claims_factory
        self._claims = factory() if factory is not None else None

    def _register_reply(self, turn_key: str, order_frame: int) -> Any | None:
        registry = self._spec.reply_registry
        if registry is None:
            return None
        try:
            return registry.register(turn_key, order_frame, group_id=self._group_id)
        except Exception:
            logger.exception("source %r: could not register a reply handle for %s", self.name, turn_key)
            return None

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
        if not self._runs:
            self._claims = None

    def forget(self, run: TurnRun) -> None:
        """Backstop for a task cancelled before it ever started."""
        self._runs.discard(run)
