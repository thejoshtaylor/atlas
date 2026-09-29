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
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from atlas.audio.ring import bytes_per_ms
from atlas.sources.frame_fanout import FrameFanout, TurnFrameSource
from atlas.sources.turn_run import TurnRun
from atlas.transports.edge import LED_IDLE, LED_LISTENING, MAX_QUEUED_FRAMES
from atlas.turn.follow_up import FollowUpChannel
from atlas.turn.turn_context import TurnContext

if TYPE_CHECKING:
    from atlas.speaker_id.tracker import PartEvent, SpeakerTracker, TurnSpeakerSpan

logger = logging.getLogger("atlas.sources.turn_group")

# Bounds on what the group remembers about parts (T-12-20).
_MAX_WOKEN_SEGMENTS = 8
_MAX_STARTED_PARTS = 64

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
    # A source with no follow-up window leaves these `None` and turns never ask a question.
    follow_up_window_s: Callable[[], float] | None = None
    follow_up_window_opens_at: Callable[[Any], float] | None = None
    follow_up_source: Callable[[Any, float], Any] | None = None


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
        # One follow-up window at a time on a source (D-10).
        self.follow_up_lock = asyncio.Lock()
        # Segment seq -> the wake hit frames in it (D-03). Only a woken segment's parts may start a turn.
        self._woken: dict[int, list[int]] = {}
        # (segment seq, part index) pairs that already started a turn: one turn per part.
        self._started_parts: dict[tuple[int, int], None] = {}
        self._watched: "SpeakerTracker | None" = None
        self._unwatch: Callable[[], None] | None = None
        self._watch_tracker(self._tracker())

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
        wake phrase and what came just before it, with no gap and no repeat.
        The turn's speaker span opens here, at the hit frame, so two hits close
        together never share one wake mark."""
        tracker = self._tracker()
        self._watch_tracker(tracker)
        run = self._start_turn(
            hit_frame_index - self._preroll_frames() + 1,
            span=self._open_span(tracker, hit_frame_index),
            split_part=False,
        )
        if tracker is not None:
            self._note_woken(tracker, hit_frame_index)
            # A late wake detector: parts that appeared before the hit are already known.
            self._consider_parts(tracker.parts_after(hit_frame_index))
        return run

    def start_part_turn(self, event: "PartEvent") -> TurnRun | None:
        """Start a turn for a second voice, with no wake word (D-01). Its audio
        replays from the part's first frame. Each part starts one turn: a part
        that already started one returns `None`. `_on_part` applies the spawn
        rules first."""
        key = (event.segment_seq, event.part_index)
        if key in self._started_parts:
            return None
        self._started_parts[key] = None
        while len(self._started_parts) > _MAX_STARTED_PARTS:
            del self._started_parts[next(iter(self._started_parts))]
        tracker = self._tracker()
        return self._start_turn(
            event.first_frame_index,
            span=self._open_span(tracker, event.first_frame_index),
            split_part=True,
        )

    def _start_turn(self, replay_from: int, *, span: "TurnSpeakerSpan | None", split_part: bool) -> TurnRun:
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
            split_part=split_part,
            speaker_span=span,
            reply_group=reply_handle,
            claims=self._claims,
            replay_until=lambda end, start=order_frame: self.fanout.slice(start, end),
        )
        turn_source = TurnFrameSource(subscription, self._source, preroll_bytes=subscription.replayed_bytes)
        turn_source.barge_in = monitor
        turn_source.turn_context = context
        if self.hooks.follow_up_window_s is not None:
            turn_source.follow_up = FollowUpChannel()
        run = TurnRun(self, turn_source, monitor, subscription, context, reply_handle)
        self._runs.add(run)
        run.start()
        if span is not None and run.task is not None:
            # Backstop: the span leaves the tracker's live list when the turn task ends.
            run.task.add_done_callback(lambda _task, span=span: span.close())
        return run

    # -- parts (D-01 trigger 1, D-02 to D-05) ---------------------------------

    def _speaker(self) -> Any | None:
        provider = self._spec.speaker_context
        if provider is None:
            return None
        try:
            return provider()
        except Exception:
            logger.exception("source %r: the speaker context failed", self.name)
            return None

    def _tracker(self) -> "SpeakerTracker | None":
        return getattr(self._speaker(), "tracker", None)

    def _watch_tracker(self, tracker: "SpeakerTracker | None") -> None:
        """Listen to `tracker` for new parts, once per tracker."""
        if tracker is None or tracker is self._watched:
            return
        if self._unwatch is not None:
            self._unwatch()
        self._watched = tracker
        self._unwatch = tracker.add_part_listener(self._on_part)

    def _open_span(self, tracker: "SpeakerTracker | None", frame_index: int) -> "TurnSpeakerSpan | None":
        if tracker is None:
            return None
        try:
            return tracker.open_turn_at(frame_index)
        except Exception:
            logger.exception("source %r: could not open a speaker span", self.name)
            return None

    def _note_woken(self, tracker: "SpeakerTracker", hit_frame_index: int) -> None:
        seq = tracker.segment_seq_for_frame(hit_frame_index)
        if seq is None:
            return
        self._woken.setdefault(seq, []).append(hit_frame_index)
        while len(self._woken) > _MAX_WOKEN_SEGMENTS:
            del self._woken[next(iter(self._woken))]

    def _on_part(self, event: "PartEvent") -> None:
        self._consider_parts([event])

    def _consider_parts(self, events: "list[PartEvent]") -> None:
        """Apply the spawn rules to each part, in order, and start a turn for
        each part that passes. Drops are logged as counts, never names."""
        dropped: dict[str, int] = {}
        for event in events:
            reason = self._part_drop_reason(event)
            if reason is None and self.at_capacity():
                self.hooks.record_blocked_hit(0.0, BLOCK_TURN_CAP, self.hooks.clock())
                reason = BLOCK_TURN_CAP
            if reason is not None:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            self.start_part_turn(event)
        for reason, count in dropped.items():
            logger.info("source %r: dropped %d speaker part(s): %s", self.name, count, reason)

    def _part_drop_reason(self, event: "PartEvent") -> str | None:
        hits = self._woken.get(event.segment_seq)
        if not hits:
            return "segment_not_woken"  # D-03
        if (event.segment_seq, event.part_index) in self._started_parts:
            return "already_started"
        context = self._speaker()
        tracker = getattr(context, "tracker", None)
        if tracker is None:
            return "no_speaker_tracker"
        # The wake part is read now, over every window embedded so far, so a
        # window that closed after the hit still counts (D-04).
        wake_parts = {index if (index := tracker.part_index_at(hit)) is not None else 0 for hit in hits}
        if event.part_index <= min(wake_parts) or event.part_index in wake_parts:
            return "not_after_wake_part"
        references = getattr(context, "references", None)
        if getattr(context, "mode", None) != "enforce" or references is None or references.enrolled_count <= 0:
            return "speaker_id_not_enforced"  # D-02
        return None

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
        """A turn ended (`TurnRun._run`'s `finally`). The ring goes idle only
        when the last live turn ends (D-11), and the group's claims end with it (D-14)."""
        self._runs.discard(run)
        if not self._runs:
            self._claims = None
            await self._set_led(LED_IDLE)

    def forget(self, run: TurnRun) -> None:
        """Backstop for a task cancelled before it ever started."""
        self._runs.discard(run)

    async def on_admitted(self) -> None:
        """A turn started. The ring shows `listening` only when nothing is
        busier, so a turn that starts while another one replies changes nothing."""
        if getattr(self._source, "led_state", LED_IDLE) == LED_IDLE:
            await self._set_led(LED_LISTENING)

    async def on_follow_up_window(self) -> None:
        """A follow-up window is about to open. The ring shows `listening` only
        when that turn is the only live turn (D-11)."""
        if len(self._runs) == 1:
            await self._set_led(LED_LISTENING)

    async def wait_for_transcripts(self, me: TurnRun) -> bool:
        """Wait until every other live turn has its final transcript or has
        ended, so a follow-up window never opens over another turn's speech
        (D-10). Returns whether it had to wait."""
        waited = False
        while True:
            pending = [run for run in list(self._runs) if run is not me and run.awaiting_transcript]
            if not pending:
                return waited
            waited = True
            await asyncio.gather(*(run.wait_transcribed() for run in pending))

    async def _set_led(self, state: str) -> None:
        """Show `state` on the source's LED ring. A source with no ring is
        skipped. A failure is logged and never reaches a turn."""
        set_led_state = getattr(self._source, "set_led_state", None)
        if set_led_state is None:
            return
        try:
            await set_led_state(state)
        except Exception:
            logger.warning("source %r: could not set the led state %r", self.name, state, exc_info=True)
