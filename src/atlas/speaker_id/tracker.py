"""Streaming speaker tracking across one edge source's whole life (D-09, D-12).

`SpeakerTracker` implements `EdgeAudioListener` (`transports/edge.py`) --
this is the answer to 11-RESEARCH.md Pitfall 2: it never opens a second
read of `EdgeAudioSource.frames()`. `SourceRunner.run()` stays the sole
reader of `frames()` for the process's life; the tracker only listens to
the same frames and `vad.start`/`vad.end`/wake-hit notifications `serve()`
already routes to every attached `EdgeAudioListener`, via
`EdgeAudioSource.add_listener`.

The tracker runs continuously, independent of any turn -- embeddings for a
segment's speech start accumulating the moment `vad.start` arrives, well
before a wake hit (if one ever comes) is known. This is what makes D-16's
"less than 100ms after vad_end" reachable: by the time a turn calls
`decide()`, most or all of the segment's embedding work has already
finished in the background.

A **tracker segment** is not the same thing as the `SpeechWindowAccumulator`'s
own segment (`speaker_id/windows.py`): a tracker segment runs from one
`vad.start` until the *next* `vad.start`, so its frame-index range includes
the tail frames after `vad.end` -- Vosk (the wake detector) often commits
the wake hit during that tail, and `open_turn()` needs the tail's frame
range to find the right segment for it. Only frames between `vad.start`
and `vad.end` ever reach the accumulator itself (D-08): tail frames extend
a tracker segment's own frame-index bookkeeping but are never windowed or
embedded.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from atlas.audio.channels import select_channel
from atlas.speaker_id.change_point import find_change_points, part_index_for_frame, split_parts
from atlas.speaker_id.matching import MatchResult, ReferenceSet, mean_embedding
from atlas.speaker_id.windows import SpeechWindow, SpeechWindowAccumulator

if TYPE_CHECKING:
    from atlas.speaker_id.embedding import EmbeddingWorker

logger = logging.getLogger("atlas.speaker_id.tracker")

# Far above the ~100ms D-16 budget, so a wedged `EmbeddingWorker` (its one
# thread stuck on a call that never returns) cannot hang a turn -- `decide()`
# gives up and reports `"decision_timeout"` instead of waiting forever.
SPEAKER_DECISION_TIMEOUT_S = 1.0

# Bounded history, oldest dropped first -- a tracker that ran for days must
# never grow without bound.
_MAX_SEGMENTS = 32
_MAX_WINDOWS_PER_SEGMENT = 240
# Live spans (one per running turn, Phase 12). Oldest dropped, so a turn that
# never closed its span cannot grow this list without bound.
_MAX_LIVE_SPANS = 8


@dataclass(eq=False)
class _PendingWindow:
    """One window already submitted to the `EmbeddingWorker`. `eq=False`
    (identity comparison) matters here for the same reason it matters on
    `_Segment` below -- this object is looked up by identity, never by
    field equality."""

    window: SpeechWindow
    future: "asyncio.Future[Any]"
    completed_at: "float | None" = None


@dataclass(frozen=True)
class PartEvent:
    """A new part (index 1 or more) of one tracker segment (Phase 12), from
    the part's first window. Emitted once per part to every part listener."""

    segment_seq: int
    part_index: int
    first_frame_index: int
    started_at: float


@dataclass(eq=False)
class _Segment:
    """One tracker segment (see module docstring). `eq=False`: `_Segment`
    instances are looked up by identity (`list.index`, `is`), never by
    field equality -- two segments could otherwise compare equal by
    accident right after `vad.start`, before either has any windows yet."""

    seq: int
    windows: "list[_PendingWindow]" = field(default_factory=list)
    in_speech: bool = True
    first_frame_index: "int | None" = None
    last_frame_index: "int | None" = None
    # Parts 1 and up already emitted to the part listeners, in order.
    parts: "list[PartEvent]" = field(default_factory=list)


@dataclass(frozen=True)
class DroppedPart:
    """One part of a segment that D-12's change detection split off from
    the kept part -- recorded as a `speaker.split` event
    (`speaker_id/turn_gate.py`) and then dropped: never a second turn
    (Phase 12), never aggregated into `decide()`'s own match.

    `first_frame_index`/`last_frame_index` mirror `SpeechWindow`'s own
    fields, spanning the part's own first and last window. `match` is
    `None` when `decide()` was given no `references` to score against.
    """

    first_frame_index: int
    last_frame_index: int
    started_at: float
    speech_ms: float
    match: "MatchResult | None"


@dataclass(frozen=True)
class SpeakerMeasurement:
    """One `TurnSpeakerSpan.decide()` result.

    `detail` is `None` for a real measurement (whether or not it later
    passes the gate's threshold), `"no_speech_measured"` when no window
    could be embedded at all, and `"decision_timeout"` when the worker did
    not finish within `timeout_s`. `speaker_id_ms` is `None` without an
    `end_of_speech_at` to measure from.

    `dropped_parts` (D-12) is every part of the span's windows that is not
    the kept part -- empty for a span whose windows never crossed
    `change_similarity_floor`, which is every measurement before plan
    11-06 and every single-voice segment after it.
    """

    match: "MatchResult | None"
    speech_ms: float
    window_count: int
    ready_at: "float | None"
    speaker_id_ms: "float | None"
    detail: "str | None"
    dropped_parts: "tuple[DroppedPart, ...]" = ()


def _collect_valid_windows(
    segments: "list[_Segment]", start_after: "float | None"
) -> "list[_PendingWindow]":
    """Every window in `segments` (in window order -- segments are
    chronological and a segment's own `windows` list is append-order),
    whose embedding future has already resolved successfully. Filtered by
    `start_after` when given (D-11: a follow-up span's own window opens
    inside an already-running segment, so an earlier window in the same
    segment does not belong to it). Shared by `TurnSpeakerSpan._check_for_split`
    (live, partial) and `decide()` (final, complete).
    """
    valid: "list[_PendingWindow]" = []
    for segment in segments:
        for pending in segment.windows:
            if start_after is not None and pending.window.started_at < start_after:
                continue
            future = pending.future
            if not future.done() or future.cancelled() or future.exception() is not None:
                continue
            valid.append(pending)
    return valid


def _kept_part_index(
    parts: "list[range]", windows: "list[SpeechWindow]", wake_frame_index: "int | None"
) -> int:
    """D-12: the part that decides the turn. For a wake turn
    (`wake_frame_index` not `None`), the part holding the wake frame index
    (`part_index_for_frame`). For a follow-up turn, the first part --
    `start_after` has already trimmed away everything before it."""
    if wake_frame_index is None:
        return 0
    return part_index_for_frame(parts, windows, wake_frame_index)


class TurnSpeakerSpan:
    """A view into the tracker's own live segment history, from a turn's
    own starting point onward -- not a snapshot. A wake turn's span (no
    `start_after`) is anchored to a specific `_Segment` object found at
    `open_turn()` time; a follow-up turn's span (`start_after` given) is
    anchored to a timestamp and re-filters every segment at `decide()`
    time, since D-11's follow-up window can span more than one segment
    that did not exist yet when the span opened.

    `split_event` (D-12) is set at most once, by `_check_for_split` --
    called from the tracker's own window-done callback each time one more
    of this span's windows finishes embedding, so a second voice's turn
    can end without waiting for `decide()` to be awaited at all.
    `split_frame_index`: first frame of the part after the kept part.
    """

    def __init__(
        self,
        tracker: "SpeakerTracker",
        *,
        start_segment: "_Segment | None" = None,
        start_after: "float | None" = None,
        wake_frame_index: "int | None" = None,
    ) -> None:
        self._tracker = tracker
        self._start_segment = start_segment
        self._start_after = start_after
        self._wake_frame_index = wake_frame_index
        self.split_event: "asyncio.Event" = asyncio.Event()
        self.split_detected_at: "float | None" = None
        self.split_frame_index: "int | None" = None
        self._closed = False

    def close(self) -> None:
        """Stop live detection. Safe to call twice; `decide()` still works."""
        self._closed = True
        self._tracker._drop_span(self)

    def _segments(self) -> "list[_Segment]":
        if self._start_after is not None:
            return list(self._tracker._segments)
        if self._start_segment is None:
            return []
        try:
            index = self._tracker._segments.index(self._start_segment)
        except ValueError:
            # The starting segment has already aged out of the bounded
            # history (`_MAX_SEGMENTS`) -- nothing left to measure from.
            return []
        return self._tracker._segments[index:]

    def _check_for_split(self, now: "float") -> None:
        """D-12 live detection. Called from `SpeakerTracker._submit_window`'s
        own window-done callback -- never a separate task (module docstring).
        Recomputes change points over every window this span holds that has
        already resolved, and sets `split_event` the first time a part later
        than the kept part exists. A change entirely before the kept part
        (someone spoke before the wake phrase) never sets it, because
        nothing after the wake phrase needs to stop -- `kept_index` already
        being the LAST part is exactly that case.
        """
        if self._closed or self.split_event.is_set():
            return
        valid = _collect_valid_windows(self._segments(), self._start_after)
        if len(valid) < 2:
            # Fewer than two embeddings: `find_change_points` cannot report
            # a change, and there is nothing to compare against yet.
            return
        embeddings = [pending.future.result() for pending in valid]
        change_points = find_change_points(
            embeddings, similarity_floor=self._tracker.change_similarity_floor
        )
        if not change_points:
            return
        windows = [pending.window for pending in valid]
        parts = split_parts(len(valid), change_points)
        kept_index = _kept_part_index(parts, windows, self._wake_frame_index)
        if kept_index < len(parts) - 1:
            self.split_frame_index = windows[parts[kept_index + 1][0]].first_frame_index
            self.split_event.set()
            self.split_detected_at = now

    async def decide(
        self,
        *,
        end_of_speech_at: "float | None",
        references: "ReferenceSet | None",
        timeout_s: float = SPEAKER_DECISION_TIMEOUT_S,
    ) -> SpeakerMeasurement:
        segments = self._segments()

        current = self._tracker._current_segment
        if segments and current is not None and segments[-1] is current and current.in_speech:
            # The latest span segment is still mid-speech (this turn's own
            # `vad.end` has not arrived yet, or this is a follow-up span
            # whose window opened inside an already-running segment) --
            # flush its partial window rather than losing it.
            flushed = self._tracker._accumulator.flush(self._tracker._clock())
            if flushed is not None:
                self._tracker._submit_window(current, flushed)

        pending_windows: "list[_PendingWindow]" = []
        for segment in segments:
            for pending in segment.windows:
                if self._start_after is not None and pending.window.started_at < self._start_after:
                    continue
                pending_windows.append(pending)

        if not pending_windows:
            return SpeakerMeasurement(
                match=None, speech_ms=0.0, window_count=0, ready_at=None, speaker_id_ms=None,
                detail="no_speech_measured",
            )

        futures = [p.future for p in pending_windows]
        done, still_pending = await asyncio.wait(futures, timeout=timeout_s)
        if still_pending:
            # The worker (a single thread) has not finished every window's
            # embedding within the budget -- treat the whole measurement as
            # unavailable rather than aggregating a partial, possibly
            # misleading result.
            return SpeakerMeasurement(
                match=None, speech_ms=0.0, window_count=0, ready_at=None, speaker_id_ms=None,
                detail="decision_timeout",
            )

        valid = [
            pending
            for pending in pending_windows
            # Skip a window whose embedding raised (or was cancelled) --
            # the rest of the turn's speech still measures something.
            if not pending.future.cancelled() and pending.future.exception() is None
        ]

        if not valid:
            return SpeakerMeasurement(
                match=None, speech_ms=0.0, window_count=0, ready_at=None, speaker_id_ms=None,
                detail="no_speech_measured",
            )

        embeddings = [pending.future.result() for pending in valid]
        windows = [pending.window for pending in valid]
        # D-12: the same split, over the same (now complete) window history
        # `_check_for_split` already watched live -- a segment with exactly
        # one voice throughout produces no change points and one part, the
        # whole span kept, `dropped_parts` empty, byte-identical to every
        # measurement before this plan.
        change_points = find_change_points(
            embeddings, similarity_floor=self._tracker.change_similarity_floor
        )
        parts = split_parts(len(valid), change_points)
        kept_index = _kept_part_index(parts, windows, self._wake_frame_index)

        dropped_parts: "list[DroppedPart]" = []
        for part_index, part_range in enumerate(parts):
            if part_index == kept_index:
                continue
            part_windows = [windows[i] for i in part_range]
            part_weights = [w.speech_ms for w in part_windows]
            part_embedding = mean_embedding([embeddings[i] for i in part_range], part_weights)
            part_match = references.match(part_embedding) if references is not None else None
            dropped_parts.append(
                DroppedPart(
                    first_frame_index=part_windows[0].first_frame_index,
                    last_frame_index=part_windows[-1].last_frame_index,
                    started_at=part_windows[0].started_at,
                    speech_ms=sum(part_weights),
                    match=part_match,
                )
            )

        kept_range = parts[kept_index]
        kept_windows = [windows[i] for i in kept_range]
        kept_weights = [w.speech_ms for w in kept_windows]
        embedding = mean_embedding([embeddings[i] for i in kept_range], kept_weights)
        match_start = self._tracker._clock()
        match = references.match(embedding) if references is not None else None
        match_elapsed_s = self._tracker._clock() - match_start

        kept_completed_ats = [
            valid[i].completed_at for i in kept_range if valid[i].completed_at is not None
        ]
        ready_at = max(kept_completed_ats) if kept_completed_ats else None
        speaker_id_ms: "float | None" = None
        if end_of_speech_at is not None and ready_at is not None:
            speaker_id_ms = (max(0.0, ready_at - end_of_speech_at) + match_elapsed_s) * 1000.0

        return SpeakerMeasurement(
            match=match,
            speech_ms=sum(kept_weights),
            window_count=len(kept_windows),
            ready_at=ready_at,
            speaker_id_ms=speaker_id_ms,
            detail=None,
            dropped_parts=tuple(dropped_parts),
        )


class SpeakerTracker:
    """Implements `EdgeAudioListener` (`transports/edge.py`). Attached once,
    at wiring time (`speaker_id/wiring.py`), for the whole edge source's
    life -- never per turn.

    `change_similarity_floor` (D-12) is the floor `TurnSpeakerSpan` reads
    for both its live detection (`_check_for_split`) and its final split
    (`decide()`) -- stored here since plan 11-04 so this plan's own wiring
    needed no change to `wiring.py`.
    """

    def __init__(
        self,
        *,
        channels: int,
        asr_channel: int,
        sample_rate: int,
        window_ms: int,
        min_window_ms: int,
        speech_rms_floor: float,
        change_similarity_floor: float,
        worker: "EmbeddingWorker",
        clock: "Any" = time.monotonic,
    ) -> None:
        self._channels = channels
        self._asr_channel = asr_channel
        self.change_similarity_floor = change_similarity_floor
        self._worker = worker
        self._clock = clock
        self._accumulator = SpeechWindowAccumulator(
            sample_rate=sample_rate,
            window_ms=window_ms,
            min_window_ms=min_window_ms,
            speech_rms_floor=speech_rms_floor,
        )
        self._segments: "list[_Segment]" = []
        self._current_segment: "_Segment | None" = None
        self._wake_mark_frame_index: "int | None" = None
        # D-12: every open span, so live detection reports to each running
        # turn (Phase 12 runs several at once). `TurnSpeakerSpan.close()` leaves it.
        self._live_spans: "list[TurnSpeakerSpan]" = []
        self._part_listeners: "list[Callable[[PartEvent], None]]" = []

    # -- EdgeAudioListener protocol -----------------------------------

    def on_vad_start(self, seq: int, at: float) -> None:
        del at  # unused -- the accumulator stamps its own window timestamps.
        segment = _Segment(seq=seq)
        self._accumulator.start_segment(seq)
        self._segments.append(segment)
        if len(self._segments) > _MAX_SEGMENTS:
            self._segments.pop(0)
        self._current_segment = segment

    def on_frame(self, chunk: bytes, frame_index: int, at: float) -> None:
        segment = self._current_segment
        if segment is None:
            # A frame that arrived before this tracker ever saw a
            # `vad.start` (a listener attached mid-connection, or a stray
            # frame outside any segment) -- nothing to attribute it to.
            return
        if segment.first_frame_index is None:
            segment.first_frame_index = frame_index
        segment.last_frame_index = frame_index

        if not segment.in_speech:
            # Tail frames (after this segment's own `vad.end`, before the
            # next `vad.start`) extend the segment's frame-index range for
            # `open_turn()` to find, but D-08 keeps them out of the
            # accumulator entirely.
            return

        asr_chunk = select_channel(chunk, self._channels, self._asr_channel)
        window = self._accumulator.push(asr_chunk, frame_index, at)
        if window is not None:
            self._submit_window(segment, window)

    def on_vad_end(self, seq: int, at: float) -> None:
        del seq, at
        segment = self._current_segment
        if segment is None:
            return
        window = self._accumulator.end_segment()
        if window is not None:
            self._submit_window(segment, window)
        segment.in_speech = False
        # Deliberately NOT cleared: `segment` keeps receiving tail frames
        # (module docstring) until the next `vad.start` replaces it.

    def on_wake_hit(self, frame_index: int) -> None:
        self._wake_mark_frame_index = frame_index

    # -- turn API -------------------------------------------------------

    def open_turn(self, *, start_after: "float | None" = None) -> TurnSpeakerSpan:
        """Open a span for one turn.

        `start_after=None` (a wake turn): consumes the last wake mark
        (`on_wake_hit`) and anchors the span to the segment whose frame
        range holds that index -- the most recent segment when there is no
        mark. `start_after=t` (a follow-up turn, D-11): anchors the span to
        every window, in any segment now or later, whose `started_at >= t`.

        D-12: the returned span joins the live list either way --
        `_submit_window`'s own window-done callback reports every newly
        finished window to it, live, until `span.close()`.
        """
        if start_after is not None:
            span = TurnSpeakerSpan(self, start_after=start_after)
        else:
            frame_index = self._wake_mark_frame_index
            self._wake_mark_frame_index = None
            start_segment = self._segment_for_frame(frame_index)
            span = TurnSpeakerSpan(self, start_segment=start_segment, wake_frame_index=frame_index)
        self._add_span(span)
        return span

    def open_turn_at(self, frame_index: int) -> TurnSpeakerSpan:
        """Open a span for the part that holds `frame_index`: a wake turn's hit
        frame or a part turn's first frame. It reads no wake mark, so two hits
        close together never share one (Phase 12). A split that already exists
        sets `split_event` at once."""
        span = TurnSpeakerSpan(
            self, start_segment=self._segment_for_frame(frame_index), wake_frame_index=frame_index
        )
        self._add_span(span)
        span._check_for_split(self._clock())
        return span

    def add_part_listener(self, callback: "Callable[[PartEvent], None]") -> "Callable[[], None]":
        """Call `callback` once per new part. A callback that raises is logged
        and kept. Returns a function that removes it."""
        self._part_listeners.append(callback)

        def _remove() -> None:
            if callback in self._part_listeners:
                self._part_listeners.remove(callback)

        return _remove

    def segment_seq_for_frame(self, frame_index: int) -> "int | None":
        """The `vad.start` seq of the segment whose frame range holds
        `frame_index`, or `None` when no recorded segment holds it."""
        segment = self._segment_holding(frame_index)
        return segment.seq if segment is not None else None

    def part_index_at(self, frame_index: int) -> "int | None":
        """The part holding `frame_index` over the windows embedded so far, or
        `None` when its segment has none."""
        segment = self._segment_holding(frame_index)
        if segment is None:
            return None
        valid = _collect_valid_windows([segment], None)
        if not valid:
            return None
        embeddings = [pending.future.result() for pending in valid]
        change_points = find_change_points(embeddings, similarity_floor=self.change_similarity_floor)
        parts = split_parts(len(valid), change_points)
        return part_index_for_frame(parts, [pending.window for pending in valid], frame_index)

    def parts_after(self, frame_index: int) -> "list[PartEvent]":
        """Parts already seen after the part holding `frame_index`, oldest first."""
        segment = self._segment_holding(frame_index)
        wake_part = self.part_index_at(frame_index)
        if segment is None or wake_part is None:
            return []
        return [event for event in segment.parts if event.part_index > wake_part]

    def _add_span(self, span: TurnSpeakerSpan) -> None:
        self._live_spans.append(span)
        while len(self._live_spans) > _MAX_LIVE_SPANS:
            self._live_spans.pop(0)

    def _drop_span(self, span: TurnSpeakerSpan) -> None:
        if span in self._live_spans:
            self._live_spans.remove(span)

    def _segment_holding(self, frame_index: int) -> "_Segment | None":
        for segment in self._segments:
            if segment.first_frame_index is None:
                continue
            last = segment.last_frame_index
            if last is None:
                last = segment.first_frame_index
            if segment.first_frame_index <= frame_index <= last:
                return segment
        return None

    def _segment_for_frame(self, frame_index: "int | None") -> "_Segment | None":
        if not self._segments:
            return None
        if frame_index is not None:
            held = self._segment_holding(frame_index)
            if held is not None:
                return held
        # No mark, or the mark named an index no recorded segment's frame
        # range holds (already aged out of history) -- the most recent
        # segment is the safest available approximation.
        return self._segments[-1]

    def _submit_window(self, segment: "_Segment", window: SpeechWindow) -> None:
        future = self._worker.submit(window.pcm)
        pending = _PendingWindow(window=window, future=future)

        def _mark_completed(done_future: "asyncio.Future[Any]", pending: "_PendingWindow" = pending) -> None:
            if done_future.cancelled():
                return
            pending.completed_at = self._clock()
            # D-12: live detection runs right here, on the loop, with no
            # extra task -- the module docstring's own promise.
            for span in list(self._live_spans):
                span._check_for_split(self._clock())
            self._emit_new_parts(segment)

        future.add_done_callback(_mark_completed)
        segment.windows.append(pending)
        if len(segment.windows) > _MAX_WINDOWS_PER_SEGMENT:
            segment.windows.pop(0)

    def _emit_new_parts(self, segment: "_Segment") -> None:
        """Phase 12 part monitor: emit a `PartEvent` for each part not yet
        emitted. Change points only append as windows arrive, so no part is
        emitted twice or withdrawn."""
        valid = _collect_valid_windows([segment], None)
        if len(valid) < 2:
            return
        embeddings = [pending.future.result() for pending in valid]
        change_points = find_change_points(embeddings, similarity_floor=self.change_similarity_floor)
        parts = split_parts(len(valid), change_points)
        for index in range(len(segment.parts) + 1, len(parts)):
            first_window = valid[parts[index][0]].window
            event = PartEvent(
                segment_seq=segment.seq,
                part_index=index,
                first_frame_index=first_window.first_frame_index,
                started_at=first_window.started_at,
            )
            segment.parts.append(event)
            for listener in list(self._part_listeners):
                try:
                    listener(event)
                except Exception:
                    logger.exception("a speaker part listener raised; it stays attached")
