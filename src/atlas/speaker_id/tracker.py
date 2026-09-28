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
from typing import TYPE_CHECKING, Any

from atlas.audio.channels import select_channel
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


@dataclass(eq=False)
class _PendingWindow:
    """One window already submitted to the `EmbeddingWorker`. `eq=False`
    (identity comparison) matters here for the same reason it matters on
    `_Segment` below -- this object is looked up by identity, never by
    field equality."""

    window: SpeechWindow
    future: "asyncio.Future[Any]"
    completed_at: "float | None" = None


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


@dataclass(frozen=True)
class SpeakerMeasurement:
    """One `TurnSpeakerSpan.decide()` result.

    `detail` is `None` for a real measurement (whether or not it later
    passes the gate's threshold), `"no_speech_measured"` when no window
    could be embedded at all, and `"decision_timeout"` when the worker did
    not finish within `timeout_s`. `speaker_id_ms` is `None` without an
    `end_of_speech_at` to measure from.
    """

    match: "MatchResult | None"
    speech_ms: float
    window_count: int
    ready_at: "float | None"
    speaker_id_ms: "float | None"
    detail: "str | None"


class TurnSpeakerSpan:
    """A view into the tracker's own live segment history, from a turn's
    own starting point onward -- not a snapshot. A wake turn's span (no
    `start_after`) is anchored to a specific `_Segment` object found at
    `open_turn()` time; a follow-up turn's span (`start_after` given) is
    anchored to a timestamp and re-filters every segment at `decide()`
    time, since D-11's follow-up window can span more than one segment
    that did not exist yet when the span opened.
    """

    def __init__(
        self,
        tracker: "SpeakerTracker",
        *,
        start_segment: "_Segment | None" = None,
        start_after: "float | None" = None,
    ) -> None:
        self._tracker = tracker
        self._start_segment = start_segment
        self._start_after = start_after

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

        embeddings = []
        weights = []
        completed_ats: "list[float]" = []
        for pending in pending_windows:
            future = pending.future
            if future.cancelled() or future.exception() is not None:
                # Skip a window whose embedding raised (or was cancelled) --
                # the rest of the turn's speech still measures something.
                continue
            embeddings.append(future.result())
            weights.append(pending.window.speech_ms)
            if pending.completed_at is not None:
                completed_ats.append(pending.completed_at)

        if not embeddings:
            return SpeakerMeasurement(
                match=None, speech_ms=0.0, window_count=0, ready_at=None, speaker_id_ms=None,
                detail="no_speech_measured",
            )

        embedding = mean_embedding(embeddings, weights)
        match_start = self._tracker._clock()
        match = references.match(embedding) if references is not None else None
        match_elapsed_s = self._tracker._clock() - match_start

        ready_at = max(completed_ats) if completed_ats else None
        speaker_id_ms: "float | None" = None
        if end_of_speech_at is not None and ready_at is not None:
            speaker_id_ms = (max(0.0, ready_at - end_of_speech_at) + match_elapsed_s) * 1000.0

        return SpeakerMeasurement(
            match=match,
            speech_ms=sum(weights),
            window_count=len(embeddings),
            ready_at=ready_at,
            speaker_id_ms=speaker_id_ms,
            detail=None,
        )


class SpeakerTracker:
    """Implements `EdgeAudioListener` (`transports/edge.py`). Attached once,
    at wiring time (`speaker_id/wiring.py`), for the whole edge source's
    life -- never per turn.

    `change_similarity_floor` is stored but unused by this plan; plan 11-06
    (D-12's change-point split) reads it from here so `wiring.py` never
    changes to add that plan's own wiring.
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
        """
        if start_after is not None:
            return TurnSpeakerSpan(self, start_after=start_after)
        frame_index = self._wake_mark_frame_index
        self._wake_mark_frame_index = None
        start_segment = self._segment_for_frame(frame_index)
        return TurnSpeakerSpan(self, start_segment=start_segment)

    def _segment_for_frame(self, frame_index: "int | None") -> "_Segment | None":
        if not self._segments:
            return None
        if frame_index is not None:
            for segment in self._segments:
                if segment.first_frame_index is None:
                    continue
                last = segment.last_frame_index
                if last is None:
                    last = segment.first_frame_index
                if segment.first_frame_index <= frame_index <= last:
                    return segment
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

        future.add_done_callback(_mark_completed)
        segment.windows.append(pending)
        if len(segment.windows) > _MAX_WINDOWS_PER_SEGMENT:
            segment.windows.pop(0)
