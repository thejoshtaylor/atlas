"""Phase 12, plan 12-06: a part of a woken segment becomes a turn of its own.

A second voice inside a VAD segment that a wake hit opened starts a turn with
no wake word, but only in enforce mode with someone enrolled, only for a part
after the wake part, and only once per part (D-01 trigger 1, D-02 to D-05).

The rig drives the real `SpeakerTracker` and the real `TurnGroup`. Only the
turn body (`hooks.run_turn`) is a stand-in that records its context. Voices
are constant PCM values: 2000 and 20000 land in different `FakeEmbedder`
buckets, so they are two known voices. Every frame is one window (100 ms).
"""

from __future__ import annotations

import asyncio
import logging
import struct
from types import SimpleNamespace

import pytest

import test_speaker_split as tss
from atlas.sources.turn_group import BLOCK_TURN_CAP, ParallelTurns, TurnGroup, TurnHooks
from atlas.speaker_id.matching import ReferenceSet
from atlas.speaker_id.tracker import PartEvent, SpeakerTracker
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext
from atlas.transports.base import SourceFormat

from tests.speaker_fakes import FakeEmbedder

_RATE = 16000
_FRAME_MS = 100
VOICE_A = 2000
VOICE_B = 20000


def _frame(value: int) -> bytes:
    samples = _RATE * _FRAME_MS // 1000
    return struct.pack(f"<{samples}h", *([value] * samples))


def _tracker() -> SpeakerTracker:
    return SpeakerTracker(
        channels=1,
        asr_channel=0,
        sample_rate=_RATE,
        window_ms=_FRAME_MS,
        min_window_ms=50,
        speech_rms_floor=0.01,
        change_similarity_floor=0.3,
        worker=tss._SyncWorker(FakeEmbedder(dim=8)),
    )


async def _settle() -> None:
    for _ in range(12):
        await asyncio.sleep(0)


class _Rig:
    """A tracker and a group on one fake source, with `run_turn` stubbed."""

    def __init__(self, *, mode: str = "enforce", enrolled: bool = True, max_concurrent: int = 4) -> None:
        self.tracker = _tracker()
        references = ReferenceSet()
        if enrolled:
            references.upsert_speaker(1, "Example Member", [(1.0, 0.0)])
        self.speaker = SpeakerIdTurnContext(
            tracker=self.tracker, references=references, mode=mode, threshold=0.5, model_id="m", worker=object()
        )
        self.sources: list = []
        self.blocked: list[tuple[float, str | None]] = []
        self.release = asyncio.Event()
        self.frame_index = 0
        source = SimpleNamespace(source_format=lambda: SourceFormat("pcm", _RATE))
        hooks = TurnHooks(
            run_turn=self._run_turn,
            new_monitor=lambda: SimpleNamespace(transcript_done=asyncio.Event()),
            watch_barge_in=self._watch,
            record_blocked_hit=lambda score, reason, at: self.blocked.append((score, reason)),
            clock=lambda: 0.0,
        )
        spec = ParallelTurns(max_concurrent=max_concurrent, preroll_ms=_FRAME_MS, speaker_context=lambda: self.speaker)
        self.group = TurnGroup("edge", source, spec, hooks)

    async def _run_turn(self, turn_source) -> None:
        self.sources.append(turn_source)
        await self.release.wait()

    async def _watch(self, monitor, frames) -> None:
        await asyncio.Event().wait()

    @property
    def contexts(self) -> list:
        return [source.turn_context for source in self.sources]

    async def feed(self, value: int, *, wake: bool = False) -> int:
        """One frame at `value`. `wake` puts a wake hit on that frame, after the
        frame reached the tracker and the fan-out, the order the runner uses."""
        index = self.frame_index
        self.frame_index += 1
        chunk = _frame(value)
        self.tracker.on_frame(chunk, index, float(index))
        self.group.push_frame(chunk, index)
        if wake:
            self.tracker.on_wake_hit(index)
            self.group.start_wake_turn(index)
        await _settle()
        return index

    async def finish(self) -> None:
        self.release.set()
        await _settle()


# ---------------------------------------------------------------------------
# Task 1: the tracer -- a second voice after the wake part starts its own turn
# ---------------------------------------------------------------------------


async def test_a_second_voice_after_the_wake_part_starts_its_own_turn():
    rig = _Rig()
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A, wake=True)
    await rig.feed(VOICE_B)

    assert len(rig.sources) == 2
    wake_context, part_context = rig.contexts
    assert wake_context.split_part is False
    assert part_context.split_part is True
    assert part_context.order_frame == 1
    assert rig.sources[1].first_frame_index == 1
    first_chunk = await asyncio.wait_for(anext(rig.sources[1].frames()), timeout=1.0)
    assert first_chunk == _frame(VOICE_B), "the part turn's audio starts at the part's first frame"

    wake_span = wake_context.speaker_span
    assert wake_span.split_event.is_set()
    assert wake_span.split_frame_index == 1

    rig.tracker.on_vad_end(2, 5.0)
    measurement = await part_context.speaker_span.decide(end_of_speech_at=None, references=None)
    assert measurement.window_count == 1, "the part turn's kept part is part B alone"
    assert [part.first_frame_index for part in measurement.dropped_parts] == [0]
    await rig.finish()


async def test_a_part_turn_after_the_wake_turn_ended_still_starts_a_group():
    rig = _Rig()
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A, wake=True)
    await rig.finish()  # the wake turn is over before B speaks
    rig.release.clear()
    await rig.feed(VOICE_B)

    assert len(rig.sources) == 2
    assert rig.contexts[1].split_part is True
    assert rig.contexts[1].group_id is not None
    await rig.finish()


async def test_a_turn_that_ends_closes_its_span():
    rig = _Rig()
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A, wake=True)
    assert len(rig.tracker._live_spans) == 1
    await rig.finish()
    assert rig.tracker._live_spans == []


async def test_a_source_with_no_speaker_context_runs_wake_turns_unchanged():
    rig = _Rig()
    rig.group._spec = ParallelTurns(max_concurrent=4, preroll_ms=_FRAME_MS)
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A, wake=True)

    assert len(rig.sources) == 1
    assert rig.contexts[0].speaker_span is None
    await rig.finish()


# ---------------------------------------------------------------------------
# The tracker: part events, live spans, open_turn_at, close()
# ---------------------------------------------------------------------------


async def _feed_tracker(tracker: SpeakerTracker, values: list[int], *, first_index: int = 0) -> None:
    for offset, value in enumerate(values):
        index = first_index + offset
        tracker.on_frame(_frame(value), index, float(index))
        await asyncio.sleep(0)
        await asyncio.sleep(0)


async def test_the_tracker_emits_one_part_event_per_new_part():
    tracker = _tracker()
    events: list[PartEvent] = []
    tracker.add_part_listener(events.append)
    tracker.on_vad_start(7, 0.0)

    await _feed_tracker(tracker, [VOICE_A, VOICE_A, VOICE_B, VOICE_B, VOICE_A])

    assert [(e.segment_seq, e.part_index, e.first_frame_index) for e in events] == [(7, 1, 2), (7, 2, 4)]
    assert tracker.segment_seq_for_frame(3) == 7
    assert tracker.segment_seq_for_frame(99) is None
    assert tracker.part_index_at(0) == 0
    assert tracker.part_index_at(3) == 1
    assert tracker.part_index_at(4) == 2
    assert [e.part_index for e in tracker.parts_after(0)] == [1, 2]
    assert [e.part_index for e in tracker.parts_after(2)] == [2]


async def test_a_listener_that_raises_is_logged_and_kept(caplog):
    tracker = _tracker()
    calls: list[int] = []

    def _bad(event: PartEvent) -> None:
        calls.append(event.part_index)
        raise RuntimeError("boom")

    good: list[PartEvent] = []
    tracker.add_part_listener(_bad)
    tracker.add_part_listener(good.append)
    tracker.on_vad_start(1, 0.0)

    with caplog.at_level(logging.ERROR, logger="atlas.speaker_id.tracker"):
        await _feed_tracker(tracker, [VOICE_A, VOICE_B, VOICE_A])

    assert calls == [1, 2], "the raising listener stays attached"
    assert [e.part_index for e in good] == [1, 2], "a later listener still hears every part"
    assert any("part listener" in r.getMessage() for r in caplog.records)


async def test_a_removed_listener_hears_nothing_more():
    tracker = _tracker()
    events: list[PartEvent] = []
    remove = tracker.add_part_listener(events.append)
    tracker.on_vad_start(1, 0.0)
    await _feed_tracker(tracker, [VOICE_A, VOICE_B])
    remove()
    remove()  # safe twice
    await _feed_tracker(tracker, [VOICE_A], first_index=2)

    assert [e.part_index for e in events] == [1]


async def test_open_turn_at_anchors_each_span_to_its_own_frame_and_reads_no_wake_mark():
    tracker = _tracker()
    tracker.on_vad_start(1, 0.0)
    await _feed_tracker(tracker, [VOICE_A, VOICE_B, VOICE_A])
    tracker.on_wake_hit(0)

    first = tracker.open_turn_at(0)
    second = tracker.open_turn_at(1)
    assert tracker._wake_mark_frame_index == 0, "open_turn_at leaves the wake mark alone"
    tracker.on_vad_end(2, 9.0)

    one = await first.decide(end_of_speech_at=None, references=None)
    two = await second.decide(end_of_speech_at=None, references=None)
    assert [p.first_frame_index for p in one.dropped_parts] == [1, 2]
    assert [p.first_frame_index for p in two.dropped_parts] == [0, 2]
    assert first.split_frame_index == 1, "the first frame of the part after the kept part"
    assert second.split_frame_index == 2


async def test_live_detection_reports_to_every_open_span():
    tracker = _tracker()
    tracker.on_vad_start(1, 0.0)
    await _feed_tracker(tracker, [VOICE_A])
    tracker.on_wake_hit(0)
    legacy = tracker.open_turn()
    anchored = tracker.open_turn_at(0)

    await _feed_tracker(tracker, [VOICE_B], first_index=1)

    assert legacy.split_event.is_set()
    assert anchored.split_event.is_set()


async def test_close_is_idempotent_and_a_closed_span_no_longer_detects_a_split():
    tracker = _tracker()
    tracker.on_vad_start(1, 0.0)
    await _feed_tracker(tracker, [VOICE_A])
    span = tracker.open_turn_at(0)

    span.close()
    span.close()
    assert span not in tracker._live_spans

    await _feed_tracker(tracker, [VOICE_B], first_index=1)
    assert not span.split_event.is_set()


async def test_the_live_span_list_is_bounded_and_drops_the_oldest():
    tracker = _tracker()
    tracker.on_vad_start(1, 0.0)
    await _feed_tracker(tracker, [VOICE_A])
    spans = [tracker.open_turn_at(0) for _ in range(12)]

    assert len(tracker._live_spans) == 8
    assert tracker._live_spans[0] is spans[4]


# ---------------------------------------------------------------------------
# Task 2: the spawn rules
# ---------------------------------------------------------------------------


async def test_parts_a_b_a_give_three_turns():
    rig = _Rig()
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A, wake=True)
    await rig.feed(VOICE_B)
    await rig.feed(VOICE_A)

    assert [context.split_part for context in rig.contexts] == [False, True, True]
    assert [context.order_frame for context in rig.contexts] == [0, 1, 2]
    await rig.finish()


async def test_a_part_before_the_wake_part_starts_no_turn():
    rig = _Rig()
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_B)
    await rig.feed(VOICE_A, wake=True)  # the wake window has not embedded yet at the hit

    assert len(rig.sources) == 1
    assert rig.contexts[0].split_part is False
    await rig.finish()


@pytest.mark.parametrize(
    ("mode", "enrolled"),
    [("record", True), ("off", True), ("enforce", False)],
    ids=["record", "off", "enforce-nobody-enrolled"],
)
async def test_a_part_is_dropped_and_logged_outside_enforce_with_someone_enrolled(mode, enrolled, caplog):
    rig = _Rig(mode=mode, enrolled=enrolled)
    rig.tracker.on_vad_start(1, 0.0)
    with caplog.at_level(logging.INFO, logger="atlas.sources.turn_group"):
        await rig.feed(VOICE_A, wake=True)
        await rig.feed(VOICE_B)

    assert len(rig.sources) == 1
    dropped = [r.getMessage() for r in caplog.records if "dropped" in r.getMessage()]
    assert len(dropped) == 1
    assert "dropped 1 speaker part" in dropped[0]
    assert "Example Member" not in dropped[0]
    await rig.finish()


async def test_parts_of_a_segment_with_no_wake_hit_start_no_turn():
    rig = _Rig()
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A, wake=True)
    await rig.finish()
    rig.tracker.on_vad_end(2, 1.0)

    rig.release.clear()
    rig.tracker.on_vad_start(3, 2.0)  # the next speaker, no wake word
    await rig.feed(VOICE_B)
    await rig.feed(VOICE_A)

    assert len(rig.sources) == 1, "a segment with no wake hit starts nothing"
    await rig.finish()


async def test_a_part_over_the_cap_is_dropped_and_recorded_as_turn_cap():
    rig = _Rig(max_concurrent=2)
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A, wake=True)
    await rig.feed(VOICE_B)
    await rig.feed(VOICE_A)

    assert len(rig.sources) == 2
    assert rig.blocked == [(0.0, BLOCK_TURN_CAP)]
    await rig.finish()


async def test_parts_seen_before_a_late_wake_hit_start_once_when_the_hit_arrives():
    rig = _Rig()
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A)
    await rig.feed(VOICE_B)
    assert rig.sources == [], "no wake hit yet: the part is dropped as unwoken"

    rig.tracker.on_wake_hit(0)
    rig.group.start_wake_turn(0)
    await _settle()

    assert [context.split_part for context in rig.contexts] == [False, True]
    assert rig.contexts[0].speaker_span.split_event.is_set(), "the split already existed"
    assert rig.contexts[0].speaker_span.split_frame_index == 1
    await rig.finish()


async def test_the_same_part_event_delivered_twice_starts_one_turn():
    rig = _Rig()
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A, wake=True)
    await rig.feed(VOICE_B)
    event = PartEvent(segment_seq=1, part_index=1, first_frame_index=1, started_at=1.0)

    rig.group._on_part(event)
    rig.group._on_part(event)
    assert rig.group.start_part_turn(event) is None
    await _settle()

    assert len(rig.sources) == 2
    await rig.finish()


async def test_a_part_that_holds_its_own_wake_hit_is_a_wake_turn_not_a_part_turn():
    rig = _Rig()
    rig.tracker.on_vad_start(1, 0.0)
    await rig.feed(VOICE_A, wake=True)
    await rig.feed(VOICE_B, wake=True)  # the second speaker says the wake word too

    assert [context.split_part for context in rig.contexts] == [False, False]
    await rig.finish()
