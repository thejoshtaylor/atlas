"""`FrameFanout` and its wrappers (Phase 12, plan 12-02): one reader of the
source, one bounded queue per turn.
"""

from __future__ import annotations

import asyncio
import logging

from atlas.sources.frame_fanout import FrameFanout, TurnFrameSource

from tests.conftest import FakeAudioSource


def _chunk(index: int) -> bytes:
    return bytes([index]) * 4


async def _read_all(subscription) -> list[bytes]:
    return [chunk async for chunk in subscription.frames()]


async def test_replay_then_live_has_no_gap_and_no_repeat():
    fanout = FrameFanout()
    for index in range(5):
        fanout.push(_chunk(index))
    subscription = fanout.subscribe(replay_from=2)
    fanout.push(_chunk(5))
    fanout.push(_chunk(6))
    fanout.close()

    assert await _read_all(subscription) == [_chunk(i) for i in range(2, 7)]
    assert subscription.first_frame_index == 2
    assert subscription.replayed_bytes == 3 * 4


async def test_a_subscription_without_replay_gets_only_live_frames():
    fanout = FrameFanout()
    fanout.push(_chunk(0))
    subscription = fanout.subscribe()
    fanout.push(_chunk(1))
    fanout.close()

    assert await _read_all(subscription) == [_chunk(1)]
    assert subscription.replayed_bytes == 0


async def test_a_live_frame_below_replay_from_is_skipped():
    fanout = FrameFanout()
    subscription = fanout.subscribe(replay_from=10)
    fanout.push(_chunk(5), frame_index=5)
    fanout.push(_chunk(10), frame_index=10)
    fanout.push(_chunk(11), frame_index=11)
    fanout.close()

    assert await _read_all(subscription) == [_chunk(10), _chunk(11)]
    assert subscription.first_frame_index == 10


async def test_the_source_index_is_used_when_given_and_counted_otherwise():
    fanout = FrameFanout()
    assert fanout.push(_chunk(0), frame_index=40) == 40
    assert fanout.push(_chunk(1)) == 41
    assert fanout.latest_index == 41


async def test_history_eviction_makes_slice_return_none():
    fanout = FrameFanout(history_frames=3)
    for index in range(6):
        fanout.push(_chunk(index))

    assert fanout.slice(0, 3) is None
    assert fanout.slice(3, 6) == [_chunk(3), _chunk(4), _chunk(5)]
    assert fanout.slice(4, 4) == []


async def test_a_full_queue_drops_its_oldest_frame_and_warns_once(caplog):
    fanout = FrameFanout(queue_frames=2)
    subscription = fanout.subscribe()
    with caplog.at_level(logging.WARNING, logger="atlas.sources.frame_fanout"):
        for index in range(5):
            fanout.push(_chunk(index))
    fanout.close()

    assert await _read_all(subscription) == [_chunk(3), _chunk(4)]
    warnings = [record for record in caplog.records if "saturated" in record.getMessage()]
    assert len(warnings) == 1


async def test_close_ends_frames_after_what_is_queued_and_a_later_call_ends_at_once():
    fanout = FrameFanout()
    subscription = fanout.subscribe()
    fanout.push(_chunk(0))
    fanout.push(_chunk(1))
    subscription.close()

    assert await _read_all(subscription) == [_chunk(0), _chunk(1)]
    assert await asyncio.wait_for(_read_all(subscription), timeout=1.0) == []
    fanout.push(_chunk(2))  # a closed subscription receives nothing more
    assert await _read_all(subscription) == []


async def test_a_second_frames_call_continues_where_the_first_stopped():
    fanout = FrameFanout()
    for index in range(3):
        fanout.push(_chunk(index))
    subscription = fanout.subscribe(replay_from=0)
    fanout.close()

    first = subscription.frames()
    assert await first.__anext__() == _chunk(0)
    await first.aclose()

    assert await _read_all(subscription) == [_chunk(1), _chunk(2)]


async def test_two_subscriptions_each_get_every_frame():
    fanout = FrameFanout()
    first = fanout.subscribe()
    second = fanout.subscribe()
    for index in range(3):
        fanout.push(_chunk(index))
    fanout.close()

    assert await _read_all(first) == [_chunk(i) for i in range(3)]
    assert await _read_all(second) == [_chunk(i) for i in range(3)]


async def test_subscribing_after_close_gives_an_ended_stream():
    fanout = FrameFanout()
    fanout.close()

    assert await asyncio.wait_for(_read_all(fanout.subscribe(replay_from=0)), timeout=1.0) == []


async def test_a_turn_frame_source_reads_its_subscription_and_forwards_the_rest():
    fanout = FrameFanout()
    wrapped = FakeAudioSource()
    subscription = fanout.subscribe(replay_from=0)
    fanout.push(_chunk(0))
    fanout.close()
    turn_source = TurnFrameSource(subscription, wrapped, preroll_bytes=7)

    assert [chunk async for chunk in turn_source.frames()] == [_chunk(0)]
    assert turn_source.preroll_bytes == 7
    assert turn_source.first_frame_index == 0
    assert turn_source.source_format() == wrapped.source_format()
    assert turn_source.sink_format() is None  # the fake declares no sink
    assert turn_source.speech_signals is None
    await turn_source.send_event({"type": "wake.heard"})  # a fake with no send_event: a no-op
    await turn_source.send_audio(b"reply")
    assert wrapped.sent_audio == [b"reply"]
