"""`ObserverRegistry`/`ObserverPublishingSource` (WEB-06, D-05, D-08).

Task 1 covers the fan-out itself: subscribe/unsubscribe, the non-waiting
drop-oldest publish, and the wrapper's event/barge-in behavior. Task 2 (the
route-level tests through a real `TestClient` WebSocket, driving a real turn
on each path) is added to this same file, below the Task 1 section, per the
plan's own instruction not to split the two.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from spire_voice.session.observers import ObserverPublishingSource, ObserverRegistry
from spire_voice.transports.base import SourceFormat


# --- Task 1: the fan-out -----------------------------------------------


def test_a_subscriber_receives_every_subsequently_published_event():
    registry = ObserverRegistry()
    queue = registry.subscribe()

    registry.publish({"type": "transcript.partial", "text": "a"})
    registry.publish({"type": "transcript.partial", "text": "b"})

    assert queue.get_nowait() == {"type": "transcript.partial", "text": "a"}
    assert queue.get_nowait() == {"type": "transcript.partial", "text": "b"}


def test_unsubscribe_stops_further_delivery():
    registry = ObserverRegistry()
    queue = registry.subscribe()
    registry.unsubscribe(queue)

    registry.publish({"type": "transcript.partial", "text": "a"})

    assert queue.empty()


def test_publishing_with_no_subscribers_does_nothing_and_raises_nothing():
    registry = ObserverRegistry()
    registry.publish({"type": "transcript.partial", "text": "a"})  # must not raise


def test_a_full_queue_drops_its_oldest_message_and_the_publisher_still_returns():
    registry = ObserverRegistry(max_queue_size=2)
    queue = registry.subscribe()

    registry.publish({"type": "e", "n": 1})
    registry.publish({"type": "e", "n": 2})
    # The queue is now full (bound = 2). A third publish must not block --
    # it drops the oldest (n=1) and enqueues the new one (n=3).
    registry.publish({"type": "e", "n": 3})

    assert queue.qsize() == 2
    assert queue.get_nowait() == {"type": "e", "n": 2}
    assert queue.get_nowait() == {"type": "e", "n": 3}


def test_publish_is_synchronous_and_awaits_nothing():
    # A synchronous call, not a coroutine -- calling it from ordinary (not
    # async) code is itself proof `publish` awaits nothing.
    registry = ObserverRegistry()
    registry.subscribe()
    result = registry.publish({"type": "e"})
    assert result is None


def test_two_subscribers_both_receive_the_same_publish():
    registry = ObserverRegistry()
    a = registry.subscribe()
    b = registry.subscribe()

    registry.publish({"type": "e"})

    assert a.get_nowait() == {"type": "e"}
    assert b.get_nowait() == {"type": "e"}


class _RaisingQueue:
    """A double satisfying just enough of `asyncio.Queue`'s put surface to
    drive `ObserverRegistry._put_dropping_oldest`'s outer `except Exception`
    branch without mocking the exception -- `put_nowait` raises a plain
    `RuntimeError`, not `asyncio.QueueFull`, so the drop-oldest path is
    never reached for this queue at all."""

    def put_nowait(self, item: Any) -> None:
        raise RuntimeError("this queue is broken")


def test_a_raising_queue_does_not_propagate_out_of_publish():
    registry = ObserverRegistry()
    good_queue = registry.subscribe()
    registry._queues.add(_RaisingQueue())  # type: ignore[arg-type]

    registry.publish({"type": "e"})  # must not raise

    assert good_queue.get_nowait() == {"type": "e"}


class _EventSource:
    """A minimal `AudioSource`-shaped double: `frames`/`send_audio`/
    `source_format`, plus an optional `send_event`, and a `barge_in`
    attribute settable at construction or afterward."""

    def __init__(self, *, with_send_event: bool = True, barge_in: Any = None) -> None:
        self.sent_audio: list[bytes] = []
        self.received_events: list[dict[str, Any]] = []
        self._has_send_event = with_send_event
        if barge_in is not None:
            self.barge_in = barge_in

    async def frames(self):
        yield b"chunk-1"
        yield b"chunk-2"

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    def source_format(self) -> SourceFormat:
        return SourceFormat("pcm", 16000)

    async def send_event(self, event: dict[str, Any]) -> None:
        if not self._has_send_event:
            raise AssertionError("send_event should not exist on this double")
        self.received_events.append(event)


async def test_send_event_publishes_with_the_source_name_added():
    registry = ObserverRegistry()
    queue = registry.subscribe()
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", registry)

    await source.send_event({"type": "transcript.partial", "text": "hello"})

    published = queue.get_nowait()
    assert published == {"type": "transcript.partial", "text": "hello", "source": "camera"}


async def test_send_event_forwards_the_original_unmodified_event_to_the_wrapped_source():
    registry = ObserverRegistry()
    registry.subscribe()
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", registry)

    await source.send_event({"type": "transcript.partial", "text": "hello"})

    # The wrapped (real) source's own send_event never sees the "source" key
    # -- that field is an observer-only addition, per the module docstring.
    assert wrapped.received_events == [{"type": "transcript.partial", "text": "hello"}]


async def test_send_event_does_nothing_extra_when_the_wrapped_source_has_no_send_event():
    class _Bare:
        def __init__(self) -> None:
            self.frames_called = False

        async def frames(self):
            yield b"x"

        async def send_audio(self, chunk: bytes) -> None:
            pass

        def source_format(self) -> SourceFormat:
            return SourceFormat("pcm", 16000)

    registry = ObserverRegistry()
    queue = registry.subscribe()
    source = ObserverPublishingSource(_Bare(), "camera", registry)

    await source.send_event({"type": "transcript.partial", "text": "hi"})  # must not raise

    assert queue.get_nowait() == {"type": "transcript.partial", "text": "hi", "source": "camera"}


async def test_frames_send_audio_and_source_format_forward_straight_through():
    registry = ObserverRegistry()
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", registry)

    chunks = [chunk async for chunk in source.frames()]
    assert chunks == [b"chunk-1", b"chunk-2"]

    await source.send_audio(b"reply-chunk")
    assert wrapped.sent_audio == [b"reply-chunk"]

    assert source.source_format() == SourceFormat("pcm", 16000)


def test_barge_in_set_before_wrapping_reads_through():
    sentinel = object()
    wrapped = _EventSource(barge_in=sentinel)
    source = ObserverPublishingSource(wrapped, "camera", ObserverRegistry())

    assert source.barge_in is sentinel


def test_barge_in_set_after_wrapping_writes_and_reads_through():
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", ObserverRegistry())
    sentinel = object()

    source.barge_in = sentinel

    assert source.barge_in is sentinel
    assert wrapped.barge_in is sentinel


def test_barge_in_defaults_to_none_when_the_wrapped_source_never_set_one():
    wrapped = _EventSource()
    source = ObserverPublishingSource(wrapped, "camera", ObserverRegistry())

    assert source.barge_in is None
