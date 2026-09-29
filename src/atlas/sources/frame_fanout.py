"""One frame reader, many bounded readers (Phase 12, D-01).

`SourceRunner.run()` is the only caller of `source.frames()` on the parallel
path. It pushes every chunk into one `FrameFanout`. Each turn, follow-up
window, and barge-in listener reads its own `FrameSubscription`, so two turns
never split one queue between them (the failure `sources/runner.py`'s module
docstring describes for `CameraAudioSource`).

Every queue is bounded (`MAX_QUEUED_FRAMES`, the same bound and the same
drop-oldest rule as `EdgeAudioSource._enqueue_frame`), because a peer on the
network controls how many frames arrive (T-12-05).

`ForwardingSource` and `TurnFrameSource` are the source wrappers a turn runs
against. `ForwardingSource` is the one forwarding base class: it passes
`send_audio`, `send_event`, `source_format`, `sink_format`, and `speech_signals`
to the wrapped source and adds nothing else.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from typing import Any, AsyncIterator

from atlas.providers.tts_xai import SinkFormat
from atlas.transports.edge import MAX_QUEUED_FRAMES

logger = logging.getLogger("atlas.sources.frame_fanout")

# Puts the end of a subscription's stream in its own queue, behind every frame
# that was already queued, so a turn still reads what arrived before the end.
_END = object()


class FrameSubscription:
    """One reader's bounded queue of frames.

    `frames()` may be called more than once (`run_turn` drains a turn's frames
    a second time when the first transcript was only the wake phrase). The
    queue lives as long as the subscription, so a second call continues where
    the first stopped and never replays.
    """

    def __init__(self, fanout: "FrameFanout", *, queue_frames: int, replay_from: int | None) -> None:
        self._fanout = fanout
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=queue_frames)
        self._min_index = replay_from
        self._closed = False
        self._ended = False
        self._saturation_warned = False
        self._queue_frames = queue_frames
        self.first_frame_index: int | None = None
        self.replayed_bytes = 0

    def _offer(self, frame_index: int, chunk: bytes, *, replayed: bool = False) -> None:
        """Queue one frame. A live frame below `replay_from` is skipped."""
        if self._closed:
            return
        if self._min_index is not None and frame_index < self._min_index:
            return
        if self.first_frame_index is None:
            self.first_frame_index = frame_index
        if replayed:
            self.replayed_bytes += len(chunk)
        self._put(chunk)

    def _put(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
            self._saturation_warned = False
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self._queue.put_nowait(item)
            if not self._saturation_warned:
                self._saturation_warned = True
                logger.warning(
                    "frame subscription saturated at %d frames; dropping the oldest",
                    self._queue_frames,
                )

    async def frames(self) -> AsyncIterator[bytes]:
        while not self._ended:
            item = await self._queue.get()
            if item is _END:
                self._ended = True
                return
            yield item

    def close(self) -> None:
        """Stop receiving frames. What is already queued is still readable,
        then `frames()` ends. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._fanout._detach(self)
        self._put(_END)


class FrameFanout:
    """Numbers each pushed chunk, keeps a bounded history, and hands every
    live subscription its own copy of the reference."""

    def __init__(self, *, history_frames: int = MAX_QUEUED_FRAMES, queue_frames: int = MAX_QUEUED_FRAMES) -> None:
        self._history: "deque[tuple[int, bytes]]" = deque(maxlen=history_frames)
        self._queue_frames = queue_frames
        self._subscriptions: list[FrameSubscription] = []
        self._closed = False
        self.latest_index: int | None = None
        self.last_chunk_bytes = 0

    def push(self, chunk: bytes, frame_index: int | None = None) -> int:
        """Record `chunk` and queue it on every subscription. `frame_index` is
        the source's own index when it has one. Otherwise the fan-out counts."""
        if frame_index is None:
            frame_index = 0 if self.latest_index is None else self.latest_index + 1
        self.latest_index = frame_index
        self.last_chunk_bytes = len(chunk)
        if self._closed:
            return frame_index
        self._history.append((frame_index, chunk))
        for subscription in list(self._subscriptions):
            subscription._offer(frame_index, chunk)
        return frame_index

    def subscribe(self, *, replay_from: int | None = None) -> FrameSubscription:
        """A new reader. With `replay_from`, its queue starts with every
        retained frame at or above that index. The replay and the
        registration happen in one synchronous step, so no frame is lost or
        repeated between the replay and the live frames."""
        subscription = FrameSubscription(self, queue_frames=self._queue_frames, replay_from=replay_from)
        if self._closed:
            subscription.close()
            return subscription
        if replay_from is not None:
            for frame_index, chunk in self._history:
                if frame_index >= replay_from:
                    subscription._offer(frame_index, chunk, replayed=True)
        self._subscriptions.append(subscription)
        return subscription

    def slice(self, start_index: int, end_index: int) -> list[bytes] | None:
        """The retained frames with `start_index <= index < end_index`, oldest
        first. `None` when `start_index` has already left the history."""
        if end_index <= start_index:
            return []
        if not self._history or start_index < self._history[0][0]:
            return None
        return [chunk for frame_index, chunk in self._history if start_index <= frame_index < end_index]

    def close(self) -> None:
        """End every subscription's `frames()`, after what is already queued."""
        self._closed = True
        for subscription in list(self._subscriptions):
            subscription.close()

    def _detach(self, subscription: FrameSubscription) -> None:
        with contextlib.suppress(ValueError):
            self._subscriptions.remove(subscription)


class ForwardingSource:
    """Wraps one `AudioSource` and forwards everything except `frames()`.

    `send_event` is a no-op when the wrapped source has none (a test double),
    the same conditional forward `FollowUpSource` uses. `sink_format` is
    `None` when the wrapped source has no sink (260922-cts)."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    async def send_audio(self, chunk: bytes) -> None:
        await self._wrapped.send_audio(chunk)

    async def send_event(self, event: dict[str, Any]) -> None:
        wrapped_send_event = getattr(self._wrapped, "send_event", None)
        if wrapped_send_event is not None:
            await wrapped_send_event(event)

    def source_format(self) -> Any:
        return self._wrapped.source_format()

    def sink_format(self) -> "SinkFormat | None":
        wrapped_sink_format = getattr(self._wrapped, "sink_format", None)
        return wrapped_sink_format() if wrapped_sink_format is not None else None

    @property
    def speech_signals(self) -> Any:
        return getattr(self._wrapped, "speech_signals", None)


class TurnFrameSource(ForwardingSource):
    """The source one turn runs against: `frames()` reads the turn's own
    subscription. The runner sets `barge_in`, `follow_up`, and `turn_context`
    on it, the same duck-typed attachments `run_turn` reads off a source."""

    def __init__(self, subscription: FrameSubscription, wrapped: Any, *, preroll_bytes: int = 0) -> None:
        super().__init__(wrapped)
        self._subscription = subscription
        self._preroll_bytes = preroll_bytes

    @property
    def preroll_bytes(self) -> int:
        """How many of the bytes `frames()` yields first are replay."""
        return self._preroll_bytes

    @property
    def first_frame_index(self) -> int | None:
        return self._subscription.first_frame_index

    async def frames(self) -> AsyncIterator[bytes]:
        async for chunk in self._subscription.frames():
            yield chunk
