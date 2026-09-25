"""A read-only fan-out of the turn event stream, for `/ws/sessions/live`
(D-05, D-08, WEB-06).

The governing constraint, stated once, the way `wake/gate.py` and
`session/retention.py` both state theirs: **a browser tab watching this
feed must never be able to slow a real turn down.** This project's reply
budget is load-bearing (PROJECT.md's own latency floor), and a fan-out that
awaits a consumer is a fan-out that can stall the producer. Every method
below that a turn's own event-emission path calls -- `ObserverRegistry.publish`
and `ObserverPublishingSource.send_event` -- is synchronous where it
matters and never awaits anything that depends on an observer's own socket,
network, or browser.

Two small classes:

- `ObserverRegistry`: a set of bounded `asyncio.Queue` objects, one per
  connected observer. `publish` is a synchronous, non-waiting fan-out --
  `put_nowait`, dropping that queue's own oldest item and retrying once when
  it is full, the same "contained" discipline `sources/runner.py`'s own
  chunk loop already applies for the identical reason (a slow consumer must
  never become the producer's problem). A queue that raises on put (a
  double a test constructs, or a genuinely broken consumer) is logged and
  skipped -- one observer's own malfunction must never stop the fan-out to
  every other observer, still less the turn itself.
- `ObserverPublishingSource`: wraps one `AudioSource`-shaped object. Its
  `send_event` publishes a *new* dictionary -- the original event plus the
  source name under a `source` key -- to the registry, then forwards the
  *original*, unmodified event to the wrapped source's own `send_event`
  when it has one. Never a mutation of the caller's own dict: the same
  object is already on its way to `SessionRecorder.record_event` (via
  `turn/controller.py`'s `_RecordingAudioSource`, which wraps this class,
  not the other way around) and to a participant's own socket, and a
  shared mutable event is exactly how two consumers come to disagree about
  what happened. `frames()`, `send_audio()`, and `source_format()` forward
  straight through with no tap -- this fan-out only ever touches events,
  never audio. `sink_format()` forwards the same way, conditionally
  (260922-cts): `None` when the wrapped source declares none, never an
  `AttributeError`. `barge_in` is exposed as a property reading and writing
  through to the wrapped source: `sources/runner.py` attaches a
  `BargeInMonitor` to the *unwrapped* source before this wrapper is ever
  constructed around it, and `run_turn` reads `source.barge_in` off
  whatever it was handed -- a wrapper that swallowed this attribute would
  silently disable barge-in on the camera path (see this module's own
  tests, and `tests/test_barge_in.py` in the plan's verification block).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, AsyncIterator

from atlas.providers.tts_xai import SinkFormat
from atlas.transports.base import SourceFormat

logger = logging.getLogger("atlas.session.observers")

# A modest bound, not a tuned one: this project's realistic scale is a
# handful of turns a day and at most a few browser tabs watching at once
# (D-06's rolling feed already caps what a client renders at twenty cards).
# Large enough that an ordinary turn's handful of events never fills it;
# small enough that a genuinely stuck consumer's backlog stays bounded.
DEFAULT_MAX_QUEUE_SIZE = 100


class ObserverRegistry:
    """Bounded, non-blocking fan-out of turn events to every subscribed
    observer queue. See this module's own docstring for the constraint
    every method here exists to uphold.
    """

    def __init__(self, max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE) -> None:
        self._max_queue_size = max_queue_size
        self._queues: set["asyncio.Queue[dict[str, Any]]"] = set()

    def subscribe(self) -> "asyncio.Queue[dict[str, Any]]":
        """Return a fresh, bounded queue that receives every event
        `publish`-ed from now on. The caller owns unsubscribing it."""
        queue: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue(maxsize=self._max_queue_size)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[dict[str, Any]]") -> None:
        """Stop `queue` receiving any further event. A no-op if it was
        never subscribed, or already unsubscribed -- every caller (a
        websocket route's own `finally`) can call this unconditionally."""
        self._queues.discard(queue)

    def publish(self, event: dict[str, Any]) -> None:
        """Fan `event` out to every currently-subscribed queue.

        Synchronous and non-waiting by construction (this module's own
        governing constraint) -- awaits nothing, so the caller (a real
        turn's own event-emission path) never blocks on a consumer it does
        not control. Publishing with no subscribers does nothing and
        raises nothing. Iterates a *copy* of the subscriber set, so an
        `unsubscribe` racing this call (a tab closing mid-publish) cannot
        mutate what is being iterated.
        """
        for queue in list(self._queues):
            try:
                self._put_dropping_oldest(queue, event)
            except Exception:
                # One observer's own malfunctioning queue must never stop
                # the fan-out to every other observer, still less the turn
                # this publish call is riding along with.
                logger.exception("observer queue raised on put; dropping this event for that observer")

    @staticmethod
    def _put_dropping_oldest(queue: "asyncio.Queue[dict[str, Any]]", event: dict[str, Any]) -> None:
        """Put `event` on `queue`, dropping the oldest item first when
        full. `put_nowait`/`get_nowait` only -- never `await queue.put(...)`,
        which is precisely the call this module exists to never make.

        The two narrow races this suppresses (another producer emptying or
        refilling the queue between the full-check and the drop, or between
        the drop and the retry) are both survivable: either this event
        lands, or -- in the vanishingly unlikely case both retries lose the
        race -- it is dropped, which is exactly what "bounded, drop-oldest"
        already promises for a queue under contention.
        """
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(event)


class ObserverPublishingSource:
    """Wraps one `AudioSource`-shaped object, publishing every event it
    emits to an `ObserverRegistry` under `source_name`. See this module's
    own docstring for the full contract.
    """

    def __init__(self, wrapped: Any, source_name: str, registry: ObserverRegistry) -> None:
        self._wrapped = wrapped
        self._source_name = source_name
        self._registry = registry

    async def frames(self) -> AsyncIterator[bytes]:
        async for chunk in self._wrapped.frames():
            yield chunk

    async def send_audio(self, chunk: bytes) -> None:
        await self._wrapped.send_audio(chunk)

    async def send_event(self, event: dict[str, Any]) -> None:
        # A new dictionary, never a mutation of `event` -- see the module
        # docstring for why a shared mutable event is unsafe here.
        self._registry.publish({**event, "source": self._source_name})
        wrapped_send_event = getattr(self._wrapped, "send_event", None)
        if wrapped_send_event is not None:
            await wrapped_send_event(event)

    def source_format(self) -> SourceFormat:
        return self._wrapped.source_format()

    def sink_format(self) -> "SinkFormat | None":
        """The wrapped source's own playback sink, when it has one.

        260922-cts: forwarded the same conditional way `send_event` above
        already is -- this method always exists on this wrapper (so
        `getattr(source, "sink_format", None)` never sees an absent
        attribute and skips calling it), but returns `None` when
        `self._wrapped` carries no `sink_format` of its own, which is
        every browser and WebRTC source today. `turn/controller.py`'s
        `_speak` already treats a `None` sink as the pre-fix browser
        default, so this wrapper adds no behavior of its own -- exactly
        the same "no tap" posture the module docstring states for
        `frames()`/`send_audio()`/`source_format()`. Without this
        forward, every camera turn would lose its sink the moment
        `_make_run_turn_for_source` wraps the source here, and the
        camera-static bug this plan fixes would still reach the speaker.
        """
        wrapped_sink_format = getattr(self._wrapped, "sink_format", None)
        return wrapped_sink_format() if wrapped_sink_format is not None else None

    @property
    def barge_in(self) -> Any:
        return getattr(self._wrapped, "barge_in", None)

    @barge_in.setter
    def barge_in(self, value: Any) -> None:
        self._wrapped.barge_in = value

    @property
    def preroll_bytes(self) -> int:
        # Same forwarding discipline as `barge_in` above: without this,
        # `turn/controller.py::run_turn`'s `getattr(source, "preroll_bytes",
        # 0)` reads off this wrapper -- never the `PrerollReplayingSource`
        # underneath it -- and always falls through to `0` for every real
        # camera turn (CR-01).
        return getattr(self._wrapped, "preroll_bytes", 0)

    @property
    def follow_up(self) -> Any:
        # Plan 09-06: forwarded exactly the way `barge_in` above already
        # is -- `sources/runner.py` attaches a `FollowUpChannel` to the
        # *unwrapped* source before this wrapper is ever constructed
        # around it, and `turn/controller.py::run_turn` reads
        # `source.follow_up` off whatever it was handed. A wrapper that
        # swallowed this attribute would silently disable the follow-up
        # window on every real camera and browser-listener turn.
        return getattr(self._wrapped, "follow_up", None)

    @follow_up.setter
    def follow_up(self, value: Any) -> None:
        self._wrapped.follow_up = value
