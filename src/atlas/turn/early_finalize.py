"""The VAD-end watch: when a Pi reports the operator stopped speaking, this
is what tells `turn/controller.py::_drain_to_final_transcript` to finalize
speech-to-text at once instead of waiting for the provider's own
endpointing (D-09 through D-13, 10-05-PLAN.md).

Duck-typed on `signals` -- this module imports nothing from `transports/`,
matching `_BargeInMonitor`'s own structural-Protocol discipline in
`turn/controller.py`. Anything with `hangover_s`, `in_speech`, and
`subscribe(callback, *, replay_segment=True) -> unsubscribe`
(`atlas.transports.edge.SpeechSignals`, as plan 10-02 left it) satisfies
this module's one function.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

__all__ = ["wait_for_end_of_speech"]


async def wait_for_end_of_speech(
    signals: Any,
    *,
    hangover_s: float,
    already_ended_counts: bool,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Return the arrival time of the `vad.end` that ends this wait.

    Subscribes with `replay_segment=False` -- this function only ever
    reacts to events published *after* it starts watching, never to a
    segment's history, so it cannot report a stale `vad.end` from before
    it was called. Always unsubscribes in `finally`, whether it returns
    normally or is cancelled (`turn/controller.py` cancels the watch task
    on every return path once a turn's drain is otherwise done).

    `already_ended_counts` decides what "speech already ended before this
    call started watching" (`signals.in_speech` already `False` at the
    moment of subscribing) means:

    - `True`: that state itself satisfies the wait, and this returns at
      once, with no event ever required. This is the wake-only first-drain
      case (260922-woc): the segment closed before speech-to-text even
      opened, so there is nothing further to wait for.
    - `False`: that state is ignored, and this function waits for a
      genuinely new `vad.end` to arrive through the subscription. This is
      the second-drain case: the operator's segment already closed once
      (on the wake phrase alone), but the command is still coming in a
      new segment, and only that new segment's own end should finalize.

    `hangover_s` (D-11) is the false-end guard: once a `vad.end` arrives,
    a `vad.start` published within `hangover_s` seconds of it cancels that
    candidate -- the operator paused, not stopped -- and this function goes
    back to waiting for the *following* `vad.end`. `hangover_s <= 0`
    (the default, D-10) returns the instant a `vad.end` arrives, with no
    added wait at all.
    """
    queue: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue()

    def _on_event(event: dict[str, Any]) -> None:
        queue.put_nowait(event)

    unsubscribe = signals.subscribe(_on_event, replay_segment=False)
    try:
        if already_ended_counts and not signals.in_speech:
            return clock()

        pending_end_at: float | None = None
        while True:
            if pending_end_at is None:
                event = await queue.get()
                if event.get("type") != "vad.end":
                    continue
                pending_end_at = clock()
                if hangover_s <= 0:
                    return pending_end_at
                continue

            # A candidate `vad.end` is pending and `hangover_s > 0`: give a
            # `vad.start` this long to arrive and cancel it.
            try:
                event = await asyncio.wait_for(queue.get(), timeout=hangover_s)
            except asyncio.TimeoutError:
                return pending_end_at
            if event.get("type") == "vad.start":
                pending_end_at = None  # cancelled -- wait for the following vad.end
            elif event.get("type") == "vad.end":
                pending_end_at = clock()  # a fresh candidate; hangover restarts
    finally:
        unsubscribe()
