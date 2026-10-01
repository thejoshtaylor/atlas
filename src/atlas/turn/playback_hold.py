"""Keep an answer's turn open until its audio has played.

TTS is batch, so `_speak_direct` writes a whole reply to the Pi in a burst
and returns long before the reply is heard. Without a hold, the turn ends
and the barge-in listener ends with it, while the Pi still has seconds of
audio queued. The hold is the only way an interrupt can reach that queued
audio (Phase 13, RESEARCH Finding 1).

Every read is duck-typed. This module imports nothing from the source
runner, which keeps the controller's Protocol discipline.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Callable


async def hold_for_playback(barge_in: Any, *, clock: Callable[[], float] = time.monotonic) -> bool:
    """Wait until the estimated playback end or an interrupt.

    Returns True when the monitor was interrupted, and False when the reply
    played to its end, or when this monitor does not hold at all.
    """
    if not getattr(barge_in, "holds_for_playback", False):
        return False
    playback = getattr(barge_in, "playback", None)
    playing_until = getattr(playback, "playing_until", None)
    if playing_until is None:
        return False
    if barge_in.interrupt_requested:
        return True
    remaining = playing_until - clock()
    if remaining > 0:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(barge_in.interrupted.wait(), timeout=remaining)
    return bool(barge_in.interrupt_requested)
