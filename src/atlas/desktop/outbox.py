"""`PanelOutbox`: the bounded queue of frames waiting for one Mac (Phase 15, D-13).

The governing rule is the one `session/observers.py` states: a Mac, its Wi-Fi,
or its sleep state must never slow a turn. `put` is synchronous and never
awaits. One sender task per connection (`DesktopHub._send_loop`) is the only
reader.

A plain drop-oldest queue is wrong here. It can drop `turn.ended` or
`timer.stopped`, and then the Mac waits for its 30 s watchdog or its 130 s
ring cap (RESEARCH Pitfall 10). D-13's "drop the oldest" therefore applies to
detail frames first: partials, then states, then the final transcript, then
cards. `wake.confirmed`, `turn.ended`, `timer.ringing` and `timer.stopped`
go last, and only when nothing else is left.

A partial transcript replaces the partial at the tail of the queue when both
carry the same `coalesce_key` (the turn id). Order is never changed: a partial
after a `transcript.final` is appended.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import NamedTuple

_PARTIAL = "transcript.partial"
# The frame types to drop first, in order. Any other type that is not
# protected follows them.
_DROP_ORDER = (_PARTIAL, "state", "transcript.final", "card")
_PROTECTED = frozenset({"wake.confirmed", "turn.ended", "timer.ringing", "timer.stopped"})


class QueuedFrame(NamedTuple):
    frame: str
    frame_type: str
    coalesce_key: str | None


class PanelOutbox:
    def __init__(self, max_frames: int = 64) -> None:
        self._max_frames = max_frames
        self._frames: deque[QueuedFrame] = deque()
        self._ready = asyncio.Event()
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._frames)

    def put(self, frame: str, frame_type: str, *, coalesce_key: str | None = None) -> None:
        """Queue one frame. Never awaits. Over the bound, detail goes first."""
        entry = QueuedFrame(frame, frame_type, coalesce_key)
        tail = self._frames[-1] if self._frames else None
        if (
            frame_type == _PARTIAL
            and tail is not None
            and tail.frame_type == _PARTIAL
            and tail.coalesce_key == coalesce_key
        ):
            self._frames[-1] = entry
        else:
            self._frames.append(entry)
            while len(self._frames) > self._max_frames:
                self._drop_one()
        self._ready.set()

    def _drop_one(self) -> None:
        for frame_type in _DROP_ORDER:
            if self._remove_oldest(lambda queued: queued.frame_type == frame_type):
                return
        if self._remove_oldest(lambda queued: queued.frame_type not in _PROTECTED):
            return
        self._frames.popleft()
        self.dropped += 1

    def _remove_oldest(self, matches) -> bool:
        for index, queued in enumerate(self._frames):
            if matches(queued):
                del self._frames[index]
                self.dropped += 1
                return True
        return False

    async def get(self) -> str:
        """Wait for a frame and return its text, oldest first."""
        while not self._frames:
            self._ready.clear()
            await self._ready.wait()
        queued = self._frames.popleft()
        if not self._frames:
            self._ready.clear()
        return queued.frame
