"""`PanelOutbox`: the bounded queue of frames waiting for one Mac (Phase 15, D-13).

The governing rule is the one `session/observers.py` states: a Mac, its Wi-Fi,
or its sleep state must never slow a turn. `put` is synchronous and never
awaits. When the queue is full it drops its oldest frame and counts the drop.
One sender task per connection (`DesktopHub._send_loop`) is the only reader.

`coalesce_key` is stored with each frame. Plan 15-05 uses it to replace a
stale partial transcript and to choose what to drop first. The signature of
`put` does not change then.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import NamedTuple


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
        """Queue one frame. Never awaits. A full queue drops its oldest frame."""
        if len(self._frames) >= self._max_frames:
            self._frames.popleft()
            self.dropped += 1
        self._frames.append(QueuedFrame(frame, frame_type, coalesce_key))
        self._ready.set()

    async def get(self) -> str:
        """Wait for a frame and return its text, oldest first."""
        while not self._frames:
            self._ready.clear()
            await self._ready.wait()
        queued = self._frames.popleft()
        if not self._frames:
            self._ready.clear()
        return queued.frame
