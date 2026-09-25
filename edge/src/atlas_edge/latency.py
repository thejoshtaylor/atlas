"""The Pi's own capture-to-send delay report (ROADMAP Phase 10 delay
budget): `SendDelayWindow` accumulates one `record(captured_at, sent_at)`
per live frame actually sent, and `drain_message()` turns whatever
accumulated since the last drain into one `protocol.latency(...)` message
-- or `None` when no live frame was sent in that window, since a message
with no data would misreport a silent second as a zero-delay one.

This module holds no timer of its own: whoever calls `drain_message()` on
a 1-second cadence (`__main__.py`'s own ticker) is what makes "one window"
mean one second in practice. `record`/`drain_message` themselves are pure
accounting over whatever calls arrive between two drains.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from atlas_edge import protocol


class SendDelayWindow:
    """`window_s` and `clock` are kept for interface symmetry with the
    plan's own signature and for a future window-aware caller; the current
    accounting is deliberately clock-agnostic -- `record` just accumulates
    delays, and whichever cadence calls `drain_message()` defines what
    "one window" means."""

    def __init__(self, window_s: float = 1.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._window_s = window_s
        self._clock = clock
        self._delays_ms: "list[float]" = []

    def record(self, captured_at: float, sent_at: float) -> None:
        """`captured_at`/`sent_at` are both `Capture`'s own monotonic
        clock values -- the delay between a frame being captured and this
        Pi finishing sending it, in milliseconds."""
        self._delays_ms.append((sent_at - captured_at) * 1000.0)

    def drain_message(self) -> "str | None":
        """The window's p50/p95/max delay and frame count as one
        `protocol.latency(...)` message, then reset for the next window.
        `None` when no frame was recorded -- a quiet second reports
        nothing rather than a manufactured zero."""
        if not self._delays_ms:
            return None
        p50, p95 = np.percentile(self._delays_ms, [50, 95])
        max_ms = max(self._delays_ms)
        frames = len(self._delays_ms)
        self._delays_ms = []
        return protocol.latency(float(p50), float(p95), float(max_ms), frames)
