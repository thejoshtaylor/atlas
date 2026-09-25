"""Reply audio playback through the XVF3800's own output stream (D-14).

**Deviation from this plan's own text, required by the orchestrator
(10-SPIKE.md hardware finding 1, and plan 10-08's `Capture`):** the
XVF3800 (firmware 2.0.6, USB `2886:001a`) delivers capture frames only
while a playback stream is open on the same device at the same time.
Plan 10-08's `Capture` therefore already opens and keeps running, for its
whole process lifetime, one `RawOutputStream` on the array -- and exposes
`enqueue_playback(bytes)` as the single seam any reply audio must go
through. A second output handle on the same hardware would conflict with
that one. `Playback` never opens a stream of its own; it writes duplicated
-to-stereo PCM16 through the injected `enqueue` callable instead (in
production, `capture.enqueue_playback`), so the reply leaves through the
array's own aux output and the XVF3800's on-chip AEC gets a real echo
reference (D-14).

Because `Capture` -- not `Playback` -- owns the output stream and its
callback, the "pads with zeros on underflow, never blocks" half of the
original plan text is already `Capture`'s own, already-proven behavior
(10-08-SUMMARY.md, `test_output_callback_writes_zeros_when_playback_queue_
is_empty`). `Playback`'s own job is narrower: convert mono to stereo, keep
an odd trailing byte for the next call, and bound how much reply audio it
will forward before that stream has a chance to drain (T-10-32) --
enforced here via a decaying byte-budget estimate (the amount of audio
`Capture`'s own real-time output callback has very likely already played,
given how much time passed since this module last forwarded anything),
since `Capture`'s own playback queue has no bound of its own to check.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2
DEFAULT_MAX_BUFFER_S = 30.0


class Playback:
    def __init__(
        self,
        enqueue: Callable[[bytes], None],
        *,
        sample_rate: int = 16000,
        channels: int = 2,
        max_buffer_s: float = DEFAULT_MAX_BUFFER_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._enqueue = enqueue
        self._sample_rate = sample_rate
        self._channels = channels
        self._clock = clock
        self._bytes_per_second = sample_rate * channels * _BYTES_PER_SAMPLE
        self._max_buffer_bytes = int(max_buffer_s * self._bytes_per_second)
        self._leftover = b""
        self._buffered_bytes = 0
        self._last_update_at: "float | None" = None
        self.dropped_bytes = 0

    def start(self) -> None:
        """No-op: `Playback` owns no stream of its own to start -- kept
        for interface symmetry with `Capture.start()`/`stop()`."""

    def stop(self) -> None:
        """No-op, for the same reason as `start()`."""

    def _decay_buffered_estimate(self) -> None:
        now = self._clock()
        if self._last_update_at is None:
            self._last_update_at = now
            return
        elapsed = max(0.0, now - self._last_update_at)
        self._last_update_at = now
        drained = int(elapsed * self._bytes_per_second)
        self._buffered_bytes = max(0, self._buffered_bytes - drained)

    async def write(self, mono_pcm16: bytes) -> None:
        """Duplicate `mono_pcm16` (PCM16, one channel) into both output
        channels -- samples `[a, b]` become `[a, a, b, b]` -- and forward
        the result through the injected `enqueue` sink. An odd trailing
        byte is kept for the next call, never dropped or misaligned
        mid-sample. When the estimated backlog already forwarded would
        exceed `max_buffer_s`, the oldest part of this call's own bytes is
        dropped (counted in `dropped_bytes`) rather than growing the
        backlog further (T-10-32)."""
        data = self._leftover + mono_pcm16
        usable = len(data) - (len(data) % 2)
        self._leftover = data[usable:]
        mono = data[:usable]

        samples = (mono[i : i + 2] for i in range(0, len(mono), 2))
        stereo_bytes = b"".join(sample + sample for sample in samples)

        self._decay_buffered_estimate()
        overflow = (self._buffered_bytes + len(stereo_bytes)) - self._max_buffer_bytes
        if overflow > 0:
            drop = min(overflow, len(stereo_bytes))
            logger.warning(
                "playback buffer over %.0fs -- dropping %d bytes of the oldest reply audio",
                self._max_buffer_bytes / self._bytes_per_second,
                drop,
            )
            stereo_bytes = stereo_bytes[drop:]
            self.dropped_bytes += drop

        self._buffered_bytes += len(stereo_bytes)
        if stereo_bytes:
            self._enqueue(stereo_bytes)
