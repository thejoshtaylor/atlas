"""sounddevice capture of the XVF3800 -- full-duplex, per 10-SPIKE.md
hardware finding 1: the array (firmware 2.0.6, USB `2886:001a`) delivers
*no capture frames* unless a playback stream is open on the same device at
the same time (a bare `arecord hw:Array` fails with `EIO`; a bare
`sd.rec()` blocks forever; spike fix, commit `f61b8ff`). `Capture` therefore
also opens, and keeps running for its whole lifetime, an output stream on
the same device -- filled from an internal playback queue, and silence
when that queue is empty. `enqueue_playback` is the one seam plan 10-09's
`Playback` writes reply audio through, so it never opens a second output
handle on hardware that cannot support one.

`sounddevice` is imported lazily (inside `_default_stream_factory`), never
at module scope, so this module imports cleanly on a dev host with no
XVF3800 and no `sounddevice` native library installed.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, AsyncIterator, Callable

logger = logging.getLogger(__name__)

# The XVF3800 (firmware 2.0.6, USB 2886:001a) ships 2ch/16kHz/S16_LE --
# GitHub issue #4 and protocol v1's FRAME_SAMPLES (256) both assume this.
SAMPLE_RATE = 16000
CHANNELS = 2
FRAME_SAMPLES = 256

_BYTES_PER_SAMPLE = 2  # int16


class DeviceNotFound(Exception):
    """Raised by `find_device` with the available device names when no
    device matches -- never falls back to the ALSA/PortAudio default
    (10-SPIKE.md hardware finding 2: the Pi's default routes through
    PipeWire, not the array)."""


def find_device(name: str, *, kind: str = "input") -> int:
    """The index of the first device whose name contains `name`
    (case-insensitive) with at least 2 channels for `kind` ("input" or
    "output"). Never the ALSA/PortAudio default device."""
    import sounddevice as sd

    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()
    for index, info in enumerate(devices):
        if name.lower() in info["name"].lower() and info[channel_key] >= 2:
            return index
    names = [info["name"] for info in devices]
    raise DeviceNotFound(f"no {kind} device named like {name!r} among: {names}")


class Capture:
    """Opens once for the process's life (`service.py` never reopens it
    on reconnect). `stream_factory(kind, **kwargs)` -- `kind` is
    `"input"` or `"output"` -- is injectable for tests; production code
    leaves it `None` and gets real `sounddevice` streams."""

    def __init__(
        self,
        device_name: str,
        *,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        frame_samples: int = FRAME_SAMPLES,
        stream_factory: "Callable[..., Any] | None" = None,
        max_queued: int = 256,
    ) -> None:
        self._device_name = device_name
        self._sample_rate = sample_rate
        self._channels = channels
        self._frame_samples = frame_samples
        self._stream_factory = stream_factory or self._default_stream_factory
        self._max_queued = max_queued
        self._block_duration_s = frame_samples / sample_rate

        self._input_stream: Any = None
        self._output_stream: Any = None
        self._loop = None
        self._queue: "deque[tuple[bytes, float]]" = deque()
        self._queue_event = None
        self.dropped = 0
        self._last_status_warn_at = 0.0

        # Read on the output-stream thread, written from whichever thread
        # calls `enqueue_playback` -- both `deque.append`/`popleft` are
        # atomic under the GIL, the same single-producer/single-consumer
        # assumption the input side's own queue already relies on.
        self._playback_queue: "deque[bytes]" = deque()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    def enqueue_playback(self, data: bytes) -> None:
        """Queue raw PCM16 bytes to play out through the shared output
        stream. This is the seam plan 10-09's `Playback` writes reply
        audio through -- the XVF3800 gets exactly one output handle."""
        self._playback_queue.append(data)

    def _default_stream_factory(self, kind: str, **kwargs: Any) -> Any:
        import sounddevice as sd

        cls = sd.RawInputStream if kind == "input" else sd.RawOutputStream
        return cls(**kwargs)

    def _warn_status(self, status: Any) -> None:
        now = time.monotonic()
        if now - self._last_status_warn_at >= 1.0:
            logger.warning("capture PortAudio status flags: %s", status)
            self._last_status_warn_at = now

    def _output_callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self._warn_status(status)
        needed = frames * self._channels * _BYTES_PER_SAMPLE
        buf = bytearray()
        while len(buf) < needed and self._playback_queue:
            buf += self._playback_queue.popleft()
        if len(buf) > needed:
            leftover = bytes(buf[needed:])
            self._playback_queue.appendleft(leftover)
            buf = buf[:needed]
        elif len(buf) < needed:
            buf += bytes(needed - len(buf))  # silence when the queue is empty
        outdata[:] = bytes(buf)

    def _input_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self._warn_status(status)
        # The reported delay must never understate the true one -- the
        # callback fires after the whole block was captured, so the
        # block's own duration is subtracted back off "now".
        captured_at = time.monotonic() - self._block_duration_s
        data = bytes(indata)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._enqueue, data, captured_at)

    def _enqueue(self, data: bytes, captured_at: float) -> None:
        if len(self._queue) >= self._max_queued:
            self._queue.popleft()
            self.dropped += 1
        self._queue.append((data, captured_at))
        if self._queue_event is not None:
            self._queue_event.set()

    def start(self) -> None:
        import asyncio

        self._loop = asyncio.get_running_loop()
        self._queue_event = asyncio.Event()
        device = find_device(self._device_name, kind="input")

        # Output starts BEFORE input -- the array delivers no capture
        # frames without an open playback stream (10-SPIKE.md finding 1).
        self._output_stream = self._stream_factory(
            "output",
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._frame_samples,
            device=device,
            callback=self._output_callback,
        )
        self._output_stream.start()

        self._input_stream = self._stream_factory(
            "input",
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._frame_samples,
            device=device,
            callback=self._input_callback,
        )
        self._input_stream.start()

    def stop(self) -> None:
        # Input stops before output -- the mirror of start()'s order.
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

    async def frames(self) -> AsyncIterator["tuple[bytes, float]"]:
        assert self._queue_event is not None, "Capture.start() was never called"
        while True:
            while not self._queue:
                await self._queue_event.wait()
                self._queue_event.clear()
            yield self._queue.popleft()
