"""Music from librespot, mixed into the array's one output stream.

librespot cannot open the XVF3800 output device, because `Capture` holds
its only output stream. So librespot writes PCM into a FIFO instead. This
module reads that FIFO and mixes the audio into `Capture`'s output
callback.

The reader opens the FIFO with O_RDWR. It never sees end of file when
librespot closes its end, and while atlas-edge runs, the open in
librespot never blocks. Music becomes mono and goes to every channel, as
reply audio does in `playback.py`. The echo canceller of the XVF3800 then
gets the same reference on both channels (D-14).

The reader stops reading when its buffer is full. The pipe then fills, and
the blocking write in librespot sets the playback speed to real time. The
output callback never waits for the reader. If the FIFO is missing or
invalid, the mixer stays off and logs one warning. Music must never stop
the microphone.
"""

from __future__ import annotations

import logging
import os
import select
import stat
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

LIBRESPOT_RATE = 44100  # librespot writes 44.1 kHz, S16LE, stereo
DEFAULT_DUCK_LEVEL = 0.05
DEFAULT_DUCK_RAMP_MS = 150
DEFAULT_MAX_BUFFER_S = 0.25

_FIFO_MODE = 0o620  # umask removes group write from os.mkfifo, so chmod after
_READ_SIZE = 8192
_FIR_TAPS = 97
_FIR_CUTOFF_HZ = 7000.0
_INT16_MAX = 32767


class _Resampler:
    """Streaming mono resampler. A windowed-sinc low-pass filter removes
    content above the output Nyquist rate. Linear interpolation then picks
    the output samples. Filter history and read position carry over between
    calls, so the output does not depend on how the input is split."""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        cutoff = _FIR_CUTOFF_HZ / in_rate
        center = (_FIR_TAPS - 1) / 2
        n = np.arange(_FIR_TAPS) - center
        taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.blackman(_FIR_TAPS)
        self._taps = (taps / taps.sum()).astype(np.float64)  # unity gain at DC
        self._history = np.zeros(_FIR_TAPS - 1, dtype=np.float64)
        self._step = in_rate / out_rate
        self._carry = np.zeros(1, dtype=np.float64)  # last filtered sample
        self._pos = 0.0

    def process(self, mono: np.ndarray) -> np.ndarray:
        if mono.size == 0:
            return np.zeros(0, dtype=np.float32)
        x = np.concatenate([self._history, mono.astype(np.float64)])
        self._history = x[-(_FIR_TAPS - 1) :]
        filtered = np.convolve(x, self._taps, mode="valid")
        y = np.concatenate([self._carry, filtered])
        last = len(y) - 1
        if self._pos >= last:
            count = 0
        else:
            count = int(np.ceil((last - self._pos) / self._step))
        positions = self._pos + self._step * np.arange(count)
        positions = positions[positions < last]
        out = np.interp(positions, np.arange(len(y)), y)
        self._pos = self._pos + self._step * len(positions) - last
        self._carry = y[-1:]
        return out.astype(np.float32)


class MusicMixer:
    """Reads S16LE stereo PCM from a FIFO and mixes it, ducked, into the
    blocks that `Capture._output_callback` builds. `duck_active` returns
    True while the music must be quiet. It runs on the audio thread, so it
    must be a plain attribute read."""

    def __init__(
        self,
        fifo_path: str,
        *,
        duck_active: Callable[[], bool],
        duck_level: float = DEFAULT_DUCK_LEVEL,
        ramp_ms: float = DEFAULT_DUCK_RAMP_MS,
        out_rate: int = 16000,
        channels: int = 2,
        max_buffer_s: float = DEFAULT_MAX_BUFFER_S,
    ) -> None:
        self._path = fifo_path
        self._duck_active = duck_active
        self._duck_level = duck_level
        self._channels = channels
        self._max_buffered = int(max_buffer_s * out_rate)
        ramp_samples = ramp_ms * out_rate / 1000
        # Gain change per sample. None means an instant change.
        self._gain_step = (1.0 - duck_level) / ramp_samples if ramp_samples > 0 else None

        self._resampler = _Resampler(LIBRESPOT_RATE, out_rate)
        self._partial = b""
        self._lock = threading.Lock()
        self._buffer: "deque[np.ndarray]" = deque()
        self._buffered = 0
        self._gain = 1.0

        self._fd: "int | None" = None
        self._thread: "threading.Thread | None" = None
        self._stop_flag = threading.Event()

    @property
    def gain(self) -> float:
        return self._gain

    def _push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        with self._lock:
            self._buffer.append(samples)
            self._buffered += samples.size

    def feed(self, raw: bytes) -> None:
        data = self._partial + raw
        usable = len(data) - len(data) % 4
        self._partial = data[usable:]
        if usable == 0:
            return
        pcm = np.frombuffer(data[:usable], dtype="<i2").reshape(-1, 2)
        mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
        self._push(self._resampler.process(mono))

    def start(self) -> None:
        try:
            if not os.path.exists(self._path):
                os.mkfifo(self._path, _FIFO_MODE)
                os.chmod(self._path, _FIFO_MODE)
            elif not stat.S_ISFIFO(os.stat(self._path).st_mode):
                logger.warning("music is off: %s is not a FIFO", self._path)
                return
            self._fd = os.open(self._path, os.O_RDWR | os.O_NONBLOCK)
        except OSError as exc:
            logger.warning("music is off: cannot use the FIFO %s: %s", self._path, exc)
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._read_loop, name="music-fifo", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _read_loop(self) -> None:
        fd = self._fd
        while not self._stop_flag.is_set():
            if self._buffered >= self._max_buffered:
                time.sleep(0.005)  # backpressure: leave the bytes in the pipe
                continue
            try:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                raw = os.read(fd, _READ_SIZE)
            except BlockingIOError:
                continue
            except (OSError, ValueError) as exc:
                logger.warning("music reader stopped: %s", exc)
                return
            if not raw:
                time.sleep(0.005)
                continue
            self.feed(raw)

    def _gains(self, frames: int, ducked: bool) -> np.ndarray:
        target = self._duck_level if ducked else 1.0
        if self._gain_step is None:
            gains = np.full(frames, target, dtype=np.float64)
        else:
            delta = self._gain_step * np.arange(1, frames + 1)
            if target < self._gain:
                gains = np.maximum(self._gain - delta, target)
            else:
                gains = np.minimum(self._gain + delta, target)
        self._gain = float(gains[-1])
        return gains

    def mix(self, block: bytes, reply_active: bool) -> bytes:
        """Add buffered music to `block` (int16, interleaved). Runs on the
        audio thread. Returns `block` itself when no music is buffered."""
        frames = len(block) // (2 * self._channels)
        if frames == 0:
            return block
        gains = self._gains(frames, reply_active or bool(self._duck_active()))

        with self._lock:
            take = min(frames, self._buffered)
            parts = []
            remaining = take
            while remaining > 0:
                head = self._buffer[0]
                if head.size <= remaining:
                    parts.append(self._buffer.popleft())
                    remaining -= head.size
                else:
                    parts.append(head[:remaining])
                    self._buffer[0] = head[remaining:]
                    remaining = 0
            self._buffered -= take
        if take == 0:
            return block

        music = np.zeros(frames, dtype=np.float64)
        music[:take] = np.concatenate(parts)
        scaled = np.rint(music * gains * _INT16_MAX).astype(np.int32)
        pcm = np.frombuffer(block, dtype="<i2").astype(np.int32).reshape(frames, self._channels)
        mixed = np.clip(pcm + scaled[:, None], -32768, _INT16_MAX)
        return mixed.astype("<i2").tobytes()
