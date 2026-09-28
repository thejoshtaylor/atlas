"""LED ring control for the XVF3800 (260928-m11).

`LedController` shows the server's turn state on the 12-LED ring. Four rules
shape it:

- The ring stays dark until the wake word. The firmware default effect
  lights the ring for any voice, the television too. The service writes
  `idle` (effect 0) at start, at every session end, and at stop.
- USB writes stay off the event loop. One worker thread runs every write,
  in submit order. `set_state` only schedules work and never blocks the
  client receive loop.
- A failure never reaches the audio path. Every write error is caught. The
  controller logs one warning per failure streak and looks up the device
  again on the next write.
- The replying color stays until the reply audio drains. The server sends
  the whole reply in milliseconds and then sends `idle`. The controller
  waits for `pending_playback_s` to reach zero before it turns the ring off.

The three colors and the animation timing are module constants. The edge
config has no LED section.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from atlas_edge import protocol, xvf3800

logger = logging.getLogger(__name__)

RING_LEDS = 12

EFFECT_OFF = 0
EFFECT_SINGLE_COLOR = 3
EFFECT_RING = 5

LISTENING_COLOR = 0x0000FF  # blue
REPLYING_COLOR = 0x00FF00  # green
THINKING_COLOR = 0xFFFFFF  # white
# Brightness of the head LED and each LED behind it, head first.
THINKING_TAIL = (1.0, 0.45, 0.2, 0.08)
FRAME_INTERVAL_S = 1 / 15


def _scale(color: int, factor: float) -> int:
    """Scale each 8-bit channel of a 0xRRGGBB color by `factor`."""
    red = int(((color >> 16) & 0xFF) * factor)
    green = int(((color >> 8) & 0xFF) * factor)
    blue = int((color & 0xFF) * factor)
    return (red << 16) | (green << 8) | blue


def ring_frame(head: int, color: int = THINKING_COLOR) -> list[int]:
    """One animation frame: 12 colors. LED `head` has the full color. The
    LEDs behind it fade along `THINKING_TAIL`. All other LEDs are off."""
    frame = [0] * RING_LEDS
    for offset, level in enumerate(THINKING_TAIL):
        frame[(head - offset) % RING_LEDS] = _scale(color, level)
    return frame


def _pack_ring(colors: list[int]) -> bytes:
    return struct.pack(f"<{RING_LEDS}I", *colors)


class LedController:
    def __init__(
        self,
        find_device: Callable[[], Any] = xvf3800.find_device,
        *,
        pending_playback_s: "Callable[[], float] | None" = None,
        frame_interval_s: float = FRAME_INTERVAL_S,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._find_device = find_device
        self._pending_playback_s = pending_playback_s
        self._frame_interval_s = frame_interval_s
        self._sleep = sleep
        # One worker keeps the writes in submit order. A cancelled frame that
        # is already in flight lands before the next state's writes.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atlas-edge-led")
        # The device lookup is lazy, so construction never touches USB.
        self._device: Any = None
        self._requested: str | None = None
        self._shown: str | None = None
        self._task: "asyncio.Task[None] | None" = None
        self._fail_streak = 0

    def set_state(self, state: str) -> None:
        """Show `state`. Synchronous and never blocking: it cancels the
        running task and schedules a new one. A repeat of the requested
        state and an unknown state do nothing."""
        if state not in protocol.LED_STATES:
            logger.warning("led: unknown state %r ignored", state)
            return
        if state == self._requested:
            return
        self._requested = state
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = asyncio.get_running_loop().create_task(self._run(state))

    async def wait(self) -> None:
        """Wait for the current task. It returns after the writes of a
        solid state. It does not return while the thinking animation runs.
        Tests and `close` use it."""
        task = self._task
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            if not task.cancelled():
                raise

    async def close(self) -> None:
        """Stop the animation, turn the ring off at once (the service is
        stopping, so there is no drain wait), and release the worker."""
        task = self._task
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
        self._requested = protocol.LED_IDLE
        self._shown = protocol.LED_IDLE
        await self._write((_EFFECT, bytes([EFFECT_OFF])))
        self._executor.shutdown(wait=False)

    async def _run(self, state: str) -> None:
        if self._shown == protocol.LED_REPLYING and state != protocol.LED_REPLYING:
            await self._wait_for_playback_drain()
        if state == protocol.LED_IDLE:
            await self._write((_EFFECT, bytes([EFFECT_OFF])))
            self._shown = state
        elif state in (protocol.LED_LISTENING, protocol.LED_REPLYING):
            color = LISTENING_COLOR if state == protocol.LED_LISTENING else REPLYING_COLOR
            await self._write(
                ("LED_COLOR", struct.pack("<I", color)),
                (_EFFECT, bytes([EFFECT_SINGLE_COLOR])),
            )
            self._shown = state
        else:
            head = 0
            await self._write(
                ("LED_RING_COLOR", _pack_ring(ring_frame(head))),
                (_EFFECT, bytes([EFFECT_RING])),
            )
            self._shown = state
            while True:
                await self._sleep(self._frame_interval_s)
                head = (head + 1) % RING_LEDS
                await self._write(("LED_RING_COLOR", _pack_ring(ring_frame(head))))

    async def _wait_for_playback_drain(self) -> None:
        while True:
            remaining = self._read_pending()
            if remaining <= 0:
                return
            await self._sleep(remaining)

    def _read_pending(self) -> float:
        if self._pending_playback_s is None:
            return 0.0
        try:
            return float(self._pending_playback_s())
        except Exception:
            return 0.0

    async def _write(self, *writes: "tuple[str, bytes]") -> None:
        """Run the writes on the worker thread. Never raises, except for
        cancellation. Logs a warning at the start of a failure streak."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._write_blocking, writes)
        except Exception as exc:
            self._fail_streak += 1
            if self._fail_streak == 1:
                logger.warning("led: USB write failed: %s", exc)
            else:
                logger.debug("led: USB write failed again: %s", exc)
            return
        self._fail_streak = 0

    def _write_blocking(self, writes: "tuple[tuple[str, bytes], ...]") -> None:
        try:
            if self._device is None:
                self._device = self._find_device()
            for name, payload in writes:
                xvf3800.write_parameter(self._device, xvf3800.PARAMETERS[name], payload)
        except Exception:
            # Look the device up again on the next write.
            self._device = None
            raise


_EFFECT = "LED_EFFECT"
