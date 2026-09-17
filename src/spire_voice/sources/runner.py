"""`SourceRunner`: the only thing that turns a wake hit into a `run_turn` call.

`run_turn` itself (`turn/controller.py`) stays exactly as ignorant of wake
words as it was before this plan -- this module is its caller, the same
way the WebSocket and WebRTC routes in `app.py` already are for the two
browser sources.

**VOICE-03 is enforced here, and nowhere else:** `run()` consumes
`source.frames()` in a single continuous loop, forwarding each raw chunk
through the source's own detector-decode step and into the wake detector.
Before a hit, nothing this runner holds is passed to `run_turn_fn` -- the
transcription provider's `stream()` is not called, and its socket is not
opened, until `wake_detector.process()` returns a `WakeHit`. Only then does
this loop await one full turn; while that await is pending, this loop makes
no other call to `source.frames()`, so the turn's own internal read of
`source.frames()` (via `run_turn` -> `stt.stream()`) is the sole consumer
of the underlying queue for the duration of the turn.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from spire_voice.wake.base import WakeDetector

logger = logging.getLogger("spire_voice.sources.runner")


class SourceRunner:
    """Ties one named source to one wake detector and one `run_turn` caller."""

    def __init__(
        self,
        name: str,
        source: Any,
        wake_detector: WakeDetector,
        decode_for_detector: Callable[[bytes], bytes],
        run_turn_fn: Callable[[Any], Awaitable[None]],
    ) -> None:
        self._name = name
        self._source = source
        self._wake_detector = wake_detector
        self._decode_for_detector = decode_for_detector
        self._run_turn_fn = run_turn_fn

    async def run(self) -> None:
        """Consume `source.frames()` until it ends, running one turn per
        wake hit along the way."""
        async for chunk in self._source.frames():
            detector_chunk = self._decode_for_detector(chunk)
            if not detector_chunk:
                continue
            hit = self._wake_detector.process(detector_chunk)
            if hit is None:
                continue
            logger.info("wake hit on source %r (score=%.3f)", self._name, hit.score)
            await self._run_turn_fn(self._source)
