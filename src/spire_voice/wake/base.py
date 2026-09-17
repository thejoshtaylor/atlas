"""The thin `WakeDetector` protocol, following `providers/base.py`'s shape.

Three members, so both engines (Vosk here, openWakeWord later) fit behind
it with no engine-specific detail leaking through:

- `process()` -- accepts one chunk of 16 kHz mono PCM16 and returns either
  nothing or a `WakeHit` carrying the score that produced it.
- `reset()` -- clears whatever state `process()` has accumulated so far,
  without releasing what `close()` would (WR-02, code review). The live
  pipeline never calls this: production audio is one continuous stream
  with no recording boundaries to reset across. It exists for
  `scripts/score_wake_engines.py`, which reuses one detector instance
  across many independent corpus recordings -- without a reset between
  them, a still-open decode from one recording's trailing audio can fire
  (or fail to fire) attributed to the *next* recording, corrupting the
  threshold-sweep evidence DBG-04 exists to produce.
- `close()` -- releases whatever the engine holds (a model, a decoder,
  native memory).

`WakeError` is raised, not returned, mirroring every other provider error
in this codebase (`SttError`, `BrainError`, `TtsError`): a caller cannot
silently continue past a failed wake engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class WakeError(Exception):
    """Raised when a wake engine cannot be constructed or fails outright --
    a missing model directory names itself here rather than producing a
    detector that silently never fires."""


@dataclass(frozen=True)
class WakeHit:
    """One wake-word detection: the score that produced it."""

    score: float


class WakeDetector(Protocol):
    def process(self, chunk: bytes) -> WakeHit | None:
        """Accept one chunk of 16 kHz mono PCM16; return a `WakeHit` if this
        chunk completed a detection, else `None`."""
        ...

    def reset(self) -> None:
        """Clear state accumulated across `process()` calls so far, ready
        for the next independent recording (see the module docstring)."""
        ...

    def close(self) -> None:
        """Release whatever this engine holds."""
        ...
