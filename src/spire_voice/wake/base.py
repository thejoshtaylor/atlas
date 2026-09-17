"""The thin `WakeDetector` protocol, following `providers/base.py`'s shape.

Two members and nothing else, so both engines (Vosk here, openWakeWord
later) fit behind it with no engine-specific detail leaking through:

- `process()` -- accepts one chunk of 16 kHz mono PCM16 and returns either
  nothing or a `WakeHit` carrying the score that produced it.
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

    def close(self) -> None:
        """Release whatever this engine holds."""
        ...
