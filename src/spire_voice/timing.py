"""Per-turn stage timing: two timestamps and one derived number, nothing else.

Field names say what they measure without needing a comment, matching
`config.example.yaml`'s own habit. Per the plan's privacy prohibition, this
module records timings only -- no transcript text, no reply text, and no
audio bytes ever reach a log line or a file here. The recorded-session store
with its own retention schedule is Phase 2 (DBG-01, DBG-06); this is the
narrower, always-on measurement that proves the 1.5 second budget, not a
substitute for it.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("spire_voice.timing")


@dataclass
class TurnTimings:
    """One turn's stage timestamps, in `time.monotonic()` seconds."""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stt_final_at: float | None = None
    first_audio_at: float | None = None

    def mark_stt_final(self) -> None:
        """Record the moment the final transcript arrived."""
        self.stt_final_at = time.monotonic()

    def mark_first_audio(self) -> None:
        """Record the moment the first reply-audio chunk left for the browser."""
        self.first_audio_at = time.monotonic()

    @property
    def end_of_speech_to_first_audio_ms(self) -> float | None:
        """The measured budget number, or `None` until both marks exist."""
        if self.stt_final_at is None or self.first_audio_at is None:
            return None
        return (self.first_audio_at - self.stt_final_at) * 1000

    def log(self) -> None:
        """Emit the one structured log line this turn produces.

        Every field here is a timestamp, a duration, or the turn id -- never
        the words that were spoken or the words spoken back.
        """
        logger.info(
            "turn timing",
            extra={
                "turn_id": self.turn_id,
                "stt_final_at": self.stt_final_at,
                "first_audio_at": self.first_audio_at,
                "end_of_speech_to_first_audio_ms": self.end_of_speech_to_first_audio_ms,
            },
        )
