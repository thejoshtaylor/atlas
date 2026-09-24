"""Per-turn stage timing: eight timestamps and their derived durations,
nothing else.

Field names say what they measure without needing a comment, matching
`config.example.yaml`'s own habit. Per the plan's privacy prohibition, this
module records timings only -- no transcript text, no reply text, and no
audio bytes ever reach a log line, a browser event, or a file here. The
recorded-session store with its own retention schedule is Phase 2 (DBG-01,
DBG-06); this is the narrower, always-on measurement that proves the
1.5 second budget, not a substitute for it.

The eight stages, in the order a turn reaches them: the turn starting, the
speech-to-text socket opening, the first partial transcript arriving, the
final transcript arriving, the first brain token arriving, the tool calls
completing, the first audio chunk leaving for the browser, and the first
chunk of the answer utterance specifically leaving for the browser. A turn
closed early by either VOICE-08 guard reaches only a prefix of these -- an
unset (`None`) stage is what "never reached" looks like, distinguishable
from a reached stage that happened to take zero time.

`first_audio_at` and `answer_audio_at` are deliberately two fields, not one
field marked twice. This phase adds a filler utterance that can play before
the answer, so "the first audio the operator heard" and "the answer" are no
longer always the same event: `first_audio_at` is the first audio of any
kind, a holding phrase included, and `answer_audio_at` is the first audio of
the answer utterance specifically. A filler starting quickly must not be
able to make a slow answer look fast, which is why each is set from its own
call site rather than sharing one.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atlas.timing")

# Encounter order: each entry's duration is measured from the previous
# *reached* stage, not necessarily the literally-preceding field, so a stage
# a turn skipped does not corrupt the duration of the one after it.
_STAGE_ORDER = (
    "turn_started_at",
    "stt_socket_open_at",
    "first_partial_at",
    "stt_final_at",
    "brain_first_token_at",
    "tool_rounds_done_at",
    "first_audio_at",
    "answer_audio_at",
)


@dataclass
class TurnTimings:
    """One turn's stage timestamps, in `time.monotonic()` seconds.

    `turn_outcome` distinguishes a turn that reached speech normally
    (`"completed"`) from the three ways VOICE-08/the round cap end one early
    (`"empty_transcript"`, `"timeout"`, `"round_cap"`) -- the closure
    mechanism controller.py picks, not something this module infers.
    """

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn_outcome: str = "unknown"
    turn_started_at: float | None = None
    stt_socket_open_at: float | None = None
    first_partial_at: float | None = None
    stt_final_at: float | None = None
    brain_first_token_at: float | None = None
    tool_rounds_done_at: float | None = None
    first_audio_at: float | None = None
    answer_audio_at: float | None = None

    def mark_turn_started(self) -> None:
        """Record the moment the turn began -- the mic toggle, in Phase 1."""
        self.turn_started_at = time.monotonic()

    def mark_stt_socket_open(self) -> None:
        """Record the moment the turn started consuming the STT stream."""
        self.stt_socket_open_at = time.monotonic()

    def mark_first_partial(self) -> None:
        """Record the first partial transcript forwarded to the page.

        Idempotent: only the first call after construction has any effect,
        matching every other mark's "this stage happened once" contract.
        """
        if self.first_partial_at is None:
            self.first_partial_at = time.monotonic()

    def mark_stt_final(self) -> None:
        """Record the moment the final transcript arrived."""
        self.stt_final_at = time.monotonic()

    def mark_brain_first_token(self) -> None:
        """Record the first `chat()` round returning.

        `BrainProvider.chat()`'s public contract returns one assembled
        `BrainReply` per round rather than per-token deltas (the streaming
        happens inside the provider; the controller never sees it) -- this
        is the earliest point a turn genuinely reaches "the model answered,"
        and the closest available proxy for "first token" without changing
        that contract. Idempotent, so a multi-round tool loop times only the
        first round.
        """
        if self.brain_first_token_at is None:
            self.brain_first_token_at = time.monotonic()

    def mark_tool_rounds_done(self) -> None:
        """Record the moment the tool-calling loop settled on a reply.

        Marked whether that reply came from zero tool calls, one round, or
        hitting the round cap -- "done" means "no more rounds will run."
        """
        self.tool_rounds_done_at = time.monotonic()

    def mark_first_audio(self) -> None:
        """Record the moment the first reply-audio chunk left for the browser.

        Idempotent, as of this phase -- deliberately, not by copying the
        unguarded style this field used in Phase 1. Phase 1 had exactly one
        utterance per turn, so a second call was unreachable and the guard
        would have been dead code. This phase adds the filler, so `_speak`
        now has two call sites that can reach this mark (filler, then
        answer); without the guard, the answer's first chunk would silently
        overwrite the filler's timestamp and this field would stop meaning
        "the first audio the operator heard" -- it would mean "the last
        utterance whose first chunk happened to mark it." The reason it was
        safe to leave unguarded no longer holds, so it is guarded now.
        """
        if self.first_audio_at is None:
            self.first_audio_at = time.monotonic()

    def mark_answer_audio(self) -> None:
        """Record the moment the first chunk of the ANSWER utterance left
        for the browser -- never a filler.

        Its only caller is the answer branch of `_speak` in
        `turn/controller.py`. Idempotent for the same reason
        `mark_first_audio` is: one answer utterance per turn, one mark.
        """
        if self.answer_audio_at is None:
            self.answer_audio_at = time.monotonic()

    @property
    def end_of_speech_to_first_audio_ms(self) -> float | None:
        """The measured budget number, or `None` until both marks exist.

        This is the number VOICE-02's 1.5-second criterion is about, and it
        is derived from `stt_final_at`/`first_audio_at` -- never measured a
        second, separate way.
        """
        if self.stt_final_at is None or self.first_audio_at is None:
            return None
        return (self.first_audio_at - self.stt_final_at) * 1000

    @property
    def end_of_speech_to_answer_audio_ms(self) -> float | None:
        """The answer's own budget number, or `None` until both marks exist.

        Derived from `stt_final_at`/`answer_audio_at`, mirroring
        `end_of_speech_to_first_audio_ms` exactly. Never rounded: the 1.5
        second budget is compared against this unrounded float, and a
        rounded duration that crossed the budget by a fraction would be
        reported as meeting it.
        """
        if self.stt_final_at is None or self.answer_audio_at is None:
            return None
        return (self.answer_audio_at - self.stt_final_at) * 1000

    def stage_durations_ms(self) -> dict[str, float | None]:
        """Each stage's duration since the previous *reached* stage.

        `None` for a stage that was never reached, or for the first reached
        stage (nothing precedes it to measure from) -- never a fabricated
        zero. This is the browser panel's only source of numbers: it renders
        this dict's values directly and computes no duration of its own.
        """
        durations: dict[str, float | None] = {}
        previous_value: float | None = None
        for name in _STAGE_ORDER:
            value = getattr(self, name)
            durations[name] = None if value is None or previous_value is None else (value - previous_value) * 1000
            if value is not None:
                previous_value = value
        return durations

    def log(self) -> None:
        """Emit the one structured log line this turn produces.

        Every field here is a timestamp, a duration, an outcome label, or
        the turn id -- never the words that were spoken or the words spoken
        back.
        """
        logger.info(
            "turn timing",
            extra={
                "turn_id": self.turn_id,
                "turn_outcome": self.turn_outcome,
                "turn_started_at": self.turn_started_at,
                "stt_socket_open_at": self.stt_socket_open_at,
                "first_partial_at": self.first_partial_at,
                "stt_final_at": self.stt_final_at,
                "brain_first_token_at": self.brain_first_token_at,
                "tool_rounds_done_at": self.tool_rounds_done_at,
                "first_audio_at": self.first_audio_at,
                "answer_audio_at": self.answer_audio_at,
                "end_of_speech_to_first_audio_ms": self.end_of_speech_to_first_audio_ms,
                "end_of_speech_to_answer_audio_ms": self.end_of_speech_to_answer_audio_ms,
            },
        )

    def to_event(self) -> dict[str, Any]:
        """The stage record sent to the browser over the transport's event
        channel -- the same values `log()` emits, shaped for
        `handleTransportMessage`'s dispatch (`type`) and pre-derived
        (`stage_durations_ms`) so the page never subtracts a timestamp
        itself.
        """
        return {
            "type": "turn.timing",
            "turn_id": self.turn_id,
            "turn_outcome": self.turn_outcome,
            "stage_durations_ms": self.stage_durations_ms(),
            "end_of_speech_to_first_audio_ms": self.end_of_speech_to_first_audio_ms,
            "end_of_speech_to_answer_audio_ms": self.end_of_speech_to_answer_audio_ms,
        }
