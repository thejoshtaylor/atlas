"""Per-turn stage timing: nine timestamps and their derived durations,
nothing else.

Field names say what they measure without needing a comment, matching
`config.example.yaml`'s own habit. Per the plan's privacy prohibition, this
module records timings only -- no transcript text, no reply text, and no
audio bytes ever reach a log line, a browser event, or a file here. The
recorded-session store with its own retention schedule is Phase 2 (DBG-01,
DBG-06); this is the narrower, always-on measurement that proves the
1.5 second budget, not a substitute for it.

The nine stages, in the order a turn reaches them: the turn starting, the
speech-to-text socket opening, the first partial transcript arriving, the
last partial whose words actually changed arriving (`speech_end_at` --
the true end of speech, before xAI's own endpointing delay), the final
transcript arriving, the first brain chat round returning, the tool calls
completing, the first audio chunk leaving for the browser, and the first
chunk of the answer utterance specifically leaving for the browser. A turn
closed early by either VOICE-08 guard reaches only a prefix of these -- an
unset (`None`) stage is what "never reached" looks like, distinguishable
from a reached stage that happened to take zero time.

260924-4iv (e): `stt_final_at` is when xAI's own endpointing decided the
utterance was over, not when the operator actually stopped talking --
there is a real gap between the two (xAI's own endpointing delay), and
before this plan nothing measured it. `speech_end_at` is the arrival time
of the last partial whose normalized words differ from the one before it
-- the best available proxy for "the operator stopped talking" from the
partial stream alone. `endpointing_delay_ms` is `stt_final_at -
speech_end_at`, and the `speech_end_to_*` properties are the *true*
end-of-speech budget numbers; the older `end_of_speech_to_*` properties
stay exactly as they were (measuring from `stt_final_at`) for backward
compatibility with every existing reader.

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
    "speech_end_at",
    "stt_final_at",
    "brain_first_round_at",
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
    speech_end_at: float | None = None
    stt_final_at: float | None = None
    brain_first_round_at: float | None = None
    tool_rounds_done_at: float | None = None
    first_audio_at: float | None = None
    answer_audio_at: float | None = None
    # 10-05-PLAN.md (D-09 through D-13): the arrival time of the Pi's own
    # `vad.end`, the edge source's headline latency measurement -- deliberately
    # NOT in `_STAGE_ORDER`/`to_event()` (kept out of the browser timing
    # contract, since a browser/camera turn never has one) but included in
    # `log()` so it reaches the structured log line every other stage does.
    vad_end_at: float | None = None

    def mark_turn_started(self) -> None:
        """Record the moment the turn began -- the mic toggle, in Phase 1."""
        self.turn_started_at = time.monotonic()

    def mark_stt_socket_open(self) -> None:
        """Record the moment the turn started consuming the STT stream."""
        self.stt_socket_open_at = time.monotonic()

    def mark_first_partial(self, at: float | None = None) -> None:
        """Record the first partial transcript forwarded to the page.

        `at`, when given (260924-4iv), is the caller's own recorded arrival
        time -- `turn/controller.py::_drain_to_final_transcript` records
        each event's arrival with the same clock this module uses, and
        passes it through rather than letting this call re-read the clock
        a moment later. `at=None` (the default) reads `time.monotonic()`
        here, exactly as before this plan.

        Idempotent: only the first call after construction has any effect,
        matching every other mark's "this stage happened once" contract.
        """
        if self.first_partial_at is None:
            self.first_partial_at = at if at is not None else time.monotonic()

    def mark_speech_end(self, at: float | None) -> None:
        """Record the arrival time of the last partial whose words actually
        changed -- the best available proxy for "the operator stopped
        talking," before xAI's own endpointing delay (260924-4iv, item e).

        Assigns unconditionally, with no first-call guard: unlike every
        other mark in this class, a later call must win. `run_turn`'s
        wake-phrase handling can drain the STT stream twice in one turn
        (a wake-only first drain, then a second drain for the actual
        command); the second drain's own speech-end time is the one that
        describes this turn, and an idempotent guard would leave the first
        drain's wake-only timing behind instead. `at=None` clears the mark
        (a drain that reached no word-changing partial before its final
        event, or before its timeout) -- also unconditional, for the
        identical reason.
        """
        self.speech_end_at = at

    def mark_stt_final(self) -> None:
        """Record the moment the final transcript arrived."""
        self.stt_final_at = time.monotonic()

    def mark_vad_end(self, at: float) -> None:
        """Record the arrival time of the Pi's own `vad.end` (D-09 through
        D-13, 10-05-PLAN.md) -- first call wins, matching every other
        idempotent mark in this class. `at` is the caller's own recorded
        arrival (`turn/early_finalize.py::wait_for_end_of_speech`'s return
        value), not a fresh `time.monotonic()` read here, the same
        already-recorded-arrival discipline `mark_first_partial`'s `at`
        parameter already established.
        """
        if self.vad_end_at is None:
            self.vad_end_at = at

    def mark_brain_first_round(self) -> None:
        """Record the first `chat()` round returning.

        `BrainProvider.chat()`'s public contract returns one assembled
        `BrainReply` per round rather than per-token deltas (the streaming
        happens inside the provider; the controller never sees it) -- this
        is the earliest point a turn genuinely reaches "the model answered
        this round," not a streamed first token (260924-4iv renamed this
        mark from `brain_first_token_at`/`mark_brain_first_token` to say
        exactly that). Idempotent, so a multi-round tool loop times only
        the first round.
        """
        if self.brain_first_round_at is None:
            self.brain_first_round_at = time.monotonic()

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

        260924-4iv: `stt_final_at` is xAI's own endpointing decision, which
        arrives after the operator actually stopped talking (see
        `endpointing_delay_ms`) -- this property is kept, unchanged, for
        every existing reader; `speech_end_to_first_audio_ms` below is the
        true end-of-speech number.
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

        260924-4iv: measures from `stt_final_at`, after xAI's own
        endpointing delay -- kept, unchanged, for every existing reader;
        `speech_end_to_answer_audio_ms` below is the true end-of-speech
        number.
        """
        if self.stt_final_at is None or self.answer_audio_at is None:
            return None
        return (self.answer_audio_at - self.stt_final_at) * 1000

    @property
    def endpointing_delay_ms(self) -> float | None:
        """How long xAI's own endpointing took past the operator's last
        word, or `None` until both marks exist (260924-4iv, item e).

        `stt_final_at - speech_end_at`: the gap between the true end of
        speech and the STT provider's own decision that the utterance was
        over. Never rounded, matching every other derived property here.
        """
        if self.stt_final_at is None or self.speech_end_at is None:
            return None
        return (self.stt_final_at - self.speech_end_at) * 1000

    @property
    def speech_end_to_first_audio_ms(self) -> float | None:
        """The true end-of-speech-to-first-audio number, or `None` until
        both marks exist (260924-4iv, item e).

        Derived from `speech_end_at`/`first_audio_at` -- unlike
        `end_of_speech_to_first_audio_ms`, this includes no part of xAI's
        own endpointing delay, only the assistant's own processing time.
        """
        if self.speech_end_at is None or self.first_audio_at is None:
            return None
        return (self.first_audio_at - self.speech_end_at) * 1000

    @property
    def speech_end_to_answer_audio_ms(self) -> float | None:
        """The true end-of-speech-to-answer-audio number, or `None` until
        both marks exist (260924-4iv, item e). Mirrors
        `speech_end_to_first_audio_ms` exactly, for the answer utterance.
        """
        if self.speech_end_at is None or self.answer_audio_at is None:
            return None
        return (self.answer_audio_at - self.speech_end_at) * 1000

    @property
    def vad_end_to_stt_final_ms(self) -> float | None:
        """The phase's headline latency number (10-05-PLAN.md): how long
        from the Pi's own `vad.end` to the final transcript arriving, or
        `None` until both marks exist -- `None` for every non-edge turn,
        which never sets `vad_end_at` at all. This is the number the
        2.1-2.6 s end-of-speech-to-final-transcript baseline this plan
        exists to cut is measured against.
        """
        if self.vad_end_at is None or self.stt_final_at is None:
            return None
        return (self.stt_final_at - self.vad_end_at) * 1000

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
                "speech_end_at": self.speech_end_at,
                "stt_final_at": self.stt_final_at,
                "brain_first_round_at": self.brain_first_round_at,
                "tool_rounds_done_at": self.tool_rounds_done_at,
                "first_audio_at": self.first_audio_at,
                "answer_audio_at": self.answer_audio_at,
                "vad_end_at": self.vad_end_at,
                "end_of_speech_to_first_audio_ms": self.end_of_speech_to_first_audio_ms,
                "end_of_speech_to_answer_audio_ms": self.end_of_speech_to_answer_audio_ms,
                "endpointing_delay_ms": self.endpointing_delay_ms,
                "speech_end_to_first_audio_ms": self.speech_end_to_first_audio_ms,
                "speech_end_to_answer_audio_ms": self.speech_end_to_answer_audio_ms,
                "vad_end_to_stt_final_ms": self.vad_end_to_stt_final_ms,
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
            "endpointing_delay_ms": self.endpointing_delay_ms,
            "speech_end_to_first_audio_ms": self.speech_end_to_first_audio_ms,
            "speech_end_to_answer_audio_ms": self.speech_end_to_answer_audio_ms,
        }
