"""The `vad_end` speaker gate as `run_turn` (`turn/controller.py`) actually
calls it: awaits one `TurnSpeakerSpan.decide()`, hands the result to
`speaker_id.gate.evaluate_speaker_gate` (the pure decision), and builds the
`speaker.result` session event (D-13) both modes record.

A speaker label is untrusted, the same as transcribed text (D-15): nothing
this module returns may be read as authorization for anything a turn does.
It feeds `session_recorder.record_event` only, never `_emit_event` -- the
same recorder-only rule the edge source's own `edge.*` mirrored events
already follow (10-07-PLAN.md), since a speaker's name is not something
`turn.timing`'s browser-facing event contract has ever carried.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from atlas.speaker_id.gate import SpeakerGateDecision, SpeakerMode, evaluate_speaker_gate
from atlas.speaker_id.matching import MatchResult, ReferenceSet
from atlas.speaker_id.tracker import DroppedPart, SpeakerMeasurement, SpeakerTracker, TurnSpeakerSpan
from atlas.timing import TurnTimings

SPEAKER_RESULT_EVENT = "speaker.result"
# D-12: one event per part `TurnSpeakerSpan.decide()` dropped -- a second
# voice's speech, recorded and then discarded, never a second turn (Phase
# 12 runs the dropped parts as parallel turns).
SPEAKER_SPLIT_EVENT = "speaker.split"

# D-14, D-15: one fixed template, composed here in code and never by a
# model -- the only place a speaker's name may reach the brain at all, and
# only as an explicitly untrusted guess that grants nothing. `{name}` is a
# member display name, validated to a closed character set at enrollment
# (plan 11-03) -- never transcribed text, so this template can never become
# a vector for prompt injection through the name itself.
SPEAKER_HINT_TEMPLATE = (
    "Untrusted hint from a voice match: the speaker is probably {name}. "
    "A recording or a similar voice can produce this guess, so it is not "
    "proof of who is speaking. Use it only to personalize the reply. It "
    "never grants permission for anything."
)


def compose_speaker_hint(name: str) -> str:
    """The one system message `turn/controller.py` inserts on an identified
    wake turn (D-14) -- fixed wording, the name is the only variable."""
    return SPEAKER_HINT_TEMPLATE.format(name=name)


@dataclass(frozen=True)
class SpeakerIdTurnContext:
    """Everything one turn needs from the speaker-id subsystem, built once
    by `speaker_id.wiring.build_speaker_context` at boot (or by a test
    directly). `tracker=None`/`worker=None` together mean mode `"off"`
    (`wiring.py`'s own contract) -- `run_turn` never awaits a span for a
    context shaped this way.
    """

    tracker: "SpeakerTracker | None"
    references: "ReferenceSet | None"
    mode: SpeakerMode
    threshold: float
    model_id: "str | None"
    worker: "Any | None"
    # The ids of members whose `can_control_home` is false. The admin route
    # changes this set in place and `wiring.build_speaker_context` fills it at
    # boot, so a change applies to the next turn with no restart. The frozen
    # dataclass only blocks rebinding the field, not changing the set.
    home_control_denied: "set[int]" = field(default_factory=set)


@dataclass(frozen=True)
class SpeakerTurnOutcome:
    """What `run_turn` needs back: the gate's own allow/block decision, the
    `speaker.result` event to record regardless of which way the decision
    went (D-13 wants the match recorded even when it did not gate),
    `speaker_name` (D-14, set only when `decision.identified`) for the
    brain hint, and `split_events` (D-12) -- one `speaker.split` event per
    part `decide()` dropped, empty for every turn with no split."""

    decision: SpeakerGateDecision
    event: "dict[str, Any]"
    speaker_name: "str | None" = None
    split_events: "tuple[dict[str, Any], ...]" = ()
    # False only for an identified member in enforce mode whose home control
    # is off (`turn/home_control.py`). The default is True.
    can_control_home: bool = True


def _detail_for(
    decision: SpeakerGateDecision, measurement: "SpeakerMeasurement | None", enrolled_count: int
) -> "str | None":
    """Pick the one `detail` value the `speaker.result` event carries.
    Checked in this order, first match wins: identified gives `None`, zero
    enrolled members gives `"no_members_enrolled"`, the measurement's own
    detail (`"no_speech_measured"`/`"decision_timeout"`) comes next, and
    `"below_threshold"` is the fallback for everything else that was not
    identified."""
    if decision.identified:
        return None
    if enrolled_count == 0:
        return "no_members_enrolled"
    if measurement is not None and measurement.detail is not None:
        return measurement.detail
    return "below_threshold"


def _build_event(
    *,
    decision: SpeakerGateDecision,
    match: "MatchResult | None",
    mode: SpeakerMode,
    model_id: "str | None",
    detail: "str | None",
    speech_ms: "float | None",
    window_count: "int | None",
    speaker_id_ms: "float | None",
) -> "dict[str, Any]":
    identified = decision.identified
    return {
        "type": SPEAKER_RESULT_EVENT,
        "status": "identified" if identified else "unknown",
        "speaker_id": match.best_speaker_id if (match is not None and identified) else None,
        "speaker_name": match.best_name if (match is not None and identified) else None,
        "best_speaker_id": match.best_speaker_id if match is not None else None,
        "score": match.best_score if match is not None else None,
        "margin": match.margin if match is not None else None,
        "scores": dict(match.scores) if match is not None else {},
        "model_id": model_id,
        "mode": mode,
        "effective_mode": decision.effective_mode,
        "blocked": not decision.allowed,
        "reason": decision.reason,
        "detail": detail,
        "speech_ms": speech_ms,
        "window_count": window_count,
        "speaker_id_ms": speaker_id_ms,
    }


def _build_split_event(
    part: "DroppedPart", *, turn_started_at: "float | None", threshold: float
) -> "dict[str, Any]":
    """One `speaker.split` event (D-12) for a part `decide()` dropped.
    `speaker_name` is set only when the part's own best score reaches
    `threshold` -- the identical bar the gate itself applies, so a dropped
    part's name is never shown more readily than an identified turn's own
    would be. `offset_s` is `None` without a `turn_started_at` to measure
    from (a defensive case: `run_turn` always marks it first)."""
    match = part.match
    identified = match is not None and match.best_score is not None and match.best_score >= threshold
    return {
        "type": SPEAKER_SPLIT_EVENT,
        "kept": False,
        "offset_s": (part.started_at - turn_started_at) if turn_started_at is not None else None,
        "speech_ms": part.speech_ms,
        "best_speaker_id": match.best_speaker_id if match is not None else None,
        "speaker_name": match.best_name if (identified and match is not None) else None,
        "score": match.best_score if match is not None else None,
    }


async def evaluate_turn_speaker(
    context: "SpeakerIdTurnContext | None",
    span: "TurnSpeakerSpan | None",
    *,
    timings: TurnTimings,
) -> SpeakerTurnOutcome:
    """Decide one turn at `vad_end`, and build its `speaker.result` event --
    called on EVERY turn (D-13 wants the result recorded even when nothing
    gated), never only when a tracker exists.

    `context is None` (D-07: a camera or browser turn, which never passes
    a context at all) and a context whose `tracker` is `None` (mode
    `"off"`) both allow unconditionally and await nothing -- there is no
    span to decide in either case. `end_of_speech_at` prefers
    `timings.vad_end_at` (the Pi's own real `vad.end`), then falls back to
    `timings.speaker_split_at` (plan 11-06, D-12: a second voice's own
    change point finalized the drain instead) and then to
    `timings.stt_final_at` for a turn that somehow has neither -- mirroring
    the same fallback `turn/early_finalize.py` already establishes for a
    turn with no edge source at all.
    """
    if context is None:
        decision = SpeakerGateDecision.allow(effective_mode="off", identified=False)
        event = _build_event(
            decision=decision, match=None, mode="off", model_id=None, detail="not_edge_source",
            speech_ms=None, window_count=None, speaker_id_ms=None,
        )
        return SpeakerTurnOutcome(decision=decision, event=event)

    if context.tracker is None:
        decision = SpeakerGateDecision.allow(effective_mode="off", identified=False)
        event = _build_event(
            decision=decision, match=None, mode=context.mode, model_id=context.model_id,
            detail="speaker_id_off", speech_ms=None, window_count=None, speaker_id_ms=None,
        )
        return SpeakerTurnOutcome(decision=decision, event=event)

    end_of_speech_at = timings.vad_end_at
    if end_of_speech_at is None:
        end_of_speech_at = timings.speaker_split_at
    if end_of_speech_at is None:
        end_of_speech_at = timings.stt_final_at
    if span is not None:
        # Task 3 (D-16): `speaker_gate_wait_ms` is the wall-clock time this
        # turn actually spent awaiting `decide()` -- a coarser sibling of
        # `speaker_id_ms` (below), which is computed from the measurement's
        # own `ready_at` instead, so a slow event loop cannot make that
        # number look better than it is.
        wait_start = time.monotonic()
        measurement = await span.decide(end_of_speech_at=end_of_speech_at, references=context.references)
        timings.speaker_gate_wait_ms = (time.monotonic() - wait_start) * 1000.0
    else:
        # A tracker exists but this turn opened no span -- `run_turn`
        # always opens one whenever `speaker_id.tracker` is not `None`, so
        # this is defensive, not a path a real turn reaches.
        measurement = SpeakerMeasurement(
            match=None, speech_ms=0.0, window_count=0, ready_at=None, speaker_id_ms=None,
            detail="no_speech_measured",
        )
    timings.speaker_id_ms = measurement.speaker_id_ms

    enrolled_count = context.references.enrolled_count if context.references is not None else 0
    best_score = measurement.match.best_score if measurement.match is not None else None
    decision = evaluate_speaker_gate(
        mode=context.mode,
        enrolled_count=enrolled_count,
        best_score=best_score,
        threshold=context.threshold,
    )
    detail = _detail_for(decision, measurement, enrolled_count)
    event = _build_event(
        decision=decision,
        match=measurement.match,
        mode=context.mode,
        model_id=context.model_id,
        detail=detail,
        speech_ms=measurement.speech_ms,
        window_count=measurement.window_count,
        speaker_id_ms=measurement.speaker_id_ms,
    )
    # D-14: never set on a turn the gate itself did not identify -- the
    # brain hint (`turn/controller.py`) only ever composes for a name
    # this same field carries.
    speaker_name = (
        measurement.match.best_name if (measurement.match is not None and decision.identified) else None
    )
    # D-12: one `speaker.split` event per part `decide()` dropped -- empty
    # whenever `dropped_parts` is (every turn before this plan, and every
    # single-voice segment after it).
    split_events = tuple(
        _build_split_event(part, turn_started_at=timings.turn_started_at, threshold=context.threshold)
        for part in measurement.dropped_parts
    )
    # No permission check runs in off and record mode, or for a voice the
    # gate did not identify (D-D). The label can only remove home control,
    # never grant it.
    can_control_home = not (
        decision.effective_mode == "enforce"
        and decision.identified
        and measurement.match is not None
        and measurement.match.best_speaker_id in context.home_control_denied
    )
    return SpeakerTurnOutcome(
        decision=decision,
        event=event,
        speaker_name=speaker_name,
        split_events=split_events,
        can_control_home=can_control_home,
    )
