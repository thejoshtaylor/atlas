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
from dataclasses import dataclass
from typing import Any

from atlas.speaker_id.gate import SpeakerGateDecision, SpeakerMode, evaluate_speaker_gate
from atlas.speaker_id.matching import MatchResult, ReferenceSet
from atlas.speaker_id.tracker import SpeakerMeasurement, SpeakerTracker, TurnSpeakerSpan
from atlas.timing import TurnTimings

SPEAKER_RESULT_EVENT = "speaker.result"


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


@dataclass(frozen=True)
class SpeakerTurnOutcome:
    """What `run_turn` needs back: the gate's own allow/block decision, and
    the `speaker.result` event to record regardless of which way the
    decision went (D-13 wants the match recorded even when it did not
    gate)."""

    decision: SpeakerGateDecision
    event: "dict[str, Any]"


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
    `timings.vad_end_at` (the Pi's own real `vad.end`) and falls back to
    `timings.stt_final_at` for a turn that somehow has no `vad_end_at`
    recorded -- mirroring the same fallback `turn/early_finalize.py`
    already establishes for a turn with no edge source at all.
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

    end_of_speech_at = timings.vad_end_at if timings.vad_end_at is not None else timings.stt_final_at
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
    return SpeakerTurnOutcome(decision=decision, event=event)
