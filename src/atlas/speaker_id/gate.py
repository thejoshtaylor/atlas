"""The vad_end gate: a narrowing control on a turn already in flight, never
the safety boundary. Every action a turn reaches still passes the same
`mcp.atlas_mcp.safety.allow_call` check unconditionally (D-15) -- no
decision from this module may be read as authorization for anything, and a
speaker label is untrusted exactly like transcribed text.

This is a *different* gate object from `atlas.wake.gate.WakeGate`. The wake
gate decides at wake-hit time, before a turn exists, and blocks are logged
through `WakeEventRow`. This gate decides at `vad_end`, inside a turn that
has already started, and a block is logged through `TurnTimings`/the
session recorder instead -- a turn *did* start; it was cut short
(11-RESEARCH.md Pitfall 3). This module does not import or extend
`atlas.wake.gate.BlockReason`; it defines its own closed reason.

This module is pure: no I/O, no imports from `turn/`, `providers/`, `db/`,
or `transports/`. It decides; it never acts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpeakerMode = Literal["off", "record", "enforce"]
SpeakerBlockReason = Literal["unknown_speaker"]


@dataclass(frozen=True)
class SpeakerGateDecision:
    """Whether a turn may continue past `vad_end`, and why not when it may
    not. `effective_mode` names the mode actually applied -- `enforce`
    degraded to `record` (D-10) reports `"record"` here, not `"enforce"`,
    so a caller never has to re-derive the degrade. `identified` is
    reported even when it did not decide the outcome (e.g. `off` mode),
    since D-13 wants the match recorded regardless of whether it gated.
    """

    allowed: bool
    reason: "SpeakerBlockReason | None"
    effective_mode: SpeakerMode
    identified: bool

    @classmethod
    def allow(cls, *, effective_mode: SpeakerMode, identified: bool) -> "SpeakerGateDecision":
        return cls(allowed=True, reason=None, effective_mode=effective_mode, identified=identified)

    @classmethod
    def block(cls, *, reason: SpeakerBlockReason, effective_mode: SpeakerMode, identified: bool) -> "SpeakerGateDecision":
        return cls(allowed=False, reason=reason, effective_mode=effective_mode, identified=identified)


def evaluate_speaker_gate(
    *,
    mode: SpeakerMode,
    enrolled_count: int,
    best_score: "float | None",
    threshold: float,
) -> SpeakerGateDecision:
    """Decide one turn at `vad_end`. Checked in this order, first match
    wins:

    1. `mode == "off"` never blocks -- there is nothing to gate.
    2. `mode == "enforce"` with zero enrolled members acts as `"record"`
       (D-10): a fresh install with nobody enrolled must not silently
       block the whole house.
    3. `identified` is `best_score is not None and best_score >= threshold`
       -- a score exactly at the threshold counts (`>=`, not `>`).
    4. The effective mode `"enforce"` and not `identified` blocks with
       `"unknown_speaker"` (D-09).
    5. Otherwise: allow.
    """
    identified = best_score is not None and best_score >= threshold

    if mode == "off":
        return SpeakerGateDecision.allow(effective_mode="off", identified=identified)

    effective_mode: SpeakerMode = mode
    if mode == "enforce" and enrolled_count == 0:
        effective_mode = "record"

    if effective_mode == "record":
        return SpeakerGateDecision.allow(effective_mode=effective_mode, identified=identified)

    if not identified:
        return SpeakerGateDecision.block(reason="unknown_speaker", effective_mode=effective_mode, identified=identified)

    return SpeakerGateDecision.allow(effective_mode=effective_mode, identified=identified)
