"""The internal-to-wire outcome map (Phase 15, D-09).

`run_turn` records how a turn ended as an open set of internal strings
(`turn_outcome`). The Mac shows only fixed text for four wire values:
`completed`, `no_speech`, `stopped` and `failed`. The text never comes from
the model. This table maps every internal value to one of the four by an
explicit entry, with no catch-all pattern.

`tests/test_desktop_outcomes.py` walks `src/atlas` for `turn_outcome`
literals and fails when one has no entry here, so a new outcome gets a
decision. A value that still gets through (a literal built at run time) maps
to `completed`, which shows no outcome line.

`unknown_speaker`, `wake_unverified` and `follow_up_wrong_speaker` map to
`no_speech`. Those turns do reach a Mac. The bridge admits a turn when the
transcript opens with the wake phrase, and that check runs before the speaker
check. The Mac has then already shown the partials and the final transcript,
and it shows "no_speech" only when the turn ends. So the Mac shows the words
of a voice that ATLAS then refuses, and the outcome line does not say that
ATLAS heard it and refused it. The wake phrase check narrows this. It does
not stop it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_NO_SPEECH = (
    "empty_transcript",
    "timeout",
    "no_command",
    "follow_up_silence",
    "answer_window_silent",
    "wake_unverified",
    "unknown_speaker",
    "follow_up_wrong_speaker",
)
_STOPPED = ("stopped", "barged_in", "ring_stopped", "cancelled")
_FAILED = (
    "failed",
    "brain_timeout",
    "round_cap",
    "macro_failed",
    "local_intent_failed",
    "home_control_refused",
    "claim_refused",
    "bulk_refused",
    "empty_answer",
    "empty_reply",
    "confirm_failed",
    "confirm_expired",
    "confirm_unavailable",
    "confirmation_unavailable",
    "proposal_store_failed",
    "proposal_invalid",
    "follow_up_limit",
    "email_read_failed",
    "email_draft_failed",
)
_COMPLETED = (
    "completed",
    "unknown",
    "confirmed",
    "amended",
    "macro",
    "timer_intent",
    "local_intent",
    "email_list",
    "email_read",
    "email_draft",
    "needs_clarification",
    "needs_confirmation",
)

OUTCOME_TO_WIRE: dict[str, str] = {
    **{value: "no_speech" for value in _NO_SPEECH},
    **{value: "stopped" for value in _STOPPED},
    **{value: "failed" for value in _FAILED},
    **{value: "completed" for value in _COMPLETED},
}

_warned: set[str] = set()


def wire_outcome(value: str) -> str:
    """The wire outcome for an internal outcome string.

    An unknown value maps to `completed` and is logged once at warning. The
    log carries the value only, which is a code-set string and never text the
    operator or a model wrote.
    """
    mapped = OUTCOME_TO_WIRE.get(value)
    if mapped is not None:
        return mapped
    if value not in _warned and len(_warned) < 64:
        _warned.add(value)
        logger.warning("desktop outcome map: no entry for %r; sending completed", value)
    return "completed"
