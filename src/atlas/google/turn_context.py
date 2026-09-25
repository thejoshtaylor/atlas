"""`build_handoff_context`: the one call each of `app.py`'s three `run_turn`
call sites makes, once per turn, to build that turn's own
`atlas.turn.handoff.HandoffContext` (plan 09-04, D-08).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atlas.turn.email_memory import EmailListMemory
from atlas.turn.handoff import HandoffContext

if TYPE_CHECKING:
    from fastapi import FastAPI


def build_handoff_context(app: "FastAPI", source_name: str) -> HandoffContext:
    """Read `app.state`'s current pending-action repository, tool host
    lookup, and brain -- called once per turn, at the call site, so every
    turn sees whatever `app.state.tool_host_lookup` currently points at
    (the same immutable-swap discipline every other reader of that
    attribute already relies on, D-08).

    Every read here is a tolerant `getattr(..., None)`: an app with no
    Google repository configured -- every deployment before this phase,
    and `tests/test_startup_smoke.py`'s own fake repository dict -- still
    boots and still runs turns exactly as before. `dispatch_handoff` reads
    `pending_actions is None` as "nothing to store a proposal through" and
    speaks `CONFIRMATION_UNAVAILABLE_REPLY`, the identical fallback it
    already gives a source with no follow-up channel attached.

    Plan 09-08 (D-16): unlike the three reads above, `app.state.email_list_memory`
    is created here, once per application, the first time any turn reaches
    this function -- `EmailListMemory` needs no external configuration
    (no repository, no credentials) to exist, so there is nothing to be
    tolerant of. Every later call reuses the same instance, matching
    `EmailListMemory`'s own "per application" contract (D-16, SRC-03).
    """
    email_memory = getattr(app.state, "email_list_memory", None)
    if email_memory is None:
        email_memory = EmailListMemory()
        app.state.email_list_memory = email_memory
    return HandoffContext(
        source_name=source_name,
        tool_host=getattr(app.state, "tool_host_lookup", None),
        pending_actions=getattr(app.state, "pending_action_repo", None),
        brain=getattr(app.state, "brain", None),
        email_memory=email_memory,
    )
