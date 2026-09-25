"""The seam every Google action (and every Gmail read, plan 09-08) passes
through: a tool result carrying `atlas_handoff` ends the tool round and code
takes over, instead of the model (D-08).

`parse_handoff`/`_compose_clarifying_question` reach back into
`turn/controller.py` through a function-body import, deferred past both
modules' load -- the identical shape `turn/brain_race.py` already uses for
`_run_tool_rounds`, and for the identical reason: `controller.py` imports
this module at load time (to dispatch a stored handoff after the tool
round), so a module-level import here of anything from `controller.py`
would deadlock the two modules' load order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from atlas_mcp.google_tools import CALENDAR_PROPOSAL_TOOL_NAMES, CODE_ONLY_TOOL_NAMES, HANDOFF_KEY

from atlas.db.pending_action_repository import PendingActionRepository
from atlas.turn.follow_up import MAX_CHAINED_FOLLOW_UPS, FollowUpRequest
from atlas.turn.pending_action import (
    CONFIRMATION_UNAVAILABLE_REPLY,
    EXECUTING_TOOL_BY_ACTION,
    FOLLOW_UP_LIMIT_REPLY,
    PROPOSAL_INVALID_REPLY,
    PROPOSAL_STORE_FAILED_REPLY,
    PendingProposal,
    compose_readback,
    confirmation_data,
    execution_arguments,
)

logger = logging.getLogger("atlas.turn.handoff")

# Plan 09-08 adds "email_list" and "email_read" -- every Gmail read hands
# off to code the same way a calendar write proposal does (D-08, D-14).
# Plan 09-09 adds "email_draft" (D-08, D-21): a reply draft is written and
# saved by code, never by the model, exactly like every other handoff
# here. A workflow handoff (already-shipped `atlas_handoff` uses
# elsewhere) stays deliberately out of this frozenset: widening it
# further is a later plan's own decision, not implied by this one.
HANDOFF_KINDS: frozenset[str] = frozenset(
    {"pending_action", "needs_clarification", "email_list", "email_read", "email_draft"}
)

# T-09-27: the fixed refusal `_run_tool_rounds` (turn/controller.py) speaks
# in place of dispatching a code-only tool a model named anyway -- the
# second of the two independent controls (`PluginManager.rebuild`'s schema
# hiding is the first) that keep `atlas_mcp.google_tools.CODE_ONLY_TOOL_NAMES`
# unreachable from the model, regardless of what a tool's own description
# says or what the schema currently hides.
CODE_ONLY_REFUSAL = "that is not something i can do directly"


def is_code_only_tool(name: str) -> bool:
    """True when `name` -- the offered name a model round actually called,
    bare or collision-prefixed (`{slug}__{bare_name}`, `plugins/naming.py`'s
    own `NAME_SEPARATOR`) -- names a code-only tool. Checked against both
    the name as given and its part after the last `"__"`, so a prefix this
    plugin never asked for (a future naming collision) still refuses the
    bare tool it wraps, and a plugin slug that itself contains `"__"` still
    resolves to its own last segment, matching `plugins/naming.py`'s own
    prefixing rule exactly."""
    bare = name.rsplit("__", 1)[-1] if "__" in name else name
    return name in CODE_ONLY_TOOL_NAMES or bare in CODE_ONLY_TOOL_NAMES


# A-CR-02: the fixed refusal `_run_tool_rounds` (turn/controller.py) speaks
# in place of dispatching a tool that is not a calendar proposal, on the one
# turn that continues an `amended` confirmation reply -- the open-mic
# window's own no-wake-word turn (D-06, D-08, D-09). This is the structural
# backstop, not the tool schema that turn is offered: a model coerced into
# naming a tool outside `CALENDAR_PROPOSAL_TOOL_NAMES` still never reaches
# `tool_host.call_tool` for it.
AMENDED_CONTINUATION_REFUSAL = (
    "that's more than i can change from a reply -- say the wake word and ask again"
)


def is_calendar_proposal_tool(name: str) -> bool:
    """True when `name` -- bare or collision-prefixed, the same shape
    `is_code_only_tool` above already unwraps -- names one of
    `atlas_mcp.google_tools.CALENDAR_PROPOSAL_TOOL_NAMES`: a call that can
    only ever build a fresh `pending_action` handoff, never execute
    anything directly (A-CR-02)."""
    bare = name.rsplit("__", 1)[-1] if "__" in name else name
    return name in CALENDAR_PROPOSAL_TOOL_NAMES or bare in CALENDAR_PROPOSAL_TOOL_NAMES


@dataclass(frozen=True)
class Handoff:
    """One parsed handoff -- `kind` is one of `HANDOFF_KINDS`, `payload` is
    the handoff's own body (everything under the `atlas_handoff` key
    besides `kind` itself is still present in `payload`, including `kind`,
    since callers key off `kind` on this object, not by re-reading the
    dict)."""

    kind: str
    payload: dict[str, Any]


def parse_handoff(result: Any) -> "Handoff | None":
    """`result`'s parsed payload, when it is a `{HANDOFF_KEY: {"kind": ...}}`
    shape with a recognized kind -- `None` for every other result,
    including an error-shaped one (a `Denied` refusal is never a handoff)
    and an ordinary tool result with no handoff key at all.
    """
    from atlas.turn.controller import _is_error, _result_payload

    if _is_error(result):
        return None
    payload = _result_payload(result)
    if not isinstance(payload, dict) or set(payload) != {HANDOFF_KEY}:
        return None
    body = payload[HANDOFF_KEY]
    if not isinstance(body, dict):
        return None
    kind = body.get("kind")
    if kind not in HANDOFF_KINDS:
        return None
    return Handoff(kind=kind, payload=body)


@dataclass
class HandoffSlot:
    """One turn's own mutable landing spot for a handoff its top tier's
    tool round produced -- mirrors `brain_race.ToolCommitment`'s "one
    instance per turn, mutated in place, read back after the race" shape.

    `unreachable_accounts` is reserved for a later plan's own use of this
    slot (a Gmail read's own per-account degrade list, plan 09-08) and is
    never populated by this plan.
    """

    handoff: "Handoff | None" = None
    bulk_refused: bool = False
    unreachable_accounts: "list[Any]" = field(default_factory=list)


@dataclass(frozen=True)
class HandoffContext:
    """Everything `dispatch_handoff` needs beyond the handoff itself --
    built once per turn by `atlas.google.turn_context.build_handoff_context`
    (Task 3), never shared across turns (the same "never shared" discipline
    `turn/controller.py::_make_run_turn_for_source`'s own `TurnTimings`
    already follows).

    `pending_actions=None` (every app with no Google repository configured,
    `tests/test_startup_smoke.py`'s own fake dict included) makes
    `dispatch_handoff` behave exactly as `follow_up_available=False` would:
    `CONFIRMATION_UNAVAILABLE_REPLY`, nothing stored.
    """

    source_name: str
    tool_host: Any
    pending_actions: "PendingActionRepository | None"
    brain: Any
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pending_ttl_s: float = 60.0
    # Plan 09-08: this source's own last-spoken-email-list memory
    # (D-16) -- `None` for every app with no Google plugin configured,
    # the same tolerant-`getattr` default `build_handoff_context` already
    # gives `tool_host`/`pending_actions`/`brain`.
    email_memory: "Any | None" = None
    # How long a quarantine round (`turn/quarantine.py`) may take before
    # `handle_email_list`/`handle_email_read`/`handle_email_draft` fall
    # back to "i couldn't summarize that one" for that one message.
    quarantine_timeout_s: float = 20.0
    # Plan 09-09: the style repository `handle_email_draft` reads a
    # drafting round's own profile/samples/signature from
    # (`atlas.db.google_repository.GoogleAccountRepository`) -- `None`
    # with the same "no Google repository configured" tolerance every
    # other Google-only field on this dataclass already has.
    style_repo: "Any | None" = None


@dataclass(frozen=True)
class HandoffOutcome:
    """What `run_turn` speaks, records as `turn_outcome`, and (for a stored
    pending action) requests on the source's own follow-up channel."""

    reply_text: str
    turn_outcome: str
    follow_up: "FollowUpRequest | None" = None


async def dispatch_handoff(
    handoff: Handoff,
    ctx: "HandoffContext | None",
    *,
    transcript: str,
    follow_up_available: bool,
    chain_depth: int,
) -> HandoffOutcome:
    """Resolve one turn's stored handoff into what to say and, for a
    pending action, what to store and what follow-up to ask for.

    `chain_depth` beyond `MAX_CHAINED_FOLLOW_UPS` ends the exchange rather
    than looping (D-06's own "never left waiting forever" extended to a
    chain of clarifications). A `needs_clarification` handoff never stores
    anything -- there is no pending action here to store -- but, on a
    source with a follow-up channel attached (`follow_up_available`), it
    now requests the same kind of open-microphone follow-up a stored
    confirmation does (plan 09-07, D-06 replacing phase 4's own D-08 for
    this question): the operator answers inside a brief window rather
    than waking the assistant again. A source with no channel is
    unchanged, exactly like the existing entity/plugin/run disambiguation
    path (`turn/controller.py::_compose_clarifying_question`).
    """
    if chain_depth > MAX_CHAINED_FOLLOW_UPS:
        return HandoffOutcome(reply_text=FOLLOW_UP_LIMIT_REPLY, turn_outcome="follow_up_limit")

    if handoff.kind == "needs_clarification":
        from atlas.turn.controller import _compose_clarifying_question

        candidates = tuple(handoff.payload.get("candidates", ()))
        question = _compose_clarifying_question(candidates, {})
        # Plan 09-07 (D-06): `playback_ends_at`/`prior_messages` are left
        # at their defaults here and filled in by `run_turn`'s own call
        # site -- that is the one place that knows this turn's own
        # `SpeechResult` and whether it was itself answering an earlier
        # follow-up (`replace(outcome.follow_up, ...)`, mirroring the
        # `pending_action` branch below).
        follow_up = (
            FollowUpRequest(
                kind="clarification",
                chain_depth=chain_depth,
                original_transcript=transcript,
                question=question,
            )
            if follow_up_available
            else None
        )
        return HandoffOutcome(reply_text=question, turn_outcome="needs_clarification", follow_up=follow_up)

    if handoff.kind == "email_list":
        from atlas.turn.email_handoff import handle_email_list

        return await handle_email_list(handoff, ctx)

    if handoff.kind == "email_read":
        from atlas.turn.email_handoff import handle_email_read

        return await handle_email_read(handoff, ctx)

    if handoff.kind == "email_draft":
        from atlas.turn.email_handoff import handle_email_draft

        return await handle_email_draft(handoff, ctx)

    # kind == "pending_action"
    if not follow_up_available or ctx is None or ctx.pending_actions is None:
        return HandoffOutcome(
            reply_text=CONFIRMATION_UNAVAILABLE_REPLY, turn_outcome="confirmation_unavailable"
        )

    body = {key: value for key, value in handoff.payload.items() if key != "kind"}
    try:
        proposal = PendingProposal(**body)
    except Exception:
        return HandoffOutcome(reply_text=PROPOSAL_INVALID_REPLY, turn_outcome="proposal_invalid")

    tool_name = EXECUTING_TOOL_BY_ACTION[proposal.action]
    arguments = execution_arguments(proposal)
    readback = compose_readback(proposal, now=ctx.now)
    expires_at = ctx.now + timedelta(seconds=ctx.pending_ttl_s)

    # A-WR-01: unguarded, a raise here would abort the turn silently
    # before the readback is ever spoken -- the operator would hear
    # nothing at all rather than a refusal, exactly the "cannot tell a
    # refusal from a crash" failure this project's own doctrine names
    # elsewhere. `proposal` itself is never stored or acted on when this
    # raises; there is nothing to resolve to a terminal status (no row
    # exists yet).
    try:
        action_row = await ctx.pending_actions.create(
            source=ctx.source_name,
            action=proposal.action,
            tool_name=tool_name,
            arguments=arguments,
            readback=readback,
            created_at=ctx.now,
            expires_at=expires_at,
        )
    except Exception:
        logger.exception("pending_actions.create raised while storing a %r proposal", proposal.action)
        return HandoffOutcome(reply_text=PROPOSAL_STORE_FAILED_REPLY, turn_outcome="proposal_store_failed")

    follow_up = FollowUpRequest(
        kind="confirmation",
        chain_depth=chain_depth,
        original_transcript=transcript,
        question=readback,
        pending_action_id=action_row.id,
        proposal=confirmation_data(proposal, now=ctx.now),
    )
    return HandoffOutcome(reply_text=readback, turn_outcome="needs_confirmation", follow_up=follow_up)
