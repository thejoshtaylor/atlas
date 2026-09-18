"""`StepOutcome` and `execute_step`: the kind dispatch the poller
(`scheduler.py`) calls once it has claimed a due `WorkflowStepRow` (D-05).

All three kinds are wired: `call_service` (plan 05-01), `wait` and `speak`
(plan 05-03). An unknown kind still raises rather than silently completing,
so a gap in coverage stays loud, not a quietly-wrong terminal state.

There is no suspension for a step's own configured duration anywhere in
this module -- confirmed by this file's own negative-grep acceptance
criterion, checked at every commit. The only step kind with a duration (`wait`)
expresses it declaratively, folded into `due_at` at creation time
(`db.repository.assign_step_due_ats`, PA-D1): its own executor below
completes instantly and calls nothing. Do not "fix" that into a sleep --
the claimed row's lock is held for the whole executor call (D-02), and
`wait` is the one kind with no external I/O to bound a sleep against
(05-RESEARCH.md Pitfall 3 and Pitfall 5).

`execute_step` also owns two things that are not any single kind's own
job: recording and, for the run's first step only, speaking a step's own
lateness (D-04), and speaking a kind's own refusal reason verbatim when it
has one (D-14) -- both through the one `speak` callable this function is
constructed with, never a workflow-local reply composer of its own
(05-RESEARCH.md "Don't Hand-Roll": `turn/controller.py`'s own composers
and verbatim-refusal path are reused, not re-implemented).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol

from spire_voice.config import WorkflowConfig
from spire_voice.db.models import WorkflowStepRow
from spire_voice.turn.controller import _is_error, _result_text

_Speak = Callable[[str], Awaitable[None]]


class _ToolHost(Protocol):
    """The same structural shape `turn/controller.py`'s own `_ToolHost`
    Protocol expects -- `McpToolHostLookup` and every `McpToolHost`
    satisfy it already, so this executor calls the identical tool host the
    live turn path calls, with no workflow-local copy of `allow_call`
    (D-13)."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class StepOutcome:
    """What `execute_step` decided a claimed step's terminal (or retried)
    state should be.

    `status` is one of `completed`, `denied`, or `failed` -- the same
    closed set `workflow_steps.status` carries (0006's own migration).
    `detail` is what `claim_and_execute_next_due_step` writes to
    `result_detail` verbatim; `speech` is what a caller that speaks a
    fire-time outcome (plan 05-03's lateness/refusal composer) reads,
    `None` when there is nothing to say about this particular outcome.
    `retry` is `True` only when this kind's own policy allows another
    attempt (PA-D3): `call_service` never retries, because a service call
    whose outcome is unknown must not be repeated -- a repeated call is
    exactly the double-execution T-05-01 names.
    """

    status: str
    detail: dict
    speech: str | None
    retry: bool


class UnwiredStepKindError(Exception):
    """Raised by `execute_step` when a step kind D-05 names has no way to
    run -- today, only a `speak` step claimed while `execute_step` was
    constructed with no `speak` callable at all (a wiring bug in the
    caller, `app.py`'s own `lifespan`), never a silent no-op that would
    look like the step completed."""


def _tool_result_payload(result: Any) -> Any:
    """The MCP result's own structured payload, when the framework
    provided one, or the plain result text otherwise -- the same
    `structuredContent`/`structured_content` presence-checked fallback
    `app.py::_tool_result_json` already applies, duplicated here rather
    than imported to avoid a `workflow.steps` -> `app` import (`app.py`
    itself imports from this package's siblings)."""
    structured = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
    if structured is not None:
        return structured
    return {"text": _result_text(result)}


def _transition_refusal(arguments: dict[str, Any]) -> str | None:
    """The same domain restriction `mcp/spire_mcp/ha.py::handle_call_service`
    enforces at the process boundary, applied a second time here, before a
    step authored in the webapp (or by voice) ever reaches that boundary
    (FLOW-03). Deliberately two layers: the operator finds out immediately
    a step will never run, and the boundary still refuses on its own if a
    `transition` value reaches it another way. Returns the refusal text,
    written to be spoken, or `None` when there is nothing to refuse."""
    transition = arguments.get("transition")
    if transition is None:
        return None
    domain = arguments.get("domain")
    if domain != "light":
        return f"transition is only supported for lights, not {domain}"
    if isinstance(transition, bool) or not isinstance(transition, (int, float)):
        return f"transition must be a non-negative number of seconds, got {transition!r}"
    if not math.isfinite(transition) or transition < 0:
        return f"transition must be a non-negative number of seconds, got {transition!r}"
    return None


async def _execute_call_service(step: WorkflowStepRow, tool_host: _ToolHost) -> StepOutcome:
    """Calls `ha_call_service` through `tool_host` -- the same tool the
    live turn path calls, via the same `McpToolHostLookup` (D-13). An
    error-shaped result (a `Denied` refusal, raised as `ToolError` by
    `mcp/spire_mcp/ha.py::ha_call_service` and caught by the MCP
    framework into `CallToolResult.is_error`) is the refusal channel --
    its text is the boundary's own words, carried into `detail`/`speech`
    verbatim, never reworded (D-14). A raised exception -- the call itself
    never reached a verdict -- is `failed` with `retry=False` for this
    kind: a service call whose outcome is unknown must not be repeated
    (PA-D3, T-05-01).

    `_transition_refusal` runs first, before `tool_host.call_tool` is ever
    awaited: a step that will only ever be refused at the far end of a
    process boundary is refused here instead, with no request made at all
    (FLOW-03's second layer)."""
    refusal = _transition_refusal(step.arguments)
    if refusal is not None:
        return StepOutcome(
            status="denied", detail={"reason": refusal}, speech=refusal, retry=False
        )
    try:
        result = await tool_host.call_tool("ha_call_service", dict(step.arguments))
    except Exception as exc:
        return StepOutcome(
            status="failed", detail={"error": str(exc)}, speech=None, retry=False
        )
    if _is_error(result):
        text = _result_text(result)
        return StepOutcome(
            status="denied", detail={"reason": text}, speech=text or None, retry=False
        )
    return StepOutcome(
        status="completed",
        detail={"result": _tool_result_payload(result)},
        speech=None,
        retry=False,
    )


async def _execute_wait() -> StepOutcome:
    """A `wait` step has no work of its own: its duration was already
    folded forward into the `due_at` of every step written after it, at
    creation time (`db.repository.assign_step_due_ats`, PA-D1). Completes
    instantly and calls nothing -- never suspends for its own configured
    duration here; see this module's own docstring for why that would be
    wrong, not merely different."""
    return StepOutcome(status="completed", detail={}, speech=None, retry=False)


async def _execute_speak(step: WorkflowStepRow, speak: "_Speak | None") -> StepOutcome:
    """Calls the injected `speak(text)` awaitable `execute_step` was
    constructed with -- this module never builds a text-to-speech
    provider, an audio sink, or a second sending loop of its own; `app.py`
    owns all three (D-14's mechanism). `speak is None` is a wiring bug in
    the caller, not a step-authoring problem, so it raises rather than
    silently completing.

    A synthesis failure is retryable for this kind and this kind only
    (PA-D3): re-speaking into a room is harmless, and an unheard sentence
    is worth another attempt up to `WorkflowConfig.max_attempts` --
    unlike `call_service`, whose outcome once sent is never safe to repeat."""
    if speak is None:
        raise UnwiredStepKindError(
            "a speak step was claimed but execute_step was given no speak callable"
        )
    text = step.arguments.get("text", "")
    try:
        await speak(text)
    except Exception as exc:
        return StepOutcome(status="failed", detail={"error": str(exc)}, speech=None, retry=True)
    return StepOutcome(status="completed", detail={"text": text}, speech=None, retry=False)


def compose_lateness_sentence(late_by_s: float) -> str:
    """The spoken lateness line for a step that fired more than
    `WorkflowConfig.late_threshold_s` past its own `due_at` (D-04's "and
    says so"). Composed here, in code, never through a model round trip --
    the same reasoning `turn/controller.py`'s own
    `_compose_clarifying_question`/`_compose_mixed_outcome_reply` already
    carry: the same `late_by_s` always produces the same sentence, and
    nothing here reads a clock or a random source.

    Rounded to whole minutes under an hour, whole hours from an hour on --
    an operator does not need second-level precision spoken aloud for
    either "fifty-one seconds late" or "three hours late", and a fixed
    rounding rule is what keeps this function's own output deterministic
    for a fixed input, asserted exactly by this file's own tests."""
    late_by_s = max(0.0, late_by_s)
    if late_by_s < 3600:
        minutes = max(1, round(late_by_s / 60))
        unit = "minute" if minutes == 1 else "minutes"
        return f"sorry, this was about {minutes} {unit} late."
    hours = max(1, round(late_by_s / 3600))
    unit = "hour" if hours == 1 else "hours"
    return f"sorry, this was about {hours} {unit} late."


def _late_by_seconds(due_at: datetime, now: datetime) -> float:
    """`due_at` (`WorkflowStepRow.due_at`) is naive UTC -- this project's
    own database-boundary convention (`db.models.WorkflowStepRow`'s own
    docstring: "lateness is `fired_at - due_at`, a fact rather than an
    inference"). `now` arrives aware UTC from every caller this project
    has (`WorkflowScheduler`'s own default `clock`) but is normalized
    here regardless rather than assumed, and rather than importing
    `db.postgres`'s own private `_to_naive_utc` across a module boundary
    this project keeps split on purpose (05-01 SUMMARY: "the module that
    owns the database boundary never imports timedelta")."""
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    return (now - due_at).total_seconds()


async def execute_step(
    step: WorkflowStepRow,
    tool_host: _ToolHost,
    config: WorkflowConfig,
    now: datetime,
    *,
    speak: "_Speak | None" = None,
) -> StepOutcome:
    """Dispatches on `step.kind`, a closed set of exactly three (D-05),
    then applies the two things that are not any single kind's own job:

    Lateness (D-04): every step whose fire time (`now`) is more than
    `config.late_threshold_s` past its own `due_at` gets `late_by_s`
    recorded in `result_detail` -- but only the run's first step
    (`position == 0`) is spoken aloud, through `speak`. A run says once
    that it is running late; every step after the first is late for the
    same reason, and repeating it per step would be noise in a room
    nobody is standing in.

    A refusal (D-14): whenever the kind's own outcome carries `speech`
    (a fire-time policy denial from `_execute_call_service`, or this
    module's own `transition`-domain refusal), it is spoken verbatim,
    through the same `speak` callable -- never reworded, never dropped.

    `speak=None` (a caller with nothing to say through, or a test that
    only cares about the kind's own outcome) skips both without raising,
    except that a claimed `speak`-kind step with no `speak` callable at
    all is a wiring bug, not a quiet no-op -- see `_execute_speak`.
    """
    if step.kind == "call_service":
        outcome = await _execute_call_service(step, tool_host)
    elif step.kind == "wait":
        outcome = await _execute_wait()
    elif step.kind == "speak":
        outcome = await _execute_speak(step, speak)
    else:
        raise ValueError(f"unknown workflow step kind: {step.kind!r}")

    late_by_s = _late_by_seconds(step.due_at, now)
    if late_by_s > config.late_threshold_s:
        detail = dict(outcome.detail)
        detail["late_by_s"] = late_by_s
        outcome = StepOutcome(
            status=outcome.status, detail=detail, speech=outcome.speech, retry=outcome.retry
        )
        if step.position == 0 and speak is not None:
            await speak(compose_lateness_sentence(late_by_s))

    if outcome.speech is not None and speak is not None:
        await speak(outcome.speech)

    return outcome
