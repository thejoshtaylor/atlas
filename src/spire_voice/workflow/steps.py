"""`StepOutcome` and `execute_step`: the kind dispatch the poller
(`scheduler.py`) calls once it has claimed a due `WorkflowStepRow` (D-05).

This task wires `call_service` only -- the single-step tracer's own kind.
`wait` and `speak` are wired in plan 05-03; calling `execute_step` against
either here raises a named error rather than silently completing, so a gap
in coverage is loud, not a quietly-wrong terminal state.

There is no suspension for a step's own configured duration anywhere in
this module. The only step kind with a duration (`wait`) expresses it
declaratively, folded into `due_at` at creation time
(`db.repository.assign_step_due_ats`, PA-D1) -- a `wait` executor that
slept for its own duration would hold the poller's claimed-row transaction
open for that whole interval, compounding the same row-lock-held-across-
execution cost the `call_service` kind already accepts for a bounded HTTP
call (05-RESEARCH.md Pitfall 3 and Pitfall 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from spire_voice.config import WorkflowConfig
from spire_voice.db.models import WorkflowStepRow
from spire_voice.turn.controller import _is_error, _result_text


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
    """Raised by `execute_step` for a step kind D-05 names (`wait`,
    `speak`) but this plan has not wired an executor for yet -- a named
    gap, never a silent no-op that would look like the step completed."""


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
    (PA-D3, T-05-01)."""
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


async def execute_step(
    step: WorkflowStepRow,
    tool_host: _ToolHost,
    config: WorkflowConfig,
    now: datetime,
) -> StepOutcome:
    """Dispatches on `step.kind`, a closed set of exactly three (D-05).
    `config`/`now` are carried through for the two kinds plan 05-03 wires
    (a `speak` step's retry backoff, a lateness sentence measured against
    `now`); this task's own `call_service` executor needs neither."""
    del config, now  # unused by call_service; carried for 05-03's kinds
    if step.kind == "call_service":
        return await _execute_call_service(step, tool_host)
    if step.kind in ("wait", "speak"):
        raise UnwiredStepKindError(
            f"workflow step kind {step.kind!r} has no executor yet -- wired in plan 05-03"
        )
    raise ValueError(f"unknown workflow step kind: {step.kind!r}")
