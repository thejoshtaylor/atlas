"""`ScheduleWorkflowRequest`, `CancelWorkflowRunRequest`,
`AppendWorkflowStepsRequest`, and `WorkflowToolHost`: the three
model-facing workflow tools (D-08, plan 05-05 widening plan 05-01's
single-step `schedule_workflow`, and adding `cancel_workflow_run`/
`append_workflow_steps`, D-09/D-11), joining the same `McpToolHostLookup`
the Home Assistant and weather children already sit in
(`mcp_client.py::McpToolHostLookup`, `app.py`'s `lifespan`).

This host is in-process, not a spawned MCP child, precisely because it
holds a repository rather than a credential -- the credential-isolated
children exist to keep `HA_TOKEN` and provider keys out of the parent
process (SAFE-09); a database connection carries no such secret, so there
is no isolation boundary to cross here.

Acceptance criterion this module holds itself to: it contains no time-
zone-name lookup, no wall-clock-string parser, and no absolute-instant
arithmetic of its own -- `workflow.schedule.resolve_schedule` is the only
place in this project that turns a caller's way of saying "when" into an
absolute instant. Every value this module hands `resolve_schedule` (the
request's own `delay_seconds`/`at`, an injected clock reading, the
server's own resolved local time zone object) is a plain pass-through.

The operator never speaks a run's id (D-09): `cancel_workflow_run` and
`append_workflow_steps` both take one, but the only place a run id ever
reaches the model is the pending-run context block plan 05-05's Task 2
injects every turn -- this module writes no sentence that asks the
operator to say a number.
"""

from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from typing import Any, Callable, Literal

from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from spire_voice.db.repository import (
    WorkflowRepository,
    WorkflowRunNotAppendableError,
    WorkflowRunNotFoundError,
    WorkflowStepSpec,
)
from spire_voice.workflow.schedule import ScheduleError, resolve_schedule

SCHEDULE_WORKFLOW_TOOL_NAME = "schedule_workflow"
CANCEL_WORKFLOW_RUN_TOOL_NAME = "cancel_workflow_run"
APPEND_WORKFLOW_STEPS_TOOL_NAME = "append_workflow_steps"


class WorkflowStepEntry(BaseModel):
    """One step of an ordered plan, in written order (D-05). `kind` is a
    closed `Literal["wait", "call_service", "speak"]` -- the same
    "impossible by construction" discipline `FillerPhrase`
    (`providers/tier_reply.py`) already applies to its own closed set: a
    fourth kind is unrepresentable here, not merely discouraged.

    `model_config`'s `extra="forbid"` closes the entry itself to any field
    beyond `kind`/`arguments` -- a condition, a branch, or a repeat count
    riding alongside a step is rejected before it can reach storage, the
    direct path to the scripting language this phase's own boundary warns
    against.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["wait", "call_service", "speak"]
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The per-kind payload. wait: duration_s (a positive number of seconds). "
            "call_service: domain, service, and optionally entity_id/area_id/device_id/"
            "label_id, plus transition (lights only). speak: text (non-empty)."
        ),
    )

    @model_validator(mode="after")
    def _validate_per_kind_arguments(self) -> "WorkflowStepEntry":
        if self.kind == "wait":
            duration = self.arguments.get("duration_s")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
                raise ValueError(
                    "a wait step's arguments must carry a positive duration_s -- got "
                    f"{duration!r}"
                )
        elif self.kind == "call_service":
            domain = self.arguments.get("domain")
            service = self.arguments.get("service")
            if not isinstance(domain, str) or not domain:
                raise ValueError(
                    "a call_service step's arguments must carry a non-empty domain"
                )
            if not isinstance(service, str) or not service:
                raise ValueError(
                    "a call_service step's arguments must carry a non-empty service"
                )
            transition = self.arguments.get("transition")
            if transition is not None and domain != "light":
                raise ValueError(
                    f"transition is only supported for lights, not domain={domain!r}"
                )
        elif self.kind == "speak":
            text = self.arguments.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("a speak step's arguments must carry non-empty text")
        return self


class ScheduleWorkflowRequest(BaseModel):
    """The model-facing shape of one `schedule_workflow` call (PA-D4).

    An ordered list of one or more steps and exactly one of two time
    fields -- `delay_seconds` (an elapsed interval, no zone read or
    needed) or `at` (an ISO-8601 instant, resolved by
    `workflow.schedule.resolve_schedule` and nowhere else). Both given, or
    neither, is a caller error, rejected here before a row is ever
    written -- the same "impossible by construction" treatment `TierReply`
    already gives its own exclusive outcomes.
    """

    model_config = ConfigDict(extra="forbid")

    steps: list[WorkflowStepEntry] = Field(
        min_length=1,
        description="The ordered steps this run performs, in the order they were spoken.",
    )
    delay_seconds: int | None = Field(
        default=None, ge=0, description="How many seconds from now the first step is due."
    )
    at: str | None = Field(
        default=None,
        description=(
            "An ISO-8601 datetime naming when the first step is due. Give a UTC offset "
            "(or 'Z') when you know one; a value with no offset is resolved against the "
            "server's own configured time zone."
        ),
    )
    summary: str = Field(
        min_length=1,
        description="What the operator would say to mean this run -- used later to cancel or extend it by voice.",
    )

    @model_validator(mode="before")
    @classmethod
    def _wrap_legacy_single_step_call(cls, data: Any) -> Any:
        """Plan 05-01's own frozen wire format -- a flat `kind`/`arguments`
        pair, one step per call (`tests/test_workflow_tracer.py`, a file
        this plan does not modify) -- is recognized here and normalized
        into the widened `steps` list before any other validator runs.
        The single-step path is a special case of the ordered list, not a
        second code path kept alive beside it."""
        if isinstance(data, dict) and "steps" not in data and "kind" in data:
            data = dict(data)
            data["steps"] = [{"kind": data.pop("kind"), "arguments": data.pop("arguments", {})}]
        return data

    @model_validator(mode="after")
    def _exactly_one_time_field(self) -> "ScheduleWorkflowRequest":
        if (self.delay_seconds is None) == (self.at is None):
            raise ValueError(
                "schedule_workflow needs exactly one of delay_seconds or at -- got "
                f"delay_seconds={self.delay_seconds!r}, at={self.at!r}"
            )
        return self


class CancelWorkflowRunRequest(BaseModel):
    """The model-facing shape of one `cancel_workflow_run` call. `run_id`
    always comes from the pending-run context block (D-09) -- this class
    carries no field for a spoken description, because the operator never
    supplies the id directly."""

    model_config = ConfigDict(extra="forbid")

    run_id: int = Field(
        description="The pending or firing run's id, from the pending-run list you were given -- never spoken by the operator."
    )


class AppendWorkflowStepsRequest(BaseModel):
    """The model-facing shape of one `append_workflow_steps` call. Same
    per-kind step validation `ScheduleWorkflowRequest.steps` uses -- an
    appended step is held to the identical rules a newly scheduled one
    is."""

    model_config = ConfigDict(extra="forbid")

    run_id: int = Field(
        description="The still-pending run's id, from the pending-run list you were given -- never spoken by the operator."
    )
    steps: list[WorkflowStepEntry] = Field(
        min_length=1, description="The steps to add to the end of this run's ordered step list, in order."
    )


class WorkflowToolHost:
    """A small in-process tool host exposing `schedule_workflow`,
    `cancel_workflow_run`, and `append_workflow_steps`, with a `.tools`
    list of `mcp.types.Tool` entries and an `async def call_tool(name,
    arguments)` -- the same structural shape `McpToolHostLookup` already
    builds its name-to-host map from (`mcp_client.py`'s own docstring: "a
    host that exposes those is routable without changing either
    function").

    A validation failure (a malformed call, a refused schedule, an
    unknown or non-appendable run id) returns an error-shaped
    `CallToolResult` the turn path already knows how to report
    (`turn/controller.py::_is_error`/`_result_text`) -- never a silently
    dropped schedule, cancellation, or append.
    """

    def __init__(
        self,
        repository: WorkflowRepository,
        *,
        zone: "tzinfo | None",
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._repository = repository
        self._zone = zone
        self._clock = clock
        self.tools: list[Tool] = [
            Tool(
                name=SCHEDULE_WORKFLOW_TOOL_NAME,
                description=(
                    "Schedule an ordered list of steps (wait, call_service, or speak) to "
                    "run later, after a delay given in seconds or at an absolute time. "
                    "Nothing happens now -- the steps run in the order given, once, later. "
                    "Give exactly one of delay_seconds or at, never both, never neither."
                ),
                input_schema=ScheduleWorkflowRequest.model_json_schema(),
            ),
            Tool(
                name=CANCEL_WORKFLOW_RUN_TOOL_NAME,
                description=(
                    "Cancel a pending or already-firing scheduled run by its id. The id "
                    "always comes from the list of pending runs already given to you in "
                    "context -- never ask the operator to say a number."
                ),
                input_schema=CancelWorkflowRunRequest.model_json_schema(),
            ),
            Tool(
                name=APPEND_WORKFLOW_STEPS_TOOL_NAME,
                description=(
                    "Add one or more steps to the end of a still-pending run's ordered "
                    "step list. Refused once that run has started firing -- a plan already "
                    "under way cannot be edited mid-execution. The id always comes from "
                    "the pending-run list already given to you in context."
                ),
                input_schema=AppendWorkflowStepsRequest.model_json_schema(),
            ),
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        if name == SCHEDULE_WORKFLOW_TOOL_NAME:
            return await self._call_schedule_workflow(arguments)
        if name == CANCEL_WORKFLOW_RUN_TOOL_NAME:
            return await self._call_cancel_workflow_run(arguments)
        if name == APPEND_WORKFLOW_STEPS_TOOL_NAME:
            return await self._call_append_workflow_steps(arguments)
        return CallToolResult(
            content=[TextContent(type="text", text=f"unknown tool: {name}")], is_error=True
        )

    async def _call_schedule_workflow(self, arguments: dict[str, Any]) -> CallToolResult:
        try:
            request = ScheduleWorkflowRequest.model_validate(arguments)
        except ValidationError as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))], is_error=True
            )

        try:
            due_at = resolve_schedule(
                delay_seconds=request.delay_seconds,
                at=request.at,
                now=self._clock(),
                zone=self._zone,
            )
        except ScheduleError as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))], is_error=True
            )

        run = await self._repository.create_run(
            origin="voice",
            summary=request.summary,
            steps=[
                WorkflowStepSpec(kind=step.kind, arguments=step.arguments)
                for step in request.steps
            ],
            base_time=due_at,
            created_by_user_id=None,
        )
        return CallToolResult(
            content=[TextContent(type="text", text=f"scheduled: {request.summary}")],
            structured_content={"run_id": run.id, "status": run.status},
        )

    async def _call_cancel_workflow_run(self, arguments: dict[str, Any]) -> CallToolResult:
        try:
            request = CancelWorkflowRunRequest.model_validate(arguments)
        except ValidationError as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))], is_error=True
            )

        cancelled = await self._repository.cancel_run(
            request.run_id, now=self._clock(), cancelled_by_user_id=None
        )
        if not cancelled:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"run {request.run_id} is not a pending or firing run -- there "
                            "is nothing to cancel"
                        ),
                    )
                ],
                is_error=True,
            )
        return CallToolResult(
            content=[TextContent(type="text", text=f"cancelled run {request.run_id}")],
            structured_content={"run_id": request.run_id, "status": "cancelled"},
        )

    async def _call_append_workflow_steps(self, arguments: dict[str, Any]) -> CallToolResult:
        try:
            request = AppendWorkflowStepsRequest.model_validate(arguments)
        except ValidationError as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))], is_error=True
            )

        specs = [
            WorkflowStepSpec(kind=step.kind, arguments=step.arguments) for step in request.steps
        ]
        try:
            run = await self._repository.append_steps(
                request.run_id, specs, now=self._clock()
            )
        except WorkflowRunNotFoundError:
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"run {request.run_id} does not exist")
                ],
                is_error=True,
            )
        except WorkflowRunNotAppendableError as exc:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"run {request.run_id} has already started (status="
                            f"{exc.status!r}) -- steps can only be added to a run that "
                            "has not started firing yet"
                        ),
                    )
                ],
                is_error=True,
            )
        return CallToolResult(
            content=[
                TextContent(
                    type="text", text=f"added {len(specs)} step(s) to run {request.run_id}"
                )
            ],
            structured_content={
                "run_id": run.id,
                "status": run.status,
                "step_count": len(run.steps),
            },
        )
