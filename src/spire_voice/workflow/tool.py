"""`ScheduleWorkflowRequest` and `WorkflowToolHost`: the model-facing
`schedule_workflow` tool (D-08), joining the same `McpToolHostLookup` the
Home Assistant and weather children already sit in
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
request's own `delay_seconds`, an injected clock reading, the server's own
resolved local time zone object) is a plain pass-through.
"""

from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from typing import Any, Callable, Literal

from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel, Field, ValidationError, model_validator

from spire_voice.db.repository import WorkflowRepository, WorkflowStepSpec
from spire_voice.workflow.schedule import ScheduleError, resolve_schedule

SCHEDULE_WORKFLOW_TOOL_NAME = "schedule_workflow"


class ScheduleWorkflowRequest(BaseModel):
    """The model-facing shape of one `schedule_workflow` call (PA-D4).

    This plan ships `delay_seconds` only and one step per call; plan 05-05
    widens this to an ordered multi-step list and an absolute `at`. `kind`
    is a closed `Literal["wait", "call_service", "speak"]` -- the same
    "impossible by construction" discipline `FillerPhrase`
    (`providers/tier_reply.py`) already applies to its own closed set
    (D-05): a fourth kind, a condition, or a loop is unrepresentable here,
    not merely discouraged.
    """

    kind: Literal["wait", "call_service", "speak"]
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="The per-kind payload. For call_service: domain, service, and optionally entity_id.",
    )
    delay_seconds: int = Field(ge=0, description="How many seconds from now this step is due.")
    summary: str = Field(
        min_length=1,
        description="What the operator would say to mean this run -- used later to cancel it by voice.",
    )

    @model_validator(mode="after")
    def _call_service_carries_a_domain_and_service(self) -> "ScheduleWorkflowRequest":
        if self.kind == "call_service":
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
        return self


class WorkflowToolHost:
    """A small in-process tool host exposing exactly one tool,
    `schedule_workflow`, with a `.tools` list of `mcp.types.Tool` entries
    and an `async def call_tool(name, arguments)` -- the same structural
    shape `McpToolHostLookup` already builds its name-to-host map from
    (`mcp_client.py`'s own docstring: "a host that exposes those is
    routable without changing either function").

    A validation failure (a malformed call, or a refused schedule) returns
    an error-shaped `CallToolResult` the turn path already knows how to
    report (`turn/controller.py::_is_error`/`_result_text`) -- never a
    silently dropped schedule.
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
                    "Schedule one step (wait, call_service, or speak) to run later, after "
                    "a delay given in seconds. Nothing happens now -- the step runs on its "
                    "own once the delay has passed."
                ),
                input_schema=ScheduleWorkflowRequest.model_json_schema(),
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        del name  # this host advertises exactly one tool; the lookup already routed here
        try:
            request = ScheduleWorkflowRequest.model_validate(arguments)
        except ValidationError as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))], is_error=True
            )

        try:
            due_at = resolve_schedule(
                delay_seconds=request.delay_seconds,
                at=None,
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
            steps=[WorkflowStepSpec(kind=request.kind, arguments=request.arguments)],
            base_time=due_at,
            created_by_user_id=None,
        )
        return CallToolResult(
            content=[TextContent(type="text", text=f"scheduled: {request.summary}")],
            structured_content={"run_id": run.id, "status": run.status},
        )
