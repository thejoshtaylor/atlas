"""`TimerToolHost`: the in-process timer and alarm tools.

It holds a repository and no credential, so there is no isolation boundary to
cross. It joins `McpToolHostLookup` with the workflow and volume hosts and has
the shape the lookup expects: `.tools` and `call_tool(name, arguments)`.

Ids come from the "Timers and alarms" block in every turn's context
(`core.describe_timers`). Every entry is listed every turn, so a shown-id
guard would add no protection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas.db.timer_repository import Timer, TimerRepository
from atlas.timers import service
from atlas.timers.core import (
    AlarmSpec,
    TimerChanges,
    TimerError,
    TimerNotFoundError,
    TimerSpec,
    clock_text,
    mask_to_days,
    remaining_seconds,
    spoken_duration,
)

SET_TIMER_TOOL_NAME = "set_timer"
SET_ALARM_TOOL_NAME = "set_alarm"
UPDATE_TIMER_TOOL_NAME = "update_timer_or_alarm"
DELETE_TIMER_TOOL_NAME = "delete_timer_or_alarm"


class UpdateTimerRequest(TimerChanges):
    """`TimerChanges` plus the id of the entry to change."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="The id from the Timers and alarms list in context.")


class DeleteTimerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="The id from the Timers and alarms list in context.")


def _error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def _ok(text: str, timer: Timer) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content={"id": timer.id, "kind": timer.kind},
    )


def _name(timer: Timer) -> str:
    return f"{timer.label} {timer.kind}" if timer.label else timer.kind


def _state_text(timer: Timer, now: datetime) -> str:
    """One spoken sentence about the entry, built in code from its state."""
    if timer.kind == "timer":
        left = spoken_duration(remaining_seconds(timer, now) or 0)
        if timer.due_at is None:
            return f"{_name(timer)} paused with {left} left"
        return f"{_name(timer)} set with {left} left"
    if not timer.enabled:
        return f"{_name(timer)} turned off"
    days = mask_to_days(timer.repeat_days)
    when = f"on {' '.join(days)}" if days else "once"
    return f"{_name(timer)} set for {clock_text(timer.time_of_day or '00:00')} {when}"


class TimerToolHost:
    def __init__(
        self,
        repository: TimerRepository,
        *,
        zone: Any,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._repository = repository
        self._zone = zone
        self._clock = clock
        self.tools: list[Tool] = [
            Tool(
                name=SET_TIMER_TOOL_NAME,
                description=(
                    "Start a countdown the user calls a timer. When the countdown ends, the house "
                    "speaker chimes and says that the timer is done. Use it, not schedule_workflow, "
                    "for any timer. The label is the short name the user gave, for example pasta, "
                    "or empty."
                ),
                input_schema=TimerSpec.model_json_schema(),
            ),
            Tool(
                name=SET_ALARM_TOOL_NAME,
                description=(
                    "Set a wake-up or an alarm at a clock time in the house's time zone. The time is "
                    "24-hour HH:MM. days is the weekdays it repeats on, and it is empty for once. "
                    "Use it, not schedule_workflow, for alarms and wake-ups."
                ),
                input_schema=AlarmSpec.model_json_schema(),
            ),
            Tool(
                name=UPDATE_TIMER_TOOL_NAME,
                description=(
                    "Change a timer (pause, resume, add or set the time left, rename) or an alarm "
                    "(time, days, turn on or off, rename). Turn off my alarm means enabled false, "
                    "not delete. The id always comes from the Timers and alarms list in context. "
                    "Never ask the user for a number."
                ),
                input_schema=UpdateTimerRequest.model_json_schema(),
            ),
            Tool(
                name=DELETE_TIMER_TOOL_NAME,
                description=(
                    "Cancel or delete a timer or an alarm by its id, taken from the Timers and "
                    "alarms list in context."
                ),
                input_schema=DeleteTimerRequest.model_json_schema(),
            ),
        ]

    async def call_tool(self, name: str, arguments: "dict[str, Any] | None") -> CallToolResult:
        try:
            return await self._dispatch(name, arguments or {})
        except (ValidationError, TimerError) as exc:
            return _error(str(exc))

    async def _dispatch(self, name: str, arguments: "dict[str, Any]") -> CallToolResult:
        now = self._clock()
        if name == SET_TIMER_TOOL_NAME:
            spec = TimerSpec.model_validate(arguments)
            timer = await service.create_timer(self._repository, spec, now=now)
            return _ok(f"{_name(timer)} set for {spoken_duration(spec.duration_seconds)}", timer)
        if name == SET_ALARM_TOOL_NAME:
            alarm_spec = AlarmSpec.model_validate(arguments)
            timer = await service.create_alarm(self._repository, alarm_spec, now=now, zone=self._zone)
            return _ok(_state_text(timer, now), timer)
        if name == UPDATE_TIMER_TOOL_NAME:
            request = UpdateTimerRequest.model_validate(arguments)
            changes = TimerChanges.model_validate(request.model_dump(exclude={"id"}))
            timer = await service.update_timer(
                self._repository, request.id, changes, now=now, zone=self._zone
            )
            return _ok(_state_text(timer, now), timer)
        if name == DELETE_TIMER_TOOL_NAME:
            delete = DeleteTimerRequest.model_validate(arguments)
            existing = await self._repository.get_timer(delete.id)
            if existing is None:
                raise TimerNotFoundError(f"there is no timer or alarm with id {delete.id}")
            await service.delete_timer(self._repository, delete.id)
            return _ok(f"deleted the {_name(existing)}", existing)
        return _error(f"unknown tool: {name}")
