"""`TimerToolHost`: the in-process timer and alarm tools.

It holds a repository and no credential, so there is no isolation boundary to
cross. It joins `McpToolHostLookup` with the workflow and volume hosts and has
the shape the lookup expects: `.tools` and `call_tool(name, arguments)`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from mcp.types import CallToolResult, TextContent, Tool
from pydantic import ValidationError

from atlas.db.timer_repository import TimerRepository
from atlas.timers import service
from atlas.timers.core import TimerError, TimerSpec, spoken_duration

SET_TIMER_TOOL_NAME = "set_timer"


def _error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


def _ok(text: str, **structured: Any) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=structured)


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
        ]

    async def call_tool(self, name: str, arguments: "dict[str, Any] | None") -> CallToolResult:
        try:
            if name == SET_TIMER_TOOL_NAME:
                spec = TimerSpec.model_validate(arguments or {})
                timer = await service.create_timer(self._repository, spec, now=self._clock())
                prefix = f"{timer.label} timer" if timer.label else "timer"
                return _ok(
                    f"{prefix} set for {spoken_duration(spec.duration_seconds)}",
                    id=timer.id,
                    kind="timer",
                )
            return _error(f"unknown tool: {name}")
        except ValidationError as exc:
            return _error(str(exc))
        except TimerError as exc:
            return _error(str(exc))
