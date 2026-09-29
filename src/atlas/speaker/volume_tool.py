"""`VolumeToolHost`: the in-process `set_speaker_volume` tool.

The brain calls it to change the volume of Atlas's own speaker, the edge
device that heard the current turn. It joins the same `McpToolHostLookup`
as the workflow tools (`workflow/tool.py`) and holds no credential, so
there is no isolation boundary to cross.

The target is a per-task `ContextVar`. `app.py` sets it in each turn's task
before `run_turn`: the edge source for an edge turn, `None` for every other
source. The default is `None`, so an unset target fails closed and the tool
refuses (the same reason `workflow/tool.py` scopes run ids per task).
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Literal, Protocol

from mcp.types import CallToolResult, TextContent, Tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from atlas.transports.edge import EdgeVolumeError

SET_SPEAKER_VOLUME_TOOL_NAME = "set_speaker_volume"


class VolumeTarget(Protocol):
    """What the tool needs from the speaker: `EdgeAudioSource.set_volume`."""

    async def set_volume(self, *, level: "int | None" = None, direction: "str | None" = None) -> int: ...


_turn_volume_target: "ContextVar[VolumeTarget | None]" = ContextVar("atlas_turn_volume_target", default=None)


def set_current_turn_target(target: "VolumeTarget | None") -> None:
    """Set the speaker this task's turn may change. `None` means none."""
    _turn_volume_target.set(target)


class SetSpeakerVolumeRequest(BaseModel):
    """Give `level` (a percent) or `direction` (one step), never both."""

    model_config = ConfigDict(extra="forbid")

    level: int | None = Field(
        default=None, ge=0, le=100, description="The volume as a percent, 0 to 100. Do not give it with direction."
    )
    direction: Literal["up", "down"] | None = Field(
        default=None, description="One step louder (up) or quieter (down). Do not give it with level."
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "SetSpeakerVolumeRequest":
        if (self.level is None) == (self.direction is None):
            raise ValueError("give exactly one of level and direction")
        return self


def _error(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)


class VolumeToolHost:
    """Satisfies the tool-host shape `McpToolHostLookup` expects: `.tools`
    and `call_tool(name, arguments)`."""

    def __init__(self) -> None:
        self.tools: list[Tool] = [
            Tool(
                name=SET_SPEAKER_VOLUME_TOOL_NAME,
                description=(
                    "Set the volume of Atlas's own speaker in the room that heard this request. "
                    "It changes the volume of Atlas's replies and of any music Atlas plays. "
                    "Give level for a percent, or direction for one step, never both. "
                    "Do not use it for a TV or a media player: the Home Assistant tools do that. "
                    "The house can limit the range, so tell the operator the level the result gives."
                ),
                input_schema=SetSpeakerVolumeRequest.model_json_schema(),
            )
        ]

    def set_current_turn_target(self, target: "VolumeTarget | None") -> None:
        set_current_turn_target(target)

    async def call_tool(self, name: str, arguments: "dict[str, Any] | None") -> CallToolResult:
        if name != SET_SPEAKER_VOLUME_TOOL_NAME:
            return _error(f"unknown tool: {name}")
        try:
            request = SetSpeakerVolumeRequest.model_validate(arguments or {})
        except ValidationError as exc:
            return _error(str(exc))
        target = _turn_volume_target.get()
        if target is None:
            return _error("this room has no speaker volume control")
        try:
            level = await target.set_volume(level=request.level, direction=request.direction)
        except EdgeVolumeError as exc:
            return _error(str(exc))
        return CallToolResult(
            content=[TextContent(type="text", text=f"speaker volume is now {level} percent")],
            structured_content={"level": level},
        )
