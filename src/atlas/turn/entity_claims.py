"""Entity claims: skeleton, filled in by the next commit."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from atlas_mcp.ha_names import HA_EXPAND_TARGET_TOOL

CLAIM_TAKEN_EVENT = "claim.taken"
CLAIM_REFUSED_EVENT = "claim.refused"


@dataclass(frozen=True)
class ClaimRefusal:
    entity_id: str
    holder_label: str | None


class ClaimRegistry:
    def claim(self, entities: frozenset[str], owner: str, label: str | None) -> ClaimRefusal | None:
        return None

    def release(self, entities: frozenset[str], owner: str) -> None:
        return None

    def holder(self, entity_id: str) -> tuple[str, str | None] | None:
        return None


class ClaimingToolHost:
    def __init__(
        self,
        inner: Any,
        *,
        registry: ClaimRegistry,
        owner: str,
        label: Callable[[], str | None],
        friendly_name: Callable[[str], str | None] = lambda entity_id: None,
        claimable_names: frozenset[str] | None = None,
        expand_tool_name: str | None = HA_EXPAND_TARGET_TOOL,
        record_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.inner = inner

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self.inner.call_tool(name, arguments)


def is_claim_refusal(result: Any) -> str | None:
    return None


_ = SimpleNamespace
