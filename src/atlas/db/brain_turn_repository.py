"""`BrainTurnEntry`, `BrainTurn` and the `BrainTurnRepository` Protocol
(quick task 261001-mp8, D-14 to D-16).

The intent log records what was said and what the brain meant for each turn
that reached the brain tier race. This store is written after a turn ends.
Only retention and future tooling read it. A turn never reads it back: no
dispatch and no matching come from these rows (D-14).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class BrainTurnEntry:
    """One brain turn, ready to store.

    `tool_calls` is the ordered list of the calls the brain made. Each item
    is a dict with `name`, `arguments` (plain JSON), `kind` ("read" or
    "write"), `ok` and `error`. The log keeps what was meant, not the
    results. `created_at` is aware UTC. `continuation` is True when the turn
    answered an earlier exchange, so its transcript needs that context.
    """

    turn_id: str
    created_at: datetime
    transcript: str
    normalized_transcript: str
    transcript_fingerprint: str
    continuation: bool
    tool_calls: "tuple[dict, ...]"
    reply_text: "str | None"
    tier_index: "int | None"
    tier_model: "str | None"
    brain_latency_ms: "int | None"
    outcome: str


@dataclass(frozen=True)
class BrainTurn(BrainTurnEntry):
    """A stored `BrainTurnEntry` with its row id."""

    id: int = 0


class BrainTurnRepository(Protocol):
    async def record_brain_turn(self, entry: BrainTurnEntry) -> None: ...

    async def list_brain_turns(self, limit: "int | None" = None) -> "list[BrainTurn]":
        """Newest first."""
        ...

    async def delete_brain_turns_before(self, cutoff: datetime) -> int: ...
