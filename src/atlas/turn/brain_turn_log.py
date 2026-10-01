"""The brain-turn intent log: what was said, and what the brain meant.

Every turn that reaches the brain tier race leaves one record. The record
holds the transcript, a wake-stripped normalized form with its sha256
fingerprint, the ordered tool calls (name, arguments, read or write, ok or
error), the spoken reply, the answering tier and the brain latency.

This module records only. It never dispatches a call and never matches a
transcript. Nothing reads a stored row back into a turn (D-14). It stores the
calls, not their results, so a future local path re-runs live tools and does
not replay stale state.

Privacy: a transcript is untrusted input from the room, and it is household
speech. Every text field has a length cap. A log line from this module names
a turn id and a count, never a transcript, a reply or an argument. The rows
expire through `RetentionScheduler`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from atlas_mcp.ha_names import HA_EXPAND_TARGET_TOOL, HA_WRITE_TOOL_NAMES

from atlas.db.brain_turn_repository import BrainTurnEntry
from atlas.turn import entity_claims
from atlas.turn.local_intent import _normalize, _strip_wake_prefix

logger = logging.getLogger("atlas.turn.brain_turn_log")

MAX_TEXT_CHARS = 2000
MAX_ARGUMENTS_JSON_CHARS = 4096
MAX_TOOL_CALLS = 32
MAX_ERROR_CHARS = 200
MAX_TOOL_NAME_CHARS = 200
MAX_OUTCOME_CHARS = 64
MAX_TIER_MODEL_CHARS = 200
MAX_PENDING_BRAIN_TURN_WRITES = 64

# The read-only tools in this codebase. A name outside this set and outside
# `HA_WRITE_TOOL_NAMES` is unknown, and unknown counts as a write.
_READ_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "ha_get_state",
        "ha_list_entities",
        HA_EXPAND_TARGET_TOOL,
        "weather_current",
        "weather_forecast",
        "calendar_list_events",
        "gmail_list_unread",
        "gmail_search",
        "gmail_read",
        "gmail_fetch_body",
    }
)


def classify_tool_call(name: str, arguments: Any) -> str:
    """"read" or "write". Unknown tools are "write"."""
    bare = entity_claims.bare_tool_name(name)
    if entity_claims.is_home_write(bare, arguments):
        return "write"
    if bare in _READ_TOOL_NAMES or bare in HA_WRITE_TOOL_NAMES:
        # The second case is a `get_` service on `ha_call_service`, which
        # `is_home_write` already ruled out as a write.
        return "read"
    return "write"


def _capped(text: Any, limit: int) -> str:
    return str(text)[:limit]


def _safe_arguments(arguments: Any) -> dict:
    """Arguments as plain JSON. Over the cap, they become a marker."""
    if not isinstance(arguments, dict):
        return {"_value": _capped(arguments, MAX_ARGUMENTS_JSON_CHARS)}
    try:
        encoded = json.dumps(arguments, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return {"_truncated": True}
    if len(encoded) > MAX_ARGUMENTS_JSON_CHARS:
        return {"_truncated": True}
    return json.loads(encoded)


def _first_text(result: Any) -> str:
    content = getattr(result, "content", None) or []
    text = getattr(content[0], "text", None) if content else None
    return text if isinstance(text, str) else ""


class RecordingToolHost:
    """Wraps a tool host and notes each call. It only observes: the inner
    result is returned unchanged, and an inner exception is re-raised. Same
    shape as `HomeControlGuardHost`: `.inner`, `.tools`, `call_tool`."""

    def __init__(self, inner: Any, record: "TurnBrainRecord") -> None:
        self.inner = inner
        self._record = record

    @property
    def tools(self) -> Any:
        return getattr(self.inner, "tools", None)

    async def call_tool(self, name: str, arguments: Any) -> Any:
        # Appended before the first await, so `asyncio.gather` keeps the
        # order the model asked for.
        call = self._record.begin_call(name, arguments)
        try:
            result = await self.inner.call_tool(name, arguments)
        except BaseException as exc:
            call["ok"] = False
            call["error"] = _capped(type(exc).__name__, MAX_ERROR_CHARS)
            raise
        if entity_claims._failed(result):
            call["ok"] = False
            call["error"] = _capped(_first_text(result), MAX_ERROR_CHARS)
        else:
            call["ok"] = True
        return result


@dataclass
class TurnBrainRecord:
    """Collects one turn's facts, then builds the `BrainTurnEntry`."""

    turn_id: str
    transcript: str
    normalized_transcript: str
    transcript_fingerprint: str
    continuation: bool
    started_at: float
    calls: list[dict] = field(default_factory=list)
    reply_text: "str | None" = None
    tier_index: "int | None" = None
    tier_model: "str | None" = None

    @classmethod
    def start(
        cls, *, turn_id: str, transcript: str, command_text: str, continuation: bool
    ) -> "TurnBrainRecord":
        normalized = _capped(_strip_wake_prefix(_normalize(command_text)), MAX_TEXT_CHARS)
        return cls(
            turn_id=turn_id,
            transcript=_capped(transcript, MAX_TEXT_CHARS),
            normalized_transcript=normalized,
            transcript_fingerprint=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            continuation=continuation,
            started_at=time.monotonic(),
        )

    def wrap(self, host: Any) -> Any:
        return host if host is None else RecordingToolHost(host, self)

    def begin_call(self, name: str, arguments: Any) -> dict:
        call = {
            "name": _capped(name, MAX_TOOL_NAME_CHARS),
            "arguments": _safe_arguments(arguments),
            "kind": classify_tool_call(name, arguments),
            "ok": None,
            "error": None,
        }
        if len(self.calls) < MAX_TOOL_CALLS:
            self.calls.append(call)
        return call

    def note_winner(self, winner: Any, tier_tasks: Any, tiers: Any) -> None:
        """Record which tier answered. Never raises."""
        try:
            for index, task in tier_tasks.items():
                if task.done() and not task.cancelled() and task.exception() is None:
                    if task.result() is winner:
                        self.tier_index = index
                        for tier in tiers:
                            if getattr(tier, "index", None) == index:
                                model = getattr(tier, "model", None)
                                self.tier_model = None if model is None else _capped(model, MAX_TIER_MODEL_CHARS)
                        return
        except Exception:  # noqa: BLE001 -- recording must never affect a turn
            return

    def note_reply(self, text: str) -> None:
        self.reply_text = _capped(text, MAX_TEXT_CHARS)

    def finish(
        self,
        *,
        outcome: str,
        failed: bool,
        cancelled: bool,
        tool_rounds_done_at: "float | None",
        now: "datetime | None" = None,
    ) -> BrainTurnEntry:
        if failed:
            final_outcome = "failed"
        elif cancelled:
            final_outcome = "cancelled"
        else:
            final_outcome = _capped(outcome, MAX_OUTCOME_CHARS)
        latency: "int | None" = None
        if tool_rounds_done_at is not None and tool_rounds_done_at >= self.started_at:
            latency = round((tool_rounds_done_at - self.started_at) * 1000)
        return BrainTurnEntry(
            turn_id=self.turn_id,
            created_at=now or datetime.now(timezone.utc),
            transcript=self.transcript,
            normalized_transcript=self.normalized_transcript,
            transcript_fingerprint=self.transcript_fingerprint,
            continuation=self.continuation,
            tool_calls=tuple(self.calls),
            reply_text=self.reply_text,
            tier_index=self.tier_index,
            tier_model=self.tier_model,
            brain_latency_ms=latency,
            outcome=final_outcome,
        )


class BrainTurnLog:
    """Bounded fire-and-forget writer. `submit` never awaits and never
    raises, so a failing or hung store cannot delay or fail a reply (D-17).
    The same shape as the wake-event writes in `sources/runner.py`."""

    def __init__(self, repo: Any, *, max_pending: int = MAX_PENDING_BRAIN_TURN_WRITES) -> None:
        self._repo = repo
        self._max_pending = max_pending
        self._pending: "set[asyncio.Task[None]]" = set()
        self._warned_full = False

    def submit(self, entry: BrainTurnEntry) -> None:
        try:
            if len(self._pending) >= self._max_pending:
                if not self._warned_full:
                    self._warned_full = True
                    logger.warning(
                        "%d brain-turn writes are already in flight and none are completing; "
                        "skipping further writes until they drain. The store is reachable "
                        "but not answering.",
                        len(self._pending),
                    )
                return
            self._warned_full = False
            task = asyncio.create_task(self._write(entry))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        except Exception as exc:  # noqa: BLE001 -- recording must never affect a turn
            logger.error(
                "failed to schedule a brain-turn write (%s); the reply was not affected",
                type(exc).__name__,
            )

    async def _write(self, entry: BrainTurnEntry) -> None:
        try:
            await self._repo.record_brain_turn(entry)
        except Exception as exc:  # noqa: BLE001
            # The exception type only, with no traceback and no message: a
            # database error can echo the statement parameters, and those
            # hold the transcript.
            logger.error(
                "failed to record brain turn %s (%s); the reply was not affected",
                entry.turn_id,
                type(exc).__name__,
            )

    async def drain(self, timeout: float = 2.0) -> None:
        """Let in-flight writes finish before the engine is disposed. Anything
        still running after `timeout` is cancelled."""
        pending = set(self._pending)
        if not pending:
            return
        _done, still_running = await asyncio.wait(pending, timeout=timeout)
        for task in still_running:
            task.cancel()
        if still_running:
            await asyncio.gather(*still_running, return_exceptions=True)
            logger.warning(
                "%d brain-turn write(s) did not finish within %.1fs of shutdown and were cancelled",
                len(still_running),
                timeout,
            )
