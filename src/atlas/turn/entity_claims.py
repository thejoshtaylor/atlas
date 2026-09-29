"""Entity claims: two turns in one group cannot change the same entity.

The first turn whose write reaches an entity claims it. A later write from
another turn in the same group on that entity is refused, and the refusal is
an error-shaped tool result whose text the turn speaks as it is written
(D-13). The turn controller already speaks such a result verbatim, so no new
reply channel exists and no model rewords the sentence.

Rules that shape this module:

- The claim key is the per-run turn key the server gives each turn. It is the
  `owner` constructor argument. `call_tool` never reads an identity from the
  model's arguments, and a speaker label is only text in the refusal (Phase 11
  D-15, T-12-08).
- A claim never expires and no turn releases it when it ends. The group drops
  the whole registry when its last turn ends (D-14).
- Only `ha_call_service` with a service whose name does not start with `get_`
  and `ha_play_spotify_playlist` take a claim. Reads and every other tool are
  never blocked (D-15). `get_` is the Home Assistant naming rule for a
  response-only service such as `todo.get_items` (Assumption A1).
- The check and the set are one synchronous step with no `await` between them,
  so the event loop needs no lock.
- `unclaimed(host)` gives back the inner host. A macro runs without claims, and
  the scheduler never sees a wrapper (D-16).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from atlas_mcp.ha_names import HA_EXPAND_TARGET_TOOL, HA_WRITE_TOOL_NAMES

CLAIM_TAKEN_EVENT = "claim.taken"
CLAIM_REFUSED_EVENT = "claim.refused"

_TARGET_ARGUMENTS: tuple[tuple[str, str], ...] = (
    ("area", "area_id"),
    ("device", "device_id"),
    ("label", "label_id"),
)

_UNRESOLVED_TARGET_REPLY = "i can't tell which devices that reaches"


@dataclass(frozen=True)
class ClaimRefusal:
    """The first entity in a request that another turn already holds, and the
    label that turn had when it took the claim."""

    entity_id: str
    holder_label: str | None


class ClaimRegistry:
    """Which turn holds which entity. One registry per turn group."""

    def __init__(self) -> None:
        self._held: dict[str, tuple[str, str | None]] = {}

    def claim(self, entities: frozenset[str], owner: str, label: str | None) -> ClaimRefusal | None:
        """Claim every entity for `owner`, or claim none and return the first
        conflict. Synchronous on purpose: no `await` sits between the check
        and the set. The same owner may claim an entity again."""
        for entity_id in sorted(entities):
            held = self._held.get(entity_id)
            if held is not None and held[0] != owner:
                return ClaimRefusal(entity_id=entity_id, holder_label=held[1])
        for entity_id in entities:
            if entity_id not in self._held:
                self._held[entity_id] = (owner, label)
        return None

    def release(self, entities: frozenset[str], owner: str) -> None:
        """Drop the entries `owner` holds. Another owner's entries stay."""
        for entity_id in entities:
            held = self._held.get(entity_id)
            if held is not None and held[0] == owner:
                del self._held[entity_id]

    def holder(self, entity_id: str) -> tuple[str, str | None] | None:
        return self._held.get(entity_id)


@dataclass(frozen=True)
class ClaimTargets:
    """What one write reaches. `entity_ids` are named directly. Each `expand`
    entry is a `(kind, target_id)` pair that only the Home Assistant child can
    turn into entity ids."""

    entity_ids: frozenset[str]
    expand: tuple[tuple[str, str], ...]


def _direct_entities(value: Any) -> frozenset[str]:
    if isinstance(value, str) and value.strip():
        return frozenset({value.strip()})
    if isinstance(value, (list, tuple)):
        return frozenset(item.strip() for item in value if isinstance(item, str) and item.strip())
    return frozenset()


def claim_targets(bare_name: str, arguments: dict) -> ClaimTargets | None:
    """The targets a call would change, or `None` when it takes no claim."""
    if bare_name not in HA_WRITE_TOOL_NAMES or not isinstance(arguments, dict):
        return None
    entity_ids = _direct_entities(arguments.get("entity_id"))
    expand: tuple[tuple[str, str], ...] = ()
    if bare_name == "ha_call_service":
        service = arguments.get("service")
        if isinstance(service, str) and service.startswith("get_"):
            return None
        expand = tuple(
            (kind, arguments[key].strip())
            for kind, key in _TARGET_ARGUMENTS
            if isinstance(arguments.get(key), str) and arguments[key].strip()
        )
    if not entity_ids and not expand:
        return None
    return ClaimTargets(entity_ids=entity_ids, expand=expand)


def _spoken_entity(entity_id: str, friendly_name: str | None) -> str:
    if friendly_name and friendly_name.strip():
        return friendly_name.strip()
    return entity_id.split(".", 1)[-1].replace("_", " ")


def claim_refusal_text(refusal: ClaimRefusal, friendly_name: str | None) -> str:
    """The sentence the refused turn speaks. Built here from the holder label,
    the entity name, and a fixed template. It never passes through a model."""
    holder = refusal.holder_label.strip() if refusal.holder_label and refusal.holder_label.strip() else None
    return f"{holder or 'another request'} just changed the {_spoken_entity(refusal.entity_id, friendly_name)}."


def _error_result(text: str, *, claim_refusal: bool) -> SimpleNamespace:
    return SimpleNamespace(isError=True, content=[SimpleNamespace(text=text)], claim_refusal=claim_refusal)


def is_claim_refusal(result: Any) -> str | None:
    """The refusal text when `result` is a claim refusal, else `None`."""
    if getattr(result, "claim_refusal", False) is not True or not getattr(result, "isError", False):
        return None
    content = getattr(result, "content", None) or []
    text = getattr(content[0], "text", None) if content else None
    return text if isinstance(text, str) else None


def _is_error_result(result: Any) -> bool:
    return bool(getattr(result, "isError", getattr(result, "is_error", False)))


def _result_payload(result: Any) -> Any:
    if isinstance(result, dict):
        return result
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    content = getattr(result, "content", None) or []
    text = getattr(content[0], "text", None) if content else None
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _failed(result: Any) -> bool:
    """True when a forwarded write changed nothing: an error result, or a dict
    payload with an `error` key (`handle_call_service` returns a non-2xx that
    way)."""
    if _is_error_result(result):
        return True
    payload = _result_payload(result)
    return isinstance(payload, dict) and "error" in payload


class ClaimingToolHost:
    """Wraps one turn's tool host and claims entities before a write goes
    through. Same shape as `RenamedToolHostView`: `.tools` and `call_tool`."""

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
        self._registry = registry
        self._owner = owner
        self._label = label
        self._friendly_name = friendly_name
        self._claimable_names = claimable_names
        self._expand_tool_name = expand_tool_name
        self._record_event = record_event

    @property
    def tools(self) -> Any:
        return self.inner.tools

    def _record(self, event_type: str, entity_ids: frozenset[str]) -> None:
        if self._record_event is not None:
            self._record_event({"type": event_type, "entity_ids": sorted(entity_ids), "turn_key": self._owner})

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        bare_name = name.rsplit("__", 1)[-1] if "__" in name else name
        if self._claimable_names is not None and bare_name not in self._claimable_names:
            return await self.inner.call_tool(name, arguments)
        targets = claim_targets(bare_name, arguments)
        if targets is None:
            return await self.inner.call_tool(name, arguments)

        entities = targets.entity_ids
        if targets.expand:
            expanded = await self._expand(name, bare_name, targets.expand)
            if not isinstance(expanded, frozenset):
                return expanded
            entities = entities | expanded

        # From here to the claim there is no `await`, so no other turn runs
        # between the check and the set.
        newly_claimed = frozenset(entity_id for entity_id in entities if self._registry.holder(entity_id) is None)
        refusal = self._registry.claim(entities, self._owner, self._label())
        if refusal is not None:
            self._record(CLAIM_REFUSED_EVENT, entities)
            text = claim_refusal_text(refusal, self._friendly_name(refusal.entity_id))
            return _error_result(text, claim_refusal=True)
        self._record(CLAIM_TAKEN_EVENT, entities)
        try:
            result = await self.inner.call_tool(name, arguments)
        except BaseException:
            self._registry.release(newly_claimed, self._owner)
            raise
        if _failed(result):
            self._registry.release(newly_claimed, self._owner)
        return result

    async def _expand(self, name: str, bare_name: str, expand: tuple[tuple[str, str], ...]) -> "frozenset[str] | Any":
        """The union of the entity ids the child gives for every target, or the
        result to return in place of the write: the child's own refusal when it
        cannot expand, or a fixed sentence when there is no way to ask."""
        if self._expand_tool_name is None:
            return _error_result(_UNRESOLVED_TARGET_REPLY, claim_refusal=False)
        # A collision prefix on the write's name (`slug__ha_call_service`)
        # belongs to the same plugin as the expand tool.
        prefix = name[: len(name) - len(bare_name)]
        expand_name = f"{prefix}{self._expand_tool_name}"
        union: set[str] = set()
        for kind, target_id in expand:
            result = await self.inner.call_tool(expand_name, {"kind": kind, "target_id": target_id})
            if _is_error_result(result):
                return result
            payload = _result_payload(result)
            ids = payload.get("entity_ids") if isinstance(payload, dict) else None
            if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids) or not ids:
                return _error_result(_UNRESOLVED_TARGET_REPLY, claim_refusal=False)
            union.update(ids)
        return frozenset(union)


def unclaimed(host: Any) -> Any:
    """The host with no claim wrapper, so a macro runs without claims (D-16)."""
    while isinstance(host, ClaimingToolHost):
        host = host.inner
    return host
