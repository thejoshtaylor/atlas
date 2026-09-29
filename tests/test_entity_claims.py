"""D-13 to D-16 (12-03-PLAN.md): two turns in one group cannot change the
same entity. The first write claims it. A later write from another turn is
refused with a sentence the refused turn speaks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from atlas.turn.entity_claims import (
    CLAIM_REFUSED_EVENT,
    CLAIM_TAKEN_EVENT,
    ClaimingToolHost,
    ClaimRegistry,
    is_claim_refusal,
)


class _FakeInnerHost:
    """Records every call and answers with a bare `{"changed": []}` result."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        return SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(text='{"changed": []}')],
            structured_content={"changed": []},
        )


def _host(
    inner: _FakeInnerHost,
    registry: ClaimRegistry,
    owner: str,
    label: str | None,
    *,
    friendly: dict[str, str] | None = None,
    events: list[dict] | None = None,
) -> ClaimingToolHost:
    return ClaimingToolHost(
        inner,
        registry=registry,
        owner=owner,
        label=lambda: label,
        friendly_name=lambda entity_id: (friendly or {}).get(entity_id),
        record_event=events.append if events is not None else None,
    )


_LAMP = {"domain": "light", "service": "turn_on", "entity_id": "light.lamp"}


async def test_second_turn_write_to_a_claimed_entity_is_refused_with_a_spoken_sentence():
    registry = ClaimRegistry()
    inner = _FakeInnerHost()
    first = _host(inner, registry, "edge:1", "Josh", friendly={"light.lamp": "lamp"})
    second = _host(inner, registry, "edge:2", "Sam", friendly={"light.lamp": "lamp"})

    first_result = await first.call_tool("ha_call_service", dict(_LAMP))
    second_result = await second.call_tool("ha_call_service", dict(_LAMP))

    assert first_result.isError is False
    assert len(inner.calls) == 1
    assert second_result.isError is True
    assert second_result.content[0].text == "Josh just changed the lamp."
    assert is_claim_refusal(second_result) == "Josh just changed the lamp."
    assert is_claim_refusal(first_result) is None


async def test_entity_without_a_friendly_name_reads_as_its_object_id():
    registry = ClaimRegistry()
    inner = _FakeInnerHost()
    first = _host(inner, registry, "edge:1", "Josh")
    second = _host(inner, registry, "edge:2", "Sam")
    arguments = {"domain": "light", "service": "turn_on", "entity_id": "light.living_room_lamp"}

    await first.call_tool("ha_call_service", dict(arguments))
    refused = await second.call_tool("ha_call_service", dict(arguments))

    assert refused.content[0].text == "Josh just changed the living room lamp."


async def test_claimer_without_a_label_reads_as_another_request():
    registry = ClaimRegistry()
    inner = _FakeInnerHost()
    first = _host(inner, registry, "edge:1", None, friendly={"light.lamp": "lamp"})
    second = _host(inner, registry, "edge:2", "Sam", friendly={"light.lamp": "lamp"})

    await first.call_tool("ha_call_service", dict(_LAMP))
    refused = await second.call_tool("ha_call_service", dict(_LAMP))

    assert refused.content[0].text == "another request just changed the lamp."


async def test_the_same_turn_may_write_the_same_entity_again():
    registry = ClaimRegistry()
    inner = _FakeInnerHost()
    first = _host(inner, registry, "edge:1", "Josh")

    await first.call_tool("ha_call_service", dict(_LAMP))
    again = await first.call_tool("ha_call_service", {**_LAMP, "service": "turn_off"})

    assert again.isError is False
    assert len(inner.calls) == 2


async def test_a_different_entity_is_not_blocked():
    registry = ClaimRegistry()
    inner = _FakeInnerHost()
    first = _host(inner, registry, "edge:1", "Josh")
    second = _host(inner, registry, "edge:2", "Sam")

    await first.call_tool("ha_call_service", dict(_LAMP))
    other = await second.call_tool("ha_call_service", {**_LAMP, "entity_id": "light.desk"})

    assert other.isError is False
    assert len(inner.calls) == 2


async def test_claim_events_carry_the_turn_key_and_entities_and_never_a_label():
    registry = ClaimRegistry()
    inner = _FakeInnerHost()
    events: list[dict] = []
    first = _host(inner, registry, "edge:1", "Josh", events=events)
    second = _host(inner, registry, "edge:2", "Sam", events=events)

    await first.call_tool("ha_call_service", dict(_LAMP))
    await second.call_tool("ha_call_service", dict(_LAMP))

    assert [event["type"] for event in events] == [CLAIM_TAKEN_EVENT, CLAIM_REFUSED_EVENT]
    assert events[0]["entity_ids"] == ["light.lamp"]
    assert events[0]["turn_key"] == "edge:1"
    assert events[1]["entity_ids"] == ["light.lamp"]
    assert events[1]["turn_key"] == "edge:2"
    assert "Josh" not in repr(events)
    assert "Sam" not in repr(events)
