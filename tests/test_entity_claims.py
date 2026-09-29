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


# ---------------------------------------------------------------------------
# Task 2: expanded targets, reads, failures, and macros
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from atlas.turn.entity_claims import claim_targets, unclaimed  # noqa: E402


class _ScriptedHost:
    """Answers `ha_expand_target` from a table, and a write with a fixed
    outcome: ok, an error result, an `{"error": ...}` payload, or a raise."""

    def __init__(self, expansions: dict[tuple[str, str], Any] | None = None, write: str = "ok") -> None:
        self.expansions = expansions or {}
        self.write = write
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        if name == "ha_expand_target":
            return self.expansions[(arguments["kind"], arguments["target_id"])]
        if self.write == "raise":
            raise RuntimeError("child crashed")
        if self.write == "error_result":
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text="denied")])
        if self.write == "error_payload":
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text='{"error": "home assistant returned 500"}')],
                structured_content={"error": "home assistant returned 500"},
            )
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text='{"changed": []}')])


def _expanded(entity_ids: list[str]) -> Any:
    return SimpleNamespace(
        isError=False,
        content=[SimpleNamespace(text=json.dumps({"entity_ids": entity_ids}))],
        structured_content={"entity_ids": entity_ids},
    )


def _refusal(text: str) -> Any:
    return SimpleNamespace(isError=True, content=[SimpleNamespace(text=text)])


import json  # noqa: E402

_OFFICE = {"domain": "light", "service": "turn_on", "area_id": "office"}


async def test_an_area_target_claims_every_entity_the_expansion_returns():
    registry = ClaimRegistry()
    inner = _ScriptedHost({("area", "office"): _expanded(["light.desk", "light.lamp"])})
    first = _host(inner, registry, "edge:1", "Josh")
    second = _host(inner, registry, "edge:2", "Sam")

    result = await first.call_tool("ha_call_service", dict(_OFFICE))
    refused = await second.call_tool("ha_call_service", {"domain": "light", "service": "turn_off", "entity_id": "light.lamp"})

    assert result.isError is False
    assert [call[0] for call in inner.calls].count("ha_expand_target") == 1
    assert inner.calls[0] == ("ha_expand_target", {"kind": "area", "target_id": "office"})
    assert refused.content[0].text == "Josh just changed the lamp."
    assert registry.holder("light.desk") == ("edge:1", "Josh")


async def test_an_expanded_target_conflicts_when_any_one_entity_is_held():
    registry = ClaimRegistry()
    inner = _ScriptedHost({("area", "office"): _expanded(["light.desk", "light.lamp"])})
    first = _host(inner, registry, "edge:1", "Josh")
    second = _host(inner, registry, "edge:2", "Sam")

    await first.call_tool("ha_call_service", {"domain": "light", "service": "turn_on", "entity_id": "light.lamp"})
    refused = await second.call_tool("ha_call_service", dict(_OFFICE))

    assert refused.isError is True
    assert is_claim_refusal(refused) == "Josh just changed the lamp."
    # One refused entity refuses the whole call, and claims nothing of the rest.
    assert registry.holder("light.desk") is None
    assert [call[0] for call in inner.calls] == ["ha_call_service", "ha_expand_target"]


async def test_an_expansion_error_is_returned_unchanged_and_nothing_is_claimed():
    registry = ClaimRegistry()
    refusal = _refusal("i don't know a area called 'office'")
    inner = _ScriptedHost({("area", "office"): refusal})
    first = _host(inner, registry, "edge:1", "Josh")

    result = await first.call_tool("ha_call_service", dict(_OFFICE))

    assert result is refusal
    assert [call[0] for call in inner.calls] == ["ha_expand_target"]
    assert registry.holder("light.desk") is None


async def test_an_entity_id_together_with_an_area_claims_the_union():
    registry = ClaimRegistry()
    inner = _ScriptedHost({("area", "office"): _expanded(["light.desk"])})
    first = _host(inner, registry, "edge:1", "Josh")

    await first.call_tool("ha_call_service", {**_OFFICE, "entity_id": "light.lamp"})

    assert registry.holder("light.desk") == ("edge:1", "Josh")
    assert registry.holder("light.lamp") == ("edge:1", "Josh")


async def test_a_missing_expand_tool_refuses_the_call_and_claims_nothing():
    registry = ClaimRegistry()
    inner = _ScriptedHost()
    host = ClaimingToolHost(inner, registry=registry, owner="edge:1", label=lambda: "Josh", expand_tool_name=None)

    result = await host.call_tool("ha_call_service", dict(_OFFICE))

    assert result.isError is True
    assert is_claim_refusal(result) is None
    assert inner.calls == []
    assert registry.holder("light.desk") is None


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("ha_call_service", {"domain": "todo", "service": "get_items", "entity_id": "todo.shopping"}),
        ("ha_get_state", {"entity_id": "light.lamp"}),
        ("ha_list_entities", {}),
        ("weather_forecast", {"entity_id": "weather.home"}),
        ("google_calendar_list", {"entity_id": "calendar.home"}),
    ],
)
async def test_reads_and_other_tools_never_claim(name, arguments):
    registry = ClaimRegistry()
    inner = _ScriptedHost()
    first = _host(inner, registry, "edge:1", "Josh")
    second = _host(inner, registry, "edge:2", "Sam")

    await first.call_tool(name, dict(arguments))
    again = await second.call_tool(name, dict(arguments))

    assert again.isError is False
    assert len(inner.calls) == 2
    assert claim_targets(name, arguments) is None


async def test_playing_a_spotify_playlist_claims_the_media_player():
    registry = ClaimRegistry()
    inner = _ScriptedHost()
    first = _host(inner, registry, "edge:1", "Josh", friendly={"media_player.den": "den speaker"})
    second = _host(inner, registry, "edge:2", "Sam", friendly={"media_player.den": "den speaker"})
    arguments = {"entity_id": "media_player.den", "name": "morning"}

    await first.call_tool("ha_play_spotify_playlist", dict(arguments))
    refused = await second.call_tool("ha_play_spotify_playlist", dict(arguments))

    assert refused.content[0].text == "Josh just changed the den speaker."


@pytest.mark.parametrize("write", ["error_result", "error_payload", "raise"])
async def test_a_failed_write_releases_only_what_that_call_newly_claimed(write):
    registry = ClaimRegistry()
    inner = _ScriptedHost(write="ok")
    first = _host(inner, registry, "edge:1", "Josh")
    await first.call_tool("ha_call_service", {"domain": "light", "service": "turn_on", "entity_id": "light.lamp"})
    inner.write = write
    inner.expansions[("area", "office")] = _expanded(["light.desk", "light.lamp"])

    if write == "raise":
        with pytest.raises(RuntimeError):
            await first.call_tool("ha_call_service", dict(_OFFICE))
    else:
        result = await first.call_tool("ha_call_service", dict(_OFFICE))
        assert result is not None

    # The lamp was claimed by an earlier, successful write and stays claimed.
    assert registry.holder("light.lamp") == ("edge:1", "Josh")
    # The desk was newly claimed by the failed call, so it is free again.
    assert registry.holder("light.desk") is None


async def test_a_failed_write_does_not_release_another_turns_claim():
    registry = ClaimRegistry()
    registry.claim(frozenset({"light.lamp"}), "edge:2", "Sam")
    registry.release(frozenset({"light.lamp"}), "edge:1")

    assert registry.holder("light.lamp") == ("edge:2", "Sam")


def test_unclaimed_returns_the_inner_host_and_a_plain_host_unchanged():
    inner = _FakeInnerHost()
    wrapper = _host(inner, ClaimRegistry(), "edge:1", "Josh")

    assert unclaimed(wrapper) is inner
    assert unclaimed(inner) is inner


def test_tools_are_forwarded_when_the_inner_host_has_them():
    inner = _FakeInnerHost()
    inner.tools = ["a-tool"]  # type: ignore[attr-defined]
    wrapper = _host(inner, ClaimRegistry(), "edge:1", "Josh")

    assert wrapper.tools == ["a-tool"]
    assert hasattr(_host(_ScriptedHost(), ClaimRegistry(), "edge:1", "Josh"), "tools") is False


async def test_a_prefixed_tool_name_still_classifies_by_its_bare_name():
    registry = ClaimRegistry()
    inner = _FakeInnerHost()
    first = _host(inner, registry, "edge:1", "Josh")
    second = _host(inner, registry, "edge:2", "Sam")

    await first.call_tool("home__ha_call_service", dict(_LAMP))
    refused = await second.call_tool("home__ha_call_service", dict(_LAMP))

    assert is_claim_refusal(refused) is not None
    assert len(inner.calls) == 1
