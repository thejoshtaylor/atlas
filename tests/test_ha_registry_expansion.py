"""An area, device, or label target expands to entity ids before
`allow_call` runs, or the call is refused -- never guessed at (SAFE-03,
SAFE-04).

`mcp/spire_mcp/ha.py`'s `handle_call_service` already forwards
`area_id`/`device_id`/`label_id` to `allow_call` as `unresolved_targets`,
which are always refused today -- Phase 1 never built expansion at all.
03-RESEARCH.md's Pitfall 2 names the specific way a naive expansion fails
even once built: reading only `entity_registry` and filtering on
`area_id` misses every entity whose area is inherited from its *device*
rather than set on the entity itself, which is the common case in a real
Home Assistant install. "Turn off everything in the office" would then
silently control fewer devices than the operator expects, with no error --
a functional bug that looks like a safety bug from the operator's chair.
The second test is the refusal side of the same decision: a target that
cannot be resolved at all must be refused with a named reason, never
treated as an empty (and therefore harmless-looking) expansion. The third
test is SAFE-04 restated for an expanded target specifically: one denied
entity inside an otherwise-allowed area must refuse the *whole* call, not
silently operate every other entity and skip the denied one.

Every id and entity id below is invented, following the rule `safety.py`'s
own self-check already states. `mcp/spire_mcp/registry.py`'s
`HaRegistryClient` answers over a real WebSocket connection this test
suite has no server for -- `tests/test_ha_registry_client.py` already
covers that client directly against a fake connection. Here, a
`_FakeRegistryClient` matching `HaRegistryClient`'s public `get_snapshot()`
surface stands in, following this codebase's own dependency-injection
convention (`FakeStt`, `FakeHomeAssistant`) rather than a subclass.
"""

from __future__ import annotations

import json

import pytest

from spire_mcp.ha import handle_call_service
from spire_mcp.registry import (
    RegistryDevice,
    RegistryEntity,
    RegistrySnapshot,
    RegistryUnavailableError,
    expand_area,
)
from spire_mcp.safety import Denied, Policy


class _FakeRegistryClient:
    """Matches `HaRegistryClient.get_snapshot()`'s public surface without
    ever touching a socket -- returns a fixed snapshot, or raises a fixed
    error, so a test drives `handle_call_service`'s expansion step against
    exactly the registry state it wants."""

    def __init__(self, *, snapshot: RegistrySnapshot | None = None, error: Exception | None = None) -> None:
        self._snapshot = snapshot
        self._error = error

    async def get_snapshot(self) -> RegistrySnapshot:
        if self._error is not None:
            raise self._error
        assert self._snapshot is not None
        return self._snapshot


def _office_snapshot() -> RegistrySnapshot:
    """One area (`area_example_office`), two devices in it, and three
    entities split the three ways expansion must tell apart: one with its
    own area set directly, one with no area of its own whose device sits
    in the office (inherited), and one whose own area overrides its
    device's area to somewhere else (excluded, not included)."""
    return RegistrySnapshot(
        areas=frozenset({"area_example_office", "area_example_workshop"}),
        devices={
            "device_example_a": RegistryDevice(device_id="device_example_a", area_id="area_example_office"),
            "device_example_b": RegistryDevice(device_id="device_example_b", area_id="area_example_office"),
        },
        labels=frozenset(),
        entities=(
            # Its own area is the office -- no device involved.
            RegistryEntity(entity_id="light.example_lamp", area_id="area_example_office", device_id=None),
            # No area of its own; inherits the office from device_example_a.
            RegistryEntity(entity_id="switch.example_fan", area_id=None, device_id="device_example_a"),
            # Its device (device_example_b) sits in the office, but this
            # entity's own area overrides that to the workshop -- excluded.
            RegistryEntity(
                entity_id="switch.example_elsewhere",
                area_id="area_example_workshop",
                device_id="device_example_b",
            ),
        ),
        fetched_at=0.0,
    )


def test_an_area_expands_to_entity_ids_including_device_inherited_area():
    """An area target must expand to every entity whose own `area_id`
    matches, plus every entity of a device whose `area_id` matches and
    whose entity has no override -- 03-RESEARCH.md Pitfall 2 -- plan 03-04
    fills this in."""
    entity_ids = expand_area(_office_snapshot(), "area_example_office")

    assert entity_ids == frozenset({"light.example_lamp", "switch.example_fan"})
    # The override case is the failure a naive expansion (device area
    # alone, ignoring an entity-level override) would get wrong the other
    # direction -- it must never appear here.
    assert "switch.example_elsewhere" not in entity_ids


async def test_an_unresolvable_target_is_refused_rather_than_guessed(fake_ha):
    """An area, device, or label id the registry does not recognize must be
    refused with a named reason, never expand to an empty or partial entity
    list treated as success -- plan 03-04 fills this in."""
    policy = Policy.from_config(None)
    snapshot = _office_snapshot()

    # Unknown: the registry answered, but has no such area.
    with pytest.raises(Denied) as unknown_exc:
        await handle_call_service(
            policy,
            fake_ha.client,
            "http://ha.invalid",
            "test-token",
            "light",
            "turn_on",
            area_id="area_example_nonexistent",
            registry=_FakeRegistryClient(snapshot=snapshot),
        )

    # Empty: the registry knows this area, and it holds nothing.
    empty_snapshot = RegistrySnapshot(
        areas=frozenset({"area_example_empty"}),
        devices={},
        labels=frozenset(),
        entities=(),
        fetched_at=0.0,
    )
    with pytest.raises(Denied) as empty_exc:
        await handle_call_service(
            policy,
            fake_ha.client,
            "http://ha.invalid",
            "test-token",
            "light",
            "turn_on",
            area_id="area_example_empty",
            registry=_FakeRegistryClient(snapshot=empty_snapshot),
        )

    # Unreachable: the registry itself could not be fetched at all -- this
    # must never fall back to expanding from a stale cached snapshot.
    with pytest.raises(Denied) as unreachable_exc:
        await handle_call_service(
            policy,
            fake_ha.client,
            "http://ha.invalid",
            "test-token",
            "light",
            "turn_on",
            area_id="area_example_office",
            registry=_FakeRegistryClient(error=RegistryUnavailableError("no route to host")),
        )

    reasons = {unknown_exc.value.reason, empty_exc.value.reason, unreachable_exc.value.reason}
    assert len(reasons) == 3, f"expected three distinguishable reasons, got {reasons}"
    # Nothing about any of the three refusals ever reached Home Assistant.
    assert len(fake_ha.requests) == 0


async def test_one_denied_entity_in_an_expanded_area_refuses_the_whole_call(fake_ha):
    """An area whose expansion includes even one denied entity (e.g.
    `switch.example_server_socket` alongside allowed entities like
    `light.example_lamp`) must refuse the entire call, not operate the
    allowed entities and skip the denied one -- plan 03-04 fills this in."""
    snapshot = RegistrySnapshot(
        areas=frozenset({"area_example_office"}),
        devices={},
        labels=frozenset(),
        entities=(
            RegistryEntity(entity_id="switch.example_fan", area_id="area_example_office", device_id=None),
            RegistryEntity(
                entity_id="switch.example_server_socket", area_id="area_example_office", device_id=None
            ),
        ),
        fetched_at=0.0,
    )
    policy = Policy.from_config({"deny_entities": ["switch.example_server_socket"]})

    with pytest.raises(Denied):
        await handle_call_service(
            policy,
            fake_ha.client,
            "http://ha.invalid",
            "test-token",
            "switch",
            "turn_off",
            area_id="area_example_office",
            registry=_FakeRegistryClient(snapshot=snapshot),
        )

    # The assertion that matters: nothing reached Home Assistant at all,
    # not even for the allowed entity in the same expanded area.
    assert len(fake_ha.requests) == 0


async def test_the_service_call_posts_every_checked_entity_id_not_only_the_first(fake_ha):
    """An expanded area carrying several allowed entities must post the
    whole checked list in one call -- the latent bug this plan fixes:
    `handle_call_service` used to post only `entity_ids[0]`, which was
    invisible while every call carried exactly one entity id."""
    snapshot = RegistrySnapshot(
        areas=frozenset({"area_example_office"}),
        devices={},
        labels=frozenset(),
        entities=(
            RegistryEntity(entity_id="switch.example_fan", area_id="area_example_office", device_id=None),
        ),
        fetched_at=0.0,
    )
    policy = Policy.from_config(None)

    result = await handle_call_service(
        policy,
        fake_ha.client,
        "http://ha.invalid",
        "test-token",
        "switch",
        "turn_off",
        "switch.example_server_socket",
        area_id="area_example_office",
        registry=_FakeRegistryClient(snapshot=snapshot),
    )

    assert len(fake_ha.requests) == 1
    request = fake_ha.requests[0]
    body = json.loads(request.read())
    assert sorted(body["entity_id"]) == ["switch.example_fan", "switch.example_server_socket"]
    assert "changed" in result
