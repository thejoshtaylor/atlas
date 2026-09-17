"""Tests for `spire_mcp.ha`'s tool handlers, gated by `spire_mcp.safety`'s
`allow_call`/`allow_read` boundary.

Turned green by plan 01-03. Every entity id below is invented, following the
rule `safety.py`'s own self-check already states.
"""

import pytest

from spire_mcp.ha import handle_call_service, handle_get_state
from spire_mcp.safety import Denied, Policy


async def test_service_call_allowed_entity(fake_ha):
    policy = Policy.from_config(None)

    result = await handle_call_service(
        policy,
        fake_ha.client,
        "http://ha.invalid",
        "test-token",
        "switch",
        "turn_on",
        "switch.example_fan",
    )

    assert len(fake_ha.requests) == 1
    request = fake_ha.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/api/services/switch/turn_on"
    assert "changed" in result


async def test_service_call_denied_entity_never_reaches_ha(fake_ha):
    policy = Policy.from_config({"deny_entities": ["switch.example_server_socket"]})

    with pytest.raises(Denied):
        await handle_call_service(
            policy,
            fake_ha.client,
            "http://ha.invalid",
            "test-token",
            "switch",
            "turn_off",
            "switch.example_server_socket",
        )

    # The assertion that matters is on the request count, not the raise: a
    # refusal that still reached Home Assistant is the failure this test
    # exists to catch.
    assert len(fake_ha.requests) == 0


async def test_read_denied_entity_succeeds(fake_ha):
    # No `Policy` is threaded into `handle_get_state` at all -- `allow_read`
    # never consults one, because the denylist blocks control and never
    # blocks reads.
    power = await handle_get_state(
        fake_ha.client, "http://ha.invalid", "test-token", "sensor.example_server_power"
    )
    assert power["state"] == "42.0"

    denied_switch = await handle_get_state(
        fake_ha.client, "http://ha.invalid", "test-token", "switch.example_server_socket"
    )
    assert denied_switch["state"] == "on"


async def test_unresolved_area_target_is_refused(fake_ha):
    policy = Policy.from_config(None)

    with pytest.raises(Denied):
        await handle_call_service(
            policy,
            fake_ha.client,
            "http://ha.invalid",
            "test-token",
            "switch",
            "turn_off",
            None,
            area_id="office",
        )

    assert len(fake_ha.requests) == 0
