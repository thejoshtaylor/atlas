"""Tests for `spire_mcp.ha`'s tool handlers, gated by `spire_mcp.safety`'s
`allow_call`/`allow_read` boundary.

Turned green by plan 01-03. Every entity id below is invented, following the
rule `safety.py`'s own self-check already states.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
from mcp.client.stdio import get_default_environment

from spire_mcp.ha import handle_call_service, handle_get_state
from spire_mcp.safety import Denied, Policy

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


# SAFE-09: plan 03-04 gives the child a second protocol to Home Assistant
# (mcp/spire_mcp/registry.py's WebSocket connection). The child must hold
# the same one credential it held before -- HA_TOKEN -- and nothing this
# plan's own parent process might carry for an unrelated purpose. Named so
# a failing assertion says exactly which secret leaked.
_HOSTILE_PARENT_SECRETS = {
    "XAI_API_KEY": "sk-hostile-parent-secret-should-never-reach-the-ha-child",
    "DATABASE_URL": "postgresql+asyncpg://hostile:secret@db.invalid/hostile",
    "SPIRE_SECRET_KEY": "hostile-fernet-key-should-never-leak-into-the-ha-child",
}


def test_the_child_never_sees_secrets_the_parent_holds_for_other_plugins(monkeypatch):
    """A deliberately hostile parent environment -- a provider API key, a
    database connection string, and an encryption key, none of which the
    Home Assistant child has any business seeing -- must never reach it.

    Asserted on the child's own view of its environment, printed from
    inside a real spawned child, not on the dictionary this test built:
    the installed `mcp` SDK merges whatever explicit `env=` a caller passes
    over its own `get_default_environment()` allow-list
    (`HOME`/`LOGNAME`/`PATH`/`SHELL`/`TERM`/`USER` on POSIX -- no secrets),
    it does not replace the process's environment outright
    (03-RESEARCH.md Pitfall 4). Building `child_env` from that same
    allow-list function, read live against this test's own polluted
    `os.environ`, is what actually exercises that merge rather than
    asserting against a guess at what it does.
    """
    for name, value in _HOSTILE_PARENT_SECRETS.items():
        monkeypatch.setenv(name, value)

    # The exact shape mcp_client.py's McpToolHost.start() produces: the
    # SDK's own allow-list, merged with the explicit three keys the real
    # child spawn passes (HA_URL, HA_TOKEN, PYTHONPATH) -- never a copy of
    # this (now hostile) process's full environment.
    child_env = get_default_environment() | {
        "PYTHONPATH": f"{_REPO_ROOT}/src:{_REPO_ROOT}/mcp",
        "HA_URL": "http://ha.invalid:8123",
        "HA_TOKEN": "not-a-real-token",
    }

    code = textwrap.dedent(
        f"""
        import json
        import os
        import spire_mcp.ha  # import the real child module under this env
        print(json.dumps({{name: (name in os.environ) for name in {list(_HOSTILE_PARENT_SECRETS)!r}}}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=child_env, timeout=60
    )

    assert proc.returncode == 0, proc.stderr
    seen = json.loads(proc.stdout.strip().splitlines()[-1])

    for name in _HOSTILE_PARENT_SECRETS:
        assert not seen[name], f"{name} leaked into the Home Assistant MCP child's environment"
