"""Tests for `atlas_mcp.ha`'s tool handlers, gated by `atlas_mcp.safety`'s
`allow_call`/`allow_read` boundary.

Turned green by plan 01-03. Every entity id below is invented, following the
rule `safety.py`'s own self-check already states.
"""

import inspect
import json
import os
import subprocess
import sys
import textwrap

import httpx
import pytest
from mcp.client.stdio import get_default_environment

import atlas_mcp.ha as ha_module
from atlas_mcp.ha import handle_call_service, handle_get_state
from atlas_mcp.safety import Denied, Policy

# The exact text Home Assistant answers with when a response-only service is
# called without `?return_response` (verified live against a real hub, see
# 260923-j8h-PLAN.md). Kept as one constant so every test that scripts this
# reply says the same thing HA actually says.
_REQUIRES_RESPONSES = (
    "Service call requires responses but caller did not ask for responses. "
    "Add ?return_response to query parameters."
)


class _ScriptedHa:
    """A Home Assistant stand-in that answers a fixed, ordered script of
    responses, one per request.

    An unbounded retry is a real bug this project must never ship quietly.
    Asking this stand-in for a response past the end of its script raises
    loudly, so a retry loop that got out of hand fails the test instead of
    hanging or silently wrapping around.
    """

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected extra request to home assistant")
        return self._responses.pop(0)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# CMD-01 follow-up (260923-j8h): a response-only service (for example
# `todo.get_items`) is always rejected the first time, because Home
# Assistant refuses to run one at all unless the caller asked for the
# response up front. The bounded retry below is what makes such a service
# reachable at all.


async def test_response_only_service_is_retried_once_with_return_response():
    policy = Policy.from_config(None)
    scripted = _ScriptedHa(
        [
            httpx.Response(400, json={"message": _REQUIRES_RESPONSES}),
            httpx.Response(
                200,
                json={
                    "changed_states": [],
                    "service_response": {
                        "todo.shopping_list": {
                            "items": [{"summary": "milk", "status": "needs_action"}]
                        }
                    },
                },
            ),
        ]
    )
    try:
        result = await handle_call_service(
            policy,
            scripted.client,
            "http://ha.invalid",
            "test-token",
            "todo",
            "get_items",
            "todo.shopping_list",
        )
    finally:
        await scripted.client.aclose()

    assert len(scripted.requests) == 2
    first, second = scripted.requests
    assert first.url.path == "/api/services/todo/get_items"
    assert second.url.path == "/api/services/todo/get_items"
    assert "return_response" not in first.url.params
    assert "return_response" in second.url.params
    assert json.loads(first.read()) == {"entity_id": ["todo.shopping_list"]}
    assert json.loads(second.read()) == {"entity_id": ["todo.shopping_list"]}
    assert result == {
        "changed": [],
        "response": {
            "todo.shopping_list": {"items": [{"summary": "milk", "status": "needs_action"}]}
        },
    }


async def test_policy_check_runs_once_across_a_return_response_retry(monkeypatch):
    real_allow_call = ha_module.allow_call
    call_count = 0

    def _counting_allow_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_allow_call(*args, **kwargs)

    monkeypatch.setattr(ha_module, "allow_call", _counting_allow_call)

    policy = Policy.from_config(None)
    scripted = _ScriptedHa(
        [
            httpx.Response(400, json={"message": _REQUIRES_RESPONSES}),
            httpx.Response(200, json={"changed_states": [], "service_response": {}}),
        ]
    )
    try:
        await handle_call_service(
            policy,
            scripted.client,
            "http://ha.invalid",
            "test-token",
            "todo",
            "get_items",
            "todo.shopping_list",
        )
    finally:
        await scripted.client.aclose()

    assert call_count == 1
    assert len(scripted.requests) == 2


# CMD-01 follow-up (260923-j8h), continued: a non-2xx reply that is not
# asking for `?return_response` must never be retried, and every non-2xx
# reply must carry Home Assistant's own reason, not a bare status code.


@pytest.mark.parametrize(
    "status_code,response_kwargs",
    [
        (
            400,
            {
                "json": {
                    "message": "Service does not support responses. Remove return_response from request."
                }
            },
        ),
        (400, {"json": {"message": "Invalid JSON specified."}}),
        (400, {"text": "not json at all"}),
        (500, {"json": {"message": "Internal Server Error"}}),
    ],
    ids=["no-response-support", "bad-json-message", "non-json-body", "server-error"],
)
async def test_a_non_2xx_that_does_not_ask_for_responses_is_not_retried(status_code, response_kwargs):
    policy = Policy.from_config(None)
    scripted = _ScriptedHa([httpx.Response(status_code, **response_kwargs)])
    try:
        result = await handle_call_service(
            policy,
            scripted.client,
            "http://ha.invalid",
            "test-token",
            "light",
            "turn_on",
            "light.example_kitchen",
        )
    finally:
        await scripted.client.aclose()

    assert len(scripted.requests) == 1
    assert set(result) == {"error"}


async def test_error_result_carries_home_assistants_message():
    policy = Policy.from_config(None)
    scripted = _ScriptedHa([httpx.Response(400, json={"message": "Invalid JSON specified."})])
    try:
        result = await handle_call_service(
            policy,
            scripted.client,
            "http://ha.invalid",
            "test-token",
            "light",
            "turn_on",
            "light.example_kitchen",
        )
    finally:
        await scripted.client.aclose()

    assert result == {"error": "home assistant returned 400: Invalid JSON specified."}


async def test_error_result_cuts_a_non_json_body_to_200_characters():
    policy = Policy.from_config(None)
    scripted = _ScriptedHa([httpx.Response(502, text="x" * 500)])
    try:
        result = await handle_call_service(
            policy,
            scripted.client,
            "http://ha.invalid",
            "test-token",
            "light",
            "turn_on",
            "light.example_kitchen",
        )
    finally:
        await scripted.client.aclose()

    assert result == {"error": "home assistant returned 502: " + "x" * 200}


async def test_a_failed_retry_returns_the_second_error_and_does_not_retry_again():
    policy = Policy.from_config(None)
    scripted = _ScriptedHa(
        [
            httpx.Response(400, json={"message": _REQUIRES_RESPONSES}),
            httpx.Response(500, json={"message": "boom"}),
        ]
    )
    try:
        result = await handle_call_service(
            policy,
            scripted.client,
            "http://ha.invalid",
            "test-token",
            "todo",
            "get_items",
            "todo.shopping_list",
        )
    finally:
        await scripted.client.aclose()

    assert len(scripted.requests) == 2
    assert result == {"error": "home assistant returned 500: boom"}


async def test_plain_success_posts_once_without_return_response(fake_ha):
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
    assert "return_response" not in request.url.params
    assert set(result) == {"changed"}
    assert isinstance(result["changed"], list)


def test_handler_signatures_gain_no_passthrough_parameter():
    assert list(inspect.signature(handle_call_service).parameters) == [
        "policy",
        "client",
        "base_url",
        "token",
        "domain",
        "service",
        "entity_id",
        "area_id",
        "device_id",
        "label_id",
        "registry",
        "transition",
    ]
    assert list(inspect.signature(ha_module.ha_call_service).parameters) == [
        "domain",
        "service",
        "entity_id",
        "area_id",
        "device_id",
        "label_id",
        "transition",
    ]


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


# FLOW-03 (plan 05-03): `transition` is new plumbing through this function's
# own POST body, not a parameter that merely goes unused today. Asserted on
# the recorded request body, not the return value -- what matters is what
# Home Assistant actually received.


async def test_transition_reaches_the_request_body_on_a_light_call(fake_ha):
    policy = Policy.from_config(None)

    await handle_call_service(
        policy,
        fake_ha.client,
        "http://ha.invalid",
        "test-token",
        "light",
        "turn_on",
        "light.example_lamp",
        transition=5.0,
    )

    assert len(fake_ha.requests) == 1
    body = json.loads(fake_ha.requests[0].read())
    assert body["transition"] == 5.0
    assert body["entity_id"] == ["light.example_lamp"]


async def test_no_transition_given_leaves_the_request_body_unchanged(fake_ha):
    policy = Policy.from_config(None)

    await handle_call_service(
        policy,
        fake_ha.client,
        "http://ha.invalid",
        "test-token",
        "light",
        "turn_on",
        "light.example_lamp",
    )

    assert len(fake_ha.requests) == 1
    body = json.loads(fake_ha.requests[0].read())
    assert body == {"entity_id": ["light.example_lamp"]}
    assert "transition" not in body


async def test_transition_on_a_non_light_domain_is_refused_before_any_request(fake_ha):
    policy = Policy.from_config(None)

    with pytest.raises(Denied):
        await handle_call_service(
            policy,
            fake_ha.client,
            "http://ha.invalid",
            "test-token",
            "switch",
            "turn_on",
            "switch.example_fan",
            transition=5.0,
        )

    assert len(fake_ha.requests) == 0


async def test_negative_transition_is_refused_before_any_request(fake_ha):
    policy = Policy.from_config(None)

    with pytest.raises(Denied):
        await handle_call_service(
            policy,
            fake_ha.client,
            "http://ha.invalid",
            "test-token",
            "light",
            "turn_on",
            "light.example_lamp",
            transition=-1.0,
        )

    assert len(fake_ha.requests) == 0


# SAFE-09: plan 03-04 gives the child a second protocol to Home Assistant
# (mcp/atlas_mcp/registry.py's WebSocket connection). The child must hold
# the same one credential it held before -- HA_TOKEN -- and nothing this
# plan's own parent process might carry for an unrelated purpose. Named so
# a failing assertion says exactly which secret leaked.
_HOSTILE_PARENT_SECRETS = {
    "XAI_API_KEY": "sk-hostile-parent-secret-should-never-reach-the-ha-child",
    "DATABASE_URL": "postgresql+asyncpg://hostile:secret@db.invalid/hostile",
    "ATLAS_SECRET_KEY": "hostile-fernet-key-should-never-leak-into-the-ha-child",
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
        import atlas_mcp.ha  # import the real child module under this env
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
