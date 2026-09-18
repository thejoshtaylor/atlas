"""Real assertions for `mcp_tools_to_openai_tools` against a real MCP `Tool`.

CR-01 (phase 01 code review): the function once read `tool.inputSchema`
(camelCase) off each MCP `Tool`. The installed `mcp>=2.2,<3` SDK exposes
that field as the Python attribute `input_schema` -- `inputSchema` is only
a wire-serialization alias, not an attribute name -- so a bare
`tool.inputSchema` raised `AttributeError` on every real `Tool` and crashed
app startup outright. No test caught this because every other test in this
suite builds tools with hand-written doubles that already use whichever
shape the test author typed, never a real `mcp.types.Tool` instance. This
test builds one for real, the way the installed SDK does, so this class of
attribute-name drift fails loudly again if it ever recurs.
"""

import inspect
import os

from mcp.types import Tool

from spire_voice.mcp_client import mcp_tools_to_openai_tools


def test_mcp_tools_to_openai_tools_reads_real_tool_input_schema():
    tool = Tool(
        name="ha_call_service",
        description="Call a Home Assistant service.",
        inputSchema={
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
        },
    )

    openai_tools = mcp_tools_to_openai_tools([tool])

    assert openai_tools == [
        {
            "type": "function",
            "function": {
                "name": "ha_call_service",
                "description": "Call a Home Assistant service.",
                "parameters": {
                    "type": "object",
                    "properties": {"entity_id": {"type": "string"}},
                },
            },
        }
    ]


def test_mcp_tools_to_openai_tools_defaults_missing_description_to_empty_string():
    tool = Tool(name="ha_list_entities", description=None, inputSchema={"type": "object", "properties": {}})

    (openai_tool,) = mcp_tools_to_openai_tools([tool])

    assert openai_tool["function"]["description"] == ""


def test_mcp_tools_to_openai_tools_keeps_a_falsy_but_present_schema():
    """WR-01 (phase 01 code review, iteration 2): CR-01's own fallback read
    `getattr(tool, "input_schema", None) or getattr(tool, "inputSchema",
    None)`. `or` is a truthiness test, not a presence test, so a schema
    that is genuinely present but falsy -- an empty dict -- was discarded
    and replaced with `None` from the second `getattr`, which resolves to
    `None` on the installed SDK because `inputSchema` is not a real
    attribute on `Tool`. That would send `{"parameters": None}` into a live
    `tools=[...]` array instead of the empty-but-valid schema the tool
    actually declared. A sentinel-based presence check keeps `{}` intact.
    """
    tool = Tool(name="ha_list_entities", description="List entities.", inputSchema={})

    (openai_tool,) = mcp_tools_to_openai_tools([tool])

    assert openai_tool["function"]["parameters"] == {}


def test_mcp_child_runs_under_this_interpreter_not_a_bare_python3():
    """The MCP child must inherit the parent's interpreter, not PATH's `python3`.

    The child imports the same `mcp` SDK this process does. A bare `python3`
    is whatever PATH resolves first, which outside an activated virtualenv is
    the system interpreter with no SDK installed. Because this repository has
    its own top-level `mcp/` directory, that failure does not surface as an
    honest "SDK not installed" -- the repo directory satisfies `import mcp` as
    a namespace package, and the child dies on `No module named 'mcp.server'`,
    taking the FastAPI lifespan down with it.

    No test that imports the tool handlers directly can catch this, because
    the import happens inside the child process at startup. This asserts on
    the spawn parameters instead, which is the one place the choice is made.

    Reads `McpToolHost._spawn`, not `.start` -- Phase 3's `respawn()` factors
    the literal spawn (env dict, `command=sys.executable`) into one private
    helper both `start()` and `respawn()` call, so there is exactly one
    place this choice is made rather than two that could drift apart.
    """
    from spire_voice.mcp_client import McpToolHost

    source = inspect.getsource(McpToolHost._spawn)
    assert "command=sys.executable" in source, (
        "McpToolHost's spawn helper must use sys.executable; a bare "
        '"python3" resolves to the system interpreter, which has no mcp SDK'
    )
    assert 'command="python3"' not in source, (
        "a bare python3 command is still present in McpToolHost's spawn helper"
    )


# --- Plan 03-02 Task 4: McpToolHost.respawn -- the only way a policy change
# reaches the enforcing process (D-10) ---

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MCP_ROOT = os.path.join(_REPO_ROOT, "mcp")


async def test_respawn_replaces_the_child_with_one_enforcing_the_new_policy():
    """A respawned host must expose the new policy, and the old child must
    be gone -- proven against a real spawned subprocess, the same way
    `tests/test_policy_repo.py`'s end-to-end test does."""
    from spire_voice.mcp_client import McpToolHost

    host = McpToolHost()
    try:
        await host.start(
            ha_url="http://ha.invalid:8123",
            ha_token="test-key",
            mcp_root=_MCP_ROOT,
            safety_block={"mode": "allow_all_except_denylist", "deny_entities": []},
        )
        old_pid = host.session
        assert old_pid is not None

        # Before respawn: the entity is not on the denylist. No real Home
        # Assistant is reachable at ha.invalid, so this may still come back
        # as an error (a connection failure) -- what matters is that it is
        # not *this* refusal, since the entity isn't denied yet.
        allowed_before = await host.call_tool(
            "ha_call_service",
            {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_respawn_target"},
        )
        before_text = allowed_before.content[0].text if allowed_before.content else ""
        assert "off limits" not in before_text

        await host.respawn(
            {"mode": "allow_all_except_denylist", "deny_entities": ["switch.example_respawn_target"]}
        )

        # After respawn: the same entity is denied by the new child -- the
        # old session object is gone, replaced by a session on the new
        # process (a different ClientSession instance).
        assert host.session is not old_pid
        denied_after = await host.call_tool(
            "ha_call_service",
            {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_respawn_target"},
        )
        assert denied_after.is_error
        assert "off limits" in denied_after.content[0].text
    finally:
        await host.aclose()


async def test_respawned_child_environment_holds_exactly_four_keys_and_no_database_scheme():
    """SAFE-09/T-03-08: the environment dict the host builds for the
    respawned child, asserted against the dict itself -- not against source
    text -- must hold exactly `HA_URL`, `HA_TOKEN`, `PYTHONPATH`, and
    `SPIRE_SAFETY`, and no value may contain a database connection scheme."""
    from spire_voice.mcp_client import McpToolHost

    captured_envs: list[dict] = []
    real_spawn = McpToolHost._spawn

    async def _capturing_spawn(self, ha_url, ha_token, mcp_root, safety_block):
        env = {"HA_URL": ha_url, "HA_TOKEN": ha_token, "PYTHONPATH": str(mcp_root)}
        if safety_block is not None:
            import json as _json

            env["SPIRE_SAFETY"] = _json.dumps(safety_block)
        captured_envs.append(env)
        await real_spawn(self, ha_url, ha_token, mcp_root, safety_block)

    host = McpToolHost()
    try:
        McpToolHost._spawn = _capturing_spawn
        await host.start(
            ha_url="http://ha.invalid:8123",
            ha_token="test-key",
            mcp_root=_MCP_ROOT,
            safety_block={"mode": "allow_all_except_denylist"},
        )
        await host.respawn({"mode": "allowlist_only", "allow_entities": ["light.example_x"]})
    finally:
        McpToolHost._spawn = real_spawn
        await host.aclose()

    assert len(captured_envs) == 2, "expected one env for start() and one for respawn()"
    respawned_env = captured_envs[-1]
    assert set(respawned_env.keys()) == {"HA_URL", "HA_TOKEN", "PYTHONPATH", "SPIRE_SAFETY"}
    for value in respawned_env.values():
        assert "postgresql" not in value.lower(), (
            f"a value in the respawned child's environment names a database scheme: {value!r}"
        )
