"""The database-backed `Policy` must be the same value a config-backed one
would have been, and the snapshot handed to the MCP child must survive that
round trip unchanged (SAFE-05).

`mcp/atlas_mcp/safety.py`'s `Policy.from_config` is the existing, tested
constructor; this phase adds a sibling, `Policy.from_db_rows`, built from
whatever a policy repository fetches. Nothing today proves the two agree --
a repository that fetches rows in the wrong order, coerces a type
differently, or drops a field `from_config` would have kept, would build a
`Policy` that is silently a different denylist than the one an equivalent
config block describes, with no test catching the divergence before an
operator's real house does. The second test guards the boundary one step
further out: what actually crosses into the MCP child over `ATLAS_SAFETY`
is a JSON-serialized snapshot, parsed a second time on the other side
(`tests/test_policy_reaches_the_child.py` already proves the existing
config-to-child leg of this; this test proves the database-to-child leg).
"""

from __future__ import annotations

from pathlib import Path

from atlas_mcp.safety import Policy
from atlas.mcp_client import McpToolHost
from atlas.policy_snapshot import safety_block_from_policy

REPO = Path(__file__).resolve().parents[1]
MCP_ROOT = REPO / "mcp"


def test_policy_rows_build_the_same_policy_as_an_equivalent_config_block():
    """`Policy.from_db_rows(...)`, given rows equivalent to a `safety:`
    block, must build a `Policy` equal to `Policy.from_config` on that same
    block, across both modes and across every rule kind."""
    block = {
        "mode": "allow_all_except_denylist",
        "deny_entities": ["switch.example_server_socket"],
        "deny_patterns": ["switch.example_camera_*"],
        "allow_entities": ["light.example_lamp"],
        "allow_patterns": ["light.example_*"],
    }
    from_config = Policy.from_config(block)
    from_rows = Policy.from_db_rows(
        mode=block["mode"],
        deny_entities=block["deny_entities"],
        deny_patterns=block["deny_patterns"],
        allow_entities=block["allow_entities"],
        allow_patterns=block["allow_patterns"],
    )
    assert from_rows == from_config

    strict_block = {
        "mode": "allowlist_only",
        "allow_entities": ["light.example_lamp"],
        "allow_patterns": ["switch.example_*"],
    }
    strict_from_config = Policy.from_config(strict_block)
    strict_from_rows = Policy.from_db_rows(
        mode=strict_block["mode"],
        deny_entities=(),
        deny_patterns=(),
        allow_entities=strict_block["allow_entities"],
        allow_patterns=strict_block["allow_patterns"],
    )
    assert strict_from_rows == strict_from_config


def test_snapshot_round_trips_through_the_child_parser():
    """`Policy.from_config(safety_block_from_policy(p)) == p` -- the round
    trip that proves a database-derived policy and a config-derived one can
    never silently disagree about what is denied, in both modes."""
    for mode in ("allow_all_except_denylist", "allowlist_only"):
        policy = Policy.from_db_rows(
            mode=mode,
            deny_entities=["switch.example_server_socket"],
            deny_patterns=["switch.example_camera_*"],
            allow_entities=["light.example_lamp"],
            allow_patterns=["light.example_*"],
        )
        block = safety_block_from_policy(policy)
        assert Policy.from_config(block) == policy


async def test_a_database_only_denied_entity_is_refused_by_the_real_child(fake_policy_repository):
    """The end-to-end slice, and the point of this file: a denied entity
    that exists only as a `FakePolicyRepository` row -- not in any source
    file -- travels through `safety_block_from_policy`, into a real
    `McpToolHost.start(..., safety_block=block)`-spawned child process over
    `ATLAS_SAFETY`, and a real `ha_call_service` MCP tool call against that
    entity id comes back refused. No Home Assistant server is needed and
    none is started: `allow_call` raises inside the child before
    `handle_call_service` ever constructs an httpx request at all.
    """
    # Invented entity id, in the switch.example_* naming convention
    # `tests/test_repo_hygiene.py` enforces -- found in no file under
    # src/, mcp/, or config/.
    denied_entity = "switch.example_database_only_denied_entity"

    repo = fake_policy_repository(deny_entities=[denied_entity])
    policy = await repo.load_policy()
    block = safety_block_from_policy(policy)

    # "test-key" is this repository's one allowlisted credential-shaped
    # placeholder (`tests/test_repo_hygiene.py::_ALLOWED_CREDENTIAL_VALUES`)
    # -- a `ha_token="..."` keyword argument is exactly the `token\s*[:=]\s*
    # ['"]` shape that hygiene scan matches, so this reuses the already-
    # allowed value rather than adding a second one.
    fake_ha_token = "test-key"

    host = McpToolHost()
    try:
        await host.start(
            ha_url="http://ha.invalid:8123",
            ha_token=fake_ha_token,
            mcp_root=MCP_ROOT,
            safety_block=block,
        )
        result = await host.call_tool(
            "ha_call_service",
            {"domain": "switch", "service": "turn_off", "entity_id": denied_entity},
        )
    finally:
        await host.aclose()

    assert result.is_error, (
        "a database-only denied entity was not refused by the real child process"
    )
    assert "off limits" in result.content[0].text, (
        f"the refusal text did not match safety.py's own wording: {result.content[0].text!r}"
    )
