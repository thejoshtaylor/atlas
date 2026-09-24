"""Build the JSON-shaped `safety:` block `Policy.from_config` already
expects, from a `Policy` object -- the other half of the round trip that
proves the database-backed and config-backed constructors agree.

This module does not serialize a `Policy` object across the MCP child's
process boundary. It serializes the same raw block `from_config` has
always accepted, so there is exactly one parser, `atlas_mcp.safety`, on
both sides of that boundary -- the same rule `config.py`'s own module
docstring already states for the config-file path. `safety_block_from_policy`
is that rule's database-path counterpart: `lifespan` calls this on the
`Policy` a `PolicyRepository` returns, then hands the resulting dict to
`McpToolHost.start`/`respawn` exactly where the raw `safety:` config block
used to go.

The round trip `Policy.from_config(safety_block_from_policy(p)) == p` either
holds or fails a test (`tests/test_policy_repo.py`) -- that equality is what
proves a database-derived policy and a config-derived one can never
silently disagree about what is denied.
"""

from __future__ import annotations

from atlas_mcp.safety import Policy


def safety_block_from_policy(policy: Policy) -> dict:
    """The `safety:`-shaped dict `Policy.from_config` would rebuild `policy`
    from. All seven fields are written explicitly, including the two
    code-owned defaults (`deny_domains`/`deny_services`) -- an explicit
    field beats one silently inherited from `from_config`'s own defaults,
    because a future default change must not retroactively alter what an
    already-snapshotted policy meant.
    """
    return {
        "mode": policy.mode,
        "deny_entities": sorted(policy.deny_entities),
        "deny_patterns": list(policy.deny_patterns),
        "deny_domains": sorted(policy.deny_domains),
        "deny_services": sorted(policy.deny_services),
        "allow_entities": sorted(policy.allow_entities),
        "allow_patterns": list(policy.allow_patterns),
    }
