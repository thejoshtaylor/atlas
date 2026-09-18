"""Editing the policy in the webapp must reach the process that enforces it,
and changing its mode must leave a record of who did it and when (SAFE-06,
SAFE-07).

`tests/test_policy_reaches_the_child.py` already proves a policy handed to
`McpToolHost` at spawn reaches the child -- it does not prove a policy
*changed after spawn* reaches a running child, because Phase 1 never
respawned the child at all. An operator who adds a denylist rule in the
webapp and sees the save succeed, while the MCP child that actually
evaluates every call keeps running on the policy it was spawned with, is a
silent SAFE-06/SAFE-07 regression: the editor works, the enforcement does
not, and nothing here fails loudly enough to notice before the next
restart. The second test guards the accountability side of D-13's two-modes
decision: switching between `allow_all_except_denylist` and `allowlist_only`
is exactly the kind of admin action this project's own prior incident makes
worth a permanent, named record.
"""

from __future__ import annotations


def test_adding_a_rule_respawns_the_tool_child_with_the_new_policy():
    """Saving a new denylist rule through the policy route must respawn
    `McpToolHost` with a policy snapshot that includes it -- a stale,
    already-spawned child evaluating the old policy is the regression this
    guards -- plan 03-07 fills this in."""
    raise AssertionError(
        "plan 03-07 fills this in (SAFE-06, SAFE-07: an edited policy reaches the running child)"
    )


def test_switching_mode_writes_an_audit_row_naming_who_and_when():
    """Changing `mode` between `allow_all_except_denylist` and
    `allowlist_only` must write an audit row naming the admin and the
    timestamp (D-13) -- plan 03-07 fills this in."""
    raise AssertionError(
        "plan 03-07 fills this in (SAFE-06, D-13: a mode change is a named, timestamped audit row)"
    )
