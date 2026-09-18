"""The database-backed `Policy` must be the same value a config-backed one
would have been, and the snapshot handed to the MCP child must survive that
round trip unchanged (SAFE-05).

`mcp/spire_mcp/safety.py`'s `Policy.from_config` is the existing, tested
constructor; this phase adds a sibling, `Policy.from_db_rows`, built from
whatever a policy repository fetches. Nothing today proves the two agree --
a repository that fetches rows in the wrong order, coerces a type
differently, or drops a field `from_config` would have kept, would build a
`Policy` that is silently a different denylist than the one an equivalent
config block describes, with no test catching the divergence before an
operator's real house does. The second test guards the boundary one step
further out: what actually crosses into the MCP child over `SPIRE_SAFETY`
is a JSON-serialized snapshot, parsed a second time on the other side
(`tests/test_policy_reaches_the_child.py` already proves the existing
config-to-child leg of this; this test proves the database-to-child leg).
"""

from __future__ import annotations


def test_policy_rows_build_the_same_policy_as_an_equivalent_config_block():
    """`Policy.from_db_rows(...)`, given rows equivalent to a `safety:`
    block, must build a `Policy` equal to `Policy.from_config` on that same
    block -- plan 03-02 fills this in."""
    raise AssertionError(
        "plan 03-02 fills this in (SAFE-05: Policy.from_db_rows agrees with Policy.from_config)"
    )


def test_snapshot_round_trips_through_the_child_parser():
    """A database-derived policy snapshot, serialized the way `app.py` hands
    it to `McpToolHost` and parsed the way `ha.py` reads `SPIRE_SAFETY`,
    must come out the same `Policy` it started as -- plan 03-02 fills this
    in."""
    raise AssertionError(
        "plan 03-02 fills this in (SAFE-05: the DB-to-child snapshot round-trips)"
    )
