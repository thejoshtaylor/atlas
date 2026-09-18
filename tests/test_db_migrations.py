"""The migration path itself, and the seed that carries an operator's real
denylist across the config-to-database cut (DEP-04, SAFE-05, D-11).

DEP-04 promises "an upgrade keeps existing data" -- the one property this
project has never had to prove before, because Phase 3 is the first phase
with a database at all. A migration that is not actually idempotent (running
it twice against an already-migrated schema errors, or silently duplicates
rows) would fail that promise the first time an operator restarts after this
phase ships. D-11 promises something narrower and higher-stakes: the
operator's real, already-deployed `safety:` block -- eight entries on this
specific house, entered before this phase existed -- must be seeded into the
database by the first migration, not discarded. A migration that runs clean
but seeds an empty policy is a silent safety regression: the process boots,
the denylist editor loads, and nothing looks wrong until the entity that used
to be denied is not.

No test here yet catches either failure, because no migration exists yet.
Both tests are marked `integration`: they need a real Postgres
(`SPIRE_TEST_DATABASE_URL`, `scripts/dev-postgres.sh`), matching this
phase's D-04 -- the suite otherwise runs with no database reachable at all.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_migrations_run_from_empty_and_are_idempotent():
    """Running every migration twice against the same database, from empty,
    must succeed both times with no duplicated row and no error -- plan
    03-02 fills this in against a real Postgres."""
    raise AssertionError("plan 03-02 fills this in (DEP-04: an upgrade keeps existing data)")


@pytest.mark.integration
def test_seeded_policy_matches_the_config_block_it_came_from():
    """The first migration's seed step must carry an equivalent-shaped
    `safety:` block into the database rows `Policy.from_db_rows` reads --
    plan 03-02 fills this in against a real Postgres."""
    raise AssertionError(
        "plan 03-02 fills this in (SAFE-05, D-11: the safety: block is seeded, not discarded)"
    )
