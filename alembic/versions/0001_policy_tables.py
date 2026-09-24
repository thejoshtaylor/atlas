"""policy tables

Revision ID: 0001
Revises:
Create Date: 2026-09-17

Creates the three tables SAFE-05's database-backed policy needs
(`safety_policy`, `policy_rules`, `audit_log`) and seeds the first row of
each from the operator's own, already-deployed configuration (D-11).

This file holds no entity id, no pattern, and no house name of its own. It
reads them, if any exist, at run time from a file that is not part of this
repository: the one `ATLAS_CONFIG` names, the same environment variable and
the same default (`config/config.example.yaml`) `app.py` already reads. A
reader checking SAFE-05 should start here and find nothing to check.

The seed step is the load-bearing half of this migration. Rejecting the
`safety:` config key (`atlas.config.Config.from_config`, this same
plan) without first carrying an operator's real denylist into the database
would leave the house with an empty denylist between those two events --
the exact window this project's prior power-cut incident is about (D-11,
`03-CONTEXT.md`). So: read the file through `Policy.from_config` -- one
parser, the same one `config.py` already uses -- and write in what it
returns as the starting policy, before the key is ever rejected.

Three distinguishable outcomes, logged differently on purpose (an operator
reading these logs after an upgrade needs to tell "there was nothing to
seed" apart from "there was something, and it came out empty"):

1. No `safety:` key in the file at all: the default policy
   (`allow_all_except_denylist`, no rules) is seeded. A real, ordinary
   choice -- most deployments start here.
2. A `safety:` key is present and carries entity ids and/or patterns: they
   are seeded verbatim.
3. A `safety:` key is present but names no entities and no patterns (an
   explicitly empty block): the policy is seeded with zero rules, same as
   case 1's row count, but logged as "found and empty," not "absent" --
   these are different operator states and only the log line tells them
   apart.

When the file itself cannot be opened or read, this migration raises rather
than seeding an empty policy: a migration that silently seeds nothing is
the very failure this decision exists to prevent.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op

from atlas_mcp.safety import Policy
from atlas.config import expand_env

logger = logging.getLogger("alembic.policy_seed")

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    _create_tables()
    _seed_policy()


def _create_tables() -> None:
    op.create_table(
        "safety_policy",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("updated_by_user_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "policy_rules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def _read_safety_block(config_path: str) -> dict | None:
    """Read `config_path` (the file `ATLAS_CONFIG` names) and return its
    `safety:` block, or `None` when the file has no such key.

    Raises when the file cannot be opened or read at all -- never seeds an
    empty policy because a path was wrong or a permission was missing.
    """
    try:
        with open(config_path, encoding="utf-8") as fh:
            raw_text = fh.read()
    except OSError as exc:
        raise RuntimeError(
            f"the policy seed migration could not read {config_path!r} (named by "
            "ATLAS_CONFIG, or its default) to seed the safety policy -- refusing to seed "
            "an empty denylist silently. Either make the file readable at that path, or "
            "point ATLAS_CONFIG at the file that holds the safety: block to carry forward."
        ) from exc

    raw = yaml.safe_load(expand_env(raw_text)) or {}
    return raw.get("safety")


def _seed_policy() -> None:
    config_path = os.environ.get("ATLAS_CONFIG", "config/config.example.yaml")
    safety_block = _read_safety_block(config_path)

    if safety_block is None:
        logger.info(
            "policy seed: no safety: key found in %s -- seeding the default policy "
            "(allow_all_except_denylist, no rules). This is 'nothing to migrate,' not "
            "'migrated nothing' -- a safety: block that is present but empty logs "
            "differently, below.",
            config_path,
        )
    else:
        logger.info("policy seed: found a safety: key in %s -- seeding it verbatim.", config_path)

    # One parser -- the same Policy.from_config every other reader of a
    # safety: block already uses -- so the seeded rows can never encode a
    # different denylist than the file described.
    policy = Policy.from_config(safety_block)

    now = datetime.now(timezone.utc)

    conn = op.get_bind()
    metadata = sa.MetaData()
    safety_policy = sa.Table("safety_policy", metadata, autoload_with=conn)
    policy_rules = sa.Table("policy_rules", metadata, autoload_with=conn)

    conn.execute(
        safety_policy.insert().values(
            id=1, mode=policy.mode, updated_at=now, updated_by_user_id=None
        )
    )

    rule_rows: list[dict] = []
    for kind, values in (
        ("deny_entity", sorted(policy.deny_entities)),
        ("deny_pattern", list(policy.deny_patterns)),
        ("allow_entity", sorted(policy.allow_entities)),
        ("allow_pattern", list(policy.allow_patterns)),
    ):
        for value in values:
            rule_rows.append(
                {
                    "kind": kind,
                    "value": value,
                    "note": None,
                    "created_at": now,
                    "created_by_user_id": None,
                }
            )

    if rule_rows:
        conn.execute(policy_rules.insert(), rule_rows)

    if safety_block is not None and not rule_rows:
        # Case 3 from the module docstring: a safety: key was present but
        # named no entities and no patterns. Same seeded row count as "no
        # key at all," a different operator state -- logged as its own
        # line so the two are never confused after the fact.
        logger.info(
            "policy seed: the safety: key in %s named no entities and no patterns -- "
            "seeded with zero rules, distinct from finding no key at all.",
            config_path,
        )

    logger.info(
        "policy seed: mode=%s, %d rule(s) seeded",
        policy.mode,
        len(rule_rows),
    )


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("policy_rules")
    op.drop_table("safety_policy")
