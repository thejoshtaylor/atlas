"""macro tables

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-18

Creates the three tables MACRO-03's database-backed macros need (`macros`,
`macro_actions`, `macro_aliases`) and seeds them from the operator's own,
already-deployed configuration (D-09) -- the same seed-then-reject move
`0001_policy_tables.py` made for the safety policy, generalized rather than
copied (D-16): `app.py`'s two-stage boot now drives one sequence over a
collection of legacy keys, and this migration is the second entry in that
collection, not a second copy of the first.

This file holds no phrase, no entity id, and no house name of its own. It
reads them, if any exist, at run time from a file that is not part of this
repository: the one `ATLAS_CONFIG` names, the same environment variable and
the same default (`config/config.example.yaml`) `app.py` already reads. A
reader checking D-09 should start here and find nothing to check.

The seed step is the load-bearing half of this migration, for the exact
reason `0001`'s own docstring gives and Phase 3's code review (CR-01) once
paid for getting backwards: rejecting the `macros:` config key
(`atlas.config.Config.from_config`) without first carrying an
operator's real macros into the database would leave the house with none
of them between those two events. So: read the file through
`MacroConfig.from_config` -- one parser, the same one `config.py` already
uses for the file path -- and write in what it returns, before the key is
ever rejected.

The operator's own instruction on this plan, not merely a Phase 3 echo:
the cross-macro collision check (`_check_macros_do_not_collide`) that
guards the config path today must run here too, against the parsed seed
set, before a single row is inserted. A macro reaching the database that
the file parser would have rejected is a macro that can shadow another one
at runtime with nothing having ever validated the pair -- the exact defect
this check exists to prevent, reached through a new door once macros can
also arrive from a browser (plan 04-06). `Macro.normalized_keys`
(`atlas.db.repository`) is what lets this same function keep
guarding the database path once a route calls it there, with no change to
the function itself.

Two distinguishable outcomes, logged differently on purpose (an operator
reading these logs after an upgrade needs to tell "there was nothing to
seed" apart from "there was something, and it came out empty" -- unlike
`safety:`, an explicitly empty `macros:` list and an absent key produce the
same zero rows either way, so there is no third case here the way `0001`
has one for a present-but-empty dict-shaped block):

1. No `macros:` key in the file at all: nothing is seeded. A real,
   ordinary choice -- most deployments start here.
2. A `macros:` key is present: every entry is parsed and seeded, in
   written order, actions in written order.

When the file itself cannot be opened or read, this migration raises
rather than seeding nothing silently -- indistinguishable, from an
operator's later reading, from "there was nothing to seed," which is
exactly the failure `0001`'s own identical rule exists to prevent.

`downgrade()` is real, not a stub: `0004_setup_state.py` set the precedent
that a migration carrying an operator's real data forward earns a real
reverse path, and this one -- the migration that is the seed source for a
second legacy key -- is exactly that case (the operator's own instruction
on this plan).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op

from atlas.config import MacroConfig, _check_macros_do_not_collide, expand_env

logger = logging.getLogger("alembic.macro_seed")

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, Sequence[str], None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    _create_tables()
    _seed_macros()


def _create_tables() -> None:
    op.create_table(
        "macros",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("phrase", sa.Text(), nullable=False),
        sa.Column("reply", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], name="fk_macros_created_by_user_id"
        ),
    )
    op.create_table(
        "macro_actions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("macro_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("tool", sa.Text(), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["macro_id"],
            ["macros.id"],
            name="fk_macro_actions_macro_id",
            ondelete="CASCADE",
        ),
    )
    op.create_table(
        "macro_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("macro_id", sa.Integer(), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["macro_id"],
            ["macros.id"],
            name="fk_macro_aliases_macro_id",
            ondelete="CASCADE",
        ),
    )


def _read_macros_block(config_path: str) -> list | None:
    """Read `config_path` (the file `ATLAS_CONFIG` names) and return its
    `macros:` list, or `None` when the file has no such key.

    Raises when the file cannot be opened or read at all -- never seeds
    nothing because a path was wrong or a permission was missing, matching
    `0001_policy_tables.py::_read_safety_block`'s identical rule.
    """
    try:
        with open(config_path, encoding="utf-8") as fh:
            raw_text = fh.read()
    except OSError as exc:
        raise RuntimeError(
            f"the macro seed migration could not read {config_path!r} (named by "
            "ATLAS_CONFIG, or its default) to seed macros -- refusing to seed nothing "
            "silently. Either make the file readable at that path, or point ATLAS_CONFIG "
            "at the file that holds the macros: block to carry forward."
        ) from exc

    raw = yaml.safe_load(expand_env(raw_text)) or {}
    return raw.get("macros")


def _seed_macros() -> None:
    config_path = os.environ.get("ATLAS_CONFIG", "config/config.example.yaml")
    macros_raw = _read_macros_block(config_path)

    if macros_raw is None:
        logger.info(
            "macro seed: no macros: key found in %s -- seeding nothing. This is "
            "'nothing to migrate,' not an error.",
            config_path,
        )
        return

    logger.info(
        "macro seed: found a macros: key in %s with %d entr%s -- seeding.",
        config_path,
        len(macros_raw),
        "y" if len(macros_raw) == 1 else "ies",
    )

    # One parser -- the same MacroConfig.from_config every other reader of
    # a macros: block already uses -- so a seeded macro can never encode a
    # phrase, alias, reply, or action list the file parser would have
    # built differently. The collision check runs against the whole
    # parsed set, before a single row is inserted, for the same reason
    # Config.from_config already runs it on the file path (the operator's
    # own instruction on this plan): a macro that would have collided is
    # a macro that must never reach the database through this door either.
    macros = tuple(MacroConfig.from_config(m) for m in macros_raw)
    _check_macros_do_not_collide(macros)

    now = datetime.now(timezone.utc)

    conn = op.get_bind()
    metadata = sa.MetaData()
    macros_table = sa.Table("macros", metadata, autoload_with=conn)
    macro_actions_table = sa.Table("macro_actions", metadata, autoload_with=conn)
    macro_aliases_table = sa.Table("macro_aliases", metadata, autoload_with=conn)

    total_actions = 0
    total_aliases = 0
    for macro in macros:
        result = conn.execute(
            macros_table.insert().values(
                phrase=macro.phrase,
                reply=macro.reply,
                created_at=now,
                updated_at=now,
                created_by_user_id=None,
            )
        )
        macro_id = result.inserted_primary_key[0]

        if macro.aliases:
            conn.execute(
                macro_aliases_table.insert(),
                [{"macro_id": macro_id, "alias": alias} for alias in macro.aliases],
            )
            total_aliases += len(macro.aliases)

        conn.execute(
            macro_actions_table.insert(),
            [
                {
                    "macro_id": macro_id,
                    "position": position,
                    "tool": action.tool,
                    "arguments": action.arguments,
                }
                for position, action in enumerate(macro.actions)
            ],
        )
        total_actions += len(macro.actions)

    logger.info(
        "macro seed: %d macro(s), %d action(s), %d alias(es) seeded",
        len(macros),
        total_actions,
        total_aliases,
    )


def downgrade() -> None:
    op.drop_table("macro_aliases")
    op.drop_table("macro_actions")
    op.drop_table("macros")
