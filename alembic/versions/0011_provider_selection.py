"""provider_selections table, seeded for all three slots

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-19

D-01, D-03, PROV-01: the three provider choices (speech to text, text to
speech, language model) move into the database, beside the credentials
Phase 3 already put there. `config.yaml`'s `stt:`/`brain:`/`tts:` blocks
stay -- they still carry url, model, voice and codec -- only the *choice
of implementation* moves here.

All three slots are seeded unconditionally with `"xai"`, the one entry
every registry holds this plan (`atlas.providers.registry`) and the
only implementation `config.example.yaml` has ever configured for any of
them. Unlike `0008_plugin_tables.py`'s seed-then-reject shape, there is no
legacy config key naming a provider *choice* to migrate away from -- this
is new state, not a cutover -- so this migration seeds and nothing more.

All three slots are seeded now, even though this plan only reads the
speech-to-text one: plan 07-02 adds a registry entry and a reader for the
other two, never a second migration.

`downgrade()` drops the table outright -- an admin's provider choice is
recoverable by re-selecting it in the browser after a downgrade, unlike
the real-credential/real-policy data `0005`/`0009` carry forward.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: Union[str, Sequence[str], None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CONSTRAINT = "uq_provider_selections_slot"

# The one entry every registry holds this plan (`registry.py`'s
# `STT_REGISTRY`/`TTS_REGISTRY`/`BRAIN_REGISTRY`) -- the same name
# `config.example.yaml`'s `stt:`/`brain:`/`tts:` blocks have always pointed
# at, for all three slots.
_SEEDED_PROVIDER_NAME = "xai"
_SLOTS = ("stt", "tts", "brain")

provider_selections_table = sa.table(
    "provider_selections",
    sa.column("slot", sa.Text),
    sa.column("provider_name", sa.Text),
    sa.column("options", sa.JSON),
    sa.column("updated_at", sa.DateTime),
    sa.column("updated_by_user_id", sa.Integer),
)


def upgrade() -> None:
    op.create_table(
        "provider_selections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("slot", sa.Text(), nullable=False),
        sa.Column("provider_name", sa.Text(), nullable=False),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column(
            "updated_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
    )
    op.create_unique_constraint(_CONSTRAINT, "provider_selections", ["slot"])

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    op.bulk_insert(
        provider_selections_table,
        [
            {
                "slot": slot,
                "provider_name": _SEEDED_PROVIDER_NAME,
                "options": {},
                "updated_at": now,
                "updated_by_user_id": None,
            }
            for slot in _SLOTS
        ],
    )


def downgrade() -> None:
    op.drop_table("provider_selections")
