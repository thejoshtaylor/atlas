"""speakers and speaker_embeddings tables -- household members and their
enrollment embeddings (D-01, D-03, Phase 11)

Revision ID: 0018
Revises: 0017
Create Date: 2026-09-28

Plan 11-03: `speakers` holds one row per household member -- a display
name and an optional link to a `users` row (D-01). A member needs no
webapp account: children and guests are ordinary members with no login at
all. `speaker_embeddings` holds one row per enrollment phrase per model id
(D-03), so a later model change can re-embed a member's already-captured
audio without a new recording.

Like `0012_wake_events.py`, this migration seeds nothing: this is new,
empty state. But unlike every other resource this project has migrated so
far -- `edge_devices.revoked_at`, `users.disabled_at` -- these two tables
break the "disable, never delete" discipline on purpose. A member's
embedding is biometric data, and deleting a member is a real, hard
`DELETE` of the member row and every one of its embedding rows (D-03),
which the `ondelete="CASCADE"` on `speaker_embeddings.speaker_id` carries
out for the embeddings automatically. Neither table has a `disabled_at`,
`revoked_at`, or `deleted_at` column, and none should ever be added.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0018"
down_revision: Union[str, Sequence[str], None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_UQ_SPEAKERS_DISPLAY_NAME = "uq_speakers_display_name"
_UQ_SPEAKER_EMBEDDINGS_SPEAKER_PHRASE_MODEL = "uq_speaker_embeddings_speaker_phrase_model"


def upgrade() -> None:
    op.create_table(
        "speakers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column(
            "linked_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("display_name", name=_UQ_SPEAKERS_DISPLAY_NAME),
    )
    op.create_table(
        "speaker_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "speaker_id",
            sa.Integer(),
            sa.ForeignKey("speakers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("phrase_index", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("vector", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "speaker_id",
            "phrase_index",
            "model_id",
            name=_UQ_SPEAKER_EMBEDDINGS_SPEAKER_PHRASE_MODEL,
        ),
    )


def downgrade() -> None:
    op.drop_table("speaker_embeddings")
    op.drop_table("speakers")
