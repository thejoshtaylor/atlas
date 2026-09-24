"""google accounts, calendars, oauth state, style, and pending actions

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-24

Phase 9 (Google accounts: multi-account Calendar and Gmail): every table
this whole phase needs, so no later plan in this phase adds a migration
(09-01-PLAN.md's own instruction). Six tables: `google_oauth_client` (the
one deployer-supplied OAuth client, D-01), `google_oauth_states` (in-flight
OAuth `state` values, D-02), `google_accounts` (a linked account and its
encrypted refresh token, D-01/D-03), `google_calendars` (per-account
discovered calendars, off until enabled, D-05), `google_account_style`
(learned writing style and signature, D-19/D-22), and `pending_actions`
(a code-held confirm/cancel action, D-06..D-09).

Like `0012_wake_events.py`, this migration seeds nothing: every table here
is new, empty, operator-populated state, not a cutover with legacy data to
carry.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0015"
down_revision: Union[str, Sequence[str], None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PENDING_ACTIONS_INDEX = "ix_pending_actions_expires_at"
_GOOGLE_CALENDARS_UNIQUE = "uq_google_calendars_account_calendar"


def upgrade() -> None:
    op.create_table(
        "google_oauth_client",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("client_secret_ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("key_version", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column(
            "updated_by_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True
        ),
    )

    op.create_table(
        "google_oauth_states",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("state_hash", sa.Text(), nullable=False, unique=True),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("redirect_uri", sa.Text(), nullable=False),
        sa.Column(
            "created_by_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("used_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "google_accounts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("label", sa.Text(), nullable=False, unique=True),
        sa.Column("email", sa.Text(), nullable=False, unique=True),
        sa.Column("refresh_token_ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("key_version", sa.Integer(), nullable=False),
        sa.Column("granted_scopes", sa.Text(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.Text(), nullable=False, server_default="ok"),
        sa.Column("status_detail", sa.Text(), nullable=True),
        sa.Column("status_at", sa.DateTime(), nullable=True),
        sa.Column("refresh_token_expires_at", sa.DateTime(), nullable=True),
        sa.Column("linked_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column(
            "linked_by_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True
        ),
    )

    op.create_table(
        "google_calendars",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("google_accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("google_calendar_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("can_write", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("access", sa.Text(), nullable=False, server_default="off"),
        sa.Column("discovered_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "account_id", "google_calendar_id", name=_GOOGLE_CALENDARS_UNIQUE
        ),
    )

    op.create_table(
        "google_account_style",
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("google_accounts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("profile", sa.Text(), nullable=False, server_default=""),
        sa.Column("samples", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("signature_html", sa.Text(), nullable=True),
        sa.Column("signature_text", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="not_learned"),
        sa.Column("status_detail", sa.Text(), nullable=True),
        sa.Column("messages_scanned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("learned_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "pending_actions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column("readback", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("result_detail", sa.Text(), nullable=True),
    )
    op.create_index(_PENDING_ACTIONS_INDEX, "pending_actions", ["expires_at"])


def downgrade() -> None:
    op.drop_index(_PENDING_ACTIONS_INDEX, table_name="pending_actions")
    op.drop_table("pending_actions")
    op.drop_table("google_account_style")
    op.drop_table("google_calendars")
    op.drop_table("google_accounts")
    op.drop_table("google_oauth_states")
    op.drop_table("google_oauth_client")
