"""accounts

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-18

Creates the three tables WEB-01/WEB-04/WEB-05's account system needs
(`users`, `invites`, `refresh_tokens`) and adds the two foreign keys
`0001_policy_tables.py` left as plain nullable columns because `users` did
not exist yet: `safety_policy.updated_by_user_id` and
`policy_rules.created_by_user_id`.

Unlike `0001`, this migration seeds nothing -- there is no operator
account to carry forward from a config file, because no account system
existed before this phase. The `users` table starts empty by design; D-08's
create-admin route is the only writer of its first row, in the browser, at
first boot.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    _create_accounts_tables()
    _add_policy_foreign_keys()


def _create_accounts_tables() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("disabled_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_table(
        "invites",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=False),
        sa.Column("accepted_at", sa.DateTime(), nullable=True),
        sa.Column("accepted_by_user_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_invites_token_hash"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], name="fk_invites_created_by_user_id"
        ),
        sa.ForeignKeyConstraint(
            ["accepted_by_user_id"], ["users.id"], name="fk_invites_accepted_by_user_id"
        ),
    )
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("issued_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("rotated_to_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_refresh_tokens_user_id"),
        sa.ForeignKeyConstraint(
            ["rotated_to_id"], ["refresh_tokens.id"], name="fk_refresh_tokens_rotated_to_id"
        ),
    )


def _add_policy_foreign_keys() -> None:
    """`0001_policy_tables.py` left these two columns as plain nullable
    integers because `users` did not exist yet -- now that it does, this
    migration is the one that resolves them into real foreign keys, per
    this plan's own action text."""
    op.create_foreign_key(
        "fk_safety_policy_updated_by_user_id",
        "safety_policy",
        "users",
        ["updated_by_user_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_policy_rules_created_by_user_id",
        "policy_rules",
        "users",
        ["created_by_user_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_policy_rules_created_by_user_id", "policy_rules", type_="foreignkey")
    op.drop_constraint(
        "fk_safety_policy_updated_by_user_id", "safety_policy", type_="foreignkey"
    )
    op.drop_table("refresh_tokens")
    op.drop_table("invites")
    op.drop_table("users")
