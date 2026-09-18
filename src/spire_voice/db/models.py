"""The SQLAlchemy declarative base every Phase 3 table inherits from, plus
the three tables plan 03-02 adds: the safety policy the code enforces.

`alembic/env.py` reads `Base.metadata` as its `target_metadata` -- a model
class that does not subclass `Base`, or a model module nobody imports, is a
model `alembic revision --autogenerate` cannot see. Plan 03-01 added only
the base itself; later plans in this phase (accounts, invites, provider
credentials, settings) add their own tables here.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    """The one declarative base for every ORM model in this project.

    A model defined against a second, unrelated `DeclarativeBase` would be
    invisible to `alembic/env.py`'s `target_metadata` and to the migration
    it drives -- there is exactly one base, imported from here, for every
    table this project ever adds.
    """


class SafetyPolicyRow(Base):
    """A single-row table (`id` always 1) carrying the one active mode.

    One `mode` column, not two independent flags, for the same reason
    `mcp/spire_mcp/safety.py`'s `Policy.mode` is a single `Literal`: a state
    that is neither `allow_all_except_denylist` nor `allowlist_only` must
    not be representable (D-13, SAFE-06). `updated_by_user_id` has no
    foreign key yet -- plan 03-05 adds the `users` table this column will
    point at; it stays a plain nullable column until then.
    """

    __tablename__ = "safety_policy"

    id: Mapped[int] = mapped_column(primary_key=True)
    mode: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_by_user_id: Mapped[int | None] = mapped_column(nullable=True)


class PolicyRuleRow(Base):
    """One denylist or allowlist entry: an entity id or a glob pattern.

    One table with a `kind` column (`deny_entity`, `deny_pattern`,
    `allow_entity`, `allow_pattern`) rather than four separate tables --
    the same one-code-path reasoning D-13 gives for `SafetyPolicyRow.mode`.
    `created_by_user_id` has no foreign key yet, for the same reason
    `SafetyPolicyRow.updated_by_user_id` does not.
    """

    __tablename__ = "policy_rules"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    created_by_user_id: Mapped[int | None] = mapped_column(nullable=True)


class AuditRow(Base):
    """One recorded administrative action -- created here so the first
    migration is the only one that has to exist before a policy can be
    stored. Written by plan 03-07's mode-switch route; nothing in plan
    03-02 writes to this table yet."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(nullable=True)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
