"""The six ORM rows migration 0015 creates for Phase 9 (Google accounts:
multi-account Calendar and Gmail) -- kept out of `db/models.py` so that
file does not grow with a whole phase's worth of tables, and so this
phase's own repository (`db/google_repository.py`) owns its rows without
touching the shared module every other phase also edits.

`alembic/env.py` imports this module so `Base.metadata` (the one
declarative base every table lives on, per `db/models.py`'s own docstring)
sees these six tables too -- a model class that does not subclass `Base`
is invisible to `alembic revision --autogenerate`, and a model module
nobody imports is invisible even if it does.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, LargeBinary, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from atlas.db.models import Base


class GoogleOAuthClientRow(Base):
    """The one deployer-supplied OAuth client (D-01) -- a single-row table
    (`id` always 1), the same singleton convention `SafetyPolicyRow` and
    `SetupStateRow` already use. The client secret is Fernet ciphertext,
    the same two-column `ciphertext`/`key_version` shape `ProviderCredentialRow`
    already establishes -- never a plaintext column.
    """

    __tablename__ = "google_oauth_client"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_id: Mapped[str] = mapped_column(Text, nullable=False)
    client_secret_ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    key_version: Mapped[int] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)


class GoogleOAuthStateRow(Base):
    """One in-flight OAuth `state` value (D-02), hashed at rest -- the
    same "opaque bearer value, hashed or store-and-compare, never trusted
    from the client alone" discipline `InviteRow.token_hash` already
    establishes for an invite token. `used_at` marks a state consumed by
    the callback, so a replayed callback with the same `state` is refused
    rather than accepted twice.
    """

    __tablename__ = "google_oauth_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    state_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    redirect_uri: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(nullable=True)


class GoogleAccountRow(Base):
    """One linked Google account (D-01, D-03): an operator label, the
    refresh token as Fernet ciphertext (T-09-02), and the account's own
    degraded-reachability status (GOOG-12). `status` is one of the closed
    set `db/google_repository.py::ACCOUNT_STATUSES` names -- the same
    "code enforces the closed set, the column is Text" split every other
    status-shaped column in this project (`WorkflowRunRow.status`,
    `PluginState`) already uses. `is_default` decides which account a
    write with no clear signal goes to (D-04).
    """

    __tablename__ = "google_accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    refresh_token_ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    key_version: Mapped[int] = mapped_column(nullable=False)
    granted_scopes: Mapped[str] = mapped_column(Text, nullable=False)
    is_default: Mapped[bool] = mapped_column(nullable=False, default=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="ok")
    status_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    status_at: Mapped[datetime | None] = mapped_column(nullable=True)
    refresh_token_expires_at: Mapped[datetime | None] = mapped_column(nullable=True)
    linked_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)
    linked_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)


class GoogleCalendarRow(Base):
    """One calendar a linked account's `calendarList` reported -- off
    (D-05) until the operator turns it on. `ondelete="CASCADE"` on
    `account_id` matches `GoogleCalendarRow`'s own analog,
    `WorkflowStepRow.run_id`: deleting an account removes its discovered
    calendars in the same statement. `uq_google_calendars_account_calendar`
    is what makes `add_calendars` a safe upsert-by-discovery: a calendar
    id already present for this account is left untouched (never reset
    back to `off`), never duplicated.
    """

    __tablename__ = "google_calendars"
    __table_args__ = (
        UniqueConstraint(
            "account_id", "google_calendar_id", name="uq_google_calendars_account_calendar"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("google_accounts.id", ondelete="CASCADE"), nullable=False
    )
    google_calendar_id: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    is_primary: Mapped[bool] = mapped_column(nullable=False, default=False)
    can_write: Mapped[bool] = mapped_column(nullable=False, default=False)
    access: Mapped[str] = mapped_column(Text, nullable=False, default="off")
    discovered_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)


class GoogleAccountStyleRow(Base):
    """One account's learned writing style (D-19, D-20) and Gmail
    signature (D-22) -- `account_id` is both this row's primary key and
    its foreign key, the one-to-one shape neither `PluginRow`/
    `PluginConfigValueRow` (one-to-many) nor any other existing table in
    this project needs, so it is spelled out plainly here rather than
    borrowed from an analog that does not quite fit. `samples` is a JSON
    array of a few short real sent-mail excerpts (D-19), never git --
    this table, like `GoogleAccountRow.refresh_token_ciphertext`, only
    ever exists in the operator's own database.
    """

    __tablename__ = "google_account_style"

    account_id: Mapped[int] = mapped_column(
        ForeignKey("google_accounts.id", ondelete="CASCADE"), primary_key=True
    )
    profile: Mapped[str] = mapped_column(Text, nullable=False, default="")
    samples: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    signature_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    signature_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="not_learned")
    status_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    messages_scanned: Mapped[int] = mapped_column(nullable=False, default=0)
    learned_at: Mapped[datetime | None] = mapped_column(nullable=True)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)


class PendingActionRow(Base):
    """One code-held pending action awaiting a spoken confirm/cancel
    (D-06 .. D-09) -- the same narrow, closed-state, indexed-by-expiry
    shape `WakeEventRow` establishes for a short-lived, append-and-expire
    row that is never operator-edited. `arguments` is the exact tool
    arguments `confirm` re-executes verbatim (D-08) -- code never
    re-parses the action from model output. `expires_at` is indexed
    (`ix_pending_actions_expires_at`) since D-09's ~6s window is what a
    later plan's own sweep or lookup reads by.
    """

    __tablename__ = "pending_actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    tool_name: Mapped[str] = mapped_column(Text, nullable=False)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    readback: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    result_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
