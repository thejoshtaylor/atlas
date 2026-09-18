"""The SQLAlchemy declarative base every Phase 3 table inherits from, plus
the three tables plan 03-02 adds (the safety policy the code enforces) and
the three plan 03-05 adds (accounts, invites, refresh tokens).

`alembic/env.py` reads `Base.metadata` as its `target_metadata` -- a model
class that does not subclass `Base`, or a model module nobody imports, is a
model `alembic revision --autogenerate` cannot see. Plan 03-01 added only
the base itself; later plans in this phase (provider credentials, settings)
add their own tables here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, LargeBinary, Text
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
    not be representable (D-13, SAFE-06). `updated_by_user_id` is a real
    foreign key as of plan 03-05, now that `users` exists -- plan
    03-02 left it a plain nullable column until then; the migration that
    adds `users` (`0002_accounts.py`) is the one that adds this constraint.
    """

    __tablename__ = "safety_policy"

    id: Mapped[int] = mapped_column(primary_key=True)
    mode: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )


class PolicyRuleRow(Base):
    """One denylist or allowlist entry: an entity id or a glob pattern.

    One table with a `kind` column (`deny_entity`, `deny_pattern`,
    `allow_entity`, `allow_pattern`) rather than four separate tables --
    the same one-code-path reasoning D-13 gives for `SafetyPolicyRow.mode`.
    `created_by_user_id` is a real foreign key as of plan 03-05, for the
    same reason `SafetyPolicyRow.updated_by_user_id` now is.
    """

    __tablename__ = "policy_rules"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )


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


class UserRow(Base):
    """One operator, admin, or viewer account (WEB-01, WEB-04, D-08).

    `email` is stored normalized to lower case (the route layer -- plan
    03-05's `routes/auth.py` -- normalizes before every read and write) with
    a unique constraint, so the database itself refuses a second account at
    the same address even if a caller ever forgot the normalization step.
    `disabled_at` is how access is removed -- never a `DELETE` -- so an
    audit row or a policy-rule row naming this user's id still resolves
    after their access ends (`SafetyPolicyRow.updated_by_user_id`,
    `PolicyRuleRow.created_by_user_id`, both above, and `InviteRow`/
    `RefreshTokenRow` below all point at this table).
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    disabled_at: Mapped[datetime | None] = mapped_column(nullable=True)


class InviteRow(Base):
    """One admin-issued invite: a role to grant, an optional email, and an
    expiry (WEB-05).

    `token_hash` stores a SHA-256 hash of the bearer token, never the token
    itself -- the same "a database read that yields live sessions is a
    different and worse incident than a database read that yields hashes"
    reasoning `RefreshTokenRow.token_hash` below carries. The plaintext
    token is returned to the admin exactly once, in the create-invite
    response body, and this table never sees it again. `accepted_at`/
    `accepted_by_user_id` are both set together, exactly once, by
    `routes/accounts.py`'s accept-invite route -- a non-`NULL` `accepted_at`
    is what makes a second acceptance attempt refusable without a second
    query.
    """

    __tablename__ = "invites"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    accepted_at: Mapped[datetime | None] = mapped_column(nullable=True)
    accepted_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )


class RefreshTokenRow(Base):
    """One refresh token in a rotation chain (D-05).

    `token_hash` stores a SHA-256 hash of the opaque bearer token, never the
    token itself -- a bearer credential is exactly the kind of value a
    database read should not be able to replay directly; storing only its
    hash means a compromised database backup cannot mint a live session.
    `rotated_to_id` is set, alongside `revoked_at`, the moment this token is
    presented and rotated -- it is the pointer `auth/tokens.py`'s chain-walk
    follows to find (and revoke) every descendant when an already-rotated
    token is presented again, which means either a bug or a theft and never
    a thing to continue through.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    issued_at: Mapped[datetime] = mapped_column(nullable=False)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(nullable=True)
    rotated_to_id: Mapped[int | None] = mapped_column(
        ForeignKey("refresh_tokens.id"), nullable=True
    )


class SetupStateRow(Base):
    """A single-row table (`id` always 1) recording whether the first-run
    wizard has ever been explicitly finished (WEB-01, WEB-02, D-08).

    `completed_at` is set exactly once, by `POST /api/wizard/finish`, and
    never cleared -- this is a terminal marker ("has the wizard been
    finished"), deliberately distinct from "are all five step conditions
    currently true," which `GET /api/wizard` recomputes live on every
    read from the real state each step names (accounts, the hub check
    row, credentials, settings, the calibration file). No row at all
    (before the first `POST /api/wizard/finish`) means the same thing as
    `completed_at IS NULL` -- both read as "not finished yet."
    """

    __tablename__ = "setup_state"

    id: Mapped[int] = mapped_column(primary_key=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)


class SetupStepRow(Base):
    """One wizard step's own record of the one condition that cannot be
    recomputed live on every read: "a call to Home Assistant returned
    successfully, and here is when." `name` is one of the five step names
    `routes/wizard.py` enumerates (`admin_account`, `hub`, `provider_set`,
    `audio_source`, `room`) -- seeded once, in this table's own migration,
    so a fresh install reads every step as present and incomplete rather
    than finding an empty table that reads as "nothing to do."

    Only the `hub` row's `completed_at`/`detail` are ever read back by
    `GET /api/wizard` -- the other four steps' completion is derived live
    from their own real source (the accounts table, the credential
    listing, the settings table, the calibration file on disk) precisely
    so an existing deployment's environment-only credential, or a
    calibration already taken outside the wizard, is recognized without
    anyone having clicked a wizard button. This table's other rows exist
    so a direct read of `setup_steps` always shows the complete, named set
    of five steps, not just the one this application code currently
    consults.
    """

    __tablename__ = "setup_steps"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class SettingRow(Base):
    """One operator-editable setting: a key, a JSON value, when it was
    last written, and by whom. The wizard's audio-source choice is the
    one key this phase ever writes (`routes/wizard.py`'s own
    `AUDIO_SOURCE_SETTING_KEY`), but this table is the general store
    Phases 4 through 8 extend -- a key/value/updated_at/updated_by shape
    with no column added per setting, the same one-code-path reasoning
    `PolicyRuleRow.kind` and `ProviderCredentialRow.slot` already apply to
    their own closed sets.
    """

    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )


class MacroRow(Base):
    """One macro: a phrase that skips the language model and runs a fixed,
    ordered list of actions (`MacroActionRow`), plus every alias it also
    matches on (`MacroAliasRow`) (D-09, D-10, MACRO-03).

    `phrase`/`reply` are exactly `spire_voice.config.MacroConfig.phrase`/
    `.reply` -- the migration that seeds this table
    (`alembic/versions/0005_macro_tables.py`) parses through that same
    class, the one parser both the file and the database go through.
    `created_by_user_id` follows the same real-foreign-key convention
    `PolicyRuleRow.created_by_user_id` already established once `users`
    existed; a seeded row (no operator authored it, the migration did)
    carries `NULL` here, matching `SafetyPolicyRow`'s own seeded-row
    convention.
    """

    __tablename__ = "macros"

    id: Mapped[int] = mapped_column(primary_key=True)
    phrase: Mapped[str] = mapped_column(Text, nullable=False)
    reply: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )


class MacroActionRow(Base):
    """One action a macro runs, in `position` order -- the same `tool`/
    `arguments` shape `spire_voice.config.MacroActionConfig` already
    carries. `position` is what fixes the written order across an update
    that reorders (plan 04-07's editor adds, removes, and reorders); `
    ondelete="CASCADE"` on `macro_id` is what makes deleting a macro also
    delete its own actions with no second query, the same guarantee
    `InviteRow`/`RefreshTokenRow`'s foreign keys give `users` rows,
    applied here to a child a macro fully owns.
    """

    __tablename__ = "macro_actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    macro_id: Mapped[int] = mapped_column(
        ForeignKey("macros.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(nullable=False)
    tool: Mapped[str] = mapped_column(Text, nullable=False)
    arguments: Mapped[dict] = mapped_column(JSON, nullable=False)


class MacroAliasRow(Base):
    """One alias a macro also matches on, as its own row rather than an
    array column -- plan 04-07's editor adds and removes aliases
    individually, and a macro that lost its aliases in a seed loses most
    of the ways it can be spoken. `ondelete="CASCADE"` matches
    `MacroActionRow.macro_id`'s own convention: a macro delete removes its
    aliases together with its actions, in one statement, with no second
    query for either child table.
    """

    __tablename__ = "macro_aliases"

    id: Mapped[int] = mapped_column(primary_key=True)
    macro_id: Mapped[int] = mapped_column(
        ForeignKey("macros.id", ondelete="CASCADE"), nullable=False
    )
    alias: Mapped[str] = mapped_column(Text, nullable=False)


class ProviderCredentialRow(Base):
    """One encrypted provider credential slot (PROV-04, D-07).

    `slot` is unique across the table and drawn from the closed set
    `spire_voice.crypto.credentials.CredentialSlot` names -- a route
    write to an unknown slot is refused before it ever reaches this
    table, so this uniqueness constraint is defense in depth, not the
    only guard. `ciphertext`/`key_version` are exactly what
    `spire_voice.crypto.credentials.encrypt_credential` returns; nothing
    in this table, nor in any repository built on it, ever holds a
    plaintext value.
    """

    __tablename__ = "provider_credentials"

    id: Mapped[int] = mapped_column(primary_key=True)
    slot: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    key_version: Mapped[int] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
