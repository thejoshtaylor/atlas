"""The real `PolicyRepository`, backed by the three tables in
`spire_voice.db.models`.

Takes its `async_sessionmaker` as a constructor argument with no default,
matching `FfmpegSupervisor`'s and `CameraAudioSource`'s own
dependency-injection-over-subclassing convention -- this class never opens
its own engine or session factory, and a test drives it against whatever
sessionmaker it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from spire_mcp.safety import Policy
from spire_voice import config as _config_module
from spire_voice.db.models import (
    AuditRow,
    InviteRow,
    MacroActionRow,
    MacroAliasRow,
    MacroRow,
    PolicyRuleRow,
    ProviderCredentialRow,
    RefreshTokenRow,
    SafetyPolicyRow,
    SettingRow,
    SetupStateRow,
    SetupStepRow,
    UserRow,
    WorkflowRunRow,
    WorkflowStepRow,
)
from spire_voice.db.repository import (
    Credential,
    Invite,
    Macro,
    MacroAction,
    PolicyRule,
    RefreshToken,
    Setting,
    SetupStep,
    User,
    WorkflowRun,
    WorkflowStep,
    WorkflowStepSpec,
    assign_step_due_ats,
    push_out_due_at,
)
from spire_voice.turn.macros import normalize as _normalize_macro_key

# Discovered while fixing WR-03 (code review): every `Mapped[datetime]`
# column in `db/models.py` maps, with no `timezone=True`, to Postgres'
# `TIMESTAMP WITHOUT TIME ZONE` -- and every datetime this module has ever
# written is `datetime.now(timezone.utc)`, timezone-*aware*. Nothing in
# this suite ever exercised a real write through the asyncpg driver before
# WR-03's own concurrency test needed one (every prior test either drove
# `FakeAccountRepository`, which is plain Python with no column type at
# all, or exercised writes through Alembic's *synchronous* `psycopg`
# driver, which tolerates the mismatch silently -- asyncpg does not: it
# raises `DataError: can't subtract offset-naive and offset-aware
# datetimes` on the very first insert). This is a real, pre-existing,
# runtime-breaking defect this review did not name (it sits outside every
# one of CR-01/WR-01/WR-02/WR-03/IN-01's cited files), caught only because
# WR-03's own fix needed a test that finally booted a real write against a
# real Postgres. Fixed module-wide here, not only in the class WR-03
# touches -- leaving `PostgresPolicyRepository`/`PostgresSetupRepository`/
# `PostgresSettingsRepository`/`PostgresCredentialRepository` broken in
# the same file, in the same way, once it is this well understood, would
# be worse than not having found it.
#
# The convention going forward: every datetime this module hands to the
# database goes through `_to_naive_utc` first; every datetime it reads
# back comes out through `_to_aware_utc`. Application code everywhere else
# in this project (route handlers, `auth/tokens.py`, `crypto/credentials.py`)
# keeps working in timezone-aware UTC throughout, exactly as before --
# only this module's own database boundary changes.


def _to_naive_utc(dt: datetime) -> datetime:
    """Strip `tzinfo` for a naive `TIMESTAMP WITHOUT TIME ZONE` column,
    normalizing to UTC first if `dt` carries a different zone -- this
    project's own convention is that every instant reaching this module is
    already `datetime.now(timezone.utc)`, so the `astimezone` call is
    almost always a no-op; doing it unconditionally makes that convention
    load-bearing rather than merely assumed."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None)


def _to_aware_utc(dt: datetime | None) -> datetime | None:
    """The read-side inverse of `_to_naive_utc`: a naive value read back
    from one of this module's own `TIMESTAMP WITHOUT TIME ZONE` columns
    is, by the same convention, always UTC -- re-attach the tzinfo so
    every datetime this module hands back is directly comparable to
    `datetime.now(timezone.utc)`, matching what every caller outside this
    module already assumes it can do. A `None` (every optional timestamp
    column here defaults to unset, never a placeholder date) passes
    through unchanged."""
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


# `setup_state` is a single-row table, the same singleton convention
# `_SINGLETON_POLICY_ID` already establishes for `safety_policy` below.
_SINGLETON_SETUP_STATE_ID = 1

# `safety_policy` is a single-row table -- `id` is always this value, never
# generated, so `load_policy`/a future write path never has to discover it.
_SINGLETON_POLICY_ID = 1


class PostgresPolicyRepository:
    """`PolicyRepository`, implemented against a real Postgres.

    Structurally satisfies `spire_voice.db.repository.PolicyRepository`
    (a `typing.Protocol`) -- there is no base class to inherit from.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def load_policy(self) -> Policy:
        """Build the current `Policy` from `safety_policy` and
        `policy_rules` -- the sibling of `Policy.from_config`, fed by rows
        instead of a config block."""
        async with self._sessionmaker() as session:
            mode_row = await session.get(SafetyPolicyRow, _SINGLETON_POLICY_ID)
            mode = mode_row.mode if mode_row is not None else "allow_all_except_denylist"

            rules = (await session.execute(select(PolicyRuleRow))).scalars().all()
            deny_entities = [r.value for r in rules if r.kind == "deny_entity"]
            deny_patterns = [r.value for r in rules if r.kind == "deny_pattern"]
            allow_entities = [r.value for r in rules if r.kind == "allow_entity"]
            allow_patterns = [r.value for r in rules if r.kind == "allow_pattern"]

        return Policy.from_db_rows(
            mode=mode,
            deny_entities=deny_entities,
            deny_patterns=deny_patterns,
            allow_entities=allow_entities,
            allow_patterns=allow_patterns,
        )

    async def list_rules(self) -> list[PolicyRule]:
        async with self._sessionmaker() as session:
            rows = (await session.execute(select(PolicyRuleRow))).scalars().all()
            return [
                PolicyRule(
                    id=row.id,
                    kind=row.kind,
                    value=row.value,
                    note=row.note,
                    created_at=_to_aware_utc(row.created_at),
                    created_by_user_id=row.created_by_user_id,
                )
                for row in rows
            ]

    async def record_audit(
        self, action: str, detail: dict, actor_user_id: int | None
    ) -> None:
        async with self._sessionmaker() as session:
            session.add(
                AuditRow(
                    at=_to_naive_utc(datetime.now(timezone.utc)),
                    actor_user_id=actor_user_id,
                    action=action,
                    detail=detail,
                )
            )
            await session.commit()

    async def add_rule(
        self, *, kind: str, value: str, note: str | None, created_by_user_id: int | None
    ) -> PolicyRule:
        async with self._sessionmaker() as session:
            row = PolicyRuleRow(
                kind=kind,
                value=value,
                note=note,
                created_at=_to_naive_utc(datetime.now(timezone.utc)),
                created_by_user_id=created_by_user_id,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return PolicyRule(
                id=row.id,
                kind=row.kind,
                value=row.value,
                note=row.note,
                created_at=_to_aware_utc(row.created_at),
                created_by_user_id=row.created_by_user_id,
            )

    async def remove_rule(self, rule_id: int) -> None:
        async with self._sessionmaker() as session:
            row = await session.get(PolicyRuleRow, rule_id)
            if row is not None:
                await session.delete(row)
                await session.commit()

    async def set_mode(self, mode: str, *, updated_by_user_id: int | None) -> None:
        async with self._sessionmaker() as session:
            row = await session.get(SafetyPolicyRow, _SINGLETON_POLICY_ID)
            now = _to_naive_utc(datetime.now(timezone.utc))
            if row is None:
                # Defensive: 0001's own seed always inserts this singleton
                # row, but a mode switch must not depend on that having
                # happened -- an absent row is created here rather than
                # raising, so this method's own contract ("set the single
                # active mode") holds regardless of seed history.
                row = SafetyPolicyRow(
                    id=_SINGLETON_POLICY_ID,
                    mode=mode,
                    updated_at=now,
                    updated_by_user_id=updated_by_user_id,
                )
                session.add(row)
            else:
                row.mode = mode
                row.updated_at = now
                row.updated_by_user_id = updated_by_user_id
            await session.commit()


def _user_from_row(row: UserRow) -> User:
    return User(
        id=row.id,
        email=row.email,
        display_name=row.display_name,
        password_hash=row.password_hash,
        role=row.role,
        created_at=_to_aware_utc(row.created_at),
        disabled_at=_to_aware_utc(row.disabled_at),
    )


def _invite_from_row(row: InviteRow) -> Invite:
    return Invite(
        id=row.id,
        token_hash=row.token_hash,
        role=row.role,
        email=row.email,
        expires_at=_to_aware_utc(row.expires_at),
        created_by_user_id=row.created_by_user_id,
        accepted_at=_to_aware_utc(row.accepted_at),
        accepted_by_user_id=row.accepted_by_user_id,
    )


def _refresh_token_from_row(row: RefreshTokenRow) -> RefreshToken:
    return RefreshToken(
        id=row.id,
        user_id=row.user_id,
        token_hash=row.token_hash,
        issued_at=_to_aware_utc(row.issued_at),
        expires_at=_to_aware_utc(row.expires_at),
        revoked_at=_to_aware_utc(row.revoked_at),
        rotated_to_id=row.rotated_to_id,
    )


# WR-03 fix (code review): an arbitrary, fixed bigint naming the
# create-admin critical section for `pg_advisory_xact_lock` -- distinct
# from any other advisory lock key this codebase might one day take, and
# stable across the life of this table (changing it would only matter if
# two old and new processes both raced the *same* moment of a rolling
# deploy, which this project's single-process deployment shape does not
# do). Spelled out as the ASCII bytes of "spire-admin", stuffed into a
# 63-bit int -- readable in a `pg_locks` dump, not that it needs to be.
_CREATE_FIRST_USER_LOCK_KEY = int.from_bytes(b"spireadm", "big") & 0x7FFFFFFFFFFFFFFF


class PostgresAccountRepository:
    """`AccountRepository`, implemented against a real Postgres.

    Structurally satisfies `spire_voice.db.repository.AccountRepository` (a
    `typing.Protocol`) -- there is no base class to inherit from, matching
    `PostgresPolicyRepository`'s own convention above.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def any_user_exists(self) -> bool:
        async with self._sessionmaker() as session:
            row = (await session.execute(select(UserRow.id).limit(1))).first()
            return row is not None

    async def create_user(
        self, *, email: str, display_name: str, password_hash: str, role: str
    ) -> User:
        async with self._sessionmaker() as session:
            row = UserRow(
                email=email,
                display_name=display_name,
                password_hash=password_hash,
                role=role,
                created_at=_to_naive_utc(datetime.now(timezone.utc)),
                disabled_at=None,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _user_from_row(row)

    async def create_user_if_no_user_exists(
        self, *, email: str, display_name: str, password_hash: str, role: str
    ) -> User | None:
        """WR-03 fix (code review): `create_admin`'s own check-then-act,
        made atomic. The old shape -- `any_user_exists()`, then, if
        `False`, `create_user(...)` -- had no transaction isolation, no
        lock, and no unique constraint spanning "the whole table has at
        most one row from this path" between the two calls: two concurrent
        `POST /api/auth/create-admin` requests could both observe an empty
        table and both insert, producing two admin accounts from a route
        this phase's own design states answers exactly once.

        `pg_advisory_xact_lock` -- a session-independent, transaction-scoped
        lock keyed by `_CREATE_FIRST_USER_LOCK_KEY` -- serializes every
        concurrent caller of this method through the same critical
        section: the first to acquire it proceeds to check-then-insert
        undisturbed; a second, concurrent caller blocks at the lock
        acquisition itself until the first's transaction commits, then
        finds a non-empty table and returns `None` instead of racing the
        insert. The lock is released automatically at transaction end
        (commit or rollback) -- never held across this call's boundary,
        so a crashed connection can never leave it stuck.
        """
        async with self._sessionmaker() as session:
            await session.execute(select(func.pg_advisory_xact_lock(_CREATE_FIRST_USER_LOCK_KEY)))
            already_exists = (await session.execute(select(UserRow.id).limit(1))).first() is not None
            if already_exists:
                return None
            row = UserRow(
                email=email,
                display_name=display_name,
                password_hash=password_hash,
                role=role,
                created_at=_to_naive_utc(datetime.now(timezone.utc)),
                disabled_at=None,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _user_from_row(row)

    async def get_user_by_email(self, email: str) -> User | None:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(select(UserRow).where(UserRow.email == email))
            ).scalar_one_or_none()
            return _user_from_row(row) if row is not None else None

    async def get_user_by_id(self, user_id: int) -> User | None:
        async with self._sessionmaker() as session:
            row = await session.get(UserRow, user_id)
            return _user_from_row(row) if row is not None else None

    async def list_users(self) -> list[User]:
        async with self._sessionmaker() as session:
            rows = (await session.execute(select(UserRow))).scalars().all()
            return [_user_from_row(row) for row in rows]

    async def disable_user(self, user_id: int) -> None:
        async with self._sessionmaker() as session:
            row = await session.get(UserRow, user_id)
            if row is not None and row.disabled_at is None:
                row.disabled_at = _to_naive_utc(datetime.now(timezone.utc))
                await session.commit()

    async def create_invite(
        self,
        *,
        token_hash: str,
        role: str,
        email: str | None,
        expires_at: datetime,
        created_by_user_id: int,
    ) -> Invite:
        async with self._sessionmaker() as session:
            row = InviteRow(
                token_hash=token_hash,
                role=role,
                email=email,
                expires_at=_to_naive_utc(expires_at),
                created_by_user_id=created_by_user_id,
                accepted_at=None,
                accepted_by_user_id=None,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _invite_from_row(row)

    async def get_invite_by_token_hash(self, token_hash: str) -> Invite | None:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(
                    select(InviteRow).where(InviteRow.token_hash == token_hash)
                )
            ).scalar_one_or_none()
            return _invite_from_row(row) if row is not None else None

    async def claim_invite(self, invite_id: int, *, now: datetime) -> bool:
        """WR-03 fix (code review): the atomic compare-and-swap
        `routes/accounts.py::accept_invite` claims an invite with, before
        it creates the user the invite is for -- not the informational
        read `get_invite_by_token_hash` above still does first (that read
        keeps deciding the early "not found/expired" refusal and which
        email/role to use; it is no longer what decides whether the
        acceptance itself is allowed to proceed).

        A single `UPDATE ... WHERE accepted_at IS NULL AND expires_at > now`
        -- Postgres evaluates the `WHERE` and the write as one atomic
        operation, so two concurrent callers claiming the same invite
        cannot both match the `WHERE` clause and both update: exactly one
        `UPDATE` affects a row, the other affects zero. Returns whether
        *this* call was the one that matched -- `True` only for the
        caller that actually flipped `accepted_at` from `NULL` to `now`.
        """
        naive_now = _to_naive_utc(now)
        async with self._sessionmaker() as session:
            result = await session.execute(
                update(InviteRow)
                .where(
                    InviteRow.id == invite_id,
                    InviteRow.accepted_at.is_(None),
                    InviteRow.expires_at > naive_now,
                )
                .values(accepted_at=naive_now)
            )
            await session.commit()
            return result.rowcount == 1

    async def record_invite_acceptor(self, invite_id: int, *, accepted_by_user_id: int) -> None:
        """The second half of accepting an invite, called only by the
        caller `claim_invite` just told it won the race (WR-03 fix): fills
        in who accepted it, once the user row that answers that question
        actually exists. Never races anyone -- by the time this runs, no
        other caller can still be contesting the same invite; `claim_invite`
        already settled that."""
        async with self._sessionmaker() as session:
            row = await session.get(InviteRow, invite_id)
            if row is not None:
                row.accepted_by_user_id = accepted_by_user_id
                await session.commit()

    async def revoke_invite(self, invite_id: int, *, revoked_at: datetime) -> None:
        async with self._sessionmaker() as session:
            row = await session.get(InviteRow, invite_id)
            if row is not None and row.accepted_at is None:
                row.expires_at = _to_naive_utc(revoked_at)
                await session.commit()

    async def list_invites(self) -> list[Invite]:
        async with self._sessionmaker() as session:
            rows = (await session.execute(select(InviteRow))).scalars().all()
            return [_invite_from_row(row) for row in rows]

    async def store_refresh_token(
        self, *, user_id: int, token_hash: str, issued_at: datetime, expires_at: datetime
    ) -> RefreshToken:
        async with self._sessionmaker() as session:
            row = RefreshTokenRow(
                user_id=user_id,
                token_hash=token_hash,
                issued_at=_to_naive_utc(issued_at),
                expires_at=_to_naive_utc(expires_at),
                revoked_at=None,
                rotated_to_id=None,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _refresh_token_from_row(row)

    async def rotate_refresh_token(
        self,
        old_token_hash: str,
        *,
        new_token_hash: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> RefreshToken | None:
        async with self._sessionmaker() as session:
            old_row = (
                await session.execute(
                    select(RefreshTokenRow).where(RefreshTokenRow.token_hash == old_token_hash)
                )
            ).scalar_one_or_none()
            if old_row is None:
                return None
            if old_row.revoked_at is not None:
                # A replay: this token was already rotated away (or
                # revoked at sign-out) and is being presented again --
                # either a bug or a theft, and the whole chain forward from
                # it must be revoked, never just refused silently.
                await self._revoke_chain_in_session(session, old_row)
                await session.commit()
                return None

            new_row = RefreshTokenRow(
                user_id=old_row.user_id,
                token_hash=new_token_hash,
                issued_at=_to_naive_utc(issued_at),
                expires_at=_to_naive_utc(expires_at),
                revoked_at=None,
                rotated_to_id=None,
            )
            session.add(new_row)
            await session.flush()  # assigns new_row.id, needed below

            old_row.revoked_at = _to_naive_utc(datetime.now(timezone.utc))
            old_row.rotated_to_id = new_row.id
            await session.commit()
            await session.refresh(new_row)
            return _refresh_token_from_row(new_row)

    async def revoke_refresh_chain(self, token_hash: str) -> None:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(
                    select(RefreshTokenRow).where(RefreshTokenRow.token_hash == token_hash)
                )
            ).scalar_one_or_none()
            if row is not None:
                await self._revoke_chain_in_session(session, row)
                await session.commit()

    async def _revoke_chain_in_session(self, session, row: RefreshTokenRow) -> None:
        """Walk `row` forward through `rotated_to_id`, revoking every row
        that is not already revoked -- the shared walk both
        `rotate_refresh_token`'s replay branch and `revoke_refresh_chain`
        use, so there is exactly one implementation of "revoke a chain,"
        not two that could disagree.
        """
        now = _to_naive_utc(datetime.now(timezone.utc))
        current: RefreshTokenRow | None = row
        while current is not None:
            if current.revoked_at is None:
                current.revoked_at = now
            next_id = current.rotated_to_id
            current = await session.get(RefreshTokenRow, next_id) if next_id is not None else None


def _setup_step_from_row(row: SetupStepRow) -> SetupStep:
    return SetupStep(
        id=row.id, name=row.name, completed_at=_to_aware_utc(row.completed_at), detail=row.detail
    )


class PostgresSetupRepository:
    """`SetupRepository`, implemented against a real Postgres.

    Structurally satisfies `spire_voice.db.repository.SetupRepository` (a
    `typing.Protocol`) -- there is no base class to inherit from, matching
    every other `Postgres*Repository` class in this module.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def is_setup_complete(self) -> bool:
        async with self._sessionmaker() as session:
            row = await session.get(SetupStateRow, _SINGLETON_SETUP_STATE_ID)
            return row is not None and row.completed_at is not None

    async def mark_setup_complete(self, *, completed_at: datetime) -> None:
        naive_completed_at = _to_naive_utc(completed_at)
        async with self._sessionmaker() as session:
            row = await session.get(SetupStateRow, _SINGLETON_SETUP_STATE_ID)
            if row is None:
                # Defensive, matching `PostgresPolicyRepository.set_mode`'s
                # own posture toward a missing singleton row: `0004`'s own
                # seed always inserts this row, but finishing the wizard
                # must not depend on that having happened.
                row = SetupStateRow(id=_SINGLETON_SETUP_STATE_ID, completed_at=naive_completed_at)
                session.add(row)
            else:
                row.completed_at = naive_completed_at
            await session.commit()

    async def get_step(self, name: str) -> SetupStep | None:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(select(SetupStepRow).where(SetupStepRow.name == name))
            ).scalar_one_or_none()
            return _setup_step_from_row(row) if row is not None else None

    async def list_steps(self) -> list[SetupStep]:
        async with self._sessionmaker() as session:
            rows = (await session.execute(select(SetupStepRow))).scalars().all()
            return [_setup_step_from_row(r) for r in rows]

    async def complete_step(self, name: str, *, detail: dict, completed_at: datetime) -> SetupStep:
        naive_completed_at = _to_naive_utc(completed_at)
        async with self._sessionmaker() as session:
            row = (
                await session.execute(select(SetupStepRow).where(SetupStepRow.name == name))
            ).scalar_one_or_none()
            if row is None:
                # Defensive, same reasoning as `mark_setup_complete` above --
                # the migration always seeds this row, but a step route
                # completing must not depend on that having happened.
                row = SetupStepRow(name=name, completed_at=naive_completed_at, detail=detail)
                session.add(row)
            else:
                row.completed_at = naive_completed_at
                row.detail = detail
            await session.commit()
            await session.refresh(row)
            return _setup_step_from_row(row)


def _setting_from_row(row: SettingRow) -> Setting:
    return Setting(
        id=row.id, key=row.key, value=row.value, updated_at=_to_aware_utc(row.updated_at),
        updated_by_user_id=row.updated_by_user_id,
    )


class PostgresSettingsRepository:
    """`SettingsRepository`, implemented against a real Postgres.

    Structurally satisfies `spire_voice.db.repository.SettingsRepository`
    (a `typing.Protocol`) -- there is no base class to inherit from,
    matching every other `Postgres*Repository` class in this module.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def get_setting(self, key: str) -> Setting | None:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(select(SettingRow).where(SettingRow.key == key))
            ).scalar_one_or_none()
            return _setting_from_row(row) if row is not None else None

    async def set_setting(
        self, key: str, value: Any, *, updated_by_user_id: int | None, updated_at: datetime
    ) -> Setting:
        naive_updated_at = _to_naive_utc(updated_at)
        async with self._sessionmaker() as session:
            row = (
                await session.execute(select(SettingRow).where(SettingRow.key == key))
            ).scalar_one_or_none()
            if row is None:
                row = SettingRow(
                    key=key,
                    value=value,
                    updated_at=naive_updated_at,
                    updated_by_user_id=updated_by_user_id,
                )
                session.add(row)
            else:
                row.value = value
                row.updated_at = naive_updated_at
                row.updated_by_user_id = updated_by_user_id
            await session.commit()
            await session.refresh(row)
            return _setting_from_row(row)


def _credential_from_row(row: ProviderCredentialRow) -> Credential:
    return Credential(
        slot=row.slot,
        ciphertext=row.ciphertext,
        key_version=row.key_version,
        updated_at=_to_aware_utc(row.updated_at),
        updated_by_user_id=row.updated_by_user_id,
    )


class PostgresCredentialRepository:
    """`CredentialRepository`, implemented against a real Postgres.

    Structurally satisfies `spire_voice.db.repository.CredentialRepository`
    (a `typing.Protocol`) -- there is no base class to inherit from,
    matching `PostgresPolicyRepository`'s and `PostgresAccountRepository`'s
    own convention above. Every method here moves ciphertext only; nothing
    in this class ever decrypts a value.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def get_credential(self, slot: str) -> Credential | None:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(
                    select(ProviderCredentialRow).where(ProviderCredentialRow.slot == slot)
                )
            ).scalar_one_or_none()
            return _credential_from_row(row) if row is not None else None

    async def list_credentials(self) -> list[Credential]:
        async with self._sessionmaker() as session:
            rows = (await session.execute(select(ProviderCredentialRow))).scalars().all()
            return [_credential_from_row(r) for r in rows]

    async def upsert_credential(
        self,
        slot: str,
        *,
        ciphertext: bytes,
        key_version: int,
        updated_by_user_id: int | None,
    ) -> Credential:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(
                    select(ProviderCredentialRow).where(ProviderCredentialRow.slot == slot)
                )
            ).scalar_one_or_none()
            now = _to_naive_utc(datetime.now(timezone.utc))
            if row is None:
                row = ProviderCredentialRow(
                    slot=slot,
                    ciphertext=ciphertext,
                    key_version=key_version,
                    updated_at=now,
                    updated_by_user_id=updated_by_user_id,
                )
                session.add(row)
            else:
                row.ciphertext = ciphertext
                row.key_version = key_version
                row.updated_at = now
                row.updated_by_user_id = updated_by_user_id
            await session.commit()
            await session.refresh(row)
            return _credential_from_row(row)


async def _load_macro(session: AsyncSession, row: MacroRow) -> Macro:
    """Assemble one whole `Macro` -- its aliases and its actions, in
    written order -- from a `MacroRow` already fetched on `session`.
    Shared by every `PostgresMacroRepository` method that returns a
    `Macro`, so 'update returns the whole macro including its ordered
    actions' (Task 2's own instruction) is one code path, not one per
    method."""
    alias_rows = (
        await session.execute(select(MacroAliasRow).where(MacroAliasRow.macro_id == row.id))
    ).scalars().all()
    action_rows = (
        await session.execute(
            select(MacroActionRow)
            .where(MacroActionRow.macro_id == row.id)
            .order_by(MacroActionRow.position)
        )
    ).scalars().all()
    return Macro(
        id=row.id,
        phrase=row.phrase,
        aliases=tuple(a.alias for a in alias_rows),
        reply=row.reply,
        actions=tuple(
            MacroAction(id=a.id, position=a.position, tool=a.tool, arguments=a.arguments)
            for a in action_rows
        ),
        created_at=_to_aware_utc(row.created_at),
        updated_at=_to_aware_utc(row.updated_at),
        created_by_user_id=row.created_by_user_id,
    )


async def _list_macros_in_session(
    session: AsyncSession, *, exclude_id: int | None = None
) -> list[Macro]:
    """The read half of both `list_macros()` and the collision recheck
    `create_macro`/`update_macro` perform under `_MACRO_WRITE_LOCK_KEY`
    (HI-01 fix, phase 4 code review) -- one query and one row-to-`Macro`
    assembly, not two copies of either. `ORDER BY id` (HI-01's second,
    smaller finding): with no order, `match()` (`turn/macros.py`) --
    which returns the *first* macro whose normalized keys contain a given
    transcript's key -- picked whichever macro Postgres happened to
    return first, not guaranteed stable across a `VACUUM` or a later
    `UPDATE` moving a heap tuple. Ordering by id (insertion order) is a
    second, independent source of determinism this fix closes regardless
    of whether the collision itself is ever hit."""
    query = select(MacroRow).order_by(MacroRow.id)
    rows = (await session.execute(query)).scalars().all()
    macros = [await _load_macro(session, row) for row in rows]
    if exclude_id is not None:
        macros = [m for m in macros if m.id != exclude_id]
    return macros


@dataclass(frozen=True)
class _CandidateMacro:
    """The would-be macro `create_macro`/`update_macro` are about to
    write, shaped exactly like `Macro`'s own duck-typed contract
    (`.phrase`, `.normalized_keys`) so it can sit in the same list
    `_config_module._check_macros_do_not_collide` walks alongside the
    real, already-stored `Macro` rows.

    This mirrors `routes/macros.py::_CandidateMacro` exactly but is not
    imported from there (HI-01 fix, phase 4 code review): a persistence
    module reaching into the HTTP route layer for a shared helper would
    invert this codebase's layering, and `routes/macros.py` already
    imports `spire_voice.db.repository` (transitively, this module) --
    the reverse import would cycle even if the layering were acceptable.
    """

    phrase: str
    aliases: tuple[str, ...]

    @property
    def normalized_keys(self) -> frozenset[str]:
        return frozenset(_normalize_macro_key(k) for k in (self.phrase, *self.aliases))


# HI-01 fix (phase 4 code review): an arbitrary, fixed bigint naming the
# macro-create/update critical section for `pg_advisory_xact_lock` -- the
# same pattern WR-03 (phase 3 code review) established with
# `_CREATE_FIRST_USER_LOCK_KEY` above, applied to a second table.
# `_check_no_collision` in `routes/macros.py` was check-then-act with no
# transaction isolation, lock, or database uniqueness constraint spanning
# the read and the write: two concurrent creates with colliding
# normalized phrases could both pass the route's pre-check and both
# commit, silently shadowing one macro behind the other with which one
# wins undefined (compounded by `list_macros()` carrying no `ORDER BY`,
# fixed above). A database-level unique constraint was not the fit here
# the way it was for `users.email`: a collision is defined over the
# *normalized* union of a phrase and every alias, not a single column, so
# the same advisory-lock-and-recheck shape WR-03 used is the more direct
# fix. Spelled out as the ASCII bytes of "spiremac1", stuffed into a
# 63-bit int -- distinct from `_CREATE_FIRST_USER_LOCK_KEY` so a macro
# write and a create-admin call never contend on the same key.
_MACRO_WRITE_LOCK_KEY = int.from_bytes(b"spiremac1", "big") & 0x7FFFFFFFFFFFFFFF


class PostgresMacroRepository:
    """`MacroRepository`, implemented against a real Postgres.

    Structurally satisfies `spire_voice.db.repository.MacroRepository` (a
    `typing.Protocol`) -- there is no base class to inherit from, matching
    every other `Postgres*Repository` class in this module. `create_macro`/
    `update_macro` replace a macro's alias and action rows wholesale rather
    than diffing them, matching the Protocol's own documented contract.

    HI-01 fix (phase 4 code review): `create_macro`/`update_macro` now
    re-run the same collision check `routes/macros.py`'s own pre-check
    already runs, under `_MACRO_WRITE_LOCK_KEY`, inside the same
    transaction the insert/update itself commits in. This does not
    contradict the Protocol's "the caller validates, this layer only
    writes" convention so much as complete it: the route's pre-check is
    still what decides the common-case, non-racing 409 and which fields
    the error names; this recheck is what makes two concurrent writes
    with colliding phrases *impossible* rather than merely unlikely, the
    same division of labor WR-03 established for `create_admin`/
    `accept_invite` (the route's own read "only decides the early refusal
    message"; the repository-level check is "the actual authority").
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def list_macros(self) -> list[Macro]:
        async with self._sessionmaker() as session:
            return await _list_macros_in_session(session)

    async def get_macro(self, macro_id: int) -> Macro | None:
        async with self._sessionmaker() as session:
            row = await session.get(MacroRow, macro_id)
            if row is None:
                return None
            return await _load_macro(session, row)

    async def create_macro(
        self,
        *,
        phrase: str,
        aliases: list[str] | tuple[str, ...],
        reply: str,
        actions: list[tuple[str, dict]] | tuple[tuple[str, dict], ...],
        created_by_user_id: int | None,
    ) -> Macro:
        now = _to_naive_utc(datetime.now(timezone.utc))
        async with self._sessionmaker() as session:
            # HI-01 fix: acquire the write lock, then re-read the current
            # macro list and recheck the collision -- both inside the
            # transaction this insert commits in -- before adding a single
            # row. A second, concurrent caller blocks at lock acquisition
            # until this transaction ends, then sees this call's own
            # committed row in its own re-read.
            await session.execute(select(func.pg_advisory_xact_lock(_MACRO_WRITE_LOCK_KEY)))
            existing = await _list_macros_in_session(session)
            candidate = _CandidateMacro(phrase=phrase, aliases=tuple(aliases))
            _config_module._check_macros_do_not_collide([*existing, candidate])

            row = MacroRow(
                phrase=phrase,
                reply=reply,
                created_at=now,
                updated_at=now,
                created_by_user_id=created_by_user_id,
            )
            session.add(row)
            await session.flush()
            for alias in aliases:
                session.add(MacroAliasRow(macro_id=row.id, alias=alias))
            for position, (tool, arguments) in enumerate(actions):
                session.add(
                    MacroActionRow(
                        macro_id=row.id, position=position, tool=tool, arguments=arguments
                    )
                )
            await session.commit()
            await session.refresh(row)
            return await _load_macro(session, row)

    async def update_macro(
        self,
        macro_id: int,
        *,
        phrase: str,
        aliases: list[str] | tuple[str, ...],
        reply: str,
        actions: list[tuple[str, dict]] | tuple[tuple[str, dict], ...],
    ) -> Macro:
        async with self._sessionmaker() as session:
            # HI-01 fix: same lock-then-recheck as create_macro, excluding
            # this macro's own current row from the collision set -- the
            # same self-exclusion `routes/macros.py`'s own pre-check
            # already performs (`exclude_macro_id=macro_id`) -- so a save
            # that keeps a macro's own phrase is never refused against
            # itself.
            await session.execute(select(func.pg_advisory_xact_lock(_MACRO_WRITE_LOCK_KEY)))
            row = await session.get(MacroRow, macro_id)
            if row is None:
                raise ValueError(f"macro {macro_id} does not exist")
            existing = await _list_macros_in_session(session, exclude_id=macro_id)
            candidate = _CandidateMacro(phrase=phrase, aliases=tuple(aliases))
            _config_module._check_macros_do_not_collide([*existing, candidate])

            row.phrase = phrase
            row.reply = reply
            row.updated_at = _to_naive_utc(datetime.now(timezone.utc))
            await session.execute(delete(MacroAliasRow).where(MacroAliasRow.macro_id == macro_id))
            await session.execute(
                delete(MacroActionRow).where(MacroActionRow.macro_id == macro_id)
            )
            for alias in aliases:
                session.add(MacroAliasRow(macro_id=macro_id, alias=alias))
            for position, (tool, arguments) in enumerate(actions):
                session.add(
                    MacroActionRow(
                        macro_id=macro_id, position=position, tool=tool, arguments=arguments
                    )
                )
            await session.commit()
            await session.refresh(row)
            return await _load_macro(session, row)

    async def delete_macro(self, macro_id: int) -> None:
        async with self._sessionmaker() as session:
            row = await session.get(MacroRow, macro_id)
            if row is None:
                return
            # `ondelete="CASCADE"` on both `MacroActionRow.macro_id` and
            # `MacroAliasRow.macro_id` (db/models.py) does the rest at the
            # database level -- this delete removes the macro and its
            # actions and aliases together in one statement, not three.
            await session.delete(row)
            await session.commit()


# The run statuses a step's own run must carry for that step to be
# claimable at all -- a cancelled or already-terminal run has no
# claimable steps (D-11, D-12), which is what makes cancellation stop
# every step that has not already been claimed with no separate
# mechanism.
_WORKFLOW_CLAIMABLE_RUN_STATUSES = ("pending", "firing")


def _workflow_step_from_row(row: WorkflowStepRow) -> WorkflowStep:
    return WorkflowStep(
        id=row.id,
        run_id=row.run_id,
        position=row.position,
        kind=row.kind,
        arguments=row.arguments,
        due_at=_to_aware_utc(row.due_at),
        status=row.status,
        attempts=row.attempts,
        result_detail=row.result_detail,
        fired_at=_to_aware_utc(row.fired_at),
    )


def _workflow_run_from_rows(
    run_row: WorkflowRunRow, step_rows: "list[WorkflowStepRow]"
) -> WorkflowRun:
    ordered = sorted(step_rows, key=lambda r: r.position)
    return WorkflowRun(
        id=run_row.id,
        origin=run_row.origin,
        status=run_row.status,
        summary=run_row.summary,
        created_at=_to_aware_utc(run_row.created_at),
        updated_at=_to_aware_utc(run_row.updated_at),
        created_by_user_id=run_row.created_by_user_id,
        steps=tuple(_workflow_step_from_row(r) for r in ordered),
    )


class PostgresWorkflowRepository:
    """`WorkflowRepository`, implemented against a real Postgres.

    Structurally satisfies `spire_voice.db.repository.WorkflowRepository`
    (a `typing.Protocol`) -- there is no base class to inherit from,
    matching every other `Postgres*Repository` class in this module.

    `claim_and_execute_next_due_step` claims with `SELECT ... FOR UPDATE
    SKIP LOCKED`, deliberately *not* `pg_advisory_xact_lock`
    (`PostgresAccountRepository.create_user_if_no_user_exists`'s own
    primitive, above, and `PostgresMacroRepository.create_macro`'s
    `_MACRO_WRITE_LOCK_KEY`). An advisory lock is the right primitive for
    a single named critical section at most one caller may enter at a
    time (creating the one admin account, writing a macro under the
    collision check) -- it is the wrong one here, where N workers should
    each claim a *different* one of M due rows without queueing behind
    each other. `SKIP LOCKED` is what lets that happen: a second,
    concurrent poller's identical query simply skips a row this one
    already holds and claims a different one instead of blocking on it
    (05-RESEARCH.md, "Don't Hand-Roll").
    """

    def __init__(
        self,
        sessionmaker: async_sessionmaker,
        *,
        max_attempts: int = 3,
        retry_backoff_s: float = 30.0,
    ) -> None:
        self._sessionmaker = sessionmaker
        # Bounds retry (PA-D3): a step whose outcome keeps asking for a
        # retry stops being retried once `attempts` reaches this, and its
        # run is left `failed` rather than retrying forever.
        self._max_attempts = max_attempts
        self._retry_backoff_s = retry_backoff_s

    async def create_run(
        self,
        *,
        origin: str,
        summary: str,
        steps: "list[WorkflowStepSpec] | tuple[WorkflowStepSpec, ...]",
        base_time: datetime,
        created_by_user_id: int | None,
    ) -> WorkflowRun:
        now = _to_naive_utc(datetime.now(timezone.utc))
        # `assign_step_due_ats` folds every `wait` step's duration forward
        # into the steps written after it (PA-D1) -- pure arithmetic on
        # `base_time`, an already-absolute instant this method never
        # resolves a wall-clock string or a zone to produce.
        due_ats = assign_step_due_ats(steps, base_time)
        async with self._sessionmaker() as session:
            run_row = WorkflowRunRow(
                origin=origin,
                status="pending",
                summary=summary,
                created_at=now,
                updated_at=now,
                created_by_user_id=created_by_user_id,
            )
            session.add(run_row)
            await session.flush()  # assigns run_row.id for the FK below

            step_rows: list[WorkflowStepRow] = []
            for position, (spec, due_at) in enumerate(zip(steps, due_ats)):
                step_row = WorkflowStepRow(
                    run_id=run_row.id,
                    position=position,
                    kind=spec.kind,
                    arguments=spec.arguments,
                    due_at=_to_naive_utc(due_at),
                    status="pending",
                    attempts=0,
                    result_detail=None,
                    fired_at=None,
                )
                session.add(step_row)
                step_rows.append(step_row)

            await session.commit()
            await session.refresh(run_row)
            for step_row in step_rows:
                await session.refresh(step_row)
            return _workflow_run_from_rows(run_row, step_rows)

    async def claim_and_execute_next_due_step(
        self,
        executor: "Callable[[WorkflowStepRow], Awaitable[Any]]",
        now: datetime,
    ) -> bool:
        """Returns `True` if a step was claimed (whatever its outcome),
        `False` if no step was due. One `async with self._sessionmaker()
        as session:` block, one transaction, no commit between the claim
        and the terminal write -- the claim and the completion share a
        transaction, which is what makes a crash mid-step recoverable
        rather than ambiguous (D-02)."""
        naive_now = _to_naive_utc(now)
        async with self._sessionmaker() as session:
            # A step whose earlier-position sibling in the same run is
            # still `pending` is not claimable -- this is what keeps two
            # steps of one run that fall due at the same instant running
            # in written order, and it holds under concurrency with no
            # extra coordination: an earlier step currently held by
            # another worker's uncommitted transaction is still `pending`
            # from this query's point of view until that transaction
            # commits.
            earlier_sibling = aliased(WorkflowStepRow)
            has_earlier_pending_sibling = (
                select(earlier_sibling.id)
                .where(
                    earlier_sibling.run_id == WorkflowStepRow.run_id,
                    earlier_sibling.position < WorkflowStepRow.position,
                    earlier_sibling.status == "pending",
                )
                .exists()
            )
            # A cancelled or already-terminal run has no claimable steps
            # -- this is what makes cancellation (a plan 05-02 write)
            # stop every step that has not already been claimed, with no
            # separate mechanism this method needs to coordinate with.
            run_is_claimable = (
                select(WorkflowRunRow.id)
                .where(
                    WorkflowRunRow.id == WorkflowStepRow.run_id,
                    WorkflowRunRow.status.in_(_WORKFLOW_CLAIMABLE_RUN_STATUSES),
                )
                .exists()
            )
            stmt = (
                select(WorkflowStepRow)
                .where(
                    WorkflowStepRow.status == "pending",
                    WorkflowStepRow.due_at <= naive_now,
                    ~has_earlier_pending_sibling,
                    run_is_claimable,
                )
                .order_by(
                    WorkflowStepRow.due_at, WorkflowStepRow.run_id, WorkflowStepRow.position
                )
                .with_for_update(skip_locked=True, of=WorkflowStepRow)
                .limit(1)
            )
            row = (await session.execute(stmt)).scalars().first()
            if row is None:
                return False

            run_row = await session.get(WorkflowRunRow, row.run_id)
            if run_row.status == "pending":
                run_row.status = "firing"

            # The work happens with the claimed row's own lock still
            # held -- deliberate, not an oversight (D-02's own
            # consequence; RESEARCH.md Pitfall 3). The executor is a
            # caller of the same tool host the live turn path calls; it
            # never reimplements `allow_call` (D-13).
            outcome = await executor(row)

            row.attempts += 1
            row.result_detail = outcome.detail
            if outcome.retry and row.attempts < self._max_attempts:
                row.status = "pending"
                row.due_at = _to_naive_utc(push_out_due_at(now, self._retry_backoff_s))
            else:
                # Either a terminal outcome, or a retry that has exhausted
                # `max_attempts` -- the latter is left `failed` rather than
                # retried forever (PA-D3, `WorkflowConfig.max_attempts`'s
                # own docstring).
                row.status = "failed" if outcome.retry else outcome.status
                row.fired_at = naive_now

            # A run reaches a terminal state once no step of its own
            # remains `pending`: `completed` when every step completed,
            # `failed` otherwise (PA-D2) -- a denied or failed step never
            # stops the rest of its run from still being attempted at its
            # own due time.
            remaining_pending = (
                await session.execute(
                    select(func.count())
                    .select_from(WorkflowStepRow)
                    .where(WorkflowStepRow.run_id == run_row.id, WorkflowStepRow.status == "pending")
                )
            ).scalar_one()
            if remaining_pending == 0:
                incomplete_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(WorkflowStepRow)
                        .where(
                            WorkflowStepRow.run_id == run_row.id,
                            WorkflowStepRow.status != "completed",
                        )
                    )
                ).scalar_one()
                run_row.status = "completed" if incomplete_count == 0 else "failed"
            run_row.updated_at = naive_now

            await session.commit()  # releases the row lock
            return True
