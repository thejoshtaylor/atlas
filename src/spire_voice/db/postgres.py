"""The real `PolicyRepository`, backed by the three tables in
`spire_voice.db.models`.

Takes its `async_sessionmaker` as a constructor argument with no default,
matching `FfmpegSupervisor`'s and `CameraAudioSource`'s own
dependency-injection-over-subclassing convention -- this class never opens
its own engine or session factory, and a test drives it against whatever
sessionmaker it is handed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from spire_mcp.safety import Policy
from spire_voice.db.models import (
    AuditRow,
    InviteRow,
    PolicyRuleRow,
    ProviderCredentialRow,
    RefreshTokenRow,
    SafetyPolicyRow,
    SettingRow,
    SetupStateRow,
    SetupStepRow,
    UserRow,
)
from spire_voice.db.repository import (
    Credential,
    Invite,
    PolicyRule,
    RefreshToken,
    Setting,
    SetupStep,
    User,
)

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
