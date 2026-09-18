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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from spire_mcp.safety import Policy
from spire_voice.db.models import (
    AuditRow,
    InviteRow,
    PolicyRuleRow,
    ProviderCredentialRow,
    RefreshTokenRow,
    SafetyPolicyRow,
    UserRow,
)
from spire_voice.db.repository import Credential, Invite, PolicyRule, RefreshToken, User

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
                    created_at=row.created_at,
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
                    at=datetime.now(timezone.utc),
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
                created_at=datetime.now(timezone.utc),
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
                created_at=row.created_at,
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
            now = datetime.now(timezone.utc)
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
        created_at=row.created_at,
        disabled_at=row.disabled_at,
    )


def _invite_from_row(row: InviteRow) -> Invite:
    return Invite(
        id=row.id,
        token_hash=row.token_hash,
        role=row.role,
        email=row.email,
        expires_at=row.expires_at,
        created_by_user_id=row.created_by_user_id,
        accepted_at=row.accepted_at,
        accepted_by_user_id=row.accepted_by_user_id,
    )


def _refresh_token_from_row(row: RefreshTokenRow) -> RefreshToken:
    return RefreshToken(
        id=row.id,
        user_id=row.user_id,
        token_hash=row.token_hash,
        issued_at=row.issued_at,
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
        rotated_to_id=row.rotated_to_id,
    )


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
                created_at=datetime.now(timezone.utc),
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
                row.disabled_at = datetime.now(timezone.utc)
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
                expires_at=expires_at,
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

    async def accept_invite(
        self, invite_id: int, *, accepted_by_user_id: int, accepted_at: datetime
    ) -> None:
        async with self._sessionmaker() as session:
            row = await session.get(InviteRow, invite_id)
            if row is not None:
                row.accepted_at = accepted_at
                row.accepted_by_user_id = accepted_by_user_id
                await session.commit()

    async def revoke_invite(self, invite_id: int, *, revoked_at: datetime) -> None:
        async with self._sessionmaker() as session:
            row = await session.get(InviteRow, invite_id)
            if row is not None and row.accepted_at is None:
                row.expires_at = revoked_at
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
                issued_at=issued_at,
                expires_at=expires_at,
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
                issued_at=issued_at,
                expires_at=expires_at,
                revoked_at=None,
                rotated_to_id=None,
            )
            session.add(new_row)
            await session.flush()  # assigns new_row.id, needed below

            old_row.revoked_at = datetime.now(timezone.utc)
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
        now = datetime.now(timezone.utc)
        current: RefreshTokenRow | None = row
        while current is not None:
            if current.revoked_at is None:
                current.revoked_at = now
            next_id = current.rotated_to_id
            current = await session.get(RefreshTokenRow, next_id) if next_id is not None else None


def _credential_from_row(row: ProviderCredentialRow) -> Credential:
    return Credential(
        slot=row.slot,
        ciphertext=row.ciphertext,
        key_version=row.key_version,
        updated_at=row.updated_at,
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
            now = datetime.now(timezone.utc)
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
