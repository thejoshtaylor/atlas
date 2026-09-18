"""The policy repository protocol: what `lifespan` and the policy routes
need from storage, with no commitment to how storage answers it.

Follows `spire_voice.providers.base`'s `Protocol`-typed shape exactly --
`PolicyRepository` is a `typing.Protocol`, not an abstract base class, so
`PostgresPolicyRepository` (real) and `FakePolicyRepository`
(`tests/conftest.py`) both satisfy it structurally, the same
dependency-injection-over-subclassing convention `FfmpegSupervisor` and
`CameraAudioSource` already use for their own real dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from spire_mcp.safety import Policy


@dataclass(frozen=True)
class PolicyRule:
    """One row from `policy_rules`, carried as a plain value object rather
    than the ORM row itself -- a caller outside `src/spire_voice/db/` has
    no reason to hold a SQLAlchemy-mapped instance open past its session."""

    id: int
    kind: str
    value: str
    note: str | None
    created_at: datetime
    created_by_user_id: int | None


class PolicyRepository(Protocol):
    """What the safety policy's storage layer must answer.

    Three members only: `load_policy` builds the frozen `Policy`
    `mcp/spire_mcp/safety.py` enforces, `list_rules` is the raw rows the
    webapp's policy editor renders, and `record_audit` is the one write
    path plan 03-07's mode switch uses. Nothing here returns an ORM row --
    a `Policy` and a `PolicyRule` are both plain value objects.
    """

    async def load_policy(self) -> Policy:
        """Build the current `Policy` from the database's rows."""
        ...

    async def list_rules(self) -> Sequence[PolicyRule]:
        """Every `policy_rules` row, for the webapp's editor."""
        ...

    async def record_audit(
        self, action: str, detail: dict, actor_user_id: int | None
    ) -> None:
        """Write one `audit_log` row."""
        ...


@dataclass(frozen=True)
class User:
    """One row from `users`, carried as a plain value object -- the same
    reason `PolicyRule` above is one rather than an ORM row: a caller
    outside `src/spire_voice/db/` has no reason to hold a SQLAlchemy-mapped
    instance open past its session. `password_hash` is included (a route
    needs it to verify a sign-in) but is never serialized into a response
    body -- that is a route-layer discipline, not something this value
    object enforces on its own.
    """

    id: int
    email: str
    display_name: str
    password_hash: str
    role: str
    created_at: datetime
    disabled_at: datetime | None


@dataclass(frozen=True)
class Invite:
    """One row from `invites`, as a plain value object. Never carries the
    plaintext token -- only `token_hash`, which a route body must never
    serialize either (the plaintext is returned exactly once, at creation,
    straight from the token this repository was handed to hash, never read
    back from storage)."""

    id: int
    token_hash: str
    role: str
    email: str | None
    expires_at: datetime
    created_by_user_id: int
    accepted_at: datetime | None
    accepted_by_user_id: int | None


@dataclass(frozen=True)
class RefreshToken:
    """One row from `refresh_tokens`, as a plain value object. Never
    carries the plaintext token, for the same reason `Invite.token_hash`
    does not."""

    id: int
    user_id: int
    token_hash: str
    issued_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    rotated_to_id: int | None


class AccountRepository(Protocol):
    """What accounts, invites, and refresh-token storage must answer
    (WEB-01, WEB-04, WEB-05, D-05, D-08).

    Structurally satisfied by `PostgresAccountRepository` (real) and
    `FakeAccountRepository` (`tests/conftest.py`), the same
    dependency-injection-over-subclassing convention `PolicyRepository`
    above already uses.
    """

    async def any_user_exists(self) -> bool:
        """True once any user row exists, of any role -- the create-admin
        gate's own question (D-08: "no user," not "no admin"; a system
        with a viewer and no admin is a system somebody got partway into)."""
        ...

    async def create_user(
        self, *, email: str, display_name: str, password_hash: str, role: str
    ) -> User:
        """Create one user row and return it. `email` must already be
        normalized (lower case, trimmed) by the caller -- this method does
        not re-normalize, so a caller cannot rely on it to paper over a
        skipped normalization step."""
        ...

    async def get_user_by_email(self, email: str) -> User | None:
        """`email` must already be normalized by the caller, matching
        `create_user`'s own contract."""
        ...

    async def get_user_by_id(self, user_id: int) -> User | None: ...

    async def list_users(self) -> Sequence[User]: ...

    async def disable_user(self, user_id: int) -> None:
        """Set `disabled_at` -- never a `DELETE` (module docstring on
        `UserRow`)."""
        ...

    async def create_invite(
        self,
        *,
        token_hash: str,
        role: str,
        email: str | None,
        expires_at: datetime,
        created_by_user_id: int,
    ) -> Invite:
        ...

    async def get_invite_by_token_hash(self, token_hash: str) -> Invite | None: ...

    async def accept_invite(
        self, invite_id: int, *, accepted_by_user_id: int, accepted_at: datetime
    ) -> None:
        """Mark one invite accepted. The caller (`routes/accounts.py`) is
        responsible for checking `accepted_at is None` and `expires_at` are
        both still satisfied before calling this -- this method itself does
        not re-check either, so a caller cannot rely on it to paper over a
        skipped check."""
        ...

    async def revoke_invite(self, invite_id: int, *, revoked_at: datetime) -> None:
        """Revoke an unaccepted invite before it is ever used, by setting
        `expires_at` to `revoked_at` -- reusing the expired-invite refusal
        `accept_invite`'s caller already has to check, rather than adding a
        second 'revoked' state alongside 'expired' for every future reader
        of this table to reason about. A no-op against an already-accepted
        invite -- revoking a used invite has no effect."""
        ...

    async def list_invites(self) -> Sequence[Invite]: ...

    async def store_refresh_token(
        self, *, user_id: int, token_hash: str, issued_at: datetime, expires_at: datetime
    ) -> RefreshToken:
        """Store the first refresh token of a session, minted at sign-in --
        `rotate_refresh_token` below is the only path for every token after
        this one."""
        ...

    async def rotate_refresh_token(
        self,
        old_token_hash: str,
        *,
        new_token_hash: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> RefreshToken | None:
        """Present `old_token_hash`; on success, store a new row, mark the
        old row revoked and pointed (`rotated_to_id`) at the new one, and
        return the new row. Returns `None` when `old_token_hash` is unknown
        or was already revoked -- the latter is a replay (either a bug or a
        theft), and an implementation must revoke the whole chain (every
        row reachable from `old_token_hash` via `rotated_to_id`) before
        returning `None`, matching `revoke_refresh_chain`'s own effect."""
        ...

    async def revoke_refresh_chain(self, token_hash: str) -> None:
        """Revoke `token_hash`'s row and every row reachable from it via
        `rotated_to_id` -- both the sign-out path (revoking the one live
        token in the chain) and the replay-detected path (revoking a whole
        chain from an old, already-rotated token forward to its still-live
        descendant) are the same operation from this method's point of
        view."""
        ...
