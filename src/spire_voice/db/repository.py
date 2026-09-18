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
from typing import Any, Protocol, Sequence

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

    async def add_rule(
        self, *, kind: str, value: str, note: str | None, created_by_user_id: int | None
    ) -> PolicyRule:
        """Insert one denylist/allowlist rule and return it. The caller
        (`routes/policy.py`) is responsible for validating `kind`/`value`
        against the shapes `mcp/spire_mcp/safety.py` enforces before
        calling this -- this method itself does not re-validate, so a
        caller cannot rely on it to paper over a skipped check."""
        ...

    async def remove_rule(self, rule_id: int) -> None:
        """Delete one rule by id. A no-op when `rule_id` does not exist --
        removing a rule that is already gone is not an error."""
        ...

    async def set_mode(self, mode: str, *, updated_by_user_id: int | None) -> None:
        """Set the single active mode. The caller is responsible for
        validating `mode` against `spire_mcp.safety.Mode`'s two literal
        values before calling this, and for writing the audit row
        (`record_audit`) that makes the change accountable -- this method
        only changes the row."""
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

    async def create_user_if_no_user_exists(
        self, *, email: str, display_name: str, password_hash: str, role: str
    ) -> User | None:
        """WR-03 (code review): the atomic form `routes/auth.py::create_admin`
        uses instead of `any_user_exists()` followed by `create_user(...)`
        -- that check-then-act pair has no transaction isolation between
        the two calls, so two concurrent create-admin requests could both
        observe an empty table and both insert. An implementation must
        make the check and the insert atomic (`PostgresAccountRepository`
        uses a `pg_advisory_xact_lock`, held for one transaction, to
        serialize concurrent callers of this specific method through the
        same critical section -- see its own docstring). Returns `None`,
        inserting nothing, when a user already exists; returns the created
        `User` otherwise."""
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

    async def claim_invite(self, invite_id: int, *, now: datetime) -> bool:
        """WR-03 (code review): the atomic compare-and-swap
        `routes/accounts.py::accept_invite` uses to claim an invite before
        creating the user it is for. Must check `accepted_at IS NULL` and
        `expires_at > now` and set `accepted_at = now` as one atomic
        database operation (`PostgresAccountRepository` uses a single
        conditional `UPDATE ... WHERE ... RETURNING`) -- not a read
        followed by a separate write, which is exactly the unguarded
        window that let two concurrent accepts of the same invite both
        pass the check and both create a user before either flipped
        `accepted_at`. Returns whether *this* call is the one that
        actually claimed it -- `False` for a caller that loses the race,
        or for an invite that is genuinely already accepted or expired."""
        ...

    async def record_invite_acceptor(self, invite_id: int, *, accepted_by_user_id: int) -> None:
        """Records who accepted an invite, called only after `claim_invite`
        has already returned `True` for the same invite -- by then no
        other caller can still be contesting it, so this method itself
        performs no check of its own."""
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


@dataclass(frozen=True)
class SetupStep:
    """One row from `setup_steps`, as a plain value object -- the same
    reason every other row in this module is one. Only the `hub` step's
    row is ever read back by `routes/wizard.py`'s own status computation
    (its module docstring says why); the other four rows exist so a direct
    read of `setup_steps` always shows the complete, named set of five
    steps a fresh install seeds."""

    id: int
    name: str
    completed_at: datetime | None
    detail: dict | None


class SetupRepository(Protocol):
    """What the first-run wizard's own persisted state must answer
    (WEB-01, WEB-02, WEB-03, D-08).

    `setup_state` (the `is_setup_complete`/`mark_setup_complete` pair) is
    the terminal "has the wizard ever been finished" marker `routes/
    wizard.py`'s own module docstring describes; `setup_steps` (the
    `get_step`/`list_steps`/`complete_step` trio) is the one step (`hub`)
    whose completion cannot be recomputed live on every read. Structurally
    satisfied by `PostgresSetupRepository` (real) and `FakeSetupRepository`
    (`tests/conftest.py`), the same dependency-injection-over-subclassing
    convention `PolicyRepository`/`AccountRepository`/`CredentialRepository`
    above already use.
    """

    async def is_setup_complete(self) -> bool:
        """`True` once `POST /api/wizard/finish` has ever succeeded."""
        ...

    async def mark_setup_complete(self, *, completed_at: datetime) -> None:
        """Write the one `setup_state` row's `completed_at`. The caller
        (`routes/wizard.py`) is responsible for having already confirmed
        every step's condition holds -- this method does not re-check any
        of them, so a caller cannot rely on it to paper over a skipped
        check."""
        ...

    async def get_step(self, name: str) -> SetupStep | None:
        """The named step's own row, or `None` if `name` is not one of
        the five names the migration seeded (never expected in practice,
        but not assumed away either)."""
        ...

    async def list_steps(self) -> Sequence[SetupStep]:
        """Every `setup_steps` row -- the seeded five, present from the
        first migration onward."""
        ...

    async def complete_step(
        self, name: str, *, detail: dict, completed_at: datetime
    ) -> SetupStep:
        """Mark `name`'s row complete with `detail` and `completed_at`.
        The caller (`routes/wizard.py`'s hub-check route) is responsible
        for having already verified the step's real condition -- this
        method only records what the caller already confirmed, matching
        every other `*_repository` write method's own "the caller
        validates, this method only writes" contract in this module."""
        ...


@dataclass(frozen=True)
class Setting:
    """One row from `settings`, as a plain value object -- the general
    operator-editable settings store `routes/wizard.py`'s audio-source
    choice is the first, but not the only, writer of."""

    id: int
    key: str
    value: Any
    updated_at: datetime
    updated_by_user_id: int | None


class SettingsRepository(Protocol):
    """What the general operator-editable settings store must answer.
    Two members only, matching `CredentialRepository`'s own read/write
    shape: `get_setting` is the read every caller needs (the wizard's own
    audio-source resolution, and any future setting a later phase adds),
    `set_setting` is the one write path. Structurally satisfied by
    `PostgresSettingsRepository` (real) and `FakeSettingsRepository`
    (`tests/conftest.py`)."""

    async def get_setting(self, key: str) -> Setting | None:
        """The stored row for `key`, or `None` when nothing has been set
        for it yet."""
        ...

    async def set_setting(
        self, key: str, value: Any, *, updated_by_user_id: int | None, updated_at: datetime
    ) -> Setting:
        """Insert or replace the one row for `key` and return it."""
        ...


@dataclass(frozen=True)
class Credential:
    """One row from `provider_credentials`, as a plain value object --
    ciphertext only, the same reason `Invite.token_hash` and
    `RefreshToken.token_hash` never carry a plaintext bearer value. This
    class never decrypts anything: decryption is
    `spire_voice.crypto.credentials.decrypt_credential` (see that module's
    own docstring for the call sites and the guarantee that actually holds
    -- WR-02, code review), never from a repository."""

    slot: str
    ciphertext: bytes
    key_version: int
    updated_at: datetime
    updated_by_user_id: int | None


class CredentialRepository(Protocol):
    """What provider-credential storage must answer (PROV-04, D-07).

    Three members, all over ciphertext: `get_credential`/`list_credentials`
    are reads, `upsert_credential` is the one write path. No member of
    this protocol, nor of either implementation, ever returns a plaintext
    value -- that boundary lives entirely in
    `spire_voice.crypto.credentials`, and only at the one call site that
    module's docstring names.
    """

    async def get_credential(self, slot: str) -> Credential | None:
        """The stored row for `slot`, or `None` when nothing has been
        saved for it yet."""
        ...

    async def list_credentials(self) -> Sequence[Credential]:
        """Every stored row -- the closed slot set itself lives in
        `spire_voice.crypto.credentials.CredentialSlot`, not here; a
        caller walks that set and calls `get_credential` per slot, or
        calls this for every row actually present."""
        ...

    async def upsert_credential(
        self,
        slot: str,
        *,
        ciphertext: bytes,
        key_version: int,
        updated_by_user_id: int | None,
    ) -> Credential:
        """Insert or replace the one row for `slot` and return it. The
        caller (`routes/credentials.py`) is responsible for validating
        `slot` against the closed set before calling this."""
        ...
