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
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Protocol, Sequence

from spire_mcp.safety import Policy
from spire_voice.turn.macros import normalize


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
class MacroAction:
    """One action a macro runs, in written order -- the same `tool`/
    `arguments` shape `spire_voice.config.MacroActionConfig` carries, plus
    the database identity (`id`, `position`) a config-parsed action has no
    need for. A plain value object, never an ORM row, matching every
    other repository in this module."""

    id: int
    position: int
    tool: str
    arguments: dict


@dataclass(frozen=True)
class Macro:
    """One macro, with its aliases and ordered actions already attached --
    a caller never reassembles one from two separate repository calls
    (Task 2's own instruction).

    Duck-type compatible with `spire_voice.config.MacroConfig` on purpose:
    `.phrase`, `.reply`, `.actions` (each exposing `.tool`/`.arguments`),
    and `.normalized_keys` below are exactly the shape `turn/macros.py`'s
    `match()`/`fire_macro()` and `spire_voice.config._check_macros_do_not_
    collide()` already read -- neither was written with this class in
    mind, and neither needs to change for a database-loaded macro to be
    interchangeable with a file-loaded one everywhere either was already
    consumed. `id`/`created_at`/`updated_at`/`created_by_user_id` are what
    `MacroConfig` has no need for and this dataclass adds, for the CRUD
    surface plan 04-06's routes build on top of this repository.
    """

    id: int
    phrase: str
    aliases: tuple[str, ...]
    reply: str
    actions: tuple[MacroAction, ...]
    created_at: datetime
    updated_at: datetime
    created_by_user_id: int | None

    @property
    def normalized_keys(self) -> frozenset[str]:
        """The same computation `MacroConfig.normalized_keys`
        (`spire_voice.config`) performs, over this row's own phrase and
        aliases rather than a parsed config block -- duplicated rather
        than imported, since importing `spire_voice.config` here would
        pull this module's own consumers (`config.py` does not import
        `db.repository`, but there is no reason to start that dependency
        for four lines built entirely from `normalize()`, which both
        modules already import from `turn/macros.py` independently).
        This is what makes `_check_macros_do_not_collide` -- the check
        that a database-authored macro must still pass (04-CONTEXT.md,
        the operator's own instruction on this plan) -- reachable against
        a `Sequence[Macro]` with no changes to that function at all.
        """
        return frozenset(normalize(k) for k in (self.phrase, *self.aliases))


class MacroRepository(Protocol):
    """What macro storage must answer (MACRO-03, D-09, D-10).

    Structurally satisfied by `PostgresMacroRepository` (real) and
    `FakeMacroRepository` (`tests/conftest.py`), the same
    dependency-injection-over-subclassing convention every other
    repository in this module already uses.

    "The caller validates, this layer only writes" (`PolicyRepository.
    add_rule`'s own wording) applies here too: a zero-action macro, a
    blank phrase or reply, and a cross-macro collision
    (`spire_voice.config._check_macros_do_not_collide`, callable directly
    against a candidate list built from `list_macros()` plus the would-be
    new or edited macro, since `Macro.normalized_keys` above makes that
    function's own duck-typed contract hold) are all the caller's
    responsibility, both at seed time (this plan's migration) and at
    write time (plan 04-06's routes) -- this protocol's own members never
    re-run that check themselves.
    """

    async def list_macros(self) -> Sequence[Macro]:
        """Every macro, with its aliases and actions already attached, in
        no particular order beyond what the database returns."""
        ...

    async def get_macro(self, macro_id: int) -> Macro | None:
        """One macro by id, or `None` if it does not exist."""
        ...

    async def create_macro(
        self,
        *,
        phrase: str,
        aliases: Sequence[str],
        reply: str,
        actions: Sequence[tuple[str, dict]],
        created_by_user_id: int | None,
    ) -> Macro:
        """Insert one macro, its aliases, and its actions (`(tool,
        arguments)` pairs, in the order given -- `position` is assigned
        from that order, not supplied by the caller) and return it whole.
        The caller is responsible for having already run the collision
        check and refused a zero-action macro before calling this -- this
        method does not re-check either."""
        ...

    async def update_macro(
        self,
        macro_id: int,
        *,
        phrase: str,
        aliases: Sequence[str],
        reply: str,
        actions: Sequence[tuple[str, dict]],
    ) -> Macro:
        """Replace `macro_id`'s phrase, reply, aliases, and actions
        wholesale -- the existing alias and action rows are deleted and
        replaced with the given ones, in the given order, rather than
        diffed, matching the add/remove/reorder editor shape plan 04-07
        builds ('here is the new list' is simpler than computing a diff
        against the old one). Returns the updated macro, including its
        new actions, so a caller never has to reassemble one from two
        calls."""
        ...

    async def delete_macro(self, macro_id: int) -> None:
        """Delete one macro and its aliases and actions together. A no-op
        when `macro_id` does not exist -- removing a macro that is
        already gone is not an error, matching `PolicyRepository.
        remove_rule`'s own convention."""
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


@dataclass(frozen=True)
class WorkflowStep:
    """One step of a `WorkflowRun`, in written order -- a plain value
    object mirroring `WorkflowStepRow`'s own columns, the same lift
    `MacroAction` already gives `MacroActionRow` (D-07). `due_at` and
    `fired_at` are always aware UTC; the repository boundary
    (`db/postgres.py`'s `_to_naive_utc`/`_to_aware_utc`) is what makes
    that true regardless of the database column's own naive storage.
    """

    id: int
    run_id: int
    position: int
    kind: str
    arguments: dict
    due_at: datetime
    status: str
    attempts: int
    result_detail: dict | None
    fired_at: datetime | None
    # Migration 0007 (CR-01 fix): `None` for a step never claimed under
    # the two-phase design, or a fake-repository step (no in-memory
    # equivalent of a durable, crash-recoverable claim exists to report).
    claimed_at: datetime | None = None


@dataclass(frozen=True)
class WorkflowRun:
    """One scheduled plan, with its ordered steps already attached -- the
    same "a caller never reassembles one from two separate repository
    calls" convention `Macro` already establishes for its own actions."""

    id: int
    origin: str
    status: str
    summary: str
    created_at: datetime
    updated_at: datetime
    created_by_user_id: int | None
    steps: tuple[WorkflowStep, ...]


@dataclass(frozen=True)
class WorkflowStepSpec:
    """One step specification a caller hands to
    `WorkflowRepository.create_run` -- a kind and its own per-kind
    `arguments` payload, in written order. Mirrors the `(tool, arguments)`
    pair `MacroRepository.create_macro`'s own `actions` parameter already
    takes for the identical job: the caller supplies written order, the
    repository assigns `position` from it.

    A `wait` step's `arguments` carries one key, `duration_s` (a
    non-negative number of seconds) -- the only kind whose payload
    `assign_step_due_ats` below reads, because it is the only kind whose
    duration folds forward into the steps that follow it (PA-D1).
    """

    kind: str
    arguments: dict


def push_out_due_at(due_at: datetime, seconds: float) -> datetime:
    """Push an already-claimed step's `due_at` (an aware UTC instant)
    forward by `seconds` -- the retry backoff
    `PostgresWorkflowRepository.claim_and_execute_next_due_step` applies
    when a step's outcome asks for a retry (PA-D3). Kept here rather than
    inline in `db/postgres.py` for the same reason `assign_step_due_ats`
    below is: that module's own acceptance criterion forbids it from
    importing `timedelta` itself, so both of this file's small absolute-
    instant-arithmetic helpers live in one place."""
    return due_at + timedelta(seconds=seconds)


def stale_claim_cutoff(now: datetime, claim_recovery_after_s: float) -> datetime:
    """The instant a `claimed` step's own `claimed_at` must be at or
    before to be eligible for CR-01's recovery-claim query (`db/postgres.
    py`'s `PostgresWorkflowRepository._lock_stale_claimed_row`): `now`
    minus `WorkflowConfig.claim_recovery_after_s`. Lives here, not inline
    in `db/postgres.py`, for the identical reason `push_out_due_at` above
    does -- arithmetic on an already-absolute instant, kept out of the one
    module whose own acceptance criterion forbids it from importing
    `timedelta` itself."""
    return now - timedelta(seconds=claim_recovery_after_s)


def next_append_due_at(steps: "Sequence[WorkflowStep]", now: datetime) -> datetime:
    """The absolute instant an appended step's own `due_at` folds forward
    from (`WorkflowRepository.append_steps`, D-11, plan 05-02): the
    existing last step's own `due_at`, plus that last step's own
    `duration_s` folded forward if it is itself a `wait` step (PA-D1) --
    an appended step must land after what is already scheduled, never
    merely after `now`. Falls back to `now` only for the pathological
    case of a run with no steps at all, never produced by `create_run`
    (which always writes at least one)."""
    if not steps:
        return now
    last = steps[-1]
    if last.kind == "wait":
        return last.due_at + timedelta(seconds=last.arguments.get("duration_s", 0))
    return last.due_at


def assign_step_due_ats(
    steps: "Sequence[WorkflowStepSpec]", base_time: datetime
) -> list[datetime]:
    """The one place `wait` step durations are folded forward into the
    `due_at` of every step written after them (PA-D1, D-04) -- called by
    `db/postgres.py`'s `PostgresWorkflowRepository.create_run`, kept here
    rather than inline so `db/postgres.py` never has to import `timedelta`
    itself (that module's own acceptance criterion: it contains no
    `ZoneInfo`, `fromisoformat`, `timedelta`, or `astimezone` of its own --
    `workflow.schedule.resolve_schedule` is the only place in this project
    that turns a caller's way of saying "when" into an absolute instant).

    This is arithmetic on an already-absolute instant (`base_time`,
    `resolve_schedule`'s own return value), never a second resolution of a
    wall-clock string -- accumulating a fixed count of seconds onto an
    aware UTC `datetime` cannot be affected by a daylight-saving boundary,
    which is exactly why `resolve_schedule` itself never needs to touch
    this function's output afterward.

    Each `wait` step's own row keeps the `due_at` it would have had if it
    completed instantly (RESEARCH.md Pitfall 5: the waiting is expressed
    declaratively, in `due_at`, never as a suspended executor holding a
    claimed row's lock); its `duration_s` is added only to the `due_at` of
    every step that comes after it in `steps`.
    """
    due_ats: list[datetime] = []
    accumulated = timedelta(0)
    for step in steps:
        due_ats.append(base_time + accumulated)
        if step.kind == "wait":
            accumulated += timedelta(seconds=step.arguments.get("duration_s", 0))
    return due_ats


class WorkflowRunNotFoundError(Exception):
    """`append_steps`/`replace_steps`: no `workflow_runs` row exists for
    the given id -- deliberately a different exception than
    `WorkflowRunNotAppendableError` below, so a caller (`routes/
    workflows.py`, plan 05-04) can tell "no such run" (404) apart from
    "that run already started" (409), the two refusals Task 3's own
    instruction says must be distinguishable."""

    def __init__(self, run_id: int) -> None:
        self.run_id = run_id
        super().__init__(f"workflow run {run_id} does not exist")


class WorkflowRunNotAppendableError(Exception):
    """`append_steps`/`replace_steps`: the run exists but its `status` is
    not `pending` -- D-11's own refusal (append-only, refused once a run
    has started firing). Carries the run's actual status so a caller can
    compose a refusal that names what state blocked it, not merely that
    something did."""

    def __init__(self, run_id: int, status: str) -> None:
        self.run_id = run_id
        self.status = status
        super().__init__(f"workflow run {run_id} is not appendable (status={status!r})")


class WorkflowRepository(Protocol):
    """What scheduled-workflow storage must answer: creating a run,
    claiming its next due step (05-01), and the authoring surface plan
    05-02 adds here -- `list_runs`, `get_run`, `cancel_run`,
    `append_steps`, `replace_steps` -- the reads and writes both the
    webapp (plan 05-04) and the voice path (plan 05-05) need, so neither
    adds a second implementation of the same rule.
    """

    async def create_run(
        self,
        *,
        origin: str,
        summary: str,
        steps: Sequence[WorkflowStepSpec],
        base_time: datetime,
        created_by_user_id: int | None,
    ) -> WorkflowRun:
        """Insert one run and its ordered steps in one transaction.
        `base_time` is already an absolute, aware UTC instant --
        `workflow.schedule.resolve_schedule`'s own return value. This
        method performs no zone handling and parses no string; it calls
        `assign_step_due_ats` to fold every `wait` step's duration forward
        (PA-D1) and writes the result."""
        ...

    async def claim_and_execute_next_due_step(
        self,
        executor: "Callable[[Any], Awaitable[Any]]",
        now: datetime,
    ) -> bool:
        """Claim the single oldest due, claimable step across every run
        with `SELECT ... FOR UPDATE SKIP LOCKED`, call `executor` against
        it, and write its outcome -- in two transactions, not one (CR-01's
        code-review fix, superseding D-02's original one-transaction
        design): the first locks the row, marks it `claimed` with a fresh
        `claimed_at`, and commits, releasing the row lock *before*
        `executor` is ever awaited; the second, opened only after
        `executor` returns, re-locks the same row by id and writes its
        terminal (or retried) state. A step is claimable only when no
        earlier-position sibling in its own run is still `pending` *or*
        `claimed` (steps run in written order, FLOW-01) and its own run's
        status is `pending` or `firing` (a cancelled or terminal run has
        no claimable steps, D-11/D-12).

        A step found `claimed` with a `claimed_at` older than
        `WorkflowConfig.claim_recovery_after_s` is a step a prior caller
        was interrupted before writing a terminal outcome for -- claimed
        again (its own `claimed_at` renewed, the same lease-renewal
        discipline a fresh claim's own commit already establishes) and
        handed to `executor` with `recovered=True` signalled on the step
        object itself (`getattr(step, "recovered", False)`), so
        `workflow.steps.execute_step` can refuse to repeat a
        `call_service` step whose first attempt's outcome is now unknown
        (T-05-01) while still safely re-running `wait`/`speak`. Returns
        `True` when a step was claimed (whatever its outcome), `False`
        when nothing was due or stale."""
        ...

    async def list_runs(self, *, statuses: Sequence[str] | None = None) -> Sequence[WorkflowRun]:
        """Every run, newest first, with its own steps already attached in
        `position` order -- explicit `ORDER BY` on both, never left to
        insertion order (Phase 4's HI-01 review: an unordered
        `list_macros()` meant which row won a collision was undefined --
        the same defect class, closed on the way in here rather than
        found in review). `statuses`, when given, restricts to runs whose
        `status` is one of the given set -- the webapp's own pending-run
        list (D-16) passes `("pending", "firing")`; `None` returns every
        run regardless of status."""
        ...

    async def get_run(self, run_id: int) -> WorkflowRun | None:
        """One run with its steps, or `None` if `run_id` does not
        exist."""
        ...

    async def cancel_run(
        self, run_id: int, *, now: datetime, cancelled_by_user_id: int | None
    ) -> bool:
        """Move `run_id` to `cancelled` and every still-`pending` step of
        its own to `cancelled`, in one transaction. Returns `False`,
        changing nothing, when `run_id` does not exist or is already
        terminal (`completed`/`cancelled`/`failed`) -- `True` only when
        this call is the one that actually moved it. Never deletes a row
        (D-12): a cancelled run stays readable for as long as the
        database does. A step the poller has already claimed (no longer
        `pending` by the time this runs) is left exactly as the poller
        left it -- the claim's own run-status guard is what stops
        anything further in this run from becoming claimable once
        `status` is `cancelled`, so this method needs no separate
        coordination with the poller to make that true."""
        ...

    async def append_steps(
        self, run_id: int, specs: Sequence[WorkflowStepSpec], *, now: datetime
    ) -> WorkflowRun:
        """Append `specs` to the end of `run_id`'s ordered step list,
        computing each new step's `due_at` from the run's own last step
        (`next_append_due_at`, never from `now`) so an appended step lands
        after what is already scheduled (PA-D1). Raises
        `WorkflowRunNotFoundError` when `run_id` does not exist, and
        `WorkflowRunNotAppendableError` when the run's `status` is not
        `pending` (D-11: append-only, refused once a run has started
        firing) -- the two are deliberately different exceptions so a
        caller can tell "no such run" apart from "that run already
        started".

        **The guard is the point of this method.** An implementation must
        take a row lock on the run first and re-read its `status` under
        that lock before inserting a single step row, all in one
        transaction -- the third application of the WR-03 (accounts and
        invites)/HI-01 (macro phrases) lesson this codebase has now
        learned twice, applied here before a review finds it a third
        time, not after. A status read followed by an unguarded insert is
        the same check-then-act shape both of those were, and the poller
        can move a run from `pending` to `firing` at any instant."""
        ...

    async def replace_steps(
        self, run_id: int, specs: Sequence[WorkflowStepSpec], *, now: datetime
    ) -> WorkflowRun:
        """The webapp editor's whole-list save (plan 05-04 only): replace
        `run_id`'s entire ordered step list with `specs`, recomputing
        every `due_at` from the run's existing first step's own `due_at`
        (the run's own schedule start, preserved across an edit that only
        changes its steps) rather than from `now`. Same row lock, same
        status refusal, and the same two exceptions as `append_steps`."""
        ...


@dataclass(frozen=True)
class PluginConfigValue:
    """One key/value pair of a plugin's own configuration (D-03, D-16) --
    a plain value object, never an ORM row, matching every other
    repository in this module.

    Carries either `value` (a plain, non-secret value) or `(ciphertext,
    key_version)` (a secret one) -- never a decrypted string. This is
    what makes a repository return value safe to serialize into a
    response body with nothing to leak (D-03, the write-only property
    PROV-04 established): the plugin-child spawn path
    (`spire_voice.plugins.host`) is the one place `ciphertext` is ever
    decrypted.
    """

    key: str
    secret: bool
    value: str | None
    ciphertext: bytes | None
    key_version: int | None


@dataclass(frozen=True)
class Plugin:
    """One plugin row, as a plain value object (D-01, D-04, PLUG-09) --
    the same reason `Macro`/`PolicyRule` above are plain value objects
    rather than ORM rows: a caller outside `src/spire_voice/db/` has no
    reason to hold a SQLAlchemy-mapped instance open past its session.
    """

    id: int
    slug: str
    display_name: str
    transport: str
    args: tuple[str, ...]
    url: str | None
    enabled: bool
    builtin: bool
    enforces_policy: bool
    timeout_ms: int
    created_at: datetime
    updated_at: datetime
    created_by_user_id: int | None


class PluginAlreadyExistsError(Exception):
    """`create_plugin`: `slug` is already taken by another row -- the
    database's own `uq_plugins_slug` constraint is the actual authority
    (`PostgresPluginRepository.create_plugin` catches the resulting
    integrity error and re-raises this), matching `WorkflowRunNotFoundError`'s
    own "a repository-level exception a route converts to a named refusal"
    shape rather than letting a raw database error surface as a bare 500."""

    def __init__(self, slug: str) -> None:
        self.slug = slug
        super().__init__(f"a plugin with slug {slug!r} already exists")


class PluginRepository(Protocol):
    """What plugin storage must answer: the read surface `PluginManager`
    (`spire_voice.plugins.manager`) actually uses (D-01, D-04), plus the
    write half `routes/plugins.py` (plan 06-06) needs for install, enable/
    disable, configuration edits, and delete.

    "The caller validates, this layer only writes" (`PolicyRepository.
    add_rule`'s own wording) applies here too: a catalog entry's own
    declared keys, the args-or-url transport exclusivity, a builtin row's
    delete refusal, and which submitted config value is actually new
    ciphertext versus "leave the existing one alone" are all
    `routes/plugins.py`'s job -- this protocol's own members never re-run
    any of those checks themselves. Every `PluginConfigValue` this
    protocol's write members accept or return already carries either a
    plain `value` or an already-encrypted `(ciphertext, key_version)` pair
    -- decryption happens at exactly one point, `PluginManager`'s own
    child-spawn path (D-03), never in this module or in `routes/plugins.py`.

    Structurally satisfied by `PostgresPluginRepository` (real) and
    `FakePluginRepository` (`tests/conftest.py`), the same
    dependency-injection-over-subclassing convention every other
    repository in this module already uses.
    """

    async def list_plugins(self) -> Sequence[Plugin]:
        """Every plugin row, enabled and not -- `PluginManager.start_all()`
        is the one caller that filters on `enabled` itself; this method
        returns the whole table."""
        ...

    async def get_plugin(self, plugin_id: int) -> "Plugin | None":
        """One plugin row by id, or `None` if it does not exist -- the
        404 `routes/plugins.py` reads to decide "unknown plugin"."""
        ...

    async def get_config_values(self, plugin_id: int) -> Sequence[PluginConfigValue]:
        """Every config value belonging to `plugin_id`, in no particular
        order beyond what the database returns -- the child's environment
        is built key by key from this list (`spire_voice.plugins.host`),
        so order does not matter the way `MacroActionRow.position` does
        for a macro's actions."""
        ...

    async def create_plugin(
        self,
        *,
        slug: str,
        display_name: str,
        transport: str,
        args: Sequence[str],
        url: "str | None",
        timeout_ms: int,
        config_values: Sequence[PluginConfigValue],
        created_by_user_id: "int | None",
    ) -> Plugin:
        """Insert one plugin row and its initial configuration values
        together, and return the created row. Always `enabled=True`,
        `builtin=False`, `enforces_policy=False` -- only the seed migration
        ever creates a builtin row (D-04), and no route in this phase may
        ever set the policy-enforcing flag (T-06-28). Raises
        `PluginAlreadyExistsError` when `slug` collides with an existing
        row -- the caller is responsible for having already chosen a slug
        it believes is free; this is the backstop the database's own
        unique constraint provides, not a second uniqueness check run
        ahead of the insert."""
        ...

    async def set_enabled(self, plugin_id: int, enabled: bool) -> Plugin:
        """Flip `enabled` and return the updated row -- changes nothing
        else (Task 2's own instruction). The caller is responsible for
        having already 404'd on an unknown `plugin_id` before calling
        this."""
        ...

    async def set_config_values(
        self, plugin_id: int, values: Sequence[PluginConfigValue]
    ) -> Sequence[PluginConfigValue]:
        """Upsert each of `values` by its own `key` -- a key already
        present is overwritten with the given row exactly as given, a new
        one is inserted. A key that already exists but is not mentioned in
        `values` is left completely alone: this method never deletes a
        config value, and "leave a secret key already set that is
        submitted blank alone" is implemented by the caller simply
        omitting that key from `values`, never by this method inspecting
        what it was asked to skip. Returns every one of this plugin's
        config values afterward, not merely the ones just written, so a
        caller never has to reassemble the full set from two calls."""
        ...

    async def delete_plugin(self, plugin_id: int) -> None:
        """Delete one plugin row and its configuration values together
        (`ondelete=CASCADE`, matching `MacroRepository.delete_macro`'s own
        convention). The caller is responsible for having already refused
        a builtin row before calling this (D-04) -- this method does not
        re-check `builtin` itself. A no-op when `plugin_id` does not
        exist, matching `PolicyRepository.remove_rule`'s own convention."""
        ...
