"""Account and invite management: `/api/accounts` and `/api/invites`,
behind `require_role(Role.ADMIN)` -- except accepting an invite, which is
unauthenticated by necessity, because the person accepting has no account
yet (WEB-04, WEB-05, T-03-24, T-03-31).
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.auth.passwords import hash_password
from spire_voice.db.repository import AccountRepository, Invite

router = APIRouter(tags=["accounts"])

# A week -- generous enough that an admin does not have to rush handing an
# invite link to the person it is for, short enough that a forgotten,
# unaccepted invite does not stay live indefinitely.
_DEFAULT_INVITE_TTL_S = 7 * 24 * 3600

_VALID_ROLES = {role.value for role in Role}


def _hash_invite_token(token: str) -> str:
    """The storage form of an invite token -- SHA-256 over the opaque
    value, the same primitive `auth/tokens.py::hash_refresh_token` uses for
    the same reason (a bearer credential's hash, never the credential
    itself, is what a database read should ever yield). Kept local rather
    than imported from `auth/tokens.py`: an invite token and a refresh
    token are different bearer credentials with different lifetimes and
    different tables, and sharing one function name across both would
    blur that they are not interchangeable."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _unknown_role_error(role: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"unknown role {role!r} -- valid roles are {sorted(_VALID_ROLES)!r}",
    )


def _invite_invalid_error() -> HTTPException:
    """One refusal for every way an invite can fail to be accepted -- not
    found, expired, or already accepted -- so accepting an invite never
    discloses which of the three actually happened (the same
    non-disclosure discipline `routes/auth.py::_invalid_credentials_error`
    already applies to sign-in)."""
    return HTTPException(status_code=400, detail="this invite is invalid, expired, or already used")


class AccountResponse(BaseModel):
    """Never a password hash -- this plan's own prohibition, asserted by a
    test over the serialized body."""

    id: int
    email: str
    display_name: str
    role: str
    disabled: bool


def _to_account_response(user) -> AccountResponse:
    return AccountResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        disabled=user.disabled_at is not None,
    )


class InviteCreateRequest(BaseModel):
    role: str
    email: str | None = None
    ttl_s: int = _DEFAULT_INVITE_TTL_S


class InviteCreatedResponse(BaseModel):
    """The one response that ever carries the plaintext invite token --
    never again from any route, per this plan's own prohibition."""

    id: int
    token: str
    role: str
    email: str | None
    expires_at: datetime


class InviteResponse(BaseModel):
    """No token, no hash -- every other invite-listing response, asserted
    against the serialized body by this plan's own acceptance criteria."""

    id: int
    role: str
    email: str | None
    expires_at: datetime
    accepted: bool


def _to_invite_response(invite: Invite) -> InviteResponse:
    return InviteResponse(
        id=invite.id,
        role=invite.role,
        email=invite.email,
        expires_at=invite.expires_at,
        accepted=invite.accepted_at is not None,
    )


class AcceptInviteRequest(BaseModel):
    """`role` is deliberately not a field here -- accepting an invite
    creates a user at the role the invite itself named, never a role from
    this request body (T-03-31). `email` is read only when the invite
    itself carries none (a generic, untargeted invite link) -- when the
    invite names an email, this field is ignored and the invite's own
    value wins, the same "the invite decides, not the request body"
    discipline `role` already gets."""

    display_name: str
    password: str
    email: str | None = None


@router.get("/api/accounts")
async def list_accounts(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[AccountResponse]:
    account_repo: AccountRepository = request.app.state.account_repo
    users = await account_repo.list_users()
    return [_to_account_response(u) for u in users]


@router.delete("/api/accounts/{account_id}", status_code=204)
async def remove_account(
    account_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> None:
    """Disables rather than deletes (`UserRow.disabled_at`'s own docstring)
    -- an audit row or a policy-rule row naming this user's id must still
    resolve after their access ends."""
    account_repo: AccountRepository = request.app.state.account_repo
    await account_repo.disable_user(account_id)


@router.get("/api/invites")
async def list_invites(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[InviteResponse]:
    account_repo: AccountRepository = request.app.state.account_repo
    invites = await account_repo.list_invites()
    return [_to_invite_response(i) for i in invites]


@router.post("/api/invites", status_code=201)
async def create_invite(
    payload: InviteCreateRequest,
    request: Request,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> InviteCreatedResponse:
    """Generates a token, stores only its hash, and returns the plaintext
    token exactly once, in this response, so the admin can pass it on --
    never returned again from any route."""
    if payload.role not in _VALID_ROLES:
        raise _unknown_role_error(payload.role)

    account_repo: AccountRepository = request.app.state.account_repo
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    invite = await account_repo.create_invite(
        token_hash=_hash_invite_token(token),
        role=payload.role,
        email=_normalize_email(payload.email) if payload.email else None,
        expires_at=now + timedelta(seconds=payload.ttl_s),
        created_by_user_id=admin.id,
    )
    return InviteCreatedResponse(
        id=invite.id, token=token, role=invite.role, email=invite.email, expires_at=invite.expires_at
    )


@router.delete("/api/invites/{invite_id}", status_code=204)
async def revoke_invite(
    invite_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> None:
    account_repo: AccountRepository = request.app.state.account_repo
    await account_repo.revoke_invite(invite_id, revoked_at=datetime.now(timezone.utc))


@router.post("/api/invites/{token}/accept", status_code=201)
async def accept_invite(token: str, payload: AcceptInviteRequest, request: Request) -> AccountResponse:
    """Unauthenticated by necessity -- the person accepting has no account
    yet. Creates the user at exactly the role the invite named (T-03-31),
    never a role the request body could ask for -- `AcceptInviteRequest`
    above has no `role` field at all, so there is nothing to ignore.

    WR-03 fix (code review): the old shape read the invite, checked
    `accepted_at is None` and `expires_at`, then created the user, then
    only afterward called `accept_invite` -- with no atomicity between the
    check and the write, two concurrent accepts of the same still-valid
    token could both pass the check and both create a user before either
    flipped `accepted_at`. The read below still decides the early,
    friendly refusal and which email/role to use, but `claim_invite` is
    now the actual authority: it is one atomic compare-and-swap, and only
    the caller it returns `True` for goes on to create a user at all.
    """
    account_repo: AccountRepository = request.app.state.account_repo

    invite = await account_repo.get_invite_by_token_hash(_hash_invite_token(token))
    if invite is None:
        raise _invite_invalid_error()
    if invite.accepted_at is not None:
        raise _invite_invalid_error()
    now = datetime.now(timezone.utc)
    if invite.expires_at <= now:
        raise _invite_invalid_error()

    if invite.email is not None:
        # The invite decides, not the request body -- the same discipline
        # `role` already gets (T-03-31), applied to the address a targeted
        # invite was created for.
        email = invite.email
    elif payload.email:
        email = _normalize_email(payload.email)
    else:
        raise HTTPException(
            status_code=400,
            detail="this invite has no email of its own -- supply one to accept it",
        )

    # The actual gate: an atomic compare-and-swap against the database,
    # not the plain-Python check above (which only produces a friendlier
    # early refusal and can itself be stale by the time this line runs).
    # A caller this returns False for lost the race, or the invite
    # genuinely became invalid between the read above and here -- either
    # way, refused the same way a never-valid token is.
    claimed = await account_repo.claim_invite(invite.id, now=now)
    if not claimed:
        raise _invite_invalid_error()

    try:
        user = await account_repo.create_user(
            email=email,
            display_name=payload.display_name,
            password_hash=hash_password(payload.password),
            role=invite.role,
        )
    except IntegrityError:
        # A unique-violation on users.email -- per WR-03's own fix
        # suggestion, this must surface as a named refusal, never an
        # unhandled 500. The invite stays claimed (its accepted_at is
        # already set): a colliding email means this specific invite
        # cannot be completed, and re-issuing a fresh one is the
        # operator's own remedy, the same as any other invite that
        # expires unused.
        raise HTTPException(
            status_code=409,
            detail="an account with this email already exists",
        ) from None

    await account_repo.record_invite_acceptor(invite.id, accepted_by_user_id=user.id)
    return _to_account_response(user)
