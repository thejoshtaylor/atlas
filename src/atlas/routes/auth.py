"""The session surface: create the first admin, sign in, sign out, refresh,
who-am-i, and the setup-status route the browser asks before it knows
whether anything else will answer (WEB-01, D-05, D-08, T-03-24, T-03-30).

Every route body follows `app.py`'s existing shape (03-PATTERNS.md): a
`pydantic.BaseModel` payload declared directly above its route, state read
off `request.app.state` and typed at the top of the function, and every
refusal from a named `_xxx_error()` factory -- copying
`_calibration_disabled_error`'s shape exactly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from atlas.auth.dependencies import CurrentUser, Role, current_user
from atlas.auth.passwords import hash_password, verify_password
from atlas.auth.tokens import (
    clear_session_cookie,
    hash_refresh_token,
    issue_access_token,
    issue_refresh_token,
    read_refresh_cookie,
    rotate_refresh_token,
    set_session_cookie,
)
from atlas.config import Config, SecurityConfig
from atlas.db.repository import AccountRepository, SetupRepository, User
from atlas.routes.wizard import _compute_all_steps

router = APIRouter(prefix="/api/auth", tags=["auth"])
setup_router = APIRouter(prefix="/api/setup", tags=["setup"])


class CreateAdminRequest(BaseModel):
    email: str
    display_name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class SessionResponse(BaseModel):
    """What a session route may return -- an id, an email, a display name,
    and a role. Never a password hash, a token, or an internal id that is
    not needed to render (this plan's own prohibition, asserted by a test
    over the serialized body)."""

    id: int
    email: str
    display_name: str
    role: str


class SetupStep(BaseModel):
    name: str
    complete: bool


class SetupStatusResponse(BaseModel):
    """Booleans and step names only -- never user data (this route is
    exempt from the setup gate by construction, so it must not leak
    anything an unauthenticated caller should not see)."""

    complete: bool
    steps: list[SetupStep]


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _create_admin_closed_error() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(
            "an account already exists -- create-admin answers only once, while no "
            "user of any role exists, and is permanently closed afterwards"
        ),
    )


def _invalid_credentials_error() -> HTTPException:
    """The same refusal for an unknown email, a wrong password, and a
    disabled account (T-03-30) -- the detail names none of the three, so
    this route never tells an unauthenticated caller which addresses exist
    or which accounts are disabled."""
    return HTTPException(status_code=401, detail="incorrect email or password")


def _session_expired_error() -> HTTPException:
    return HTTPException(status_code=401, detail="session expired -- sign in again")


def _to_session_response(user: User) -> SessionResponse:
    return SessionResponse(id=user.id, email=user.email, display_name=user.display_name, role=user.role)


async def _issue_new_session(
    response: Response, user: User, security: SecurityConfig, account_repo: AccountRepository
) -> None:
    """Mint a fresh access token and the *first* refresh token of a new
    session (create-admin, login) -- every refresh token after this one
    goes through `rotate_refresh_token` instead, never through this
    function again for the same session."""
    access_token = issue_access_token(user_id=user.id, role=user.role, security=security)
    refresh_token = issue_refresh_token()
    now = datetime.now(timezone.utc)
    await account_repo.store_refresh_token(
        user_id=user.id,
        token_hash=hash_refresh_token(refresh_token),
        issued_at=now,
        expires_at=now + timedelta(seconds=security.refresh_token_ttl_s),
    )
    set_session_cookie(response, security, access_token=access_token, refresh_token=refresh_token)


@router.post("/create-admin", status_code=201)
async def create_admin(
    payload: CreateAdminRequest, request: Request, response: Response
) -> SessionResponse:
    """Answers only while no user of any role exists (D-08: "no user," not
    "no admin"). There is no default password and no printed setup token
    at any point -- the person creating the admin chooses the password
    here, in the browser, and nothing about it is ever written to a log.

    WR-03 fix (code review): `any_user_exists()` followed by a separate
    `create_user(...)` used to be a check-then-act pair with no atomicity
    between them -- two concurrent requests could both observe an empty
    table and both insert, producing two admin accounts from a route this
    docstring itself says answers exactly once. `create_user_if_no_user_exists`
    makes the check and the insert one atomic database operation; only the
    caller it actually returns a `User` for goes on to get a session.
    """
    config: Config = request.app.state.config
    account_repo: AccountRepository = request.app.state.account_repo

    try:
        user = await account_repo.create_user_if_no_user_exists(
            email=_normalize_email(payload.email),
            display_name=payload.display_name,
            password_hash=hash_password(payload.password),
            role=Role.ADMIN.value,
        )
    except IntegrityError:
        # Defense in depth, per WR-03's own fix suggestion -- the
        # advisory-lock serialization above already makes this
        # unreachable for two `create-admin` calls racing each other, but
        # a unique-violation from `users.email` must still surface as this
        # route's own named refusal, never as an unhandled 500 naming a
        # database column an unauthenticated caller has no reason to see.
        raise _create_admin_closed_error() from None

    if user is None:
        raise _create_admin_closed_error()

    await _issue_new_session(response, user, config.security, account_repo)
    return _to_session_response(user)


@router.post("/login")
async def login(payload: LoginRequest, request: Request, response: Response) -> SessionResponse:
    config: Config = request.app.state.config
    account_repo: AccountRepository = request.app.state.account_repo

    user = await account_repo.get_user_by_email(_normalize_email(payload.email))
    if (
        user is None
        or user.disabled_at is not None
        or not verify_password(payload.password, user.password_hash)
    ):
        raise _invalid_credentials_error()

    await _issue_new_session(response, user, config.security, account_repo)
    return _to_session_response(user)


@router.post("/logout", status_code=204)
async def logout(request: Request, response: Response) -> None:
    config: Config = request.app.state.config
    account_repo: AccountRepository = request.app.state.account_repo

    presented = read_refresh_cookie(request.cookies, config.security)
    if presented is not None:
        await account_repo.revoke_refresh_chain(hash_refresh_token(presented))
    clear_session_cookie(response, config.security)


@router.post("/refresh")
async def refresh(request: Request, response: Response) -> SessionResponse:
    config: Config = request.app.state.config
    account_repo: AccountRepository = request.app.state.account_repo
    security = config.security

    presented = read_refresh_cookie(request.cookies, security)
    if presented is None:
        raise _session_expired_error()

    rotated = await rotate_refresh_token(account_repo, presented, security=security)
    if rotated is None:
        # Unknown or already-revoked (replay) -- the whole chain is
        # revoked by `rotate_refresh_token` itself; the browser's session
        # is over either way, so both cookies are cleared here too.
        clear_session_cookie(response, security)
        raise _session_expired_error()

    new_refresh_token, user_id = rotated
    user = await account_repo.get_user_by_id(user_id)
    if user is None or user.disabled_at is not None:
        clear_session_cookie(response, security)
        raise _session_expired_error()

    access_token = issue_access_token(user_id=user.id, role=user.role, security=security)
    set_session_cookie(
        response, security, access_token=access_token, refresh_token=new_refresh_token
    )
    return _to_session_response(user)


@router.get("/me")
async def me(user: CurrentUser = Depends(current_user)) -> SessionResponse:
    return SessionResponse(id=user.id, email=user.email, display_name=user.display_name, role=user.role.value)


@setup_router.get("/status")
async def setup_status(request: Request) -> SetupStatusResponse:
    """Exempt from the setup gate by construction (`SETUP_GATE_EXEMPT_PATHS`,
    `auth/dependencies.py`) -- this is the route the browser asks before it
    knows whether anything else will answer.

    Plan 03-09 moves this onto the full five-step wizard state
    (`routes/wizard.py`'s own `_compute_all_steps`, the same computation
    `GET /api/wizard` uses, so the two routes can never disagree about
    what each step's condition is). `complete` reads the `setup_state`
    row's own terminal marker (`POST /api/wizard/finish` has actually been
    called), not merely "every condition happens to be true right now" --
    finishing is a deliberate act, matching WEB-03's "the wizard cannot
    finish before the mic/speaker test runs" being a property of a route,
    not of a computed boolean nobody had to act on. `steps` here carries
    step names and booleans only, never the `detail` field `GET
    /api/wizard` returns to an authenticated admin -- this route answers
    before authentication is even possible (T-03-55), so it must leak
    nothing about the house: no hub address, no entity id, no email.
    """
    setup_repo: SetupRepository = request.app.state.setup_repo
    complete = await setup_repo.is_setup_complete()
    steps = await _compute_all_steps(request)
    return SetupStatusResponse(
        complete=complete,
        steps=[SetupStep(name=s.name, complete=s.complete) for s in steps],
    )
