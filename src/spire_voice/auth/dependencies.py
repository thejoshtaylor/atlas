"""`Role`, `current_user`, `require_role`, and `require_setup_complete` --
the FastAPI dependencies every account-aware route is built on (WEB-01,
WEB-04, D-08).

Every refusal here comes from a small, named `_xxx_error() -> HTTPException`
factory, copying `app.py`'s own `_calibration_disabled_error` shape exactly
(that function's docstring: "a disabled route must be distinguishable from
a route nobody wrote"). This is the house convention for a refusal in this
codebase, and a fourth style invented here would be a fourth thing to
learn.

`current_user` reads the access-token cookie, verifies it, and loads the
user *fresh from the repository* by the id the token names -- the role
`require_role` checks is always the row's current `role`, never a claim
copied out of the JWT payload. Two things this buys, both load-bearing:
an admin's role change (or a disable) takes effect on this user's very next
request, not after their token expires, matching CONTEXT.md's "read live
at each check" posture for accounts the same way it already applies to
policy; and it structurally rules out the one thing WEB-04's own second
test exists to catch -- a role read from anything the client supplied.

Every function here is typed on `starlette.requests.HTTPConnection`
(`Request`'s and `WebSocket`'s shared base -- both expose `.cookies` and
`.app`), not `Request` -- confirmed against the installed `fastapi==0.141.1`
directly (a `Request`-typed dependency raises a bare `TypeError` when
resolved inside a WebSocket route's dependency chain; `HTTPConnection`
resolves correctly against either). `GET /ws/turn` (Task 3, below) needs
`require_role` to gate it exactly like every HTTP route, so every
dependency in this file has to work for both from the start, not be
special-cased per connection type at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fastapi import Depends, HTTPException
from starlette.requests import HTTPConnection

from spire_voice.auth.tokens import InvalidAccessToken, verify_access_token
from spire_voice.config import Config, SecurityConfig
from spire_voice.db.repository import AccountRepository


class Role(str, Enum):
    """The three roles WEB-04 names, in ascending rank order (`_ROLE_RANK`
    below) -- `admin` reaches everything, `operator` edits policy/macros
    and uses the voice surfaces, `viewer` reads and writes nothing."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


_ROLE_RANK: dict[Role, int] = {Role.VIEWER: 0, Role.OPERATOR: 1, Role.ADMIN: 2}


@dataclass(frozen=True)
class CurrentUser:
    """The authenticated caller -- never carries a password hash, a token,
    or anything a route should not turn around and put in a response body
    (`GET /api/auth/me`'s own return shape mirrors this directly)."""

    id: int
    email: str
    display_name: str
    role: Role


def _unauthenticated_error() -> HTTPException:
    """T-03-30's "same refusal for an unknown email and a wrong password"
    reasoning extends here too: one generic 401, never a detail that tells
    an unauthenticated caller *why* (expired vs. malformed vs. no cookie at
    all) -- that distinction is not actionable to a caller who is not
    signed in, and finer detail here would be exactly the account/session
    enumeration surface T-03-30 already closes on the sign-in route."""
    return HTTPException(status_code=401, detail="not authenticated")


def _forbidden_error(minimum: Role) -> HTTPException:
    return HTTPException(status_code=403, detail=f"this action requires the {minimum.value} role")


def _setup_incomplete_error() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "setup incomplete: no admin account exists yet -- finish the "
            "first-run setup before this route will answer"
        ),
    )


async def current_user(conn: HTTPConnection) -> CurrentUser:
    """Read the access-token cookie, verify it, and load the user it names
    fresh from `app.state.account_repo` -- see the module docstring for why
    this is a repository read on every call rather than a claim copied out
    of the token payload."""
    config: Config = conn.app.state.config
    security: SecurityConfig = config.security
    account_repo: AccountRepository = conn.app.state.account_repo

    token = conn.cookies.get(security.cookie_name)
    if not token:
        raise _unauthenticated_error()

    try:
        payload = verify_access_token(token, security)
    except InvalidAccessToken:
        raise _unauthenticated_error()

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        raise _unauthenticated_error()

    user = await account_repo.get_user_by_id(user_id)
    if user is None or user.disabled_at is not None:
        # A disabled user's otherwise-valid token is refused exactly like
        # an invalid one -- the caller learns nothing about which case it
        # was, the same non-disclosure T-03-30 already establishes.
        raise _unauthenticated_error()

    return CurrentUser(id=user.id, email=user.email, display_name=user.display_name, role=Role(user.role))


def require_role(minimum: Role):
    """Build a dependency admitting `current_user` only at rank `minimum`
    or above -- `require_role(Role.OPERATOR)` admits an admin too, without
    enumerating every role combination that should pass."""

    async def _dependency(user: CurrentUser = Depends(current_user)) -> CurrentUser:
        if _ROLE_RANK[user.role] < _ROLE_RANK[minimum]:
            raise _forbidden_error(minimum)
        return user

    return _dependency


# The only routes that may answer while `users` is empty (D-08): the one
# route that can clear the gate, the route the browser asks before it
# knows whether anything else will answer, and the health check --
# everything else in this application, present or future, is behind this
# gate by construction (registered as an application-level dependency in
# `app.py`, not repeated per-route).
#
# Plan 03-09 adds the wizard's own routes (`routes/wizard.py`): the
# create-admin route is permanently closed once an admin exists, so a
# half-finished install is reachable only by signing in, and a gate that
# blocks the very routes which would finish setup is a lockout -- the
# thing WEB-02's "never a lockout" guarantee exists to rule out. Every one
# of these routes still sits behind `require_role(Role.ADMIN)`
# (`routes/wizard.py`'s own router), so this is an exemption from the
# setup gate specifically, never from authentication.
SETUP_GATE_EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/api/auth/create-admin",
        "/api/setup/status",
        "/health",
        "/api/wizard",
        "/api/wizard/steps/hub/check",
        "/api/wizard/audio-source",
        "/api/wizard/finish",
    }
)


async def require_setup_complete(conn: HTTPConnection) -> None:
    """Refuse every route but the three named in `SETUP_GATE_EXEMPT_PATHS`
    with a 503 naming "setup incomplete" while no user of any role exists
    (D-08: "no user," not "no admin" -- a system with a viewer and no
    admin is a system somebody got partway into)."""
    if conn.url.path in SETUP_GATE_EXEMPT_PATHS:
        return
    account_repo: AccountRepository = conn.app.state.account_repo
    if not await account_repo.any_user_exists():
        raise _setup_incomplete_error()
