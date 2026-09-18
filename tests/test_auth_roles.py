"""The route enforces a role, not the user interface (WEB-04).

CONTEXT.md is explicit that a hidden button is not access control: a
`require_role(...)` FastAPI dependency must sit on each route that needs
one, checked server-side, every request. The failure mode this file exists
to catch is a policy or account route that is reachable by any authenticated
user regardless of role -- a viewer editing the denylist, say -- because the
webapp merely hid the control rather than the backend refusing the call.
The second test guards a narrower, easier mistake: a role read from
somewhere the caller controls (a request body field, a header) rather than
from the signed cookie the server itself issued, which would let any
authenticated request claim whatever role it wants.

The first two tests below build a minimal, throwaway `FastAPI()` app with
one route gated by `require_role` -- Task 2's own scope is the auth
primitives in isolation, before `src/spire_voice/routes/` exists at all
(that is Task 3). The third test, added in Task 3, walks the real
application's own registered routes instead.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from spire_voice.auth.dependencies import Role, require_role
from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig

# A plainly fictional value shaped like a real generated key -- this file
# never calls `validate_secret_key_strength`, so the shape does not need to
# pass that check, only `read_secret_key`'s "is it set at all."
_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


class _RoleClaim(BaseModel):
    """A request body a client could shape to *claim* a role -- accepted
    and ignored by every route below, proving the claim has no effect."""

    role: str | None = None


def _build_test_app(security: SecurityConfig, account_repo) -> FastAPI:
    app = FastAPI()
    # `current_user` (auth/dependencies.py) reads `request.app.state.config.security`
    # and `request.app.state.account_repo` -- a `SimpleNamespace` standing in
    # for `Config` here is enough, since only `.security` is ever read off it.
    app.state.config = SimpleNamespace(security=security)
    app.state.account_repo = account_repo

    @app.post("/operator-only", dependencies=[Depends(require_role(Role.OPERATOR))])
    async def operator_only(payload: _RoleClaim | None = None) -> dict:
        return {"ok": True}

    return app


def test_a_viewer_cannot_reach_an_operator_route(monkeypatch, fake_account_repository):
    """A `viewer`-role account calling a route gated
    `require_role(Role.OPERATOR)` must get a 403, not a hidden-but-reachable
    success."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    app = _build_test_app(security, account_repo)

    import asyncio

    viewer = asyncio.run(
        account_repo.create_user(
            email="viewer@example.invalid",
            display_name="A Viewer",
            password_hash="not-checked-by-this-test",
            role="viewer",
        )
    )
    token = issue_access_token(user_id=viewer.id, role="viewer", security=security)

    client = TestClient(app, cookies={security.cookie_name: token})
    response = client.post("/operator-only")

    assert response.status_code == 403
    assert "operator" in response.json()["detail"].lower()


# --- Task 3: every registered route either enforces a role, or is named ----
# --- here as a deliberate exemption ----------------------------------------


def _flatten_routes(routes):
    """`fastapi==0.141.1` (installed) wraps every `include_router`-added
    route in a `fastapi.routing._IncludedRouter` -- `app.routes` no longer
    lists the actual endpoint routes directly for anything registered
    through `register_routers` (verified directly against this repo's own
    `.venv` this session). `.effective_candidates()` is the documented way
    back to the real, matchable route objects; this recurses because an
    `_IncludedRouter` can itself nest another one."""
    flat = []
    for route in routes:
        if hasattr(route, "effective_candidates"):
            flat.extend(_flatten_routes(route.effective_candidates()))
        else:
            flat.append(route)
    return flat


def _dependant_reaches_current_user(dependant, seen=None) -> bool:
    """True when `dependant`'s own dependency graph includes
    `spire_voice.auth.dependencies.current_user` anywhere -- true for a
    route gated directly with `Depends(current_user)` and for one gated
    with `require_role(...)`, since every `require_role`-built dependency
    itself depends on `current_user` (`auth/dependencies.py`). `seen`
    guards against re-visiting a shared sub-dependency FastAPI may cache."""
    if seen is None:
        seen = set()
    if id(dependant) in seen:
        return False
    seen.add(id(dependant))
    from spire_voice.auth.dependencies import current_user as _current_user

    if dependant.call is _current_user:
        return True
    return any(_dependant_reaches_current_user(sub, seen) for sub in dependant.dependencies)


# Every path here is a deliberate exemption from "carries a role
# dependency," with the reason recorded inline -- not an oversight a
# sampling test could have missed. Framework-internal routes (the OpenAPI/
# docs endpoints FastAPI itself registers) carry no `dependant` at all and
# are excluded from the walk separately, below.
_ROLE_EXEMPT_PATHS = {
    # D-08: the one route that can clear the setup gate cannot itself
    # require an authenticated caller -- nobody is authenticated yet.
    "/api/auth/create-admin",
    # How a caller becomes authenticated in the first place.
    "/api/auth/login",
    # Must succeed harmlessly even from an already-logged-out browser --
    # requiring a live session to sign out would make an expired session
    # impossible to clear.
    "/api/auth/logout",
    # The whole point of a refresh route is to mint a new access token
    # once the old one may already be expired -- it cannot itself require
    # a currently-valid access token.
    "/api/auth/refresh",
    # Exempt from the setup gate by construction and must answer with no
    # user data before the browser knows whether anything else will.
    "/api/setup/status",
    # Unauthenticated by necessity: the person accepting has no account
    # yet (T-03-31's own reasoning).
    "/api/invites/{token}/accept",
    # A `StaticFiles` mount, not an endpoint route -- no `dependant` to
    # inspect at all.
    "/static",
    # The Phase 1 dev-mic index page. Plan 03-05's own Task 4 replaces
    # this route entirely with `app.frontend(...)`; recorded here rather
    # than left for Task 4 to silently change the shape of this test.
    "/",
    # A health check answering only to an authenticated caller is not a
    # health check a container orchestrator or load balancer can use --
    # also exempt from the setup gate by name.
    "/health",
    # Discloses a deployment configuration choice ("websocket"/"webrtc"),
    # never house data, and is not a control surface -- documented in
    # `app.py`'s own route docstring alongside this entry.
    "/transport",
}

# FastAPI's own built-in routes -- not application content, and carry no
# `dependant` attribute at all (they are plain Starlette routes under the
# hood), so they cannot be walked the same way as an `APIRoute`.
_FRAMEWORK_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def test_every_registered_route_enforces_a_role_or_is_named_exempt():
    """Walk every route this application registers (after `register_routers`
    and every `app.py` route decorator have run) and assert each one either
    reaches `current_user` somewhere in its dependency graph, or is named
    in `_ROLE_EXEMPT_PATHS`/`_FRAMEWORK_PATHS` above with a reason. A route
    that forgets its dependency is the entire failure mode of
    dependency-based access control (T-03-25), and a test that samples a
    handful of routes cannot catch the one that was forgotten.
    """
    import spire_voice.app as app_module

    flat_routes = _flatten_routes(app_module.app.routes)

    unexpected_unenforced: list[str] = []
    seen_paths: set[str] = set()
    for route in flat_routes:
        path = getattr(route, "path", None) or getattr(route, "path_format", None)
        assert path is not None, f"a route with no discoverable path exists: {route!r}"
        seen_paths.add(path)

        if path in _FRAMEWORK_PATHS:
            continue
        if path in _ROLE_EXEMPT_PATHS:
            continue

        dependant = getattr(route, "dependant", None)
        if dependant is None or not _dependant_reaches_current_user(dependant):
            unexpected_unenforced.append(path)

    assert not unexpected_unenforced, (
        "the following routes carry no role dependency and are not named in "
        f"_ROLE_EXEMPT_PATHS: {unexpected_unenforced!r} -- either add "
        "require_role(...) to the route, or add it to the exemption list "
        "with a reason"
    )

    # Every exemption named above must correspond to a route that actually
    # exists -- an exemption naming a path this application no longer
    # registers would silently stop meaning anything.
    missing_exemptions = (_ROLE_EXEMPT_PATHS | _FRAMEWORK_PATHS) - seen_paths
    assert not missing_exemptions, (
        f"these exempted paths are not registered by the application: {missing_exemptions!r} -- "
        "the exemption list has drifted from the real route table"
    )


def test_a_role_is_read_from_the_cookie_not_the_request_body(monkeypatch, fake_account_repository):
    """A request body or header claiming a higher role than the caller's
    actual signed cookie carries must not elevate what `require_role`
    grants."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    app = _build_test_app(security, account_repo)

    import asyncio

    viewer = asyncio.run(
        account_repo.create_user(
            email="viewer2@example.invalid",
            display_name="Another Viewer",
            password_hash="not-checked-by-this-test",
            role="viewer",
        )
    )
    token = issue_access_token(user_id=viewer.id, role="viewer", security=security)

    client = TestClient(app, cookies={security.cookie_name: token})
    # The body and the header both claim admin -- neither is ever read by
    # `current_user`/`require_role`, so this must be refused exactly like
    # the plain-viewer call above.
    response = client.post(
        "/operator-only",
        json={"role": "admin"},
        headers={"X-Role": "admin"},
    )

    assert response.status_code == 403
    assert "operator" in response.json()["detail"].lower()
