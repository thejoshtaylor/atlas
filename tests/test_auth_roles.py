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
