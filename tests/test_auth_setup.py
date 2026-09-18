"""There is no default password at any point (WEB-01, D-08).

While the `users` table is empty, only the create-admin route may answer --
every other route must return 503 naming "setup incomplete". Once an admin
exists, the create-admin route must be permanently closed. The failure mode
this file exists to catch is not "the wizard looks broken"; it is "a
half-installed spire-voice is a reachable spire-voice" -- an admin panel
that quietly serves its ordinary routes before an operator has ever set a
password would let a television-shaped sentence, or a stranger on the same
network before setup finishes, reach a route this project's whole safety
posture assumes only an authenticated operator can reach.

Every test here boots the real application through `lifespan`, reusing
`tests/test_startup_smoke.py`'s fake builders -- the failure mode this file
guards against is specifically about the real, wired-together route table,
not a handler called in isolation.

Two tests below are additional named acceptance criteria from this plan's
Task 3 (not "fill this scaffold" work -- there were only two pre-written
scaffolds in this file): sign-in's account-enumeration resistance (T-03-30)
and the plan-wide prohibition on any response body carrying a password
hash, a refresh token, or a JWT.
"""

from __future__ import annotations

import re

import test_startup_smoke as smoke
from fastapi.testclient import TestClient

import spire_voice.app as app_module
from spire_voice.auth.dependencies import SETUP_GATE_EXEMPT_PATHS

_PATH_PARAM_RE = re.compile(r"\{([^}]+)\}")


def _fill_path_params(path: str) -> str:
    """Substitute every `{param}` segment with a value shaped for the
    real parameter -- `account_id`/`invite_id` are `int` fields (a
    non-numeric placeholder would fail FastAPI's own path-parameter
    validation with a 422 before this test's target dependency ever runs,
    which is not the thing this test is checking)."""

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in ("account_id", "invite_id"):
            return "1"
        return "placeholder-value"

    return _PATH_PARAM_RE.sub(_sub, path)


def _boot_with_empty_accounts(tmp_path, monkeypatch) -> TestClient:
    """The real application, booted through `lifespan`, with an *empty*
    `FakeAccountRepository` -- unlike `test_startup_smoke.py`'s own
    pre-seeded default (deliberately "already set up," so its unrelated
    wiring assertions are not all 503s), this file's whole point is the
    genuinely-empty state.
    """
    import conftest

    def _empty_repositories(config: object, engine: object) -> dict:
        return {
            "policy_repo": conftest.FakePolicyRepository(),
            "account_repo": conftest.FakeAccountRepository(),
        }

    # `test_startup_smoke.py`'s own autouse fixture only applies within
    # that module -- this file needs the same structurally-valid test key
    # set explicitly (`validate_secret_key_strength`, plan 03-05).
    monkeypatch.setenv("SPIRE_SECRET_KEY", smoke._TEST_SECRET_KEY)
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(smoke._write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "McpToolHost", smoke._FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _empty_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", smoke._fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", smoke._FakeCameraSource)

    return TestClient(app_module.app)


def test_create_admin_answers_only_while_no_user_exists(tmp_path, monkeypatch):
    """`POST /api/auth/create-admin` must succeed while `users` is empty,
    and must refuse (the route "permanently closed", D-08) once an admin
    exists -- asserted by calling it twice."""
    with _boot_with_empty_accounts(tmp_path, monkeypatch) as client:
        payload = {
            "email": "First.Admin@Example.Invalid",
            "display_name": "First Admin",
            "password": "a-plainly-fictional-test-password",
        }
        first = client.post("/api/auth/create-admin", json=payload)
        assert first.status_code == 201, first.text
        body = first.json()
        assert body["email"] == "first.admin@example.invalid"  # normalized
        assert body["role"] == "admin"
        # No response body from this route may carry a password hash, a
        # refresh token, or a JWT (this plan's own prohibition).
        assert "password" not in body
        assert "password_hash" not in body

        second = client.post(
            "/api/auth/create-admin",
            json={
                "email": "second.admin@example.invalid",
                "display_name": "Second Admin",
                "password": "another-plainly-fictional-password",
            },
        )
        assert second.status_code == 409, second.text
        assert "already exists" in second.json()["detail"].lower()


def test_every_other_route_reports_setup_incomplete_until_an_admin_exists(tmp_path, monkeypatch):
    """Every route other than the three named in `SETUP_GATE_EXEMPT_PATHS`
    must answer 503 naming "setup incomplete" while `users` is empty --
    enumerated from the application's own registered routes, not sampled,
    so the exemption list this test asserts against is the same complete
    list `auth/dependencies.py` itself enforces from."""
    from test_auth_roles import _flatten_routes

    with _boot_with_empty_accounts(tmp_path, monkeypatch) as client:
        flat_routes = _flatten_routes(app_module.app.routes)

        checked_any = False
        for route in flat_routes:
            path = getattr(route, "path", None) or getattr(route, "path_format", None)
            if path is None or path in SETUP_GATE_EXEMPT_PATHS:
                continue
            if path in ("/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"):
                continue
            methods = getattr(route, "methods", None)
            if methods is None:
                # A WebSocket route (/ws/turn) or a StaticFiles mount --
                # neither answers a plain client.get() the way an HTTP
                # route does; the setup gate itself is exercised over
                # every HTTP route, which is exhaustive enough to prove
                # the gate is real (a WebSocket connection attempt against
                # a gated route is exercised separately, below).
                continue
            method = "GET" if "GET" in methods else next(iter(methods))
            response = client.request(method, _fill_path_params(path))
            assert response.status_code == 503, (
                f"{method} {path} answered {response.status_code}, not 503 -- "
                "every route but SETUP_GATE_EXEMPT_PATHS must report setup "
                "incomplete while no user exists"
            )
            assert "setup incomplete" in response.json()["detail"].lower()
            checked_any = True

        assert checked_any, "no non-exempt route was found to check -- this test would pass vacuously"

        # The WebSocket turn route: a denied connection during the
        # handshake, not a 200 upgrade.
        try:
            with client.websocket_connect("/ws/turn"):
                raise AssertionError("/ws/turn accepted a connection with no admin account yet")
        except Exception as exc:  # noqa: BLE001 -- asserting on the denial itself
            status_code = getattr(exc, "status_code", None)
            assert status_code == 503, f"/ws/turn denial carried status {status_code!r}, not 503"

        # The three exempt routes must still answer normally.
        assert client.post(
            "/api/auth/create-admin",
            json={"email": "x@example.invalid", "display_name": "X", "password": "y" * 12},
        ).status_code in (201, 409)
        assert client.get("/api/setup/status").status_code == 200
        assert client.get("/health").status_code == 200


def test_sign_in_gives_the_same_refusal_for_an_unknown_email_and_a_wrong_password(
    tmp_path, monkeypatch
):
    """T-03-30: an unknown email and a wrong password must produce the same
    refusal -- compared directly, not read off the code -- so `/api/auth/
    login` never tells an unauthenticated caller which addresses exist."""
    with _boot_with_empty_accounts(tmp_path, monkeypatch) as client:
        create_response = client.post(
            "/api/auth/create-admin",
            json={
                "email": "known@example.invalid",
                "display_name": "Known Admin",
                "password": "the-real-fictional-password",
            },
        )
        assert create_response.status_code == 201, create_response.text

        wrong_password = client.post(
            "/api/auth/login",
            json={"email": "known@example.invalid", "password": "not-the-real-password"},
        )
        unknown_email = client.post(
            "/api/auth/login",
            json={"email": "nobody-here@example.invalid", "password": "does-not-matter-either"},
        )

        assert wrong_password.status_code == unknown_email.status_code == 401
        assert wrong_password.json() == unknown_email.json(), (
            "a wrong password and an unknown email must produce byte-identical "
            "refusal bodies, or the difference itself discloses which accounts exist"
        )


def test_no_response_body_carries_a_password_hash_a_refresh_token_or_a_jwt(tmp_path, monkeypatch):
    """No route this plan adds may return a password hash, a refresh
    token, or a JWT in its response body -- the session travels in an
    HttpOnly cookie and nowhere else. Walks the real create-admin -> login
    -> refresh -> me -> invite-create -> invite-list -> invite-accept flow
    and scans every serialized body.
    """
    with _boot_with_empty_accounts(tmp_path, monkeypatch) as client:
        bodies: list[tuple[str, str]] = []

        def _record(label: str, response) -> None:
            bodies.append((label, response.text))

        _record(
            "create-admin",
            client.post(
                "/api/auth/create-admin",
                json={
                    "email": "scan-admin@example.invalid",
                    "display_name": "Scan Admin",
                    "password": "yet-another-fictional-password",
                },
            ),
        )
        _record(
            "login",
            client.post(
                "/api/auth/login",
                json={"email": "scan-admin@example.invalid", "password": "yet-another-fictional-password"},
            ),
        )
        _record("refresh", client.post("/api/auth/refresh"))
        _record("me", client.get("/api/auth/me"))
        invite_create = client.post(
            "/api/invites", json={"role": "viewer", "email": "scan-invitee@example.invalid"}
        )
        _record("invite-create", invite_create)
        invite_token = invite_create.json()["token"]
        _record("invite-list", client.get("/api/invites"))
        _record("accounts-list", client.get("/api/accounts"))
        _record(
            "invite-accept",
            client.post(
                f"/api/invites/{invite_token}/accept",
                json={"display_name": "Scan Invitee", "password": "a-fourth-fictional-password"},
            ),
        )

        # The access-token cookie itself is a real JWT -- three
        # dot-separated base64url segments. Every *response body* above is
        # scanned for that shape (a JWT leaking into a body, not the
        # cookie header, which is the one sanctioned place it travels) and
        # for the literal substrings a leaked hash/refresh token would
        # contain.
        jwt_shape = re.compile(r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
        forbidden_keys = ("password_hash", "refresh_token", "access_token", "token_hash")

        for label, text in bodies:
            assert not jwt_shape.search(text), f"{label} response body looks like it contains a JWT: {text!r}"
            for key in forbidden_keys:
                assert key not in text, f"{label} response body contains forbidden key {key!r}: {text!r}"
