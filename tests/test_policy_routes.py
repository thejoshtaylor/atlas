"""Editing the policy in the webapp must reach the process that enforces it,
and changing its mode must leave a record of who did it and when (SAFE-06,
SAFE-07).

`tests/test_policy_reaches_the_child.py` already proves a policy handed to
`McpToolHost` at spawn reaches the child -- it does not prove a policy
*changed after spawn* reaches a running child, because Phase 1 never
respawned the child at all. An operator who adds a denylist rule in the
webapp and sees the save succeed, while the MCP child that actually
evaluates every call keeps running on the policy it was spawned with, is a
silent SAFE-06/SAFE-07 regression: the editor works, the enforcement does
not, and nothing here fails loudly enough to notice before the next
restart. The second test guards the accountability side of D-13's two-modes
decision: switching between `allow_all_except_denylist` and `allowlist_only`
is exactly the kind of admin action this project's own prior incident makes
worth a permanent, named record.

Every test here builds a small, throwaway `FastAPI()` app carrying only
`spire_voice.routes.policy`'s own router -- the same "primitives in
isolation" shape `tests/test_auth_roles.py`'s first two tests use, since
this file's whole point is the policy routes themselves, not the rest of
the application.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig
from spire_voice.mcp_client import McpToolHost
from spire_voice.routes.policy import router as policy_router

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MCP_ROOT = REPO_ROOT / "mcp"


def _build_policy_app(security, account_repo, policy_repo, tool_host) -> FastAPI:
    app = FastAPI()
    app.state.config = SimpleNamespace(security=security)
    app.state.account_repo = account_repo
    app.state.policy_repo = policy_repo
    app.state.tool_host = tool_host
    app.state.safety_block = None
    app.include_router(policy_router)
    return app


def _issue_cookie(security, account_repo, *, role: str):
    """Create a user at `role` and return it -- every sync test below
    then issues its own access token from the returned user's id and
    passes it to `TestClient(app, cookies={...})`."""
    user = asyncio.run(
        account_repo.create_user(
            email=f"{role}@example.invalid",
            display_name=f"A {role.title()}",
            password_hash="not-checked-by-this-test",
            role=role,
        )
    )
    return user


class _RecordingToolHost:
    """Stands in for `McpToolHost`: records every `respawn(safety_block)`
    call rather than spawning a real subprocess, so a test can assert
    against the block the route actually rebuilt from the repository."""

    def __init__(self) -> None:
        self.respawn_calls: list[dict] = []

    async def respawn(self, safety_block: dict) -> None:
        self.respawn_calls.append(safety_block)


class _SlowRecordingToolHost:
    """Like `_RecordingToolHost`, but `respawn` takes a moment -- proves
    the route awaits it before returning rather than firing a detached
    task and answering early."""

    def __init__(self) -> None:
        self.respawn_calls: list[dict] = []
        self.respawn_completed = False

    async def respawn(self, safety_block: dict) -> None:
        await asyncio.sleep(0.05)
        self.respawn_calls.append(safety_block)
        self.respawn_completed = True


class _FailingToolHost:
    """`respawn` always raises -- the write already committed by the time
    this runs (`routes/policy.py`'s own docstring), so the route must
    report a named, retryable failure rather than a bare 500."""

    async def respawn(self, safety_block: dict) -> None:
        raise RuntimeError("simulated respawn failure")


def test_adding_a_rule_respawns_the_tool_child_with_the_new_policy(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()
    tool_host = _RecordingToolHost()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_policy_app(security, account_repo, policy_repo, tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/policy/rules",
        json={"kind": "deny_entity", "value": "switch.example_new_denied", "note": None},
    )
    assert response.status_code == 201, response.text

    assert len(tool_host.respawn_calls) == 1, (
        "adding a rule must respawn the tool child exactly once, not zero and not twice"
    )
    block = tool_host.respawn_calls[0]
    assert "switch.example_new_denied" in block["deny_entities"], (
        "the block the child was respawned with does not contain the new rule -- "
        f"got {block!r}"
    )


def test_the_route_awaits_respawn_before_returning(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()
    tool_host = _SlowRecordingToolHost()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_policy_app(security, account_repo, policy_repo, tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/policy/rules",
        json={"kind": "deny_entity", "value": "switch.example_awaited", "note": None},
    )
    assert response.status_code == 201, response.text
    assert tool_host.respawn_completed is True, (
        "the response returned before the slow respawn finished -- the route must "
        "await respawn, not fire it and answer early"
    )


def test_removing_a_rule_respawns_the_tool_child_with_the_updated_policy(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository(deny_entities=["switch.example_to_remove"])
    tool_host = _RecordingToolHost()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_policy_app(security, account_repo, policy_repo, tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    rule_id = policy_repo.rules[0].id
    response = client.delete(f"/api/policy/rules/{rule_id}")
    assert response.status_code == 204, response.text

    assert len(tool_host.respawn_calls) == 1
    block = tool_host.respawn_calls[0]
    assert "switch.example_to_remove" not in block["deny_entities"], (
        "the removed rule is still present in the block the child was respawned with"
    )


def test_switching_mode_writes_an_audit_row_naming_who_and_when(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    """Changing `mode` between `allow_all_except_denylist` and
    `allowlist_only` must write an audit row naming the admin and the
    timestamp (D-13)."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository(mode="allow_all_except_denylist")
    tool_host = _RecordingToolHost()

    admin = _issue_cookie(security, account_repo, role="admin")
    token = issue_access_token(user_id=admin.id, role="admin", security=security)

    app = _build_policy_app(security, account_repo, policy_repo, tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.put("/api/policy/mode", json={"mode": "allowlist_only"})
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "allowlist_only"

    assert len(policy_repo.audit_log) == 1, (
        f"expected exactly one audit row, found {len(policy_repo.audit_log)}: "
        f"{policy_repo.audit_log!r}"
    )
    entry = policy_repo.audit_log[0]
    assert entry["actor_user_id"] == admin.id
    assert entry["detail"]["old_mode"] == "allow_all_except_denylist"
    assert entry["detail"]["new_mode"] == "allowlist_only"


def test_an_unknown_mode_value_is_refused(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()
    tool_host = _RecordingToolHost()

    admin = _issue_cookie(security, account_repo, role="admin")
    token = issue_access_token(user_id=admin.id, role="admin", security=security)

    app = _build_policy_app(security, account_repo, policy_repo, tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.put("/api/policy/mode", json={"mode": "permissive"})
    assert response.status_code == 400
    assert "permissive" in response.json()["detail"]
    assert not tool_host.respawn_calls, "an invalid mode must never reach a respawn"
    assert not policy_repo.audit_log, "an invalid mode must never write an audit row"


def test_an_invalid_entity_value_is_refused_on_write(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    """An entity value the boundary's own matcher (`safety.allow_read`)
    would not accept is refused before it is ever stored."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()
    tool_host = _RecordingToolHost()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_policy_app(security, account_repo, policy_repo, tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/policy/rules",
        json={"kind": "deny_entity", "value": "not an entity id", "note": None},
    )
    assert response.status_code == 400
    assert not tool_host.respawn_calls, "an invalid rule must never reach a respawn"
    assert not policy_repo.rules, "an invalid rule must never be stored"


def test_rule_reads_and_edits_require_an_operator(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()
    tool_host = _RecordingToolHost()

    viewer = _issue_cookie(security, account_repo, role="viewer")
    token = issue_access_token(user_id=viewer.id, role="viewer", security=security)

    app = _build_policy_app(security, account_repo, policy_repo, tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    assert client.get("/api/policy").status_code == 403
    assert (
        client.post(
            "/api/policy/rules",
            json={"kind": "deny_entity", "value": "switch.example_viewer_attempt"},
        ).status_code
        == 403
    )
    assert not tool_host.respawn_calls


def test_the_mode_switch_requires_an_admin(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()
    tool_host = _RecordingToolHost()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_policy_app(security, account_repo, policy_repo, tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.put("/api/policy/mode", json={"mode": "allowlist_only"})
    assert response.status_code == 403
    assert not tool_host.respawn_calls
    assert not policy_repo.audit_log


def test_a_respawn_failure_produces_a_named_error_telling_the_operator_to_retry(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()
    tool_host = _FailingToolHost()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_policy_app(security, account_repo, policy_repo, tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/policy/rules",
        json={"kind": "deny_entity", "value": "switch.example_respawn_fails", "note": None},
    )
    assert response.status_code == 502
    assert "retry" in response.json()["detail"].lower()
    # The row is already committed by the time respawn was attempted --
    # the route's own docstring states this, and the test proves it.
    assert any(r.value == "switch.example_respawn_fails" for r in policy_repo.rules)


async def test_a_denylist_rule_added_through_the_route_is_refused_end_to_end_by_the_real_child(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    """The same end-to-end shape plan 03-02's tracer proved
    (`tests/test_policy_repo.py`), now driven from the route an operator
    actually uses: add a denylist rule through `POST /api/policy/rules`,
    then make a real service call through the real spawned child and
    confirm it is refused.

    Uses `httpx.AsyncClient` over `httpx.ASGITransport` rather than
    `fastapi.testclient.TestClient` -- `TestClient` is a *synchronous*
    wrapper that runs the ASGI app on its own thread's own event loop (an
    `anyio` blocking portal). The real `McpToolHost` this test spawns
    binds its subprocess streams to *this* test's own event loop
    (`asyncio_mode = auto`); handing that host to a route running on
    `TestClient`'s separate loop deadlocks the awaited `respawn()` call,
    since the underlying anyio stream objects are not safe to use from a
    second event loop. `ASGITransport` runs the app in-process on the
    caller's own loop, exactly like every other call this test makes.
    """
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()

    operator = await account_repo.create_user(
        email="operator@example.invalid",
        display_name="An Operator",
        password_hash="not-checked-by-this-test",
        role="operator",
    )
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    denied_entity = "switch.example_route_denied_entity"

    from spire_voice.policy_snapshot import safety_block_from_policy

    initial_block = safety_block_from_policy(await policy_repo.load_policy())

    host = McpToolHost()
    try:
        # "test-key" is this repository's one allowlisted credential-shaped
        # placeholder (tests/test_repo_hygiene.py).
        await host.start(
            ha_url="http://ha.invalid:8123",
            ha_token="test-key",
            mcp_root=MCP_ROOT,
            safety_block=initial_block,
        )

        app = _build_policy_app(security, account_repo, policy_repo, host)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            cookies={security.cookie_name: token},
        ) as client:
            response = await client.post(
                "/api/policy/rules",
                json={"kind": "deny_entity", "value": denied_entity, "note": None},
            )
            assert response.status_code == 201, response.text

        result = await host.call_tool(
            "ha_call_service",
            {"domain": "switch", "service": "turn_off", "entity_id": denied_entity},
        )
    finally:
        await host.aclose()

    assert result.is_error, (
        "a rule added through the route was not enforced by the real child process"
    )
    assert "off limits" in result.content[0].text, (
        f"the refusal text did not match safety.py's own wording: {result.content[0].text!r}"
    )
