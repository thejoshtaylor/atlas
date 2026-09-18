"""Workflow authoring over HTTP (FLOW-09, FLOW-10): the routes give an
operator a control surface over the runs plan 05-02 moved into the
database -- list, read, author, edit, and cancel -- with the exact same
read-only safety-conflict annotation `routes/macros.py` already reads from
`routes/conflict.py` (never a second copy of that check).

Every test here builds a small, throwaway `FastAPI()` app carrying only
`spire_voice.routes.workflows`'s own router, the same "primitives in
isolation" shape `tests/test_macro_routes.py` already uses.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig
from spire_voice.db.repository import WorkflowStepSpec
from spire_voice.providers.tts_xai import SinkFormat
from spire_voice.routes.workflows import router as workflows_router

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


def _build_workflow_app(
    security,
    account_repo,
    workflow_repo,
    *,
    policy_repo=None,
    tool_host=None,
    tts=None,
    filler_cache=None,
    cache_dir: Path | None = None,
    server_timezone: ZoneInfo | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.config = SimpleNamespace(
        security=security,
        tts=SimpleNamespace(
            cache_dir=str(cache_dir) if cache_dir is not None else "/tmp/spire-test-workflow-tts-cache",
            voice_id="eve",
        ),
    )
    app.state.account_repo = account_repo
    app.state.workflow_repo = workflow_repo
    app.state.policy_repo = policy_repo
    app.state.tool_host = tool_host
    app.state.tts = tts
    app.state.filler_cache = filler_cache if filler_cache is not None else {}
    app.state.server_timezone = server_timezone
    app.include_router(workflows_router)
    return app


async def _create_user(account_repo, *, role: str):
    return await account_repo.create_user(
        email=f"{role}@example.invalid",
        display_name=f"A {role.title()}",
        password_hash="not-checked-by-this-test",
        role=role,
    )


def _issue_cookie(security, account_repo, *, role: str):
    import asyncio

    return asyncio.run(_create_user(account_repo, role=role))


class _EntityListingToolHost:
    """A fake `tool_host` whose `ha_list_entities` reports a fixed live
    catalog -- the shape `routes/conflict.py::known_entity_ids` reads."""

    def __init__(self, known_entity_ids: list[str]) -> None:
        self._known_entity_ids = known_entity_ids
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append(name)
        assert name == "ha_list_entities"
        return SimpleNamespace(
            structuredContent=[{"entity_id": e, "friendly_name": e} for e in self._known_entity_ids],
            content=[],
        )


class _UnreachableToolHost:
    """`call_tool` always raises -- simulates Home Assistant being
    unreachable, so `known_entity_ids` must return `None`, not an empty
    catalog."""

    async def call_tool(self, name: str, arguments: dict):
        raise RuntimeError("simulated home assistant unreachable")


class _RaisingPolicyRepository:
    """`load_policy` always raises -- simulates a policy read failure
    independent of the catalog, so `load_policy_or_none` must return
    `None`."""

    async def load_policy(self):
        raise RuntimeError("simulated policy read failure")


class _FakeWorkflowTts:
    """A `precache_all`-compatible fake: `browser_sink()` plus a
    `synthesize(text_deltas, sink=...)` that records the text it was asked
    to render and either succeeds with scripted bytes or raises."""

    def __init__(self, chunks: tuple[bytes, ...] = (b"synthesized-audio",), fail: bool = False) -> None:
        self._chunks = chunks
        self._fail = fail
        self.synthesized_texts: list[str] = []

    def browser_sink(self) -> SinkFormat:
        return SinkFormat(codec="pcm", sample_rate=24000)

    async def synthesize(self, text_deltas, sink=None):
        text = "".join([delta async for delta in text_deltas])
        self.synthesized_texts.append(text)
        if self._fail:
            raise RuntimeError("simulated synthesis failure")
        for chunk in self._chunks:
            yield chunk


def _call_service_step(entity_id: str = "switch.example_fan", **extra_arguments) -> dict:
    return {
        "kind": "call_service",
        "arguments": {
            "domain": "switch",
            "service": "turn_off",
            "entity_id": entity_id,
            **extra_arguments,
        },
    }


def _wait_step(duration_s: float = 60) -> dict:
    return {"kind": "wait", "arguments": {"duration_s": duration_s}}


def _speak_step(text: str = "the lights are off") -> dict:
    return {"kind": "speak", "arguments": {"text": text}}


def _future_run_at(now: datetime, *, minutes: int = 30) -> str:
    """An ISO-8601 string carrying an explicit UTC offset (rule 3,
    `resolve_schedule`'s own docstring) -- used by every test in this file
    that is not itself testing the zone-less PA-03 form, so those tests
    need no `server_timezone` wired into the app to pass."""
    return (now + timedelta(minutes=minutes)).isoformat()


# --- Task 2: the five routes, refused by name -----------------------------


def test_reading_and_listing_reject_an_unauthenticated_request_and_a_viewer(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    anonymous_client = TestClient(app)
    assert anonymous_client.get("/api/workflows").status_code == 401

    viewer = _issue_cookie(security, account_repo, role="viewer")
    token = issue_access_token(user_id=viewer.id, role="viewer", security=security)
    viewer_client = TestClient(app, cookies={security.cookie_name: token})
    assert viewer_client.get("/api/workflows").status_code == 403


def test_every_workflow_route_requires_operator(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    """Every route -- reads included -- is gated (T-05-19)."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    viewer = _issue_cookie(security, account_repo, role="viewer")
    token = issue_access_token(user_id=viewer.id, role="viewer", security=security)
    client = TestClient(app, cookies={security.cookie_name: token})

    assert client.get("/api/workflows").status_code == 403
    assert client.get("/api/workflows/1").status_code == 403
    assert client.post("/api/workflows", json={"summary": "x", "run_at": "2027-01-01T00:00:00Z", "steps": []}).status_code == 403
    assert client.put("/api/workflows/1", json={"steps": []}).status_code == 403
    assert client.post("/api/workflows/1/cancel").status_code == 403


async def test_listing_returns_both_origins_marked_and_only_pending_or_firing(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    base_time = datetime.now(timezone.utc) + timedelta(hours=1)
    voice_run = await workflow_repo.create_run(
        origin="voice",
        summary="spoken run",
        steps=[WorkflowStepSpec(kind="wait", arguments={"duration_s": 5})],
        base_time=base_time,
        created_by_user_id=None,
    )
    webapp_run = await workflow_repo.create_run(
        origin="webapp",
        summary="authored run",
        steps=[WorkflowStepSpec(kind="wait", arguments={"duration_s": 5})],
        base_time=base_time,
        created_by_user_id=1,
    )
    # A completed run must not appear in the pending list at all.
    already_ran = await workflow_repo.create_run(
        origin="voice",
        summary="a run that already ran",
        steps=[WorkflowStepSpec(kind="wait", arguments={"duration_s": 5})],
        base_time=base_time,
        created_by_user_id=None,
    )
    workflow_repo._runs[already_ran.id]["status"] = "completed"  # type: ignore[index]

    operator = await _create_user(account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get("/api/workflows")
    assert response.status_code == 200, response.text
    body = response.json()
    origins = {row["id"]: row["origin"] for row in body}
    assert origins == {voice_run.id: "voice", webapp_run.id: "webapp"}
    assert all(row["step_count"] == 1 for row in body)


def test_reading_an_unknown_run_id_returns_a_named_404(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get("/api/workflows/999")
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


async def test_a_call_service_step_targeting_a_denied_entity_is_annotated_denied(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository(deny_entities=["switch.example_denied"])
    tool_host = _EntityListingToolHost(["switch.example_denied"])


    base_time = datetime.now(timezone.utc) + timedelta(hours=1)
    run = await workflow_repo.create_run(
        origin="webapp",
        summary="turn off the denied switch",
        steps=[
            WorkflowStepSpec(
                kind="call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_denied"},
            )
        ],
        base_time=base_time,
        created_by_user_id=1,
    )

    operator = await _create_user(account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(
        security, account_repo, workflow_repo, policy_repo=policy_repo, tool_host=tool_host
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get(f"/api/workflows/{run.id}")
    assert response.status_code == 200, response.text
    assert response.json()["steps"][0]["conflict"] == "denied"


async def test_the_unknown_annotation_is_returned_when_the_catalog_cannot_be_read_and_differs_from_no_conflict(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()
    tool_host = _UnreachableToolHost()


    base_time = datetime.now(timezone.utc) + timedelta(hours=1)
    run = await workflow_repo.create_run(
        origin="webapp",
        summary="turn off a switch",
        steps=[
            WorkflowStepSpec(
                kind="call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_a"},
            )
        ],
        base_time=base_time,
        created_by_user_id=1,
    )

    operator = await _create_user(account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(
        security, account_repo, workflow_repo, policy_repo=policy_repo, tool_host=tool_host
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get(f"/api/workflows/{run.id}")
    assert response.status_code == 200, response.text
    conflict = response.json()["steps"][0]["conflict"]
    assert conflict == "unknown"
    assert conflict != "ok"


async def test_the_unknown_annotation_is_returned_when_the_policy_cannot_be_read(
    monkeypatch, fake_account_repository, fake_workflow_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = _RaisingPolicyRepository()
    tool_host = _EntityListingToolHost(["switch.example_a"])


    base_time = datetime.now(timezone.utc) + timedelta(hours=1)
    run = await workflow_repo.create_run(
        origin="webapp",
        summary="turn off a switch",
        steps=[
            WorkflowStepSpec(
                kind="call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_a"},
            )
        ],
        base_time=base_time,
        created_by_user_id=1,
    )

    operator = await _create_user(account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(
        security, account_repo, workflow_repo, policy_repo=policy_repo, tool_host=tool_host
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get(f"/api/workflows/{run.id}")
    assert response.status_code == 200, response.text
    assert response.json()["steps"][0]["conflict"] == "unknown"


# --- POST /api/workflows: creation, validation, scheduling -----------------


def test_creating_a_workflow_with_a_summary_run_at_and_steps_returns_it_stored_in_order(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={
            "summary": "turn off the fan in ten minutes",
            "run_at": _future_run_at(now, minutes=30),
            "steps": [_wait_step(60), _call_service_step()],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["origin"] == "webapp"
    assert body["status"] == "pending"
    assert [s["kind"] for s in body["steps"]] == ["wait", "call_service"]
    assert body["created_by_user_id"] == operator.id


def test_creating_a_workflow_with_no_steps_is_refused(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={"summary": "nothing to do", "run_at": _future_run_at(now), "steps": []},
    )
    assert response.status_code == 400
    assert len(workflow_repo._runs) == 0  # type: ignore[attr-defined]


def test_creating_a_workflow_with_a_negative_wait_duration_is_refused(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={"summary": "x", "run_at": _future_run_at(now), "steps": [_wait_step(-5)]},
    )
    assert response.status_code == 400


def test_creating_a_call_service_step_with_no_domain_or_service_is_refused(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={
            "summary": "x",
            "run_at": _future_run_at(now),
            "steps": [{"kind": "call_service", "arguments": {"entity_id": "switch.example_fan"}}],
        },
    )
    assert response.status_code == 400


def test_a_transition_on_a_non_light_domain_is_refused_at_authoring_time(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={
            "summary": "x",
            "run_at": _future_run_at(now),
            "steps": [_call_service_step(entity_id="switch.example_fan", transition=5)],
        },
    )
    assert response.status_code == 400
    assert "light" in response.json()["detail"]


def test_creating_a_speak_step_with_blank_words_is_refused(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={"summary": "x", "run_at": _future_run_at(now), "steps": [_speak_step("")]},
    )
    assert response.status_code == 400


def test_a_schedule_time_in_the_past_is_refused(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={
            "summary": "x",
            "run_at": _future_run_at(now, minutes=-30),
            "steps": [_wait_step(5)],
        },
    )
    assert response.status_code == 400
    assert "future" in response.json()["detail"]


def test_a_zoneless_run_at_is_resolved_against_the_servers_own_configured_zone(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    """PA-03/D-04: the browser's `datetime-local` string carries no offset
    -- this must resolve against `app.state.server_timezone`, never the
    test process's own local zone."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    zone = ZoneInfo("America/New_York")
    app = _build_workflow_app(
        security, account_repo, workflow_repo, policy_repo=policy_repo, server_timezone=zone
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/workflows",
        json={
            "summary": "x",
            "run_at": "2027-06-01T19:30:00",  # zone-less, resolved against `zone` above
            "steps": [_wait_step(5)],
        },
    )
    assert response.status_code == 201, response.text
    run_id = response.json()["id"]
    run = workflow_repo._runs[run_id]  # type: ignore[index]
    # 19:30 America/New_York in June is EDT (-04:00) -> 23:30 UTC.
    first_step = next(s for s in workflow_repo._steps.values() if s["run_id"] == run_id)  # type: ignore[attr-defined]
    assert first_step["due_at"] == datetime(2027, 6, 1, 23, 30, tzinfo=timezone.utc)


def test_an_ambiguous_run_at_is_refused_with_its_own_named_error(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    zone = ZoneInfo("America/New_York")
    app = _build_workflow_app(
        security, account_repo, workflow_repo, policy_repo=policy_repo, server_timezone=zone
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/workflows",
        json={"summary": "x", "run_at": "2027-11-07T01:30:00", "steps": [_wait_step(5)]},
    )
    assert response.status_code == 400
    assert "ambiguous" in response.json()["detail"]


def test_a_nonexistent_run_at_is_refused_with_its_own_named_error(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    zone = ZoneInfo("America/New_York")
    app = _build_workflow_app(
        security, account_repo, workflow_repo, policy_repo=policy_repo, server_timezone=zone
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/workflows",
        json={"summary": "x", "run_at": "2027-03-14T02:30:00", "steps": [_wait_step(5)]},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "does not exist" in detail or "skip" in detail
    # Distinguishable from the ambiguous-time refusal's own wording.
    assert "ambiguous" not in detail


# --- PUT /api/workflows/{run_id}: replace steps, refused once firing -------


async def test_replacing_steps_on_a_pending_run_persists_the_new_list(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()


    base_time = datetime.now(timezone.utc) + timedelta(hours=1)
    run = await workflow_repo.create_run(
        origin="webapp",
        summary="original",
        steps=[WorkflowStepSpec(kind="wait", arguments={"duration_s": 5})],
        base_time=base_time,
        created_by_user_id=1,
    )

    operator = await _create_user(account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.put(
        f"/api/workflows/{run.id}",
        json={"steps": [_call_service_step(entity_id="switch.example_replaced")]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["kind"] for s in body["steps"]] == ["call_service"]
    assert body["steps"][0]["arguments"]["entity_id"] == "switch.example_replaced"


def test_replacing_steps_on_an_unknown_run_returns_a_named_404(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.put("/api/workflows/999", json={"steps": [_wait_step(5)]})
    assert response.status_code == 404


async def test_replacing_steps_on_a_firing_run_is_refused_distinguishably_from_a_404(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()


    base_time = datetime.now(timezone.utc) + timedelta(hours=1)
    run = await workflow_repo.create_run(
        origin="webapp",
        summary="already firing",
        steps=[WorkflowStepSpec(kind="wait", arguments={"duration_s": 5})],
        base_time=base_time,
        created_by_user_id=1,
    )
    workflow_repo._runs[run.id]["status"] = "firing"  # type: ignore[index]

    operator = await _create_user(account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.put(
        f"/api/workflows/{run.id}", json={"steps": [_call_service_step()]}
    )
    assert response.status_code != 404
    assert response.status_code == 409


# --- POST /api/workflows/{run_id}/cancel: never deletes --------------------


async def test_cancelling_a_pending_run_moves_it_to_a_terminal_status_and_keeps_the_row(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()


    base_time = datetime.now(timezone.utc) + timedelta(hours=1)
    run = await workflow_repo.create_run(
        origin="webapp",
        summary="to be cancelled",
        steps=[WorkflowStepSpec(kind="wait", arguments={"duration_s": 5})],
        base_time=base_time,
        created_by_user_id=1,
    )

    operator = await _create_user(account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(f"/api/workflows/{run.id}/cancel")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    # Never deleted: the row is still readable afterward (D-12).
    read_back = client.get(f"/api/workflows/{run.id}")
    assert read_back.status_code == 200
    assert read_back.json()["status"] == "cancelled"


async def test_cancelling_an_unknown_run_and_cancelling_it_twice_answer_differently(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()


    base_time = datetime.now(timezone.utc) + timedelta(hours=1)
    run = await workflow_repo.create_run(
        origin="webapp",
        summary="to be cancelled twice",
        steps=[WorkflowStepSpec(kind="wait", arguments={"duration_s": 5})],
        base_time=base_time,
        created_by_user_id=1,
    )

    operator = await _create_user(account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    not_found = client.post("/api/workflows/999/cancel")
    assert not_found.status_code == 404

    first = client.post(f"/api/workflows/{run.id}/cancel")
    assert first.status_code == 200

    second = client.post(f"/api/workflows/{run.id}/cancel")
    assert second.status_code == 409
    assert second.status_code != not_found.status_code


async def test_cancelling_a_run_that_moved_on_between_the_pre_read_and_the_cancel_names_the_current_status(
    monkeypatch, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    """WR-02 (code review): the 409 body must name the run's status at the
    moment the response is composed, not a snapshot read before
    `cancel_run` ran. Simulated here by a `cancel_run` stand-in that moves
    the run to a *different* terminal status than the one `existing`
    (read before this call) saw -- exactly what a concurrent poller
    finishing the run between those two reads would do."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()

    base_time = datetime.now(timezone.utc) + timedelta(hours=1)
    run = await workflow_repo.create_run(
        origin="webapp",
        summary="races the poller",
        steps=[WorkflowStepSpec(kind="wait", arguments={"duration_s": 5})],
        base_time=base_time,
        created_by_user_id=1,
    )
    # The pre-read `cancel_workflow` does before ever calling `cancel_run`
    # will see this run as "firing".
    workflow_repo._runs[run.id]["status"] = "firing"  # type: ignore[attr-defined]

    async def _cancel_run_that_races_a_concurrent_poller(run_id, *, now, cancelled_by_user_id):
        # By the time this (stand-in for `cancel_run`'s own atomic check)
        # runs, a concurrent poller has already moved the run all the way
        # to "completed" -- a status `existing` (read earlier, above)
        # never saw.
        workflow_repo._runs[run_id]["status"] = "completed"  # type: ignore[attr-defined]
        return False

    workflow_repo.cancel_run = _cancel_run_that_races_a_concurrent_poller

    operator = await _create_user(account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(security, account_repo, workflow_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(f"/api/workflows/{run.id}/cancel")

    assert response.status_code == 409
    assert "completed" in response.json()["detail"]
    assert "firing" not in response.json()["detail"]


# --- Task 3: a speak step's words, prepared while someone is still there --


def test_a_saved_speak_step_is_synthesized_before_the_response_and_reports_ready(
    monkeypatch, tmp_path, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()
    tts = _FakeWorkflowTts()
    filler_cache: dict = {}

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(
        security,
        account_repo,
        workflow_repo,
        policy_repo=policy_repo,
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={
            "summary": "say something",
            "run_at": _future_run_at(now),
            "steps": [_speak_step("the lights are off now")],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["reply_synthesis_degraded"] is False
    assert body["steps"][0]["speak_cached"] is True
    assert "the lights are off now" in filler_cache
    assert tts.synthesized_texts == ["the lights are off now"]


def test_a_synthesis_failure_returns_a_degraded_success_and_the_run_still_commits(
    monkeypatch, tmp_path, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()
    tts = _FakeWorkflowTts(fail=True)
    filler_cache: dict = {}

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(
        security,
        account_repo,
        workflow_repo,
        policy_repo=policy_repo,
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={
            "summary": "say something that fails",
            "run_at": _future_run_at(now),
            "steps": [_speak_step("this will fail to synthesize")],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["reply_synthesis_degraded"] is True
    assert body["reply_synthesis_message"]
    assert body["steps"][0]["speak_cached"] is False
    assert "this will fail to synthesize" not in filler_cache

    # The run is still committed and readable, not lost to the failure.
    run_id = body["id"]
    read_back = client.get(f"/api/workflows/{run_id}")
    assert read_back.status_code == 200
    assert read_back.json()["steps"][0]["arguments"]["text"] == "this will fail to synthesize"
    # A read never re-synthesizes -- the degraded flag is only meaningful
    # on the save response itself.
    assert read_back.json()["reply_synthesis_degraded"] is False


def test_a_second_save_of_the_same_words_does_not_resynthesize(
    monkeypatch, tmp_path, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()
    tts = _FakeWorkflowTts()
    filler_cache: dict = {}

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(
        security,
        account_repo,
        workflow_repo,
        policy_repo=policy_repo,
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    first = client.post(
        "/api/workflows",
        json={
            "summary": "say the same thing",
            "run_at": _future_run_at(now),
            "steps": [_speak_step("unchanging words")],
        },
    )
    assert first.status_code == 201, first.text
    run_id = first.json()["id"]
    assert tts.synthesized_texts == ["unchanging words"]

    second = client.put(
        f"/api/workflows/{run_id}",
        json={"steps": [_speak_step("unchanging words")]},
    )
    assert second.status_code == 200, second.text
    assert tts.synthesized_texts == ["unchanging words"], "an unchanged text must never re-synthesize"


def test_a_second_save_after_a_failed_synthesis_synthesizes_again(
    monkeypatch, tmp_path, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()
    tts = _FakeWorkflowTts(fail=True)
    filler_cache: dict = {}

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(
        security,
        account_repo,
        workflow_repo,
        policy_repo=policy_repo,
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    first = client.post(
        "/api/workflows",
        json={
            "summary": "will fail then retry",
            "run_at": _future_run_at(now),
            "steps": [_speak_step("retry me")],
        },
    )
    assert first.status_code == 201, first.text
    assert first.json()["reply_synthesis_degraded"] is True
    run_id = first.json()["id"]
    assert tts.synthesized_texts == ["retry me"]

    tts._fail = False  # the underlying provider recovers before the next save

    second = client.put(
        f"/api/workflows/{run_id}",
        json={"steps": [_speak_step("retry me")]},
    )
    assert second.status_code == 200, second.text
    assert second.json()["reply_synthesis_degraded"] is False
    assert second.json()["steps"][0]["speak_cached"] is True
    assert tts.synthesized_texts == ["retry me", "retry me"], "a failed synthesis must retry on the next save"


def test_a_run_with_no_speak_step_never_calls_the_synthesizer(
    monkeypatch, tmp_path, fake_account_repository, fake_workflow_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    workflow_repo = fake_workflow_repository()
    policy_repo = fake_policy_repository()
    tts = _FakeWorkflowTts()
    filler_cache: dict = {}

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_workflow_app(
        security,
        account_repo,
        workflow_repo,
        policy_repo=policy_repo,
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    now = datetime.now(timezone.utc)
    response = client.post(
        "/api/workflows",
        json={
            "summary": "no speaking here",
            "run_at": _future_run_at(now),
            "steps": [_wait_step(5), _call_service_step()],
        },
    )
    assert response.status_code == 201, response.text
    assert tts.synthesized_texts == []
    assert response.json()["reply_synthesis_degraded"] is False
