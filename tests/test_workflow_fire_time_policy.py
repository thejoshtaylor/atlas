"""SAFE-08's own evidence: a scheduled step re-checks the policy at the
moment it fires, not only when it was created -- and the check that
matters is the one the *enforcing process* has heard, not the one the
database holds.

Deliberately not an integration test, and deliberately not built on
`tests/conftest.py`'s `FakeWorkflowRepository` or a real Postgres. Two
reasons: this file's subject is policy freshness across a process
boundary, not durable storage, which has its own tests in plan 05-02;
and a test that skips whenever no database happens to be up is the wrong
place to keep this project's single most consequential safety claim --
it runs on every commit. `_StubWorkflowRepository` below is a small,
in-memory `WorkflowRepository`, built for this file alone, the same
in-file-stub convention `tests/test_policy_routes.py` already uses for
its own `_RecordingToolHost`.

The MCP child that enforces `allow_call` holds a policy *snapshot*,
captured once at spawn from `SPIRE_SAFETY` and refreshed only by
`respawn()` (`mcp/spire_mcp/ha.py::_load_policy`). SAFE-08's "re-checks
the policy when it fires" is therefore only as current as the last
respawn -- so the policy change below goes through the real route
(`POST /api/policy/rules`, `routes/policy.py`) against a real,
spawned `McpToolHost`, exactly the way `tests/test_policy_routes.py`'s
own
`test_a_denylist_rule_added_through_the_route_is_refused_end_to_end_by_the_real_child`
proves the live-turn side of this same seam. Writing to the policy
repository directly, or constructing a `Policy` object, would prove the
database is live -- it proves nothing about whether the *enforcing
process* heard about the change, which is the only thing SAFE-08 claims
(04-REVIEW.md's CR-01: the writer-vs-writer respawn race this
requirement's correctness now depends on transitively).

`httpx.ASGITransport`, never FastAPI's synchronous test-client wrapper:
that wrapper runs the ASGI app on its own thread's own event loop (an
`anyio` blocking portal), and the real `McpToolHost` this file spawns
binds its subprocess streams to *this* test's own event loop -- handing
that host to a route running on the wrapper's separate loop deadlocks the
awaited `respawn()` call. `ASGITransport` runs the app in-process on the
caller's own loop, exactly like every other call this file makes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI

from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig, WorkflowConfig
from spire_voice.db.models import WorkflowRunRow, WorkflowStepRow
from spire_voice.db.repository import WorkflowRun, WorkflowStep, WorkflowStepSpec, assign_step_due_ats, push_out_due_at
from spire_voice.mcp_client import McpToolHost
from spire_voice.policy_snapshot import safety_block_from_policy
from spire_voice.routes.policy import router as policy_router
from spire_voice.workflow.scheduler import WorkflowScheduler
from spire_voice.workflow.steps import execute_step

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MCP_ROOT = REPO_ROOT / "mcp"

# RFC 2606-reserved, guaranteed-unreachable -- the same convention
# `tests/test_policy_routes.py`'s own end-to-end respawn test uses, and
# for the identical reason: a real call must actually be attempted (never
# mocked), and it must never reach a real network.
_HA_URL = "http://ha.invalid:8123"
_DENIED_ENTITY = "switch.example_workflow_fire_time_denied_entity"
_WORKFLOW_CLAIMABLE_RUN_STATUSES = ("pending", "firing")


def _naive_utc(dt: datetime) -> datetime:
    """The same naive-UTC storage convention `db/postgres.py`'s own
    `_to_naive_utc` establishes for every `due_at`/`fired_at` column --
    duplicated here, in three lines, rather than imported across the
    `db.postgres` module boundary that module's own acceptance criteria
    keep deliberately private."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None)


class _StubWorkflowRepository:
    """An in-memory `WorkflowRepository` (`spire_voice.db.repository`'s
    own Protocol), built for this file alone -- see the module docstring
    for why this is not `tests/conftest.py`'s `FakeWorkflowRepository`.

    Mirrors `PostgresWorkflowRepository`'s own claim-and-complete-in-one-
    step shape (there is no real transaction to share here, so "one
    transaction" is simply "no `await` between claim and write" -- the
    same property, without a database to prove it against) closely enough
    that a test exercising this stub exercises the same contract
    `execute_step`'s own callers depend on, never a shortcut around it.
    """

    def __init__(self, *, max_attempts: int = 3, retry_backoff_s: float = 30.0) -> None:
        self._runs: dict[int, WorkflowRunRow] = {}
        self._steps: dict[int, WorkflowStepRow] = {}
        self._next_run_id = 1
        self._next_step_id = 1
        self._max_attempts = max_attempts
        self._retry_backoff_s = retry_backoff_s

    async def create_run(
        self,
        *,
        origin: str,
        summary: str,
        steps: "list[WorkflowStepSpec] | tuple[WorkflowStepSpec, ...]",
        base_time: datetime,
        created_by_user_id: int | None,
    ) -> WorkflowRun:
        now = _naive_utc(datetime.now(timezone.utc))
        due_ats = assign_step_due_ats(steps, base_time)
        run_id = self._next_run_id
        self._next_run_id += 1
        run_row = WorkflowRunRow(
            id=run_id,
            origin=origin,
            status="pending",
            summary=summary,
            created_at=now,
            updated_at=now,
            created_by_user_id=created_by_user_id,
        )
        self._runs[run_id] = run_row

        step_rows: list[WorkflowStepRow] = []
        for position, (spec, due_at) in enumerate(zip(steps, due_ats)):
            step_id = self._next_step_id
            self._next_step_id += 1
            step_row = WorkflowStepRow(
                id=step_id,
                run_id=run_id,
                position=position,
                kind=spec.kind,
                arguments=spec.arguments,
                due_at=_naive_utc(due_at),
                status="pending",
                attempts=0,
                result_detail=None,
                fired_at=None,
            )
            self._steps[step_id] = step_row
            step_rows.append(step_row)

        return WorkflowRun(
            id=run_row.id,
            origin=run_row.origin,
            status=run_row.status,
            summary=run_row.summary,
            created_at=now.replace(tzinfo=timezone.utc),
            updated_at=now.replace(tzinfo=timezone.utc),
            created_by_user_id=created_by_user_id,
            steps=tuple(
                WorkflowStep(
                    id=r.id,
                    run_id=r.run_id,
                    position=r.position,
                    kind=r.kind,
                    arguments=r.arguments,
                    due_at=r.due_at.replace(tzinfo=timezone.utc),
                    status=r.status,
                    attempts=r.attempts,
                    result_detail=r.result_detail,
                    fired_at=r.fired_at,
                )
                for r in step_rows
            ),
        )

    async def claim_and_execute_next_due_step(self, executor, now: datetime) -> bool:
        naive_now = _naive_utc(now)
        candidates = [
            step
            for step in self._steps.values()
            if step.status == "pending"
            and step.due_at <= naive_now
            and self._runs[step.run_id].status in _WORKFLOW_CLAIMABLE_RUN_STATUSES
            and not any(
                other.status == "pending"
                for other in self._steps.values()
                if other.run_id == step.run_id and other.position < step.position
            )
        ]
        if not candidates:
            return False
        candidates.sort(key=lambda s: (s.due_at, s.run_id, s.position))
        step = candidates[0]
        run = self._runs[step.run_id]
        if run.status == "pending":
            run.status = "firing"

        outcome = await executor(step)

        step.attempts += 1
        step.result_detail = outcome.detail
        if outcome.retry and step.attempts < self._max_attempts:
            step.status = "pending"
            step.due_at = _naive_utc(push_out_due_at(now, self._retry_backoff_s))
        else:
            step.status = "failed" if outcome.retry else outcome.status
            step.fired_at = naive_now

        remaining_pending = sum(
            1 for s in self._steps.values() if s.run_id == run.id and s.status == "pending"
        )
        if remaining_pending == 0:
            incomplete = sum(
                1 for s in self._steps.values() if s.run_id == run.id and s.status != "completed"
            )
            run.status = "completed" if incomplete == 0 else "failed"
        run.updated_at = naive_now
        return True


class _RecordingSpeak:
    """Records every text `execute_step` was asked to say -- the SAFE-08
    test's own evidence for D-14: the spoken text must equal the
    recorded refusal reason exactly, never a substring match standing in
    for equality."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def __call__(self, text: str) -> None:
        self.spoken.append(text)


def _build_policy_app(security, account_repo, policy_repo, tool_host) -> FastAPI:
    """The same throwaway, router-only `FastAPI()` shape
    `tests/test_policy_routes.py::_build_policy_app` builds -- duplicated
    here rather than imported, since that function lives in a test module
    this project's own convention treats as a leaf, not a shared library
    (no `tests/` module imports another test module's helpers elsewhere
    in this suite)."""
    app = FastAPI()
    app.state.config = type("Config", (), {"security": security})()
    app.state.account_repo = account_repo
    app.state.policy_repo = policy_repo
    app.state.tool_host = tool_host
    app.state.safety_block = None
    app.include_router(policy_router)
    return app


async def _make_run(repo: _StubWorkflowRepository, *, entity_id: str, due_at: datetime) -> WorkflowStepRow:
    """Builds one pending run with one `call_service` step targeting
    `entity_id`, due at `due_at`. Shared by the SAFE-08 test and its own
    control so the policy write is the only difference between them --
    a separately constructed run would leave open which difference
    produced the refusal.

    Returns the step's own `WorkflowStepRow` (`_StubWorkflowRepository`'s
    own storage object, mutated in place by `claim_and_execute_next_due_step`)
    so a test can read `status`/`result_detail` straight off it after
    polling, with no second lookup.
    """
    run = await repo.create_run(
        origin="voice",
        summary=f"turn off {entity_id}",
        steps=[
            WorkflowStepSpec(
                kind="call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": entity_id},
            )
        ],
        base_time=due_at,
        created_by_user_id=None,
    )
    return repo._steps[run.steps[0].id]  # the same object claim_and_execute_next_due_step mutates


async def test_a_run_scheduled_before_the_entity_was_denied_is_refused_when_it_fires(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()  # default mode, no rules -- the target is NOT denied yet

    operator = await account_repo.create_user(
        email="operator@example.invalid",
        display_name="An Operator",
        password_hash="not-checked-by-this-test",
        role="operator",
    )
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    initial_block = safety_block_from_policy(await policy_repo.load_policy())

    host = McpToolHost()
    try:
        await host.start(
            ha_url=_HA_URL,
            ha_token="test-key",  # this repository's one allowlisted credential-shaped placeholder
            mcp_root=MCP_ROOT,
            safety_block=initial_block,
        )

        repo = _StubWorkflowRepository()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        due_at = t0 + timedelta(seconds=30)
        step = await _make_run(repo, entity_id=_DENIED_ENTITY, due_at=due_at)
        assert step.status == "pending"

        # 1. Add a deny rule for the target entity through the real route,
        #    over `httpx.ASGITransport`, so the real `respawn()` runs
        #    (module docstring; RESEARCH.md Pitfall 2). Load-bearing --
        #    writing `policy_repo` directly or handing `host` a
        #    constructed `Policy` would prove the database is live, not
        #    that the enforcing process heard about the change.
        app = _build_policy_app(security, account_repo, policy_repo, host)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", cookies={security.cookie_name: token}
        ) as client:
            response = await client.post(
                "/api/policy/rules",
                json={"kind": "deny_entity", "value": _DENIED_ENTITY, "note": None},
            )
            assert response.status_code == 201, response.text

        # 2. Advance past `due_at` and run one poll -- `WorkflowScheduler`
        #    itself, an injectable clock frozen at a fixed instant, never
        #    a real sleep (this project's own "count, never time it" rule).
        speak = _RecordingSpeak()
        config = WorkflowConfig()
        fire_time = due_at + timedelta(seconds=1)
        executor = lambda step, now: execute_step(step, host, config, now, speak=speak)
        scheduler = WorkflowScheduler(repo, executor, config, clock=lambda: fire_time)
        await scheduler._poll_once()

        # 3. The step was refused, carrying the boundary's own wording
        #    (`Denied`'s own text, `mcp/spire_mcp/safety.py`) -- the same
        #    "off limits" substring `test_policy_routes.py`'s own
        #    end-to-end respawn test asserts against, the installed MCP
        #    SDK's own "Error executing tool {name}: {message}" prefix
        #    included, never silently skipped (D-14). The load-bearing
        #    equality is between what was recorded and what was spoken --
        #    never a substring-containment stand-in for that one.
        assert step.status == "denied"
        assert step.result_detail is not None
        reason = step.result_detail["reason"]
        assert "off limits" in reason
        assert speak.spoken == [reason]
    finally:
        await host.aclose()


async def test_the_control_the_same_run_and_poll_with_no_policy_change_is_not_denied(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    """Without this control, "the step was refused" above passes for a
    bug that refuses every step regardless of policy -- this proves the
    refusal in the test above came from the deny rule, not from anything
    else (the target Home Assistant host is unreachable by design, so
    this step fails for its own, unrelated reason)."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()  # no rules -- and none are ever added in this test

    initial_block = safety_block_from_policy(await policy_repo.load_policy())

    host = McpToolHost()
    try:
        await host.start(
            ha_url=_HA_URL, ha_token="test-key", mcp_root=MCP_ROOT, safety_block=initial_block
        )

        repo = _StubWorkflowRepository()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        due_at = t0 + timedelta(seconds=30)
        step = await _make_run(repo, entity_id=_DENIED_ENTITY, due_at=due_at)

        speak = _RecordingSpeak()
        config = WorkflowConfig()
        fire_time = due_at + timedelta(seconds=1)
        executor = lambda step, now: execute_step(step, host, config, now, speak=speak)
        scheduler = WorkflowScheduler(repo, executor, config, clock=lambda: fire_time)
        await scheduler._poll_once()

        assert step.status != "denied"
        reason = (step.result_detail or {}).get("reason", "")
        assert "off limits" not in reason
    finally:
        await host.aclose()


async def test_a_denied_first_step_does_not_stop_the_runs_second_step_from_firing(
    monkeypatch, fake_account_repository, fake_policy_repository
):
    """PA-D3/PA-D2 (05-01): a denied or failed step never stops the rest
    of its run -- "turn the lights off, then lock the door" must not
    silently drop the door because the light was refused. The second
    step is policy-checked at its own fire time too, the same as the
    first."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()

    operator = await account_repo.create_user(
        email="operator2@example.invalid",
        display_name="Another Operator",
        password_hash="not-checked-by-this-test",
        role="operator",
    )
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    initial_block = safety_block_from_policy(await policy_repo.load_policy())

    host = McpToolHost()
    try:
        await host.start(
            ha_url=_HA_URL, ha_token="test-key", mcp_root=MCP_ROOT, safety_block=initial_block
        )

        repo = _StubWorkflowRepository()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        due_at = t0 + timedelta(seconds=30)
        second_entity = "switch.example_workflow_fire_time_second_entity"
        run = await repo.create_run(
            origin="voice",
            summary="turn off the light, then lock the door",
            steps=[
                WorkflowStepSpec(
                    kind="call_service",
                    arguments={"domain": "switch", "service": "turn_off", "entity_id": _DENIED_ENTITY},
                ),
                WorkflowStepSpec(
                    kind="call_service",
                    arguments={"domain": "switch", "service": "turn_off", "entity_id": second_entity},
                ),
            ],
            base_time=due_at,
            created_by_user_id=None,
        )
        first_step = repo._steps[run.steps[0].id]
        second_step = repo._steps[run.steps[1].id]

        app = _build_policy_app(security, account_repo, policy_repo, host)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", cookies={security.cookie_name: token}
        ) as client:
            response = await client.post(
                "/api/policy/rules",
                json={"kind": "deny_entity", "value": _DENIED_ENTITY, "note": None},
            )
            assert response.status_code == 201, response.text

        speak = _RecordingSpeak()
        # `max_steps_per_poll=1`: both steps are due at the same instant,
        # and `_poll_once()` otherwise drains every due step in one tick
        # (T-05-05) -- capping the drain at one is what makes "the first
        # poll claims the denied step only" an actual two-poll story this
        # test can observe, rather than both steps firing within a single
        # `_poll_once()` call.
        config = WorkflowConfig(max_steps_per_poll=1)
        fire_time = due_at + timedelta(seconds=1)
        executor = lambda step, now: execute_step(step, host, config, now, speak=speak)
        scheduler = WorkflowScheduler(repo, executor, config, clock=lambda: fire_time)

        # Two claimable steps this tick -- position order (D-02's own
        # claim predicate) means the first poll claims the denied step
        # only; the second step is still `pending` after it, due at its
        # own `due_at` (the same instant here), and claimed on the next
        # poll -- never claimed early, and never silently dropped because
        # its sibling was refused.
        await scheduler._poll_once()
        assert first_step.status == "denied"
        assert second_step.status == "pending"

        await scheduler._poll_once()
        assert second_step.status in ("denied", "failed", "completed")
        assert second_step.attempts >= 1
    finally:
        await host.aclose()
