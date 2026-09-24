"""FLOW-08's own evidence: a step runs exactly once even when two workers
poll at the same time, and the losers of that race come back clean.

Copies `tests/test_account_repository_concurrency.py`'s harness shape
exactly (`pytestmark = pytest.mark.integration`, `skip_without_postgres`,
`_migration_url`, `_reset_schema`'s explicit table list, `_run_upgrade_head`
via `asyncio.to_thread`, `asyncio.gather` over N real repository calls) --
determinism here comes from `SELECT ... FOR UPDATE SKIP LOCKED` actually
being exercised by concurrent database transactions, never from timing the
test (05-RESEARCH.md Pitfall 4).

`SKIP LOCKED`'s contract has two halves, and this file asserts both on
every test below, not only the winner count: a caller that finds every due
row already claimed returns `False`, cleanly, with no exception and no
executor invoked. Counting winners alone would let nine raising losers
hide behind one correct winner -- `asyncio.gather(...,
return_exceptions=True)` surfaces exactly that failure mode instead of
swallowing it.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas.db.postgres import PostgresWorkflowRepository
from atlas.db.repository import WorkflowStepSpec
from atlas.workflow.steps import StepOutcome

_TEST_DB_URL = os.environ.get("ATLAS_TEST_DATABASE_URL")

pytestmark = pytest.mark.integration

skip_without_postgres = pytest.mark.skipif(
    _TEST_DB_URL is None,
    reason=(
        "ATLAS_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite, to exercise these tests instead of skipping them"
    ),
)


def _migration_url(async_url: str) -> str:
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
    """The same table list `tests/test_account_repository_concurrency.py`'s
    own `_reset_schema` already carries -- `workflow_steps`/`workflow_runs`
    were added there in plan 05-01, so no further extension is needed here;
    kept local rather than imported, matching every other integration test
    file's own convention in this suite."""
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
        # Phase 6 (plan 06-01 follow-up): drop the whole schema, never a
        # hand-maintained table list. Migration 0008 added `plugins` and
        # `plugin_config_values`, and not one of the nine copies of this
        # helper knew about them -- so a reset dropped `alembic_version` but
        # left those two tables standing, and the very next `upgrade head`
        # died on `DuplicateTable: relation "plugins" already exists`. Every
        # Postgres-backed test after the first one failed that way. A list
        # that has to be edited in nine files each time a migration lands is
        # itself the defect; a schema drop cannot drift out of date.
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


def _run_upgrade_head(async_url: str) -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(async_url))
    command.upgrade(cfg, "head")


def _set_migration_env(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    # Phase 7 (D-15): server.bind_host / security.cookie_secure -- the two
    # new ${VAR} placeholders config.example.yaml expands.
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)


@pytest.fixture
async def workflow_repo(monkeypatch):
    _set_migration_env(monkeypatch)
    await _reset_schema(_TEST_DB_URL)
    await asyncio.to_thread(_run_upgrade_head, _TEST_DB_URL)
    engine = create_async_engine(_TEST_DB_URL)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repo = PostgresWorkflowRepository(sessionmaker)
    yield repo
    await engine.dispose()


class _RecordingExecutor:
    """Records every claimed step id this specific caller's own executor
    was invoked with -- a caller that lost the race must have an empty
    `.calls`, never merely "a smaller number than the winner"."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def __call__(self, step) -> StepOutcome:
        self.calls.append(step.id)
        return StepOutcome(status="completed", detail={}, speech=None, retry=False)


async def _seed_due_run(repo, *, entity_id: str, due_offset_s: float = -60.0):
    """One run, one already-due `call_service` step."""
    base_time = datetime.now(timezone.utc) + timedelta(seconds=due_offset_s)
    return await repo.create_run(
        origin="voice",
        summary=f"example scheduled step for {entity_id}",
        steps=[
            WorkflowStepSpec(
                kind="call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": entity_id},
            )
        ],
        base_time=base_time,
        created_by_user_id=None,
    )


async def _count_completed_steps(async_url: str) -> int:
    engine = create_async_engine(async_url)
    try:
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    text("SELECT count(*) FROM workflow_steps WHERE status = 'completed'")
                )
            ).scalar_one()
    finally:
        await engine.dispose()


def _assert_clean_race(results: list) -> None:
    """The shared assertion every test below runs on its own gathered
    results: no gathered result is a `BaseException` -- a raising loser is
    a failed assertion here, never something `return_exceptions=True`
    quietly absorbs."""
    assert not any(isinstance(r, BaseException) for r in results), (
        f"a caller that lost the claim race raised instead of returning "
        f"False cleanly: {results!r}"
    )


@skip_without_postgres
async def test_ten_concurrent_callers_on_one_due_step_produce_exactly_one_winner(
    workflow_repo,
):
    """One due step, ten concurrent callers: exactly one `True`, exactly
    one executor invoked, and the step's own row shows a terminal,
    completed state -- FLOW-08's whole claim, and the reason D-02 named
    `SELECT ... FOR UPDATE SKIP LOCKED` for this job."""
    run = await _seed_due_run(workflow_repo, entity_id="switch.example_ten_racers")
    now = datetime.now(timezone.utc)
    executors = [_RecordingExecutor() for _ in range(10)]

    results = await asyncio.gather(
        *(workflow_repo.claim_and_execute_next_due_step(ex, now) for ex in executors),
        return_exceptions=True,
    )

    _assert_clean_race(results)
    assert results.count(True) == 1, f"expected exactly one winner, got {results!r}"
    assert results.count(False) == 9, f"expected nine clean losers, got {results!r}"

    called = [ex for ex in executors if ex.calls]
    assert len(called) == 1, "exactly one executor must have been invoked"
    assert called[0].calls == [run.steps[0].id]

    completed = await _count_completed_steps(_TEST_DB_URL)
    assert completed == 1, f"expected exactly one completed step row, found {completed}"


@skip_without_postgres
async def test_two_concurrent_callers_on_two_different_runs_claim_two_different_rows(
    workflow_repo,
):
    """Two due steps in two different runs, two concurrent callers: both
    win, and each claims a *different* step id -- the property that
    distinguishes `SKIP LOCKED` from `pg_advisory_xact_lock` (an advisory
    lock would still produce two eventual winners, but serialized through
    one critical section rather than claiming two rows at once)."""
    run_a = await _seed_due_run(workflow_repo, entity_id="switch.example_two_workers_a")
    run_b = await _seed_due_run(workflow_repo, entity_id="switch.example_two_workers_b")
    now = datetime.now(timezone.utc)
    ex1, ex2 = _RecordingExecutor(), _RecordingExecutor()

    results = await asyncio.gather(
        workflow_repo.claim_and_execute_next_due_step(ex1, now),
        workflow_repo.claim_and_execute_next_due_step(ex2, now),
        return_exceptions=True,
    )

    _assert_clean_race(results)
    assert results == [True, True], f"expected both callers to win their own row: {results!r}"
    claimed_ids = ex1.calls + ex2.calls
    assert len(claimed_ids) == 2
    assert set(claimed_ids) == {run_a.steps[0].id, run_b.steps[0].id}


@skip_without_postgres
async def test_five_concurrent_callers_on_two_due_rows_yield_two_winners_and_three_clean_losers(
    workflow_repo,
):
    """Raising the count on the same shape: five concurrent callers, two
    due rows -- exactly two winners on two different ids, and the other
    three come back `False`, having raised nothing and called nothing."""
    run_a = await _seed_due_run(workflow_repo, entity_id="switch.example_five_racers_a")
    run_b = await _seed_due_run(workflow_repo, entity_id="switch.example_five_racers_b")
    now = datetime.now(timezone.utc)
    executors = [_RecordingExecutor() for _ in range(5)]

    results = await asyncio.gather(
        *(workflow_repo.claim_and_execute_next_due_step(ex, now) for ex in executors),
        return_exceptions=True,
    )

    _assert_clean_race(results)
    assert results.count(True) == 2, f"expected exactly two winners, got {results!r}"
    assert results.count(False) == 3, f"expected three clean losers, got {results!r}"

    called = [ex for ex in executors if ex.calls]
    assert len(called) == 2, "exactly two executors must have been invoked"
    claimed_ids = [ex.calls[0] for ex in called]
    assert len(set(claimed_ids)) == 2
    assert set(claimed_ids) == {run_a.steps[0].id, run_b.steps[0].id}


@skip_without_postgres
async def test_two_due_steps_of_one_run_cannot_both_be_claimed_at_once(workflow_repo):
    """Two due steps in the SAME run, two concurrent callers: exactly one
    winner, and it must be the step at `position` 0 -- the ordering
    guarantee the claim predicate's `NOT EXISTS`-on-earlier-siblings check
    exists for. The loser comes back `False`, not an exception, the same
    clean shape a row-lock loser gets: the poller treats "blocked by
    ordering" and "blocked by SKIP LOCKED" identically -- both mean
    "nothing to do right now"."""
    base_time = datetime.now(timezone.utc) - timedelta(seconds=60)
    run = await workflow_repo.create_run(
        origin="voice",
        summary="two ordered example steps in one run",
        steps=[
            WorkflowStepSpec(
                kind="call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.example_ordered_first",
                },
            ),
            WorkflowStepSpec(
                kind="call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_on",
                    "entity_id": "switch.example_ordered_second",
                },
            ),
        ],
        base_time=base_time,
        created_by_user_id=None,
    )
    now = datetime.now(timezone.utc)
    ex1, ex2 = _RecordingExecutor(), _RecordingExecutor()

    results = await asyncio.gather(
        workflow_repo.claim_and_execute_next_due_step(ex1, now),
        workflow_repo.claim_and_execute_next_due_step(ex2, now),
        return_exceptions=True,
    )

    _assert_clean_race(results)
    assert results.count(True) == 1, f"expected exactly one winner, got {results!r}"
    assert results.count(False) == 1, f"expected exactly one clean loser, got {results!r}"

    called = [ex for ex in (ex1, ex2) if ex.calls]
    assert len(called) == 1, "the loser's executor must never have been invoked"
    claimed_step_id = called[0].calls[0]
    assert claimed_step_id == run.steps[0].id, (
        "the step claimed under contention must be the one at position 0 -- "
        "the later step is not claimable while the earlier one is still pending"
    )
