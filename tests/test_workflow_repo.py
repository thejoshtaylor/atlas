"""The authoring surface's own behaviour (plan 05-02): `list_runs`/
`get_run` ordering, `cancel_run`'s terminal-run refusal, `append_steps`
extending a run's own schedule past its existing tail rather than from
`now`, and -- marked `integration` -- the append guard actually holding
once a concurrent claim has moved the run to `firing`.

Every test but the last drives `tests/conftest.py`'s
`FakeWorkflowRepository`, which reimplements the same claim/append/cancel
predicates in plain Python (`fake_workflow_repository`, a fixture factory
matching every other repository double in this suite). Its own docstring
states plainly what it cannot prove: it has no interleaving to race, so
the append guard's real claim -- that it is enforced under a genuine
database row lock, not merely a Python-level check -- is earned only by
this file's one `integration`-marked test, against a real, disposable
Postgres.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas.db.postgres import PostgresWorkflowRepository
from atlas.db.repository import (
    WorkflowRunNotAppendableError,
    WorkflowRunNotFoundError,
    WorkflowStepSpec,
)
from atlas.workflow.steps import StepOutcome

_TEST_DB_URL = os.environ.get("ATLAS_TEST_DATABASE_URL")

skip_without_postgres = pytest.mark.skipif(
    _TEST_DB_URL is None,
    reason=(
        "ATLAS_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite, to exercise these tests instead of skipping them"
    ),
)


def _spec(entity_id: str, service: str = "turn_off") -> WorkflowStepSpec:
    return WorkflowStepSpec(
        kind="call_service",
        arguments={"domain": "switch", "service": service, "entity_id": entity_id},
    )


# ---------------------------------------------------------------------------
# list_runs / get_run ordering
# ---------------------------------------------------------------------------


async def test_list_runs_orders_newest_first(fake_workflow_repository):
    repo = fake_workflow_repository()
    base_time = datetime(2027, 6, 1, tzinfo=timezone.utc)
    run1 = await repo.create_run(
        origin="voice", summary="example run one", steps=[_spec("switch.example_one")],
        base_time=base_time, created_by_user_id=None,
    )
    run2 = await repo.create_run(
        origin="voice", summary="example run two", steps=[_spec("switch.example_two")],
        base_time=base_time, created_by_user_id=None,
    )
    run3 = await repo.create_run(
        origin="voice", summary="example run three", steps=[_spec("switch.example_three")],
        base_time=base_time, created_by_user_id=None,
    )

    runs = await repo.list_runs()

    assert [r.id for r in runs] == [run3.id, run2.id, run1.id]


async def test_list_runs_filters_by_status(fake_workflow_repository):
    repo = fake_workflow_repository()
    still_pending = await repo.create_run(
        origin="voice",
        summary="still pending",
        steps=[_spec("switch.example_pending")],
        base_time=datetime.now(timezone.utc) + timedelta(hours=1),
        created_by_user_id=None,
    )
    already_completed = await repo.create_run(
        origin="voice",
        summary="already completed",
        steps=[_spec("switch.example_completed")],
        base_time=datetime.now(timezone.utc) - timedelta(seconds=60),
        created_by_user_id=None,
    )

    async def _executor(step) -> StepOutcome:
        return StepOutcome(status="completed", detail={}, speech=None, retry=False)

    # Only already_completed's own step is due -- one claim call resolves
    # exactly that run, leaving still_pending untouched, with no
    # dependence on wall-clock timing to pick which one gets claimed.
    claimed = await repo.claim_and_execute_next_due_step(_executor, datetime.now(timezone.utc))
    assert claimed is True

    pending_only = await repo.list_runs(statuses=("pending",))
    assert [r.id for r in pending_only] == [still_pending.id]

    completed_only = await repo.list_runs(statuses=("completed",))
    assert [r.id for r in completed_only] == [already_completed.id]


async def test_a_runs_steps_are_ordered_by_position(fake_workflow_repository):
    repo = fake_workflow_repository()
    base_time = datetime(2027, 6, 1, tzinfo=timezone.utc)
    run = await repo.create_run(
        origin="voice",
        summary="three ordered example steps",
        steps=[
            _spec("switch.example_pos_0"),
            _spec("switch.example_pos_1"),
            _spec("switch.example_pos_2"),
        ],
        base_time=base_time,
        created_by_user_id=None,
    )

    fetched = await repo.get_run(run.id)

    assert [s.position for s in fetched.steps] == [0, 1, 2]
    assert [s.arguments["entity_id"] for s in fetched.steps] == [
        "switch.example_pos_0",
        "switch.example_pos_1",
        "switch.example_pos_2",
    ]


async def test_get_run_on_a_missing_id_returns_none(fake_workflow_repository):
    repo = fake_workflow_repository()

    assert await repo.get_run(999) is None


# ---------------------------------------------------------------------------
# cancel_run
# ---------------------------------------------------------------------------


async def test_cancel_run_on_a_pending_run_cancels_it_and_its_pending_steps(
    fake_workflow_repository,
):
    repo = fake_workflow_repository()
    base_time = datetime(2027, 6, 1, tzinfo=timezone.utc) + timedelta(hours=1)
    run = await repo.create_run(
        origin="voice",
        summary="example run to cancel",
        steps=[_spec("switch.example_cancel_a"), _spec("switch.example_cancel_b")],
        base_time=base_time,
        created_by_user_id=None,
    )
    now = datetime.now(timezone.utc)

    cancelled = await repo.cancel_run(run.id, now=now, cancelled_by_user_id=7)

    assert cancelled is True
    fetched = await repo.get_run(run.id)
    assert fetched.status == "cancelled"
    assert all(s.status == "cancelled" for s in fetched.steps)
    assert all(s.result_detail == {"cancelled_by_user_id": 7} for s in fetched.steps)


async def test_cancel_run_on_an_already_terminal_run_returns_false_and_changes_nothing(
    fake_workflow_repository,
):
    repo = fake_workflow_repository()
    base_time = datetime.now(timezone.utc) - timedelta(seconds=60)
    run = await repo.create_run(
        origin="voice", summary="example run already done", steps=[_spec("switch.example_done")],
        base_time=base_time, created_by_user_id=None,
    )

    async def _executor(step) -> StepOutcome:
        return StepOutcome(status="completed", detail={}, speech=None, retry=False)

    await repo.claim_and_execute_next_due_step(_executor, datetime.now(timezone.utc))
    before = await repo.get_run(run.id)
    assert before.status == "completed"

    cancelled = await repo.cancel_run(run.id, now=datetime.now(timezone.utc), cancelled_by_user_id=1)

    assert cancelled is False
    after = await repo.get_run(run.id)
    assert after.status == "completed"
    assert after.steps[0].status == "completed"


async def test_cancel_run_on_a_missing_id_returns_false(fake_workflow_repository):
    repo = fake_workflow_repository()

    assert await repo.cancel_run(999, now=datetime.now(timezone.utc), cancelled_by_user_id=None) is False


# ---------------------------------------------------------------------------
# append_steps
# ---------------------------------------------------------------------------


async def test_append_steps_extends_due_at_past_the_existing_tail_not_from_now(
    fake_workflow_repository,
):
    repo = fake_workflow_repository()
    base_time = datetime(2027, 7, 1, 9, 0, tzinfo=timezone.utc)
    run = await repo.create_run(
        origin="voice", summary="example run with one step", steps=[_spec("switch.example_tail")],
        base_time=base_time, created_by_user_id=None,
    )
    # `now` is well *before* the existing tail's own due_at -- if
    # append_steps computed the new step's due_at from `now` instead of
    # the run's own last step, the appended step would land earlier than
    # what is already scheduled.
    now = base_time - timedelta(hours=1)

    updated = await repo.append_steps(run.id, [_spec("switch.example_tail_appended")], now=now)

    assert len(updated.steps) == 2
    assert updated.steps[1].position == 1
    assert updated.steps[1].due_at == base_time


async def test_append_steps_folds_a_trailing_wait_steps_duration_forward(
    fake_workflow_repository,
):
    """PA-D1's own fold rule, exercised across the append boundary: if the
    run's existing last step is itself a `wait`, an appended step must
    land after that wait's own duration, not merely at the wait step's
    own due_at."""
    repo = fake_workflow_repository()
    base_time = datetime(2027, 7, 1, 9, 0, tzinfo=timezone.utc)
    run = await repo.create_run(
        origin="voice",
        summary="example run ending in a wait",
        steps=[WorkflowStepSpec(kind="wait", arguments={"duration_s": 300})],
        base_time=base_time,
        created_by_user_id=None,
    )

    updated = await repo.append_steps(
        run.id, [_spec("switch.example_after_wait")], now=base_time - timedelta(hours=1)
    )

    assert updated.steps[1].due_at == base_time + timedelta(seconds=300)


async def test_append_steps_raises_not_found_for_a_missing_run(fake_workflow_repository):
    repo = fake_workflow_repository()

    with pytest.raises(WorkflowRunNotFoundError):
        await repo.append_steps(999, [_spec("switch.example_missing")], now=datetime.now(timezone.utc))


async def test_append_steps_is_refused_once_the_run_has_left_pending(fake_workflow_repository):
    repo = fake_workflow_repository()
    base_time = datetime.now(timezone.utc) - timedelta(seconds=60)
    run = await repo.create_run(
        origin="voice",
        summary="example run that has started firing",
        steps=[_spec("switch.example_firing")],
        base_time=base_time,
        created_by_user_id=None,
    )

    async def _executor(step) -> StepOutcome:
        # A retried outcome leaves the run in `firing` (the step returns
        # to `pending`, but the run's own status, once flipped away from
        # `pending`, is never reset back by this fake -- matching
        # PostgresWorkflowRepository's own claim path, which likewise
        # never moves a run back to `pending`).
        return StepOutcome(status="failed", detail={}, speech=None, retry=True)

    claimed = await repo.claim_and_execute_next_due_step(_executor, datetime.now(timezone.utc))
    assert claimed is True
    firing_run = await repo.get_run(run.id)
    assert firing_run.status == "firing"

    with pytest.raises(WorkflowRunNotAppendableError) as excinfo:
        await repo.append_steps(
            run.id, [_spec("switch.example_too_late")], now=datetime.now(timezone.utc)
        )

    assert excinfo.value.run_id == run.id
    assert excinfo.value.status == "firing"


async def test_replace_steps_recomputes_due_at_from_the_runs_original_first_step(
    fake_workflow_repository,
):
    repo = fake_workflow_repository()
    base_time = datetime(2027, 7, 1, 9, 0, tzinfo=timezone.utc)
    run = await repo.create_run(
        origin="voice",
        summary="example run to replace",
        steps=[_spec("switch.example_replace_original")],
        base_time=base_time,
        created_by_user_id=None,
    )
    later_now = base_time + timedelta(hours=3)

    updated = await repo.replace_steps(
        run.id, [_spec("switch.example_replace_new")], now=later_now
    )

    assert len(updated.steps) == 1
    assert updated.steps[0].arguments["entity_id"] == "switch.example_replace_new"
    assert updated.steps[0].due_at == base_time, (
        "replace_steps must preserve the run's own original schedule start, "
        "never reset it to the moment of the edit"
    )


# ---------------------------------------------------------------------------
# The append guard under a real, concurrent-with-the-poller Postgres race.
# ---------------------------------------------------------------------------


def _migration_url(async_url: str) -> str:
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
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


@pytest.mark.integration
@skip_without_postgres
async def test_append_steps_is_refused_once_a_concurrent_claim_has_moved_the_run_to_firing(
    monkeypatch,
):
    """T-05-09: two independent connections -- the same "two independent
    workers" shape `tests/test_workflow_scheduler_restart.py`'s own
    FLOW-07 test uses, not one shared session -- prove the guard re-reads
    live, *committed* state under its own row lock rather than trusting a
    read taken before the race. The run is seeded with two due steps so
    the claim genuinely leaves it in `firing` (one step still pending)
    rather than skipping straight to a terminal status, matching this
    guard's own wording ("refused once the run has started firing").

    Deterministic sequencing across the two connections, not a timed
    interleave: 05-RESEARCH.md's own Pitfall 4 is explicit that a
    concurrency guarantee is proved by the primitive actually being
    exercised against a real transaction boundary, never by hoping two
    calls land at the same instant.
    """
    _set_migration_env(monkeypatch)
    await _reset_schema(_TEST_DB_URL)
    await asyncio.to_thread(_run_upgrade_head, _TEST_DB_URL)

    engine_claim = create_async_engine(_TEST_DB_URL)
    sessionmaker_claim = async_sessionmaker(engine_claim, expire_on_commit=False)
    repo_claim = PostgresWorkflowRepository(sessionmaker_claim)

    due_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    run = await repo_claim.create_run(
        origin="voice",
        summary="example run raced by a concurrent claim",
        steps=[
            WorkflowStepSpec(
                kind="call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.example_race_first",
                },
            ),
            WorkflowStepSpec(
                kind="call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_on",
                    "entity_id": "switch.example_race_second",
                },
            ),
        ],
        base_time=due_at,
        created_by_user_id=None,
    )

    async def _executor(step) -> StepOutcome:
        return StepOutcome(status="completed", detail={}, speech=None, retry=False)

    claimed = await repo_claim.claim_and_execute_next_due_step(_executor, datetime.now(timezone.utc))
    assert claimed is True
    await engine_claim.dispose()

    engine_append = create_async_engine(_TEST_DB_URL)
    sessionmaker_append = async_sessionmaker(engine_append, expire_on_commit=False)
    repo_append = PostgresWorkflowRepository(sessionmaker_append)

    firing_run = await repo_append.get_run(run.id)
    assert firing_run.status == "firing", (
        "the claim of the first of two steps must leave the run firing, "
        "not skip straight to a terminal status"
    )

    with pytest.raises(WorkflowRunNotAppendableError) as excinfo:
        await repo_append.append_steps(
            run.id,
            [
                WorkflowStepSpec(
                    kind="call_service",
                    arguments={
                        "domain": "switch",
                        "service": "turn_on",
                        "entity_id": "switch.example_race_too_late",
                    },
                )
            ],
            now=datetime.now(timezone.utc),
        )
    await engine_append.dispose()

    assert excinfo.value.run_id == run.id
    assert excinfo.value.status == "firing"

    check_engine = create_async_engine(_TEST_DB_URL)
    async with check_engine.connect() as conn:
        step_count = (
            await conn.execute(
                text("SELECT count(*) FROM workflow_steps WHERE run_id = :run_id"),
                {"run_id": run.id},
            )
        ).scalar_one()
    await check_engine.dispose()
    assert step_count == 2, "the refused append must not have landed a third step row"
