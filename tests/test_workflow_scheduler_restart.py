"""FLOW-07's own evidence -- a run survives a restart of the process that
created it -- plus the loop contract `WorkflowScheduler` must honour
regardless: it drains, it is bounded, a raising poll does not end the
schedule, and `stop()` actually stops it.

Two halves, per 05-RESEARCH.md Pitfall 4's second half: a literal OS
process restart is not needed and must not be simulated with one. FLOW-07
is proven with a *second*, independent `WorkflowScheduler` over a second
sessionmaker against the same database, which never saw the run created
and shares no in-memory state with the scheduler that created it -- this
directly tests D-03/D-04's own claim (state lives in Postgres, not in the
process).

The loop-contract half needs no database at all and runs against a small,
local stub `WorkflowRepository` declared in this module rather than
`tests/conftest.py`'s `FakeWorkflowRepository` -- Task 3 of this same plan
is what adds that fake, and a test depending on a later task in its own
plan is a test that cannot be run when it is written. These four tests
mirror `tests/test_retention.py`'s own shape for `RetentionScheduler`
exactly: an injected `sleep` that yields once rather than actually
waiting, and assertions built entirely from counters, never from timing.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from spire_voice.config import WorkflowConfig
from spire_voice.db.postgres import PostgresWorkflowRepository
from spire_voice.db.repository import WorkflowStepSpec
from spire_voice.workflow.scheduler import WorkflowScheduler
from spire_voice.workflow.steps import execute_step

_TEST_DB_URL = os.environ.get("SPIRE_TEST_DATABASE_URL")

skip_without_postgres = pytest.mark.skipif(
    _TEST_DB_URL is None,
    reason=(
        "SPIRE_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite, to exercise these tests instead of skipping them"
    ),
)


# ---------------------------------------------------------------------------
# The durable half: real Postgres, two independent scheduler instances.
# ---------------------------------------------------------------------------


def _migration_url(async_url: str) -> str:
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
        for table in (
            "workflow_steps",
            "workflow_runs",
            "macro_actions",
            "macro_aliases",
            "macros",
            "settings",
            "setup_steps",
            "setup_state",
            "provider_credentials",
            "refresh_tokens",
            "invites",
            "users",
            "policy_rules",
            "safety_policy",
            "audit_log",
            "alembic_version",
        ):
            await conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    await engine.dispose()


def _run_upgrade_head(async_url: str) -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(async_url))
    command.upgrade(cfg, "head")


def _set_migration_env(monkeypatch) -> None:
    monkeypatch.setenv("SPIRE_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)


class _RecordingToolHost:
    """Stands in for the real Home Assistant MCP child, matching
    `tests/test_workflow_tracer.py`'s own `_RecordingToolHost` shape
    exactly: records every call and returns a fixed, success-shaped
    `CallToolResult`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        self.calls.append((name, dict(arguments)))
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structured_content={"changed": [arguments.get("entity_id")]},
        )


@pytest.mark.integration
@skip_without_postgres
async def test_a_second_scheduler_instance_fires_a_step_it_did_not_create(monkeypatch):
    """FLOW-07: scheduler A creates a run and is torn down (its own engine
    disposed, no in-memory state survives). A second, independent
    `WorkflowScheduler` B, built over a second sessionmaker against the
    same database, never saw that run created -- yet claims and fires its
    due step on its first poll, because the state that made it claimable
    lives in Postgres, not in scheduler A's process."""
    _set_migration_env(monkeypatch)
    await _reset_schema(_TEST_DB_URL)
    await asyncio.to_thread(_run_upgrade_head, _TEST_DB_URL)

    engine_a = create_async_engine(_TEST_DB_URL)
    sessionmaker_a = async_sessionmaker(engine_a, expire_on_commit=False)
    repo_a = PostgresWorkflowRepository(sessionmaker_a)

    due_at = datetime(2027, 3, 1, 9, 0, tzinfo=timezone.utc) + timedelta(seconds=600)
    run = await repo_a.create_run(
        origin="voice",
        summary="restart-test example step",
        steps=[
            WorkflowStepSpec(
                kind="call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.example_restart",
                },
            )
        ],
        base_time=due_at,
        created_by_user_id=None,
    )
    # Scheduler A is gone: its own engine disposed, nothing about it kept
    # in this test beyond the `run` value object already returned.
    await engine_a.dispose()

    engine_b = create_async_engine(_TEST_DB_URL)
    sessionmaker_b = async_sessionmaker(engine_b, expire_on_commit=False)
    repo_b = PostgresWorkflowRepository(sessionmaker_b)
    recording_host = _RecordingToolHost()
    config = WorkflowConfig()
    scheduler_b = WorkflowScheduler(
        repo_b,
        lambda step, now: execute_step(step, recording_host, config, now),
        config,
        clock=lambda: due_at + timedelta(seconds=1),
    )

    await scheduler_b._poll_once()
    await engine_b.dispose()

    assert scheduler_b.claimed_count == 1
    assert recording_host.calls == [
        (
            "ha_call_service",
            {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_restart"},
        )
    ]

    check_engine = create_async_engine(_TEST_DB_URL)
    async with check_engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT status FROM workflow_steps WHERE run_id = :run_id"),
                {"run_id": run.id},
            )
        ).one()
    await check_engine.dispose()
    assert row.status == "completed"


@pytest.mark.integration
@skip_without_postgres
async def test_an_overdue_step_fires_late_rather_than_being_dropped(monkeypatch):
    """D-04: a step whose `due_at` passed while nothing was polling is
    claimed on the next poll and runs, late, rather than being discarded --
    and its recorded `fired_at - due_at` is the real overdue interval, not
    silently zeroed or truncated."""
    _set_migration_env(monkeypatch)
    await _reset_schema(_TEST_DB_URL)
    await asyncio.to_thread(_run_upgrade_head, _TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repo = PostgresWorkflowRepository(sessionmaker)

    due_at = datetime(2027, 3, 1, 8, 0, tzinfo=timezone.utc)
    await repo.create_run(
        origin="voice",
        summary="overdue example step",
        steps=[
            WorkflowStepSpec(
                kind="call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.example_overdue",
                },
            )
        ],
        base_time=due_at,
        created_by_user_id=None,
    )

    lateness = timedelta(seconds=2400)  # 40 minutes late
    poll_time = due_at + lateness
    recording_host = _RecordingToolHost()
    config = WorkflowConfig()
    scheduler = WorkflowScheduler(
        repo,
        lambda step, now: execute_step(step, recording_host, config, now),
        config,
        clock=lambda: poll_time,
    )

    await scheduler._poll_once()
    await engine.dispose()

    assert scheduler.claimed_count == 1
    assert recording_host.calls, "an overdue step must still fire, not be dropped"

    check_engine = create_async_engine(_TEST_DB_URL)
    async with check_engine.connect() as conn:
        row = (
            await conn.execute(text("SELECT due_at, fired_at, status FROM workflow_steps"))
        ).one()
    await check_engine.dispose()
    assert row.status == "completed"
    assert row.fired_at - row.due_at == lateness, (
        "the recorded lateness must be the real overdue interval, not silently zeroed"
    )


# ---------------------------------------------------------------------------
# The loop-contract half: no Postgres, a local stub repository.
# ---------------------------------------------------------------------------


class _StubWorkflowRepository:
    """A minimal `WorkflowRepository` stand-in for this file's own
    loop-contract tests only -- deliberately not `tests/conftest.py`'s
    `FakeWorkflowRepository`, which Task 3 of this same plan adds. Exposes
    a fixed number of due steps to claim, optionally raising on the first
    claim to prove a raising poll does not end the schedule."""

    def __init__(self, due_count: int = 0, *, raise_first: bool = False) -> None:
        self._pending = due_count
        self._raise_first = raise_first
        self._raised_once = False

    @property
    def remaining_pending(self) -> int:
        return self._pending

    async def claim_and_execute_next_due_step(self, executor, now: datetime) -> bool:
        if self._raise_first and not self._raised_once:
            self._raised_once = True
            raise RuntimeError("simulated: claim blew up")
        if self._pending <= 0:
            return False
        self._pending -= 1
        await executor(object())
        return True


async def _noop_executor(step: Any, now: datetime) -> None:
    return None


async def _wait_for(predicate, *, attempts: int = 200, interval: float = 0.01) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"condition never became true after {attempts * interval:.2f}s")


async def _instant_sleep(_seconds: float) -> None:
    """A `WorkflowScheduler`-injected sleep that yields control back to the
    event loop exactly once rather than actually waiting, matching
    `tests/test_retention.py`'s own `_instant_sleep` -- lets a test drive
    many "intervals" with no real clock wait."""
    await asyncio.sleep(0)


async def test_stop_between_ticks_ends_the_loop():
    repo = _StubWorkflowRepository(due_count=0)
    config = WorkflowConfig(poll_interval_s=0.01)
    scheduler = WorkflowScheduler(repo, _noop_executor, config, sleep=_instant_sleep)

    scheduler.start()
    await _wait_for(lambda: scheduler.poll_count >= 2)
    await scheduler.stop()

    count_after_stop = scheduler.poll_count
    await asyncio.sleep(0.05)

    assert scheduler.poll_count == count_after_stop, (
        "no further poll must run once stop() has returned"
    )


async def test_a_poll_that_raises_is_logged_and_the_schedule_continues(caplog):
    """The counter still advances on the next tick after a raising poll --
    `asyncio.CancelledError` is re-raised rather than swallowed
    (`workflow/scheduler.py`'s own `except asyncio.CancelledError: raise`
    ahead of its blanket `except Exception`), which is also what lets
    `stop()` above terminate the loop cleanly rather than hang."""
    repo = _StubWorkflowRepository(due_count=0, raise_first=True)
    config = WorkflowConfig(poll_interval_s=0.01)
    scheduler = WorkflowScheduler(repo, _noop_executor, config, sleep=_instant_sleep)

    with caplog.at_level(logging.ERROR, logger="spire_voice.workflow.scheduler"):
        scheduler.start()
        try:
            await _wait_for(lambda: scheduler.poll_count >= 2)
        finally:
            await scheduler.stop()

    assert any("workflow poll raised" in record.message for record in caplog.records)
    assert scheduler.poll_count >= 2, (
        "one failed poll must not silently end the schedule for the life of the process"
    )


async def test_poll_once_drains_all_due_steps_in_one_tick():
    repo = _StubWorkflowRepository(due_count=5)
    config = WorkflowConfig(poll_interval_s=1000.0, max_steps_per_poll=20)
    scheduler = WorkflowScheduler(repo, _noop_executor, config, sleep=_instant_sleep)

    await scheduler._poll_once()

    assert scheduler.claimed_count == 5
    assert repo.remaining_pending == 0


async def test_poll_once_is_bounded_by_max_steps_per_poll():
    repo = _StubWorkflowRepository(due_count=5)
    config = WorkflowConfig(poll_interval_s=1000.0, max_steps_per_poll=2)
    scheduler = WorkflowScheduler(repo, _noop_executor, config, sleep=_instant_sleep)

    await scheduler._poll_once()

    assert scheduler.claimed_count == 2
    assert repo.remaining_pending == 3, (
        "a large backlog must not be drained past max_steps_per_poll in one tick"
    )
