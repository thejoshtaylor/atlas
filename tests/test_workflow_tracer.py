"""The end-to-end tracer (plan 05-01): a `schedule_workflow` tool call
writes a durable run and step, nothing fires at that moment, and one poll
after the step's `due_at` fires it through the tool host and writes its
terminal state in the transaction that claimed it.

Marked `integration` and skipped without `SPIRE_TEST_DATABASE_URL`,
following `tests/test_account_repository_concurrency.py`'s own
skip/reset/upgrade harness shape exactly.

A recording in-process tool host stands in for the Home Assistant MCP
child -- this project's `fake_ha` fixture is an in-process
`httpx.MockTransport` and cannot answer a request from a *separately
spawned* child process (STATE.md, plan 02-01's own finding), so a real
spawned child is not what this test drives. It drives the real,
Postgres-backed claim path (`PostgresWorkflowRepository`) and the real
poller (`WorkflowScheduler._poll_once`) against a recording stand-in for
the one thing this tracer does not re-prove: that the real MCP child
enforces policy. That is plan 05-03's own SAFE-08 test's job, against the
real policy route, which needs no HTTP server at all to prove.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from mcp.types import CallToolResult, TextContent
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from spire_voice.config import WorkflowConfig
from spire_voice.db.postgres import PostgresWorkflowRepository
from spire_voice.workflow.scheduler import WorkflowScheduler
from spire_voice.workflow.steps import execute_step
from spire_voice.workflow.tool import WorkflowToolHost

_TEST_DB_URL = os.environ.get("SPIRE_TEST_DATABASE_URL")

pytestmark = pytest.mark.integration

skip_without_postgres = pytest.mark.skipif(
    _TEST_DB_URL is None,
    reason=(
        "SPIRE_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite, to exercise these tests instead of skipping them"
    ),
)


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
    monkeypatch.setenv("SPIRE_CONFIG", "config/config.example.yaml")
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


class _RecordingToolHost:
    """Stands in for the real Home Assistant MCP child: records every
    call it receives and returns a fixed, success-shaped
    `CallToolResult`, the same shape `mcp_client.py::McpToolHost.call_tool`
    returns for a real, allowed `ha_call_service` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        self.calls.append((name, dict(arguments)))
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structured_content={"changed": [arguments.get("entity_id")]},
        )


async def _build_repo(monkeypatch) -> PostgresWorkflowRepository:
    _set_migration_env(monkeypatch)
    await _reset_schema(_TEST_DB_URL)
    _run_upgrade_head(_TEST_DB_URL)
    engine = create_async_engine(_TEST_DB_URL)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    return PostgresWorkflowRepository(sessionmaker, max_attempts=3, retry_backoff_s=30.0)


async def _fetch_run_and_step(async_url: str) -> tuple[tuple, tuple]:
    engine = create_async_engine(async_url)
    try:
        async with engine.connect() as conn:
            run_row = (
                await conn.execute(
                    text("SELECT id, origin, status, summary FROM workflow_runs")
                )
            ).one()
            step_row = (
                await conn.execute(
                    text(
                        "SELECT id, run_id, position, kind, status, fired_at, due_at "
                        "FROM workflow_steps WHERE run_id = :run_id"
                    ),
                    {"run_id": run_row.id},
                )
            ).one()
            return run_row, step_row
    finally:
        await engine.dispose()


@skip_without_postgres
async def test_a_scheduled_call_service_step_fires_exactly_once_after_due_at(monkeypatch):
    repo = await _build_repo(monkeypatch)
    recording_host = _RecordingToolHost()

    clock_time = [datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)]
    tool_host = WorkflowToolHost(repo, zone=None, clock=lambda: clock_time[0])

    result = await tool_host.call_tool(
        "schedule_workflow",
        {
            "kind": "call_service",
            "arguments": {
                "domain": "switch",
                "service": "turn_off",
                "entity_id": "switch.example_tracer_socket",
            },
            "delay_seconds": 1200,
            "summary": "turn off the example socket",
        },
    )
    assert not getattr(result, "is_error", False), result

    run_row, step_row = await _fetch_run_and_step(_TEST_DB_URL)
    assert run_row.origin == "voice"
    assert run_row.status == "pending"
    assert run_row.summary == "turn off the example socket"
    assert step_row.kind == "call_service"
    assert step_row.status == "pending"
    assert step_row.fired_at is None

    # Nothing fired at the moment the sentence was spoken (D-08, FLOW-01's
    # own boundary): the recording host was never called.
    assert recording_host.calls == []

    # A poll taken before due_at claims nothing and calls nothing.
    config = WorkflowConfig()
    scheduler = WorkflowScheduler(
        repo,
        lambda step, now: execute_step(step, recording_host, config, now),
        config,
        clock=lambda: clock_time[0],
    )
    await scheduler._poll_once()
    assert scheduler.claimed_count == 0
    assert recording_host.calls == []

    # Advance the injected clock past due_at (1200s = 20 minutes) and poll
    # once more.
    clock_time[0] = clock_time[0] + timedelta(seconds=1201)
    await scheduler._poll_once()

    assert scheduler.claimed_count == 1
    assert recording_host.calls == [
        (
            "ha_call_service",
            {
                "domain": "switch",
                "service": "turn_off",
                "entity_id": "switch.example_tracer_socket",
            },
        )
    ]

    run_row, step_row = await _fetch_run_and_step(_TEST_DB_URL)
    assert step_row.status == "completed"
    assert step_row.fired_at is not None
    assert run_row.status == "completed"


@skip_without_postgres
async def test_a_poll_before_due_at_claims_nothing_and_calls_nothing(monkeypatch):
    repo = await _build_repo(monkeypatch)
    recording_host = _RecordingToolHost()

    clock_time = [datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)]
    tool_host = WorkflowToolHost(repo, zone=None, clock=lambda: clock_time[0])

    await tool_host.call_tool(
        "schedule_workflow",
        {
            "kind": "call_service",
            "arguments": {
                "domain": "switch",
                "service": "turn_on",
                "entity_id": "switch.example_tracer_lamp",
            },
            "delay_seconds": 3600,
            "summary": "turn on the example lamp",
        },
    )

    config = WorkflowConfig()
    scheduler = WorkflowScheduler(
        repo,
        lambda step, now: execute_step(step, recording_host, config, now),
        config,
        clock=lambda: clock_time[0],
    )
    # Advance, but not past due_at (3600s = 1 hour) -- 30 minutes in.
    clock_time[0] = clock_time[0] + timedelta(seconds=1800)
    await scheduler._poll_once()

    assert scheduler.claimed_count == 0
    assert recording_host.calls == []

    run_row, step_row = await _fetch_run_and_step(_TEST_DB_URL)
    assert step_row.status == "pending"
    assert run_row.status == "pending"
