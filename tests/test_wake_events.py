"""The persistent wake-hit record (D-13, D-14): the `wake_events` table,
its repository pair, migration `0012`, and the two `SourceRunner` call
sites that write to it.

DBG-05's whole premise is that moving the threshold re-partitions numbers
already on disk. Before this plan there were no numbers on disk --
`sources/runner.py::_record_blocked_hit` emitted a structured log line
and nothing else, and no session directory was ever created for a hit the
gate blocked (`SessionRecorder` is only constructed after a hit is
allowed). This file proves: the table's field set is closed and carries
no transcript/audio-shaped column (T-08-06); both writes actually happen
and carry the right `allowed`/`block_reason`/`score`; and a failing store
never stops a source listening or delays the turn a wake hit started
(T-08-08).

The Postgres-backed tests below are marked `integration` and skipped
without a reachable Postgres, matching `tests/test_plugin_repo.py`'s own
precedent -- the direct sibling this file follows for its `sessionmaker`
fixture (reset schema, migrate to head, hand back a fresh
`async_sessionmaker`). The `SourceRunner` tests need no Postgres at all:
they drive `FakeWakeEventRepository` (`tests/conftest.py`) the same way
`tests/test_wake_gating.py` already drives a fake wake detector.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from spire_voice.config import GateConfig, WakeConfig
from spire_voice.db.engine import get_current_revision
from spire_voice.db.models import WakeEventRow
from spire_voice.db.postgres import PostgresWakeEventRepository
from spire_voice.db.repository import WakeEvent
from spire_voice.sources.runner import SourceRunner

from tests.conftest import FakeAudioSource, FakeWakeEventRepository, FakeWakeHit

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
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


def _run_upgrade_to(async_url: str, revision: str) -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(async_url))
    command.upgrade(cfg, revision)


@pytest.fixture
async def sessionmaker(monkeypatch):
    """`tests/test_plugin_repo.py`'s own `sessionmaker` fixture, unchanged
    -- reset schema, migrate to head, hand back a fresh
    `async_sessionmaker`."""
    monkeypatch.setenv("SPIRE_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("SPIRE_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    await _reset_schema(_TEST_DB_URL)
    _run_upgrade_to(_TEST_DB_URL, "head")

    engine = create_async_engine(_TEST_DB_URL)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


# === Task 1: the table, the repository, the closed field set ===========


def test_wake_event_field_names_are_exactly_the_seven_named_fields():
    """Reads the names off `dataclasses.fields(WakeEvent)` so this fails
    on an ADDED field, not only a renamed one -- a later field carrying
    speech cannot land here without this test objecting."""
    names = sorted(f.name for f in dataclasses.fields(WakeEvent))
    assert names == sorted(
        ["id", "source", "engine", "score", "allowed", "block_reason", "recorded_at"]
    )


@skip_without_postgres
async def test_a_recorded_wake_event_round_trips_with_a_real_id(sessionmaker):
    repo = PostgresWakeEventRepository(sessionmaker)
    recorded_at = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)

    written = await repo.record_wake_event(
        source="camera",
        engine="vosk",
        score=1.0,
        allowed=True,
        block_reason=None,
        recorded_at=recorded_at,
    )

    assert written.id is not None
    assert written.source == "camera"
    assert written.engine == "vosk"
    assert written.score == 1.0
    assert written.allowed is True
    assert written.block_reason is None
    assert written.recorded_at == recorded_at


@skip_without_postgres
async def test_list_wake_events_with_no_limit_returns_every_row_newest_first(sessionmaker):
    repo = PostgresWakeEventRepository(sessionmaker)
    base = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        await repo.record_wake_event(
            source="camera",
            engine="vosk",
            score=1.0,
            allowed=True,
            block_reason=None,
            recorded_at=base + timedelta(seconds=i),
        )

    events = await repo.list_wake_events()
    assert len(events) == 5
    assert [e.recorded_at for e in events] == sorted(
        (e.recorded_at for e in events), reverse=True
    )
    assert events[0].recorded_at == base + timedelta(seconds=4)


@skip_without_postgres
async def test_list_wake_events_with_a_limit_returns_the_twenty_newest(sessionmaker):
    repo = PostgresWakeEventRepository(sessionmaker)
    base = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(25):
        await repo.record_wake_event(
            source="camera",
            engine="vosk",
            score=1.0,
            allowed=True,
            block_reason=None,
            recorded_at=base + timedelta(seconds=i),
        )

    events = await repo.list_wake_events(limit=20)
    assert len(events) == 20
    assert events[0].recorded_at == base + timedelta(seconds=24)
    assert events[-1].recorded_at == base + timedelta(seconds=5)


@skip_without_postgres
async def test_delete_wake_events_before_removes_only_what_is_older_than_the_cutoff(sessionmaker):
    """The retention sweep's own call, against a real Postgres (WR-06).
    The boundary matches `sweep_expired_sessions`'s: strictly older goes,
    exactly at the cutoff stays."""
    repo = PostgresWakeEventRepository(sessionmaker)
    cutoff = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    for offset in (timedelta(seconds=-1), timedelta(0), timedelta(seconds=1)):
        await repo.record_wake_event(
            source="camera",
            engine="vosk",
            score=1.0,
            allowed=True,
            block_reason=None,
            recorded_at=cutoff + offset,
        )

    removed = await repo.delete_wake_events_before(cutoff)

    assert removed == 1
    remaining = await repo.list_wake_events()
    assert [event.recorded_at for event in remaining] == [
        cutoff + timedelta(seconds=1),
        cutoff,
    ]

    # Idempotent: a second sweep over the same window removes nothing and
    # says so, the way the directory sweep's empty run does.
    assert await repo.delete_wake_events_before(cutoff) == 0


@skip_without_postgres
async def test_two_events_recorded_at_the_same_instant_both_persist(sessionmaker):
    """No uniqueness rule could collapse them -- every row is a distinct
    event, not a per-key singleton."""
    repo = PostgresWakeEventRepository(sessionmaker)
    same_instant = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)

    first = await repo.record_wake_event(
        source="camera",
        engine="vosk",
        score=1.0,
        allowed=True,
        block_reason=None,
        recorded_at=same_instant,
    )
    second = await repo.record_wake_event(
        source="camera",
        engine="vosk",
        score=1.0,
        allowed=True,
        block_reason=None,
        recorded_at=same_instant,
    )

    assert first.id != second.id
    events = await repo.list_wake_events()
    assert {e.id for e in events} == {first.id, second.id}


@skip_without_postgres
async def test_block_reason_is_none_for_allowed_and_a_gate_reason_for_blocked(sessionmaker):
    repo = PostgresWakeEventRepository(sessionmaker)
    allowed = await repo.record_wake_event(
        source="camera",
        engine="vosk",
        score=1.0,
        allowed=True,
        block_reason=None,
        recorded_at=datetime.now(timezone.utc),
    )
    blocked = await repo.record_wake_event(
        source="camera",
        engine="openwakeword",
        score=0.1,
        allowed=False,
        block_reason="below_threshold",
        recorded_at=datetime.now(timezone.utc),
    )

    assert allowed.block_reason is None
    assert blocked.block_reason == "below_threshold"


@skip_without_postgres
async def test_migration_0012_applies_onto_0011_and_reverses_cleanly(monkeypatch):
    from alembic import command
    from alembic.config import Config as AlembicConfig

    # Migration 0001 reads `SPIRE_CONFIG`'s own `${VAR}`-expanded safety
    # block while seeding -- the same env vars `sessionmaker` above sets,
    # needed here too since this test drives Alembic directly rather than
    # through that fixture (it stops partway to `0011` before `0012`).
    monkeypatch.setenv("SPIRE_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("SPIRE_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    await _reset_schema(_TEST_DB_URL)
    _run_upgrade_to(_TEST_DB_URL, "0011")
    assert get_current_revision(_migration_url(_TEST_DB_URL)) == "0011"

    _run_upgrade_to(_TEST_DB_URL, "0012")
    assert get_current_revision(_migration_url(_TEST_DB_URL)) == "0012"

    engine = create_async_engine(_TEST_DB_URL)
    async with engine.connect() as conn:
        count = (await conn.execute(text("SELECT count(*) FROM wake_events"))).scalar_one()
    await engine.dispose()
    assert count == 0, "0012 seeds nothing -- new, empty, append-only state (D-14)"

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(_TEST_DB_URL))
    command.downgrade(cfg, "0011")
    assert get_current_revision(_migration_url(_TEST_DB_URL)) == "0011"

    _run_upgrade_to(_TEST_DB_URL, "0012")
    assert get_current_revision(_migration_url(_TEST_DB_URL)) == "0012"


# === Task 2: both call sites write, neither slows a turn down ==========


class _AlwaysHitWakeDetector:
    """Fires a hit of a fixed score on every call, matching
    `tests/test_wake_gating.py`'s own fake of the same name."""

    def __init__(self, score: float = 1.0) -> None:
        self._score = score

    def process(self, chunk: bytes) -> FakeWakeHit:
        return FakeWakeHit(score=self._score)

    def close(self) -> None:
        pass


async def test_a_blocked_hit_records_one_event_with_the_gates_reason(caplog):
    repo = FakeWakeEventRepository()
    source = FakeAudioSource(frames=[b"\x00"])
    detector = _AlwaysHitWakeDetector(score=0.1)
    wake_config = WakeConfig(engine="openwakeword", refractory_s=0.0)  # default threshold 0.55
    gate_config = GateConfig()

    async def _run_turn(src: object) -> None:
        return None

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
        wake_event_repo=repo,
    )

    with caplog.at_level(logging.INFO, logger="spire_voice.sources.runner"):
        await runner.run()
        await asyncio.sleep(0)  # let the scheduled write's task actually run

    # The existing structured log line still fires, unchanged, alongside
    # the new write.
    blocked_logs = [r for r in caplog.records if getattr(r, "block_reason", None) is not None]
    assert len(blocked_logs) == 1

    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.allowed is False
    assert event.block_reason == "below_threshold"
    assert event.score == 0.1
    assert event.source == "camera"
    assert event.engine == "openwakeword"


async def test_an_allowed_hit_records_one_event_with_no_block_reason():
    repo = FakeWakeEventRepository()
    source = FakeAudioSource(frames=[b"\x00"])
    detector = _AlwaysHitWakeDetector(score=1.0)
    wake_config = WakeConfig(engine="vosk", refractory_s=0.0)
    gate_config = GateConfig()

    turns_started: list[object] = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
        wake_event_repo=repo,
    )

    await runner.run()
    await asyncio.sleep(0)

    assert len(turns_started) == 1
    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.allowed is True
    assert event.block_reason is None
    assert event.score == 1.0
    assert event.engine == "vosk"


async def test_a_source_runner_with_no_repository_behaves_exactly_as_before():
    source = FakeAudioSource(frames=[b"\x00"])
    detector = _AlwaysHitWakeDetector(score=1.0)
    wake_config = WakeConfig(engine="vosk", refractory_s=0.0)
    gate_config = GateConfig()

    turns_started: list[object] = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
    )

    await runner.run()

    assert len(turns_started) == 1  # no write, no error, no change


class _RaisingWakeEventRepository:
    """`record_wake_event` always raises -- proves a failing store never
    stops a source's listening loop and never prevents the turn from
    running (T-08-08)."""

    async def record_wake_event(self, **kwargs: object) -> None:
        raise RuntimeError("the store is down")

    async def list_wake_events(self, limit: int | None = None) -> list:
        return []


async def test_a_raising_repository_does_not_stop_the_source_or_the_turn(caplog):
    source = FakeAudioSource(frames=[b"\x00", b"\x01"])
    detector = _AlwaysHitWakeDetector(score=1.0)
    wake_config = WakeConfig(engine="vosk", refractory_s=0.0)
    gate_config = GateConfig()

    turns_started: list[object] = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
        wake_event_repo=_RaisingWakeEventRepository(),
    )

    with caplog.at_level(logging.ERROR, logger="spire_voice.sources.runner"):
        await runner.run()
        await asyncio.sleep(0)  # let the failing scheduled write actually raise

    assert len(turns_started) == 2, (
        "both hits still started a turn despite the raising repository -- a store "
        "that is down must never cost a house its listening"
    )
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("failed to record a wake event" in r.getMessage() for r in errors)


async def test_recording_never_delays_the_allowed_path_the_turn_starts_before_the_write():
    """The allowed branch schedules the write and returns without
    awaiting it -- proven here by a repository whose write never
    completes on its own: if the turn had to wait for it, `runner.run()`
    would hang and this test would time out."""
    write_started = asyncio.Event()
    write_may_finish = asyncio.Event()

    class _SlowWakeEventRepository:
        async def record_wake_event(self, **kwargs: object) -> None:
            write_started.set()
            await write_may_finish.wait()

        async def list_wake_events(self, limit: int | None = None) -> list:
            return []

    source = FakeAudioSource(frames=[b"\x00"])
    detector = _AlwaysHitWakeDetector(score=1.0)
    wake_config = WakeConfig(engine="vosk", refractory_s=0.0)
    gate_config = GateConfig()

    turns_started: list[object] = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
        wake_event_repo=_SlowWakeEventRepository(),
    )

    await asyncio.wait_for(runner.run(), timeout=1.0)
    assert len(turns_started) == 1, "the turn ran to completion without ever awaiting the slow write"

    write_may_finish.set()  # let the still-pending write task finish cleanly
    await asyncio.sleep(0)


# === Task 3: the boundary holds structurally, not by discipline ========


def test_wake_event_row_column_set_is_exactly_the_expected_seven_columns():
    names = {c.name for c in WakeEventRow.__table__.columns}
    assert names == {"id", "source", "engine", "score", "allowed", "block_reason", "recorded_at"}


def test_no_wake_event_row_column_is_a_json_or_generic_object_type():
    """T-08-06: there is nowhere for a later change to casually put a
    transcript -- enforced structurally, with a narrow scalar-only row
    schema, rather than with a generic extra-JSON column."""
    from sqlalchemy.types import JSON

    for column in WakeEventRow.__table__.columns:
        assert not isinstance(column.type, JSON), (
            f"{column.name} is a JSON column -- widening this record must never be this easy"
        )
        assert column.type.python_type in (int, str, float, bool, datetime), (
            f"{column.name} has an unexpected type {column.type.python_type!r}"
        )


async def test_booting_the_real_lifespan_assigns_app_state_wake_event_repo(tmp_path, monkeypatch):
    """Boots the real application through the same fake-builder shape
    `tests/test_startup_smoke.py` uses (its own fakes and monkeypatch
    targets, reused rather than duplicated) so a future refactor that
    drops the wiring is caught at boot, not by a screen showing nothing."""
    from fastapi.testclient import TestClient

    from spire_voice.plugins import manager as plugin_manager_module
    from tests import test_startup_smoke as smoke

    def _fake_build_repositories_with_wake_events(config: object, engine: object) -> dict:
        repositories = smoke._fake_build_repositories(config, engine)
        repositories["wake_event_repo"] = FakeWakeEventRepository()
        return repositories

    monkeypatch.setenv("SPIRE_SECRET_KEY", smoke._TEST_SECRET_KEY)
    monkeypatch.setattr(smoke.app_module, "CONFIG_PATH", str(smoke._write_fake_config(tmp_path)))
    monkeypatch.setattr(plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(smoke.app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(smoke.app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(smoke.app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(
        smoke.app_module, "_build_repositories", _fake_build_repositories_with_wake_events
    )
    monkeypatch.setattr(smoke.app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(smoke.app_module, "_build_wake_detector", smoke._fake_build_wake_detector)
    monkeypatch.setattr(
        smoke.app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor
    )

    with TestClient(smoke.app_module.app) as client:
        response = client.get("/transport")
        assert response.status_code == 200

        repo = smoke.app_module.app.state.wake_event_repo
        assert repo is not None
        assert hasattr(repo, "record_wake_event")
        assert hasattr(repo, "list_wake_events")


# === Code review WR-05: the last write before a restart ================


class _SlowWakeEventRepository:
    """Answers, but not instantly. The write is genuinely still in flight
    when the listening loop ends, which is what a real repository over a
    real socket looks like and what an in-memory fake never does."""

    def __init__(self) -> None:
        self.events: list = []

    async def record_wake_event(self, **kwargs):
        await asyncio.sleep(0.05)
        self.events.append(kwargs)

    async def list_wake_events(self, limit=None):
        return list(self.events)


class _HangingWakeEventRepository:
    """Reachable, and never answering. The failure mode a raising fake
    cannot reproduce: the write neither succeeds nor fails, so the task
    stays in flight and the set it lives in never drains."""

    def __init__(self) -> None:
        self.started = 0
        self.released = asyncio.Event()

    async def record_wake_event(self, **_kwargs):
        self.started += 1
        await self.released.wait()

    async def list_wake_events(self, limit=None):
        return []


async def test_a_pending_wake_event_write_is_drained_before_shutdown():
    """The wake immediately before a restart is the one D-14 says must not
    disappear silently. Nothing drained the scheduled write, so the engine
    was disposed underneath it (WR-05)."""
    repo = _SlowWakeEventRepository()
    source = FakeAudioSource(frames=[b"\x00"])
    detector = _AlwaysHitWakeDetector(score=1.0)

    async def _run_turn(src: object) -> None:
        return None

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=WakeConfig(engine="vosk", refractory_s=0.0),
        gate_config=GateConfig(),
        wake_event_repo=repo,
    )

    await runner.run()
    # Still in flight when the listening loop ends -- which is precisely
    # the moment `lifespan`'s teardown used to dispose the engine under it.
    assert runner._pending_wake_event_tasks
    assert repo.events == []

    await runner.drain_pending_wake_events()

    assert len(repo.events) == 1
    assert not runner._pending_wake_event_tasks


async def test_a_store_that_never_answers_bounds_the_pending_set_and_says_so(caplog):
    """A repository that hangs rather than raises leaves every scheduled
    write in flight forever, and a talking television never stops waking
    the house. The set has to be bounded, and the skip has to be said out
    loud -- D-14 forbids a silent drop, not a bounded one (WR-05)."""
    from spire_voice.sources.runner import MAX_PENDING_WAKE_EVENT_WRITES

    repo = _HangingWakeEventRepository()
    source = FakeAudioSource(frames=[b"\x00"])
    detector = _AlwaysHitWakeDetector(score=1.0)

    async def _run_turn(src: object) -> None:
        return None

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=WakeConfig(engine="vosk", refractory_s=0.0),
        gate_config=GateConfig(),
        wake_event_repo=repo,
    )

    with caplog.at_level(logging.WARNING, logger="spire_voice.sources.runner"):
        for _ in range(MAX_PENDING_WAKE_EVENT_WRITES + 20):
            runner._schedule_wake_event_write(score=1.0, allowed=True, block_reason=None)
            await asyncio.sleep(0)

        assert len(runner._pending_wake_event_tasks) == MAX_PENDING_WAKE_EVENT_WRITES
        saturation_warnings = [
            record for record in caplog.records if "already in flight" in record.getMessage()
        ]
        # Said once per episode, not once per wake hit.
        assert len(saturation_warnings) == 1

        # And a shutdown does not wait on a store that is not answering:
        # every write is genuinely finished when the drain returns, not
        # merely asked to stop.
        scheduled = set(runner._pending_wake_event_tasks)
        await runner.drain_pending_wake_events(timeout=0.01)
        assert all(task.done() for task in scheduled)

    repo.released.set()
