"""`PostgresSpeakerRepository` against a real Postgres (D-01, D-03, Phase
11, plan 11-03) -- the `<behavior>` cases from `11-03-PLAN.md`'s Task 1,
proven against real Postgres.

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_edge_device_repository.py`'s own precedent. The one test that
needs no Postgres at all (the hard-delete column-name check) carries no
marker, so it runs everywhere.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas.db.speaker_models import SpeakerEmbeddingRow, SpeakerRow
from atlas.db.speaker_postgres import PostgresSpeakerRepository

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


def test_speaker_tables_carry_no_soft_delete_column():
    """D-03: `speakers` and `speaker_embeddings` are a deliberate break
    from this project's "disable, never delete" discipline
    (`EdgeDeviceRow.revoked_at`, `UserRow.disabled_at`). Neither table's
    column set may hold a soft-delete marker -- the only column ending in
    `_at` on either table is `created_at`. Needs no Postgres: this reads
    the ORM's own declared columns."""
    for table in (SpeakerRow, SpeakerEmbeddingRow):
        at_columns = [name for name in table.__table__.columns.keys() if name.endswith("_at")]
        assert at_columns == ["created_at"], (
            f"{table.__name__} must carry no soft-delete column, found: {at_columns}"
        )


def _migration_url(async_url: str) -> str:
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


def _run_upgrade_head(async_url: str) -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(async_url))
    command.upgrade(cfg, "head")


@pytest.fixture
async def sessionmaker(monkeypatch):
    monkeypatch.setenv("ATLAS_CONFIG", "config/config.example.yaml")
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
    _run_upgrade_head(_TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@skip_without_postgres
async def test_create_speaker_then_list_returns_it_with_no_linked_user(sessionmaker):
    repo = PostgresSpeakerRepository(sessionmaker)
    now = datetime.now(timezone.utc)

    created = await repo.create_speaker(display_name="Member A", linked_user_id=None, created_at=now)
    assert created.linked_user_id is None
    assert created.display_name == "Member A"

    [listed] = await repo.list_speakers()
    assert listed == created


@skip_without_postgres
async def test_a_second_speaker_with_the_same_display_name_raises_unique_violation(sessionmaker):
    repo = PostgresSpeakerRepository(sessionmaker)
    now = datetime.now(timezone.utc)

    await repo.create_speaker(display_name="Member A", linked_user_id=None, created_at=now)

    with pytest.raises(IntegrityError):
        await repo.create_speaker(display_name="Member A", linked_user_id=None, created_at=now)


@skip_without_postgres
async def test_upsert_embedding_twice_leaves_one_row_with_the_second_vector(sessionmaker):
    repo = PostgresSpeakerRepository(sessionmaker)
    speaker = await repo.create_speaker(
        display_name="Member A", linked_user_id=None, created_at=datetime.now(timezone.utc)
    )

    await repo.upsert_embedding(
        speaker_id=speaker.id,
        phrase_index=0,
        model_id="model-a",
        vector=[0.1, 0.2, 0.3],
        created_at=datetime.now(timezone.utc),
    )
    await repo.upsert_embedding(
        speaker_id=speaker.id,
        phrase_index=0,
        model_id="model-a",
        vector=[0.9, 0.8, 0.7],
        created_at=datetime.now(timezone.utc),
    )

    [reference] = await repo.list_reference_embeddings("model-a")
    assert reference.vector == [0.9, 0.8, 0.7]

    # A different model id adds a second row -- never replaces the first.
    await repo.upsert_embedding(
        speaker_id=speaker.id,
        phrase_index=0,
        model_id="model-b",
        vector=[0.5, 0.5, 0.5],
        created_at=datetime.now(timezone.utc),
    )
    assert len(await repo.list_reference_embeddings("model-a")) == 1
    assert len(await repo.list_reference_embeddings("model-b")) == 1


@skip_without_postgres
async def test_list_reference_embeddings_returns_only_the_requested_model_with_display_name(sessionmaker):
    repo = PostgresSpeakerRepository(sessionmaker)
    member_a = await repo.create_speaker(
        display_name="Member A", linked_user_id=None, created_at=datetime.now(timezone.utc)
    )
    member_b = await repo.create_speaker(
        display_name="Member B", linked_user_id=None, created_at=datetime.now(timezone.utc)
    )
    await repo.upsert_embedding(
        speaker_id=member_a.id,
        phrase_index=0,
        model_id="model-a",
        vector=[0.1, 0.2],
        created_at=datetime.now(timezone.utc),
    )
    await repo.upsert_embedding(
        speaker_id=member_b.id,
        phrase_index=0,
        model_id="model-a",
        vector=[0.3, 0.4],
        created_at=datetime.now(timezone.utc),
    )
    await repo.upsert_embedding(
        speaker_id=member_b.id,
        phrase_index=0,
        model_id="model-b",
        vector=[0.5, 0.6],
        created_at=datetime.now(timezone.utc),
    )

    references = await repo.list_reference_embeddings("model-a")
    assert {r.speaker_id for r in references} == {member_a.id, member_b.id}
    by_speaker = {r.speaker_id: r for r in references}
    assert by_speaker[member_a.id].display_name == "Member A"
    assert by_speaker[member_a.id].vector == [0.1, 0.2]
    assert by_speaker[member_b.id].display_name == "Member B"
    assert all(isinstance(v, float) for v in by_speaker[member_a.id].vector)


@skip_without_postgres
async def test_delete_speaker_removes_the_member_and_every_embedding(sessionmaker):
    repo = PostgresSpeakerRepository(sessionmaker)
    speaker = await repo.create_speaker(
        display_name="Member A", linked_user_id=None, created_at=datetime.now(timezone.utc)
    )
    await repo.upsert_embedding(
        speaker_id=speaker.id,
        phrase_index=0,
        model_id="model-a",
        vector=[0.1, 0.2],
        created_at=datetime.now(timezone.utc),
    )
    await repo.upsert_embedding(
        speaker_id=speaker.id,
        phrase_index=1,
        model_id="model-a",
        vector=[0.3, 0.4],
        created_at=datetime.now(timezone.utc),
    )

    assert await repo.delete_speaker(speaker.id) is True

    async with sessionmaker() as session:
        remaining = (
            await session.execute(
                text("SELECT count(*) FROM speaker_embeddings WHERE speaker_id = :id"),
                {"id": speaker.id},
            )
        ).scalar_one()
    assert remaining == 0
    assert await repo.get_speaker(speaker.id) is None

    # A second delete of the same, now-gone, id returns False.
    assert await repo.delete_speaker(speaker.id) is False


@skip_without_postgres
async def test_count_embeddings_counts_phrases_enrolled_per_member_for_one_model(sessionmaker):
    repo = PostgresSpeakerRepository(sessionmaker)
    member_a = await repo.create_speaker(
        display_name="Member A", linked_user_id=None, created_at=datetime.now(timezone.utc)
    )
    for phrase_index in range(3):
        await repo.upsert_embedding(
            speaker_id=member_a.id,
            phrase_index=phrase_index,
            model_id="model-a",
            vector=[0.1, 0.2],
            created_at=datetime.now(timezone.utc),
        )

    counts = await repo.count_embeddings("model-a")
    assert counts == {member_a.id: 3}


@skip_without_postgres
async def test_upsert_embedding_refuses_an_empty_or_oversized_or_non_finite_vector(sessionmaker):
    repo = PostgresSpeakerRepository(sessionmaker)
    speaker = await repo.create_speaker(
        display_name="Member A", linked_user_id=None, created_at=datetime.now(timezone.utc)
    )

    with pytest.raises(ValueError):
        await repo.upsert_embedding(
            speaker_id=speaker.id,
            phrase_index=0,
            model_id="model-a",
            vector=[],
            created_at=datetime.now(timezone.utc),
        )
    with pytest.raises(ValueError):
        await repo.upsert_embedding(
            speaker_id=speaker.id,
            phrase_index=0,
            model_id="model-a",
            vector=[0.1] * 1025,
            created_at=datetime.now(timezone.utc),
        )
    with pytest.raises(ValueError):
        await repo.upsert_embedding(
            speaker_id=speaker.id,
            phrase_index=0,
            model_id="model-a",
            vector=[float("nan")],
            created_at=datetime.now(timezone.utc),
        )


@skip_without_postgres
async def test_delete_embedding_removes_the_phrase_row_for_every_model(sessionmaker):
    repo = PostgresSpeakerRepository(sessionmaker)
    speaker = await repo.create_speaker(
        display_name="Member A", linked_user_id=None, created_at=datetime.now(timezone.utc)
    )
    for phrase_index, model_id in ((100, "a"), (100, "b"), (101, "a")):
        await repo.upsert_embedding(
            speaker_id=speaker.id,
            phrase_index=phrase_index,
            model_id=model_id,
            vector=[0.1, 0.2],
            created_at=datetime.now(timezone.utc),
        )

    assert await repo.delete_embedding(speaker_id=speaker.id, phrase_index=100) == 2
    assert await repo.delete_embedding(speaker_id=speaker.id, phrase_index=100) == 0

    [remaining] = await repo.list_reference_embeddings("a")
    assert remaining.phrase_index == 101
    assert await repo.list_reference_embeddings("b") == []
