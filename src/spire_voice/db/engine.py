"""Two module-level functions, and nothing else: the database engine's
construction, and the migration run that must complete before it serves
anything.

Both are module-level, not methods on a class, matching `_build_wake_detector`'s
and `_build_ffmpeg_supervisor`'s existing shape in `app.py` -- a separate,
easily monkeypatched function is what lets `tests/test_startup_smoke.py`
substitute a fake for each without a reachable Postgres.
"""

from __future__ import annotations

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from spire_voice.config import DatabaseConfig


def get_current_revision(migration_url: str) -> str | None:
    """The Alembic revision `migration_url`'s database is stamped at right
    now, or `None` when nothing has ever been applied (a genuinely fresh
    database, with no `alembic_version` table at all).

    A short-lived, synchronous connection -- opened, queried, and disposed
    within this call -- matching `run_migrations`'s own synchronous,
    unbridged shape (D-02, 03-RESEARCH.md Pitfall 1) rather than borrowing
    the async runtime engine for a one-off read.

    `app.py`'s `lifespan` calls this *before* `run_migrations` (CR-01 fix):
    `None` here means the migration this call precedes is about to seed the
    safety policy for the first time ever on this database, which is the
    one signal that distinguishes a first boot (a lingering `safety:` key
    should be tolerated and logged) from a later one (the same key should
    be rejected, because a previous boot already seeded it). `MigrationContext.
    get_current_revision` is Alembic's own API for this exact question -- it
    handles the missing-table case itself, so this function raises on
    nothing an unreachable or malformed database wouldn't already raise on.
    """
    engine = create_engine(migration_url)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            return context.get_current_revision()
    finally:
        engine.dispose()


def run_migrations(migration_url: str) -> None:
    """Run every pending Alembic migration up to `head`, against `migration_url`
    (the synchronous `psycopg` connection string, `DatabaseConfig.migration_url`).

    A plain, synchronous call -- never bridged through the async runtime
    engine's `run_sync()` (D-02, 03-RESEARCH.md Pitfall 1: a bridge means a
    second async engine object with its own pool to leak, and a single
    un-awaited `create_task` anywhere in that chain lets the caller continue
    past a migration that never actually ran, with no error). The caller
    (`app.py`'s `lifespan`) is responsible for running this inside
    `asyncio.to_thread` and awaiting it -- this function itself has no
    async awareness at all, matching Alembic's own `command.upgrade` API.
    """
    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", migration_url)
    command.upgrade(cfg, "head")


def build_engine(database_config: DatabaseConfig) -> AsyncEngine:
    """Build the one async engine `lifespan` holds for the life of the
    process -- created once, never per-request (D-02)."""
    return create_async_engine(database_config.url)
