from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# atlas: this is the plain (sync) Alembic template, not `-t async`
# (D-02) -- the runtime process uses an async `asyncpg` engine, but Alembic
# runs its own, fully separate synchronous `psycopg` connection, never
# bridged through the async engine's `run_sync()`. See 03-RESEARCH.md
# Pattern 1 and Pitfall 1 for why: a bridge means a second async engine
# object that must itself be explicitly disposed, and a single un-awaited
# `create_task` anywhere in that chain lets `lifespan` continue past a
# migration that never actually ran, with no error.
from atlas.db.models import Base

# Phase 9: a model module nobody imports is invisible to `Base.metadata`
# even though it subclasses `Base` (`db/models.py`'s own docstring) --
# this import is what makes the six Google tables visible to Alembic.
from atlas.db import google_models  # noqa: F401

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
#
# atlas: `disable_existing_loggers=False` is not Alembic's generated
# default -- it is required here. `run_migrations` (db/engine.py) runs
# `command.upgrade()` inside this application's own long-lived process
# (D-02), not a short-lived CLI invocation Alembic's template assumes.
# `fileConfig`'s own default (`disable_existing_loggers=True`) tears down
# every logger already configured in this process the moment a migration
# runs -- the application's own runtime logging, and (found the hard way,
# via test_retention.py/test_speaker_fifo.py/test_wake_gating.py's caplog
# assertions failing only when they ran after a migration in the same
# suite) pytest's own log-capture handler. `disable_existing_loggers=False`
# is the documented remedy for exactly this embedding pattern.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# `Base.metadata` is the single import that makes every model this project
# ever adds visible to `alembic revision --autogenerate` -- a model class
# that does not subclass `atlas.db.models.Base` is a model this
# environment cannot see.
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    `config.get_section(...)` reads `sqlalchemy.url` from this `Config`
    object, not from `alembic.ini` on disk -- `alembic.ini` leaves that key
    blank deliberately, and the caller (`app.py`'s `lifespan`, via
    `cfg.set_main_option("sqlalchemy.url", database.migration_url)`) sets it
    at run time instead, so the connection string lives in one place
    (`DatabaseConfig`), not two that could disagree.

    `poolclass=pool.NullPool` (kept from Alembic's own generated template,
    not added here) matters more than usual in this project: this runs
    *inside* the long-lived FastAPI process, not a short-lived CLI
    invocation, so a pooled connection from a one-shot migration would be a
    permanent leak for the rest of the process's life (03-RESEARCH.md
    Pitfall 5).
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
