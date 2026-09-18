"""The SQLAlchemy declarative base every Phase 3 table inherits from.

`alembic/env.py` reads `Base.metadata` as its `target_metadata` -- a model
class that does not subclass `Base`, or a model module nobody imports, is a
model `alembic revision --autogenerate` cannot see. This plan adds only the
base itself; later plans in this phase (accounts, invites, the safety
policy, provider credentials, settings, audit rows) add their tables here.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """The one declarative base for every ORM model in this project.

    A model defined against a second, unrelated `DeclarativeBase` would be
    invisible to `alembic/env.py`'s `target_metadata` and to the migration
    it drives -- there is exactly one base, imported from here, for every
    table this project ever adds.
    """
