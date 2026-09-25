"""`EdgeDeviceRow` -- the one table migration 0017 creates for Phase 10
(D-03). Kept out of `db/models.py`, following `db/google_models.py`'s own
precedent: this phase's own repository (`db/edge_postgres.py`) owns its
row without touching the shared module every other phase also edits.

`alembic/env.py` imports this module, next to `google_models`, so
`Base.metadata` sees this table too -- a model class that does not
subclass `Base`, or a model module nobody imports, is invisible to
`alembic revision --autogenerate` (`db/models.py`'s own docstring).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column

from atlas.db.models import Base


class EdgeDeviceRow(Base):
    """One paired Raspberry Pi (D-03) -- `token_hash` stores the SHA-256
    hash of the device's bearer token, never the token itself, the same
    "a database read that yields live credentials is a different and
    worse incident than a database read that yields hashes" reasoning
    `InviteRow.token_hash` already carries. The plaintext token is
    returned to the admin exactly once, in the create-device response
    body, and this table never sees it again.

    `revoked_at` is how access is removed -- never a `DELETE`, the same
    "disable, never delete" discipline `UserRow.disabled_at` already
    establishes -- so a revoked device's row still exists for
    `list_devices` and for whatever later audit reads it.
    """

    __tablename__ = "edge_devices"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_connected_at: Mapped[datetime | None] = mapped_column(nullable=True)
