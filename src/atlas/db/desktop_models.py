"""`DesktopDeviceRow` -- the one table migration 0021 creates for Phase 14
(D-01). Kept out of `db/models.py`, following `db/edge_models.py`: this
phase's own repository (`db/desktop_postgres.py`) owns its row without
touching the shared module every other phase also edits.

`alembic/env.py` imports this module, next to `edge_models`, so
`Base.metadata` sees this table too.

The two partial unique indexes (`uq_desktop_devices_single_default` for D-14,
`uq_desktop_devices_active_name` for D-28) live in migration 0021 only, the
way 0016 holds the Google single-default index: a partial index with a
`lower(name)` expression is not something this model needs to describe.
"""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column

from atlas.db.models import Base


class DesktopDeviceRow(Base):
    """One paired Mac (D-01). `token_hash` stores the SHA-256 hash of the
    Mac's bearer token, never the token itself. The plaintext token is
    returned to the admin once, in the create response, and this table
    never sees it again.

    `revoked_at` removes access, never a `DELETE`, so a revoked Mac's row
    stays for `list_devices` and any later audit read. `edge_device_id` is
    the room mapping (D-06).
    """

    __tablename__ = "desktop_devices"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(nullable=True)
    edge_device_id: Mapped[int | None] = mapped_column(ForeignKey("edge_devices.id"), nullable=True)
    is_default: Mapped[bool] = mapped_column(
        nullable=False, default=False, server_default=sa.false()
    )
