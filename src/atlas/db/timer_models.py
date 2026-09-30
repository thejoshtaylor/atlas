"""`TimerRow`: the `timers` table that migration 0020 creates.

It holds voice timers and alarms in one table (`kind` tells them apart).
`alembic/env.py` imports this module so `Base.metadata` sees the table.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, SmallInteger, Text, true
from sqlalchemy.orm import Mapped, mapped_column

from atlas.db.models import Base


class TimerRow(Base):
    """One timer or one alarm. Datetimes are naive UTC, the project's
    database convention (`db/postgres.py`)."""

    __tablename__ = "timers"
    __table_args__ = (CheckConstraint("kind IN ('timer', 'alarm')", name="ck_timers_kind"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    due_at: Mapped[datetime | None] = mapped_column(nullable=True)
    remaining_s: Mapped[int | None] = mapped_column(nullable=True)
    duration_s: Mapped[int | None] = mapped_column(nullable=True)
    time_of_day: Mapped[str | None] = mapped_column(Text, nullable=True)
    repeat_days: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=true())
    created_at: Mapped[datetime] = mapped_column(nullable=False)
