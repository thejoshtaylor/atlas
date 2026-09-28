"""`SpeakerRow` and `SpeakerEmbeddingRow` -- the two tables migration 0018
creates for Phase 11 (D-01, D-03). Kept out of `db/models.py`, following
`db/edge_models.py`'s own precedent: this phase's own repository
(`db/speaker_postgres.py`) owns its rows without touching the shared
module every other phase also edits.

`alembic/env.py` imports this module, next to `edge_models`, so
`Base.metadata` sees these two tables too -- a model class that does not
subclass `Base`, or a model module nobody imports, is invisible to
`alembic revision --autogenerate` (`db/models.py`'s own docstring).

**Deliberate break from "disable, never delete":** every other resource
migrated so far disables rather than deletes (`EdgeDeviceRow.revoked_at`,
`UserRow.disabled_at`). A speaker embedding is biometric data, and D-03
requires a real, hard `DELETE` of a member and every one of that member's
embeddings when they are removed. Neither row below carries a
`disabled_at`, `revoked_at`, or `deleted_at` column -- do not copy that
column from `edge_models.py` onto either class here.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from atlas.db.models import Base


class SpeakerRow(Base):
    """One household member (D-01) -- a display name and an optional link
    to a `users` row. `linked_user_id` is `ondelete="SET NULL"`: a user
    account can be removed without also erasing the member's enrollment.

    No `disabled_at`/`revoked_at` column here -- `delete_speaker`
    (`db/speaker_postgres.py`) issues a real `DELETE`, and the foreign key
    cascade on `SpeakerEmbeddingRow.speaker_id` removes that member's
    embeddings along with it (D-03).
    """

    __tablename__ = "speakers"

    id: Mapped[int] = mapped_column(primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    linked_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False)


class SpeakerEmbeddingRow(Base):
    """One enrollment-phrase embedding for one speaker, under one model id
    (D-03). Storing the model id on the row (rather than assuming a single
    global model) is what lets a later model change re-embed without a new
    recording: the enrollment clips themselves stay on disk, outside
    Postgres, in the gitignored data root.

    `speaker_id` is `ondelete="CASCADE"` -- deleting a `SpeakerRow` removes
    every one of that member's embedding rows for good, no soft-delete
    column anywhere in this table either.
    """

    __tablename__ = "speaker_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    speaker_id: Mapped[int] = mapped_column(
        ForeignKey("speakers.id", ondelete="CASCADE"), nullable=False
    )
    phrase_index: Mapped[int] = mapped_column(Integer, nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    vector: Mapped[list[float]] = mapped_column(ARRAY(Float()), nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
