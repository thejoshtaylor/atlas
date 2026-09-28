"""Admin routes to list, add, and delete household members (D-01, D-03,
D-04, Phase 11, plan 11-03).

Mirrors `routes/edge_devices.py`'s admin-guard/validation/response-model
shape. The delete route is where this module diverges from that analog:
`revoke_edge_device` sets `revoked_at` and keeps the row, but D-03 requires
a real, hard delete here -- a member's embeddings are biometric data, and
`delete_speaker` (`db/speaker_postgres.py`) removes the member row and
every one of its embedding rows for good, via a foreign key cascade.
Plan 11-07 extends this same delete to that member's enrollment clips on
disk.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator
from sqlalchemy.exc import IntegrityError

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.db.speaker_repository import Speaker, SpeakerRepository

router = APIRouter(tags=["speakers"])

# 1-64 characters, a closed set: letters, digits, space, `-`, `_`, `.` --
# the same closed-charset discipline `routes/edge_devices.py::_NAME_RE`
# already establishes for this exact kind of operator-supplied label.
_NAME_RE = re.compile(r"^[A-Za-z0-9 _.-]{1,64}$")


class SpeakerCreateRequest(BaseModel):
    display_name: str
    linked_user_id: "int | None" = None

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not _NAME_RE.match(trimmed):
            raise ValueError(
                "display_name must be 1-64 characters of letters, digits, space, '-', '_' "
                "and '.' (after trimming)"
            )
        return trimmed


class SpeakerResponse(BaseModel):
    id: int
    display_name: str
    linked_user_id: "int | None"
    created_at: datetime


def _to_response(speaker: Speaker) -> SpeakerResponse:
    return SpeakerResponse(
        id=speaker.id,
        display_name=speaker.display_name,
        linked_user_id=speaker.linked_user_id,
        created_at=speaker.created_at,
    )


def _no_such_speaker_error() -> HTTPException:
    return HTTPException(status_code=404, detail="no such household member")


def _display_name_conflict_error() -> HTTPException:
    return HTTPException(status_code=409, detail="a member with this display name already exists")


def _unknown_linked_user_error() -> HTTPException:
    return HTTPException(status_code=422, detail="linked_user_id does not match an existing user")


@router.get("/api/speakers")
async def list_speakers(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[SpeakerResponse]:
    repo: SpeakerRepository = request.app.state.speaker_repo
    return [_to_response(speaker) for speaker in await repo.list_speakers()]


@router.post("/api/speakers", status_code=201)
async def create_speaker(
    payload: SpeakerCreateRequest,
    request: Request,
    _admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> SpeakerResponse:
    repo: SpeakerRepository = request.app.state.speaker_repo

    if payload.linked_user_id is not None:
        account_repo = request.app.state.account_repo
        linked_user = await account_repo.get_user_by_id(payload.linked_user_id)
        if linked_user is None:
            raise _unknown_linked_user_error()

    try:
        speaker = await repo.create_speaker(
            display_name=payload.display_name,
            linked_user_id=payload.linked_user_id,
            created_at=datetime.now(timezone.utc),
        )
    except IntegrityError:
        # A unique-violation on speakers.display_name -- surfaced as a
        # named refusal, never an unhandled 500, matching
        # `routes/accounts.py`'s own `create_user` IntegrityError handling.
        raise _display_name_conflict_error() from None

    return _to_response(speaker)


@router.delete("/api/speakers/{speaker_id}", status_code=204)
async def delete_speaker(
    speaker_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> None:
    """A real `DELETE` (D-03) -- the member row and every one of its
    embedding rows are gone for good, not merely flagged. 404 for an
    unknown id; a second delete of the same id also 404s, since the row no
    longer exists to look up."""
    repo: SpeakerRepository = request.app.state.speaker_repo
    existing = await repo.get_speaker(speaker_id)
    if existing is None:
        raise _no_such_speaker_error()

    await repo.delete_speaker(speaker_id)
