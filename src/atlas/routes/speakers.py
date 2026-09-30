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

import asyncio
import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictBool, field_validator
from sqlalchemy.exc import IntegrityError

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.config import Config
from atlas.db.speaker_repository import Speaker, SpeakerRepository
from atlas.speaker_id.enrollment import EnrollmentError, enroll_phrase
from atlas.speaker_id.phrases import ENROLLMENT_PHRASES
from atlas.speaker_id.retroactive import RETROACTIVE_MIN_INDEX

logger = logging.getLogger("atlas.routes.speakers")

router = APIRouter(tags=["speakers"])

# D-02: the fixed number of prompted phrases an enrollment needs -- the
# length of the one phrase list this project ships (`speaker_id/phrases.py`),
# never a number restated independently of it.
REQUIRED_ENROLLMENT_PHRASES = len(ENROLLMENT_PHRASES)

# 1-64 characters, a closed set: letters, digits, space, `-`, `_`, `.` --
# the same closed-charset discipline `routes/edge_devices.py::_NAME_RE`
# already establishes for this exact kind of operator-supplied label.
_NAME_RE = re.compile(r"^[A-Za-z0-9 _.-]{1,64}$")


def validate_display_name(value: str) -> str:
    """Trim `value` and check it against the member-name rule. Shared by
    `POST /api/speakers` and the voice inbox assign route."""
    trimmed = value.strip()
    if not _NAME_RE.match(trimmed):
        raise ValueError(
            "display_name must be 1-64 characters of letters, digits, space, '-', '_' "
            "and '.' (after trimming)"
        )
    return trimmed


class SpeakerCreateRequest(BaseModel):
    display_name: str
    linked_user_id: "int | None" = None

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        return validate_display_name(value)


class SpeakerUpdateRequest(BaseModel):
    """`PATCH /api/speakers/{id}` (260929-p12, D-B). One field, strict."""

    model_config = ConfigDict(extra="forbid")

    can_control_home: StrictBool


class SpeakerResponse(BaseModel):
    id: int
    display_name: str
    linked_user_id: "int | None"
    created_at: datetime
    # Prompted phrases only (index under REQUIRED_ENROLLMENT_PHRASES).
    enrolled_phrases: int
    required_phrases: int
    model_id: "str | None"
    # Clips from recorded turns (index 100 or more). Counted apart, so they
    # never end prompted enrollment early.
    retroactive_clips: int = 0
    # D-A: false stops this member's home writes in enforce mode.
    can_control_home: bool = True


def _to_response(
    speaker: Speaker, *, enrolled_phrases: int, model_id: "str | None", retroactive_clips: int = 0
) -> SpeakerResponse:
    return SpeakerResponse(
        id=speaker.id,
        display_name=speaker.display_name,
        linked_user_id=speaker.linked_user_id,
        created_at=speaker.created_at,
        enrolled_phrases=enrolled_phrases,
        required_phrases=REQUIRED_ENROLLMENT_PHRASES,
        model_id=model_id,
        retroactive_clips=retroactive_clips,
        can_control_home=speaker.can_control_home,
    )


async def phrase_counts(repo: SpeakerRepository, model_id: "str | None") -> "dict[int, tuple[int, int]]":
    """`speaker_id -> (prompted phrases, retroactive clips)` under `model_id`.
    Empty when no model is set."""
    if model_id is None:
        return {}
    counts: "dict[int, tuple[int, int]]" = {}
    for row in await repo.list_reference_embeddings(model_id):
        prompted, retroactive = counts.get(row.speaker_id, (0, 0))
        if row.phrase_index < REQUIRED_ENROLLMENT_PHRASES:
            prompted += 1
        elif row.phrase_index >= RETROACTIVE_MIN_INDEX:
            retroactive += 1
        counts[row.speaker_id] = (prompted, retroactive)
    return counts


def _no_such_speaker_error() -> HTTPException:
    return HTTPException(status_code=404, detail="no such household member")


def _display_name_conflict_error() -> HTTPException:
    return HTTPException(status_code=409, detail="a member with this display name already exists")


def _unknown_linked_user_error() -> HTTPException:
    return HTTPException(status_code=422, detail="linked_user_id does not match an existing user")


def _no_such_enrollment_phrase_error() -> HTTPException:
    return HTTPException(status_code=404, detail="no such enrollment phrase")


class EnrollmentPhrasesResponse(BaseModel):
    phrases: "list[str]"


class EnrollmentRequest(BaseModel):
    device_id: int


class EnrollmentResponse(BaseModel):
    speaker_id: int
    phrase_index: int
    speech_ms: float
    enrolled_phrases: int
    required_phrases: int


async def create_speaker_or_409(
    repo: SpeakerRepository, *, display_name: str, linked_user_id: "int | None"
) -> Speaker:
    """Create a member. A duplicate display name is a named 409, never a
    500, matching `routes/accounts.py`'s own `create_user` handling."""
    try:
        return await repo.create_speaker(
            display_name=display_name,
            linked_user_id=linked_user_id,
            created_at=datetime.now(timezone.utc),
        )
    except IntegrityError:
        raise _display_name_conflict_error() from None


@router.get("/api/speakers")
async def list_speakers(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[SpeakerResponse]:
    """`enrolled_phrases`/`model_id` (plan 11-07): read under the currently
    configured model -- with `speaker_id.model` unset, every member reports
    `enrolled_phrases: 0` and `model_id: null` rather than a spurious count
    against no model at all. `enrolled_phrases` counts prompted phrases
    only. `retroactive_clips` counts clips from recordings, so those clips
    never end the prompted enrollment panel early (260929-j08)."""
    repo: SpeakerRepository = request.app.state.speaker_repo
    config: Config = request.app.state.config
    model_id = config.speaker_id.model_id
    counts = await phrase_counts(repo, model_id)
    return [
        _to_response(
            speaker,
            enrolled_phrases=counts.get(speaker.id, (0, 0))[0],
            model_id=model_id,
            retroactive_clips=counts.get(speaker.id, (0, 0))[1],
        )
        for speaker in await repo.list_speakers()
    ]


@router.post("/api/speakers", status_code=201)
async def create_speaker(
    payload: SpeakerCreateRequest,
    request: Request,
    _admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> SpeakerResponse:
    repo: SpeakerRepository = request.app.state.speaker_repo
    config: Config = request.app.state.config

    if payload.linked_user_id is not None:
        account_repo = request.app.state.account_repo
        linked_user = await account_repo.get_user_by_id(payload.linked_user_id)
        if linked_user is None:
            raise _unknown_linked_user_error()

    speaker = await create_speaker_or_409(
        repo, display_name=payload.display_name, linked_user_id=payload.linked_user_id
    )
    return _to_response(speaker, enrolled_phrases=0, model_id=config.speaker_id.model_id)


@router.patch("/api/speakers/{speaker_id}")
async def update_speaker(
    speaker_id: int,
    payload: SpeakerUpdateRequest,
    request: Request,
    _admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> SpeakerResponse:
    """Set a member's home control flag (D-B). The live denied set changes in
    the same request, so the next turn uses the new value with no restart."""
    repo: SpeakerRepository = request.app.state.speaker_repo
    config: Config = request.app.state.config
    speaker = await repo.set_can_control_home(speaker_id, payload.can_control_home)
    if speaker is None:
        raise _no_such_speaker_error()

    context = getattr(request.app.state, "speaker_id_context", None)
    if context is not None:
        if payload.can_control_home:
            context.home_control_denied.discard(speaker_id)
        else:
            context.home_control_denied.add(speaker_id)

    model_id = config.speaker_id.model_id
    counts = await phrase_counts(repo, model_id)
    prompted, retroactive = counts.get(speaker.id, (0, 0))
    return _to_response(speaker, enrolled_phrases=prompted, model_id=model_id, retroactive_clips=retroactive)


def _clip_removal_failed_error() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail=(
            "the member's rows are removed, but the enrollment clips could not be -- they "
            "will be removed at the next start"
        ),
    )


@router.delete("/api/speakers/{speaker_id}", status_code=204)
async def delete_speaker(
    speaker_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> None:
    """A real `DELETE` (D-03) -- the member row and every one of its
    embedding rows are gone for good, not merely flagged. 404 for an
    unknown id; a second delete of the same id also 404s, since the row no
    longer exists to look up.

    Order: the rows first (`delete_speaker`), then the clip directory
    (`ClipStore.delete_speaker_dir`, plan 11-07), then the live reference
    (`ReferenceSet.remove_speaker` when a context exists). Rows go first
    because a row with no clip can never be re-embedded, while a clip
    directory with no row is exactly what the startup reconcile (plan
    11-07 Task 3) removes on its own -- the opposite order would risk a
    row surviving with no way back to its clips. When the clip directory
    cannot be removed, the rows and the live reference are still gone; this
    returns `500` naming that the clips are removed at the next start
    (D-03), not a failed delete to retry, and logs the member id only.
    """
    repo: SpeakerRepository = request.app.state.speaker_repo
    existing = await repo.get_speaker(speaker_id)
    if existing is None:
        raise _no_such_speaker_error()

    await repo.delete_speaker(speaker_id)

    clip_removal_failed = False
    clip_store = getattr(request.app.state, "speaker_clip_store", None)
    if clip_store is not None:
        try:
            await asyncio.to_thread(clip_store.delete_speaker_dir, speaker_id)
        except OSError:
            clip_removal_failed = True
            logger.warning("speaker delete: could not remove the clip directory for member %s", speaker_id)

    context = getattr(request.app.state, "speaker_id_context", None)
    if context is not None and context.references is not None:
        context.references.remove_speaker(speaker_id)
    if context is not None:
        context.home_control_denied.discard(speaker_id)

    if clip_removal_failed:
        raise _clip_removal_failed_error()


@router.get("/api/speakers/enrollment-phrases")
async def list_enrollment_phrases(
    _admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> EnrollmentPhrasesResponse:
    return EnrollmentPhrasesResponse(phrases=list(ENROLLMENT_PHRASES))


@router.post("/api/speakers/{speaker_id}/enrollment/{phrase_index}")
async def record_enrollment_phrase(
    speaker_id: int,
    phrase_index: int,
    payload: EnrollmentRequest,
    request: Request,
    _admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> EnrollmentResponse:
    """Capture one prompted phrase through the real edge microphone
    (D-02) and store its embedding (D-03) -- see `speaker_id/enrollment.py
    ::enroll_phrase` for the capture itself. `404` for an unknown member
    or a phrase index outside the phrase list; every other refusal is an
    `EnrollmentError` this route maps to its own named status/detail."""
    repo: SpeakerRepository = request.app.state.speaker_repo
    speaker = await repo.get_speaker(speaker_id)
    if speaker is None:
        raise _no_such_speaker_error()
    if not (0 <= phrase_index < REQUIRED_ENROLLMENT_PHRASES):
        raise _no_such_enrollment_phrase_error()

    config: Config = request.app.state.config
    try:
        result = await enroll_phrase(
            state=request.app.state,
            speaker=speaker,
            phrase_index=phrase_index,
            device_id=payload.device_id,
            speaker_repo=repo,
            speaker_config=config.speaker_id,
        )
    except EnrollmentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None

    counts = await phrase_counts(repo, config.speaker_id.model_id)
    return EnrollmentResponse(
        speaker_id=result.speaker_id,
        phrase_index=result.phrase_index,
        speech_ms=result.speech_ms,
        enrolled_phrases=counts.get(speaker_id, (0, 0))[0],
        required_phrases=REQUIRED_ENROLLMENT_PHRASES,
    )
