"""Admin routes to add a voice from a recorded edge turn (quick task 260929-j08).

The Speakers page lists recent edge turns that no member matched, plays
the trimmed speech, and assigns it to a member as a retroactive clip. See
`speaker_id/retroactive.py` for the store rules and the trim.

Every route needs `Role.ADMIN`: the audio and the embeddings are biometric
data. The inbox response carries no speaker name and no speaker id. A
speaker label never grants permission (D-15), and no route here reaches
policy, the denylist, or the brain.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, field_validator, model_validator

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.config import Config
from atlas.db.speaker_repository import SpeakerRepository
from atlas.session.audio_wrap import wrap_pcm16_as_wav

# Private helpers of `routes/sessions.py`, imported so that one confinement
# and retention check serves both modules.
from atlas.routes.sessions import (
    _audio_not_recorded_error,
    _has_audio,
    _list_session_directories,
    _read_events,
    _read_timing_payload,
    _require_timing_payload,
    _resolve_readable_session_directory,
    _transcript_before_stt_final,
)
from atlas.routes.speakers import (
    _no_such_speaker_error,
    create_speaker_or_409,
    validate_display_name,
)
from atlas.speaker_id.embedding import SAMPLE_RATE
from atlas.speaker_id.enrollment import ClipStore, EnrollmentError, _too_short_error
from atlas.speaker_id.retroactive import (
    RetroactiveAssignResult,
    assigned_session_ids,
    is_edge_turn,
    resolve_embedding_worker,
    retroactive_lock,
    store_retroactive_clip,
    trim_turn_speech,
)
from atlas.transports.edge import FRAME_SAMPLES

logger = logging.getLogger("atlas.routes.speaker_inbox")

router = APIRouter(tags=["speakers"])

VOICE_INBOX_LIMIT = 50
_TRANSCRIPT_LIMIT = 120
_BYTES_PER_SAMPLE = 2


class VoiceInboxItemResponse(BaseModel):
    session_id: str
    started_at: datetime
    transcript: "str | None"
    speech_ms: "float | None"
    score: "float | None"
    blocked: bool


class VoiceInboxResponse(BaseModel):
    items: "list[VoiceInboxItemResponse]"


class VoiceAssignRequest(BaseModel):
    speaker_id: "int | None" = None
    new_speaker_name: "str | None" = None

    @field_validator("new_speaker_name")
    @classmethod
    def _validate_new_speaker_name(cls, value: "str | None") -> "str | None":
        return None if value is None else validate_display_name(value)

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "VoiceAssignRequest":
        if (self.speaker_id is None) == (self.new_speaker_name is None):
            raise ValueError("give exactly one of speaker_id and new_speaker_name")
        return self


class VoiceAssignResponse(BaseModel):
    speaker_id: int
    phrase_index: int
    speech_ms: float


def _http_error(exc: EnrollmentError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


def _last_speaker_event(events: "list[dict]") -> "dict | None":
    result: "dict | None" = None
    for event in events:
        if event.get("type") == "speaker.result":
            result = event
    return result


def _number_or_none(value: Any) -> "float | None":
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _snippet(text: "str | None") -> "str | None":
    if text is None:
        return None
    if len(text) <= _TRANSCRIPT_LIMIT:
        return text
    return text[:_TRANSCRIPT_LIMIT] + "…"


def _inbox_item(
    config: Config, name: str, started_at: datetime, directory: Path, assigned: "set[str]"
) -> "VoiceInboxItemResponse | None":
    if name in assigned:
        return None
    timing = _read_timing_payload(directory)
    if timing is None:
        return None
    if timing.get("turn_outcome") == "wake_unverified":
        return None
    # No pre-roll means a follow-up turn. Its recording can hold reply readback.
    if not timing.get("preroll_bytes", 0) > 0:
        return None
    audio_format = timing.get("audio_format")
    events = _read_events(directory)
    speaker_event = _last_speaker_event(events)
    if speaker_event is None or speaker_event.get("status") != "unknown":
        return None
    if not is_edge_turn(audio_format, speaker_event, edge_channels=config.edge.channels):
        return None
    if not _has_audio(directory, audio_format):
        return None

    minimum_ms = config.speaker_id.min_enrollment_speech_ms
    frame_bytes = config.edge.channels * _BYTES_PER_SAMPLE
    recorded_ms = (directory / "audio.pcm").stat().st_size / frame_bytes / SAMPLE_RATE * 1000.0
    if recorded_ms < minimum_ms:
        return None
    speech_ms = _number_or_none(speaker_event.get("speech_ms"))
    if speech_ms is not None and speech_ms < minimum_ms:
        return None

    return VoiceInboxItemResponse(
        session_id=name,
        started_at=started_at,
        transcript=_snippet(_transcript_before_stt_final(events, timing.get("stt_final_at"))),
        speech_ms=speech_ms,
        score=_number_or_none(speaker_event.get("score")),
        blocked=speaker_event.get("blocked") is True,
    )


def _collect_inbox(config: Config, clip_store: "ClipStore | None") -> "list[VoiceInboxItemResponse]":
    assigned = assigned_session_ids(clip_store) if clip_store is not None else set()
    items: "list[VoiceInboxItemResponse]" = []
    for name, started_at, directory in _list_session_directories(config.session):
        try:
            item = _inbox_item(config, name, started_at, directory, assigned)
        except (OSError, ValueError, TypeError):
            continue
        if item is None:
            continue
        items.append(item)
        if len(items) >= VOICE_INBOX_LIMIT:
            break
    return items


def _read_edge_turn(config: Config, session_id: str) -> bytes:
    """The raw recording of one edge turn, read once into memory. A
    retention sweep can remove the directory at any time."""
    directory, _started_at = _resolve_readable_session_directory(config.session, session_id)
    timing = _require_timing_payload(directory, session_id)
    audio_format = timing.get("audio_format")
    if not _has_audio(directory, audio_format):
        raise _audio_not_recorded_error(session_id)
    speaker_event = _last_speaker_event(_read_events(directory))
    if not is_edge_turn(audio_format, speaker_event, edge_channels=config.edge.channels):
        raise HTTPException(
            status_code=409, detail="this turn was not recorded from the edge microphone at 16 kHz"
        )
    if config.edge.asr_channel is None:
        raise HTTPException(status_code=409, detail="edge.asr_channel is not set")
    try:
        return (directory / "audio.pcm").read_bytes()
    except OSError:
        raise _audio_not_recorded_error(session_id) from None


def _trim(config: Config, raw: bytes) -> "tuple[bytes, float]":
    speaker_config = config.speaker_id
    return trim_turn_speech(
        raw,
        channels=config.edge.channels,
        asr_channel=config.edge.asr_channel,
        frame_samples=FRAME_SAMPLES,
        sample_rate=SAMPLE_RATE,
        speech_rms_floor=speaker_config.speech_rms_floor,
        max_seconds=speaker_config.enrollment_max_phrase_s,
    )


# -- voice inbox (static paths first) --------------------------------------


@router.get("/api/speakers/voice-inbox")
async def list_voice_inbox(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> VoiceInboxResponse:
    config: Config = request.app.state.config
    clip_store = getattr(request.app.state, "speaker_clip_store", None)
    items = await asyncio.to_thread(_collect_inbox, config, clip_store)
    return VoiceInboxResponse(items=items)


@router.get("/api/speakers/voice-inbox/{session_id}/audio")
async def voice_inbox_audio(
    session_id: str, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> Response:
    config: Config = request.app.state.config

    def _read_and_trim() -> "tuple[bytes, float]":
        return _trim(config, _read_edge_turn(config, session_id))

    pcm, _speech_ms = await asyncio.to_thread(_read_and_trim)
    if not pcm:
        raise _http_error(_too_short_error())
    wav_bytes = wrap_pcm16_as_wav(pcm, SAMPLE_RATE, 1)
    return Response(
        content=wav_bytes, media_type="audio/wav", headers={"Content-Length": str(len(wav_bytes))}
    )


@router.post("/api/speakers/voice-inbox/{session_id}/assign")
async def assign_voice(
    session_id: str,
    payload: VoiceAssignRequest,
    request: Request,
    _admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> VoiceAssignResponse:
    state = request.app.state
    config: Config = state.config
    repo: SpeakerRepository = state.speaker_repo
    try:
        worker = resolve_embedding_worker(state, config.speaker_id)
    except EnrollmentError as exc:
        raise _http_error(exc) from None
    clip_store: "ClipStore | None" = getattr(state, "speaker_clip_store", None)
    if clip_store is None:
        raise HTTPException(status_code=409, detail="speaker storage is not configured")

    speaker = None
    if payload.speaker_id is not None:
        speaker = await repo.get_speaker(payload.speaker_id)
        if speaker is None:
            raise _no_such_speaker_error()

    def _read_and_trim() -> "tuple[bytes, float]":
        return _trim(config, _read_edge_turn(config, session_id))

    pcm, speech_ms = await asyncio.to_thread(_read_and_trim)
    if speech_ms < config.speaker_id.min_enrollment_speech_ms:
        raise _http_error(_too_short_error())

    model_id = config.speaker_id.model_id
    assert model_id is not None  # resolve_embedding_worker checked the model.
    async with retroactive_lock(state):
        if session_id in await asyncio.to_thread(assigned_session_ids, clip_store):
            raise HTTPException(status_code=409, detail="this turn is already assigned to a member")
        created = False
        if speaker is None:
            speaker = await create_speaker_or_409(
                repo, display_name=payload.new_speaker_name, linked_user_id=None
            )
            created = True
        try:
            result: RetroactiveAssignResult = await store_retroactive_clip(
                state=state,
                clip_store=clip_store,
                speaker=speaker,
                session_id=session_id,
                pcm16_mono=pcm,
                speech_ms=speech_ms,
                worker=worker,
                speaker_repo=repo,
                model_id=model_id,
            )
        except Exception as exc:
            if created:
                await repo.delete_speaker(speaker.id)
                try:
                    await asyncio.to_thread(clip_store.delete_speaker_dir, speaker.id)
                except OSError:
                    logger.warning("voice assign: could not remove the clip directory for member %s", speaker.id)
            if isinstance(exc, EnrollmentError):
                raise _http_error(exc) from None
            raise

    return VoiceAssignResponse(
        speaker_id=result.speaker_id, phrase_index=result.phrase_index, speech_ms=result.speech_ms
    )
