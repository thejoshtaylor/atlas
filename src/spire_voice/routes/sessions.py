"""Session review over HTTP: list recorded turns and read one in full,
straight off disk (D-01, D-02, D-04, WEB-07).

Every route here requires `Role.OPERATOR` (D-03): a recording is reachable
at the same trust level `/policy`, `/macros`, and `/dev-mic` already carry,
never `Role.ADMIN` -- 08-UI-SPEC.md's "Read this first" and D-03 both place
every screen this phase adds at that tier.

There is no session table and this module creates none (D-01): a household
produces a handful of turns a day and `session/retention.py`'s sweep
already bounds the total, so there is no scale problem an index would
solve that the filesystem does not already answer. This module reuses that
module's own directory-name parser (`parse_session_timestamp`,
`SESSION_DIRECTORY_RE`, promoted to public names there for exactly this
import) rather than writing a second one -- two independently written
parsers would disagree at a boundary case, and D-04 depends on both
agreeing about a session's age.

A listed session carries no source label, deliberately. Which source
heard a turn is not among the artifacts `session/recorder.py` writes, and
recording it would be a change to what the recorder captures -- a change
08-CONTEXT.md defers by name. The source label lives on `/live` (D-08),
where the source is genuinely known at the moment the event is emitted;
this module never infers one from the audio format.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict

from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.config import SessionConfig
from spire_voice.session.audio_wrap import wrap_session_audio
from spire_voice.session.recorder import EVENTS_FILENAME, TIMING_FILENAME
from spire_voice.session.retention import parse_session_timestamp
from spire_voice.session.timeline import TIMELINE_FILENAME, regenerate_timeline

logger = logging.getLogger("spire_voice.routes.sessions")

router = APIRouter(tags=["sessions"])

# A directory whose `timing.json` is missing or unreadable is listed with
# null fields under this named outcome, never dropped -- a half-written
# turn is a real fact, and hiding it would make a crashed turn
# indistinguishable from one that never happened (the same reasoning
# `SessionRecorder` gives for creating its directory eagerly).
#
# This is the *only* name the list route gives a damaged directory, and it
# is deliberately the same one the detail and audio routes refuse by
# (`_incomplete_recording_error`): one definition of "a session whose own
# files can be read", honoured two ways, so the list can never advertise a
# session the single-session routes cannot open. The definition is
# `_read_timing_payload` below. Anything else that goes wrong while
# summarising one directory lands here too -- one damaged directory is one
# damaged row, never a failed list.
_INCOMPLETE_OUTCOME = "recording_incomplete"


# --- Named refusals (`routes/plugins.py`'s own house convention; D-04's
# --- first-class-"unknown" discipline, `routes/conflict.py`) --------------


def _unrecognised_session_id_error(session_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{session_id!r} is not a recognised session id")


def _removed_by_retention_error(session_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"session {session_id!r} has been removed by the retention sweep",
    )


def _session_not_found_error(session_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"no session with id {session_id!r}")


def _audio_not_recorded_error(session_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"session {session_id!r} has no recorded audio",
    )


# --- Response models --------------------------------------------------------


class SessionSummaryResponse(BaseModel):
    id: str
    started_at: datetime
    turn_outcome: str
    reply_text: "str | None"
    duration_ms: "float | None"
    has_audio: bool


class SessionsListResponse(BaseModel):
    sessions: list[SessionSummaryResponse]


class TimelineEntryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    ts: float
    kind: str
    # The one field this route adds beyond what `render_timeline` itself
    # writes: the entry's own timestamp minus the turn's start mark, both
    # drawn from the same `time.monotonic()` clock domain -- one clock
    # compared against itself, never a re-sort against a second one.
    offset_s: "float | None" = None


class SessionDetailResponse(BaseModel):
    id: str
    started_at: datetime
    turn_outcome: str
    transcript: "str | None"
    reply_text: "str | None"
    stage_durations_ms: dict
    end_of_speech_to_first_audio_ms: "float | None"
    end_of_speech_to_answer_audio_ms: "float | None"
    audio_format: "dict | None"
    has_audio: bool
    timeline: list[TimelineEntryResponse]


# --- Filesystem helpers ------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Every JSON object on its own line. A line that does not parse as one
    is skipped, not raised.

    `events.jsonl` is appended one line at a time while the turn is still
    running (`SessionRecorder.record_event`), so a power loss, an OOM kill
    or a container stop mid-append leaves exactly one partial line behind.
    That partial line is a real fact about one turn; it is not a reason to
    refuse every *other* session in the list, which is what raising here
    did. The skip is logged with the path and a count and never the
    content -- `session/retention.py`'s privacy boundary, which this
    module reads the same directories under, holds here too.
    """
    if not path.is_file():
        return []
    entries: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(parsed, dict):
            skipped += 1
            continue
        entries.append(parsed)
    if skipped:
        logger.warning(
            "session review: skipped %d unreadable line(s) in %s",
            skipped,
            path,
        )
    return entries


def _read_events(directory: Path) -> list[dict]:
    return _read_jsonl(directory / EVENTS_FILENAME)


def _last_reply_text(events: list[dict]) -> "str | None":
    reply_text: "str | None" = None
    for event in events:
        if event.get("type") == "reply.text":
            reply_text = event.get("text")
    return reply_text


def _transcript_before_stt_final(events: list[dict], stt_final_at: "float | None") -> "str | None":
    """The text of the last `transcript.partial` event whose recorded
    instant is at or before `stt_final_at` -- 08-UI-SPEC.md's own flagged
    assumption: there is no `transcript.final` event today, and every
    partial recorded to `events.jsonl` is exactly that. `None` when no
    partial precedes the mark (or the mark itself was never reached) --
    the screen says so rather than showing a blank."""
    if stt_final_at is None:
        return None
    transcript: "str | None" = None
    for event in events:
        if event.get("type") != "transcript.partial":
            continue
        recorded_at = event.get("recorded_at")
        if recorded_at is None or recorded_at > stt_final_at:
            continue
        transcript = event.get("text")
    return transcript


def _sum_durations(stage_durations_ms: dict) -> "float | None":
    values = [value for value in stage_durations_ms.values() if value is not None]
    if not values:
        return None
    return sum(values)


def _has_audio(directory: Path, audio_format: "dict | None") -> bool:
    if not audio_format:
        return False
    encoding = audio_format.get("encoding")
    if not encoding:
        return False
    return (directory / f"audio.{encoding}").is_file()


def _list_session_directories(root: Path) -> "list[tuple[str, datetime, Path]]":
    """Every directory under `root` whose name the recorder wrote, newest
    first. A directory whose name does not parse is skipped -- the same
    discipline `sweep_expired_sessions` already applies to one it does not
    recognise."""
    try:
        resolved_root = root.resolve()
    except OSError:
        return []
    if not resolved_root.is_dir():
        return []
    candidates: "list[tuple[str, datetime, Path]]" = []
    for entry in resolved_root.iterdir():
        if not entry.is_dir():
            continue
        parsed = parse_session_timestamp(entry.name)
        if parsed is None:
            continue
        candidates.append((entry.name, parsed, entry))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _read_timing_payload(directory: Path) -> "dict | None":
    """`timing.json` read back as the object `TurnTimings` wrote, or `None`
    when it cannot be read as one.

    This is the single definition of "this session's own files can be
    read" that every route in this module shares. `_session_summary` turns
    a `None` into an `_INCOMPLETE_OUTCOME` row; `_require_timing_payload`
    turns the same `None` into a named refusal. Neither re-derives the
    check, so the list and the single-session routes cannot come to
    disagree about which recordings are openable.
    """
    try:
        payload = json.loads((directory / TIMING_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _incomplete_summary(name: str, started_at: datetime) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        id=name,
        started_at=started_at,
        turn_outcome=_INCOMPLETE_OUTCOME,
        reply_text=None,
        duration_ms=None,
        has_audio=False,
    )


def _session_summary(name: str, started_at: datetime, directory: Path) -> SessionSummaryResponse:
    try:
        timing_payload = _read_timing_payload(directory)
        if timing_payload is None:
            return _incomplete_summary(name, started_at)

        events = _read_events(directory)
        stage_durations_ms = timing_payload.get("stage_durations_ms") or {}
        return SessionSummaryResponse(
            id=name,
            started_at=started_at,
            turn_outcome=timing_payload.get("turn_outcome", _INCOMPLETE_OUTCOME),
            reply_text=_last_reply_text(events),
            duration_ms=_sum_durations(stage_durations_ms),
            has_audio=_has_audio(directory, timing_payload.get("audio_format")),
        )
    except (OSError, ValueError, TypeError):
        # The blast radius of one unreadable directory is that one row.
        # `list_sessions` builds every row from a separate directory, so
        # there is no failure here that says anything true about the other
        # sessions -- and an operator whose whole Sessions screen goes to
        # `ErrorState` because one turn crashed mid-write cannot reach any
        # recording at all until someone edits a file on the server.
        # `json.JSONDecodeError` and pydantic's own `ValidationError` are
        # both `ValueError`, so a malformed field value lands here too.
        logger.warning(
            "session review: session %s could not be summarised; listing it as %s",
            name,
            _INCOMPLETE_OUTCOME,
            exc_info=True,
        )
        return _incomplete_summary(name, started_at)


def _resolve_session_directory(root: Path, session_id: str) -> "tuple[Path, datetime]":
    """Match `session_id` against the recorder's own directory-name shape
    before it is ever joined onto a path, then confirm the resolved path
    is genuinely a direct child of the resolved root -- the same two-step
    `sweep_expired_sessions` already performs before it deletes anything.
    A literal traversal segment and a percent-encoded one both fail this
    containment check identically, because both are resolved through the
    same `Path.resolve()` call before comparison -- there is nothing
    percent-decoding could do here that changes which check runs."""
    parsed = parse_session_timestamp(session_id)
    if parsed is None:
        raise _unrecognised_session_id_error(session_id)
    resolved_root = root.resolve()
    candidate = (resolved_root / session_id).resolve()
    if candidate.parent != resolved_root:
        raise _unrecognised_session_id_error(session_id)
    return candidate, parsed


def _resolve_readable_session_directory(session_config: SessionConfig, session_id: str) -> "tuple[Path, datetime]":
    """The full three-way D-04 check every route that reads one session's
    own files needs: shape-validated and traversal-contained
    (`_resolve_session_directory`), then not yet swept by retention, then
    actually present on disk. `get_session_detail` and the audio route
    below both call this rather than each re-deriving the same three
    checks -- exactly the "one code path, one confinement check" this
    module's own docstring requires (T-08-16)."""
    directory, parsed_at = _resolve_session_directory(Path(session_config.dir), session_id)

    age = datetime.now(timezone.utc) - parsed_at
    if age > timedelta(days=session_config.retain_days):
        raise _removed_by_retention_error(session_id)

    if not directory.is_dir():
        raise _session_not_found_error(session_id)

    return directory, parsed_at


def _session_detail(session_id: str, started_at: datetime, directory: Path) -> SessionDetailResponse:
    timing_payload = json.loads((directory / TIMING_FILENAME).read_text(encoding="utf-8"))
    events = _read_events(directory)

    timeline_path = directory / TIMELINE_FILENAME
    if not timeline_path.is_file():
        # `regenerate_timeline` exists precisely to prove the rendering is
        # derivable from `events.jsonl` and `timing.json` alone (D-02).
        regenerate_timeline(directory)
    raw_timeline = _read_jsonl(timeline_path)

    turn_started_at = timing_payload.get("turn_started_at")
    timeline = [
        TimelineEntryResponse(
            offset_s=(entry["ts"] - turn_started_at) if turn_started_at is not None else None,
            **entry,
        )
        for entry in raw_timeline
    ]

    stage_durations_ms = timing_payload.get("stage_durations_ms") or {}
    audio_format = timing_payload.get("audio_format")

    return SessionDetailResponse(
        id=session_id,
        started_at=started_at,
        turn_outcome=timing_payload.get("turn_outcome", _INCOMPLETE_OUTCOME),
        transcript=_transcript_before_stt_final(events, timing_payload.get("stt_final_at")),
        reply_text=_last_reply_text(events),
        stage_durations_ms=stage_durations_ms,
        end_of_speech_to_first_audio_ms=timing_payload.get("end_of_speech_to_first_audio_ms"),
        end_of_speech_to_answer_audio_ms=timing_payload.get("end_of_speech_to_answer_audio_ms"),
        audio_format=audio_format,
        has_audio=_has_audio(directory, audio_format),
        timeline=timeline,
    )


# --- Routes ------------------------------------------------------------------


@router.get("/api/sessions")
async def list_sessions(
    request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> SessionsListResponse:
    session_config: SessionConfig = request.app.state.config.session
    directories = _list_session_directories(Path(session_config.dir))
    sessions = [_session_summary(name, started_at, directory) for name, started_at, directory in directories]
    return SessionsListResponse(sessions=sessions)


@router.get("/api/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    request: Request,
    _user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> SessionDetailResponse:
    session_config: SessionConfig = request.app.state.config.session
    directory, parsed_at = _resolve_readable_session_directory(session_config, session_id)

    return _session_detail(session_id, parsed_at, directory)


@router.get("/api/sessions/{session_id}/audio")
async def get_session_audio(
    session_id: str,
    request: Request,
    _user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> Response:
    """The recording, wrapped as a WAV a browser can play, served whole in
    one response (D-11) -- no range handling, deliberately: a range
    request header changes nothing about what is returned, and a test
    below asserts that directly so a later change that adds range support
    has to face that test rather than slipping in quietly.

    `DEFAULT_ALAW_WRAPPING` (`session/audio_wrap.py`) stayed on the safe
    PCM16-decoded default: the Task 3 checkpoint of plan 08-02 was
    answered on the strength of PCM16's universal browser support, not on
    the outcome of an actual browser test against A-law-tagged WAV -- that
    question remains open (see `audio_wrap.py`'s own module docstring and
    `scripts/dev-write-sample-wav.py`, which exists to settle it later).
    """
    session_config: SessionConfig = request.app.state.config.session
    directory, _parsed_at = _resolve_readable_session_directory(session_config, session_id)

    timing_payload = json.loads((directory / TIMING_FILENAME).read_text(encoding="utf-8"))
    audio_format = timing_payload.get("audio_format")
    if not _has_audio(directory, audio_format):
        raise _audio_not_recorded_error(session_id)

    encoding = audio_format["encoding"]
    raw = (directory / f"audio.{encoding}").read_bytes()
    wav_bytes = wrap_session_audio(encoding, audio_format["sample_rate"], raw)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Length": str(len(wav_bytes))},
    )
