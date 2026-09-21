"""Session review over HTTP: list recorded turns, straight off disk
(D-01, D-02, WEB-07).

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
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.config import SessionConfig
from spire_voice.session.recorder import EVENTS_FILENAME, TIMING_FILENAME
from spire_voice.session.retention import parse_session_timestamp

router = APIRouter(tags=["sessions"])

# A directory whose `timing.json` is missing or unreadable is listed with
# null fields under this named outcome, never dropped -- a half-written
# turn is a real fact, and hiding it would make a crashed turn
# indistinguishable from one that never happened (the same reasoning
# `SessionRecorder` gives for creating its directory eagerly).
_INCOMPLETE_OUTCOME = "recording_incomplete"


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


# --- Filesystem helpers ------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_events(directory: Path) -> list[dict]:
    return _read_jsonl(directory / EVENTS_FILENAME)


def _last_reply_text(events: list[dict]) -> "str | None":
    reply_text: "str | None" = None
    for event in events:
        if event.get("type") == "reply.text":
            reply_text = event.get("text")
    return reply_text


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


def _session_summary(name: str, started_at: datetime, directory: Path) -> SessionSummaryResponse:
    timing_path = directory / TIMING_FILENAME
    try:
        timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SessionSummaryResponse(
            id=name,
            started_at=started_at,
            turn_outcome=_INCOMPLETE_OUTCOME,
            reply_text=None,
            duration_ms=None,
            has_audio=False,
        )

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


# --- Routes ------------------------------------------------------------------


@router.get("/api/sessions")
async def list_sessions(
    request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> SessionsListResponse:
    session_config: SessionConfig = request.app.state.config.session
    directories = _list_session_directories(Path(session_config.dir))
    sessions = [_session_summary(name, started_at, directory) for name, started_at, directory in directories]
    return SessionsListResponse(sessions=sessions)
