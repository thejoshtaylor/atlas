"""The wake-hit tuning surface over HTTP: one read of what was heard, one
write of what should count (D-13, D-14a, D-15, D-16, DBG-05).

A router of its own, not an extension of `routes/sessions.py`: these two
routes read a database table and mutate running objects, where the
sessions router reads a directory tree -- keeping them apart keeps one
file's concerns single. Every route here requires `Role.OPERATOR`, the
same trust level D-03 places every Phase 8 screen at (`routes/providers.py`'s
own shape for the request/response and named-refusal conventions this
module follows).

`PUT /api/wake-threshold` stores first, then applies (T-08-29's own
"repudiation" mitigation reads the same updated-by/updated-at columns
`SettingRow` already carries for every other row in that store): a failed
write never leaves the running system disagreeing with what was saved.
Applying means calling `SourceRunner.set_wake_threshold` on every entry in
`app.state.source_runners` -- that list is assigned once at startup and
lives for the process's life (`app.py`'s own `lifespan`), so this is the
only place the running gates are reachable through. No restart (D-15):
this differs from Phase 7's provider slots on purpose -- a provider is a
client the turn pipeline is constructed around, and a threshold is a
float the detector reads on every hit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.db.repository import SettingsRepository, WakeEventRepository
from spire_voice.session.retention import parse_session_timestamp

router = APIRouter(tags=["wake"])

# The general operator-editable settings store's one key for this plan
# (the same store `routes/wizard.py`'s `AUDIO_SOURCE_SETTING_KEY` already
# writes under) -- imported by `app.py`'s `lifespan` so the boot-time read
# and this route's write can never drift onto two different key strings.
WAKE_THRESHOLD_SETTING_KEY = "wake_threshold"

# The only engine this project ships whose score is genuinely graded
# (D-14a): `wake/vosk_engine.py` reports `score=1.0` for every hit,
# because a match against a constrained grammar is a categorical hit, not
# a graded one. Derived from the configured engine's own name here --
# never inferred from the spread of scores already stored, which would
# misreport a graded engine that has so far only heard one voice.
_GRADING_ENGINES: frozenset[str] = frozenset({"openwakeword"})

# The most events one response will carry. The table's real bound is the
# retention sweep, which removes wake events on the same `retain_days` as
# the recordings they describe (WR-06); this is the backstop for the window
# between two sweeps in a household with an ambient noise source, where a
# television can trigger the gate all day. Generous on purpose -- the
# screen fetches once and re-partitions locally on every drag (08-PATTERNS.md's
# own no-network-call-per-drag discipline), so it needs the history, not a
# page of it -- and never silent: `capped` says so when it bites.
MAX_WAKE_EVENTS_IN_RESPONSE = 2000


def _engine_grades(engine: str) -> bool:
    return engine in _GRADING_ENGINES


def _count_sessions_not_yet_scored(session_dir: str, earliest_recorded_at: "datetime | None") -> int:
    """Session directories under `session_dir` whose parsed timestamp
    predates `earliest_recorded_at` -- D-16's separately counted
    "not scored" figure. `None` (no wake event recorded at all) means
    every session predates it, matching `_list_wake_events_response`'s own
    call site below. A directory whose name does not parse is skipped,
    the same discipline `routes/sessions.py::_list_session_directories`
    already applies."""
    try:
        resolved_root = Path(session_dir).resolve()
    except OSError:
        return 0
    if not resolved_root.is_dir():
        return 0
    count = 0
    for entry in resolved_root.iterdir():
        if not entry.is_dir():
            continue
        parsed = parse_session_timestamp(entry.name)
        if parsed is None:
            continue
        if earliest_recorded_at is None or parsed < earliest_recorded_at:
            count += 1
    return count


# --- Response models --------------------------------------------------------


class WakeEventResponse(BaseModel):
    id: int
    source: str
    engine: str
    score: "float | None"
    allowed: bool
    block_reason: "str | None"
    recorded_at: datetime


class WakeEventsResponse(BaseModel):
    events: list[WakeEventResponse]
    # The three facts the screen cannot derive for itself (D-14a, D-15,
    # D-16) -- read fresh at response-build time, never a stored copy.
    engine: str
    engine_grades: bool
    threshold: float
    not_scored_session_count: int
    # True when older wake events exist that this response does not carry
    # (`MAX_WAKE_EVENTS_IN_RESPONSE`). The screen states it rather than
    # presenting a partial history as a whole one -- the same discipline
    # D-16 applies to the sessions that could not be scored at all.
    capped: bool


class SetWakeThresholdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The same closed `[0.0, 1.0]` range `OpenWakeWordConfig.from_config`
    # already enforces for the graded engine's own threshold
    # (`config.py`) -- reused here on the request model itself rather than
    # a second, independently written bound that could disagree with it
    # (T-08-26).
    threshold: float = Field(ge=0.0, le=1.0)


class SetWakeThresholdResponse(BaseModel):
    threshold: float


# --- Routes -----------------------------------------------------------


@router.get("/api/wake-events")
async def list_wake_events(
    request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> WakeEventsResponse:
    wake_event_repo: WakeEventRepository = request.app.state.wake_event_repo
    config = request.app.state.config

    # The recorded set as a whole, up to a stated bound: the tuning screen
    # fetches once and re-partitions locally on every drag (08-PATTERNS.md's
    # own "no-network-call-per-drag" discipline), which needs every scored
    # attempt on hand, not merely the newest few. One row past the bound is
    # asked for so the response can say whether the bound bit, without a
    # second count query.
    events = await wake_event_repo.list_wake_events(limit=MAX_WAKE_EVENTS_IN_RESPONSE + 1)
    capped = len(events) > MAX_WAKE_EVENTS_IN_RESPONSE
    if capped:
        events = events[:MAX_WAKE_EVENTS_IN_RESPONSE]

    # `list_wake_events` returns newest first, so the last element is the
    # oldest -- the earliest recorded wake attempt this D-16 count is
    # measured against. When `capped`, that is the earliest attempt *being
    # shown*, not the earliest on record, so the count runs high rather
    # than low: it over-reports how much of the operator's history is
    # unscored. That is the safe direction for D-16, whose whole purpose is
    # to stop a threshold looking better-evidenced than it is -- and
    # `capped` is on the response so the screen never presents the number
    # without the reason.
    earliest_recorded_at = events[-1].recorded_at if events else None
    not_scored_session_count = _count_sessions_not_yet_scored(
        config.session.dir, earliest_recorded_at
    )

    # The live threshold in force, read from the first running source
    # rather than from configuration -- so the number the screen shows is
    # the number the gate is actually using after any live change (D-15).
    source_runners = request.app.state.source_runners
    threshold = source_runners[0].wake_threshold

    return WakeEventsResponse(
        events=[
            WakeEventResponse(
                id=event.id,
                source=event.source,
                engine=event.engine,
                score=event.score,
                allowed=event.allowed,
                block_reason=event.block_reason,
                recorded_at=event.recorded_at,
            )
            for event in events
        ],
        engine=config.wake.engine,
        engine_grades=_engine_grades(config.wake.engine),
        threshold=threshold,
        not_scored_session_count=not_scored_session_count,
        capped=capped,
    )


@router.put("/api/wake-threshold")
async def set_wake_threshold(
    payload: SetWakeThresholdRequest,
    request: Request,
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> SetWakeThresholdResponse:
    settings_repo: SettingsRepository = request.app.state.settings_repo

    # Store first, then apply (module docstring): a failed write never
    # leaves the running system disagreeing with what was saved.
    await settings_repo.set_setting(
        WAKE_THRESHOLD_SETTING_KEY,
        payload.threshold,
        updated_by_user_id=user.id,
        updated_at=datetime.now(timezone.utc),
    )

    # Every running source, not just one -- `app.state.source_runners` is
    # assigned once at startup and lives for the process's life, and it is
    # the only reference the running gates are reachable through (D-15).
    for runner in request.app.state.source_runners:
        runner.set_wake_threshold(payload.threshold)

    return SetWakeThresholdResponse(threshold=payload.threshold)
