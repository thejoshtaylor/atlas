"""The follow-up window's own admin-editable length (D-09, T-09-40, T-09-41,
plan 09-07): how long ATLAS listens, with no wake word, for the answer to a
calendar readback or a clarifying question -- plan 09-06's own configured
default (`config.follow_up.window_s`), now overridable live from the
Settings screen.

Mirrors `routes/wizard.py`'s `TIMEZONE_SETTING_KEY` GET/PUT shape and
`routes/wake.py`'s own "a stored value wins over configuration, and applies
live with no restart" precedent (D-15's sibling here) -- one settings-store
key, resolved once at boot by `app.py`'s own `_resolve_follow_up_window_s`
(mirroring that file's `_resolve_wake_threshold`), and read on every
follow-up window through `app.state.follow_up_window_s`, which both
`SourceRunner` constructions in `app.py` read through their own
`follow_up_window_s` callable. T-09-41: the 3-15 second range this route
enforces on the way in is the only thing standing between an admin and a
widened no-wake-word listening window, so it is enforced here directly,
never left to a client's own good behavior.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.config import Config
from atlas.db.repository import SettingsRepository

router = APIRouter(tags=["settings"])

# The general operator-editable settings store's own key for this plan --
# the same store `routes/wizard.py`'s `AUDIO_SOURCE_SETTING_KEY`/
# `TIMEZONE_SETTING_KEY` and `routes/wake.py`'s `WAKE_THRESHOLD_SETTING_KEY`
# already write under. Imported by `app.py`'s `lifespan` so the boot-time
# read and this route's write can never drift onto two different key
# strings.
FOLLOW_UP_WINDOW_SETTING_KEY = "follow_up_window_s"

# `FollowUpConfig.from_config`'s own bounds (`config.py`) -- reused here
# rather than a second, independently written range that could disagree
# with it (the same T-08-26 discipline `routes/wake.py`'s
# `SetWakeThresholdRequest` already follows for its own `[0.0, 1.0]`).
FOLLOW_UP_WINDOW_MIN_S = 3.0
FOLLOW_UP_WINDOW_MAX_S = 15.0


def _out_of_range_error() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=(
            f"the follow-up window must be a number between {FOLLOW_UP_WINDOW_MIN_S:g} and "
            f"{FOLLOW_UP_WINDOW_MAX_S:g} seconds"
        ),
    )


class SetFollowUpWindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Typed `Any`, not `float`: a boolean or a string must be refused with
    # this route's own named-range 400, not pydantic's own type-coercion
    # 422 (a bare `float` annotation would silently accept `True` as
    # `1.0`, since `bool` is a `float`-coercible subtype of `int` in lax
    # mode) -- validated by hand below, the same `isinstance(x, bool) or
    # not isinstance(x, (int, float))` shape `FollowUpConfig.from_config`
    # already uses for the identical two-part check.
    window_s: Any = None


class FollowUpWindowResponse(BaseModel):
    window_s: float
    default_s: float
    resolved_from: str  # "database" | "config"


def _stored_value_in_range(setting: Any) -> "float | None":
    """The stored row's own value, when it is a real, in-range number --
    `None` for no row, a non-numeric value, a boolean, or an out-of-range
    number, all of which this route's own `PUT` can never have produced
    (it validates before ever writing), so any of them reading back means
    the row came from somewhere else entirely."""
    if setting is None:
        return None
    value = setting.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not (FOLLOW_UP_WINDOW_MIN_S <= value <= FOLLOW_UP_WINDOW_MAX_S):
        return None
    return value


async def _follow_up_window_status(request: Request) -> FollowUpWindowResponse:
    config: Config = request.app.state.config
    settings_repo: SettingsRepository = request.app.state.settings_repo
    setting = await settings_repo.get_setting(FOLLOW_UP_WINDOW_SETTING_KEY)
    stored = _stored_value_in_range(setting)
    return FollowUpWindowResponse(
        window_s=request.app.state.follow_up_window_s,
        default_s=config.follow_up.window_s,
        resolved_from="database" if stored is not None else "config",
    )


@router.get("/api/settings/follow-up-window")
async def get_follow_up_window(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> FollowUpWindowResponse:
    return await _follow_up_window_status(request)


@router.put("/api/settings/follow-up-window")
async def set_follow_up_window(
    payload: SetFollowUpWindowRequest,
    request: Request,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> FollowUpWindowResponse:
    value = payload.window_s
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not (FOLLOW_UP_WINDOW_MIN_S <= value <= FOLLOW_UP_WINDOW_MAX_S)
    ):
        raise _out_of_range_error()
    value = float(value)

    settings_repo: SettingsRepository = request.app.state.settings_repo
    await settings_repo.set_setting(
        FOLLOW_UP_WINDOW_SETTING_KEY,
        value,
        updated_by_user_id=admin.id,
        updated_at=datetime.now(timezone.utc),
    )

    # Live effect, no restart (D-09/T-09-41): both `SourceRunner`
    # constructions in `app.py` read this exact attribute through their
    # own `follow_up_window_s` callable, so this write reaches the very
    # next window this process opens with nothing else to notify.
    request.app.state.follow_up_window_s = value

    return await _follow_up_window_status(request)
