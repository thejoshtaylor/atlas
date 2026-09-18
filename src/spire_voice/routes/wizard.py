"""The first-run wizard's own server state, and a definition of "finished"
that cannot be satisfied by clicking through (WEB-01, WEB-02, WEB-03,
D-08).

A step marked complete without the thing it claims is the failure this
module's whole design exists to prevent. Four of the five steps
(`admin_account`, `provider_set`, `audio_source`, `room`) are recomputed
live, from the real state each one names, on every `GET /api/wizard` --
never from a claim a browser made earlier. This is deliberate: an existing
deployment with `XAI_API_KEY` already set in its environment has a complete
provider set the moment this route is first called, with nobody having
clicked anything, and a calibration taken directly through Phase 2's own
route (or by a previous wizard session) is recognized the same way. Only
`hub` is stored (`setup_steps`'s own row): "a call to Home Assistant
returned successfully" is a point-in-time fact a live read cannot
reconstruct without repeating the network call on every page load, so this
is the one step whose own `POST` route this file exposes.

Every route here sits behind `require_role(Role.ADMIN)` and is exempt from
the application-level setup gate (`auth/dependencies.py`'s
`SETUP_GATE_EXEMPT_PATHS`) -- an exemption from the gate, never from
authentication, so the wizard stays reachable by signing in even while
setup is incomplete (never a lockout, per `03-UI-SPEC.md`'s own "partial:
wizard, abandoned partway" row).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from spire_mcp.ha import handle_list_entities
from spire_mcp.safety import Policy

from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.calibration.runner import find_latest_calibration
from spire_voice.config import Config
from spire_voice.crypto.credentials import (
    CredentialSlot,
    resolve_credential_source,
    resolve_credential_value,
)
from spire_voice.db.repository import (
    AccountRepository,
    CredentialRepository,
    SettingsRepository,
    SetupRepository,
)

router = APIRouter(prefix="/api/wizard", tags=["wizard"])

# The five steps, in the order `03-09-PLAN.md`'s own "Step set, chosen
# under CD-4" section names them. Steps 2-4 (hub/provider_set/
# audio_source) carry no ordering constraint the server enforces between
# themselves -- only this list's own position decides which name is
# reported as "first unfinished," a browser-presentation convenience, not
# a rule.
STEP_ORDER: tuple[str, ...] = ("admin_account", "hub", "provider_set", "audio_source", "room")

# The provider set this phase actually has: the three AI-provider credential
# slots. The Home Assistant slot is checked by the `hub` step instead (a
# live call, not merely "is a value stored") -- CD-4's "one connected hub"
# is a stronger claim than "a token is set," which is why it is not folded
# into this list.
_PROVIDER_SET_SLOTS: tuple[CredentialSlot, ...] = (
    CredentialSlot.STT,
    CredentialSlot.BRAIN,
    CredentialSlot.TTS,
)

# The one always-listening source this application builds today
# (`app.py`'s `lifespan`: `CameraAudioSource` is the only `SourceRunner`
# constructed). A closed set, the same reasoning `CredentialSlot` and
# `PolicyRuleRow.kind` already apply to their own closed sets -- a second
# source name arrives as a code change, never as a string a route parameter
# invents.
VALID_AUDIO_SOURCES: frozenset[str] = frozenset({"camera"})

AUDIO_SOURCE_SETTING_KEY = "audio_source"
DEFAULT_AUDIO_SOURCE = "camera"


class WizardStepStatus(BaseModel):
    """One step's status: whether it is complete, and the evidence its own
    condition produced -- never a claim, always what actually happened
    (`load_policy`/etc.'s own "value object, not a claim" convention).
    """

    name: str
    complete: bool
    detail: dict[str, Any] | None = None


class WizardStatusResponse(BaseModel):
    steps: list[WizardStepStatus]
    first_unfinished_step: str | None


class WizardFinishResponse(BaseModel):
    complete: bool
    completed_at: datetime


class AudioSourceRequest(BaseModel):
    source: str


def _unknown_audio_source_error(source: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"unknown audio source {source!r} -- valid sources are {sorted(VALID_AUDIO_SOURCES)!r}",
    )


def _finish_incomplete_error(outstanding: list[str]) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(
            "setup is not finished -- outstanding steps: " + ", ".join(outstanding)
        ),
    )


class _HubCheckFailed(Exception):
    """Raised by `_probe_home_assistant` with one of three named
    categories -- `unreachable` (a refused/failed connection), `unauthorized`
    (the token was rejected), or `unparseable` (a response came back but
    could not be made sense of as Home Assistant's own API) -- so the route
    can report which of the three sends the operator to a different place
    (Task 2's own instruction), rather than one generic failure."""

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        self.message = message
        super().__init__(message)


def _hub_check_failed_error(exc: _HubCheckFailed) -> HTTPException:
    return HTTPException(status_code=502, detail=f"{exc.category}: {exc.message}")


async def _probe_home_assistant(base_url: str, token: str, *, client: httpx.AsyncClient) -> int:
    """Call `mcp.spire_mcp.ha.handle_list_entities` -- the exact read
    `app.py`'s own catalog fetch performs through the tool host at startup
    -- and return how many entities it listed. Never invents a second
    probe endpoint: a hub that passes this check is a hub the assistant
    can use.

    A read is never denied by `allow_read` regardless of policy content
    (`handle_list_entities`'s own docstring), so a fresh, unrestricted
    `Policy` is enough here -- this call is about reachability, not about
    what a real turn would be allowed to see.
    """
    policy = Policy.from_config(None)
    try:
        entities = await handle_list_entities(policy, client, base_url, token)
    except httpx.TransportError as exc:
        raise _HubCheckFailed("unreachable", str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            raise _HubCheckFailed(
                "unauthorized", f"Home Assistant rejected the token (HTTP {status})"
            ) from exc
        raise _HubCheckFailed("unparseable", f"Home Assistant returned HTTP {status}") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise _HubCheckFailed(
            "unparseable", f"Home Assistant's response could not be parsed: {exc}"
        ) from exc
    return len(entities)


async def resolve_audio_source(config: Config, settings_repo: SettingsRepository) -> tuple[str, str]:
    """`(source, resolved_from)` for the always-listening audio source --
    the one function that decides between a stored `settings` row and this
    application's own shipped default, mirroring
    `spire_voice.crypto.credentials.resolve_credential_value`'s
    database-then-config precedence (03-07 Task 3). `app.py`'s `lifespan`
    and this module's own `_audio_source_status` both call this rather than
    re-deriving the same check -- `grep -rn` for a second place choosing
    between a stored setting and a configuration value is expected to find
    nothing.
    """
    setting = await settings_repo.get_setting(AUDIO_SOURCE_SETTING_KEY)
    if setting is not None and isinstance(setting.value, str) and setting.value:
        return setting.value, "database"
    return DEFAULT_AUDIO_SOURCE, "config"


async def _admin_account_status(account_repo: AccountRepository) -> WizardStepStatus:
    complete = await account_repo.any_user_exists()
    return WizardStepStatus(name="admin_account", complete=complete, detail=None)


async def _hub_status(setup_repo: SetupRepository) -> WizardStepStatus:
    step = await setup_repo.get_step("hub")
    complete = step is not None and step.completed_at is not None
    detail = step.detail if (step is not None and complete) else None
    return WizardStepStatus(name="hub", complete=complete, detail=detail)


async def _provider_set_status(config: Config, credential_repo: CredentialRepository) -> WizardStepStatus:
    missing: list[str] = []
    for slot in _PROVIDER_SET_SLOTS:
        is_set, _source, _updated_at = await resolve_credential_source(slot, credential_repo, config)
        if not is_set:
            missing.append(slot.value)
    complete = not missing
    detail = {"missing": missing} if missing else {"slots": [s.value for s in _PROVIDER_SET_SLOTS]}
    return WizardStepStatus(name="provider_set", complete=complete, detail=detail)


async def _audio_source_status(config: Config, settings_repo: SettingsRepository) -> WizardStepStatus:
    source, resolved_from = await resolve_audio_source(config, settings_repo)
    complete = resolved_from == "database"
    detail = {"source": source, "resolved_from": resolved_from, "applies_live": False}
    return WizardStepStatus(name="audio_source", complete=complete, detail=detail)


async def _room_status(config: Config) -> WizardStepStatus:
    calibration = find_latest_calibration(config.calibration.dir)
    if calibration is None:
        return WizardStepStatus(name="room", complete=False, detail=None)
    detail = {"taken_at": calibration.taken_at.isoformat(), "source": calibration.source}
    return WizardStepStatus(name="room", complete=True, detail=detail)


async def _compute_all_steps(request: Request) -> list[WizardStepStatus]:
    """Recompute every step's status from its own real source, in
    `STEP_ORDER`. Used by both `GET /api/wizard` and `POST
    /api/wizard/finish` so the two routes can never disagree about what
    "complete" means."""
    account_repo: AccountRepository = request.app.state.account_repo
    setup_repo: SetupRepository = request.app.state.setup_repo
    credential_repo: CredentialRepository = request.app.state.credential_repo
    settings_repo: SettingsRepository = request.app.state.settings_repo
    config: Config = request.app.state.config

    by_name = {
        "admin_account": await _admin_account_status(account_repo),
        "hub": await _hub_status(setup_repo),
        "provider_set": await _provider_set_status(config, credential_repo),
        "audio_source": await _audio_source_status(config, settings_repo),
        "room": await _room_status(config),
    }
    return [by_name[name] for name in STEP_ORDER]


@router.get("")
async def get_wizard(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> WizardStatusResponse:
    steps = await _compute_all_steps(request)
    first_unfinished = next((s.name for s in steps if not s.complete), None)
    return WizardStatusResponse(steps=steps, first_unfinished_step=first_unfinished)


@router.post("/steps/hub/check")
async def check_hub_step(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> WizardStepStatus:
    config: Config = request.app.state.config
    credential_repo: CredentialRepository = request.app.state.credential_repo
    setup_repo: SetupRepository = request.app.state.setup_repo

    ha_config = config.mcp_servers.get("ha")
    base_url = ha_config.env.get("HA_URL", "") if ha_config is not None else ""
    token, _source = await resolve_credential_value(CredentialSlot.HOME_ASSISTANT, config, credential_repo)

    if not base_url:
        raise _hub_check_failed_error(
            _HubCheckFailed("unreachable", "no Home Assistant URL is configured (HA_URL)")
        )

    # A test seam, not a production one: a test builds its own throwaway
    # app and sets `app.state.ha_http_client` to a `FakeHomeAssistant`-backed
    # client it owns and closes itself, so this route never closes a client
    # it does not own. Production (`app.py`'s `lifespan`) sets nothing
    # here, so every real hub check opens and closes its own short-lived
    # connection -- an admin-triggered, infrequent action with no reason to
    # hold a persistent client open for its sake alone.
    injected_client = getattr(request.app.state, "ha_http_client", None)
    owns_client = injected_client is None
    client = injected_client if injected_client is not None else httpx.AsyncClient(timeout=10.0)
    try:
        entity_count = await _probe_home_assistant(base_url, token, client=client)
    except _HubCheckFailed as exc:
        raise _hub_check_failed_error(exc) from exc
    finally:
        if owns_client:
            await client.aclose()

    now = datetime.now(timezone.utc)
    detail = {"checked_at": now.isoformat(), "entity_count": entity_count}
    step = await setup_repo.complete_step("hub", detail=detail, completed_at=now)
    return WizardStepStatus(name=step.name, complete=True, detail=step.detail)


@router.put("/audio-source")
async def set_audio_source(
    payload: AudioSourceRequest,
    request: Request,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> WizardStepStatus:
    if payload.source not in VALID_AUDIO_SOURCES:
        raise _unknown_audio_source_error(payload.source)

    settings_repo: SettingsRepository = request.app.state.settings_repo
    config: Config = request.app.state.config
    await settings_repo.set_setting(
        AUDIO_SOURCE_SETTING_KEY,
        payload.source,
        updated_by_user_id=admin.id,
        updated_at=datetime.now(timezone.utc),
    )
    return await _audio_source_status(config, settings_repo)


@router.post("/finish")
async def finish_wizard(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> WizardFinishResponse:
    steps = await _compute_all_steps(request)
    outstanding = [s.name for s in steps if not s.complete]
    if outstanding:
        raise _finish_incomplete_error(outstanding)

    setup_repo: SetupRepository = request.app.state.setup_repo
    now = datetime.now(timezone.utc)
    await setup_repo.mark_setup_complete(completed_at=now)
    return WizardFinishResponse(complete=True, completed_at=now)
