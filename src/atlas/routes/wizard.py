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

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from atlas_mcp.ha import handle_list_entities
from atlas_mcp.safety import Policy

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.calibration.runner import find_latest_calibration
from atlas.config import Config, SecurityConfig
from atlas.crypto.credentials import (
    CredentialSlot,
    decrypt_credential,
    resolve_credential_source,
)
from atlas.db.repository import (
    AccountRepository,
    CredentialRepository,
    PluginRepository,
    SettingsRepository,
    SetupRepository,
)

router = APIRouter(prefix="/api/wizard", tags=["wizard"])

logger = logging.getLogger("atlas.routes.wizard")

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

# The always-listening sources this application builds
# (`app.py`'s `lifespan`: `resolved_audio_source` branches between
# `CameraAudioSource` and `EdgeAudioSource`). A closed set, the same
# reasoning `CredentialSlot` and `PolicyRuleRow.kind` already apply to
# their own closed sets -- a source name arrives as a code change, never
# as a string a route parameter invents. `"edge"` is the second name this
# comment's own prior revision anticipated (Phase 10, D-15): a Raspberry
# Pi + XVF3800 array, paired through `/api/edge-devices`.
VALID_AUDIO_SOURCES: frozenset[str] = frozenset({"camera", "edge"})

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


class TimezoneRequest(BaseModel):
    """`PUT /api/wizard/timezone`'s own body -- one field, the zone name
    an admin typed. Never trimmed here (260924-h2f Task 2's own
    instruction): a name with leading/trailing whitespace is simply not a
    `zoneinfo` key, and `PUT` refuses it the same way it refuses any other
    unknown name, rather than silently correcting a typo."""

    zone: str


class TimezoneStatusResponse(BaseModel):
    """`GET`/`PUT /api/wizard/timezone`'s own shape -- `zone`/`resolved_from`/
    `warning` are always the boot's own `TimezoneResolution` (never
    re-resolved per request, per that dataclass's own docstring);
    `stored` is the current `settings` row (`None` when nothing has been
    saved through the webapp yet); `applies_live` is always `False` --
    a saved zone applies on the next restart, matching `audio_source`'s
    own `detail.applies_live` (this file's own precedent)."""

    zone: str
    resolved_from: str
    stored: "str | None"
    warning: "str | None"
    applies_live: bool


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


async def _ha_plugin_connection_info(
    plugin_repo: PluginRepository, security: SecurityConfig
) -> tuple[str, str]:
    """`(base_url, token)` for the Home Assistant plugin -- read from
    `plugins`/`plugin_config_values` (plan 06-01, D-01), never from
    `Config.mcp_servers`, which the same plan retires: `HA_URL`/`HA_TOKEN`
    live in the database now, seeded by `alembic/versions/
    0008_plugin_tables.py`, the same single source of truth `PluginManager`
    itself reads to spawn the child this route is probing the reachability
    of. `HA_TOKEN` is decrypted here -- the third of the three server-side
    points D-03 names (IN-04), beside the two on `PluginManager`'s own
    spawn path -- never returned in this
    route's own response body (`check_hub_step`'s own `WizardStepStatus`
    carries only `checked_at`/`entity_count`, unchanged by this plan).
    Returns `("", "")` when no `ha` plugin row exists at all.
    """
    plugins = await plugin_repo.list_plugins()
    ha_plugin = next((plugin for plugin in plugins if plugin.slug == "ha"), None)
    if ha_plugin is None:
        return "", ""
    base_url = ""
    token = ""
    for value in await plugin_repo.get_config_values(ha_plugin.id):
        if value.key == "HA_URL":
            base_url = value.value or ""
        elif value.key == "HA_TOKEN":
            if value.secret and value.ciphertext is not None and value.key_version is not None:
                token = decrypt_credential(value.ciphertext, value.key_version, security)
            else:
                token = value.value or ""
    return base_url, token


async def _probe_home_assistant(base_url: str, token: str, *, client: httpx.AsyncClient) -> int:
    """Call `mcp.atlas_mcp.ha.handle_list_entities` -- the exact read
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
    `atlas.crypto.credentials.resolve_credential_value`'s
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


# 260924-h2f: the settings key an admin's saved time zone lives under
# (`PUT /api/wizard/timezone`), and the text this application shows and
# logs when nothing gives it a zone and the process itself runs in UTC --
# a house told in UTC with no explanation is issue #1's whole complaint.
TIMEZONE_SETTING_KEY = "timezone"

TIMEZONE_UTC_WARNING = (
    "No time zone is set and Home Assistant did not give one. This server runs "
    "in UTC, so ATLAS gives times, dates, and 'at' schedules in UTC. Set the "
    "time zone in the admin webapp under Settings, or set server.timezone in "
    "the configuration file."
)


@dataclass(frozen=True)
class TimezoneResolution:
    """The one resolved house time zone. `lifespan` stores this on
    `app.state.timezone_resolution`, and no other code resolves the zone
    again (260924-h2f).

    `zone` is `None` only for `resolved_from == "process"` -- every other
    source loaded successfully with `zoneinfo`. `name` is always a string
    worth showing an operator: the IANA key for the first three sources, or
    the process's own abbreviated name (or "the local zone") for the last.
    `child_tz` is the value `plugins.manager.PluginManager` writes into
    every stdio MCP child's `TZ` -- the same as `name` for the first three
    sources, the parent process's own `TZ` (or `None`) for the last, so a
    child never disagrees with the parent about which zone "no zone
    resolved" means.
    """

    zone: "ZoneInfo | None"
    name: str
    resolved_from: str
    child_tz: "str | None"
    warning: "str | None"


def _process_zone_is_utc() -> bool:
    """True when this process's own local zone is UTC -- checked by its
    `tzname()`, not by comparing against a literal `"UTC"` string anywhere
    else, so a container whose `TZ` is unset (which Python reports as UTC)
    and one explicitly set to `TZ=UTC` are both caught the same way.
    Called by its module-level name, not inlined into `resolve_timezone`,
    so a test can monkeypatch this one attribute and drive both branches
    with no real clock or environment change.
    """
    return datetime.now().astimezone().tzname() == "UTC"


def _current_process_zone_name() -> str:
    return datetime.now().astimezone().tzname() or "the local zone"


async def _fetch_home_assistant_time_zone(
    base_url: str, token: str, *, client: httpx.AsyncClient
) -> str:
    """`GET {base_url}/api/config`'s own `time_zone` field, built the same
    way `atlas_mcp.ha`'s own `/api/states` call is (same header, same
    bearer-token shape) -- raises on anything that is not a non-empty
    string, which `resolve_timezone` below treats as "Home Assistant did
    not give a usable zone" and falls through from.
    """
    response = await client.get(
        f"{base_url}/api/config", headers={"Authorization": f"Bearer {token}"}
    )
    response.raise_for_status()
    value = response.json()["time_zone"]
    if not isinstance(value, str) or not value:
        raise ValueError(f"time_zone was {value!r}, not a non-empty string")
    return value


async def resolve_timezone(
    config: Config,
    settings_repo: SettingsRepository,
    plugin_repo: PluginRepository,
    *,
    client: httpx.AsyncClient,
) -> TimezoneResolution:
    """`TimezoneResolution` for the house's own zone (260924-h2f, issue
    #1) -- the one function that decides between four sources, in one
    fixed order, mirroring `resolve_audio_source`'s own "one function
    decides" pattern immediately above:

    1. The zone an admin saved through the webapp (`settings` row keyed
       `TIMEZONE_SETTING_KEY`) -- `resolved_from == "database"`. A stored
       value that no longer loads with `zoneinfo` logs a `WARNING` naming
       the key and falls through, rather than raising: a boot must never
       refuse over a setting a route already validated when it was
       written.
    2. `config.server.timezone`, already validated by
       `ServerConfig.from_config` -- `resolved_from == "config"`.
    3. Home Assistant's own `GET /api/config` `time_zone` field, read
       through the `ha` plugin row's connection info
       (`_ha_plugin_connection_info`) -- `resolved_from ==
       "home_assistant"`. Skipped with no request at all when no `HA_URL`
       is configured. Any failure (decrypt, transport, HTTP status, JSON
       shape, an unknown zone name) is caught broadly and logged as a
       `WARNING` naming only the exception's type and message -- never
       the token -- and falls through: Home Assistant being unreachable
       must never stop the boot (D-07's same "a degraded plugin boots
       anyway" posture).
    4. The process's own zone -- `resolved_from == "process"`, `zone`
       `None`. `warning` carries `TIMEZONE_UTC_WARNING` when the process
       itself is running in UTC (`_process_zone_is_utc`), so an operator
       who never configured a zone anywhere is told, rather than silently
       served UTC.

    For sources 1-3, `name` and `child_tz` are the same IANA zone key.
    """
    setting = await settings_repo.get_setting(TIMEZONE_SETTING_KEY)
    if setting is not None and isinstance(setting.value, str) and setting.value:
        try:
            zone = ZoneInfo(setting.value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            logger.warning(
                "stored setting %r (%r) is not a valid time zone: %s -- falling "
                "through to the next source",
                TIMEZONE_SETTING_KEY,
                setting.value,
                exc,
            )
        else:
            return TimezoneResolution(
                zone=zone,
                name=setting.value,
                resolved_from="database",
                child_tz=setting.value,
                warning=None,
            )

    if config.server.timezone:
        zone = ZoneInfo(config.server.timezone)
        return TimezoneResolution(
            zone=zone,
            name=config.server.timezone,
            resolved_from="config",
            child_tz=config.server.timezone,
            warning=None,
        )

    base_url, token = await _ha_plugin_connection_info(plugin_repo, config.security)
    if base_url:
        try:
            tz_name = await _fetch_home_assistant_time_zone(base_url, token, client=client)
            zone = ZoneInfo(tz_name)
        except Exception as exc:  # noqa: BLE001 -- D-07: Home Assistant down is survivable
            logger.warning(
                "could not resolve the time zone from Home Assistant: %s: %s",
                type(exc).__name__,
                exc,
            )
        else:
            return TimezoneResolution(
                zone=zone,
                name=tz_name,
                resolved_from="home_assistant",
                child_tz=tz_name,
                warning=None,
            )

    name = _current_process_zone_name()
    warning = TIMEZONE_UTC_WARNING if _process_zone_is_utc() else None
    child_tz = os.environ.get("TZ") or None
    return TimezoneResolution(
        zone=None, name=name, resolved_from="process", child_tz=child_tz, warning=warning
    )


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


async def _room_status(config: Config, settings_repo: SettingsRepository) -> WizardStepStatus:
    """Complete once the room's own microphone path is proven -- for
    `"camera"` (or no stored choice), by a real echo-path calibration on
    file; for `"edge"` (D-16), calibration is never the gate at all. The
    camera's echo-path calibration feeds only the camera's correlation
    barge-in gate; the edge source's own barge-in listens for a Pi VAD
    start during playback instead (D-16), so a stored calibration -- or
    the lack of one -- says nothing about whether an edge-sourced room is
    ready.
    """
    source, _resolved_from = await resolve_audio_source(config, settings_repo)
    if source == "edge":
        return WizardStepStatus(
            name="room", complete=True, detail={"source": "edge", "calibration": "not_used"}
        )

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
        "room": await _room_status(config, settings_repo),
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
    plugin_repo: PluginRepository = request.app.state.plugin_repo
    setup_repo: SetupRepository = request.app.state.setup_repo

    base_url, token = await _ha_plugin_connection_info(plugin_repo, config.security)

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


async def _timezone_status(request: Request) -> TimezoneStatusResponse:
    """`request.app.state.timezone_resolution` -- the boot's own value,
    read here and never re-resolved per request (`TimezoneResolution`'s
    own docstring) -- paired with whatever `TIMEZONE_SETTING_KEY` reads
    right now, so a `PUT` a moment ago is reflected in `stored` even
    though `zone`/`resolved_from`/`warning` stay the boot's own values
    until the next restart (`applies_live` is always `False`, matching
    `audio_source`)."""
    resolution: TimezoneResolution = request.app.state.timezone_resolution
    settings_repo: SettingsRepository = request.app.state.settings_repo
    setting = await settings_repo.get_setting(TIMEZONE_SETTING_KEY)
    stored = setting.value if (setting is not None and isinstance(setting.value, str) and setting.value) else None
    return TimezoneStatusResponse(
        zone=resolution.name,
        resolved_from=resolution.resolved_from,
        stored=stored,
        warning=resolution.warning,
        applies_live=False,
    )


def _unknown_timezone_error(zone: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"unknown time zone {zone!r} -- use an IANA name such as 'Europe/Berlin'",
    )


@router.get("/timezone")
async def get_timezone(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> TimezoneStatusResponse:
    return await _timezone_status(request)


@router.put("/timezone")
async def set_timezone(
    payload: TimezoneRequest,
    request: Request,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> TimezoneStatusResponse:
    try:
        ZoneInfo(payload.zone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise _unknown_timezone_error(payload.zone) from exc

    settings_repo: SettingsRepository = request.app.state.settings_repo
    await settings_repo.set_setting(
        TIMEZONE_SETTING_KEY,
        payload.zone,
        updated_by_user_id=admin.id,
        updated_at=datetime.now(timezone.utc),
    )
    return await _timezone_status(request)


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
