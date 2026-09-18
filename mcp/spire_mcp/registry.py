"""Home Assistant's area, device, label, and entity registries -- fetched
once per child process, refreshed on a bound, and expanded into entity ids
for `ha.py`'s `handle_call_service` to check.

Two things worth stating before any code, the same way `safety.py`'s own
docstring states its rules first:

1. No REST endpoint answers this. `developers.home-assistant.io/docs/api/
   rest/` enumerates every `/api/*` path Home Assistant documents, and none
   of them touch the area, device, or label registry -- confirmed by two
   independent community threads asking for exactly this as a missing
   feature. Only the WebSocket API's `config/*_registry/list` command
   family exposes them (03-RESEARCH.md Pitfall 3). This is a MEDIUM
   confidence finding: corroborated by official documentation and by the
   community, but never falsified against a live Home Assistant instance,
   because none was reachable while this module was written. That is what
   `scripts/probe_ha_registry.py` exists to settle, against the operator's
   own house, before this module's expansion logic is trusted blind.

2. This module gives the child a second protocol to Home Assistant, and
   the child still holds exactly one credential in total: `HA_TOKEN`.
   `HaRegistryClient` authenticates with that same token `ha.py` already
   holds, over a connection to the same host `HA_URL` already names.
   Nothing here reads an environment variable of its own, opens a
   database connection of any kind, or asks the parent process for
   anything the child did not already have -- a plugin child scoped to
   Home Assistant stays scoped to Home Assistant even once it speaks a
   second protocol to it (SAFE-09). `tests/test_ha_tool.py`'s
   hostile-parent-environment test proves this against a real spawned
   child, not just this paragraph.

`HaRegistryClient` is the only stateful thing in this module, and its
state is exactly one cached `RegistrySnapshot` plus the time it was taken.
Everything that turns a snapshot into entity ids -- `expand_area`,
`expand_device`, `expand_label`, `expand_target` -- is a pure function over
that snapshot and the target id, taking no client, no socket, and no
environment, so a test drives the real expansion logic directly against
fixture data.

Reopening a fresh WebSocket connection repeats Home Assistant's full
`auth_required` -> `auth` -> `auth_ok` handshake, a multi-message exchange
every connection pays again. This project's latency budget has no room to
pay it per voice command, so `HaRegistryClient` follows the same "open
once, never per turn" doctrine this codebase already applies to the
`httpx` client, the MCP session, and the `ffmpeg` process: one connection,
opened on first use, cached for `REGISTRY_REFRESH_INTERVAL_S` before the
next call triggers a fresh fetch. A refresh that fails never falls back to
serving the stale snapshot underneath it -- a registry this process could
not reach right now must be refused as unreachable, not answered from
whatever it last happened to know.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import websockets

# Reopening a connection repeats Home Assistant's multi-message auth
# handshake, and this project's latency budget (VOICE-02: already 10.8s
# median against a 1.5s target, per STATE.md) has no room to pay that on
# every voice command. The registry is fetched once per child process and
# refreshed only when it is older than this -- never per call -- matching
# the "open once, never per turn" doctrine already applied to the httpx
# client, the MCP session, and the ffmpeg process. The bound exists
# because a registry is not static: an operator who moves a switch to a
# new area, or adds a device, should not have to restart the assistant for
# voice control to see it.
REGISTRY_REFRESH_INTERVAL_S = 300.0

_REGISTRY_COMMANDS: Mapping[str, str] = {
    "areas": "config/area_registry/list",
    "devices": "config/device_registry/list",
    "labels": "config/label_registry/list",
    "entities": "config/entity_registry/list",
}


class RegistryError(Exception):
    """Base for every named registry failure. Never caught silently -- a
    caller that wants to tell three failures apart catches the three
    subclasses below, not this one."""


class RegistryAuthError(RegistryError):
    """Home Assistant rejected `HA_TOKEN` during the WebSocket handshake.

    Raised by name rather than returning an empty snapshot, for the same
    reason `ha.py`'s own policy loader exits on a malformed block: a
    registry that looks fetched and is empty expands every area to
    nothing, and "expands to nothing" must never be confusable with "the
    house has nothing there".
    """


class RegistryUnavailableError(RegistryError):
    """The registry could not be fetched this call -- a connection
    failure, a rejected command, or a malformed reply. Never answered from
    a stale cached snapshot instead: the caller must refuse the target as
    unreachable, not guess from what this process last happened to know.
    """


class UnknownRegistryTargetError(RegistryError):
    """No area, device, or label with this id exists in the registry that
    was actually fetched -- distinct from `RegistryUnavailableError`,
    because this means the registry answered and the id is simply not in
    it, which is a different spoken reason than "could not be reached"."""

    def __init__(self, kind: str, target_id: str) -> None:
        super().__init__(f"no such {kind}: {target_id}")
        self.kind = kind
        self.target_id = target_id


@dataclass(frozen=True)
class RegistryDevice:
    """One row of `config/device_registry/list`, trimmed to what
    expansion needs: the area it sits in (if any), and the labels it
    carries."""

    device_id: str
    area_id: str | None
    labels: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RegistryEntity:
    """One row of `config/entity_registry/list`, trimmed to what
    expansion needs: its own area override (if any), the device it
    belongs to (if any), and the labels it carries directly."""

    entity_id: str
    area_id: str | None
    device_id: str | None
    labels: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RegistrySnapshot:
    """A frozen picture of four Home Assistant registries, taken at one
    moment. `areas` and `labels` are id sets -- expansion only needs to
    know whether a given id exists, not any other field of the area or
    label row itself. `devices` is keyed by device id, since `entities`
    looks devices up by id to resolve inherited area. `entities` is every
    entity, not filtered to any target -- expansion filters per call.
    """

    areas: frozenset[str]
    devices: dict[str, RegistryDevice]
    labels: frozenset[str]
    entities: tuple[RegistryEntity, ...]
    fetched_at: float

    @classmethod
    def from_raw(cls, raw: Mapping[str, list[dict[str, Any]]], *, fetched_at: float) -> "RegistrySnapshot":
        """Build a snapshot from the four `config/*_registry/list` reply
        bodies, keyed the same way `_REGISTRY_COMMANDS` names them."""
        areas = frozenset(row["area_id"] for row in raw.get("areas", ()))
        labels = frozenset(row["label_id"] for row in raw.get("labels", ()))
        devices = {
            row["id"]: RegistryDevice(
                device_id=row["id"],
                area_id=row.get("area_id"),
                labels=frozenset(row.get("labels") or ()),
            )
            for row in raw.get("devices", ())
        }
        entities = tuple(
            RegistryEntity(
                entity_id=row["entity_id"],
                area_id=row.get("area_id"),
                device_id=row.get("device_id"),
                labels=frozenset(row.get("labels") or ()),
            )
            for row in raw.get("entities", ())
        )
        return cls(areas=areas, devices=devices, labels=labels, entities=entities, fetched_at=fetched_at)


def _effective_area(entity: RegistryEntity, devices: Mapping[str, RegistryDevice]) -> str | None:
    """An entity's *effective* area: its own `area_id` if it set one,
    otherwise its device's `area_id` if it has a device and that device
    has one. Home Assistant's area model has exactly these two levels, and
    `config/entity_registry/list`'s own reply does not resolve the second
    one for the caller (03-RESEARCH.md Pitfall 2) -- reading only the
    entity registry silently drops every entity that inherits its area
    from its device, which is the common case in a real install.
    """
    if entity.area_id is not None:
        return entity.area_id
    if entity.device_id is not None:
        device = devices.get(entity.device_id)
        if device is not None:
            return device.area_id
    return None


def expand_area(snapshot: RegistrySnapshot, area_id: str) -> frozenset[str]:
    """Every entity whose effective area (own, or inherited from its
    device) is `area_id`. Raises `UnknownRegistryTargetError` when
    `area_id` is not in the registry at all; returns an empty set, not an
    error, when the area exists but holds nothing -- the caller
    distinguishes those two outcomes itself."""
    if area_id not in snapshot.areas:
        raise UnknownRegistryTargetError("area", area_id)
    return frozenset(
        entity.entity_id
        for entity in snapshot.entities
        if _effective_area(entity, snapshot.devices) == area_id
    )


def expand_device(snapshot: RegistrySnapshot, device_id: str) -> frozenset[str]:
    """Every entity belonging to `device_id`. Raises
    `UnknownRegistryTargetError` when the device is not in the registry."""
    if device_id not in snapshot.devices:
        raise UnknownRegistryTargetError("device", device_id)
    return frozenset(entity.entity_id for entity in snapshot.entities if entity.device_id == device_id)


def expand_label(snapshot: RegistrySnapshot, label_id: str) -> frozenset[str]:
    """Every entity carrying `label_id` directly, plus every entity of a
    device carrying `label_id` -- a label on a device reaches the whole
    device, the same way an area on a device does. Raises
    `UnknownRegistryTargetError` when the label is not in the registry."""
    if label_id not in snapshot.labels:
        raise UnknownRegistryTargetError("label", label_id)
    labeled_device_ids = {
        device.device_id for device in snapshot.devices.values() if label_id in device.labels
    }
    return frozenset(
        entity.entity_id
        for entity in snapshot.entities
        if label_id in entity.labels or entity.device_id in labeled_device_ids
    )


_EXPANDERS: Mapping[str, Callable[[RegistrySnapshot, str], frozenset[str]]] = {
    "area": expand_area,
    "device": expand_device,
    "label": expand_label,
}


def expand_target(snapshot: RegistrySnapshot, kind: str, target_id: str) -> frozenset[str]:
    """Dispatch to `expand_area`/`expand_device`/`expand_label` by `kind`
    (`"area"`, `"device"`, or `"label"`) -- the one entry point `ha.py`
    calls, so it never has to know which specific function a target kind
    maps to."""
    try:
        expander = _EXPANDERS[kind]
    except KeyError:
        raise ValueError(f"unknown target kind: {kind!r}") from None
    return expander(snapshot, target_id)


def ws_url_from_http(base_url: str) -> str:
    """Home Assistant's WebSocket API lives at `/api/websocket` on the
    same host `HA_URL` already names, with `http`/`https` swapped for
    `ws`/`wss` -- the standard scheme mapping for a WebSocket upgrade over
    the same connection an HTTP request would use."""
    if base_url.startswith("https://"):
        ws_base = "wss://" + base_url[len("https://") :]
    elif base_url.startswith("http://"):
        ws_base = "ws://" + base_url[len("http://") :]
    else:
        raise ValueError(f"HA_URL must start with http:// or https://, got {base_url!r}")
    return ws_base.rstrip("/") + "/api/websocket"


class HaRegistryClient:
    """Fetches and caches Home Assistant's area/device/label/entity
    registries over the WebSocket API, authenticating with the same
    `HA_TOKEN` `ha.py` already holds.

    `connect` is injectable, following the dependency-injection-over-
    subclassing convention `CameraAudioSource(open_container=...)` already
    set in this codebase: a test drives the real handshake and the real
    out-of-order reply matching against a fake connection object exposing
    `send`/`recv`/`__aenter__`/`__aexit__`, never a real socket and never a
    subclass override.
    """

    def __init__(
        self,
        ws_url: str,
        token: str,
        *,
        connect: Callable[[str], Any] = websockets.connect,
        clock: Callable[[], float] = time.monotonic,
        refresh_interval_s: float = REGISTRY_REFRESH_INTERVAL_S,
    ) -> None:
        self._ws_url = ws_url
        self._token = token
        self._connect = connect
        self._clock = clock
        self._refresh_interval_s = refresh_interval_s
        self._cached: RegistrySnapshot | None = None

    async def get_snapshot(self) -> RegistrySnapshot:
        """The cached snapshot, refreshed first if it is missing or older
        than `refresh_interval_s`. A refresh failure propagates -- it
        never falls back to the stale value underneath it."""
        now = self._clock()
        if self._cached is None or (now - self._cached.fetched_at) > self._refresh_interval_s:
            self._cached = await self._fetch_snapshot()
        return self._cached

    async def _fetch_snapshot(self) -> RegistrySnapshot:
        try:
            async with self._connect(self._ws_url) as ws:
                await self._authenticate(ws)
                raw = await self._fetch_registry_lists(ws)
        except RegistryError:
            raise
        except Exception as exc:  # connection refused, DNS failure, closed socket, timeout, ...
            raise RegistryUnavailableError(
                f"could not reach home assistant's registry: {exc}"
            ) from exc
        return RegistrySnapshot.from_raw(raw, fetched_at=self._clock())

    async def _authenticate(self, ws: Any) -> None:
        """Home Assistant's documented handshake: it speaks first with
        `auth_required`, the client replies with the long-lived token it
        already has, and the server answers `auth_ok` or `auth_invalid`
        (developers.home-assistant.io/docs/api/websocket/)."""
        raw = await ws.recv()
        message = json.loads(raw)
        if message.get("type") != "auth_required":
            raise RegistryUnavailableError(
                f"unexpected first message from home assistant: {message!r}"
            )
        await ws.send(json.dumps({"type": "auth", "access_token": self._token}))
        raw = await ws.recv()
        message = json.loads(raw)
        if message.get("type") != "auth_ok":
            raise RegistryAuthError(
                message.get("message", "home assistant rejected the registry token")
            )

    async def _fetch_registry_lists(self, ws: Any) -> dict[str, list[dict[str, Any]]]:
        """Issue the four `config/*_registry/list` commands with
        incrementing message ids, then read replies until all four have
        answered -- matched to their request by id, not by arrival order,
        because nothing about Home Assistant's WebSocket protocol
        promises replies arrive in the order they were sent."""
        id_to_key: dict[int, str] = {}
        for message_id, (key, command_type) in enumerate(_REGISTRY_COMMANDS.items(), start=1):
            id_to_key[message_id] = key
            await ws.send(json.dumps({"id": message_id, "type": command_type}))

        results: dict[str, list[dict[str, Any]]] = {}
        while len(results) < len(_REGISTRY_COMMANDS):
            raw = await ws.recv()
            message = json.loads(raw)
            message_id = message.get("id")
            key = id_to_key.get(message_id)
            if key is None:
                # Not one of the four registry replies this fetch is
                # waiting on -- an event, or a reply to a command this
                # client never sent. Matching is by id alone, so this is
                # ignored rather than treated as an error.
                continue
            if not message.get("success", False):
                error = message.get("error", {}) or {}
                raise RegistryUnavailableError(
                    f"home assistant rejected {_REGISTRY_COMMANDS[key]}: "
                    f"{error.get('message', message)}"
                )
            results[key] = message.get("result", []) or []
        return results
