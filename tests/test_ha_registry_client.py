"""Tests for `atlas_mcp.registry.HaRegistryClient`: the WebSocket handshake,
out-of-order reply matching, and the failure-closed behaviors the target
expansion in `mcp/atlas_mcp/ha.py` depends on (SAFE-03, SAFE-09).

Every id and every entity id below is invented, following the rule
`safety.py`'s own self-check already states.

None of these tests touch a real socket. `HaRegistryClient(connect=...)`
takes the connection factory as an injected callable -- the dependency-
injection-over-subclassing convention `CameraAudioSource(open_container=...)`
already set in this codebase -- so `_FakeWs` below drives the real
handshake and matching logic directly.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from atlas_mcp.registry import (
    RegistryAuthError,
    RegistryUnavailableError,
    REGISTRY_REFRESH_INTERVAL_S,
    HaRegistryClient,
)


class _FakeWs:
    """A scripted WebSocket connection: `recv()` yields the next message
    in `script`, in order; `send()` just records what was sent. Used as
    both the connect-time object and the async context manager, matching
    how `websockets.connect(...)` itself is used (`async with connect(url)
    as ws:`)."""

    def __init__(self, script: list[dict]) -> None:
        self._script = [json.dumps(m) for m in script]
        self.sent: list[dict] = []

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return self._script.pop(0)

    async def __aenter__(self) -> "_FakeWs":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def _connect_returning(ws: _FakeWs):
    def connect(url: str) -> _FakeWs:
        return ws

    return connect


_AUTH_REQUIRED = {"type": "auth_required", "ha_version": "2026.9.1"}
_AUTH_OK = {"type": "auth_ok", "ha_version": "2026.9.1"}
_AUTH_INVALID = {"type": "auth_invalid", "message": "invalid access token"}


def _result(message_id: int, result: list[dict]) -> dict:
    return {"id": message_id, "type": "result", "success": True, "result": result}


async def test_authentication_failure_raises_a_named_error_not_an_empty_snapshot():
    """`auth_invalid` must raise `RegistryAuthError`, never return a
    snapshot -- an empty snapshot expands every area to nothing, which is
    indistinguishable from a house with nothing in it."""
    ws = _FakeWs([_AUTH_REQUIRED, _AUTH_INVALID])
    client = HaRegistryClient("ws://ha.invalid/api/websocket", "bad-token", connect=_connect_returning(ws))

    with pytest.raises(RegistryAuthError):
        await client.get_snapshot()

    assert ws.sent == [{"type": "auth", "access_token": "bad-token"}]


async def test_registry_replies_are_matched_by_id_not_by_arrival_order():
    """The four `config/*_registry/list` replies arrive scrambled, in the
    reverse order the commands were sent -- the client must still place
    each result under the right key, because nothing about Home
    Assistant's WebSocket protocol promises in-order replies."""
    script = [
        _AUTH_REQUIRED,
        _AUTH_OK,
        # Sent order is areas(1), devices(2), labels(3), entities(4).
        # Replies arrive entities, labels, devices, areas -- fully reversed.
        _result(4, [{"entity_id": "light.example_lamp", "area_id": "area_example_office", "device_id": None}]),
        _result(3, [{"label_id": "label_example_important"}]),
        _result(2, [{"id": "device_example_a", "area_id": "area_example_office"}]),
        _result(1, [{"area_id": "area_example_office"}]),
    ]
    ws = _FakeWs(script)
    client = HaRegistryClient("ws://ha.invalid/api/websocket", "good-token", connect=_connect_returning(ws))

    snapshot = await client.get_snapshot()

    assert snapshot.areas == frozenset({"area_example_office"})
    assert snapshot.labels == frozenset({"label_example_important"})
    assert "device_example_a" in snapshot.devices
    assert snapshot.devices["device_example_a"].area_id == "area_example_office"
    assert snapshot.entities[0].entity_id == "light.example_lamp"


async def test_a_reply_to_an_unrequested_id_is_ignored_not_an_error():
    """A message carrying an id this client never sent (a stray event, or
    a reply to a command from a different caller sharing the connection)
    is skipped, not treated as a protocol violation."""
    script = [
        _AUTH_REQUIRED,
        _AUTH_OK,
        {"id": 999, "type": "result", "success": True, "result": []},  # unrequested id
        _result(1, []),
        _result(2, []),
        _result(3, []),
        _result(4, []),
    ]
    ws = _FakeWs(script)
    client = HaRegistryClient("ws://ha.invalid/api/websocket", "good-token", connect=_connect_returning(ws))

    snapshot = await client.get_snapshot()

    assert snapshot.areas == frozenset()
    assert snapshot.entities == ()


async def test_a_rejected_registry_command_raises_unavailable():
    """`success: false` on any of the four commands must raise
    `RegistryUnavailableError` -- a partial registry is not a usable one."""
    script = [
        _AUTH_REQUIRED,
        _AUTH_OK,
        {"id": 1, "type": "result", "success": False, "error": {"message": "unknown_command"}},
    ]
    ws = _FakeWs(script)
    client = HaRegistryClient("ws://ha.invalid/api/websocket", "good-token", connect=_connect_returning(ws))

    with pytest.raises(RegistryUnavailableError):
        await client.get_snapshot()


async def test_a_connection_failure_raises_unavailable_not_an_import_level_exception():
    """Whatever `connect()` raises when it cannot reach Home Assistant at
    all (a real `OSError`/`ConnectionRefusedError` from `websockets`) is
    wrapped into the one named error `ha.py` catches, rather than leaking
    the underlying transport exception type to a caller that only knows
    about this module's own errors."""

    def _connect_that_fails(url: str):
        raise ConnectionRefusedError("no route to host")

    client = HaRegistryClient("ws://ha.invalid/api/websocket", "good-token", connect=_connect_that_fails)

    with pytest.raises(RegistryUnavailableError):
        await client.get_snapshot()


async def test_the_snapshot_is_cached_and_not_refetched_within_the_interval():
    """A second `get_snapshot()` call inside the refresh interval must not
    reopen the connection -- 'never per call', per this module's own
    docstring."""
    script = [_AUTH_REQUIRED, _AUTH_OK, _result(1, []), _result(2, []), _result(3, []), _result(4, [])]
    ws = _FakeWs(script)
    connect_calls: list[str] = []

    def connect(url: str) -> _FakeWs:
        connect_calls.append(url)
        return ws

    client = HaRegistryClient("ws://ha.invalid/api/websocket", "good-token", connect=connect)

    first = await client.get_snapshot()
    second = await client.get_snapshot()

    assert first is second
    assert len(connect_calls) == 1


async def test_a_refresh_failure_never_falls_back_to_the_stale_snapshot():
    """Once the cache is stale, a failed refetch must raise -- not quietly
    keep answering from the snapshot it already has. A clock the test
    controls fast-forwards past `refresh_interval_s` between the two
    calls."""
    good_script = [_AUTH_REQUIRED, _AUTH_OK, _result(1, []), _result(2, []), _result(3, []), _result(4, [])]
    calls = {"n": 0}

    def connect(url: str):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeWs(good_script)
        raise ConnectionRefusedError("home assistant restarted mid-refresh")

    # tick 1: first get_snapshot()'s unused "now" read (cache starts empty).
    # tick 2: fetched_at recorded for the first, successful fetch (0.0).
    # tick 3: second get_snapshot()'s "now" read -- far enough past tick 2
    # to force a refetch, which then fails.
    clock_values = iter([0.0, 0.0, REGISTRY_REFRESH_INTERVAL_S + 1.0])

    def clock() -> float:
        return next(clock_values)

    client = HaRegistryClient(
        "ws://ha.invalid/api/websocket", "good-token", connect=connect, clock=clock
    )

    first = await client.get_snapshot()
    assert first.areas == frozenset()

    with pytest.raises(RegistryUnavailableError):
        await client.get_snapshot()


class _HangingWs:
    """A connection that accepts the auth handshake and then never answers
    again -- the shape WR-01's finding names: a half-open connection
    behind a NAT, or a Home Assistant restart that drops the listener but
    leaves an established connection dangling. `recv()` after the
    handshake awaits an `Event` this test never sets, standing in for a
    socket read that would otherwise block forever."""

    def __init__(self) -> None:
        self._replies = [json.dumps(_AUTH_REQUIRED), json.dumps(_AUTH_OK)]
        self.sent: list[dict] = []

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        if self._replies:
            return self._replies.pop(0)
        await asyncio.Event().wait()  # pragma: no cover -- never returns

    async def __aenter__(self) -> "_HangingWs":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


async def test_a_hung_connection_refuses_within_the_fetch_timeout_instead_of_blocking_forever():
    """WR-01 fix (code review): neither `_authenticate` nor
    `_fetch_registry_lists` bounded `ws.recv()` before this fix -- a
    connection that answered the auth handshake and then stopped
    responding hung `get_snapshot()` (and, through `McpToolHost.call_tool`'s
    one lock, every other tool call) for the life of the process. A tiny
    `fetch_timeout_s` (not the real ten-second default -- this test must
    stay fast) proves the bound is real and wall-clock-measured, not
    merely present in the signature.
    """
    ws = _HangingWs()
    client = HaRegistryClient(
        "ws://ha.invalid/api/websocket",
        "good-token",
        connect=_connect_returning(ws),
        fetch_timeout_s=0.05,
    )

    started = time.monotonic()
    with pytest.raises(RegistryUnavailableError, match="did not answer within"):
        await client.get_snapshot()
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, (
        f"get_snapshot() took {elapsed:.2f}s against a 0.05s fetch_timeout_s -- "
        "the timeout is not actually bounding the hung recv()"
    )
