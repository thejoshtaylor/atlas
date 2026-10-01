"""`DesktopHub`: the live `/ws/desktop` connections, one per paired Mac
(Phase 14, D-02, D-09, D-15).

Online state is whatever this hub holds right now, never a database
column. The route has already accepted the socket and authenticated the
device token before `serve` runs. Log lines name the device id and a close
code only: never token material, and never hello text beyond `app_version`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from atlas.db.desktop_repository import DesktopDevice, DesktopDeviceRepository
from atlas.desktop.protocol import (
    CLOSE_GOING_AWAY,
    CLOSE_POLICY_VIOLATION,
    CLOSE_PROTOCOL_MISMATCH,
    HELLO_TIMEOUT_S,
    PROTOCOL_VERSION,
    TEST_TIMEOUT_S,
    DesktopHello,
    DesktopPing,
    DesktopPong,
    DesktopProtocolError,
    build_error,
    build_hello_ack,
    build_ping,
    build_pong,
    parse_client_message,
)

logger = logging.getLogger(__name__)


class DesktopNotConnected(Exception):
    """No live socket for this Mac, or it dropped while a ping was pending."""


class DesktopConnection:
    """One live Mac socket and the pings waiting on it."""

    def __init__(
        self, websocket: Any, device_id: int, hello: DesktopHello, connected_at: datetime
    ) -> None:
        self.websocket = websocket
        self.device_id = device_id
        self.hello = hello
        self.connected_at = connected_at
        self._pending_pings: dict[int, asyncio.Future[None]] = {}
        self._next_ping_id = 1

    async def ping(self, timeout_s: float, clock: Callable[[], float]) -> float | None:
        """Send a ping and wait for the Mac's pong with the same id. Returns
        the round trip in milliseconds, or `None` on timeout."""
        ping_id = self._next_ping_id
        self._next_ping_id += 1
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending_pings[ping_id] = future
        try:
            started = clock()
            await self.websocket.send_text(build_ping(ping_id))
            try:
                await asyncio.wait_for(future, timeout_s)
            except asyncio.TimeoutError:
                return None
            return (clock() - started) * 1000.0
        finally:
            self._pending_pings.pop(ping_id, None)

    def resolve_pong(self, ping_id: int) -> bool:
        future = self._pending_pings.get(ping_id)
        if future is None or future.done():
            return False
        future.set_result(None)
        return True

    def fail_pending(self) -> None:
        """The socket ended: every waiting ping learns the Mac is gone."""
        for future in list(self._pending_pings.values()):
            if not future.done():
                future.set_exception(DesktopNotConnected("the Mac disconnected"))


class DesktopHub:
    def __init__(
        self,
        *,
        device_repo: DesktopDeviceRepository | None = None,
        clock: Callable[[], float] = time.monotonic,
        hello_timeout_s: float = HELLO_TIMEOUT_S,
        ping_timeout_s: float = TEST_TIMEOUT_S,
    ) -> None:
        self._device_repo = device_repo
        self._clock = clock
        self.hello_timeout_s = hello_timeout_s
        self.ping_timeout_s = ping_timeout_s
        self._connections: dict[int, DesktopConnection] = {}

    async def serve(self, websocket: Any, device: DesktopDevice) -> None:
        """Run one Mac connection to its end. The caller has accepted."""
        try:
            first = await asyncio.wait_for(websocket.receive(), self.hello_timeout_s)
        except asyncio.TimeoutError:
            await self._close(websocket, CLOSE_POLICY_VIOLATION, "hello_timeout")
            return
        if first.get("type") == "websocket.disconnect":
            return

        text = first.get("text")
        hello = None
        if text is not None:
            try:
                hello = parse_client_message(text)
            except DesktopProtocolError:
                hello = None
        if not isinstance(hello, DesktopHello):
            await self._refuse(websocket, "bad_hello", "the first frame must be a hello")
            await self._close(websocket, CLOSE_POLICY_VIOLATION, "bad_hello")
            return

        if hello.protocol != PROTOCOL_VERSION:
            await self._refuse(
                websocket, "protocol_mismatch", f"server speaks protocol {PROTOCOL_VERSION}"
            )
            await self._close(websocket, CLOSE_PROTOCOL_MISMATCH, "protocol_mismatch")
            return

        connection = DesktopConnection(websocket, device.id, hello, datetime.now(timezone.utc))
        self._connections[device.id] = connection
        try:
            await websocket.send_text(build_hello_ack(device.id))
            await self._mark_seen(device.id)
            logger.info("desktop %s connected, app %s", device.id, hello.app_version)
            await self._receive_loop(websocket, connection)
        finally:
            if self._connections.get(device.id) is connection:
                del self._connections[device.id]
            connection.fail_pending()
            await self._mark_seen(device.id)
            logger.info("desktop %s disconnected", device.id)

    async def _receive_loop(self, websocket: Any, connection: DesktopConnection) -> None:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            text = message.get("text")
            if text is None:
                continue
            try:
                parsed = parse_client_message(text)
            except DesktopProtocolError:
                continue
            if isinstance(parsed, DesktopPing):
                await websocket.send_text(build_pong(parsed.id))
            elif isinstance(parsed, DesktopPong):
                connection.resolve_pong(parsed.id)
            # Any other type, including one this server has never heard of,
            # is ignored so an older server keeps working with a newer app.

    async def _mark_seen(self, device_id: int) -> None:
        if self._device_repo is None:
            return
        try:
            await self._device_repo.mark_seen(device_id, at=datetime.now(timezone.utc))
        except Exception:
            logger.exception("desktop %s: mark_seen failed", device_id)

    @staticmethod
    async def _refuse(websocket: Any, code: str, detail: str) -> None:
        try:
            await websocket.send_text(build_error(code, detail))
        except Exception:
            logger.debug("desktop refusal frame not delivered", exc_info=True)

    @staticmethod
    async def _close(websocket: Any, code: int, reason: str) -> None:
        try:
            await websocket.close(code=code, reason=reason)
        except Exception:
            logger.debug("desktop close %s not delivered", code, exc_info=True)

    def snapshot(self) -> dict[int, dict[str, Any]]:
        return {
            device_id: {
                "connected": True,
                "connected_at": connection.connected_at,
                "app_version": connection.hello.app_version,
            }
            for device_id, connection in self._connections.items()
        }

    def is_connected(self, device_id: int) -> bool:
        return device_id in self._connections

    async def ping(self, device_id: int, *, timeout_s: float | None = None) -> float | None:
        connection = self._connections.get(device_id)
        if connection is None:
            raise DesktopNotConnected(f"desktop {device_id} is not connected")
        return await connection.ping(
            self.ping_timeout_s if timeout_s is None else timeout_s, self._clock
        )

    async def close(self) -> None:
        """Server shutdown: close every live socket with 1001."""
        for connection in list(self._connections.values()):
            try:
                await connection.websocket.close(code=CLOSE_GOING_AWAY, reason="going_away")
            except Exception:
                logger.debug("desktop %s: close on shutdown failed", connection.device_id)
