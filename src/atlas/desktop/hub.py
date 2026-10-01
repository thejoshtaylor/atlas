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

from starlette.websockets import WebSocketDisconnect

from atlas.db.desktop_repository import DesktopDevice, DesktopDeviceRepository
from atlas.desktop.protocol import (
    CLOSE_GOING_AWAY,
    CLOSE_POLICY_VIOLATION,
    CLOSE_PROTOCOL_MISMATCH,
    CLOSE_REVOKED,
    CLOSE_SUPERSEDED,
    HELLO_TIMEOUT_S,
    IDLE_TIMEOUT_S,
    MAX_INVALID_MESSAGES,
    MAX_TEXT_FRAME_BYTES,
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
    raw_hello_protocol,
)

logger = logging.getLogger(__name__)

# What a send raises when the Mac is already gone (sleep, lid close, network
# drop): Starlette's disconnect, a send after close, or a broken pipe. This is
# a routine event, not a fault.
_SEND_FAILED = (WebSocketDisconnect, RuntimeError, OSError)


class DesktopNotConnected(Exception):
    """No live socket for this Mac when a ping was asked for."""


class DesktopConnection:
    """One live Mac socket and the pings waiting on it."""

    def __init__(
        self, websocket: Any, device_id: int, hello: DesktopHello, connected_at: datetime
    ) -> None:
        self.websocket = websocket
        self.device_id = device_id
        self.hello = hello
        self.connected_at = connected_at
        self._pending_pings: dict[int, asyncio.Future[bool]] = {}
        self._next_ping_id = 1
        # Set by the hub just before it cancels this connection's task
        # (supersede or revoke), so `serve` can end quietly for that cause
        # alone and still let any other cancellation, such as a server
        # shutdown, propagate.
        self.ended_by_hub = False

    async def ping(self, timeout_s: float, clock: Callable[[], float]) -> float | None:
        """Send a ping and wait for the Mac's pong with the same id. Returns
        the round trip in milliseconds, or `None` on timeout or disconnect."""
        ping_id = self._next_ping_id
        self._next_ping_id += 1
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_pings[ping_id] = future
        try:
            started = clock()
            try:
                await self.websocket.send_text(build_ping(ping_id))
            except _SEND_FAILED:
                return None  # the Mac is gone, so the Test reports it as unanswered
            try:
                answered = await asyncio.wait_for(future, timeout_s)
            except asyncio.TimeoutError:
                return None
            if not answered:
                return None  # the socket ended while the ping waited
            return (clock() - started) * 1000.0
        finally:
            self._pending_pings.pop(ping_id, None)

    def resolve_pong(self, ping_id: int) -> bool:
        future = self._pending_pings.get(ping_id)
        if future is None or future.done():
            return False
        future.set_result(True)
        return True

    def fail_pending(self) -> None:
        """The socket ended: every waiting ping returns None at once, not
        after its full timeout."""
        for future in list(self._pending_pings.values()):
            if not future.done():
                future.set_result(False)


class DesktopHub:
    def __init__(
        self,
        *,
        device_repo: DesktopDeviceRepository | None = None,
        clock: Callable[[], float] = time.monotonic,
        hello_timeout_s: float = HELLO_TIMEOUT_S,
        ping_timeout_s: float = TEST_TIMEOUT_S,
        idle_timeout_s: float = IDLE_TIMEOUT_S,
    ) -> None:
        self._device_repo = device_repo
        self._clock = clock
        self.hello_timeout_s = hello_timeout_s
        self.ping_timeout_s = ping_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self._connections: dict[int, DesktopConnection] = {}
        self._tasks: dict[int, asyncio.Task[Any]] = {}

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
        # The first frame gets the same size cap as every later frame.
        if text is not None and len(text.encode("utf-8")) > MAX_TEXT_FRAME_BYTES:
            text = None
        # Name a protocol mismatch before the hello is validated: a newer
        # protocol may change the hello shape, and that hello would otherwise
        # fail validation here and read as "bad_hello" (D-09).
        claimed = raw_hello_protocol(text) if text is not None else None
        if claimed is not None and claimed != PROTOCOL_VERSION:
            await self._refuse(
                websocket, "protocol_mismatch", f"server speaks protocol {PROTOCOL_VERSION}"
            )
            await self._close(websocket, CLOSE_PROTOCOL_MISMATCH, "protocol_mismatch")
            return

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

        # A revoke can land between the handshake check and this hello. Look
        # the row up again so a revoked Mac is never registered (PAIR-03).
        if self._device_repo is not None:
            try:
                current = await self._device_repo.get_device(device.id)
            except Exception:
                logger.exception("desktop %s: revoke re-check failed", device.id)
                current = None
            if current is not None and current.revoked_at is not None:
                await self._close(websocket, CLOSE_REVOKED, "revoked")
                return

        await self._supersede(device.id)
        connection = DesktopConnection(websocket, device.id, hello, datetime.now(timezone.utc))
        self._connections[device.id] = connection
        task = asyncio.current_task()
        if task is not None:
            self._tasks[device.id] = task
        try:
            try:
                await websocket.send_text(build_hello_ack(device.id))
            except _SEND_FAILED:
                logger.info("desktop %s: gone before the hello.ack", device.id)
                return
            await self._mark_seen(device.id)
            logger.info("desktop %s connected, app %s", device.id, hello.app_version)
            await self._receive_loop(websocket, connection)
        except asyncio.CancelledError:
            if not connection.ended_by_hub:
                raise
            # The hub cancelled this task on purpose, after it closed the
            # socket. End the call normally: a cancellation that escaped
            # into the ASGI app would be logged as an application error.
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
        finally:
            # A superseded call's cleanup must never clear the connection
            # that replaced it, so only the registered owner removes entries.
            if self._connections.get(device.id) is connection:
                del self._connections[device.id]
                self._tasks.pop(device.id, None)
            connection.fail_pending()
            await self._mark_seen(device.id)
            logger.info("desktop %s disconnected", device.id)

    async def _supersede(self, device_id: int) -> None:
        """Close the older socket for this Mac, if any, and cancel its task.
        A dead old socket must never block the new connection (CR-02)."""
        previous = self._connections.get(device_id)
        if previous is None:
            return
        try:
            await previous.websocket.close(code=CLOSE_SUPERSEDED, reason="superseded")
        except Exception:
            logger.info("desktop %s: previous connection already gone", device_id)
        previous_task = self._tasks.get(device_id)
        previous.ended_by_hub = True
        if previous_task is not None:
            previous_task.cancel()

    async def disconnect_device(self, device_id: int, *, code: int, reason: str) -> None:
        """Close the live socket for `device_id` at once (revoke, D-01). A
        no-op when the Mac is not connected. The task is cancelled even when
        the close fails, because a dead socket must not keep a revoked Mac
        registered (CR-01)."""
        connection = self._connections.get(device_id)
        if connection is None:
            return
        task = self._tasks.get(device_id)
        try:
            await connection.websocket.close(code=code, reason=reason)
        except Exception:
            logger.warning("desktop %s: close on disconnect failed (already gone)", device_id)
        connection.ended_by_hub = True
        if task is not None:
            task.cancel()

    async def _receive_loop(self, websocket: Any, connection: DesktopConnection) -> None:
        invalid_count = 0
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), self.idle_timeout_s)
            except asyncio.TimeoutError:
                await self._close(websocket, CLOSE_POLICY_VIOLATION, "idle")
                return
            if message.get("type") == "websocket.disconnect":
                return
            invalid = False
            text = message.get("text")
            if text is None or len(text.encode("utf-8")) > MAX_TEXT_FRAME_BYTES:
                invalid = True  # a binary frame, or a text frame over the cap
            else:
                try:
                    parsed = parse_client_message(text)
                except DesktopProtocolError:
                    invalid = True
                else:
                    if isinstance(parsed, DesktopPing):
                        try:
                            await websocket.send_text(build_pong(parsed.id))
                        except _SEND_FAILED:
                            return  # the Mac is gone; `serve` cleans up
                    elif isinstance(parsed, DesktopPong):
                        # The edge T-10-25 rule: a pong nobody waits for is
                        # counted like any other bad frame.
                        invalid = not connection.resolve_pong(parsed.id)
                    # Any other type, including one this server has never
                    # heard of, is ignored so an older server keeps working
                    # with a newer app.
            if invalid:
                invalid_count += 1
                if invalid_count >= MAX_INVALID_MESSAGES:
                    await self._close(websocket, CLOSE_POLICY_VIOLATION, "too_many_invalid")
                    return

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
