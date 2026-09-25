"""One authenticated session from the Pi to `/ws/edge`.

Stdlib and `websockets` only -- this module never imports `atlas`. The Pi
dials the server and owns the reconnect loop (D-02); `run_session` is one
connection attempt, not the retry loop around it (that lives in whatever
`edge/`'s own service entry point becomes, later in this phase).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from ipaddress import ip_address
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import urlsplit

import websockets
from websockets.exceptions import ConnectionClosed

from atlas_edge.protocol import Hello, Ping, ProtocolError, parse_server_message, pong


def _is_loopback_host(hostname: str | None) -> bool:
    """Whether `hostname` names this machine itself -- the same rule
    `src/atlas/plugins/host.py::_is_loopback_host` applies on the server
    side, copied here rather than imported (this package never imports
    `atlas`)."""
    if hostname is None:
        return False
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_server_url(url: str, allow_plaintext: bool = False) -> None:
    """Accept `wss://` always; accept `ws://` only for a loopback host or
    when `allow_plaintext` is true. Refuses everything else -- a device
    token crossing a real network unencrypted is exactly what this stops
    (T-10-04)."""
    parsed = urlsplit(url)
    if parsed.scheme == "wss":
        return
    if parsed.scheme == "ws" and (allow_plaintext or _is_loopback_host(parsed.hostname)):
        return
    raise ValueError(
        f"server url {url!r} is not allowed -- only wss:// is allowed for a non-loopback "
        "host, since a plain ws:// connection would send the device token over the "
        "network in the clear"
    )


async def run_session(
    url: str,
    token: str,
    *,
    make_outbound: Callable[[Hello], AsyncIterator["bytes | str"]],
    on_reply_audio: Callable[[bytes], "Awaitable[None] | None"],
    allow_plaintext: bool = False,
    stop: "asyncio.Event | None" = None,
    connect: Callable[..., Any] = websockets.connect,
    hello_timeout_s: float = 10.0,
) -> None:
    """One authenticated connection to `/ws/edge`: waits for the server's
    hello, then runs two concurrent loops -- sending whatever
    `make_outbound(hello)` yields, and receiving reply audio (binary) and
    answering a `ping` with a `pong` (text) -- until the server closes the
    connection, or `stop` is set. Closes with code 1000 either way.
    """
    validate_server_url(url, allow_plaintext)
    stop_event = stop if stop is not None else asyncio.Event()

    async with connect(url, additional_headers={"Authorization": f"Bearer {token}"}) as ws:
        hello_text = await asyncio.wait_for(ws.recv(), timeout=hello_timeout_s)
        hello = parse_server_message(hello_text)
        if not isinstance(hello, Hello):
            raise ProtocolError(f"expected the server's hello as the first message, got {hello!r}")

        async def _send_outbound() -> None:
            async for item in make_outbound(hello):
                await ws.send(item)

        async def _receive_inbound() -> None:
            async for message in ws:
                if isinstance(message, bytes):
                    outcome = on_reply_audio(message)
                    if inspect.isawaitable(outcome):
                        await outcome
                    continue
                parsed = parse_server_message(message)
                if isinstance(parsed, Ping):
                    await ws.send(pong(parsed.id, parsed.server_t_ms))

        send_task = asyncio.ensure_future(_send_outbound())
        receive_task = asyncio.ensure_future(_receive_inbound())
        stop_task = asyncio.ensure_future(stop_event.wait())
        try:
            # `send_task` finishing (its `make_outbound` generator ran
            # out of items) is not, on its own, a reason for this session
            # to end -- reply audio can keep arriving long after the
            # outbound stream is exhausted. Only the connection closing
            # (`receive_task` ends) or `stop` being set ends the wait.
            await asyncio.wait({receive_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (send_task, receive_task, stop_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(send_task, receive_task, stop_task, return_exceptions=True)

        # A closed connection is the ordinary "the server ended the
        # session" outcome, never a failure this function raises -- any
        # other exception from either worker task is real and propagates.
        for task in (send_task, receive_task):
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None and not isinstance(exc, ConnectionClosed):
                raise exc

        with contextlib.suppress(ConnectionClosed):
            await ws.close(code=1000)
