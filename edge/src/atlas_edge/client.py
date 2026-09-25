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
import logging
import random
import time
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import urlsplit

import websockets
from websockets.exceptions import ConnectionClosed

from atlas_edge.protocol import Hello, Ping, ProtocolError, parse_server_message, pong

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class LiveFrame:
    """One live (not pre-roll) audio frame, carrying the monotonic time it
    was captured at so `on_live_frame_sent` can report the true
    capture-to-send delay -- pre-roll audio (plain `bytes`) never counts
    toward that figure, since it is history, not something that just
    happened."""

    data: bytes
    captured_at: float


async def run_session(
    url: str,
    token: str,
    *,
    make_outbound: Callable[[Hello], AsyncIterator["bytes | str | LiveFrame"]],
    on_reply_audio: Callable[[bytes], "Awaitable[None] | None"],
    on_live_frame_sent: "Callable[[float, float], Awaitable[None] | None] | None" = None,
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

    `make_outbound` may yield `LiveFrame` items (sent as binary, then
    reported once each via `on_live_frame_sent(captured_at, sent_at)`) in
    addition to plain `bytes` and `str`, which are still sent as-is.
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
                if isinstance(item, LiveFrame):
                    await ws.send(item.data)
                    if on_live_frame_sent is not None:
                        outcome = on_live_frame_sent(item.captured_at, time.monotonic())
                        if inspect.isawaitable(outcome):
                            await outcome
                else:
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


@dataclass(frozen=True)
class ReconnectPolicy:
    """Backoff constants (Claude's discretion per 10-08-PLAN.md
    `<interfaces>`): initial 0.5s, factor 2, capped at 30s, jitter
    +/-20%, an auth-refused connection backs off to 300s, and a session
    that stayed connected at least `reset_after_s` resets the attempt
    count back to 0."""

    initial_s: float = 0.5
    factor: float = 2.0
    cap_s: float = 30.0
    jitter: float = 0.20
    auth_refused_s: float = 300.0
    reset_after_s: float = 60.0

    def next_delay(
        self,
        attempt: int,
        auth_refused: bool = False,
        *,
        rng: Callable[[], float] = random.random,
    ) -> float:
        """`attempt` 0, 1, 2, 3 ... gives 0.5, 1, 2, 4 ... seconds,
        capped at `cap_s`, jittered by +/- `jitter` under the injected
        `rng` (each call to `rng()` must return a value in [0, 1)).
        `auth_refused=True` ignores `attempt` and returns `auth_refused_s`
        flat -- a refused token backs off the same amount every time,
        never escalating and never resetting on its own (T-10-27)."""
        if auth_refused:
            return self.auth_refused_s
        base = min(self.initial_s * (self.factor**attempt), self.cap_s)
        jitter_factor = 1.0 + (rng() * 2.0 - 1.0) * self.jitter
        return base * jitter_factor


async def run_forever(
    config: Any,
    *,
    make_outbound: Callable[[Hello], AsyncIterator["bytes | str | LiveFrame"]],
    on_reply_audio: Callable[[bytes], "Awaitable[None] | None"],
    on_live_frame_sent: "Callable[[float, float], Awaitable[None] | None] | None" = None,
    policy: ReconnectPolicy = ReconnectPolicy(),
    session: Callable[..., Awaitable[None]] = run_session,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: Callable[[], float] = random.random,
    stop: "asyncio.Event | None" = None,
) -> None:
    """The Pi dials the server and owns the reconnect loop (D-02): one
    `session(...)` attempt after another, forever, until `stop` is set.

    Classifies each ending: a 403 handshake refusal (the token may be
    revoked, T-10-27) backs off `policy.auth_refused_s` and logs one error
    per refusal streak that never includes the token itself; every other
    ending (a closed connection -- including a 4000/4009 close, which
    `run_session` already treats as an ordinary end -- an `OSError`, or
    any other exception) backs off with `policy.next_delay`'s normal
    schedule. A session that lasted at least `policy.reset_after_s`
    resets the attempt count, so a Pi that reconnects cleanly every few
    minutes never climbs toward the 30s cap.
    """
    stop_event = stop if stop is not None else asyncio.Event()
    attempt = 0
    in_auth_refused_streak = False

    while not stop_event.is_set():
        started_at = clock()
        auth_refused = False
        host = urlsplit(config.server_url).hostname

        try:
            await session(
                config.server_url,
                config.token,
                make_outbound=make_outbound,
                on_reply_audio=on_reply_audio,
                on_live_frame_sent=on_live_frame_sent,
                allow_plaintext=config.allow_plaintext,
                stop=stop_event,
            )
        except websockets.exceptions.InvalidStatus as exc:
            if getattr(exc.response, "status_code", None) == 403:
                auth_refused = True
                # Once per refusal *streak*, not once per attempt -- a
                # revoked token retries for as long as this Pi runs, and
                # logging the same error every 300s forever would be
                # nothing but noise.
                if not in_auth_refused_streak:
                    logger.error(
                        "edge server at %s refused this device's token (403) -- "
                        "it may be revoked; backing off %.0fs",
                        host,
                        policy.auth_refused_s,
                    )
                in_auth_refused_streak = True
            else:
                logger.warning(
                    "edge handshake to %s failed with status %s; reconnecting",
                    host,
                    getattr(exc.response, "status_code", "?"),
                )
        except OSError as exc:
            logger.warning(
                "edge connection to %s failed (%s); reconnecting",
                host,
                type(exc).__name__,
            )
        except Exception:
            logger.exception(
                "edge session with %s ended with an unexpected error; reconnecting",
                host,
            )
        else:
            logger.warning("edge connection to %s ended; reconnecting", host)

        if not auth_refused:
            in_auth_refused_streak = False

        if stop_event.is_set():
            return

        connected_s = clock() - started_at
        if connected_s >= policy.reset_after_s:
            attempt = 0

        delay = policy.next_delay(attempt, auth_refused=auth_refused, rng=rng)
        attempt += 1
        await sleep(delay)
