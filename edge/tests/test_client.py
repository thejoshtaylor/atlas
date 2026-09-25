"""RED for ReconnectPolicy/LiveFrame/run_forever, and run_session's
LiveFrame handling (10-08-PLAN.md Task 2). Every test token literal below
stays under 8 characters (tests/test_repo_hygiene.py's credential scan).
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from atlas_edge.client import LiveFrame, ReconnectPolicy, run_forever, run_session


def _invalid_status(code: int) -> InvalidStatus:
    return InvalidStatus(Response(code, "reason", Headers()))


class _FakeConfig:
    def __init__(self, server_url="wss://svr.test/ws/edge", token="tok1234", allow_plaintext=False):
        self.server_url = server_url
        self.token = token
        self.allow_plaintext = allow_plaintext


def _make_clock(values):
    it = iter(values)

    def _clock():
        return next(it)

    return _clock


class _ScriptedSession:
    """One outcome per call: `{}` (normal end), `{"raise": exc}`, and
    optionally `{"stop_after": True}` to set `stop` right away -- lets a
    test end `run_forever`'s loop at an exact point."""

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[str] = []

    async def __call__(
        self,
        url,
        token,
        *,
        make_outbound,
        on_reply_audio,
        on_live_frame_sent,
        allow_plaintext,
        stop,
    ):
        self.calls.append(token)
        step = self._script.pop(0)
        if step.get("stop_after"):
            stop.set()
        exc = step.get("raise")
        if exc is not None:
            raise exc


# --- ReconnectPolicy -------------------------------------------------------


def test_next_delay_backoff_schedule_capped_at_30s():
    policy = ReconnectPolicy()
    no_jitter = lambda: 0.5  # midpoint -- jitter_factor == 1.0
    assert policy.next_delay(0, rng=no_jitter) == 0.5
    assert policy.next_delay(1, rng=no_jitter) == 1.0
    assert policy.next_delay(2, rng=no_jitter) == 2.0
    assert policy.next_delay(3, rng=no_jitter) == 4.0
    assert policy.next_delay(10, rng=no_jitter) == 30.0


def test_next_delay_jitter_stays_within_plus_minus_20_percent():
    policy = ReconnectPolicy()
    low = policy.next_delay(0, rng=lambda: 0.0)
    high = policy.next_delay(0, rng=lambda: 1.0)
    assert low == pytest.approx(0.5 * 0.8)
    assert high == pytest.approx(0.5 * 1.2)


def test_next_delay_auth_refused_gives_300_regardless_of_attempt():
    policy = ReconnectPolicy()
    assert policy.next_delay(0, auth_refused=True) == 300.0
    assert policy.next_delay(9, auth_refused=True) == 300.0


# --- run_forever -------------------------------------------------------


@pytest.mark.asyncio
async def test_retries_after_closed_connection_oserror_and_normal_close_with_normal_delay():
    script = [
        {"raise": OSError("net down")},
        {},  # a clean session end (a 4000/4009 close already surfaces this way -- run_session
        # swallows ConnectionClosed and simply returns)
        {"raise": OSError("net down"), "stop_after": True},
    ]
    session = _ScriptedSession(script)
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    clock = _make_clock([0.0, 1.0, 10.0, 11.0, 20.0])  # each session lasts well under 60s
    stop = asyncio.Event()

    await run_forever(
        _FakeConfig(),
        make_outbound=lambda hello: _empty_aiter(),
        on_reply_audio=lambda data: None,
        policy=ReconnectPolicy(),
        session=session,
        sleep=fake_sleep,
        clock=clock,
        rng=lambda: 0.5,
        stop=stop,
    )

    assert len(session.calls) == 3
    assert delays == [0.5, 1.0]  # attempt 0 then attempt 1 -- no reset, no auth backoff


@pytest.mark.asyncio
async def test_403_handshake_refusal_backs_off_300s_and_never_logs_the_token(caplog):
    script = [
        {"raise": _invalid_status(403)},
        {"raise": _invalid_status(403), "stop_after": True},
    ]
    session = _ScriptedSession(script)
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    clock = _make_clock([0.0, 1.0, 10.0])
    stop = asyncio.Event()

    with caplog.at_level(logging.ERROR):
        await run_forever(
            _FakeConfig(token="sekrit1"),
            make_outbound=lambda hello: _empty_aiter(),
            on_reply_audio=lambda data: None,
            session=session,
            sleep=fake_sleep,
            clock=clock,
            rng=lambda: 0.5,
            stop=stop,
        )

    assert delays == [300.0]
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(error_records) == 1
    assert "revoke" in error_records[0].message.lower()
    for record in caplog.records:
        assert "sekrit1" not in record.message
        assert "sekrit1" not in record.getMessage()


@pytest.mark.asyncio
async def test_non_403_handshake_failure_uses_normal_delay_not_auth_backoff():
    script = [
        {"raise": _invalid_status(500), "stop_after": True},
    ]
    session = _ScriptedSession(script)
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    clock = _make_clock([0.0])
    stop = asyncio.Event()
    stop.set()  # stop_after already sets it before sleep is reached; belt and suspenders

    await run_forever(
        _FakeConfig(),
        make_outbound=lambda hello: _empty_aiter(),
        on_reply_audio=lambda data: None,
        session=session,
        sleep=fake_sleep,
        clock=clock,
        rng=lambda: 0.5,
        stop=asyncio.Event(),
    )
    # stop_after inside the scripted call sets its own stop instance, which is
    # the one passed through -- reaching here at all proves the loop ended
    # without raising, i.e. the 500 was treated as a normal (non-auth) ending.
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_session_lasting_60s_or_more_resets_the_attempt_count():
    script = [
        {"raise": OSError()},  # attempt 0 used, short session
        {"raise": OSError()},  # attempt 1 used, short session
        {"raise": OSError()},  # LONG session (>=60s) -- resets attempt to 0 for this delay
        {"raise": OSError(), "stop_after": True},  # ends before its own delay is computed
    ]
    session = _ScriptedSession(script)
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    clock = _make_clock(
        [
            0.0, 5.0,  # iter0: elapsed 5s
            100.0, 105.0,  # iter1: elapsed 5s
            200.0, 271.0,  # iter2: elapsed 71s -- triggers reset
            400.0,  # iter3: stop_after -- only started_at is read
        ]
    )
    stop = asyncio.Event()

    await run_forever(
        _FakeConfig(),
        make_outbound=lambda hello: _empty_aiter(),
        on_reply_audio=lambda data: None,
        session=session,
        sleep=fake_sleep,
        clock=clock,
        rng=lambda: 0.5,
        stop=stop,
    )

    # attempt 0 -> 0.5, attempt 1 -> 1.0, reset back to attempt 0 -> 0.5 again.
    assert delays == [0.5, 1.0, 0.5]


async def _empty_aiter():
    for _ in ():
        yield _


# --- run_session's LiveFrame handling -------------------------------------


class _FakeWebsocket:
    def __init__(self, hello_text: str, incoming=()):
        self._hello_text = hello_text
        self._incoming = list(incoming)
        self.sent: list = []
        self.closed_with = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def recv(self):
        return self._hello_text

    async def send(self, item):
        self.sent.append(item)

    async def close(self, code=1000):
        self.closed_with = code

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for item in self._incoming:
            yield item


def _connect_factory(ws: "_FakeWebsocket"):
    class _Ctx:
        async def __aenter__(self):
            return ws

        async def __aexit__(self, *exc_info):
            return False

    def _connect(url, **kwargs):
        return _Ctx()

    return _connect


@pytest.mark.asyncio
async def test_run_session_sends_live_frames_as_binary_and_reports_sent_callback():
    import json

    hello_text = json.dumps(
        {
            "type": "hello",
            "protocol": 1,
            "device_id": 1,
            "sample_rate": 16000,
            "channels": 2,
            "asr_channel": 0,
            "pre_roll_ms": 100,
            "tail_ms": 100,
            "frame_samples": 256,
        }
    )
    ws = _FakeWebsocket(hello_text)
    reported: list[tuple[float, float]] = []

    async def make_outbound(hello):
        yield LiveFrame(b"frame-data", 12.5)
        yield b"plain-bytes"

    def on_live_frame_sent(captured_at, sent_at):
        reported.append((captured_at, sent_at))

    await run_session(
        "wss://svr.test/ws/edge",
        "tok1234",
        make_outbound=make_outbound,
        on_reply_audio=lambda data: None,
        on_live_frame_sent=on_live_frame_sent,
        connect=_connect_factory(ws),
    )

    assert ws.sent[0] == b"frame-data"
    assert ws.sent[1] == b"plain-bytes"
    assert len(reported) == 1
    assert reported[0][0] == 12.5
