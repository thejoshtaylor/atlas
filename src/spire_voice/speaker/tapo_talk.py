"""`TapoTalkSupervisor`: drive the Tapo camera speaker directly over
pytapo's authenticated 8800 media session, in place of go2rtc's `tapo://`
backchannel.

The C121's current firmware demands a SHA256 / `encrypt_type=3` digest on
its 8800 "talk" endpoint that go2rtc's built-in `tapo://` client cannot
satisfy -- every backchannel push through it returns 401 / "can't find
consumer". pytapo (the client Home Assistant's own Tapo integration uses)
implements exactly that auth. This module owns the read side of
`speaker.fifo_path` -- the role `FfmpegSupervisor` plays for the `go2rtc`
backend -- and streams the A-law bytes the existing TTS/turn path already
writes there into the camera's talk session, framed as MPEG-TS
(`speaker/mpegts.py`), byte-for-byte per the proven prototype
(`reference/tapo_talk.py`, run live against the real camera this session).

Public shape mirrors `FfmpegSupervisor`: `start()` (fire-and-forget),
async `stop()`, async `handle_reconnect()` -- so `app.py`'s lifecycle
wiring and the `CameraAudioSource(on_reconnect=...)` callback need no
branch on which backend is running.

`Tapo(...)` construction and `getMediaSession()` are synchronous and run
pytapo's own internal event loop -- calling either inline on this process's
running loop is the primary failure mode this module avoids; both happen
in an executor thread (`_default_build_session`).

The Tapo cloud-account password crosses one trust boundary here: read from
`TAPO_CLOUD_PASSWORD` in the process environment at the point of use,
mirroring `auth/tokens.py`'s `read_secret_key` posture for
`SPIRE_SECRET_KEY`. It is never stored on `SpeakerConfig` (which is
`repr`'d/logged elsewhere in this codebase), never logged by value, and
never placed on a command line -- pytapo is an in-process library call,
unlike the `ffmpeg` subprocess path, so there is no argv surface for it to
leak through at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from spire_voice.config import SpeakerConfig
from spire_voice.speaker import mpegts

logger = logging.getLogger("spire_voice.speaker.tapo_talk")

TAPO_CLOUD_PASSWORD_ENV = "TAPO_CLOUD_PASSWORD"
_TAPO_PORT = 8800
_TAPO_USER = "admin"
_TALK_START_REQUEST = '{"params":{"talk":{"mode":"aec"},"method":"get"},"type":"request"}'
# Bounds each handshake step (client build, media-session connect, talk
# start). A stalled camera then fails into the supervisor's normal backoff
# and does not hang the loop with no end. The executor thread behind a
# timed-out build keeps running until pytapo returns; the loop does not
# wait for it.
_CONNECT_TIMEOUT_S = 15.0
# Ceiling for the exponential backoff in `_supervise`.
_MAX_BACKOFF_S = 600.0

# 20ms @ 8kHz mono A-law -- the frame size the camera's talk endpoint
# expects, matching the proven prototype's own pacing.
_FRAME_BYTES = 160
_FRAME_S = 0.02
_PTS_STEP = 90000 * _FRAME_BYTES // 8000  # the PES clock runs at 90kHz

SessionBuilderFn = Callable[[str, str], Awaitable[Any]]
FifoReaderOpenFn = Callable[[str], Any]


def camera_host_from_rtsp_url(rtsp_url: str) -> str:
    """The one place the camera's host is derived from
    `config.camera.rtsp_url` -- never a second, independently-configured
    host field, and never a hardcoded literal (this repository is public;
    T-vqa-02)."""
    return urlparse(rtsp_url).hostname or ""


async def _default_build_session(host: str, cloud_password: str) -> Any:
    """Build the `Tapo` client and its media session off the event loop
    (module docstring: both calls are synchronous and run pytapo's own
    internal event loop)."""
    from pytapo import Tapo

    def _build() -> Any:
        client = Tapo(host, user=_TAPO_USER, password=cloud_password, cloudPassword=cloud_password)
        return client.getMediaSession()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _build)


def _default_open_fifo_reader(path: str) -> Any:
    """Open `path` for reading, creating the FIFO node first if only its
    directory exists yet -- the same defensive create `FifoWriter`'s own
    `_blocking_open` performs on the write side, since either side may be
    the first to run. Runs in an executor thread; never called directly on
    the event loop (opening a FIFO blocks at the OS level until the other
    side attaches)."""
    if not os.path.exists(path):
        with contextlib.suppress(FileExistsError):
            os.mkfifo(path)
    return open(path, "rb", buffering=0)


async def _drain(writer: Any) -> None:
    drain = getattr(writer, "drain", None)
    if drain is None:
        return
    result = drain()
    if asyncio.iscoroutine(result):
        await result


class TapoTalkSupervisor:
    """Owns the read side of `speaker.fifo_path` for the `tapo_talk`
    backend, streaming A-law frames into the camera's 8800 talk session.

    Supervised like `FfmpegSupervisor`: a loop that restarts the whole
    session (fresh `Tapo`/media-session construction, fresh FIFO open) on
    any failure or clean EOF, backed off by `respawn_backoff_s`, never a
    busy loop against an unreachable camera (T-vqa-03).
    """

    def __init__(
        self,
        config: SpeakerConfig,
        camera_host: str,
        *,
        build_session: SessionBuilderFn = _default_build_session,
        open_fifo_reader: FifoReaderOpenFn = _default_open_fifo_reader,
        connect_timeout_s: float = _CONNECT_TIMEOUT_S,
    ) -> None:
        self._config = config
        self._camera_host = camera_host
        self._build_session = build_session
        self._open_fifo_reader = open_fifo_reader
        self._connect_timeout_s = connect_timeout_s
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    def start(self) -> None:
        """Start the supervisor loop. Never awaited by the caller -- the
        same fire-and-forget shape `FfmpegSupervisor.start()` uses."""
        self._task = asyncio.create_task(self._supervise())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def handle_reconnect(self) -> None:
        """No-op: `_supervise`'s own loop already restarts the talk session
        from scratch on any failure, and a camera reconnect looks like
        exactly that from this side. Kept only so `app.py`'s
        `CameraAudioSource(on_reconnect=...)` wiring stays uniform across
        both backends (module docstring)."""
        return

    async def _supervise(self) -> None:
        # Consecutive failures double the wait, up to _MAX_BACKOFF_S. The
        # camera's talk port locks out every client after repeated failed
        # auths, and a fixed short retry can keep that lockout armed with
        # no end (seen live: 20+ minutes of 401s at a 30 s retry). A session
        # that ends cleanly resets the wait to `respawn_backoff_s`.
        failures = 0
        while not self._stopping:
            try:
                await self._run_once()
                failures = 0
                delay = self._config.respawn_backoff_s
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- logged and retried, never crashes the app
                failures += 1
                delay = min(self._config.respawn_backoff_s * 2 ** (failures - 1), _MAX_BACKOFF_S)
                logger.warning(
                    "tapo_talk session ended (%s: %s); restarting in %.1fs",
                    type(exc).__name__,
                    exc,
                    delay,
                )
            if self._stopping:
                return
            await asyncio.sleep(delay)

    async def _run_once(self) -> None:
        cloud_password = os.environ.get(TAPO_CLOUD_PASSWORD_ENV)
        if not cloud_password:
            logger.warning("%s not set; tapo_talk speaker staying in backoff", TAPO_CLOUD_PASSWORD_ENV)
            return

        loop = asyncio.get_running_loop()
        async with asyncio.timeout(self._connect_timeout_s):
            session = await self._build_session(self._camera_host, cloud_password)
        # No timeout on the FIFO open: it blocks until a writer attaches,
        # which is normal idle time, not a stalled handshake.
        fifo_fh = await loop.run_in_executor(None, self._open_fifo_reader, self._config.fifo_path)
        try:
            async with contextlib.AsyncExitStack() as stack:
                async with asyncio.timeout(self._connect_timeout_s):
                    await stack.enter_async_context(session)
                    sid = await self._start_talk(session)
                if sid is None:
                    logger.warning("tapo_talk: camera returned no talk session id")
                    return
                await self._stream(session, sid, fifo_fh)
        finally:
            await loop.run_in_executor(None, fifo_fh.close)

    @staticmethod
    async def _start_talk(session: Any) -> Any:
        started = None
        async for resp in session.transceive(_TALK_START_REQUEST):
            started = resp
            break
        return getattr(started, "session", None) if started is not None else None

    async def _stream(self, session: Any, sid: Any, fifo_fh: Any) -> None:
        loop = asyncio.get_running_loop()
        writer = session._writer
        boundary = b"--" + session.client_boundary

        def send_part(body: bytes) -> None:
            header = (
                boundary + b"\r\n"
                b"Content-Type: audio/mp2t\r\n"
                b"X-If-Encrypt: 0\r\n"
                b"X-Session-Id: " + str(sid).encode() + b"\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
            )
            writer.write(header + body)

        send_part(mpegts.build_header())
        await _drain(writer)

        pes = mpegts.PesState()
        ts = 0
        buffer = bytearray()
        while not self._stopping:
            chunk = await loop.run_in_executor(None, fifo_fh.read, _FRAME_BYTES)
            if not chunk:
                break  # every writer has closed -- a clean EOF, not an error
            buffer.extend(chunk)
            while len(buffer) >= _FRAME_BYTES:
                frame = bytes(buffer[:_FRAME_BYTES])
                del buffer[:_FRAME_BYTES]
                send_part(mpegts.get_payload(pes, ts, frame))
                ts += _PTS_STEP
                await _drain(writer)
                await asyncio.sleep(_FRAME_S)
        if buffer and not self._stopping:
            send_part(mpegts.get_payload(pes, ts, bytes(buffer)))
            await _drain(writer)
