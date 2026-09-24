"""`TapoTalkSupervisor`: the FIFO reader for the `tapo_talk` speaker
backend, streaming A-law into an injected fake media session.

No camera, no pytapo import at the network layer -- `build_session` is
replaced with a fake factory, matching `test_startup_smoke.py`'s own
monkeypatched-factory pattern for `_build_ffmpeg_supervisor`. The FIFO
itself is real (`test_speaker_fifo.py`'s own real-pipe technique): the
supervisor's blocking-open/blocking-read discipline is the operating
system's, and faking it would test the fake, not the reader.
"""

from __future__ import annotations

import asyncio
import os
import threading

import pytest

from atlas.config import SpeakerConfig
from atlas.speaker import mpegts
from atlas.speaker.tapo_talk import TAPO_CLOUD_PASSWORD_ENV, TapoTalkSupervisor, camera_host_from_rtsp_url


class _FakeWriter:
    """Captures every `write()` call as one part -- the fake `_writer`
    `HttpMediaSession` exposes for `send_part` to write multipart bodies
    into."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.parts.append(bytes(data))


class _FakeResponse:
    def __init__(self, session_id: int) -> None:
        self.session = session_id


class _FakeMediaSession:
    """Stands in for `pytapo`'s `HttpMediaSession`: an async context
    manager, `transceive()` as an async generator yielding one response
    carrying `.session`, `client_boundary`, and a `_writer` that captures
    every write."""

    def __init__(self, session_id: int = 4242) -> None:
        self.client_boundary = b"----client-stream-boundary--"
        self._writer = _FakeWriter()
        self._session_id = session_id
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "_FakeMediaSession":
        self.entered = True
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exited = True

    async def transceive(self, data: str):
        yield _FakeResponse(self._session_id)


async def _wait_for(predicate, *, attempts: int = 300, interval: float = 0.01) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"condition never became true after {attempts * interval:.2f}s")


def test_camera_host_from_rtsp_url_derives_the_host_only():
    assert camera_host_from_rtsp_url("rtsp://admin:secret@192.0.2.5:554/stream1") == "192.0.2.5"


async def test_tapo_talk_supervisor_frames_header_then_alaw_to_the_fake_session(tmp_path, monkeypatch):
    monkeypatch.setenv(TAPO_CLOUD_PASSWORD_ENV, "not-a-real-password")

    fifo_path = str(tmp_path / "speaker.fifo")
    os.mkfifo(fifo_path)

    fake_session = _FakeMediaSession()

    async def fake_build_session(host: str, cloud_password: str):
        assert host == "192.0.2.5"
        assert cloud_password == "not-a-real-password"
        return fake_session

    config = SpeakerConfig(fifo_path=fifo_path, respawn_backoff_s=0.05)
    supervisor = TapoTalkSupervisor(config, "192.0.2.5", build_session=fake_build_session)
    supervisor.start()

    # Two 20ms A-law frames, written by a real pipe writer once the
    # supervisor's own reader has attached -- os.mkfifo above only creates
    # the node; opening it for writing blocks until a reader shows up,
    # exactly like the real FifoWriter/ffmpeg rendezvous this backend
    # replaces.
    def _write_frames() -> None:
        with open(fifo_path, "wb", buffering=0) as fh:
            fh.write(b"\xd5" * 160)
            fh.write(b"\xd5" * 160)

    writer_thread = threading.Thread(target=_write_frames, daemon=True)
    writer_thread.start()
    writer_thread.join(timeout=2)

    # header part + two A-law frame parts
    await _wait_for(lambda: len(fake_session._writer.parts) >= 3)

    await supervisor.stop()

    assert fake_session.entered
    assert fake_session.exited

    header_part = fake_session._writer.parts[0]
    assert mpegts.build_header() in header_part
    assert b"Content-Type: audio/mp2t" in header_part
    assert b"X-If-Encrypt: 0" in header_part
    assert b"X-Session-Id: 4242" in header_part

    frame_part = fake_session._writer.parts[1]
    assert b"Content-Type: audio/mp2t" in frame_part
    assert b"X-If-Encrypt: 0" in frame_part


async def test_tapo_talk_supervisor_stop_closes_the_session_while_blocked_on_a_fifo_read(
    tmp_path, monkeypatch, caplog
):
    """260923-spd: a live pod restart left the camera holding a stale talk
    session and refusing port 8800 with 401 until a power cycle, because
    `stop()` relied on task cancellation reaching `_stream`'s blocking
    `fifo_fh.read()` -- cancelling a task blocked on an OS-level read in an
    executor thread does not interrupt that read; the thread keeps
    blocking until a frame arrives or the write end closes. This attaches
    a real writer (so the supervisor's own reader unblocks and the talk
    session starts) that then sends nothing -- the exact "idle turn, no
    TTS audio in flight" shape the live bug hit -- and proves `stop()`
    still closes the session (and returns) within a bounded time."""
    monkeypatch.setenv(TAPO_CLOUD_PASSWORD_ENV, "not-a-real-password")

    fifo_path = str(tmp_path / "speaker.fifo")
    os.mkfifo(fifo_path)

    fake_session = _FakeMediaSession()

    async def fake_build_session(host: str, cloud_password: str):
        return fake_session

    config = SpeakerConfig(fifo_path=fifo_path, respawn_backoff_s=0.05)
    supervisor = TapoTalkSupervisor(config, "192.0.2.5", build_session=fake_build_session)
    supervisor.start()

    # Opening the write end blocks until the supervisor's own reader
    # attaches (real pipe rendezvous, matching the first test above) --
    # once open, it is never written to, so `_stream`'s read blocks
    # exactly like an idle turn with no TTS audio queued.
    writer_fh = await asyncio.get_running_loop().run_in_executor(None, open, fifo_path, "wb", 0)
    try:
        await _wait_for(lambda: fake_session.entered)
        # Let `_stream`'s executor thread actually reach its blocking
        # `fifo_fh.read()` call before `stop()` runs -- without this, a
        # `task.cancel()` that lands before the executor thread has picked
        # up the read from its queue can succeed trivially (the work item
        # never started), which would make this test pass against the old,
        # buggy code by sheer timing luck rather than exercising the case
        # it names.
        await asyncio.sleep(0.3)

        import logging
        import time

        # `asyncio.wait_for` here is a safety net only, generous enough
        # (5s) that it never fires against correct code -- it must not be
        # the thing that proves `stop()` is bounded. `wait_for` itself
        # forces a cancellation into whatever `stop()` is awaiting once
        # ITS OWN deadline arrives, so wrapping a hanging `stop()` in
        # `wait_for(..., timeout=2)` would make `stop()` return (via the
        # `contextlib.suppress(asyncio.CancelledError)` already inside it)
        # right around that 2s mark even with the bug still present --
        # passing this test for the wrong reason. Measuring elapsed time
        # directly, against a threshold far below the safety net, is what
        # actually distinguishes "closed promptly on its own" from
        # "only closed because this test's own timeout forced it."
        with caplog.at_level(logging.WARNING, logger="atlas.speaker.tapo_talk"):
            started = time.monotonic()
            await asyncio.wait_for(supervisor.stop(), timeout=5)
            elapsed = time.monotonic() - started

        assert fake_session.exited, "stop() must close the session even while _stream is blocked on a FIFO read"
        assert elapsed < 1.0, (
            f"stop() took {elapsed:.2f}s -- it must notice a stop request within "
            "one poll interval of the blocked FIFO read, not rely on this test's "
            "own safety-net timeout to force it through"
        )
        assert any("session closed for shutdown" in record.message for record in caplog.records)
    finally:
        writer_fh.close()


async def test_tapo_talk_supervisor_stays_in_backoff_without_the_cloud_password(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv(TAPO_CLOUD_PASSWORD_ENV, raising=False)

    fifo_path = str(tmp_path / "speaker.fifo")
    os.mkfifo(fifo_path)

    build_calls = 0

    async def fake_build_session(host: str, cloud_password: str):
        nonlocal build_calls
        build_calls += 1
        return _FakeMediaSession()

    config = SpeakerConfig(fifo_path=fifo_path, respawn_backoff_s=0.02)
    supervisor = TapoTalkSupervisor(config, "192.0.2.5", build_session=fake_build_session)

    import logging

    with caplog.at_level(logging.WARNING, logger="atlas.speaker.tapo_talk"):
        supervisor.start()
        await asyncio.sleep(0.1)
        await supervisor.stop()

    assert build_calls == 0
    assert any("not set" in record.message for record in caplog.records)


async def test_tapo_talk_supervisor_times_out_a_stalled_connect_into_backoff(tmp_path, monkeypatch, caplog):
    """A handshake that never returns must fail into the normal backoff and
    retry, not hang the supervisor loop forever."""
    monkeypatch.setenv(TAPO_CLOUD_PASSWORD_ENV, "not-a-real-password")

    fifo_path = str(tmp_path / "speaker.fifo")
    os.mkfifo(fifo_path)

    build_calls = 0

    async def stalled_build_session(host: str, cloud_password: str):
        nonlocal build_calls
        build_calls += 1
        await asyncio.Event().wait()  # never set: the camera never answers

    config = SpeakerConfig(fifo_path=fifo_path, respawn_backoff_s=0.01)
    supervisor = TapoTalkSupervisor(
        config, "192.0.2.5", build_session=stalled_build_session, connect_timeout_s=0.02
    )

    import logging

    with caplog.at_level(logging.WARNING, logger="atlas.speaker.tapo_talk"):
        supervisor.start()
        await _wait_for(lambda: build_calls >= 2)
        await supervisor.stop()

    assert any("TimeoutError" in record.message for record in caplog.records)


async def test_tapo_talk_supervisor_doubles_its_backoff_on_consecutive_failures(tmp_path, monkeypatch):
    """A fixed short retry kept the camera's talk-port lockout armed for 20+
    minutes live; consecutive failures must back off exponentially."""
    monkeypatch.setenv(TAPO_CLOUD_PASSWORD_ENV, "not-a-real-password")

    async def failing_build_session(host: str, cloud_password: str):
        raise RuntimeError("401")

    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def recording_sleep(seconds: float) -> None:
        if seconds >= 1:  # the supervisor's backoff, not `_wait_for`'s own polling
            delays.append(seconds)
        await real_sleep(0)

    import atlas.speaker.tapo_talk as tapo_talk_module

    monkeypatch.setattr(tapo_talk_module.asyncio, "sleep", recording_sleep)
    config = SpeakerConfig(fifo_path=str(tmp_path / "speaker.fifo"), respawn_backoff_s=30.0)
    supervisor = TapoTalkSupervisor(config, "192.0.2.5", build_session=failing_build_session)
    supervisor.start()
    await _wait_for(lambda: len(delays) >= 7)
    await supervisor.stop()

    assert delays[:7] == [30.0, 60.0, 120.0, 240.0, 480.0, 600.0, 600.0]
