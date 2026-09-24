"""`FfmpegSupervisor`: one long-lived `ffmpeg` child reading the speaker FIFO.

VOICE-05's whole point: a fresh `ffmpeg` process per utterance costs
300-800 ms, the single largest avoidable delay in the round trip
(`config.example.yaml`'s own comment). This module starts exactly one
child, once, and restarts it after `respawn_backoff_s` only if it exits --
never a process pool or a second process-tracking structure. The
subprocess object's own `returncode`/`wait()` are the single source of
truth for whether the child is alive (RESEARCH.md "Don't Hand-Roll":
FIFO/subprocess lifecycle); a supervisor that tracked liveness a second way
would just be a second thing that could go stale.

A child that dies and never comes back would mean the assistant silently
never speaks again (T-02-11) -- every exit and every restart is logged with
the child's exit status, so that failure mode is visible in the log rather
than inferred from silence.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Awaitable, Callable

import httpx

from spire_voice.config import SpeakerConfig

logger = logging.getLogger("spire_voice.speaker.ffmpeg_supervisor")

SpawnFn = Callable[[list[str]], Awaitable[Any]]


def build_ffmpeg_argv(config: SpeakerConfig) -> list[str]:
    """The `ffmpeg` invocation: read the camera's own raw A-law bytes from
    the speaker FIFO and push them to go2rtc's stream with no re-encode
    (`-c copy`) -- the bytes written into the FIFO are already the format
    the camera speaker consumes, so there is nothing for `ffmpeg` to
    transcode here either.

    go2rtc's exact ingest URL shape for a named stream is one of this
    phase's research open questions (02-RESEARCH.md Open Question 2, A3) --
    this builds the most direct reading of `go2rtc_url`/`stream` and is
    exactly what the Task 1 live human check exists to confirm against the
    real deployment.
    """
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "alaw",
        "-ar",
        "8000",
        "-ac",
        "1",
        "-i",
        config.fifo_path,
        "-c",
        "copy",
        "-f",
        "rtsp",
        f"{config.go2rtc_url}/{config.stream}",
    ]


def build_tcp_argv(config: SpeakerConfig) -> list[str]:
    """The `ffmpeg` invocation for the `tcp` backend (260923-pds): the FIFO
    already holds raw 8 kHz A-law, so `-c copy` again -- there is nothing to
    transcode. `-flush_packets 1` sends each packet as soon as `ffmpeg` has
    it, so a short reply is not held in an output buffer waiting for more
    data. The far end is a listener on another machine (for example a
    Raspberry Pi) that plays what it receives.
    """
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "alaw",
        "-ar",
        "8000",
        "-ac",
        "1",
        "-i",
        config.fifo_path,
        "-c",
        "copy",
        "-flush_packets",
        "1",
        "-f",
        "alaw",
        config.tcp_url,
    ]


async def _default_spawn(argv: list[str]) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(*argv)


class FfmpegSupervisor:
    """Owns one supervised `ffmpeg` child for the life of the application."""

    def __init__(
        self,
        config: SpeakerConfig,
        *,
        spawn: SpawnFn = _default_spawn,
        build_argv: Callable[[SpeakerConfig], list[str]] = build_ffmpeg_argv,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._spawn = spawn
        self._build_argv = build_argv
        self._http_client = http_client
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    def start(self) -> None:
        """Start the supervisor loop. Never awaited by the caller -- the
        loop runs for the life of the application, the same fire-and-forget
        shape `CameraAudioSource.start()` uses."""
        self._task = asyncio.create_task(self._supervise())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def handle_reconnect(self) -> None:
        """The third trigger for the backchannel request (plan 02-07),
        alongside startup and every child restart: a camera whose network
        path came back may well have lost the receiving service's producer
        for its own speaker along with it, and a working microphone with a
        dead speaker is a turn that runs and answers into nothing.

        Called as a callback -- something else's composition wires this
        method to a camera source's own reconnect supervisor, never an
        import of that module into this one, so the two supervisors stay
        independent enough to reason about alone (module docstring).
        """
        await self._ensure_backchannel()

    async def _supervise(self) -> None:
        argv = self._build_argv(self._config)
        while not self._stopping:
            await self._ensure_backchannel()
            process = await self._spawn(argv)
            returncode = await process.wait()
            if self._stopping:
                return
            logger.warning(
                "speaker ffmpeg exited with code %s; restarting in %.1fs",
                returncode,
                self._config.respawn_backoff_s,
            )
            await asyncio.sleep(self._config.respawn_backoff_s)

    async def _ensure_backchannel(self) -> None:
        """Issue the idempotent go2rtc backchannel PUT, at startup, again
        after every child restart, and again on a camera reconnect
        (`handle_reconnect`, plan 02-07).

        `ensure_url` arrives whole from a secret and already encodes
        everything go2rtc needs (`config.example.yaml`'s own comment) --
        this code issues the request and never assembles or interprets
        go2rtc's own API shape. A non-success response is logged, matching
        `tts_xai.py`'s own error style, and never stops the supervisor: the
        operator may legitimately bring go2rtc up after this process.
        """
        if not self._config.ensure_url or self._http_client is None:
            return
        try:
            # No request body is sent. RESEARCH.md's Open Question 2 could
            # not confirm go2rtc's exact expected request shape for this
            # endpoint (query params vs. body) -- this is a deliberate,
            # documented gap in what this codebase has verified, not
            # evidence that go2rtc wants no body. Treated as an integration
            # detail the operator resolves when constructing `ensure_url`
            # itself, per this module's own "issues, never assembles"
            # posture above.
            response = await self._http_client.put(self._config.ensure_url, timeout=10.0)
            if response.status_code >= 400:
                logger.warning("speaker backchannel ensure_url PUT failed: %s", response.status_code)
        except httpx.HTTPError as exc:
            logger.warning("speaker backchannel ensure_url PUT raised: %s", type(exc).__name__)
