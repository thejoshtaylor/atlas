"""260923-sfi (D3): a real-ffmpeg guarantee test for the tail-flush pad.

Measured at planning time with ffmpeg 8.1.2: `build_tcp_argv`'s sender
holds the partial last packet of its raw demuxer -- at most a tenth of a
second of audio -- until more bytes arrive. The Pi receiver command
(`-fflags nobuffer -probesize 32 -analyzeduration 0 -f nut -listen 1`)
does not hold anything at the tail; it drops the first packet of every new
connection instead, because `-fflags nobuffer` discards the packet the
probe reads. This test's receiver mirrors that command exactly (with the
output side changed to a pipe this test can read).

The primer write below exists only to take that dropped first packet --
it is not part of what this test measures. The real writer stays open for
the whole test: closing it would flush the held packet through EOF and
hide the exact bug this test exists to catch. The unpadded case's
half-match (the front of the audio arrives, the back does not) proves the
chain actually delivered *something*, so a missing tail is the held
packet and not a dead harness.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import socket

import numpy as np
import pytest

from atlas.audio.alaw import alaw_to_pcm16, pcm16_to_alaw
from atlas.audio.cue import silence
from atlas.config import SpeakerConfig
from atlas.providers.tts_xai import SinkFormat
from atlas.speaker.ffmpeg_supervisor import build_tcp_argv
from atlas.speaker.fifo_writer import FifoWriter

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="this test needs a real ffmpeg on PATH")

_POLL_INTERVAL_S = 0.01
_DEADLINE_S = 1.5


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _wait_for_listener(port: int, timeout_s: float = 5.0) -> None:
    """A fresh `bind()` on the same port, not `connect()`: the receiver's
    `-listen 1` accepts exactly one connection, so a probe `connect()`
    would take the one slot this test's own sender needs. `bind()` instead
    fails with `EADDRINUSE` the instant ffmpeg has the port bound, with no
    side effect on the listener itself.
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return  # EADDRINUSE: ffmpeg already holds the port
        else:
            probe.close()
            await asyncio.sleep(_POLL_INTERVAL_S)
        finally:
            probe.close()
    raise AssertionError(f"no listener bound to 127.0.0.1:{port} within {timeout_s}s")


async def _wait_until(predicate, timeout_s: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(_POLL_INTERVAL_S)


def _build_audio(sink: SinkFormat) -> tuple[bytes, bytes]:
    """`(to_write, expected)`: 0.4s of non-silent, non-periodic audio in
    `sink`'s codec, and the PCM16 bytes a correct delivery reconstructs to.
    """
    n = int(sink.sample_rate * 0.4)
    pcm16 = np.round(8000 * np.sin(np.arange(n) / 5)).astype("<i2").tobytes()
    if sink.codec == "alaw":
        return pcm16_to_alaw(pcm16), alaw_to_pcm16(pcm16_to_alaw(pcm16))
    return pcm16, pcm16


@pytest.mark.parametrize(
    "sink, padded",
    [
        (SinkFormat("pcm", 24000), True),
        (SinkFormat("pcm", 24000), False),
        (SinkFormat("alaw", 8000), True),
        (SinkFormat("alaw", 8000), False),
    ],
    ids=["pcm24000-padded", "pcm24000-unpadded", "alaw8000-padded", "alaw8000-unpadded"],
)
async def test_tail_flush_delivers_the_last_write_only_when_padded(tmp_path, sink, padded):
    fifo_path = str(tmp_path / "speaker.fifo")
    os.mkfifo(fifo_path)  # before the sender starts -- ffmpeg exits on a missing input path

    port = _free_port()
    receiver = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-f",
        "nut",
        "-listen",
        "1",
        "-i",
        f"tcp://127.0.0.1:{port}",
        "-f",
        "s16le",
        "-flush_packets",
        "1",
        "pipe:1",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    received = bytearray()

    async def _read_forever() -> None:
        assert receiver.stdout is not None
        while True:
            chunk = await receiver.stdout.read(65536)
            if not chunk:
                return
            received.extend(chunk)

    reader_task = asyncio.create_task(_read_forever())
    sender = None
    writer = None
    try:
        await _wait_for_listener(port)
        await asyncio.sleep(0.1)

        sender_config = SpeakerConfig(backend="tcp", tcp_url=f"tcp://127.0.0.1:{port}", fifo_path=fifo_path)
        sender = await asyncio.create_subprocess_exec(
            *build_tcp_argv(sender_config, sink),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        writer = FifoWriter(
            fifo_path,
            pad_bytes=silence(sink, 0.2) if padded else b"",
            pad_idle_s=0.15,
        )
        await asyncio.wait_for(writer.open(), 5)

        # The primer takes the connection's first packet, which the
        # receiver drops (module docstring) -- without it, the audio this
        # test actually measures would be the one eaten by that drop.
        await writer.write(silence(sink, 0.3))
        await _wait_until(lambda: len(received) > 0, 5)
        await asyncio.sleep(0.3)
        received.clear()

        to_write, expected = _build_audio(sink)
        await writer.write(to_write)

        deadline = asyncio.get_running_loop().time() + _DEADLINE_S
        while asyncio.get_running_loop().time() < deadline:
            if expected in bytes(received):
                break
            await asyncio.sleep(_POLL_INTERVAL_S)

        got = bytes(received)
        if padded:
            assert expected in got, "padded audio must arrive in full within 1.5s"
        else:
            assert expected not in got, "unpadded audio must NOT arrive in full -- the tail stays held"
            assert expected[: len(expected) // 2] in got, (
                "the front half must still arrive -- otherwise this proves nothing "
                "about a held tail, only a dead harness"
            )
    finally:
        if writer is not None:
            await writer.close()
        for proc in (sender, receiver):
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
        reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader_task
