"""`FifoWriter` and `FfmpegSupervisor`: one process for every utterance, and
a pipe that survives losing its reader (VOICE-05, T-02-10, T-02-11).

Every case here uses a real named pipe under the test's own temporary
directory, with a real reader thread on the other end -- the pipe
semantics being tested (blocking open, broken-pipe-on-writer, atomic short
writes) are the operating system's, and faking them would test the fake,
not the writer.

The `ffmpeg` child itself is faked (`_FakeSpawner`/`_FakeProcess`): what
these tests prove is the supervisor's own spawn/restart bookkeeping, not
`ffmpeg`'s own behavior, which is out of this process's control.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading

from spire_voice.config import SpeakerConfig
from spire_voice.speaker.ffmpeg_supervisor import FfmpegSupervisor
from spire_voice.speaker.fifo_writer import FifoWriter


class _FakeProcess:
    """A `_FakeSpawner`-issued stand-in for `asyncio.subprocess.Process`:
    only `wait()`/`returncode`, the two members `FfmpegSupervisor` actually
    reads (its own module docstring's "single source of truth" doctrine)."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self._exited = asyncio.Event()

    def exit(self, returncode: int) -> None:
        self.returncode = returncode
        self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode


class _FakeSpawner:
    """Counts spawns and hands back one scripted `_FakeProcess` per call --
    the conftest-style fake the plan asks for, local to this file since no
    other test needs it."""

    def __init__(self, processes: list[_FakeProcess]) -> None:
        self._processes = list(processes)
        self.spawn_count = 0
        self.spawned_argv: list[list[str]] = []

    async def __call__(self, argv: list[str]) -> _FakeProcess:
        self.spawned_argv.append(argv)
        process = self._processes[self.spawn_count]
        self.spawn_count += 1
        return process


async def _wait_for(predicate, *, attempts: int = 200, interval: float = 0.01) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"condition never became true after {attempts * interval:.2f}s")


def _drain_forever(path: str, out: list[bytes], stop: threading.Event) -> None:
    """A always-attached reader standing in for the real `ffmpeg` child's
    own read side -- these tests exercise the writer's behavior, not a real
    subprocess, so this thread plays that role directly."""
    with open(path, "rb", buffering=0) as fh:
        while not stop.is_set():
            chunk = fh.read(4096)
            if not chunk:
                break
            out.append(chunk)


async def test_speaker_fifo_reuses_one_subprocess_across_two_utterances(tmp_path):
    """VOICE-05: two consecutive utterances through the writer must not
    cost a second `ffmpeg` spawn -- asserted by counting spawns, not by
    timing them, since a timing assertion would pass on a slow machine that
    spawned twice anyway."""
    fifo_path = str(tmp_path / "speaker.alaw")
    os.mkfifo(fifo_path)

    received: list[bytes] = []
    stop_reader = threading.Event()
    reader_thread = threading.Thread(target=_drain_forever, args=(fifo_path, received, stop_reader), daemon=True)
    reader_thread.start()

    spawner = _FakeSpawner([_FakeProcess()])
    config = SpeakerConfig(fifo_path=fifo_path, respawn_backoff_s=0.01)
    supervisor = FfmpegSupervisor(config, spawn=spawner)
    supervisor.start()
    await _wait_for(lambda: spawner.spawn_count == 1)

    writer = FifoWriter(fifo_path)
    await writer.open()
    await writer.write(b"utterance-one")
    await writer.write(b"utterance-two")
    await _wait_for(lambda: b"".join(received) == b"utterance-oneutterance-two")

    assert spawner.spawn_count == 1

    await supervisor.stop()
    await writer.close()
    stop_reader.set()
    reader_thread.join(timeout=2)


async def test_dead_child_restarts_after_backoff_and_logs_exit_status(tmp_path, caplog):
    """T-02-11: a child that exits is restarted after the configured
    backoff, and the restart is visible in the log -- a silent drop is the
    exact failure mode this supervisor exists to prevent."""
    fifo_path = str(tmp_path / "speaker.alaw")
    os.mkfifo(fifo_path)

    first_process = _FakeProcess()
    second_process = _FakeProcess()
    spawner = _FakeSpawner([first_process, second_process])
    config = SpeakerConfig(fifo_path=fifo_path, respawn_backoff_s=0.01)
    supervisor = FfmpegSupervisor(config, spawn=spawner)

    with caplog.at_level(logging.WARNING, logger="spire_voice.speaker.ffmpeg_supervisor"):
        supervisor.start()
        await _wait_for(lambda: spawner.spawn_count == 1)

        first_process.exit(17)
        await _wait_for(lambda: spawner.spawn_count == 2)

    await supervisor.stop()

    assert any("exited with code" in record.message and "17" in record.message for record in caplog.records)


async def test_write_after_reader_loss_reopens_the_pipe(tmp_path):
    """RESEARCH.md Pitfall 5: a write after every reader has closed must
    reopen the pipe rather than propagating `BrokenPipeError` to the
    caller."""
    fifo_path = str(tmp_path / "speaker.alaw")
    os.mkfifo(fifo_path)

    first_reader_done = threading.Event()

    def _first_reader() -> None:
        with open(fifo_path, "rb", buffering=0) as fh:
            fh.read(64)
        first_reader_done.set()

    first_reader_thread = threading.Thread(target=_first_reader, daemon=True)
    first_reader_thread.start()

    writer = FifoWriter(fifo_path)
    await writer.open()
    await writer.write(b"first-chunk")

    await asyncio.get_running_loop().run_in_executor(None, first_reader_done.wait, 2)
    assert first_reader_done.is_set(), "the first reader never attached/closed"

    received: list[bytes] = []
    second_reader_done = threading.Event()

    def _second_reader() -> None:
        with open(fifo_path, "rb", buffering=0) as fh:
            received.append(fh.read(64))
        second_reader_done.set()

    second_reader_thread = threading.Thread(target=_second_reader, daemon=True)
    second_reader_thread.start()

    # The write that hits the broken pipe -- must not raise to the caller.
    await writer.write(b"second-chunk")

    await asyncio.get_running_loop().run_in_executor(None, second_reader_done.wait, 2)
    assert second_reader_done.is_set(), "the reopened pipe never found its new reader"
    assert b"".join(received) == b"second-chunk"

    await writer.close()
    first_reader_thread.join(timeout=2)
    second_reader_thread.join(timeout=2)
