"""`FifoWriter`: the write side of the camera speaker's named pipe.

Opening a FIFO for writing blocks at the operating-system level until a
reader attaches (RESEARCH.md Pitfall 4) -- every open here runs in an
executor thread and never inline on the event loop. An inline open would
hang the whole application with no exception and no log line.

That "runs in a thread" discipline stops an open from blocking the event
loop; it does nothing about an open that blocks the *thread* forever. The
initial `open()` accepts that: "no reader yet" is an expected
startup-ordering state, and no timeout is correct for "however long the
rest of the application takes to wire itself up," so it stays unbounded.
The reopen `write()` performs after a reader loss has no such excuse -- by
then a reader is expected to already be running or imminently respawning,
so it is bounded by `reopen_timeout_s` (CR-04, code review: an earlier
version of this reopen shared the initial open's blocking call and could
hang the write path forever, silently, with the executor thread it
consumed never coming back).

`SpeakerError` is raised on an unrecoverable open, in the shape
`providers/base.py` establishes for every other provider error.

260923-sfi (D1): idle tail padding. ffmpeg's raw demuxer reads the FIFO in
fixed packets of at most 0.1 s and holds a partial last packet until more
audio arrives -- the end of a reply then waits for the next reply to push
it out. One silence write, after an idle gap, pushes that packet out at
once. The pad and every real write share one lock, so a pad can never
split a chunk. A real write cancels a pad that has not started yet. A pad
that has already started finishes first, and it never re-arms itself --
the next real write is what arms the next one.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import os
import time
from typing import Any

logger = logging.getLogger("atlas.speaker.fifo_writer")

# The reopen retry loop's own backoff (distinct from `respawn_backoff_s`,
# which paces ffmpeg child restarts): starts fast enough to catch a reader
# that reattaches almost immediately, doubles up to the cap so a reopen
# that is genuinely going to time out does not spend its whole budget
# spinning against ENXIO.
_REOPEN_RETRY_INITIAL_S = 0.05
_REOPEN_RETRY_MAX_S = 0.5


class SpeakerError(Exception):
    """Raised on an unrecoverable FIFO open."""


class FifoWriter:
    """Owns the write side of `fifo_path`, opened off the event loop thread."""

    def __init__(
        self,
        fifo_path: str,
        *,
        reopen_timeout_s: float = 10.0,
        pad_bytes: bytes = b"",
        pad_idle_s: float = 0.15,
    ) -> None:
        if pad_bytes and pad_idle_s <= 0:
            raise ValueError(
                f"pad_idle_s must be positive when pad_bytes is set, got {pad_idle_s!r}"
            )
        self._fifo_path = fifo_path
        self._fh: Any | None = None
        self._reopen_timeout_s = reopen_timeout_s
        self._pad_bytes = pad_bytes
        self._pad_idle_s = pad_idle_s
        # The tail-flush pad and every real write share this one lock, so a
        # pad can never split a chunk (module docstring).
        self._write_lock = asyncio.Lock()
        self._pad_task: asyncio.Task[None] | None = None
        # True from the moment a pad task decides to write until it is
        # done -- write() reads this to know a pad is past the point where
        # cancelling it is safe (module docstring).
        self._pad_writing = False

    async def open(self) -> None:
        """Open the FIFO for writing. Blocks (in an executor thread) until
        a reader attaches, with no timeout -- see the module docstring."""
        loop = asyncio.get_running_loop()
        try:
            self._fh = await loop.run_in_executor(None, self._blocking_open)
        except OSError as exc:
            raise SpeakerError(
                f"could not open speaker FIFO {self._fifo_path!r}: {exc}"
            ) from exc

    async def write(self, chunk: bytes) -> None:
        """Write `chunk`, transparently reopening the FIFO if every reader
        has closed since the last write.

        A FIFO's writer and reader are independent opens against the same
        path (RESEARCH.md Pitfall 5): once every reader closes, the next
        write raises `BrokenPipeError`, and respawning the egress
        supervisor's `ffmpeg` child alone does not repair this side's own
        file descriptor. Catching that here, closing, and reopening is what
        makes losing a reader a reopen rather than a permanent end to every
        future reply -- the caller never sees the broken pipe at all.

        The reopen itself is bounded by `reopen_timeout_s` (module
        docstring). A reopen that never finds a reader within that window
        raises `SpeakerError` instead: `sources/runner.py`'s per-chunk
        containment (CR-03) logs and continues from it, which is strictly
        better than the caller -- and the executor thread it was
        awaiting on -- never coming back at all.

        260923-sfi: this real write always wins over the tail-flush pad. A
        pad that has not started yet is cancelled outright; a pad that has
        already started keeps writing first, and this write simply waits
        its turn on `_write_lock`. A new pad is armed only once this write
        actually succeeds -- an exception here arms nothing.
        """
        self._cancel_pending_pad()
        async with self._write_lock:
            await self._write_unlocked(chunk)
        if self._pad_bytes:
            self._arm_pad()

    async def _write_unlocked(self, chunk: bytes) -> None:
        """The write itself -- unchanged from before the tail-flush pad
        existed. Callers hold `_write_lock` first (`write()` and
        `_pad_after_idle()`), so a pad write can never land in the middle
        of a real one, or the other way around."""
        loop = asyncio.get_running_loop()
        if self._fh is None:
            # A previous reopen gave up. Try again, bounded the same way:
            # without this, one timed-out reopen left `_fh` at None for the
            # life of the process, and the speaker stayed silent after the
            # talk session came back (seen live after a camera lockout).
            await self._reopen(loop)
            await loop.run_in_executor(None, self._fh.write, chunk)
            return
        try:
            await loop.run_in_executor(None, self._fh.write, chunk)
        except BrokenPipeError:
            logger.warning("speaker FIFO reader disappeared; reopening %r", self._fifo_path)
            # Closed inline, not via the executor: unlike an open, closing a
            # pipe fd never blocks on a reader, and every tick this stays
            # open past the failed write is a tick a brand new reader can
            # spend rendezvousing with this dead-but-not-yet-closed fd
            # instead of the reopen below -- getting an instant, spurious
            # EOF the moment this line finally runs, rather than the
            # fresh connection it was waiting for.
            self._fh.close()
            self._fh = None
            await self._reopen(loop)
            await loop.run_in_executor(None, self._fh.write, chunk)

    def _cancel_pending_pad(self) -> None:
        """Cancel `_pad_task` if it exists and has not started writing yet.
        A pad already writing (`_pad_writing`) is left alone -- it finishes
        first, and whatever called this simply waits for `_write_lock`."""
        if self._pad_task is not None and not self._pad_writing:
            self._pad_task.cancel()

    def _arm_pad(self) -> None:
        """Schedule the next tail-flush pad, replacing any pad that has not
        started yet."""
        self._cancel_pending_pad()
        self._pad_task = asyncio.create_task(self._pad_after_idle())

    async def _pad_after_idle(self) -> None:
        """Wait for `pad_idle_s` of silence, then write `pad_bytes` once.

        Sets `_pad_writing` the instant the wait ends -- before its first
        await past that point -- so a real write arriving right after
        cannot cancel this pad out from under it (module docstring): from
        here, this pad either writes or a real write is already waiting
        for `_write_lock` behind it. Never calls `_reopen` on a broken
        pipe: the next real write repairs that the same way it always has.
        Never re-arms itself -- only a real write does that.
        """
        try:
            await asyncio.sleep(self._pad_idle_s)
            self._pad_writing = True
            async with self._write_lock:
                if self._fh is None:
                    return
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(None, self._fh.write, self._pad_bytes)
                except OSError as exc:
                    logger.debug("tail-flush pad write failed for %r: %s", self._fifo_path, exc)
        finally:
            self._pad_writing = False
            if self._pad_task is asyncio.current_task():
                self._pad_task = None

    async def _reopen(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            self._fh = await loop.run_in_executor(
                None, self._blocking_reopen, self._reopen_timeout_s
            )
        except OSError as exc:
            raise SpeakerError(
                f"speaker FIFO {self._fifo_path!r} found no reader within "
                f"{self._reopen_timeout_s:g}s: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the FIFO, and cancel a pending tail-flush pad first so
        teardown never races a pad write. Does not take `_write_lock` --
        the same as before the pad existed -- so a write stuck in the
        reopen loop cannot hang teardown."""
        if self._pad_task is not None:
            self._pad_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pad_task
            self._pad_task = None
        if self._fh is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._fh.close)
            self._fh = None

    def _blocking_open(self) -> Any:
        """Create the FIFO node if the mount only provided its directory,
        then open it for writing -- blocks until a reader attaches, with no
        timeout (see the module docstring for why). Runs in an executor
        thread -- never called directly from the event loop."""
        if not os.path.exists(self._fifo_path):
            os.mkfifo(self._fifo_path)
        return open(self._fifo_path, "wb", buffering=0)

    def _blocking_reopen(self, timeout_s: float) -> Any:
        """Reopen the FIFO for writing after a reader loss, bounded by
        `timeout_s` (module docstring explains why the reopen is bounded
        and the initial open is not). Runs in an executor thread -- never
        called directly from the event loop.

        A blocking `open()` call has no way to give up partway through --
        the kernel does not return until a reader attaches, full stop.
        Opening `O_NONBLOCK` instead turns "no reader yet" into an
        immediate `ENXIO` this loop can retry on its own schedule, so the
        timeout is enforced here, in Python, rather than left to the
        kernel to never enforce at all. `os.set_blocking` restores ordinary
        blocking writes on the fd once a reader has actually attached --
        the resulting file object behaves exactly like the initial open's.
        """
        deadline = time.monotonic() + timeout_s
        delay = _REOPEN_RETRY_INITIAL_S
        while True:
            try:
                fd = os.open(self._fifo_path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as exc:
                if exc.errno != errno.ENXIO:  # not "no reader attached yet"
                    raise
            else:
                os.set_blocking(fd, True)
                return os.fdopen(fd, "wb", buffering=0)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no reader attached to speaker FIFO {self._fifo_path!r} "
                    f"within {timeout_s:g}s"
                )
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, _REOPEN_RETRY_MAX_S)
