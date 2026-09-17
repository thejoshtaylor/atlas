"""`FifoWriter`: the write side of the camera speaker's named pipe.

Opening a FIFO for writing blocks at the operating-system level until a
reader attaches (RESEARCH.md Pitfall 4) -- every open here, including the
reopen Task 2 adds for the reader-loss case, runs in an executor thread and
never inline on the event loop. An inline open would hang the whole
application with no exception and no log line; "no reader yet" is an
expected startup-ordering state, not an error, so an in-flight open never
raises on that basis alone.

`SpeakerError` is raised on an unrecoverable open, in the shape
`providers/base.py` establishes for every other provider error.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger("spire_voice.speaker.fifo_writer")


class SpeakerError(Exception):
    """Raised on an unrecoverable FIFO open."""


class FifoWriter:
    """Owns the write side of `fifo_path`, opened off the event loop thread."""

    def __init__(self, fifo_path: str) -> None:
        self._fifo_path = fifo_path
        self._fh: Any | None = None

    async def open(self) -> None:
        """Open the FIFO for writing. Blocks (in an executor thread) until
        a reader attaches -- see the module docstring."""
        loop = asyncio.get_running_loop()
        try:
            self._fh = await loop.run_in_executor(None, self._blocking_open)
        except OSError as exc:
            raise SpeakerError(
                f"could not open speaker FIFO {self._fifo_path!r}: {exc}"
            ) from exc

    async def write(self, chunk: bytes) -> None:
        if self._fh is None:
            raise SpeakerError("FifoWriter.write called before open()")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._fh.write, chunk)

    async def close(self) -> None:
        if self._fh is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._fh.close)
            self._fh = None

    def _blocking_open(self) -> Any:
        """Create the FIFO node if the mount only provided its directory,
        then open it for writing. Runs in an executor thread -- never
        called directly from the event loop."""
        if not os.path.exists(self._fifo_path):
            os.mkfifo(self._fifo_path)
        return open(self._fifo_path, "wb", buffering=0)
