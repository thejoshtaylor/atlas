"""Session retention sweep (DBG-06): audio of a home does not outlive the
operator's configured window because nobody noticed.

This is the first scheduled or periodic module in this codebase -- Phase 1
and Phase 01.1 have no cron-like task of any kind to copy from -- so its own
governing constraint is stated here the way `timing.py` and `wake/gate.py`
both state theirs: this module deletes directories under one configured
root and nothing else. It reads no recording's contents. It never touches a
directory outside that root, and it never touches a directory under the
root whose name it does not recognize as one `session/recorder.py` itself
wrote.

Two pieces live here:

- `sweep_expired_sessions()`: one run. Takes the session root, the
  retention window and a clock as explicit parameters, evaluates every
  candidate directory's age against exactly one clock reading, removes the
  expired ones oldest first, and logs the run whether or not it removed
  anything -- an operator asking "is my audio being deleted" needs to tell
  "ran and had nothing to do" apart from "never ran", and a silent empty
  run makes those identical.
- `RetentionScheduler`: runs the sweep once at startup and again every
  configured interval, for the life of the process -- the same
  `start()`/`stop()` shape `speaker/ffmpeg_supervisor.py`'s `FfmpegSupervisor`
  already uses, with no scheduling dependency, because this is one loop
  that sleeps and calls one function.

Privacy boundary, extended verbatim from `timing.py` and
`session/recorder.py`: every log line this module writes names a path, an
age and a count -- never a transcript, a reply, or an audio byte, because
deleting the file is not a license to read it first.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger("spire_voice.session.retention")

# `session/recorder.py`'s own `_directory_name()`: a UTC timestamp in
# `%Y%m%dT%H%M%S%fZ` form (8 date digits, "T", 12 time-plus-microsecond
# digits, "Z"), a dash, then the turn id. Age is read from this name, never
# from filesystem mtime, which a backup, a copy or a container restore can
# rewrite without anything having changed about the recording (T-02-35). A
# directory under the root whose name does not match this shape is left
# untouched -- it is not one this module owns. Matched with `fullmatch`,
# never `match`: `$` accepts a trailing newline, so `re.match` would parse
# a name the recorder could not have written as though it had.
#
# Public as of Phase 8 (08-04-PLAN.md, D-01): `routes/sessions.py` reads the
# session list straight off disk and needs this exact pattern, not a second,
# independently written one that could disagree with this sweep's own at a
# boundary case. `_SESSION_DIRECTORY_RE` stays as a module alias for any
# caller already using the old, private name.
SESSION_DIRECTORY_RE = re.compile(r"^(\d{8}T\d{12}Z)-.+$")
_SESSION_DIRECTORY_RE = SESSION_DIRECTORY_RE


def parse_session_timestamp(name: str) -> datetime | None:
    match = SESSION_DIRECTORY_RE.fullmatch(name)
    if match is None:
        return None
    stamp = match.group(1)
    try:
        # Drop the trailing literal "Z"; the value is already UTC by
        # construction (`recorder.py` uses `datetime.now(timezone.utc)`).
        return datetime.strptime(stamp[:-1], "%Y%m%dT%H%M%S%f").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# Alias kept for the same reason as `_SESSION_DIRECTORY_RE` above.
_parse_session_timestamp = parse_session_timestamp


@dataclass(frozen=True)
class RemovedSession:
    """One directory this sweep actually removed."""

    path: Path
    age_days: float


def sweep_expired_sessions(
    root: str | Path,
    retain_days: int,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    remove: Callable[[Path], None] = shutil.rmtree,
) -> list[RemovedSession]:
    """Remove every session directory under `root` strictly older than
    `retain_days`, oldest first, and log the run whether or not it removed
    anything.

    `clock` is read exactly once, at the top of this call, and every
    directory's age is measured against that single reading (CD/D-14): a
    run that re-read the clock per directory could keep a session at the
    top of a long list and remove an identically-aged one further down,
    meaning a session could be both kept and removed within one run
    depending on how long the run took. One reading makes that impossible.

    The boundary: a session whose age is exactly `retain_days` is kept, one
    strictly older is removed. `retain_days` is a floor the operator raises,
    never a cache size to shave a session off early.

    `remove` defaults to `shutil.rmtree` and is the one injectable seam
    besides `clock` -- it exists so a test can simulate an undeletable
    directory (a permission error, a directory a recorder still holds open)
    deterministically, without depending on the host's own permission model
    or on timing a real file lock. A failure removing one directory is
    logged and stepped over; it does not abort the run, because aborting on
    the first failure would let one undeletable path pin unbounded disk --
    audio of a house accumulating indefinitely, exactly what this schedule
    exists to prevent.
    """
    root_path = Path(root)
    now = clock()
    window = timedelta(days=retain_days)

    try:
        resolved_root = root_path.resolve()
    except OSError:
        logger.warning("session retention sweep: configured root could not be resolved, skipping", extra={"root": str(root_path)})
        return []

    candidates: list[tuple[datetime, Path]] = []
    if resolved_root.is_dir():
        for entry in resolved_root.iterdir():
            if not entry.is_dir():
                continue
            session_time = _parse_session_timestamp(entry.name)
            if session_time is None:
                continue
            try:
                resolved_entry = entry.resolve()
            except OSError:
                continue
            # Never follow a symlink (or any other indirection) out of the
            # configured root -- a retention bug that walks outside its own
            # root would delete something that is not a session (T-02-34).
            if resolved_entry != resolved_root and resolved_root not in resolved_entry.parents:
                continue
            candidates.append((session_time, resolved_entry))

    candidates.sort(key=lambda item: item[0])  # oldest first

    removed: list[RemovedSession] = []
    for session_time, path in candidates:
        age = now - session_time
        if age <= window:
            continue
        age_days = age.total_seconds() / 86400.0
        try:
            remove(path)
        except OSError:
            logger.warning(
                "session retention: could not remove expired session path=%s age_days=%.2f",
                path,
                age_days,
                exc_info=True,
            )
            continue
        removed.append(RemovedSession(path=path, age_days=age_days))
        logger.info(
            "session retention: removed expired session path=%s age_days=%.2f",
            path,
            age_days,
        )

    # Logged unconditionally -- including the empty case. A run that
    # removed nothing must be visible as "ran, nothing to do", never
    # silence indistinguishable from "never ran" (T-02-36).
    logger.info(
        "session retention sweep ran: removed=%d checked=%d root=%s",
        len(removed),
        len(candidates),
        resolved_root,
    )
    return removed


class RetentionScheduler:
    """Runs `sweep_expired_sessions` once immediately and again every
    `interval_s`, for the life of the application.

    The exact `start()`/`stop()` shape `FfmpegSupervisor` already uses: an
    internal `_stopping` flag, checked both before and after the interval
    sleep. A process that restarts more often than `interval_s` would
    otherwise never sweep at all, which is why the first sweep runs
    immediately rather than waiting out the first interval. The flag is
    re-checked once the sleep resolves, matching the shape plan 02-07 used
    for the camera's reconnect backoff, for the same reason: a sweep that
    wakes right after `stop()` and starts deleting is a leak with worse
    consequences than most, and relying on task cancellation alone leaves a
    real, if narrow, window between the sleep resolving and the
    cancellation being delivered.

    A sweep that raises is logged and the loop continues on schedule --
    `asyncio.CancelledError` is re-raised rather than swallowed, since that
    is `stop()`'s own shutdown signal, not a failed run.
    """

    def __init__(
        self,
        root: str | Path,
        retain_days: int,
        interval_s: float,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._root = root
        self._retain_days = retain_days
        self._interval_s = interval_s
        self._clock = clock
        self._sleep = sleep
        self._stopping = False
        self._task: asyncio.Task[None] | None = None
        # Exposed for tests to prove the loop ran more than once by
        # counting, never by timing it.
        self.sweep_count = 0

    def start(self) -> None:
        """Start the schedule. Never awaited by the caller -- the loop runs
        for the life of the application, the same fire-and-forget shape
        `FfmpegSupervisor.start()`/`CameraAudioSource.start()` both use."""
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while not self._stopping:
            self.sweep_count += 1
            try:
                sweep_expired_sessions(self._root, self._retain_days, clock=self._clock)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("session retention sweep raised; the schedule continues")
            if self._stopping:
                return
            await self._sleep(self._interval_s)
            if self._stopping:
                return
