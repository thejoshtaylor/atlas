"""Session retention sweep (DBG-06): a session at exactly the configured
window is kept, one strictly older is gone, the boundary is decided against
one clock reading, an undeletable directory does not stop the run, a run
with nothing to remove still says so, and nothing outside the configured
root is ever touched.

Every case here drives `sweep_expired_sessions`/`RetentionScheduler`
directly against a real temporary directory tree, named exactly as
`session/recorder.py` names its own session directories -- the retention
sweep parses that name, so a test fixture built any other way would not be
proving what the sweep actually does against a real session directory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from atlas.session.retention import RemovedSession, RetentionScheduler, sweep_expired_sessions

_NOW = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


def _session_dir_name(when: datetime, turn_id: str = "deadbeef") -> str:
    """Exactly `session/recorder.py`'s own `_directory_name()` shape:
    `%Y%m%dT%H%M%S%fZ` plus a dash plus the turn id."""
    return f"{when.strftime('%Y%m%dT%H%M%S%f')}Z-{turn_id}"


def _make_session(root: Path, when: datetime, turn_id: str = "deadbeef") -> Path:
    session_dir = root / _session_dir_name(when, turn_id)
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text("", encoding="utf-8")
    return session_dir


async def _wait_for(predicate, *, attempts: int = 200, interval: float = 0.01) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"condition never became true after {attempts * interval:.2f}s")


async def _instant_sleep(_seconds: float) -> None:
    """A `RetentionScheduler`-injected sleep that yields control back to the
    event loop exactly once, rather than actually waiting -- this is what
    lets a test drive many "intervals" without a real clock wait, while
    still letting other coroutines (the test's own polling) run between
    iterations."""
    await asyncio.sleep(0)


def test_a_session_exactly_at_the_retention_age_is_kept(tmp_path):
    session = _make_session(tmp_path, _NOW - timedelta(days=7))

    removed = sweep_expired_sessions(tmp_path, retain_days=7, clock=lambda: _NOW)

    assert removed == []
    assert session.exists()


def test_a_session_one_step_older_than_the_retention_age_is_removed(tmp_path):
    session = _make_session(tmp_path, _NOW - timedelta(days=7, seconds=1))

    removed = sweep_expired_sessions(tmp_path, retain_days=7, clock=lambda: _NOW)

    assert len(removed) == 1
    assert removed[0].path == session
    assert not session.exists()


def test_a_session_one_step_younger_than_the_retention_age_is_kept(tmp_path):
    session = _make_session(tmp_path, _NOW - timedelta(days=6, hours=23))

    removed = sweep_expired_sessions(tmp_path, retain_days=7, clock=lambda: _NOW)

    assert removed == []
    assert session.exists()


def test_removal_order_is_oldest_first(tmp_path):
    oldest = _make_session(tmp_path, _NOW - timedelta(days=30), turn_id="oldest")
    middle = _make_session(tmp_path, _NOW - timedelta(days=20), turn_id="middle")
    newest_expired = _make_session(tmp_path, _NOW - timedelta(days=10), turn_id="newest")

    removal_order: list[Path] = []

    def recording_remove(path: Path) -> None:
        removal_order.append(path)
        shutil.rmtree(path)

    removed = sweep_expired_sessions(tmp_path, retain_days=7, clock=lambda: _NOW, remove=recording_remove)

    assert removal_order == [oldest, middle, newest_expired]
    assert [r.path for r in removed] == [oldest, middle, newest_expired]


def test_an_undeletable_directory_is_logged_and_the_run_continues_past_it(tmp_path, caplog):
    stuck = _make_session(tmp_path, _NOW - timedelta(days=30), turn_id="stuck")
    after = _make_session(tmp_path, _NOW - timedelta(days=20), turn_id="after")

    def failing_for_stuck(path: Path) -> None:
        if path == stuck:
            raise OSError("simulated: directory held open")
        shutil.rmtree(path)

    with caplog.at_level(logging.WARNING, logger="atlas.session.retention"):
        removed = sweep_expired_sessions(tmp_path, retain_days=7, clock=lambda: _NOW, remove=failing_for_stuck)

    assert stuck.exists()
    assert not after.exists()
    assert [r.path for r in removed] == [after]
    assert any("could not remove" in record.message for record in caplog.records)


def test_a_run_with_nothing_to_remove_still_logs_that_it_ran(tmp_path, caplog):
    _make_session(tmp_path, _NOW - timedelta(days=1))  # well within the window

    with caplog.at_level(logging.INFO, logger="atlas.session.retention"):
        removed = sweep_expired_sessions(tmp_path, retain_days=7, clock=lambda: _NOW)

    assert removed == []
    assert any("removed=0" in record.message for record in caplog.records), (
        "an empty run must say it ran and removed nothing -- silence here is "
        "indistinguishable from the sweep never having run at all"
    )


def test_a_directory_outside_the_configured_root_is_never_touched(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    outside_target = tmp_path / "not-a-session-root"
    outside_target.mkdir()
    marker = outside_target / "marker.txt"
    marker.write_text("still here", encoding="utf-8")

    # A symlink *inside* the configured root, named exactly like an expired
    # session, whose target resolves *outside* the root -- the guard this
    # proves is the resolved-path containment check, not merely "iterdir
    # only sees the root's own children" (T-02-34).
    escaping_link = root / _session_dir_name(_NOW - timedelta(days=365), turn_id="escape")
    os.symlink(outside_target, escaping_link, target_is_directory=True)

    removed = sweep_expired_sessions(root, retain_days=7, clock=lambda: _NOW)

    assert removed == []
    assert marker.exists()
    assert escaping_link.exists()


def test_age_is_read_from_the_directory_name_not_filesystem_mtime(tmp_path):
    # Named as expired, but its mtime is rewritten to "now" -- must still be
    # removed, because age comes from the name (T-02-35).
    named_expired = _make_session(tmp_path, _NOW - timedelta(days=30), turn_id="named-expired")
    os.utime(named_expired, (_NOW.timestamp(), _NOW.timestamp()))

    # Named as fresh, but its mtime is rewritten to look ancient -- must
    # still be kept, for the same reason.
    named_fresh = _make_session(tmp_path, _NOW - timedelta(hours=1), turn_id="named-fresh")
    ancient = (_NOW - timedelta(days=3650)).timestamp()
    os.utime(named_fresh, (ancient, ancient))

    removed = sweep_expired_sessions(tmp_path, retain_days=7, clock=lambda: _NOW)

    assert [r.path for r in removed] == [named_expired]
    assert not named_expired.exists()
    assert named_fresh.exists()


def test_the_clock_is_read_exactly_once_per_run(tmp_path):
    _make_session(tmp_path, _NOW - timedelta(days=30), turn_id="a")
    _make_session(tmp_path, _NOW - timedelta(days=20), turn_id="b")
    _make_session(tmp_path, _NOW - timedelta(days=1), turn_id="c")

    calls = []

    def counting_clock() -> datetime:
        calls.append(_NOW)
        return _NOW

    sweep_expired_sessions(tmp_path, retain_days=7, clock=counting_clock)

    assert len(calls) == 1, "one clock reading per run -- re-reading it per directory could keep and remove identically-aged sessions in the same run"


async def test_the_schedule_performs_more_than_one_sweep_over_time(tmp_path):
    scheduler = RetentionScheduler(
        tmp_path,
        retain_days=7,
        interval_s=0.01,
        clock=lambda: _NOW,
        sleep=_instant_sleep,
    )

    scheduler.start()
    try:
        await _wait_for(lambda: scheduler.sweep_count >= 3)
    finally:
        await scheduler.stop()

    assert scheduler.sweep_count >= 3


async def test_cancelling_the_schedule_stops_further_sweeps(tmp_path):
    scheduler = RetentionScheduler(
        tmp_path,
        retain_days=7,
        interval_s=0.01,
        clock=lambda: _NOW,
        sleep=_instant_sleep,
    )

    scheduler.start()
    await _wait_for(lambda: scheduler.sweep_count >= 2)
    await scheduler.stop()

    count_after_stop = scheduler.sweep_count
    await asyncio.sleep(0.05)

    assert scheduler.sweep_count == count_after_stop


async def test_a_sweep_that_raises_is_logged_and_the_schedule_continues(tmp_path, caplog, monkeypatch):
    import atlas.session.retention as retention_module

    call_count = {"n": 0}
    real_sweep = sweep_expired_sessions

    def flaky_sweep(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated: sweep blew up")
        return real_sweep(*args, **kwargs)

    monkeypatch.setattr(retention_module, "sweep_expired_sessions", flaky_sweep)

    scheduler = RetentionScheduler(
        tmp_path,
        retain_days=7,
        interval_s=0.01,
        clock=lambda: _NOW,
        sleep=_instant_sleep,
    )

    with caplog.at_level(logging.ERROR, logger="atlas.session.retention"):
        scheduler.start()
        try:
            await _wait_for(lambda: call_count["n"] >= 2)
        finally:
            await scheduler.stop()

    assert any("sweep raised" in record.message for record in caplog.records)
    assert call_count["n"] >= 2, "one failed run must not silently end the schedule for the life of the process"


# === Code review WR-06: the wake-event table's own retention ============


class _RecordingWakeEventRepo:
    """Counts deletions and remembers the cutoff it was asked for. Small on
    purpose -- what this proves is that the schedule reaches the table at
    all, and with the same window."""

    def __init__(self, *, rows: int = 3) -> None:
        self.cutoffs: list = []
        self._rows = rows

    async def delete_wake_events_before(self, cutoff) -> int:
        self.cutoffs.append(cutoff)
        removed, self._rows = self._rows, 0
        return removed


class _RaisingWakeEventRepo:
    async def delete_wake_events_before(self, cutoff) -> int:
        raise RuntimeError("the store is down")


async def test_the_schedule_sweeps_wake_events_on_the_same_window(tmp_path, caplog):
    """`wake_events` was the one piece of persistent state in this project
    with no owner and no bound -- every gate-blocked hit a television
    triggers, kept for the life of the deployment, while the recordings it
    describes expire on `retain_days` (WR-06)."""
    repo = _RecordingWakeEventRepo()
    scheduler = RetentionScheduler(
        tmp_path,
        retain_days=7,
        interval_s=0.01,
        clock=lambda: _NOW,
        sleep=_instant_sleep,
        wake_event_repo=repo,
    )

    with caplog.at_level(logging.INFO, logger="atlas.session.retention"):
        scheduler.start()
        try:
            await _wait_for(lambda: len(repo.cutoffs) >= 2)
        finally:
            await scheduler.stop()

    # The same window the directory sweep uses, measured from the same clock.
    assert repo.cutoffs[0] == _NOW - timedelta(days=7)
    # Logged whether or not it removed anything, the way the directory
    # sweep logs its own -- "ran and had nothing to do" must stay
    # distinguishable from "never ran".
    sweep_lines = [r for r in caplog.records if "wake-event retention sweep ran" in r.getMessage()]
    assert len(sweep_lines) >= 2
    assert "removed=3" in sweep_lines[0].getMessage()
    assert "removed=0" in sweep_lines[1].getMessage()


async def test_a_wake_event_sweep_that_raises_does_not_stop_the_directory_sweep(tmp_path, caplog):
    """The two sweeps are separately guarded on purpose: a store that is
    down must not stop the one that bounds actual audio (WR-06)."""
    expired = _make_session(tmp_path, _NOW - timedelta(days=30))
    scheduler = RetentionScheduler(
        tmp_path,
        retain_days=7,
        interval_s=0.01,
        clock=lambda: _NOW,
        sleep=_instant_sleep,
        wake_event_repo=_RaisingWakeEventRepo(),
    )

    with caplog.at_level(logging.ERROR, logger="atlas.session.retention"):
        scheduler.start()
        try:
            await _wait_for(lambda: not expired.exists())
        finally:
            await scheduler.stop()

    assert not expired.exists()
    assert any("wake-event retention sweep raised" in r.getMessage() for r in caplog.records)


async def test_no_wake_event_repository_means_no_wake_event_sweep(tmp_path):
    """A deployment with no repository at all behaves exactly as it did
    before -- the same absent-configuration discipline the rest of this
    codebase carries."""
    scheduler = RetentionScheduler(
        tmp_path, retain_days=7, interval_s=0.01, clock=lambda: _NOW, sleep=_instant_sleep
    )
    scheduler.start()
    try:
        await _wait_for(lambda: scheduler.sweep_count >= 2)
    finally:
        await scheduler.stop()
    assert scheduler.sweep_count >= 2
