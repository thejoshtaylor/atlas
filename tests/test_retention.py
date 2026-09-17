"""Wave-0 scaffold for session retention expiry (DBG-06).

Red on purpose -- `src/spire_voice/session/retention.py` does not exist
yet. Matches the shape plans 01-01 and 01.1-01 both used for their own
Wave-0 scaffolds.
"""

from __future__ import annotations


def test_retention_expiry_removes_old_sessions_and_logs_what_it_removed():
    """DBG-06: a session older than the configured retention window must be
    removed, and what was removed must be logged -- turned green by plan
    02-08.
    """
    raise AssertionError("plan 02-08 turns this green: session retention does not exist yet")
