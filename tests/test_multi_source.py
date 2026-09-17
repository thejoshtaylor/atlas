"""Wave-0 scaffold for concurrent multi-source operation (SRC-03).

Red on purpose. Matches the shape plans 01-01 and 01.1-01 both used for
their own Wave-0 scaffolds.
"""

from __future__ import annotations


def test_two_sources_run_concurrently_with_independent_wake_and_refractory_state():
    """SRC-03: two audio sources (e.g. the camera and the browser
    microphone) must run at once, each with its own wake and refractory
    state -- a hit or a refractory window on one must never affect the
    other -- turned green by plan 02-04.
    """
    raise AssertionError("plan 02-04 turns this green: multi-source concurrency does not exist yet")
