"""Wave-0 scaffold for offline wake-engine scoring (DBG-04).

Red on purpose -- `scripts/score_wake_engines.py` does not exist yet, and
no recorded corpus exists to score against until the capture-mode script
has been run against a real camera. Matches the shape plans 01-01 and
01.1-01 both used for their own Wave-0 scaffolds.
"""

from __future__ import annotations


def test_offline_scoring_runs_both_engines_against_the_same_fixture():
    """DBG-04: openWakeWord and Vosk must both be scored offline, against
    the same recorded audio -- never live, which would give each engine a
    different acoustic moment and prove nothing about which is better --
    turned green by plan 02-09.
    """
    raise AssertionError("plan 02-09 turns this green: the scoring harness does not exist yet")
