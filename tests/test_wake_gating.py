"""Wave-0 scaffold for wake-hit gating (VOICE-03).

Red on purpose -- `src/spire_voice/wake/gate.py` does not exist yet.
Matches the shape plans 01-01 and 01.1-01 both used for their own Wave-0
scaffolds.
"""

from __future__ import annotations


def test_wake_hit_spawns_a_turn_no_audio_leaves_before_it_fires():
    """VOICE-03: a wake-word hit that clears the gate must spawn a turn, and
    no audio may leave this host before that hit fires -- turned green by
    plan 02-04.
    """
    raise AssertionError("plan 02-04 turns this green: wake gating does not exist yet")
