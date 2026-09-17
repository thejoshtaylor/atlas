"""Wave-0 scaffolds for the per-turn session recorder (DBG-01, DBG-02).

Red on purpose -- `src/spire_voice/session/recorder.py` does not exist yet.
Matches the shape plans 01-01 and 01.1-01 both used for their own Wave-0
scaffolds.
"""

from __future__ import annotations


def test_turn_produces_a_session_folder_with_audio_events_and_a_timeline():
    """DBG-01: one turn must produce one session folder holding the raw
    captured audio, a JSONL event stream, and a rendered merged timeline
    derived from that stream -- turned green by plan 02-05.
    """
    raise AssertionError("plan 02-05 turns this green: the session recorder does not exist yet")


def test_turn_timings_is_embedded_in_the_session_folder():
    """DBG-02: the already-built `TurnTimings` record must be embedded in
    the session folder verbatim, not measured a second way -- turned green
    by plan 02-05.
    """
    raise AssertionError("plan 02-05 turns this green: the session recorder does not exist yet")
