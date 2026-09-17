"""Wave-0 scaffold for the pre-roll ring buffer (VOICE-06).

Red on purpose -- the pre-roll buffer does not exist yet. Matches the shape
plans 01-01 and 01.1-01 both used for their own Wave-0 scaffolds.
"""

from __future__ import annotations


def test_preroll_buffer_replays_into_the_transcript_stream():
    """VOICE-06: a speaker usually runs straight from "hey spire" into the
    command with no pause, so the buffered pre-roll audio must replay into
    the transcript stream ahead of live frames, recovering the command's
    first syllables -- turned green by plan 02-04.
    """
    raise AssertionError("plan 02-04 turns this green: the pre-roll buffer does not exist yet")
