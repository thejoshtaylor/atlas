"""Wave-0 scaffold for the long-lived speaker FIFO writer (VOICE-05).

Red on purpose -- `src/spire_voice/speaker/fifo_writer.py` does not exist
yet. Matches the shape plans 01-01 and 01.1-01 both used for their own
Wave-0 scaffolds.
"""

from __future__ import annotations


def test_speaker_fifo_reuses_one_subprocess_across_two_utterances():
    """VOICE-05: a fresh `ffmpeg` process per utterance costs 300-800 ms
    before the first sample reaches the speaker -- the FIFO writer must
    hold one long-lived subprocess across two separate utterances, not
    spawn a second one -- turned green by plan 02-03.
    """
    raise AssertionError("plan 02-03 turns this green: the FIFO writer does not exist yet")
