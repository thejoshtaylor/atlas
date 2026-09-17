"""Stubs for the turn-controller validation map.

Turned green by plan 01-02 (happy path) and plan 01-05 (the closing-turn and
tool-round-cap edges). Every body raises until then, so the suite collects
cleanly and stays red until its owning task lands.
"""


def test_full_turn_happy_path():
    raise AssertionError("not implemented: VOICE-01")


def test_empty_transcript_closes_turn():
    raise AssertionError("not implemented: VOICE-08")


def test_silence_timeout_closes_turn():
    raise AssertionError("not implemented: VOICE-08")


def test_next_turn_runs_after_a_closed_turn():
    raise AssertionError("not implemented: VOICE-08")


def test_tool_round_cap_is_enforced():
    raise AssertionError("not implemented: VOICE-01")
