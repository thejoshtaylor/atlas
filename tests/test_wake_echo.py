"""The wake-only predicate `turn/controller.py` checks before deciding
whether a final transcript needs a second drain for the command that
follows (260922-woc).
"""

import pytest

from atlas.turn.wake_echo import is_wake_only

_PHRASE = "hey atlas"

# Samples re-picked for "hey atlas" from the 260922-woc-PLAN.md shape -- the
# first four are wake-only (an echo of the phrase or its tail), the last
# three are real commands that must never be misread as one.
_WAKE_ONLY_SAMPLES = ["at last", "at less", "hey, atlas", "a atlas"]
_COMMAND_SAMPLES = ["stop", "turn on the lights", "at last the lights"]


@pytest.mark.parametrize("text", _WAKE_ONLY_SAMPLES)
def test_wake_only_samples_are_recognized(text: str) -> None:
    assert is_wake_only(text, _PHRASE) is True


@pytest.mark.parametrize("text", _COMMAND_SAMPLES)
def test_command_samples_are_not_wake_only(text: str) -> None:
    assert is_wake_only(text, _PHRASE) is False


def test_empty_text_is_not_wake_only() -> None:
    """The existing empty-transcript path in `run_turn` already handles
    this case; this function has nothing to add for it."""
    assert is_wake_only("", _PHRASE) is False


def test_wake_phrase_followed_by_a_command_is_not_wake_only() -> None:
    """The word-count guard alone rejects this: five words is more than
    the two-word phrase's `len(phrase.split()) + 1` allowance."""
    assert is_wake_only("hey atlas turn on the lights", _PHRASE) is False
