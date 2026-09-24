"""Table tests for `turn/controller.py`'s pre-brain transcript guard
(260923-kao): a wake-only or filler-only final transcript must never reach
a macro, the local on/off matcher, or the tier race.
"""

import pytest

from atlas.turn.transcript_guard import is_no_command

# Wake-only against the default keyword ("atlas", used when no wake_phrase
# is configured) -- an echo of the wake phrase, or a near-miss of it.
_WAKE_ONLY = ("Space Atlas", "Stay Atlas", "Hey, Atlas.", "atlas")

# Filler-only, or empty -- nothing is left once stopwords are stripped.
# "It’s" uses the curly apostrophe (U+2019), the shape STT actually
# produces; "hay" is the wake-prefix mishearing, not deliberate filler.
_FILLER_ONLY = ("It's", "It’s", "uh, um", "Okay.", "hay", "", "   ")

# Real commands, three words or fewer or longer -- must never be dropped,
# with no wake_phrase configured and with one configured.
_COMMANDS = ("turn off the atlas lamp", "what's the weather in paris", "turn on the lights", "stop")


@pytest.mark.parametrize("text", _WAKE_ONLY)
def test_wake_only_default_keyword_is_no_command(text: str) -> None:
    assert is_no_command(text) is True


@pytest.mark.parametrize("text", ("Space Atlas", "Hey Atlas, uh"))
def test_wake_only_with_configured_wake_phrase_is_no_command(text: str) -> None:
    assert is_no_command(text, "hey atlas") is True


@pytest.mark.parametrize("text", _FILLER_ONLY)
def test_filler_only_or_empty_is_no_command(text: str) -> None:
    assert is_no_command(text) is True


@pytest.mark.parametrize("text", _COMMANDS)
@pytest.mark.parametrize("wake_phrase", (None, "hey atlas"))
def test_commands_are_not_no_command(text: str, wake_phrase: str | None) -> None:
    assert is_no_command(text, wake_phrase) is False


def test_keyword_comes_from_the_configured_wake_phrase() -> None:
    """The last word of `wake_phrase`, not the default "atlas", is the
    keyword this rule matches against."""
    assert is_no_command("computer", "okay computer") is True
    assert is_no_command("atlas", "okay computer") is False


@pytest.mark.parametrize("wake_phrase", ("", None))
def test_blank_or_missing_wake_phrase_falls_back_to_the_default_keyword(wake_phrase: str | None) -> None:
    assert is_no_command("atlas", wake_phrase) is True
