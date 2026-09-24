"""Table tests for `turn/controller.py`'s pre-brain transcript guard
(260923-kao): a wake-only or filler-only final transcript must never reach
a macro, the local on/off matcher, or the tier race.

Also tests `asks_for_information` (260924-4it): the pure predicate the
first-round done shortcut in `_run_tool_rounds` checks before speaking a
canned "done" reply instead of taking a second model round.
"""

import pytest

from atlas.turn.transcript_guard import asks_for_information, is_no_command

# 260924-4it: True cases -- the operator may be waiting for spoken
# information, so the done shortcut must not fire.
_ASKS = (
    "What's the temperature and turn on the fan",
    "turn on the fan and tell me the temperature",
    "Is the door locked?",
    "hey atlas is the door locked",
    "turn on the fan and is the window open",
    "turn on the fan?",
    "how warm is it in here",
    "which lights are on",
    "turn off the lamp. what time is it",
    "check if the door is locked and turn on the lamp",
    "read me the forecast then turn off the lamp",
)

# False cases -- plain commands, nothing to answer aloud.
_DOES_NOT_ASK = (
    "turn on the fan",
    "Turn off the lamp.",
    "could you turn off the lamp",
    "can you switch on the fan please",
    "hey atlas turn off the lamp",
    "set the lamp to fifty percent",
    "turn on the light that is by the door",
    "turn off the fan and the lamp",
    "",
)


@pytest.mark.parametrize("text", _ASKS)
def test_asks_for_information_is_true(text: str) -> None:
    assert asks_for_information(text) is True


@pytest.mark.parametrize("text", _DOES_NOT_ASK)
def test_asks_for_information_is_false(text: str) -> None:
    assert asks_for_information(text) is False

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
