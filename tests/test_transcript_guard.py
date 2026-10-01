"""Table tests for `turn/controller.py`'s pre-brain transcript guard
(260923-kao): a wake-only or filler-only final transcript must never reach
a macro, the local on/off matcher, or the tier race.

Also tests `asks_for_information` (260924-4it): the pure predicate the
first-round done shortcut in `_run_tool_rounds` checks before speaking a
canned "done" reply instead of taking a second model round.
"""

import pytest

from atlas.turn.transcript_guard import (
    asks_for_information,
    is_no_command,
    looks_unfinished,
    missing_target_question,
)

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
# The alarm commands are real transcripts the guard once dropped, because
# "alarm" and "alarms" sound like "atlas".
_COMMANDS = (
    "turn off the atlas lamp",
    "what's the weather in paris",
    "turn on the lights",
    "stop",
    "Cancel all alarms.",
    "cancel my alarm.",
    "Off my alarm.",
    "set alarm",
    "cancel alarms",
)


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


# 261001-ibf: a control command that names no device.
@pytest.mark.parametrize(
    ("text", "question"),
    [
        ("Turn off.", "turn off what?"),
        ("Turn on the.", "turn on what?"),
        ("shut off", "shut off what?"),
        ("switch on", "switch on what?"),
        ("turn of the", "turn off what?"),
        ("please turn off", "turn off what?"),
    ],
)
@pytest.mark.parametrize("has_referent", (False, True))
def test_a_bare_verb_gets_the_question(text: str, question: str, has_referent: bool) -> None:
    assert missing_target_question(text, has_referent=has_referent) == question


@pytest.mark.parametrize("text", ["turn it on", "turn it back on", "Turn that off."])
def test_a_pronoun_with_no_referent_gets_the_question(text: str) -> None:
    assert missing_target_question(text, has_referent=False) is not None
    assert missing_target_question(text, has_referent=True) is None


def test_the_pronoun_question_names_the_verb_and_particle() -> None:
    assert missing_target_question("turn it on", has_referent=False) == "turn on what?"


@pytest.mark.parametrize(
    "text",
    [
        "lights off",
        "stop",
        "pause",
        "turn off the swamp cooler",
        "turn on the lights in the",
        "turn up the volume",
        "whose turn is it",
        "is the power on",
        "turn everything off",
        "",
    ],
)
@pytest.mark.parametrize("has_referent", (False, True))
def test_a_command_with_a_target_or_no_verb_gets_no_question(text: str, has_referent: bool) -> None:
    assert missing_target_question(text, has_referent=has_referent) is None


@pytest.mark.parametrize(
    "text",
    [
        "Turn off.",
        "turn on",
        "switch off",
        "Turn on the.",
        "set it to",
        "turn off the lamp and",
        "turn on the\u2026",
        "turn on the...",
    ],
)
def test_looks_unfinished_is_true_for_a_command_that_stops_short(text: str) -> None:
    assert looks_unfinished(text) is True


@pytest.mark.parametrize(
    "text",
    ["turn off the swamp cooler.", "turn it off", "lights off", "stop", "", "Hey Atlas.", "what is the weather"],
)
def test_looks_unfinished_is_false_for_a_finished_text(text: str) -> None:
    assert looks_unfinished(text) is False
