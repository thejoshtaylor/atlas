"""The wake-only predicate `turn/controller.py` checks before deciding
whether a final transcript needs a second drain for the command that
follows (260922-woc).
"""

import pytest

from atlas.turn.wake_echo import (
    WakeHold,
    completes_wake_command,
    is_wake_only,
    is_wake_without_command,
    strip_wake_phrase,
)

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


# --- 260929-icf: strip_wake_phrase verifies and removes a leading phrase ---

_STRIP_SAMPLES = [
    ("Hey Atlas.", ""),
    ("hey atlas", ""),
    ("Hey, Atlas!", ""),
    ("Hey, Atlas, turn on the lights.", "turn on the lights."),
    ("Atlas.", ""),
    ("Atlas, what time is it?", "what time is it?"),
    ("At last.", ""),
    ("At last, turn off the fan.", "turn off the fan."),
    ("At less.", ""),
    ("A atlas.", ""),
    ("Hey at last, stop.", "stop."),
    ("Hey Atlas. Turn off the fan.", "Turn off the fan."),
    ("the atlas", ""),
    ("heyatlas", ""),
    ("hey, atlas turn on", "turn on"),
    ("at last turn on", "turn on"),
    ("Atlas play music", "play music"),
    ("Okay Atlas, lights off.", "lights off."),
    ("Oh atlas.", ""),
    ("Hi Atlas, what time is it?", "what time is it?"),
    ("Hay Atlas, stop.", "stop."),
    # Accepted miss: "at"+"least" is a split keyword (0.83).
    ("At least, I think so.", "I think so."),
]

_NOT_A_WAKE_SAMPLES = [
    "",
    "Yeah",
    "Yeah.",
    "Huh?",
    "Hey.",
    "Okay.",
    "Hey Alice.",
    "Hey, what's up?",
    "She left the car in the lot.",
    "Terms and conditions apply.",
    "Puppies usually sleep in a crate.",
    "Turn on the lights.",
    "stop",
    "You always say AM in the morning.",
    "So atlas is a book.",
    "Yeah atlas.",
    "You atlas.",
]


@pytest.mark.parametrize(("text", "expected"), _STRIP_SAMPLES)
def test_strip_wake_phrase_returns_the_remainder(text: str, expected: str) -> None:
    assert strip_wake_phrase(text, _PHRASE) == expected


@pytest.mark.parametrize("text", _NOT_A_WAKE_SAMPLES)
def test_strip_wake_phrase_returns_none_when_text_does_not_open_with_it(text: str) -> None:
    assert strip_wake_phrase(text, _PHRASE) is None


def test_strip_wake_phrase_with_a_blank_phrase_returns_none() -> None:
    assert strip_wake_phrase("hey atlas", "") is None


def test_strip_wake_phrase_works_for_a_one_word_phrase() -> None:
    assert strip_wake_phrase("Computer, lights on.", "computer") == "lights on."


# --- 261001-glr: the helpers that keep one stream open ---


@pytest.mark.parametrize(
    ("text", "verify", "expected"),
    [
        ("Hey Atlas.", True, True),
        ("Hey Atlas, turn on the lights.", True, False),
        ("She left the car in the lot.", True, False),
        ("at less", False, True),
        ("", False, False),
    ],
)
def test_is_wake_without_command(text: str, verify: bool, expected: bool) -> None:
    assert is_wake_without_command(text, _PHRASE, verify=verify) is expected


def test_wake_hold_holds_a_wake_only_final_and_the_empty_final_after_it() -> None:
    hold = WakeHold(_PHRASE, verify=True)

    assert hold("") is False
    assert hold("Hey Atlas.") is True
    assert hold("") is True
    assert hold("turn on the lights") is False
    assert hold.held == ["Hey Atlas.", ""]


def test_wake_hold_heard_text_puts_the_held_phrase_back() -> None:
    hold = WakeHold(_PHRASE, verify=True)
    assert hold.heard_text("turn on the lights") == "turn on the lights"

    hold("Hey Atlas.")

    assert hold.heard_text("turn on the lights") == "Hey Atlas. turn on the lights"
    assert hold.heard_text("Hey Atlas, turn on") == "Hey Atlas, turn on"


# Debug wake-command-needs-beep-wait: the early transcription of a one-breath
# wake segment ends the turn only on the phrase and a finished command. The
# false samples are real Parakeet decodes of recorded wake segments ("Yeah.",
# "Partless", "Hey Alice.") and the shapes the drain must keep waiting for.
@pytest.mark.parametrize(
    ("text", "verify", "expected"),
    [
        ("Hey Atlas, turn on the T V.", True, True),
        ("Hey Atlas, what time is it?", True, True),
        ("The Atlas, turn it off.", True, True),
        ("Hey Atlas.", True, False),
        ("Hey Atlas", True, False),
        ("Hey Atlas, at last.", True, False),
        ("Hey Atlas, turn off.", True, False),
        ("Hey Atlas, turn on the", True, False),
        ("Hey yeah, let's turn off the fan.", True, False),
        ("Yeah.", True, False),
        ("Partless", True, False),
        ("Hey Alice.", True, False),
        ("", True, False),
        ("It is very nice in the spring.", True, False),
        ("It is very nice in the spring.", False, True),
        ("Turn off.", False, False),
    ],
)
def test_completes_wake_command(text: str, verify: bool, expected: bool) -> None:
    assert completes_wake_command(text, _PHRASE, verify=verify) is expected
