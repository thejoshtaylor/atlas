"""The panel text sanitizer, the closed card union and the text card builder
(Phase 15, D-11, CARD-08, PANEL-03).

`hostile_text.json` is shared: `swift test` reads the same vectors for
`TextSanitizer.sanitize`, so the Mac and the server give the same output.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.desktop import cards, display_text, protocol

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "desktop" / "protocol" / "v1"
_VECTORS = json.loads((PROTOCOL_DIR / "hostile_text.json").read_text())["vectors"]
assert len(_VECTORS) >= 12

_TURN = "5f1c2a9e7b3d4c60a8e2f7b1c9d04e3a"


@pytest.mark.parametrize("vector", _VECTORS, ids=lambda v: v["name"])
def test_sanitizer_gives_the_shared_vector_output(vector):
    result = display_text.sanitize_display_text(
        vector["input"], vector["max"], keep=vector["keep"]
    )
    assert result == vector["expected"]
    assert len(result) <= vector["max"]


def test_sanitizer_result_is_never_longer_than_the_cap():
    assert len(display_text.sanitize_display_text("z" * 5000, 600)) == 600
    assert len(display_text.sanitize_display_text("z " * 2000, 7, keep="end")) <= 7


def test_sanitizer_rejects_a_cap_too_small_to_cut():
    with pytest.raises(AssertionError):
        display_text.sanitize_display_text("abc", 1)


def test_build_text_card_takes_only_the_turn_the_card_and_the_text():
    assert list(inspect.signature(cards.build_text_card).parameters) == [
        "turn_id",
        "card_id",
        "text",
    ]


@pytest.mark.parametrize("text", ["", "   ", "\u0000‮\u007f", "\n\t "])
def test_build_text_card_returns_none_for_text_empty_after_sanitizing(text):
    assert cards.build_text_card(_TURN, f"{_TURN}-text", text) is None


def test_build_text_card_sanitizes_and_caps_at_600():
    built = cards.build_text_card(_TURN, f"{_TURN}-text", "pay‮pal " + "x" * 2000)
    assert built is not None
    card = json.loads(built)
    assert len(card["data"]["text"]) == display_text.CARD_TEXT_MAX
    assert card["data"]["text"].startswith("paypal x")
    assert card["data"]["text"].endswith("…")
    assert card["fallback_text"] == card["data"]["text"]
    assert set(card) == {"type", "turn_id", "card_id", "kind", "fallback_text", "data"}


def test_markdown_link_syntax_stays_literal_in_a_card():
    built = cards.build_text_card(_TURN, f"{_TURN}-text", "[text](https://example.com)")
    assert built is not None
    assert json.loads(built)["data"]["text"] == "[text](https://example.com)"


def test_build_timer_card_names_the_timer_in_the_fallback():
    card = json.loads(cards.build_timer_card(_TURN, f"{_TURN}-timer", 12, "timer", "pa‮sta"))
    assert card["fallback_text"] == "Timer: pasta"
    assert card["data"] == {"timer_id": 12, "timer_kind": "timer", "label": "pasta"}
    bare = json.loads(cards.build_timer_card(_TURN, f"{_TURN}-timer", 3, "alarm", ""))
    assert bare["fallback_text"] == "Alarm"


def _text_card() -> dict:
    return {
        "type": "card",
        "turn_id": _TURN,
        "card_id": f"{_TURN}-text",
        "kind": "text",
        "fallback_text": "Hi.",
        "data": {"text": "Hi."},
    }


def test_a_valid_card_validates_through_the_server_union():
    parsed = protocol.SERVER_MESSAGE_ADAPTER.validate_python(_text_card())
    assert isinstance(parsed, cards.TextCard)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(extra="x"),
        lambda c: c["data"].update(markup="<b>"),
        lambda c: c.update(kind="weather"),
        lambda c: c.pop("card_id"),
        lambda c: c.update(card_id=""),
        lambda c: c["data"].update(text="x" * 601),
    ],
    ids=["extra_key", "extra_data_key", "unknown_kind", "no_card_id", "empty_card_id", "text_601"],
)
def test_a_malformed_card_fails_server_validation(mutate):
    card = _text_card()
    mutate(card)
    with pytest.raises(ValidationError):
        protocol.SERVER_MESSAGE_ADAPTER.validate_python(card)


def test_a_timer_card_validates_and_picks_the_timer_model():
    card = {
        "type": "card",
        "turn_id": _TURN,
        "card_id": f"{_TURN}-timer",
        "kind": "timer",
        "fallback_text": "Timer: pasta",
        "data": {"timer_id": 12, "timer_kind": "timer", "label": "pasta"},
    }
    assert isinstance(protocol.SERVER_MESSAGE_ADAPTER.validate_python(card), cards.TimerCard)


def test_turn_ended_clamps_its_integers_into_range():
    built = json.loads(protocol.build_turn_ended(_TURN, "completed", -5, 9_999_999))
    assert built["follow_up_window_ms"] == 0
    assert built["playback_ms_left"] == display_text.MS_MAX


def test_a_transcript_keeps_the_end_when_it_is_over_the_cap():
    text = "a" * 1500 + " tail"
    built = json.loads(protocol.build_transcript_partial(_TURN, text))
    assert len(built["text"]) == display_text.TRANSCRIPT_TEXT_MAX
    assert built["text"].startswith("…")
    assert built["text"].endswith("tail")
