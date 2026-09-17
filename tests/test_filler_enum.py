"""Proves the CMD-07 boundary: no holding line can assert an outcome.

Four assertions, matching plan 01.1-01's `<behavior>` block: an out-of-set
filler value is rejected rather than coerced, no member of the closed set
survives a word-boundary check against outcome-asserting words, the enum
and the spoken-text mapping cannot drift apart, and a confident reply must
carry a real answer.
"""

import re

import pytest
from pydantic import ValidationError

from spire_voice.providers.tier_reply import (
    FILLER_TEXT,
    FillerPhrase,
    OUTCOME_ASSERTING_WORDS,
    TierReply,
)

_WORD_RE = re.compile(r"[a-z']+")


def test_out_of_set_filler_value_raises_validation_error():
    with pytest.raises(ValidationError):
        TierReply(answer="", confident=False, needs_tool=False, filler="turning that off")


def test_no_filler_text_asserts_an_outcome():
    for member, text in FILLER_TEXT.items():
        words = set(_WORD_RE.findall(text.lower()))
        offending = words & OUTCOME_ASSERTING_WORDS
        assert not offending, f"{member!r} ({text!r}) contains outcome-asserting word(s): {offending}"


def test_filler_text_and_filler_phrase_do_not_drift():
    assert set(FILLER_TEXT) == set(FillerPhrase)


def test_confident_reply_must_carry_a_non_empty_answer():
    with pytest.raises(ValidationError):
        TierReply(answer="", confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)
