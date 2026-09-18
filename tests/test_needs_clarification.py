"""Proves CMD-09's boundary: asking which entity was meant is a third,
exclusive outcome on `TierReply`, never combinable with an answer or a tool
call, and never triggered by a single candidate wearing a question's
clothes.

Task 1's cases (envelope validation) match this plan's own `<behavior>`
block. Task 3 adds turn-level cases below, driving the real `run_turn`.
"""

import pytest
from pydantic import ValidationError

from spire_voice.providers.tier_reply import FillerPhrase, TierReply


def test_a_reply_asking_which_entity_with_two_candidates_validates():
    reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=("light.example_lamp", "light.example_desk_lamp"),
    )
    assert reply.needs_clarification is True
    assert reply.candidates == ("light.example_lamp", "light.example_desk_lamp")


def test_needs_clarification_and_confident_cannot_both_be_true():
    with pytest.raises(ValidationError):
        TierReply(
            answer="turned it off",
            confident=True,
            needs_tool=False,
            filler=FillerPhrase.LET_ME_CHECK,
            needs_clarification=True,
            candidates=("light.example_lamp", "light.example_desk_lamp"),
        )


def test_needs_clarification_and_needs_tool_cannot_both_be_true():
    with pytest.raises(ValidationError):
        TierReply(
            answer="",
            confident=False,
            needs_tool=True,
            filler=FillerPhrase.LET_ME_CHECK,
            needs_clarification=True,
            candidates=("light.example_lamp", "light.example_desk_lamp"),
        )


def test_needs_clarification_with_no_candidates_raises():
    with pytest.raises(ValidationError):
        TierReply(
            answer="",
            confident=False,
            needs_tool=False,
            filler=FillerPhrase.LET_ME_CHECK,
            needs_clarification=True,
            candidates=(),
        )


def test_needs_clarification_with_exactly_one_candidate_raises():
    """One candidate is not ambiguity -- it is a guess wearing a question's
    clothes, and this mechanism exists specifically to make that
    unrepresentable."""
    with pytest.raises(ValidationError):
        TierReply(
            answer="",
            confident=False,
            needs_tool=False,
            filler=FillerPhrase.LET_ME_CHECK,
            needs_clarification=True,
            candidates=("light.example_lamp",),
        )


def test_an_ordinary_confident_reply_is_unchanged_by_the_new_fields():
    reply = TierReply(answer="sure", confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)
    assert reply.needs_clarification is False
    assert reply.candidates == ()


def test_an_ordinary_needs_tool_reply_is_unchanged_by_the_new_fields():
    reply = TierReply(answer="", confident=False, needs_tool=True, filler=FillerPhrase.LET_ME_CHECK)
    assert reply.needs_clarification is False
    assert reply.candidates == ()
