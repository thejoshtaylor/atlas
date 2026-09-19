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


# --- Task 3: turn-level cases, driving the real `run_turn` -----------------


def test_the_same_candidate_list_produces_the_same_clarifying_sentence_twice():
    """Composed in code, never through a second model round (D-14): the
    same candidates, in the same order, always produce the same sentence."""
    from spire_voice.turn.controller import _compose_clarifying_question

    candidates = ("light.example_lamp", "light.example_desk_lamp")
    first = _compose_clarifying_question(candidates, {})
    second = _compose_clarifying_question(candidates, {})

    assert first == second
    assert "light.example_lamp" in first
    assert "light.example_desk_lamp" in first


async def test_a_needs_clarification_winner_speaks_a_question_and_makes_no_tool_call(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
    """CMD-09/D-07, end to end: a triage tier's `needs_clarification` reply
    ends the race before the (deliberately slow) top tier ever reaches the
    tool host. The operator hears a question naming every candidate, the
    tool host recorded zero calls, and `turn_outcome` names this case.
    """
    import asyncio
    from types import SimpleNamespace

    from spire_voice.providers.base import FinalTranscript
    from spire_voice.timing import TurnTimings
    from spire_voice.turn import brain_race
    from spire_voice.turn.controller import run_turn

    class _RecordingToolHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])

    class _NeverFinishesBrain:
        """The top tier's tool round never returns before the race is
        already won -- proof the clarifying reply pre-empts it rather than
        racing it to a tool call."""

        async def chat(self, messages, tools=None):
            await asyncio.sleep(10)
            raise AssertionError("should have been cancelled before this line")  # pragma: no cover

    clarifying_reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=("light.example_lamp", "light.example_desk_lamp"),
    )
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=clarifying_reply, delay_s=0.0),
        calls_tools=False,
    )
    top_brain = _NeverFinishesBrain()
    top_tier = brain_race.TierBrain(
        index=1,
        model="top-model",
        brain=top_brain,
        envelope_client=fake_envelope_client(reply=None, delay_s=0.0),
        calls_tools=True,
    )

    tool_host = _RecordingToolHost()
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn on the lamp")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        top_brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier, top_tier],
        filler_after_ms=1000,
        filler_cache=None,
    )

    assert tool_host.calls == []
    assert len(tts.received_text) == 1
    spoken = tts.received_text[0]
    assert "light.example_lamp" in spoken
    assert "light.example_desk_lamp" in spoken
    assert timings.turn_outcome == "needs_clarification"


async def test_a_needs_clarification_question_prefers_friendly_names_from_injected_state(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
    """When this turn's own injected live-state fetch carries a friendly
    name for a candidate, the spoken question uses it and keeps the raw
    entity id out of the sentence entirely -- an id is not what a person
    says out loud."""
    import asyncio

    from spire_voice.providers.base import FinalTranscript
    from spire_voice.timing import TurnTimings
    from spire_voice.turn import brain_race
    from spire_voice.turn.controller import run_turn

    async def _state_fetch():
        return [
            {"entity_id": "light.example_lamp", "friendly_name": "the lamp", "state": "off"},
            {"entity_id": "light.example_desk_lamp", "friendly_name": "the desk lamp", "state": "off"},
        ]

    clarifying_reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=("light.example_lamp", "light.example_desk_lamp"),
    )
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=clarifying_reply, delay_s=0.0),
        calls_tools=False,
    )

    class _NeverFinishesBrain:
        async def chat(self, messages, tools=None):
            await asyncio.sleep(10)
            raise AssertionError("should have been cancelled before this line")  # pragma: no cover

    top_brain = _NeverFinishesBrain()
    top_tier = brain_race.TierBrain(
        index=1,
        model="top-model",
        brain=top_brain,
        envelope_client=fake_envelope_client(reply=None, delay_s=0.0),
        calls_tools=True,
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn on the lamp")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        top_brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier, top_tier],
        filler_after_ms=1000,
        filler_cache=None,
        state_fetch=_state_fetch,
    )

    spoken = tts.received_text[0]
    assert "the lamp" in spoken
    assert "the desk lamp" in spoken
    assert "light.example_lamp" not in spoken
    assert "light.example_desk_lamp" not in spoken


# --- Plan 05-05 Task 3: a pending run is a new candidate type, not a new ---
# --- mechanism (D-10) -- same envelope, same validators, unchanged.     ---


def test_a_reply_asking_which_run_with_two_run_summaries_validates():
    reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=("turn off the porch light", "start the coffee maker"),
    )
    assert reply.needs_clarification is True
    assert reply.candidates == ("turn off the porch light", "start the coffee maker")


def test_needs_clarification_with_exactly_one_run_summary_raises():
    """The same at-least-two rule applies unchanged, whether the
    candidates are entity ids or run summaries (D-10): a single candidate
    is a guess wearing a question's clothes either way."""
    with pytest.raises(ValidationError):
        TierReply(
            answer="",
            confident=False,
            needs_tool=False,
            filler=FillerPhrase.LET_ME_CHECK,
            needs_clarification=True,
            candidates=("turn off the porch light",),
        )


def test_a_run_disambiguation_reply_cannot_also_claim_needs_tool():
    """The assistant cannot ask which run was meant and cancel or extend
    one in the same breath (D-10): the same exclusivity validator that
    already forbids asking-and-acting for an entity forbids it for a run,
    unchanged."""
    with pytest.raises(ValidationError):
        TierReply(
            answer="",
            confident=False,
            needs_tool=True,
            filler=FillerPhrase.LET_ME_CHECK,
            needs_clarification=True,
            candidates=("turn off the porch light", "start the coffee maker"),
        )


def test_compose_clarifying_question_over_two_run_summaries_names_both_in_order():
    """`_compose_clarifying_question` already speaks a candidate verbatim
    when no friendly name is known for it -- a run summary has no entry in
    `friendly_names` (that mapping is built from entity state, never from
    workflow runs), so this is the existing fallback path, not new
    behavior."""
    from spire_voice.turn.controller import _compose_clarifying_question

    candidates = ("turn off the porch light", "start the coffee maker")
    question = _compose_clarifying_question(candidates, {})

    assert "turn off the porch light" in question
    assert "start the coffee maker" in question
    assert question.index("turn off the porch light") < question.index("start the coffee maker")


# --- Plan 06-05 Task 1: a plugin two plugins both publish a capability -----
# --- for is a third candidate type (D-11) -- same envelope, same          -
# --- validators, unchanged.                                                -


def test_a_reply_asking_which_plugin_with_two_plugin_names_validates():
    reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=("Weather", "Garden Sensors"),
    )
    assert reply.needs_clarification is True
    assert reply.candidates == ("Weather", "Garden Sensors")


def test_needs_clarification_with_exactly_one_plugin_name_raises():
    """The same at-least-two rule applies unchanged, whether the
    candidates are entity ids, run summaries, or plugin display names
    (D-11): a single candidate is a guess wearing a question's clothes
    either way."""
    with pytest.raises(ValidationError):
        TierReply(
            answer="",
            confident=False,
            needs_tool=False,
            filler=FillerPhrase.LET_ME_CHECK,
            needs_clarification=True,
            candidates=("Weather",),
        )


def test_a_plugin_disambiguation_reply_cannot_also_claim_needs_tool():
    """The assistant cannot ask which plugin was meant and call a tool in
    the same breath (D-11): the same exclusivity validator that already
    forbids asking-and-acting for an entity or a run forbids it for a
    plugin, unchanged."""
    with pytest.raises(ValidationError):
        TierReply(
            answer="",
            confident=False,
            needs_tool=True,
            filler=FillerPhrase.LET_ME_CHECK,
            needs_clarification=True,
            candidates=("Weather", "Garden Sensors"),
        )


def test_compose_clarifying_question_over_two_plugin_names_names_both_in_order():
    """`_compose_clarifying_question` already speaks a candidate verbatim
    when no friendly name is known for it -- a plugin's display name has
    no entry in `friendly_names` (that mapping is built from entity
    state, never from installed plugins), so this is the existing
    fallback path, not new behavior."""
    from spire_voice.turn.controller import _compose_clarifying_question

    candidates = ("Weather", "Garden Sensors")
    question = _compose_clarifying_question(candidates, {})

    assert "Weather" in question
    assert "Garden Sensors" in question
    assert question.index("Weather") < question.index("Garden Sensors")
