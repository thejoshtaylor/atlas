"""Turn-level proofs for CMD-03, CMD-04, and CMD-05 (04-CONTEXT.md D-01
through D-04).

Each test here asserts on where a call went or whether one happened at
all, never on how the spoken sentence read -- reading only the answer
would pass a turn that got the words right after a pointless tool round,
which is exactly the failure CMD-03/CMD-04 forbid (D-04).

Task 1 (CMD-03): a time or date question is answered from the per-turn
state message with zero calls to the tool host at all -- the production
turn always starts a state fetch before the transcript is drained (see the
test's own docstring below for the caveat this repository has already had
to correct twice), but that fetch is not a call made *in service of*
answering the time; a `get_time` tool would be.
"""

from __future__ import annotations

from typing import Any

from spire_voice.providers.base import FinalTranscript
from spire_voice.providers.tier_reply import FillerPhrase, TierReply
from spire_voice.timing import TurnTimings
from spire_voice.turn import brain_race
from spire_voice.turn.controller import run_turn


class _RecordingToolHost:
    """Records every `(name, arguments)` pair `call_tool` receives, in
    order, and returns a scripted result keyed by tool name.

    Unlike the `_RaisingToolHost` fixture `test_turn_controller.py` uses
    for its own zero-tool-call proof, this host does not raise on a call
    -- it records one, so a test can assert the recorded list is empty
    (or, for the weather-routing proof in a later task, assert exactly
    which host received the call), rather than relying on an uncalled
    `AssertionError` to prove the same thing indirectly.
    """

    def __init__(self, results_by_tool: dict[str, Any] | None = None) -> None:
        self._results = dict(results_by_tool or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        return self._results.get(name)


async def test_a_time_question_is_answered_with_zero_tool_calls(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
    """CMD-03: a confident triage tier answers a spoken time question from
    the state message `run_turn` already injects, with zero calls
    recorded on the tool host.

    Caveat, stated plainly per this plan's own instruction (and the two
    prior corrections this project has already had to make on exactly
    this kind of claim): a production turn always starts a `state_fetch`
    before the transcript is drained (`run_turn`'s own dispatch order) --
    that fetch is itself one call to the tool host (`ha_list_entities`),
    made on every turn regardless of what was asked. This test supplies
    `state_fetch` as a fake that does not go through the tool host at all
    (mirroring `test_criterion_4_a_confident_triage_tier_answers_with_zero_tool_calls`
    in `tests/test_turn_controller.py`), so the zero this test asserts is
    a clean zero: no call was made in service of answering the time
    question itself, which is the claim CMD-03 is actually about -- not
    that a turn makes no tool call ever.
    """
    triage_reply = TierReply(
        answer="it's a quarter past two",
        confident=True,
        needs_tool=False,
        filler=FillerPhrase.ONE_MOMENT,
    )
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=triage_reply),
        calls_tools=False,
    )

    async def _state_fetch() -> list[dict[str, str]]:
        return [{"entity_id": "light.example_lamp", "friendly_name": "the lamp", "state": "on"}]

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what time is it")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    tool_host = _RecordingToolHost()
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        None,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier],
        state_fetch=_state_fetch,
    )

    assert tts.received_text == ["it's a quarter past two"]
    assert tool_host.calls == []
    assert len(triage_tier.envelope_client.calls) == 1
