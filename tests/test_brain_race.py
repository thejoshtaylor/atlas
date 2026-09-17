"""The tier race (VOICE-02) -- the tracer proof lands in plan 01.1-04.

The remaining scaffold below (the one-entry-list invariant) is turned green
by plan 01.1-04's Task 2; every plan in this phase scopes its own suite run
to explicit files for exactly this reason.
"""

from spire_voice.providers.base import BrainReply


async def test_tracer_races_tiers_covers_the_wait_and_answers(
    fake_audio_source, fake_stt, fake_brain, fake_tts, fake_envelope_client
):
    """The phase's tracer slice, driven through `run_turn` end to end: two
    tiers race, a startup-cached holding phrase covers the wait once the
    deadline passes, the winning answer speaks afterwards, and the two audio
    marks stay distinct.

    The triage tier resolves immediately with a non-confident reply; the top
    tier's tool-round settles immediately too, but its envelope call is
    delayed, so it "returns later" without a real network call. The fake
    clock advances past `filler_after_ms` after exactly one real poll cycle,
    which is what gives the fast triage tier a chance to complete before the
    deadline fires -- mirroring `test_silence_timeout_closes_turn`'s idiom.
    """
    from spire_voice.providers.base import FinalTranscript
    from spire_voice.providers.tier_reply import FILLER_TEXT, FillerPhrase, TierReply
    from spire_voice.timing import TurnTimings
    from spire_voice.turn import brain_race
    from spire_voice.turn.controller import run_turn

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what time is it")])

    triage_reply = TierReply(answer="", confident=False, needs_tool=False, filler=FillerPhrase.STILL_LOOKING)
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=triage_reply, delay_s=0.0),
        calls_tools=False,
    )

    top_answer = "it is teatime"
    top_reply = TierReply(answer=top_answer, confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)
    top_tier = brain_race.TierBrain(
        index=1,
        model="top-model",
        brain=fake_brain(replies=[BrainReply(text=top_answer)]),
        envelope_client=fake_envelope_client(reply=top_reply, delay_s=0.05),
        calls_tools=True,
    )

    tts = fake_tts(chunks=[b"\x09\x0a"])
    filler_text = FILLER_TEXT[FillerPhrase.STILL_LOOKING]
    filler_bytes = b"\xfe\xff"
    filler_cache = {filler_text: filler_bytes}
    timings = TurnTimings()

    fake_now = [0.0]

    def clock() -> float:
        fake_now[0] += 0.3
        return fake_now[0]

    await run_turn(
        source,
        stt,
        top_tier.brain,  # unused: `tiers` below overrides the positional `brain`
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier, top_tier],
        filler_after_ms=600,
        filler_cache=filler_cache,
        clock=clock,
        poll_interval_s=0.01,
    )

    # The filler's bytes reached the source before the answer's -- proof
    # D-09's ordering held, not just that both eventually arrived.
    assert source.sent_audio == [filler_bytes, b"\x09\x0a"]

    # Two distinct marks, filler first -- the exact regression Pitfall 1 and
    # `mark_first_audio`'s idempotence guard exist to prevent.
    assert timings.first_audio_at is not None
    assert timings.answer_audio_at is not None
    assert timings.first_audio_at != timings.answer_audio_at
    assert timings.first_audio_at < timings.answer_audio_at

    # The answer FakeTts received is the top tier's, and it received nothing
    # else -- proof the filler never touched the live synthesis path.
    assert tts.received_text == [top_answer]


def test_a_one_entry_tier_list_still_resolves_through_the_list():
    """The companion invariant from the assumption-delta decision: a turn
    resolves through the tier list for every configured list length,
    including one, so a future single-model assumption goes red immediately."""
    raise AssertionError("Wave 0 scaffold - turned green by plan 01.1-04")
