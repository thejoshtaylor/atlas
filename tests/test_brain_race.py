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


async def test_a_one_entry_tier_list_still_resolves_through_the_list(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """The companion invariant from the assumption-delta decision: a turn
    resolves through the tier list for every configured list length,
    including one, so a future single-model assumption goes red immediately.

    Drives `run_turn` end to end, not `race_tiers` alone, with `tiers` left
    at its default (`None`) -- the exact seam a future single-model bypass
    would have to skip past.
    """
    from spire_voice.providers.base import FinalTranscript
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = fake_brain(replies=[BrainReply(text="the only tier answered")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert tts.received_text == ["the only tier answered"]
    # No race was lost and no filler played -- the single tier is also the
    # top tier, so its arrival ends the race with nothing left pending. Only
    # the one answer chunk ever reached the source.
    assert source.sent_audio == [b"\x01\x02"]


async def test_two_confident_tiers_in_one_batch_resolve_to_the_lower_index():
    """Inside one `asyncio.wait` `FIRST_COMPLETED` batch, ties break by
    ascending tier index, never by set iteration order -- repeated 20 times
    so a pass cannot be luck against Python's own set ordering.
    """
    import asyncio

    from spire_voice.providers.tier_reply import FillerPhrase, TierReply
    from spire_voice.turn import brain_race

    low_reply = TierReply(answer="low", confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)
    high_reply = TierReply(answer="high", confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)

    async def _immediate(reply: TierReply) -> TierReply:
        return reply

    for _ in range(20):
        tasks = {
            0: asyncio.create_task(_immediate(low_reply)),
            1: asyncio.create_task(_immediate(high_reply)),
        }
        winner = await brain_race.race_tiers(tasks)
        assert winner is low_reply


async def test_losers_are_cancelled_and_their_unwind_is_awaited():
    """When a winner is found, every still-pending task reaches `cancelled()`,
    and `race_tiers` does not return before a loser's own `finally` block has
    actually run -- proof the cancellation was awaited, not fired and
    forgotten.
    """
    import asyncio

    from spire_voice.providers.tier_reply import FillerPhrase, TierReply
    from spire_voice.turn import brain_race

    winner_reply = TierReply(answer="fast", confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)
    loser_finally_ran = False

    async def _fast_winner() -> TierReply:
        return winner_reply

    async def _slow_loser() -> TierReply:
        nonlocal loser_finally_ran
        try:
            await asyncio.sleep(10)
            raise AssertionError("the loser should have been cancelled before this line")
        finally:
            loser_finally_ran = True

    loser_task = asyncio.create_task(_slow_loser())
    tasks = {0: asyncio.create_task(_fast_winner()), 1: loser_task}

    winner = await brain_race.race_tiers(tasks)

    assert winner is winner_reply
    assert loser_task.cancelled()
    assert loser_finally_ran


async def test_three_tiers_the_middle_one_confident_first_wins_and_cancels_the_rest():
    """A three-entry tier list where the middle tier is confident first
    speaks the middle tier's answer and cancels the other two -- proof the
    winner is not always tier 0 or the top tier.
    """
    import asyncio

    from spire_voice.providers.tier_reply import FillerPhrase, TierReply
    from spire_voice.turn import brain_race

    middle_reply = TierReply(answer="middle", confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)

    async def _never_confident_but_slow() -> TierReply:
        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled")

    async def _middle() -> TierReply:
        return middle_reply

    bottom_task = asyncio.create_task(_never_confident_but_slow())
    top_task = asyncio.create_task(_never_confident_but_slow())
    tasks = {0: bottom_task, 1: asyncio.create_task(_middle()), 2: top_task}

    winner = await brain_race.race_tiers(tasks)

    assert winner is middle_reply
    assert bottom_task.cancelled()
    assert top_task.cancelled()


async def test_a_needs_tool_triage_reply_never_reaches_the_tool_host(
    fake_audio_source, fake_stt, fake_brain, fake_tts, fake_envelope_client
):
    """A triage tier's `needs_tool=True` reply makes zero calls to the tool
    host across a whole turn -- only the top tier's own tool round reaches
    it. `fake_envelope_client` itself fails the test if a `tools` keyword
    ever reaches an envelope call (see `tests/conftest.py`).
    """
    from spire_voice.providers.base import FinalTranscript, ToolCall
    from spire_voice.providers.tier_reply import FillerPhrase, TierReply
    from spire_voice.timing import TurnTimings
    from spire_voice.turn import brain_race
    from spire_voice.turn.controller import run_turn

    class _RecordingToolHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call_tool(self, name, arguments):
            from types import SimpleNamespace

            self.calls.append((name, arguments))
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])

    triage_reply = TierReply(answer="", confident=False, needs_tool=True, filler=FillerPhrase.STILL_LOOKING)
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=triage_reply, delay_s=0.0),
        calls_tools=False,
    )

    top_answer = "flipped it"
    top_reply = TierReply(answer=top_answer, confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)
    tool_call = ToolCall(
        name="ha_call_service",
        arguments={"domain": "switch", "service": "turn_on", "entity_id": "switch.example_fan"},
    )
    top_brain = fake_brain(replies=[BrainReply(tool_calls=[tool_call]), BrainReply(text=top_answer)])
    top_tier = brain_race.TierBrain(
        index=1,
        model="top-model",
        brain=top_brain,
        envelope_client=fake_envelope_client(reply=top_reply, delay_s=0.05),
        calls_tools=True,
    )

    tool_host = _RecordingToolHost()
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
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
        filler_after_ms=0,
        filler_cache=None,
    )

    assert tool_host.calls == [("ha_call_service", tool_call.arguments)]
    assert tts.received_text == [top_answer]


async def test_a_raising_triage_tier_is_dropped_and_the_race_still_resolves():
    """A triage tier that raises is logged and dropped from the race; the
    remaining tier still resolves the race normally.
    """
    import asyncio

    from spire_voice.providers.tier_reply import FillerPhrase, TierReply
    from spire_voice.turn import brain_race

    top_reply = TierReply(answer="fine without it", confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)

    async def _raise() -> TierReply:
        raise RuntimeError("triage tier exploded")

    async def _top() -> TierReply:
        return top_reply

    tasks = {0: asyncio.create_task(_raise()), 1: asyncio.create_task(_top())}
    winner = await brain_race.race_tiers(tasks)

    assert winner is top_reply


async def test_a_raising_top_tier_propagates_out_of_run_turn(
    fake_audio_source, fake_stt, fake_tts
):
    """The top tier raising propagates all the way out of `run_turn` --
    there is no fallback behind it to swallow the exception.
    """
    import pytest

    from spire_voice.providers.base import FinalTranscript
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    class _RaisingBrain:
        async def chat(self, messages, tools=None):
            raise RuntimeError("top tier exploded")

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    tts = fake_tts(chunks=[])
    timings = TurnTimings()

    with pytest.raises(RuntimeError, match="top tier exploded"):
        await run_turn(
            source,
            stt,
            _RaisingBrain(),
            tts,
            None,
            tools_schema=[],
            system_prompt="you control a home",
            max_tool_rounds=3,
            timings=timings,
        )


async def test_racing_many_times_does_not_leak_tasks():
    """RESEARCH.md assumption A3: a cancelled tier's task must not linger.

    30 consecutive races, each cancelling one always-pending loser; the live
    `asyncio` task count returns to its pre-loop value afterward. This is a
    task-accounting proxy for A3's real concern (`httpx` connection-pool
    cleanup under cancellation), not a live-network soak -- no real
    `instructor`/`httpx` client is exercised here, so a leak specific to
    `httpx`'s own connection pool would not show up in this count. Recorded
    in the SUMMARY, not silently assumed away.
    """
    import asyncio

    from spire_voice.providers.tier_reply import FillerPhrase, TierReply
    from spire_voice.turn import brain_race

    winner_reply = TierReply(answer="ok", confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)

    async def _fast_winner() -> TierReply:
        return winner_reply

    async def _slow_loser() -> TierReply:
        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled")

    baseline = len(asyncio.all_tasks())
    for _ in range(30):
        tasks = {0: asyncio.create_task(_fast_winner()), 1: asyncio.create_task(_slow_loser())}
        winner = await brain_race.race_tiers(tasks)
        assert winner is winner_reply

    await asyncio.sleep(0)
    assert len(asyncio.all_tasks()) == baseline
