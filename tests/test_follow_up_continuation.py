"""Plan 09-07 Task 1: a clarifying question -- from a triage tier's own
`needs_clarification` win, or from a Google tool's own `needs_clarification`
handoff -- opens the same brief no-wake-word window a calendar confirmation
does (D-06 of phase 9, replacing phase 4's own D-08 for this question), and
the answer completes the original request with every earlier exchange in
context, chained up to `MAX_CHAINED_FOLLOW_UPS` links.

Layers, each proven at the level that actually exercises it:

- `_continuation_messages` (`turn/controller.py`) -- the pure message-list
  builder both a clarification's own answer and an amendment share.
- `dispatch_handoff`'s `needs_clarification` case (`turn/handoff.py`) --
  whether it requests a follow-up, driven directly with no full turn.
- `run_turn`'s own branches -- the triage `needs_clarification` win
  requesting a follow-up (or not, past the chain cap, or on a source with
  no channel at all), the clarification-answer continuation running the
  ordinary pipeline with the whole chain, and silence speaking
  `CANCELLED_REPLY`.
- One end-to-end run through a real `SourceRunner`: an ambiguous account
  proposal, answered without the wake word, confirmed, and inserted.

Every account, calendar, and entity name below is invented -- no real house
appears in this file (`tests/conftest.py`'s and `mcp/atlas_mcp/safety.py`'s
own convention).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

from atlas_mcp.google import handle_calendar_insert_event, handle_calendar_propose_event
from atlas_mcp.google_boundary import AccountGrant, CalendarGrant
from atlas_mcp.safety import Denied

from atlas.config import GateConfig, WakeConfig
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.providers.tier_reply import FillerPhrase, TierReply
from atlas.sources.runner import SourceRunner
from atlas.timing import TurnTimings
from atlas.transports.base import SourceFormat
from atlas.turn import brain_race
from atlas.turn.controller import _continuation_messages, run_turn
from atlas.turn.follow_up import MAX_CHAINED_FOLLOW_UPS, FollowUpChannel, FollowUpRequest
from atlas.turn.handoff import Handoff, HandoffContext, dispatch_handoff

from brain_fakes import RecordingFakeBrain
from google_fakes import FakeGoogle
from pending_action_fakes import FakePendingActionRepository

_ZONE = ZoneInfo("America/New_York")
_NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


# --- _continuation_messages: the pure builder -------------------------------


def test_continuation_messages_is_prior_messages_then_original_then_question():
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=2,
        original_transcript="home",
        question="did you mean the home calendar?",
        prior_messages=(
            {"role": "user", "content": "add dentist on friday at 3"},
            {"role": "assistant", "content": "which account -- home, work?"},
        ),
    )

    messages = _continuation_messages(incoming)

    assert messages == [
        {"role": "user", "content": "add dentist on friday at 3"},
        {"role": "assistant", "content": "which account -- home, work?"},
        {"role": "user", "content": "home"},
        {"role": "assistant", "content": "did you mean the home calendar?"},
    ]


def test_continuation_messages_with_no_prior_messages_is_just_the_one_exchange():
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question="add dentist to the home calendar, friday at 3, for an hour?",
        pending_action_id=1,
    )

    assert _continuation_messages(incoming) == [
        {"role": "user", "content": "add dentist on friday at 3"},
        {"role": "assistant", "content": "add dentist to the home calendar, friday at 3, for an hour?"},
    ]


# --- dispatch_handoff's needs_clarification case ----------------------------


async def test_dispatch_handoff_needs_clarification_requests_a_follow_up_when_available():
    handoff = Handoff(
        kind="needs_clarification",
        payload={"kind": "needs_clarification", "about": "account", "candidates": ["home", "work"]},
    )

    outcome = await dispatch_handoff(
        handoff, None, transcript="add dentist on friday at 3", follow_up_available=True, chain_depth=1
    )

    assert outcome.follow_up is not None
    assert outcome.follow_up.kind == "clarification"
    assert outcome.follow_up.chain_depth == 1
    assert outcome.follow_up.original_transcript == "add dentist on friday at 3"
    assert "home" in outcome.follow_up.question
    assert "work" in outcome.follow_up.question
    assert outcome.turn_outcome == "needs_clarification"
    assert outcome.reply_text == outcome.follow_up.question


async def test_dispatch_handoff_needs_clarification_requests_nothing_when_follow_up_unavailable():
    handoff = Handoff(
        kind="needs_clarification",
        payload={"kind": "needs_clarification", "about": "account", "candidates": ["home", "work"]},
    )

    outcome = await dispatch_handoff(
        handoff, None, transcript="add dentist on friday at 3", follow_up_available=False, chain_depth=1
    )

    assert outcome.follow_up is None
    assert outcome.turn_outcome == "needs_clarification"


async def test_dispatch_handoff_needs_clarification_still_respects_the_chain_cap():
    handoff = Handoff(
        kind="needs_clarification",
        payload={"kind": "needs_clarification", "about": "account", "candidates": ["home", "work"]},
    )

    outcome = await dispatch_handoff(
        handoff,
        None,
        transcript="still not sure",
        follow_up_available=True,
        chain_depth=MAX_CHAINED_FOLLOW_UPS + 1,
    )

    assert outcome.follow_up is None
    assert outcome.turn_outcome == "follow_up_limit"


# --- A triage tier's own needs_clarification win, through run_turn ---------


async def test_a_triage_clarification_winner_requests_a_follow_up_when_a_channel_is_attached(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
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

    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel()
    stt = fake_stt(events=[FinalTranscript(text="turn on the lamp")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        None,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier],
    )

    assert timings.turn_outcome == "needs_clarification"
    requested = source.follow_up.requested
    assert requested is not None
    assert requested.kind == "clarification"
    assert requested.chain_depth == 1
    assert requested.original_transcript == "turn on the lamp"
    assert requested.playback_ends_at is not None
    assert requested.prior_messages == ()


async def test_a_triage_clarification_winner_on_a_source_with_no_channel_requests_nothing(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
    """`tests/test_needs_clarification.py`'s own turn-level cases already
    prove nothing about this turn changes with no channel attached; this
    proves the one new fact those cases do not: no follow-up is stored
    either, since there is no channel to store one on."""
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

    source = fake_audio_source(frames=[b"\x00\x01"])
    assert getattr(source, "follow_up", None) is None
    stt = fake_stt(events=[FinalTranscript(text="turn on the lamp")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        None,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier],
    )

    assert timings.turn_outcome == "needs_clarification"
    assert len(tts.received_text) == 1


async def test_a_triage_clarification_past_the_chain_cap_opens_no_window(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=MAX_CHAINED_FOLLOW_UPS,
        original_transcript="add dentist on friday at 3",
        question="which one -- home, work?",
    )
    clarifying_reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=("home", "work"),
    )
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=clarifying_reply, delay_s=0.0),
        calls_tools=False,
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="still not sure")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        None,
        tts,
        None,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier],
    )

    assert timings.turn_outcome == "needs_clarification"
    assert source.follow_up.requested is None
    assert len(tts.received_text) == 1


async def test_a_follow_up_raised_inside_a_continuation_carries_every_earlier_message(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
    """D-06: a follow-up a continuation turn itself raises carries the
    whole chain forward, not just the last hop -- `prior_messages` equals
    the continuation messages this turn ran with, and `original_transcript`
    is this turn's own transcript."""
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question="i'm not sure which one you mean -- home, work?",
    )
    clarifying_reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=("home", "work"),
    )
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=clarifying_reply, delay_s=0.0),
        calls_tools=False,
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="the calendar one")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        None,
        tts,
        None,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier],
    )

    requested = source.follow_up.requested
    assert requested is not None
    assert requested.chain_depth == 2
    assert requested.original_transcript == "the calendar one"
    assert requested.prior_messages == (
        {"role": "user", "content": "add dentist on friday at 3"},
        {"role": "assistant", "content": "i'm not sure which one you mean -- home, work?"},
    )


# --- A clarification's own answer: the continuation turn --------------------


async def test_a_clarification_answer_runs_the_ordinary_pipeline_with_the_whole_chain(
    fake_audio_source, fake_stt, fake_tts
):
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question="i'm not sure which one you mean -- home, work?",
    )
    brain = RecordingFakeBrain(replies=[BrainReply(text="done, it's on your calendar")])
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="home")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
    )

    assert timings.turn_outcome != "needs_clarification"
    assert timings.turn_outcome != "follow_up_silence"
    assert brain.call_count == 1
    messages = brain.calls[0].messages
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "add dentist on friday at 3"}
    assert messages[2] == {"role": "assistant", "content": "i'm not sure which one you mean -- home, work?"}
    assert messages[3] == {"role": "user", "content": "home"}
    assert tts.received_text == ["done, it's on your calendar"]


async def test_silence_after_a_clarifying_question_speaks_cancelled_and_stores_nothing(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question="i'm not sure which one you mean -- home, work?",
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming, window_opens_at=0.0, window_s=10.0)
    stt = fake_stt(hang=True)
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    fake_now = [0.0]

    def clock() -> float:
        fake_now[0] += 5.0
        return fake_now[0]

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        max_utterance_s=100,
        clock=clock,
        poll_interval_s=0.01,
    )

    assert tts.received_text == ["cancelled, nothing was changed"]
    assert timings.turn_outcome == "follow_up_silence"
    assert source.follow_up.requested is None


async def test_an_unreadable_reply_after_a_clarifying_question_also_cancels(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """`is_no_command` catches more than literal silence -- a bare wake
    phrase or filler-only reply reads the same way (D-09's own "however
    many reasons code has for choosing it" doctrine, `pending_action.py`'s
    module docstring)."""
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question="i'm not sure which one you mean -- home, work?",
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="um")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
    )

    assert tts.received_text == ["cancelled, nothing was changed"]
    assert timings.turn_outcome == "follow_up_silence"


# --- Existing turn-level clarification cases keep passing with no channel --


async def test_needs_clarification_turn_level_cases_are_unaffected_by_this_plan(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
    """A light re-statement of `tests/test_needs_clarification.py`'s own
    turn-level proof, from this file, so a reviewer of this plan's diff
    sees the "no channel, nothing changes" guarantee stated alongside the
    new behavior it is a boundary for."""
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

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn on the lamp")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        None,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier],
    )

    assert timings.turn_outcome == "needs_clarification"
    spoken = tts.received_text[0]
    assert "light.example_lamp" in spoken
    assert "light.example_desk_lamp" in spoken


# --- End to end, through one real SourceRunner ------------------------------


def _home_account() -> AccountGrant:
    return AccountGrant(
        label="home",
        email="home@example.com",
        is_default=False,
        access_token="at-home",
        unreachable_reason=None,
        calendars=(CalendarGrant(calendar_id="cal-home-primary", name="Home", primary=True, access="read_write"),),
    )


def _work_account() -> AccountGrant:
    return AccountGrant(
        label="work",
        email="work@example.com",
        is_default=False,
        access_token="at-work",
        unreachable_reason=None,
        calendars=(CalendarGrant(calendar_id="cal-work-primary", name="Work", primary=True, access="read_write"),),
    )


class _GoogleToolHost:
    """The same functional-double shape `test_follow_up_window.py`'s own
    `_GoogleToolHost` establishes, reduced to only the two tools this
    end-to-end run ever calls."""

    def __init__(self, accounts: "tuple[AccountGrant, ...]", google_client: Any) -> None:
        self._accounts = accounts
        self._google_client = google_client
        self.calls: "list[tuple[str, dict]]" = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        try:
            if name == "calendar_propose_event":
                result = await handle_calendar_propose_event(self._accounts, _ZONE, **arguments)
            elif name == "calendar_insert_event":
                result = await handle_calendar_insert_event(self._accounts, self._google_client, **arguments)
            else:
                raise AssertionError(f"unknown tool: {name}")
        except Denied as exc:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text=str(exc))])
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(result))])


class _FixedFramesSource:
    def __init__(self, chunks: "list[bytes]") -> None:
        self._chunks = chunks

    async def frames(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def send_audio(self, chunk: bytes) -> None:
        pass

    def source_format(self) -> SourceFormat:
        return SourceFormat("pcm", 16000)


class _AlwaysHitWakeDetector:
    def process(self, chunk: bytes) -> Any:
        from tests.conftest import FakeWakeHit

        return FakeWakeHit(score=1.0)

    def close(self) -> None:
        pass


class _ScriptedBrain:
    """One brain across the whole chain -- `tools:` truthy is always the
    confirmation round (`CONFIRM_CANCEL_TOOLS`); the first falsy-`tools`
    call is the wake turn's own ambiguous proposal, the second is the
    clarification answer's own proposal, now carrying `account="home"`."""

    def __init__(self) -> None:
        self.ordinary_calls = 0

    async def chat(self, messages: "list[dict]", tools: Any = None) -> BrainReply:
        if tools:
            return BrainReply(tool_calls=[ToolCall(name="confirm", arguments={})])
        self.ordinary_calls += 1
        arguments: dict[str, Any] = {"title": "Dentist", "start": "2026-10-02T15:00"}
        if self.ordinary_calls > 1:
            arguments["account"] = "home"
        return BrainReply(tool_calls=[ToolCall(name="calendar_propose_event", arguments=arguments)])


async def test_end_to_end_two_accounts_no_default_clarified_without_the_wake_word_then_confirmed(fake_stt):
    accounts = (_home_account(), _work_account())
    fake_google = FakeGoogle()
    tool_host = _GoogleToolHost(accounts, fake_google.client)
    pending_actions = FakePendingActionRepository()
    brain = _ScriptedBrain()

    source = _FixedFramesSource([b"\x00"])
    sent_audio: list[bytes] = []

    async def _send_audio(chunk: bytes) -> None:
        sent_audio.append(chunk)

    source.send_audio = _send_audio  # type: ignore[method-assign]

    clarification_stt = fake_stt(events=[FinalTranscript(text="home")])
    confirmation_stt = fake_stt(events=[FinalTranscript(text="yes")])
    turns_run: list[int] = []

    async def run_turn_fn(turn_source: Any) -> None:
        turns_run.append(1)
        handoff_context = HandoffContext(
            source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
        )
        from tests.conftest import FakeStt, FakeTts

        if len(turns_run) == 1:
            stt = FakeStt(events=[FinalTranscript(text="add dentist on friday at 3")])
        elif len(turns_run) == 2:
            stt = clarification_stt
        else:
            stt = confirmation_stt

        await run_turn(
            turn_source,
            stt,
            brain,
            FakeTts(chunks=[b"\x01\x02"]),
            tool_host,
            tools_schema=[],
            system_prompt="you manage a calendar",
            max_tool_rounds=3,
            timings=TurnTimings(),
            handoff_context=handoff_context,
        )

    runner = SourceRunner(
        "camera",
        source,
        _AlwaysHitWakeDetector(),
        lambda chunk: chunk,
        run_turn_fn,
        wake_config=WakeConfig(engine="vosk", refractory_s=0.0),
        gate_config=GateConfig(),
        follow_up_window_s=lambda: 6.0,
        follow_up_echo_tail_s=0.0,
    )

    await runner.run()

    assert turns_run == [1, 1, 1]
    assert [name for name, _ in tool_host.calls] == [
        "calendar_propose_event",
        "calendar_propose_event",
        "calendar_insert_event",
    ]
    rows = list(pending_actions._rows.values())
    assert len(rows) == 1
    assert rows[0].status == "executed"
    assert rows[0].arguments["account"] == "home"
    assert sent_audio  # the clarifying question, the readback, and "done" all wrote audio
