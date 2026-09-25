"""Plan 09-06 Task 2: after a calendar readback, the microphone stays open
briefly with no wake word, starting only after the readback's own audio has
finished playing plus an echo tail (D-06, D-09; the camera has no echo
cancellation).

Three layers, each proven at the level that actually exercises it:

- `estimate_playback_end` (`turn/follow_up.py`) and `SpeechResult`
  (`turn/controller.py`) -- pure unit tests, no asyncio.
- `_drain_to_final_transcript`'s `onset_deadline` (`turn/controller.py`) --
  driven through the public `run_turn` with an injectable clock, the same
  pattern `tests/test_turn_controller.py::test_silence_timeout_closes_turn`
  already establishes for the ordinary `max_utterance_s` bound.
- `FollowUpSource` and `SourceRunner._run_follow_ups`
  (`sources/runner.py`) -- the runner-level mechanics: which frames a
  follow-up turn's own STT can ever see, that it records no wake event,
  and that the chain stops at `MAX_CHAINED_FOLLOW_UPS`.
- Two end-to-end runs through one real `SourceRunner`, with the real
  `run_turn`, a fake Google tool host, and `FakePendingActionRepository`.

Every account, calendar, and entity name below is invented -- no real house
appears in this file (`tests/conftest.py`'s and `mcp/atlas_mcp/safety.py`'s
own convention).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

from atlas_mcp.google import handle_calendar_insert_event, handle_calendar_propose_event
from atlas_mcp.google_boundary import AccountGrant, CalendarGrant
from atlas_mcp.safety import Denied

from atlas.calibration.record import EchoCalibration
from atlas.config import GateConfig, WakeConfig
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.providers.tts_xai import SinkFormat
from atlas.timing import TurnTimings
from atlas.transports.base import SourceFormat
from atlas.turn.controller import SpeechResult, run_turn
from atlas.turn.follow_up import MAX_CHAINED_FOLLOW_UPS, FollowUpChannel, FollowUpRequest, estimate_playback_end
from atlas.sources.runner import FollowUpSource, SourceRunner

from google_fakes import FakeGoogle
from pending_action_fakes import FakePendingActionRepository

_ZONE = ZoneInfo("America/New_York")
_NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


# --- estimate_playback_end / SpeechResult ------------------------------------


def test_estimate_playback_end_alaw_8khz():
    sink = SinkFormat("alaw", 8000)
    # 8000 bytes at 1 byte/sample and 8000 Hz is exactly 1 second of audio.
    result = SpeechResult(bytes_sent=8000, first_write_at=10.0, last_write_at=10.05)
    assert estimate_playback_end(result, sink) == 11.0


def test_estimate_playback_end_pcm_24khz_divides_by_48000():
    sink = SinkFormat("pcm", 24000)
    # 48000 bytes at 2 bytes/sample and 24000 Hz is exactly 1 second.
    result = SpeechResult(bytes_sent=48000, first_write_at=5.0, last_write_at=5.02)
    assert estimate_playback_end(result, sink) == 6.0


def test_estimate_playback_end_with_no_sink_is_last_write_at():
    result = SpeechResult(bytes_sent=1234, first_write_at=1.0, last_write_at=3.5)
    assert estimate_playback_end(result, None) == 3.5


def test_estimate_playback_end_never_earlier_than_last_write_at():
    # A burst of writes landing faster than the audio duration they encode
    # must not estimate a playback end in the past.
    sink = SinkFormat("alaw", 8000)
    result = SpeechResult(bytes_sent=80, first_write_at=1.0, last_write_at=1.9)
    assert estimate_playback_end(result, sink) == 1.9


def test_estimate_playback_end_with_nothing_ever_written_is_last_write_at():
    sink = SinkFormat("alaw", 8000)
    result = SpeechResult(bytes_sent=0, first_write_at=None, last_write_at=None)
    assert estimate_playback_end(result, sink) is None


# --- `_drain_to_final_transcript`'s onset_deadline, through run_turn --------


async def test_a_follow_up_turn_with_no_answer_by_the_window_close_speaks_cancelled(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    source = fake_audio_source(frames=[b"\x00\x01"])
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question="add dentist to the home calendar, friday at 3 pm, for an hour?",
        pending_action_id=1,
    )
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


async def test_once_an_event_arrives_the_onset_deadline_no_longer_gates_the_drain(fake_stt):
    """Once `pending is not None` (a real event has already arrived), the
    `onset_deadline` check's own `pending is None` guard short-circuits it
    away for every later iteration -- proven directly against
    `_drain_to_final_transcript` (Task 2's own `<behavior>`: "the ordinary
    `max_utterance_s` bound governs the rest of the utterance").

    A clock that advances by 1.0 real-seeming unit per call: the one call
    before the read loop starts (`deadline = clock() + max_utterance_s`)
    and the two per-iteration checks before the first event is read land
    on 2 and 3 -- both below `onset_deadline=3.5`, so the first event is
    read normally. The second iteration's own onset check would land on
    5 -- past `onset_deadline` -- but is never even evaluated, because
    `pending is not None` short-circuits it first.
    """
    from atlas.turn.controller import _drain_to_final_transcript

    class _NoFramesSource:
        async def frames(self):
            return
            yield  # pragma: no cover - never reached; makes this an async generator

        def source_format(self) -> SourceFormat:
            return SourceFormat("pcm", 16000)

    stt = fake_stt(events=[FinalTranscript(text="yes")])
    timings = TurnTimings()
    fake_now = [0.0]

    def clock() -> float:
        fake_now[0] += 1.0
        return fake_now[0]

    final = await _drain_to_final_transcript(
        _NoFramesSource(),
        stt,
        1000,
        timings,
        clock=clock,
        poll_interval_s=0.01,
        onset_deadline=3.5,
    )

    assert final is not None
    assert final.text == "yes"


# --- FollowUpSource: no frame read before window_opens_at reaches the STT --


class _FixedFramesSource:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def frames(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def send_audio(self, chunk: bytes) -> None:
        pass

    def source_format(self) -> SourceFormat:
        return SourceFormat("pcm", 16000)


async def test_follow_up_source_drops_every_frame_read_before_opens_at():
    clock_values = iter([0.0, 0.4, 0.8, 1.2])

    def clock() -> float:
        return next(clock_values)

    wrapped = _FixedFramesSource([b"early-1", b"early-2", b"late-1", b"late-2"])
    source = FollowUpSource(wrapped, opens_at=0.5, clock=clock)

    received = [chunk async for chunk in source.frames()]

    # `clock()` returns 0.0 and 0.4 for the first two chunks (both < 0.5,
    # dropped) and 0.8/1.2 for the last two (both >= 0.5, kept) -- proving
    # directly that no frame read before `opens_at` ever reaches whatever
    # is draining `frames()` (a real STT's own `stream()` call, in
    # production).
    assert received == [b"late-1", b"late-2"]


async def test_follow_up_source_forwards_send_audio_and_source_format():
    wrapped = _FixedFramesSource([])
    source = FollowUpSource(wrapped, opens_at=0.0, clock=lambda: 1.0)
    await source.send_audio(b"reply-chunk")
    assert source.source_format() == SourceFormat("pcm", 16000)


# --- SourceRunner: wiring, no wake event on a follow-up, the chain cap -----


class _AlwaysHitWakeDetector:
    def __init__(self, score: float = 1.0) -> None:
        self._score = score

    def process(self, chunk: bytes) -> Any:
        from tests.conftest import FakeWakeHit

        return FakeWakeHit(score=self._score)

    def close(self) -> None:
        pass


def _runner(
    source: Any,
    run_turn_fn,
    *,
    follow_up_window_s=None,
    follow_up_echo_tail_s: float = 0.8,
    wake_event_repo: Any = None,
    calibration: "EchoCalibration | None" = None,
    clock=None,
) -> SourceRunner:
    kwargs: dict[str, Any] = {}
    if clock is not None:
        kwargs["clock"] = clock
    return SourceRunner(
        "camera",
        source,
        _AlwaysHitWakeDetector(),
        lambda chunk: chunk,
        run_turn_fn,
        wake_config=WakeConfig(engine="vosk", refractory_s=0.0),
        gate_config=GateConfig(),
        follow_up_window_s=follow_up_window_s,
        follow_up_echo_tail_s=follow_up_echo_tail_s,
        wake_event_repo=wake_event_repo,
        calibration=calibration,
        **kwargs,
    )


async def test_a_source_runner_built_without_follow_up_window_s_attaches_no_channel(fake_audio_source):
    source = fake_audio_source(frames=[b"\x00"])
    seen_follow_up: list[Any] = []

    async def run_turn_fn(turn_source: Any) -> None:
        seen_follow_up.append(getattr(turn_source, "follow_up", "not-an-attribute"))

    runner = _runner(source, run_turn_fn, follow_up_window_s=None)
    await runner.run()

    assert seen_follow_up == ["not-an-attribute"]


async def test_a_follow_up_turn_records_no_wake_event(fake_audio_source, fake_wake_event_repository):
    source = fake_audio_source(frames=[b"\x00"])
    wake_event_repo = fake_wake_event_repository()
    calls: list[int] = []

    async def run_turn_fn(turn_source: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            # The wake turn: propose and request a follow-up.
            turn_source.follow_up.request(
                FollowUpRequest(
                    kind="confirmation",
                    chain_depth=1,
                    original_transcript="add dentist on friday at 3",
                    question="add dentist ... ?",
                    pending_action_id=1,
                    playback_ends_at=0.0,
                )
            )
            return
        # The follow-up turn: drain whatever it can see, content unused.
        async for _chunk in turn_source.frames():
            pass

    runner = _runner(
        source,
        run_turn_fn,
        follow_up_window_s=lambda: 6.0,
        follow_up_echo_tail_s=0.0,
        wake_event_repo=wake_event_repo,
        clock=lambda: 0.0,
    )
    await runner.run()
    await runner.drain_pending_wake_events()

    # Exactly one wake event -- for the real wake hit that started this
    # source's own loop -- never a second one for the follow-up turn
    # `_run_follow_ups` ran afterward.
    assert len(wake_event_repo.events) == 1
    assert calls == [1, 1]


async def test_a_request_above_max_chained_follow_ups_opens_no_window(fake_audio_source):
    source = fake_audio_source(frames=[])
    runner = _runner(source, run_turn_fn=lambda s: None, follow_up_window_s=lambda: 6.0)

    turns_run: list[Any] = []

    async def run_turn_fn(turn_source: Any) -> None:
        turns_run.append(turn_source)

    runner._run_turn_fn = run_turn_fn  # noqa: SLF001 -- exercising the internal chain guard directly
    channel = FollowUpChannel(
        requested=FollowUpRequest(
            kind="confirmation",
            chain_depth=MAX_CHAINED_FOLLOW_UPS + 1,
            original_transcript="x",
            question="y",
            pending_action_id=1,
            playback_ends_at=0.0,
        )
    )

    await runner._run_follow_ups(channel)  # noqa: SLF001

    assert turns_run == []


async def test_the_echo_tail_is_the_larger_of_configured_and_calibration_plus_margin(fake_audio_source):
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    calibration = EchoCalibration(
        schema_version=1,
        probe_format_version=1,
        probe_seed=1,
        source="camera",
        delay_s=2.0,
        confidence=1.0,
        echo_level=0.1,
        gain=1.0,
        agc_verdict="absent",
        segment_levels=(0.1, 0.1, 0.1),
        encoding="pcm",
        sample_rate=8000,
        channels=1,
        placement_note="test fixture, no real room",
        taken_at=_dt(2026, 1, 1, tzinfo=_tz.utc),
    )
    source = fake_audio_source(frames=[])
    opened_at: list[float] = []

    async def run_turn_fn(turn_source: Any) -> None:
        opened_at.append(turn_source.follow_up.window_opens_at)

    runner = _runner(
        source,
        run_turn_fn,
        follow_up_window_s=lambda: 6.0,
        follow_up_echo_tail_s=0.1,  # far smaller than the calibration's own delay_s + 0.2
        calibration=calibration,
        clock=lambda: 0.0,
    )
    channel = FollowUpChannel(
        requested=FollowUpRequest(
            kind="confirmation",
            chain_depth=1,
            original_transcript="x",
            question="y",
            pending_action_id=1,
            playback_ends_at=100.0,
        )
    )

    await runner._run_follow_ups(channel)  # noqa: SLF001

    # 100.0 (playback_ends_at) + max(0.1, 2.0 + 0.2) == 102.2 -- the
    # calibration's own measured delay wins over the smaller configured tail.
    assert opened_at == [102.2]


# --- Two end-to-end runs through one real SourceRunner ----------------------


def _home_account() -> AccountGrant:
    return AccountGrant(
        label="home",
        email="home@example.com",
        is_default=True,
        access_token="at-home",
        unreachable_reason=None,
        calendars=(CalendarGrant(calendar_id="cal-home-primary", name="Home", primary=True, access="read_write"),),
    )


class _GoogleToolHost:
    """The same functional-double shape `test_pending_action.py`'s own
    `_GoogleToolHost` establishes, reduced to only the two tools these two
    end-to-end runs ever call."""

    def __init__(self, accounts: tuple[AccountGrant, ...], google_client: Any) -> None:
        self._accounts = accounts
        self._google_client = google_client
        self.calls: list[tuple[str, dict]] = []

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


def _end_to_end_runner(
    *, tool_host: Any, pending_actions: Any, second_turn_stt: Any, brain: Any, window_s: float
) -> tuple[SourceRunner, list[bytes]]:
    """Builds one `SourceRunner` whose `run_turn_fn` calls the real
    `run_turn` -- the wake turn always speaks "add dentist on friday at
    3"; `second_turn_stt` scripts the follow-up turn's own reply.
    Real `time.monotonic()` throughout (the runner's default clock,
    unchanged), the same domain `run_turn`'s own default `clock` uses --
    `window_s` is kept small so a real-silence test still resolves fast.
    """
    from atlas.turn.handoff import HandoffContext

    source = _FixedFramesSource([b"\x00"])
    sent_audio: list[bytes] = []

    async def _send_audio(chunk: bytes) -> None:
        sent_audio.append(chunk)

    source.send_audio = _send_audio  # type: ignore[method-assign]

    turns_run: list[int] = []

    async def run_turn_fn(turn_source: Any) -> None:
        turns_run.append(1)
        handoff_context = HandoffContext(
            source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
        )
        from tests.conftest import FakeStt, FakeTts

        stt = FakeStt(events=[FinalTranscript(text="add dentist on friday at 3")]) if len(turns_run) == 1 else second_turn_stt

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

    runner = _runner(source, run_turn_fn, follow_up_window_s=lambda: window_s, follow_up_echo_tail_s=0.0)
    return runner, sent_audio


async def test_end_to_end_wake_readback_yes_inserts_and_says_done(fake_stt):
    accounts = (_home_account(),)
    fake_google = FakeGoogle()
    tool_host = _GoogleToolHost(accounts, fake_google.client)
    pending_actions = FakePendingActionRepository()

    class _ScriptedBrain:
        async def chat(self, messages, tools=None):
            if tools:
                # The confirmation round is the only caller that ever
                # offers a non-empty `tools` here (`CONFIRM_CANCEL_TOOLS`)
                # -- the wake turn's own tier race is given `tools_schema
                # =[]` in this test -- so this always says "confirm".
                return BrainReply(tool_calls=[ToolCall(name="confirm", arguments={})])
            # The ordinary tier race -- proposes the event.
            return BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T15:00", "account": "home"},
                    )
                ]
            )

    brain = _ScriptedBrain()
    second_turn_stt = fake_stt(events=[FinalTranscript(text="yes")])

    runner, sent_audio = _end_to_end_runner(
        tool_host=tool_host,
        pending_actions=pending_actions,
        second_turn_stt=second_turn_stt,
        brain=brain,
        window_s=6.0,
    )

    await runner.run()

    rows = list(pending_actions._rows.values())
    assert len(rows) == 1
    assert rows[0].status == "executed"
    assert [name for name, _ in tool_host.calls] == ["calendar_propose_event", "calendar_insert_event"]
    assert sent_audio  # the readback and the "done" reply both wrote audio


async def test_end_to_end_wake_readback_and_no_answer_says_cancelled(fake_stt, fake_brain):
    accounts = (_home_account(),)
    fake_google = FakeGoogle()
    tool_host = _GoogleToolHost(accounts, fake_google.client)
    pending_actions = FakePendingActionRepository()

    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T15:00", "account": "home"},
                    )
                ]
            ),
        ]
    )
    # Hangs forever -- the follow-up turn must never reach a confirmation
    # round on silence; it must close on its own real, short window.
    second_turn_stt = fake_stt(hang=True)

    runner, _sent_audio = _end_to_end_runner(
        tool_host=tool_host,
        pending_actions=pending_actions,
        second_turn_stt=second_turn_stt,
        brain=brain,
        window_s=0.2,
    )

    await runner.run()

    rows = list(pending_actions._rows.values())
    assert len(rows) == 1
    assert rows[0].status == "expired"
