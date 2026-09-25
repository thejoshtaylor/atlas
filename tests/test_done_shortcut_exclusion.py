"""Plan 09-05 Task 2, GOOG-11/D-13: a Google tool must never join
`turn/controller.py::_DONE_SHORTCUT_TOOLS` -- a canned "done" for a
calendar or email action would assert an outcome no read-back confirmed
(CMD-07). This file holds the positive test that fails the suite the
moment any Google tool name is added to that set, plus the two behavioral
cases naming why: `_round_settles_as_done` never fires for a Google tool
call, and a real turn calling one always takes the model's own second
round rather than a canned confirmation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from atlas_mcp.google_tools import ALL_TOOL_NAMES

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import _DONE_SHORTCUT_TOOLS, _round_settles_as_done, run_turn


def test_google_tools_never_in_done_shortcut_set():
    assert ALL_TOOL_NAMES.isdisjoint(_DONE_SHORTCUT_TOOLS)


@pytest.mark.parametrize("tool_name", sorted(ALL_TOOL_NAMES))
def test_round_settles_as_done_is_false_for_every_google_tool(tool_name):
    reply = SimpleNamespace(text="", tool_calls=[SimpleNamespace(name=tool_name)])
    results = [SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps({"changed": []}))])]
    messages = [{"role": "user", "content": "add dentist on friday at 3"}]
    assert _round_settles_as_done(reply, results, messages) is False


async def test_a_turn_calling_calendar_list_events_takes_a_second_round_and_never_says_done(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    class _ListEventsToolHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call_tool(self, name: str, arguments: dict):
            self.calls.append((name, arguments))
            payload = {"time_zone": "UTC", "events": [], "unreachable_accounts": []}
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(payload))])

    tool_host = _ListEventsToolHost()
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_list_events",
                        arguments={"start": "2026-10-02", "end": "2026-10-03"},
                    )
                ]
            ),
            BrainReply(text="you have nothing on your calendar tomorrow"),
        ]
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what's on my calendar tomorrow")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
    )

    assert brain.call_count == 2
    assert tts.received_text == ["you have nothing on your calendar tomorrow"]
    assert timings.turn_outcome != "local_intent"
