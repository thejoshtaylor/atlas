"""GOOG-12: an account whose token could not be refreshed, or whose
calendar call fails, is named in the result as unreachable -- never
silently left out, never a crash.

09-01-PLAN.md Task 2 adds `test_revoked_account_is_named_by_the_child`
(child-level, no network call for the revoked account itself). Plan 09-05
adds the turn-level `test_revoked_account_named_not_crashed` to this same
file later.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from atlas_mcp.google import handle_calendar_list_events
from atlas_mcp.google_boundary import parse_accounts_env

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn

from google_fakes import FakeGoogle

_ZONE = ZoneInfo("UTC")


async def test_revoked_account_is_named_by_the_child():
    """An account the token service could not refresh (`access_token`
    `None`, `unreachable_reason="needs_relink"`) is named in
    `unreachable_accounts` with that exact reason, and no request is ever
    made for it -- while a healthy account alongside it still answers."""
    fake_google = FakeGoogle()
    fake_google.add_events(
        "at-work",
        "cal-on",
        [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}],
    )

    accounts_env = json.dumps(
        {
            "accounts": [
                {
                    "label": "work",
                    "email": "work@example.com",
                    "is_default": True,
                    "access_token": "at-work",
                    "unreachable_reason": None,
                    "calendars": [
                        {"calendar_id": "cal-on", "name": "Team", "primary": True, "access": "read_only"}
                    ],
                },
                {
                    "label": "revoked",
                    "email": "revoked@example.com",
                    "is_default": False,
                    "access_token": None,
                    "unreachable_reason": "needs_relink",
                    "calendars": [
                        {"calendar_id": "cal-x", "name": "Personal", "primary": True, "access": "read_only"}
                    ],
                },
            ]
        }
    )
    accounts = parse_accounts_env(accounts_env)

    result = await handle_calendar_list_events(
        accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
    )

    assert {e["account"] for e in result["events"]} == {"work"}
    assert result["unreachable_accounts"] == [{"account": "revoked", "reason": "needs_relink"}]
    assert fake_google.requests == fake_google.requests_by_bearer("at-work")


_ACCOUNTS_ENV = json.dumps(
    {
        "accounts": [
            {
                "label": "work",
                "email": "work@example.com",
                "is_default": True,
                "access_token": "at-work",
                "unreachable_reason": None,
                "calendars": [
                    {"calendar_id": "cal-work", "name": "Team", "primary": True, "access": "read_only"}
                ],
            },
            {
                "label": "home",
                "email": "home@example.com",
                "is_default": False,
                "access_token": None,
                "unreachable_reason": "needs_relink",
                "calendars": [
                    {"calendar_id": "cal-home", "name": "Personal", "primary": True, "access": "read_only"}
                ],
            },
        ]
    }
)


class _ListEventsToolHost:
    def __init__(self, accounts, client) -> None:
        self._accounts = accounts
        self._client = client
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        result = await handle_calendar_list_events(self._accounts, self._client, _ZONE, **arguments)
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(result))])


async def test_revoked_account_named_not_crashed(fake_audio_source, fake_stt, fake_brain, fake_tts):
    """GOOG-12, plan 09-05 Task 3: an answer that names only the reachable
    account still gets the unreachable one appended, by code, never left
    silently out."""
    accounts = parse_accounts_env(_ACCOUNTS_ENV)
    fake_google = FakeGoogle()
    fake_google.add_events(
        "at-work",
        "cal-work",
        [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}],
    )
    tool_host = _ListEventsToolHost(accounts, fake_google.client)
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(name="calendar_list_events", arguments={"start": "2026-10-02", "end": "2026-10-03"})
                ]
            ),
            BrainReply(text="you have standup on your work calendar"),
        ]
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what's on my calendar")])
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

    assert tts.received_text == [
        "you have standup on your work calendar i can't reach your home account right now."
    ]


async def test_an_answer_already_naming_the_unreachable_account_gets_no_duplicate_note(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = parse_accounts_env(_ACCOUNTS_ENV)
    fake_google = FakeGoogle()
    fake_google.add_events(
        "at-work",
        "cal-work",
        [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}],
    )
    tool_host = _ListEventsToolHost(accounts, fake_google.client)
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(name="calendar_list_events", arguments={"start": "2026-10-02", "end": "2026-10-03"})
                ]
            ),
            BrainReply(text="i couldn't reach your home account, but work has a standup"),
        ]
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what's on my calendar")])
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

    assert tts.received_text == ["i couldn't reach your home account, but work has a standup"]


async def test_an_empty_winning_answer_still_names_the_unreachable_account(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """A-WR-02 regression: GOOG-12's own "never silently omit an
    unreachable account" doctrine must also cover the one case it missed
    -- a winning answer that is empty (whitespace-only text with no tool
    calls, `confident=False`) reaches `_CANNOT_DO_REPLY`, but the
    unreachable-account note must still be spoken, not silently dropped
    with the fallback."""
    accounts = parse_accounts_env(_ACCOUNTS_ENV)
    fake_google = FakeGoogle()
    fake_google.add_events(
        "at-work",
        "cal-work",
        [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}],
    )
    tool_host = _ListEventsToolHost(accounts, fake_google.client)
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(name="calendar_list_events", arguments={"start": "2026-10-02", "end": "2026-10-03"})
                ]
            ),
            # Whitespace, not a bare "": `_run_tool_rounds` only substitutes
            # its own fixed `_EMPTY_REPLY` for a bare-empty `reply.text` --
            # a whitespace-only reply passes through unchanged and reaches
            # `run_turn`'s own `winner.answer`, blank after `.strip()`.
            BrainReply(text="   "),
        ]
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what's on my calendar")])
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

    assert tts.received_text == ["i can't do that one i can't reach your home account right now."]
    assert timings.turn_outcome == "empty_answer"
