"""Plan 09-05 Task 2 (T-09-27): two independent controls keep
`calendar_insert_event`/`calendar_delete_event` unreachable from the
model -- hidden from `PluginManager.tools_schema` (this file's schema
test, spawning the real `atlas_mcp.google` child) and refused before
dispatch by `atlas.turn.handoff.is_code_only_tool`, bare or
collision-prefixed (this file's dispatch-refusal tests, at the
`run_turn` level). The drift guard proves the two controls actually
agree with the real child's own registered tool names.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from atlas.config import SecurityConfig
from atlas.db.repository import Plugin
from atlas.plugins.manager import PluginManager
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.handoff import CODE_ONLY_REFUSAL

from atlas_mcp.google_tools import ALL_TOOL_NAMES, CODE_ONLY_TOOL_NAMES, GOOGLE_PLUGIN_MODULE

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MCP_ROOT = os.path.join(_REPO_ROOT, "mcp")


def _google_plugin(plugin_id: int = 1) -> Plugin:
    now = datetime.now(timezone.utc)
    return Plugin(
        id=plugin_id,
        slug="google",
        display_name="google",
        transport="stdio",
        args=("-m", "atlas_mcp.google"),
        url=None,
        enabled=True,
        builtin=True,
        enforces_policy=False,
        timeout_ms=5000,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )


async def _no_policy() -> "dict | None":
    return None


async def _empty_google_env(plugin: Plugin) -> "dict[str, str]":
    """No accounts linked -- enough for the real child to boot; this file
    proves schema/dispatch shape, not any real Google call."""
    return {"GOOGLE_ACCOUNTS_JSON": "{}"}


# --- schema hiding + lookup dispatch ----------------------------------------


async def test_code_only_tools_are_hidden_from_schema_but_still_reach_the_child(fake_plugin_repository):
    security = SecurityConfig()
    repo = fake_plugin_repository(plugins=[_google_plugin()], config_values={1: []})
    manager = PluginManager(
        repo,
        mcp_root=_MCP_ROOT,
        security=security,
        safety_block_provider=_no_policy,
        custom_env_builders={GOOGLE_PLUGIN_MODULE: _empty_google_env},
        hidden_tools_by_module={GOOGLE_PLUGIN_MODULE: CODE_ONLY_TOOL_NAMES},
    )
    try:
        await manager.start_all()

        offered_names = {entry["function"]["name"] for entry in manager.tools_schema}
        assert offered_names.isdisjoint(CODE_ONLY_TOOL_NAMES)
        assert "calendar_list_events" in offered_names
        assert "calendar_propose_event" in offered_names

        # Hidden from the schema, but the lookup still routes to the real
        # child -- reaching `handle_calendar_insert_event`'s own `Denied`
        # (no account linked) proves this call was actually dispatched,
        # not turned away by `McpToolHostLookup` itself as an unknown name.
        result = await manager.tool_host_lookup.call_tool(
            "calendar_insert_event",
            {
                "account": "home",
                "calendar_id": "cal-1",
                "title": "Standup",
                "start": "2026-10-02T09:00:00Z",
                "end": "2026-10-02T09:15:00Z",
                "all_day": False,
                "time_zone": "UTC",
            },
        )
        assert result.is_error
        assert "google account is linked" in result.content[0].text or "google account called" in result.content[0].text
    finally:
        await manager.stop_all()


# --- drift guard -------------------------------------------------------------


async def test_every_registered_google_tool_name_is_in_all_tool_names():
    from atlas_mcp import google as google_module

    listed = await google_module.mcp_server.list_tools()
    names = {tool.name for tool in listed}
    assert names, "the real child registered no tools at all"
    assert names <= ALL_TOOL_NAMES


# --- dispatch refusal, at the turn level ------------------------------------


class _RecordingToolHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])


async def test_a_bare_code_only_tool_call_is_never_dispatched(fake_audio_source, fake_stt, fake_brain, fake_tts):
    tool_host = _RecordingToolHost()
    brain = fake_brain(
        replies=[BrainReply(tool_calls=[ToolCall(name="calendar_insert_event", arguments={"account": "home"})])]
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="add the event")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source, stt, brain, tts, tool_host, tools_schema=[], system_prompt="x", max_tool_rounds=3, timings=timings
    )

    assert tool_host.calls == []
    assert tts.received_text == [CODE_ONLY_REFUSAL]


async def test_a_collision_prefixed_code_only_tool_call_is_never_dispatched(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    tool_host = _RecordingToolHost()
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[ToolCall(name="google__calendar_insert_event", arguments={"account": "home"})]
            )
        ]
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="add the event")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source, stt, brain, tts, tool_host, tools_schema=[], system_prompt="x", max_tool_rounds=3, timings=timings
    )

    assert tool_host.calls == []
    assert tts.received_text == [CODE_ONLY_REFUSAL]


async def test_the_manager_reports_the_google_plugins_offered_proposal_names(fake_plugin_repository):
    """R2-WR-04: the restricted turn's allowlist comes from the running
    Google plugin's own offered names, found by module, never by slug."""
    from atlas_mcp.google_tools import CALENDAR_PROPOSAL_TOOL_NAMES

    repo = fake_plugin_repository(plugins=[_google_plugin()], config_values={1: []})
    manager = PluginManager(
        repo,
        mcp_root=_MCP_ROOT,
        security=SecurityConfig(),
        safety_block_provider=_no_policy,
        custom_env_builders={GOOGLE_PLUGIN_MODULE: _empty_google_env},
        hidden_tools_by_module={GOOGLE_PLUGIN_MODULE: CODE_ONLY_TOOL_NAMES},
    )
    assert manager.offered_tool_names_for_module(GOOGLE_PLUGIN_MODULE, CALENDAR_PROPOSAL_TOOL_NAMES) == frozenset()
    try:
        await manager.start_all()
        assert (
            manager.offered_tool_names_for_module(GOOGLE_PLUGIN_MODULE, CALENDAR_PROPOSAL_TOOL_NAMES)
            == CALENDAR_PROPOSAL_TOOL_NAMES
        )
        assert manager.offered_tool_names_for_module("atlas_mcp.other", CALENDAR_PROPOSAL_TOOL_NAMES) == frozenset()
    finally:
        await manager.stop_all()
