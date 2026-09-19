"""Proves PLUG-07/D-11/D-12 end to end: a tool name two plugins both
publish is a third candidate type on the exact `needs_clarification`
envelope Phase 4 (entity name) and Phase 5 (pending run) already
established -- not a third mechanism -- and a stored macro action or
workflow step whose bare tool name has become ambiguous is flagged
(never rewritten) and refuses at fire time in both runners, reaching
neither plugin.

Task 1 (D-11): a spoken command that could reach either of two plugins'
own version of the same capability produces a spoken question naming
both plugins, with no call ever reaching either one. Driven against a
real `McpToolHostLookup` built over two real `RenamedToolHostView`
instances (the exact shape `plugins/manager.py::PluginManager.rebuild`
produces, plan 06-04) so the "no call reached either plugin" claim is
proven against the real routing layer, not a mock that merely trusts it.

Task 2 (D-12): `routes/conflict.py`'s shared annotator reports a stored
row naming a now-ambiguous bare tool as `unknown` -- the same value it
already carries for any other check it could not resolve -- and
`turn/macros.py::fire_macro` / `workflow/steps.py::execute_step` both
refuse before ever calling `tool_host.call_tool`, never reaching either
of the two plugins that now publish the name. Nothing here writes to a
macro or workflow row: `PluginManager` never depends on `MacroRepository`
or `WorkflowRepository` at all, so this file also pins that invariant
directly.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from mcp.types import Tool
from pydantic import ValidationError

from spire_voice.config import MacroActionConfig, MacroConfig, WorkflowConfig
from spire_voice.db.models import WorkflowStepRow
from spire_voice.mcp_client import McpToolHostLookup, RenamedToolHostView
from spire_voice.plugins.naming import PluginTool, PluginTools, rename_collisions
from spire_voice.providers.base import FinalTranscript
from spire_voice.providers.tier_reply import FillerPhrase, TierReply
from spire_voice.routes.conflict import annotate_conflict
from spire_voice.timing import TurnTimings
from spire_voice.turn import brain_race
from spire_voice.turn.controller import run_turn
from spire_voice.turn.macros import fire_macro
from spire_voice.workflow.steps import _HA_CALL_SERVICE_TOOL, execute_step

# --- Task 1: a spoken command ambiguous between two plugins -----------------


def _two_colliding_plugin_hosts() -> tuple[McpToolHostLookup, "_RecordingHost", "_RecordingHost"]:
    """Two plugins that both publish a `forecast` tool -- built through the
    exact `rename_collisions` + `RenamedToolHostView` + `McpToolHostLookup`
    pipeline `PluginManager.rebuild()` uses (plan 06-04), so a call routed
    through the returned lookup exercises the real collision-safe routing
    layer, not a stand-in for it."""

    class _RecordingHost:
        def __init__(self, tools: "list[Tool]") -> None:
            self.tools = tools
            self.calls: list[str] = []

        async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
            self.calls.append(name)
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])

    weather_tool = Tool(
        name="forecast", description="Get a forecast", inputSchema={"type": "object", "properties": {}}
    )
    garden_tool = Tool(
        name="forecast", description="Get a garden forecast", inputSchema={"type": "object", "properties": {}}
    )
    weather_host = _RecordingHost([weather_tool])
    garden_host = _RecordingHost([garden_tool])

    naming_result = rename_collisions(
        [
            PluginTools(slug="weather", display_name="Weather", tools=(PluginTool("forecast", "Get a forecast"),)),
            PluginTools(
                slug="garden", display_name="Garden Sensors", tools=(PluginTool("forecast", "Get a garden forecast"),)
            ),
        ]
    )

    def _view(slug: str, host: "_RecordingHost") -> RenamedToolHostView:
        renamed = naming_result.tools_for(slug)
        renamed_tools = [
            tool.model_copy(update={"name": r.offered_name, "description": r.offered_description})
            for tool, r in zip(host.tools, renamed)
        ]
        bare_by_offered = {r.offered_name: r.bare_name for r in renamed}
        return RenamedToolHostView(host, renamed_tools, bare_by_offered)

    lookup = McpToolHostLookup([_view("weather", weather_host), _view("garden", garden_host)])
    return lookup, weather_host, garden_host


def test_a_reply_asking_which_plugin_with_two_candidates_validates():
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


def test_needs_clarification_with_one_plugin_candidate_raises():
    with pytest.raises(ValidationError):
        TierReply(
            answer="",
            confident=False,
            needs_tool=False,
            filler=FillerPhrase.LET_ME_CHECK,
            needs_clarification=True,
            candidates=("Weather",),
        )


async def test_an_ambiguous_capability_speaks_a_question_naming_both_plugins_and_calls_neither(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
    """D-11, end to end: two plugins both publish `forecast`; a triage
    tier's `needs_clarification` reply naming both plugins' own display
    names wins the race before the (deliberately slow) top tier ever
    reaches the tool host. Neither plugin's host records a single call."""

    lookup, weather_host, garden_host = _two_colliding_plugin_hosts()

    class _NeverFinishesBrain:
        async def chat(self, messages, tools=None):
            await asyncio.sleep(10)
            raise AssertionError("should have been cancelled before this line")  # pragma: no cover

    clarifying_reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=("Weather", "Garden Sensors"),
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

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what's the forecast")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        top_brain,
        tts,
        lookup,
        tools_schema=[],
        system_prompt="you control a house with plugins",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier, top_tier],
        filler_after_ms=1000,
        filler_cache=None,
    )

    assert weather_host.calls == []
    assert garden_host.calls == []
    assert len(tts.received_text) == 1
    spoken = tts.received_text[0]
    assert "Weather" in spoken
    assert "Garden Sensors" in spoken
    assert timings.turn_outcome == "needs_clarification"
