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
from typing import Callable

import pytest
from mcp.types import Tool
from pydantic import ValidationError

from atlas.config import MacroActionConfig, MacroConfig, WorkflowConfig
from atlas.db.models import WorkflowStepRow
from atlas.db.repository import WorkflowStepSpec
from atlas.mcp_client import McpToolHostLookup, RenamedToolHostView
from atlas.plugins.naming import PluginTool, PluginTools, rename_collisions
from atlas.providers.base import FinalTranscript
from atlas.providers.tier_reply import FillerPhrase, TierReply
from atlas.routes.conflict import annotate_conflict
from atlas.timing import TurnTimings
from atlas.turn import brain_race
from atlas.turn.controller import run_turn
from atlas.turn.macros import fire_macro
from atlas.workflow.steps import _HA_CALL_SERVICE_TOOL, execute_step

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


# --- Task 2: a stored row naming a now-ambiguous tool is flagged, never ---
# --- rewritten, and refuses at fire time in both runners (D-12) ----------


def _owners(*slugs: str) -> "Callable[[str], tuple[str, ...]]":
    """A `tool_owners` callable that always answers `slugs`, standing in
    for `PluginManager.owners_of_bare_name` -- every caller under test
    here takes this exact shape, matching the real accessor's own
    signature (`str -> tuple[str, ...]`)."""

    def _fn(_bare_name: str) -> tuple[str, ...]:
        return slugs

    return _fn


def test_annotate_conflict_is_ok_for_a_tool_only_one_plugin_publishes():
    """Unchanged from before this plan: a tool name with exactly one
    owner is not ambiguous, and the entity-conflict check runs exactly as
    it always has (here: no target in `arguments` at all, so `ok`)."""
    annotation = annotate_conflict({}, None, None, tool_name="forecast", tool_owners=_owners("weather"))
    assert annotation == "ok"


def test_annotate_conflict_is_unknown_for_a_tool_two_plugins_now_publish():
    annotation = annotate_conflict(
        {}, None, None, tool_name="forecast", tool_owners=_owners("weather", "garden")
    )
    assert annotation == "unknown"


def test_annotate_conflict_with_no_tool_owners_given_is_unaffected():
    """A caller that predates this plan (`tool_name`/`tool_owners` both
    omitted) reaches the pre-existing entity-conflict logic unchanged --
    here, no target in `arguments`, so `ok`."""
    assert annotate_conflict({}, None, None) == "ok"


async def test_fire_macro_refuses_an_ambiguous_action_and_calls_no_tool_host():
    class _RecordingToolHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call_tool(self, name: str, arguments: dict):
            self.calls.append((name, arguments))
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])

    macro = MacroConfig(
        phrase="what's the forecast",
        aliases=(),
        reply="here you go",
        actions=(MacroActionConfig(tool="forecast", arguments={}),),
    )
    tool_host = _RecordingToolHost()

    outcome = await fire_macro(macro, tool_host, tool_owners=_owners("weather", "garden"))

    assert outcome.succeeded is False
    assert "forecast" in outcome.text
    assert tool_host.calls == []


async def test_fire_macro_with_no_ambiguity_still_calls_the_tool_host():
    class _RecordingToolHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call_tool(self, name: str, arguments: dict):
            self.calls.append((name, arguments))
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])

    macro = MacroConfig(
        phrase="what's the forecast",
        aliases=(),
        reply="here you go",
        actions=(MacroActionConfig(tool="forecast", arguments={}),),
    )
    tool_host = _RecordingToolHost()

    outcome = await fire_macro(macro, tool_host, tool_owners=_owners("weather"))

    assert outcome.succeeded is True
    assert tool_host.calls == [("forecast", {})]


def _make_call_service_step(arguments: dict) -> WorkflowStepRow:
    return WorkflowStepRow(
        id=1,
        run_id=1,
        position=0,
        kind="call_service",
        arguments=arguments,
        due_at=datetime(2026, 1, 1, 12, 0, 0),
        status="pending",
        attempts=0,
        result_detail=None,
        fired_at=None,
    )


class _RecordingWorkflowToolHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return SimpleNamespace(is_error=False, content=[], name=name)


async def test_execute_step_refuses_an_ambiguous_call_service_step_and_calls_no_tool_host():
    step = _make_call_service_step({"domain": "climate", "service": "set_temperature"})
    tool_host = _RecordingWorkflowToolHost()
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    outcome = await execute_step(
        step,
        tool_host,
        WorkflowConfig(),
        now,
        tool_owners=_owners("ha", "another_ha_lookalike"),
    )

    assert outcome.status == "denied"
    assert outcome.retry is False
    assert _HA_CALL_SERVICE_TOOL in (outcome.speech or "")
    assert tool_host.calls == []


async def test_execute_step_with_no_ambiguity_still_reaches_the_tool_host():
    step = _make_call_service_step({"domain": "climate", "service": "set_temperature"})
    tool_host = _RecordingWorkflowToolHost()
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    outcome = await execute_step(
        step,
        tool_host,
        WorkflowConfig(),
        now,
        tool_owners=_owners("ha"),
    )

    assert outcome.status == "completed"
    assert tool_host.calls == [(_HA_CALL_SERVICE_TOOL, {"domain": "climate", "service": "set_temperature"})]


async def test_a_wait_step_is_unaffected_by_tool_owners_since_it_calls_no_tool():
    step = WorkflowStepRow(
        id=1,
        run_id=1,
        position=0,
        kind="wait",
        arguments={},
        due_at=datetime(2026, 1, 1, 12, 0, 0),
        status="pending",
        attempts=0,
        result_detail=None,
        fired_at=None,
    )
    tool_host = _RecordingWorkflowToolHost()
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    outcome = await execute_step(
        step, tool_host, WorkflowConfig(), now, tool_owners=_owners("ha", "another_ha_lookalike")
    )

    assert outcome.status == "completed"
    assert tool_host.calls == []


async def test_installing_a_colliding_plugin_writes_nothing_to_macro_or_workflow_storage(
    fake_plugin_repository, fake_macro_repository, fake_workflow_repository
):
    """D-12's own must_have, pinned directly: `PluginManager` never takes
    a `MacroRepository` or a `WorkflowRepository` as a constructor
    argument at all, so a collision discovered by `rebuild()` -- driven
    here exactly the way `tests/test_plugin_manager.py`'s own two-plugin
    collision tests drive it, with no real subprocess -- cannot write to
    either repository. This test proves it empirically: a macro action and
    a workflow step naming the bare tool name that becomes contested are
    read back byte-identical after the collision is introduced."""
    from mcp.types import Tool

    from atlas.config import SecurityConfig
    from atlas.db.repository import Plugin
    from atlas.plugins.manager import PluginManager, PluginState, RunningPlugin

    async def _no_policy():
        return None

    class _FakeHost:
        def __init__(self, tool_specs: "list[tuple[str, str]]") -> None:
            self.tools = [
                Tool(name=name, description=description, inputSchema={"type": "object", "properties": {}})
                for name, description in tool_specs
            ]

        async def call_tool(self, name: str, arguments: dict):
            return SimpleNamespace(is_error=False, content=[], name=name)

    now = datetime.now(timezone.utc)
    ha_plugin = Plugin(
        id=1,
        slug="ha",
        display_name="ha",
        transport="stdio",
        args=("-m", "atlas_mcp.ha"),
        url=None,
        enabled=True,
        builtin=True,
        enforces_policy=False,
        timeout_ms=5000,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )
    other_plugin = Plugin(
        id=2,
        slug="lookalike",
        display_name="lookalike",
        transport="stdio",
        args=("-m", "atlas_mcp.weather"),
        url=None,
        enabled=True,
        builtin=False,
        enforces_policy=False,
        timeout_ms=5000,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )

    manager = PluginManager(
        fake_plugin_repository(plugins=[]),
        mcp_root=".",
        security=SecurityConfig(),
        safety_block_provider=_no_policy,
    )
    manager._plugins[1] = RunningPlugin(
        plugin=ha_plugin, host=_FakeHost([("ha_call_service", "call an HA service")]), state=PluginState.RUNNING
    )
    manager.rebuild()
    assert manager.owners_of_bare_name("ha_call_service") == ("ha",)

    macro_repo = fake_macro_repository(
        macros=[("good night", (), "good night", [("ha_call_service", {"domain": "light"})])]
    )
    workflow_repo = fake_workflow_repository()
    run = await workflow_repo.create_run(
        origin="webapp",
        summary="turn off the lights",
        steps=[WorkflowStepSpec(kind="call_service", arguments={"domain": "light"})],
        base_time=now,
        created_by_user_id=None,
    )
    before_macro_tools = [action.tool for macro in await macro_repo.list_macros() for action in macro.actions]
    before_step_kinds = [(s.kind, dict(s.arguments)) for s in (await workflow_repo.get_run(run.id)).steps]

    # A second plugin now also publishes `ha_call_service` -- the
    # collision `owners_of_bare_name` must report, with no write to
    # either repository as a side effect of discovering it.
    manager._plugins[2] = RunningPlugin(
        plugin=other_plugin,
        host=_FakeHost([("ha_call_service", "a lookalike service")]),
        state=PluginState.RUNNING,
    )
    manager.rebuild()
    assert len(manager.owners_of_bare_name("ha_call_service")) == 2

    after_macro_tools = [action.tool for macro in await macro_repo.list_macros() for action in macro.actions]
    after_step_kinds = [(s.kind, dict(s.arguments)) for s in (await workflow_repo.get_run(run.id)).steps]

    assert after_macro_tools == before_macro_tools == ["ha_call_service"]
    assert after_step_kinds == before_step_kinds == [("call_service", {"domain": "light"})]
