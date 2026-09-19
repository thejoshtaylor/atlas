"""`plugins.naming` -- the collision-only tool-name prefixing pre-pass
(D-09, PLUG-07).

Every test here builds `PluginTools`/`PluginTool` by hand and calls
`rename_collisions` directly -- no session, no repository, no started
plugin -- proving the whole module as a pure table of inputs and outputs,
per its own module docstring.
"""

from __future__ import annotations

import pytest
from mcp.types import Tool

from spire_voice.mcp_client import AmbiguousToolError, McpToolHostLookup
from spire_voice.plugins.naming import (
    NAME_SEPARATOR,
    PluginTool,
    PluginTools,
    rename_collisions,
)


def _plugin(slug: str, display_name: str, tool_names: "list[str]") -> PluginTools:
    return PluginTools(
        slug=slug,
        display_name=display_name,
        tools=tuple(PluginTool(name=name, description=f"{name} description") for name in tool_names),
    )


# --- Task 1: only the contested names move --------------------------------


def test_an_uncontested_name_is_returned_byte_identical():
    result = rename_collisions([_plugin("weather", "Weather", ["forecast", "current"])])

    tools = result.tools_for("weather")
    assert {tool.offered_name for tool in tools} == {"forecast", "current"}
    for tool in tools:
        assert tool.offered_name == tool.bare_name
        assert tool.prefixed is False


def test_only_the_one_contested_name_is_rewritten_on_both_owners_everything_else_unchanged():
    ha = _plugin("ha", "Home Assistant", ["list_entities", "notify"])
    weather = _plugin("weather", "Weather", ["notify", "forecast"])

    result = rename_collisions([ha, weather])

    ha_names = {tool.offered_name for tool in result.tools_for("ha")}
    weather_names = {tool.offered_name for tool in result.tools_for("weather")}

    # The uncontested names on each side survive bare.
    assert "list_entities" in ha_names
    assert "forecast" in weather_names

    # Only "notify" -- the one name both plugins publish -- was rewritten,
    # and it was rewritten on *both* owners.
    assert "notify" not in ha_names
    assert "notify" not in weather_names
    assert f"ha{NAME_SEPARATOR}notify" in ha_names
    assert f"weather{NAME_SEPARATOR}notify" in weather_names


def test_no_collisions_at_all_returns_every_name_unchanged():
    plugins = [
        _plugin("ha", "Home Assistant", ["list_entities", "call_service"]),
        _plugin("weather", "Weather", ["forecast", "current"]),
        _plugin("workflow", "Workflow", ["schedule_workflow"]),
    ]

    result = rename_collisions(plugins)

    for plugin in plugins:
        for tool in plugin.tools:
            renamed = [t for t in result.tools_for(plugin.slug) if t.bare_name == tool.name][0]
            assert renamed.offered_name == tool.name
            assert renamed.offered_description == tool.description
            assert renamed.prefixed is False


def test_three_plugins_publishing_the_same_name_produce_three_distinct_prefixed_names():
    plugins = [
        _plugin("alpha", "Alpha", ["notify"]),
        _plugin("bravo", "Bravo", ["notify"]),
        _plugin("charlie", "Charlie", ["notify"]),
    ]

    result = rename_collisions(plugins)

    offered_names = {result.tools_for(plugin.slug)[0].offered_name for plugin in plugins}
    assert offered_names == {
        f"alpha{NAME_SEPARATOR}notify",
        f"bravo{NAME_SEPARATOR}notify",
        f"charlie{NAME_SEPARATOR}notify",
    }
    assert len(offered_names) == 3


def test_the_result_never_contains_a_duplicate_offered_name():
    plugins = [
        _plugin("ha", "Home Assistant", ["list_entities", "notify", "shared_two"]),
        _plugin("weather", "Weather", ["forecast", "notify", "shared_two"]),
        _plugin("workflow", "Workflow", ["schedule_workflow", "shared_two"]),
    ]

    result = rename_collisions(plugins)

    all_offered_names = [
        tool.offered_name for plugin in plugins for tool in result.tools_for(plugin.slug)
    ]
    assert len(all_offered_names) == len(set(all_offered_names))


def test_owner_of_offered_name_answers_for_both_bare_and_prefixed_names():
    ha = _plugin("ha", "Home Assistant", ["list_entities", "notify"])
    weather = _plugin("weather", "Weather", ["notify"])

    result = rename_collisions([ha, weather])

    # Bare, uncontested name.
    assert result.owner_of_offered_name("list_entities") == "ha"
    # Prefixed, contested names.
    assert result.owner_of_offered_name(f"ha{NAME_SEPARATOR}notify") == "ha"
    assert result.owner_of_offered_name(f"weather{NAME_SEPARATOR}notify") == "weather"
    # A name nobody offers at all.
    assert result.owner_of_offered_name("does_not_exist") is None


def test_owners_of_bare_name_reports_every_publishing_plugin():
    ha = _plugin("ha", "Home Assistant", ["list_entities", "notify"])
    weather = _plugin("weather", "Weather", ["notify"])

    result = rename_collisions([ha, weather])

    assert set(result.owners_of_bare_name("notify")) == {"ha", "weather"}
    assert result.owners_of_bare_name("list_entities") == ("ha",)
    assert result.owners_of_bare_name("does_not_exist") == ()


def test_a_plugin_advertising_the_same_bare_name_twice_counts_as_one_owner_not_two():
    """This module decides prefixing from *how many plugins*, not how many
    tools, publish a bare name -- a plugin's own duplicate advertisement
    (a shape this module has no opinion on) must not itself trigger
    prefixing against a single other, unrelated plugin."""
    solo_with_dupe = _plugin("solo", "Solo", ["notify", "notify"])

    result = rename_collisions([solo_with_dupe])

    for tool in result.tools_for("solo"):
        assert tool.offered_name == "notify"
        assert tool.prefixed is False
    assert result.owners_of_bare_name("notify") == ("solo",)


# --- Task 2: the invariant this pre-pass exists in front of, unchanged ----


class _RawToolHost:
    """A bare host exposing only `.tools` -- exactly the shape
    `McpToolHostLookup.__init__` reads, with no pre-pass run over it.
    Standing in for `McpToolHost` itself, which this test has no reason
    to spawn for real."""

    def __init__(self, tool_specs: "list[str]") -> None:
        self.tools = [
            Tool(name=name, description="", inputSchema={"type": "object", "properties": {}})
            for name in tool_specs
        ]


def test_constructing_a_lookup_directly_over_two_hosts_sharing_a_name_still_raises_unchanged():
    """`McpToolHostLookup`'s own construction-time refusal is the
    invariant this phase's `06-CONTEXT.md` says to keep, not delete --
    proven here by skipping this module's own pre-pass entirely and
    handing the lookup two raw, colliding hosts directly. Collision
    safety for the real assistant comes from `plugins/manager.py::
    PluginManager.rebuild` always running the pre-pass first (Pitfall 6),
    never from a change to this constructor."""
    host_a = _RawToolHost(["notify"])
    host_b = _RawToolHost(["notify"])

    with pytest.raises(AmbiguousToolError):
        McpToolHostLookup([host_a, host_b])


# --- Task 3: the model is told who owns what, in the field it reads ------


def test_a_prefixed_tools_description_names_its_owner_in_front_of_the_original_text():
    ha = _plugin("ha", "Home Assistant", ["notify"])
    weather = _plugin("weather", "Weather", ["notify"])

    result = rename_collisions([ha, weather])

    ha_notify = result.tools_for("ha")[0]
    weather_notify = result.tools_for("weather")[0]
    assert ha_notify.offered_description.startswith("[Home Assistant]")
    assert "notify description" in ha_notify.offered_description
    assert weather_notify.offered_description.startswith("[Weather]")
    assert "notify description" in weather_notify.offered_description


def test_an_uncontested_tools_description_is_unchanged():
    weather = _plugin("weather", "Weather", ["forecast"])

    result = rename_collisions([weather])

    tool = result.tools_for("weather")[0]
    assert tool.offered_description == "forecast description"


def test_ownership_prompt_carries_one_line_per_prefixed_tool_and_nothing_for_uncontested_ones():
    ha = _plugin("ha", "Home Assistant", ["list_entities", "notify"])
    weather = _plugin("weather", "Weather", ["notify", "forecast"])

    result = rename_collisions([ha, weather])

    assert "list_entities" not in result.ownership_prompt
    assert "forecast" not in result.ownership_prompt
    assert f"ha{NAME_SEPARATOR}notify" in result.ownership_prompt
    assert "Home Assistant" in result.ownership_prompt
    assert f"weather{NAME_SEPARATOR}notify" in result.ownership_prompt
    assert "Weather" in result.ownership_prompt
    # Exactly two lines -- one per prefixed tool, no more.
    assert len([line for line in result.ownership_prompt.splitlines() if line.strip()]) == 2


def test_ownership_prompt_is_empty_when_nothing_collides():
    plugins = [
        _plugin("ha", "Home Assistant", ["list_entities"]),
        _plugin("weather", "Weather", ["forecast"]),
    ]

    result = rename_collisions(plugins)

    assert result.ownership_prompt == ""


def test_two_rebuilds_with_no_plugin_change_produce_the_same_ownership_prompt_bytes():
    plugins = [
        _plugin("ha", "Home Assistant", ["list_entities", "notify"]),
        _plugin("weather", "Weather", ["notify", "forecast"]),
    ]

    first = rename_collisions(plugins)
    second = rename_collisions(plugins)

    assert first.ownership_prompt == second.ownership_prompt
