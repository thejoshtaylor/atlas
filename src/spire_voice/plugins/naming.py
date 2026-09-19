"""Collision-only tool-name prefixing pre-pass (D-09, PLUG-07).

`McpToolHostLookup.__init__` (`mcp_client.py`) raises `AmbiguousToolError`
at construction the moment two hosts advertise the same tool name -- right
when the plugin list was fixed at three, wrong the instant an admin can
install a fourth. This module is the pre-pass that runs before that
construction: only a bare tool name two or more plugins actually publish
is rewritten, on every one of its owners, as `{slug}__{name}` -- a name
exactly one plugin publishes is returned byte-identical to its input,
because every stored macro action and every workflow step already names a
bare tool, and prefixing everything would rewrite every one of those rows
(D-09's own reasoning, `06-CONTEXT.md`).

`NAME_SEPARATOR` is two characters and, by convention, never appears in a
real MCP tool name -- this is an assumption this module leans on, not an
enforced invariant (`06-RESEARCH.md` Pitfall 4: the provider API's own
name length/character-set limit could not be confirmed from a first-party
source this session, so no truncation logic is built against an
unverified number). As long as that convention holds, a prefixed name can
never collide with a bare, uncontested name from a different plugin; if
it somehow did, `McpToolHostLookup`'s own construction-time
`AmbiguousToolError` -- the invariant this phase's `06-CONTEXT.md` says to
keep, not delete -- is still there to catch it, one layer up, since
`plugins/manager.py::rebuild` still builds that lookup over this module's
own output.

Pure transform: no session, no repository, no application state. That is
what makes every one of this module's own behaviours testable as a table
of inputs and outputs (Task 1's own instruction, plan 06-04).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

# Two characters, chosen for brevity per `06-CONTEXT.md`'s "Claude's
# Discretion" -- keep it short rather than build truncation logic against
# an unverified downstream name-length limit (`06-RESEARCH.md` Pitfall 4).
NAME_SEPARATOR = "__"


@dataclass(frozen=True)
class PluginTool:
    """One tool exactly as a started plugin advertised it -- the bare
    name and description straight off `McpToolHost.tools`, before this
    module's pre-pass runs. Deliberately duck-typed to two plain fields,
    not `mcp.types.Tool` itself, so this module takes no dependency on
    the MCP SDK and stays a pure table of inputs and outputs.
    """

    name: str
    description: str


@dataclass(frozen=True)
class PluginTools:
    """One started plugin's own identity and the tools it advertises --
    everything the pre-pass needs to know about a plugin, with no
    dependency on `Plugin`/`McpToolHost` themselves."""

    slug: str
    display_name: str
    tools: "Sequence[PluginTool]"


@dataclass(frozen=True)
class RenamedTool:
    """One tool as it is offered to the model after the pre-pass.

    `offered_name`/`offered_description` are what actually reaches the
    schema; `bare_name` is what the owning plugin actually advertised and
    the name a call must carry when it reaches that plugin -- the plugin
    never learns it was renamed (`prefixed` is `False` for every tool
    whose `offered_name == bare_name`).
    """

    offered_name: str
    offered_description: str
    bare_name: str
    owner_slug: str
    prefixed: bool


class NamingResult:
    """The whole pre-pass outcome: each plugin's own renamed tool list, in
    that plugin's own tool order, plus the two owner lookups the rest of
    this phase needs (plan 06-04's own `key_links` -- plan 06-05's
    stored-row flag and plan 06-06's routes both read
    `owners_of_bare_name`).

    A plain class, not a `dataclass`, so the two lookup methods below are
    real accessors over private dict state rather than public fields a
    caller could reassign.
    """

    def __init__(
        self,
        by_plugin: "Mapping[str, tuple[RenamedTool, ...]]",
        owner_of_offered_name: "Mapping[str, str]",
        owners_of_bare_name: "Mapping[str, tuple[str, ...]]",
    ) -> None:
        self._by_plugin = dict(by_plugin)
        self._owner_of_offered_name = dict(owner_of_offered_name)
        self._owners_of_bare_name = dict(owners_of_bare_name)

    def tools_for(self, slug: str) -> "tuple[RenamedTool, ...]":
        """Every tool `slug` offers post-pre-pass, in its own original
        order -- `()` for a slug this result was never built over."""
        return self._by_plugin.get(slug, ())

    def owner_of_offered_name(self, offered_name: str) -> "str | None":
        """Which plugin's slug owns `offered_name` -- answers for both a
        bare, uncontested name and a `{slug}__{name}` prefixed one.
        `None` when no plugin in this result offers that name at all."""
        return self._owner_of_offered_name.get(offered_name)

    def owners_of_bare_name(self, bare_name: str) -> "tuple[str, ...]":
        """Every plugin slug that publishes `bare_name`, regardless of
        whether the pre-pass ended up prefixing it -- a length of 2 or
        more is exactly what makes a bare name ambiguous (plan 06-05).
        `()` when no plugin in this result publishes that bare name."""
        return self._owners_of_bare_name.get(bare_name, ())


def rename_collisions(plugins: "Sequence[PluginTools]") -> NamingResult:
    """The pre-pass itself: rewrite only a bare name two or more plugins
    publish, on every one of its owners, as `{slug}{NAME_SEPARATOR}{name}`
    -- every other name (and every uncontested plugin's whole tool list)
    is returned byte-identical to its input.
    """
    bare_name_counts: "Counter[str]" = Counter()
    owners_of_bare_name: "defaultdict[str, list[str]]" = defaultdict(list)
    for plugin in plugins:
        # A `set` guards against one plugin advertising the same bare
        # name twice in its own list counting itself twice here -- this
        # module has no opinion about whether that shape is itself a
        # defect; it only decides prefixing from *how many plugins*, not
        # how many tools, publish a name.
        for name in {tool.name for tool in plugin.tools}:
            bare_name_counts[name] += 1
            owners_of_bare_name[name].append(plugin.slug)

    by_plugin: "dict[str, tuple[RenamedTool, ...]]" = {}
    owner_of_offered_name: "dict[str, str]" = {}
    for plugin in plugins:
        renamed_tools: list[RenamedTool] = []
        for tool in plugin.tools:
            contested = bare_name_counts[tool.name] > 1
            offered_name = f"{plugin.slug}{NAME_SEPARATOR}{tool.name}" if contested else tool.name
            renamed_tools.append(
                RenamedTool(
                    offered_name=offered_name,
                    offered_description=tool.description,
                    bare_name=tool.name,
                    owner_slug=plugin.slug,
                    prefixed=contested,
                )
            )
            owner_of_offered_name[offered_name] = plugin.slug
        by_plugin[plugin.slug] = tuple(renamed_tools)

    return NamingResult(
        by_plugin=by_plugin,
        owner_of_offered_name=owner_of_offered_name,
        owners_of_bare_name={name: tuple(slugs) for name, slugs in owners_of_bare_name.items()},
    )
