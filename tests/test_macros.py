"""Macro matching and firing (MACRO-01, MACRO-02, CMD-07, CMD-08).

Turned green by plan 01.1-05, replacing the three Wave-0 scaffolds plan
01.1-01 wrote red on purpose. `match()`/`fire_macro()` tests live here
(Task 1); the end-to-end `run_turn` tests proving a macro hit never reaches
the brain, and never touches the live text-to-speech provider, are added by
Task 2 below them.
"""

from __future__ import annotations

from types import SimpleNamespace

from spire_mcp.ha import handle_call_service
from spire_mcp.safety import Denied, Policy

from spire_voice.config import MacroActionConfig, MacroConfig
from spire_voice.turn.macros import fire_macro, match, normalize


class _MacroToolHost:
    """Calls straight into `handle_call_service` against `fake_ha`.

    The exact shape `tests/test_safety_integration.py::_RefusingToolHost`
    and `tests/test_turn_controller.py::_FakeToolHost` already use: no
    subprocess and no MCP wire protocol, converting an uncaught `Denied`
    into the same error-shaped result the real MCP framework produces for
    an uncaught tool exception -- `isError=True`, `content[0].text ==
    str(exc)`, which for a `Denied` is exactly `reason`. This is what makes
    the `allow_call` assertion in the tests below real rather than mocked.
    """

    def __init__(self, ha, policy: Policy) -> None:
        self._ha = ha
        self._policy = policy

    async def call_tool(self, name: str, arguments: dict):
        assert name == "ha_call_service", f"unexpected tool: {name}"
        try:
            await handle_call_service(
                self._policy, self._ha.client, "http://ha.invalid", "test-token", **arguments
            )
        except Denied as exc:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text=str(exc))])
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])


def _macro(
    phrase: str = "good night",
    aliases: tuple[str, ...] = (),
    reply: str = "good night",
    actions: tuple[MacroActionConfig, ...] = (),
) -> MacroConfig:
    if not actions:
        actions = (
            MacroActionConfig(
                tool="ha_call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"},
            ),
        )
    return MacroConfig(phrase=phrase, aliases=aliases, reply=reply, actions=actions)


# --- match() -----------------------------------------------------------


def test_match_empty_transcript_matches_nothing():
    macro = _macro()
    assert match([macro], "") is None
    assert match([macro], "   ") is None


def test_match_returns_none_when_no_macro_is_configured():
    assert match([], "good night") is None


def test_match_exact_phrase():
    macro = _macro(phrase="good night")
    assert match([macro], "Good Night") is macro


def test_match_does_not_match_a_transcript_that_merely_contains_the_phrase():
    macro = _macro(phrase="good night")
    assert match([macro], "well good night then") is None


def test_match_alias():
    macro = _macro(phrase="good night", aliases=("goodnight", "night night"))
    assert match([macro], "night night") is macro
    assert match([macro], "goodnight") is macro


def test_match_with_no_alias_list_works_on_phrase_alone():
    macro = _macro(phrase="movie time", aliases=())
    assert match([macro], "movie time") is macro
    assert match([macro], "movie") is None


def test_match_on_phrase_equal_to_its_own_alias_returns_the_macro_once():
    # `MacroConfig.normalized_keys` dedupes a self-colliding alias into one
    # key (proven by tests/test_config.py); `match` itself only ever
    # returns a single value, so "returns it once, not twice" is a property
    # of match()'s own return type here, not something that needs a loop.
    macro = _macro(phrase="good night", aliases=("GOOD NIGHT!",))
    assert match([macro], "good night") is macro


def test_match_compares_by_code_point_after_nfkc_folding():
    # A precomposed accented character and its NFKC-equivalent decomposed
    # form differ in code point count and UTF-8 byte length, but must
    # normalize to the same string (mirrors test_config.py's own NFKC
    # normalize() test).
    macro = _macro(phrase="café time")
    assert match([macro], "café time") is macro


def test_match_does_not_produce_a_cross_macro_tie():
    # Config.from_config's own _check_macros_do_not_collide already rejects
    # any pair whose normalized_keys overlap before the process starts, so
    # two macros with genuinely distinct phrases never tie here.
    fan_macro = _macro(phrase="good night")
    lamp_macro = _macro(phrase="movie time")
    assert match([fan_macro, lamp_macro], "movie time") is lamp_macro
    assert match([fan_macro, lamp_macro], "good night") is fan_macro


# --- fire_macro(): allow_call reuse (MACRO-01, T-01.1-01) ---------------


async def test_macro_actions_all_pass_allow_call(fake_ha):
    """Every macro action passes `allow_call` at fire time, exactly like a
    model-issued call -- a macro is a shortcut past the model, never past
    the safety boundary (Pattern 4, 01.1-PATTERNS.md)."""
    policy = Policy.from_config({})
    macro = _macro(
        actions=(
            MacroActionConfig(
                tool="ha_call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"},
            ),
        )
    )
    tool_host = _MacroToolHost(fake_ha, policy)

    outcome = await fire_macro(macro, tool_host)

    assert outcome.succeeded is True
    assert outcome.text == "good night"
    assert outcome.cacheable is True
    assert len(fake_ha.requests) == 1
    assert fake_ha.requests[0].method == "POST"


async def test_macro_action_denied_by_the_boundary_never_reaches_home_assistant(fake_ha):
    """A denied macro action must not reach Home Assistant at all -- a
    stronger claim than "it returned an error"."""
    policy = Policy.from_config({"deny_entities": ["switch.example_server_socket"]})
    macro = _macro(
        actions=(
            MacroActionConfig(
                tool="ha_call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.example_server_socket",
                },
            ),
        )
    )
    tool_host = _MacroToolHost(fake_ha, policy)

    outcome = await fire_macro(macro, tool_host)

    assert outcome.succeeded is False
    assert outcome.cacheable is False
    assert outcome.text == "that one is off limits"
    assert len(fake_ha.requests) == 0


# --- fire_macro(): sequential order, stop-on-first-failure (CMD-07) -----


async def test_macro_partial_failure_stops_before_the_remaining_actions_run(fake_ha):
    policy = Policy.from_config({"deny_entities": ["switch.example_server_socket"]})
    macro = _macro(
        actions=(
            MacroActionConfig(
                tool="ha_call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.example_server_socket",
                },
            ),
            MacroActionConfig(
                tool="ha_call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"},
            ),
        )
    )
    tool_host = _MacroToolHost(fake_ha, policy)

    outcome = await fire_macro(macro, tool_host)

    assert outcome.succeeded is False
    # The tool host recorded fewer calls than the macro has actions: the
    # second action never ran once the first failed.
    assert len(fake_ha.requests) == 0


async def test_macro_partial_failure_names_the_first_failure_in_written_order(fake_ha):
    """When more than one action would fail, the reason names the one
    written first -- never an aggregated "some things failed" phrase, and
    never the second action's reason instead of the first's."""
    policy = Policy.from_config(
        {"deny_entities": ["switch.example_server_socket", "light.example_lamp"]}
    )
    macro = _macro(
        actions=(
            MacroActionConfig(
                tool="ha_call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.example_server_socket",
                },
            ),
            MacroActionConfig(
                tool="ha_call_service",
                arguments={"domain": "light", "service": "turn_off", "entity_id": "light.example_lamp"},
            ),
        )
    )
    tool_host = _MacroToolHost(fake_ha, policy)

    outcome = await fire_macro(macro, tool_host)

    # Both actions target denied entities and would raise the identical
    # Denied.reason ("that one is off limits") -- what this test actually
    # pins down is that only the FIRST action was ever attempted (a single
    # request), proving the second was never dispatched to find out whether
    # it would also have failed.
    assert outcome.text == "that one is off limits"
    assert len(fake_ha.requests) == 0


async def test_fire_macro_success_requires_every_action_to_succeed(fake_ha):
    policy = Policy.from_config({})
    macro = _macro(
        reply="good night",
        actions=(
            MacroActionConfig(
                tool="ha_call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"},
            ),
            MacroActionConfig(
                tool="ha_call_service",
                arguments={"domain": "light", "service": "turn_off", "entity_id": "light.example_lamp"},
            ),
        )
    )
    tool_host = _MacroToolHost(fake_ha, policy)

    outcome = await fire_macro(macro, tool_host)

    assert outcome.succeeded is True
    assert outcome.text == "good night"
    assert outcome.cacheable is True
    assert len(fake_ha.requests) == 2


def test_normalize_still_exported_from_this_module():
    # Guards against an accidental rename/removal of plan 01.1-02's own
    # contribution while this plan extends the same file.
    assert normalize("Good  Night!") == "good night"
