"""Proves the safety boundary's refusal reaches the spoken reply verbatim.

Turned green by plan 01-03. The first test here exercises the end-to-end
shape `mcp/atlas_mcp/ha.py`'s own docstring documents: an uncaught `Denied`
becomes an error-shaped MCP result whose text is `Denied.reason`, and the
turn controller speaks that text directly with no second brain call in
between.

Plan 01.1-05 extends this file rather than replacing it (CMD-08): a refusal
reaching the turn through the macro path must behave identically to one
reaching it through the tier path -- same verbatim text, same empty-reason
fallback, same absence of any length check, whichever entrance it came
through.
"""

from types import SimpleNamespace

from atlas_mcp.ha import handle_call_service
from atlas_mcp.safety import Denied, Policy

from atlas.config import MacroActionConfig, MacroConfig
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall


class _RefusingToolHost:
    """Calls straight into `handle_call_service` against `fake_ha`.

    Converts an uncaught `Denied` into the same error-shaped result the real
    MCP framework produces for an uncaught tool exception -- `is_error=True`,
    `content[0].text == str(exc)`, which for a `Denied` is exactly `reason`
    (see `ha.py`'s module docstring). No subprocess and no MCP wire protocol,
    matching `_FakeToolHost` in `tests/test_turn_controller.py`.
    """

    def __init__(self, ha, policy: Policy) -> None:
        self._ha = ha
        self._policy = policy

    async def call_tool(self, name: str, arguments: dict):
        assert name == "ha_call_service", f"unexpected tool: {name}"
        try:
            await handle_call_service(
                self._policy,
                self._ha.client,
                "http://ha.invalid",
                "test-token",
                **arguments,
            )
        except Denied as exc:
            return SimpleNamespace(is_error=True, content=[SimpleNamespace(text=str(exc))])
        raise AssertionError("call was allowed; this test needs a denied entity")


async def test_denied_reason_reaches_the_reply_verbatim(
    fake_audio_source, fake_stt, fake_brain, fake_tts, fake_ha
):
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the server socket")])
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="ha_call_service",
                        arguments={
                            "domain": "switch",
                            "service": "turn_off",
                            "entity_id": "switch.example_server_socket",
                        },
                    )
                ]
            ),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
    policy = Policy.from_config({"deny_entities": ["switch.example_server_socket"]})
    tool_host = _RefusingToolHost(fake_ha, policy)
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    # The reason string safety.py actually raises, not a substring or a
    # paraphrase.
    assert tts.received_text == ["that one is off limits"]
    # The refusal does not make a second round trip to the language model.
    assert brain.call_count == 1
    # Nothing left the process for the denied call.
    assert len(fake_ha.requests) == 0


async def test_denied_reason_reaches_the_reply_verbatim_through_the_macro_path(
    fake_audio_source, fake_stt, fake_brain, fake_tts, fake_ha
):
    """The macro-path twin of the test above (CMD-08): the exact same
    `Denied.reason` reaches speech through `fire_macro`, with no macro-
    specific bypass and no second brain call -- `_RefusingToolHost` is
    reused unmodified, because a macro action reaches `allow_call` through
    the identical `tool_host.call_tool` entry a model-issued call uses."""
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    macro = MacroConfig(
        phrase="good night",
        aliases=(),
        reply="good night",
        actions=(
            MacroActionConfig(
                tool="ha_call_service",
                arguments={
                    "domain": "switch",
                    "service": "turn_off",
                    "entity_id": "switch.example_server_socket",
                },
            ),
        ),
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="good night")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    policy = Policy.from_config({"deny_entities": ["switch.example_server_socket"]})
    tool_host = _RefusingToolHost(fake_ha, policy)
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        macros=(macro,),
    )

    assert tts.received_text == ["that one is off limits"]
    assert brain.call_count == 0
    assert len(fake_ha.requests) == 0
    assert timings.turn_outcome == "macro_failed"


class _ScriptedErrorToolHost:
    """Returns a fixed error-shaped result for every call, carrying
    whatever `reason` a test supplies.

    `safety.py`'s own reasons are short, fixed strings -- proving a reason
    "at least four times `brain.max_tokens`' default in characters" or
    carrying non-ASCII text needs a reason the real policy machinery cannot
    conveniently produce. This still exercises the exact error-shaped-result
    shape the real MCP boundary produces (`isError=True`,
    `content[0].text == reason`); it only replaces where the reason comes
    from, not the shape it arrives in.
    """

    def __init__(self, reason: str) -> None:
        self._reason = reason
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict):
        self.calls += 1
        return SimpleNamespace(isError=True, content=[SimpleNamespace(text=self._reason)])


# `brain.max_tokens` defaults to 400 (`BrainConfig.max_tokens`); "at least
# four times that in characters" is comfortably exceeded by 2,000.
_LONG_REASON = (
    "the entity you asked about is off limits because it controls power to "
    "equipment the operator has specifically excluded from voice control, "
    "and this explanation keeps going on at length to prove that nothing "
    "in the refusal path measures or truncates by length. "
) * 12


async def test_long_denied_reason_reaches_speech_whole_via_the_tier_path(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """A refusal never passes through a model, so no `max_tokens`
    truncation applies -- `Denied.reason` reaches text-to-speech whole, at
    any length."""
    from atlas.config import BrainConfig
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    assert len(_LONG_REASON) >= BrainConfig().max_tokens * 4

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the server socket")])
    brain = fake_brain(
        replies=[
            BrainReply(tool_calls=[ToolCall(name="ha_call_service", arguments={})]),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
    tool_host = _ScriptedErrorToolHost(_LONG_REASON)
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert tts.received_text == [_LONG_REASON]
    assert brain.call_count == 1


async def test_long_denied_reason_reaches_speech_whole_via_the_macro_path(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """The tier path and the macro path behave identically for the same
    over-length reason -- neither entrance truncates."""
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    macro = MacroConfig(
        phrase="good night",
        aliases=(),
        reply="good night",
        actions=(MacroActionConfig(tool="ha_call_service", arguments={}),),
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="good night")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    tool_host = _ScriptedErrorToolHost(_LONG_REASON)
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        macros=(macro,),
    )

    assert tts.received_text == [_LONG_REASON]
    assert brain.call_count == 0


async def test_empty_denied_reason_speaks_the_fallback_via_the_tier_path(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """A `Denied` whose reason is empty speaks a fixed fallback sentence
    rather than silence, so a refused command is never indistinguishable
    from a dropped turn."""
    from atlas.timing import TurnTimings
    from atlas.turn.controller import _DENIED_FALLBACK_REPLY, run_turn

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the server socket")])
    brain = fake_brain(
        replies=[
            BrainReply(tool_calls=[ToolCall(name="ha_call_service", arguments={})]),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
    tool_host = _ScriptedErrorToolHost("")
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert tts.received_text == [_DENIED_FALLBACK_REPLY]
    assert _DENIED_FALLBACK_REPLY != ""


async def test_empty_denied_reason_speaks_the_fallback_via_the_macro_path(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """The macro path's empty-reason case uses the identical fallback the
    tier path uses -- `_DENIED_FALLBACK_REPLY` is the one constant both call
    sites reach for, not two independently-worded sentences."""
    from atlas.timing import TurnTimings
    from atlas.turn.controller import _DENIED_FALLBACK_REPLY, run_turn

    macro = MacroConfig(
        phrase="good night",
        aliases=(),
        reply="good night",
        actions=(MacroActionConfig(tool="ha_call_service", arguments={}),),
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="good night")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    tool_host = _ScriptedErrorToolHost("")
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        macros=(macro,),
    )

    assert tts.received_text == [_DENIED_FALLBACK_REPLY]


async def test_non_ascii_denied_reason_reaches_speech_unchanged(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """A reason containing non-ASCII characters reaches text-to-speech
    unchanged -- nothing in the macro or refusal path measures or
    truncates by byte length. Exact equality, not a normalized or slugged
    comparison: a non-ASCII character surviving a lossy round trip as a
    replacement character would still pass a normalized comparison."""
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    reason = "そのスイッチは操作できません -- ceci est refusé -- ☃"
    macro = MacroConfig(
        phrase="good night",
        aliases=(),
        reply="good night",
        actions=(MacroActionConfig(tool="ha_call_service", arguments={}),),
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="good night")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    tool_host = _ScriptedErrorToolHost(reason)
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        macros=(macro,),
    )

    assert tts.received_text == [reason]
    assert tts.received_text[0] == reason  # byte-for-byte, not a normalized form
