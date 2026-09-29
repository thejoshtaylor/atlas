"""D-13, D-15, D-16 (12-08-PLAN.md, task 1): a refused claim is the refused
turn's spoken reply on every command path, and a macro stays outside claims.

Two grouped `run_turn` calls share one `ClaimRegistry`. Each wraps one fake
Home Assistant host in its own `ClaimingToolHost`. The second turn's speech-to-
text waits a short time, so the first turn's write always lands first.

Every id, name, and phrase here is a generic placeholder.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from atlas.config import MacroActionConfig, MacroConfig
from atlas.providers.base import BrainReply, ToolCall
from atlas.turn.entity_claims import ClaimingToolHost, ClaimRegistry
from atlas.turn.reply_group import GroupSpeaker
from tests.test_turn_group_speech import _Turn, _run_both

_LAMP = {"domain": "light", "service": "turn_on", "entity_id": "light.example_lamp"}
_STATES = [{"entity_id": "light.example_lamp", "friendly_name": "lamp", "state": "off"}]


class _FakeHomeAssistant:
    """Records each `ha_call_service` call and answers with success."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        return SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(text='{"changed": []}')],
            structured_content={"changed": []},
        )


class _DelayedStt:
    """Waits, then plays the wrapped fake's scripted events."""

    def __init__(self, inner, delay_s: float) -> None:
        self._inner = inner
        self._delay_s = delay_s

    async def stream(self, frames, source_format=None, **_kwargs):
        await asyncio.sleep(self._delay_s)
        async for event in self._inner.stream(frames, source_format):
            yield event


async def _state_fetch():
    return list(_STATES)


def _claiming(inner: _FakeHomeAssistant, registry: ClaimRegistry, turn: _Turn) -> ClaimingToolHost:
    # Built the way plan 12-09 builds it in `app.py`.
    context = turn.context
    return ClaimingToolHost(
        inner,
        registry=registry,
        owner=context.turn_key,
        label=lambda: context.speaker_label,
        friendly_name=lambda entity_id: context.friendly_names.get(entity_id),
    )


def _lamp_call_reply() -> BrainReply:
    return BrainReply(tool_calls=[ToolCall(name="ha_call_service", arguments=dict(_LAMP))])


def _pair(tmp_path, fake_stt, fake_brain, fake_tts, *, josh_says, sam_says, josh_brain=None, sam_brain=None, **kwargs):
    speaker = GroupSpeaker(merge_wait_s=0.3)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    common = dict(live_tts=live_tts, fake_stt=fake_stt, fake_brain=fake_brain)
    registry = ClaimRegistry()
    home = _FakeHomeAssistant()
    josh = _Turn(
        tmp_path, speaker, key="src:1", order_frame=100, member=(1, "Josh"), transcript=josh_says, brain=josh_brain, **common
    )
    sam = _Turn(
        tmp_path, speaker, key="src:2", order_frame=300, member=(2, "Sam"), transcript=sam_says, brain=sam_brain, **common
    )
    sam.stt = _DelayedStt(sam.stt, 0.1)
    josh.tool_host = _claiming(home, registry, josh)
    sam.tool_host = _claiming(home, registry, sam)
    for turn in (josh, sam):
        turn.run_kwargs = {"state_fetch": _state_fetch, **kwargs}
    return josh, sam, home, live_tts


async def test_a_refused_claim_is_spoken_on_the_model_path_and_home_assistant_gets_one_call(
    tmp_path, fake_stt, fake_brain, fake_tts
):
    josh, sam, home, live_tts = _pair(
        tmp_path,
        fake_stt,
        fake_brain,
        fake_tts,
        josh_says="turn on the lamp",
        sam_says="turn on the lamp please",
        josh_brain=fake_brain(replies=[_lamp_call_reply()]),
        sam_brain=fake_brain(replies=[_lamp_call_reply()]),
    )

    await _run_both(josh, sam)

    assert len(home.calls) == 1
    (spoken,) = live_tts.received_text
    assert "Sam, Josh just changed the lamp." in spoken


async def test_a_refused_claim_replaces_the_cannot_do_reply_on_the_local_intent_path(
    tmp_path, fake_stt, fake_brain, fake_tts
):
    josh, sam, home, live_tts = _pair(
        tmp_path,
        fake_stt,
        fake_brain,
        fake_tts,
        josh_says="turn on the lamp",
        sam_says="turn on the lamp",
        local_intents=True,
    )

    await _run_both(josh, sam)

    assert len(home.calls) == 1
    (spoken,) = live_tts.received_text
    assert "Sam, Josh just changed the lamp." in spoken
    assert "can't do that one" not in spoken
    assert sam.timings.turn_outcome == "claim_refused"
    assert josh.timings.turn_outcome == "local_intent"


async def test_the_refusal_names_the_entity_from_the_refused_turns_own_state_fetch(
    tmp_path, fake_stt, fake_brain, fake_tts
):
    josh, sam, home, live_tts = _pair(
        tmp_path,
        fake_stt,
        fake_brain,
        fake_tts,
        josh_says="turn on the lamp",
        sam_says="turn on the lamp",
        local_intents=True,
    )

    await _run_both(josh, sam)

    assert sam.context.friendly_names == {"light.example_lamp": "lamp"}
    assert "the lamp." in live_tts.received_text[0]


async def test_a_macro_runs_through_the_unclaimed_inner_host_and_takes_no_claim(
    tmp_path, fake_stt, fake_brain, fake_tts
):
    macro = MacroConfig(
        phrase="lamp time",
        aliases=(),
        reply="lamp done",
        actions=(MacroActionConfig(tool="ha_call_service", arguments=dict(_LAMP)),),
    )
    josh, sam, home, live_tts = _pair(
        tmp_path,
        fake_stt,
        fake_brain,
        fake_tts,
        josh_says="turn on the lamp",
        sam_says="lamp time",
        josh_brain=fake_brain(replies=[_lamp_call_reply()]),
        macros=(macro,),
    )

    await _run_both(josh, sam)

    # Josh's claimed write and Sam's macro both reached Home Assistant.
    assert len(home.calls) == 2
    assert sam.timings.turn_outcome == "macro"
    assert "just changed" not in live_tts.received_text[0]
