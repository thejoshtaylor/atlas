"""Real assertions for the per-turn stage-timing validation map.

Turned green by plan 01-05.
"""

import pytest


async def test_stage_timestamps_recorded(fake_audio_source, fake_stt, fake_brain, fake_tts):
    """A full turn driven by fakes records all seven stage timestamps, each
    at or after the one before it, with none left unset.

    The scripted STT events include a partial before the final, so
    `first_partial_at` has something real to record -- a fixture with only
    the final transcript would leave that one stage genuinely unreached,
    which is a different (and also correct) case this test does not cover.
    """
    from types import SimpleNamespace

    from spire_voice.providers.base import BrainReply, FinalTranscript, PartialTranscript, ToolCall
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    class _StubToolHost:
        async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])

    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(
        events=[
            PartialTranscript(text="turn"),
            PartialTranscript(text="turn on"),
            FinalTranscript(text="turn on the fan"),
        ]
    )
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="ha_call_service",
                        arguments={"domain": "switch", "service": "turn_on", "entity_id": "switch.example_fan"},
                    )
                ]
            ),
            BrainReply(text="turned on the fan"),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        _StubToolHost(),
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    stages = [
        timings.turn_started_at,
        timings.stt_socket_open_at,
        timings.first_partial_at,
        timings.stt_final_at,
        timings.brain_first_token_at,
        timings.tool_rounds_done_at,
        timings.first_audio_at,
    ]
    assert all(stage is not None for stage in stages), stages
    assert stages == sorted(stages)
    assert timings.turn_outcome == "completed"


def test_end_of_speech_to_first_audio_is_measured():
    """The budget number is derived from the two recorded timestamps it
    names, not measured a second, separate way."""
    from spire_voice.timing import TurnTimings

    timings = TurnTimings()
    timings.stt_final_at = 10.0
    timings.first_audio_at = 10.35

    assert timings.end_of_speech_to_first_audio_ms == pytest.approx(350.0)
