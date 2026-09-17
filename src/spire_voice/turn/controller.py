"""The turn state machine: frames to transcript to tools to speech.

`TurnController` never imports a concrete transport module -- it takes an
`AudioSource` and three providers, whichever they are, and the turn
controller has no idea which transport or which provider implementation
produced them. That is D-02's whole purpose.

The final transcript is whichever speech-to-text event the stream yields
last; every earlier event is a partial forwarded to the page live. This is
deliberate: it avoids an `isinstance` check against a specific
`PartialTranscript`/`FinalTranscript` class, so a test's own
structurally-identical fake classes (see `tests/conftest.py`) work exactly
like the real ones from `spire_voice.providers.base`.

`Denied`, when a tool call is refused, reaches the reply path directly: its
`reason` becomes the text handed to text-to-speech, with no second language
model call in between to reword it. That is the one thing this path must
not do, per `spire_mcp.safety.Denied`'s own doctrine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from spire_voice.timing import TurnTimings

logger = logging.getLogger("spire_voice.turn.controller")

_NO_SPEECH_REPLY = "sorry, i didn't catch that"
_TOO_MANY_ROUNDS_REPLY = "that needs more steps than i can take at once"


class _AudioSource(Protocol):
    def frames(self) -> AsyncIterator[bytes]: ...
    async def send_audio(self, chunk: bytes) -> None: ...


class _SttProvider(Protocol):
    def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Any]: ...


class _BrainProvider(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any: ...


class _TtsProvider(Protocol):
    def synthesize(self, text_deltas: AsyncIterator[str]) -> AsyncIterator[bytes]: ...


class _ToolHost(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


@dataclass
class TurnController:
    """Everything one turn needs, wired once by `app.py`'s lifespan."""

    source: _AudioSource
    stt: _SttProvider
    brain: _BrainProvider
    tts: _TtsProvider
    tool_host: _ToolHost
    tools_schema: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    max_tool_rounds: int = 3

    async def run(self, timings: TurnTimings) -> None:
        await run_turn(
            self.source,
            self.stt,
            self.brain,
            self.tts,
            self.tool_host,
            self.tools_schema,
            self.system_prompt,
            self.max_tool_rounds,
            timings,
        )


async def run_turn(
    source: _AudioSource,
    stt: _SttProvider,
    brain: _BrainProvider,
    tts: _TtsProvider,
    tool_host: _ToolHost,
    tools_schema: list[dict[str, Any]],
    system_prompt: str,
    max_tool_rounds: int,
    timings: TurnTimings,
) -> None:
    """Drive one turn end to end: frames -> transcript -> tool calls -> speech."""
    final = await _drain_to_final_transcript(source, stt)
    timings.mark_stt_final()

    final_text = getattr(final, "text", "") if final is not None else ""
    if not final_text:
        # No speech, or nothing intelligible -- VOICE-08 closes the turn with
        # no language model call. The client-side no-speech timeout and the
        # "next turn still runs" guarantee are plan 01-05's; this tracer only
        # needs the happy path not to hang on an empty utterance.
        await _emit_event(source, {"type": "reply.text", "text": _NO_SPEECH_REPLY})
        return

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_text},
    ]

    reply_text = await _run_tool_rounds(brain, tool_host, tools_schema, messages, max_tool_rounds)

    await _speak(source, tts, timings, reply_text)
    await _emit_event(
        source,
        {"type": "turn.timing", "end_of_speech_to_first_audio_ms": timings.end_of_speech_to_first_audio_ms},
    )
    timings.log()


async def _drain_to_final_transcript(source: _AudioSource, stt: _SttProvider) -> Any | None:
    """Forward every event but the last as a partial; return the last, if any."""
    pending: Any | None = None
    async for event in stt.stream(source.frames()):
        if pending is not None:
            await _emit_event(source, {"type": "transcript.partial", "text": getattr(pending, "text", "")})
        pending = event
    return pending


async def _run_tool_rounds(
    brain: _BrainProvider,
    tool_host: _ToolHost,
    tools_schema: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    max_tool_rounds: int,
) -> str:
    """Run up to `max_tool_rounds` brain calls, executing any tool calls in between.

    A command needing more rounds than the configured cap is a
    misunderstanding, not a complex request (per `BrainConfig.max_tool_rounds`'s
    own doctrine) -- the turn ends by saying so rather than continuing.
    """
    for _round_num in range(max_tool_rounds):
        reply = await brain.chat(messages, tools=tools_schema)
        if not reply.tool_calls:
            return reply.text

        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for i, tc in enumerate(reply.tool_calls)
                ],
            }
        )

        for i, tc in enumerate(reply.tool_calls):
            result = await tool_host.call_tool(tc.name, tc.arguments)
            content_text = _result_text(result)
            if _is_error(result):
                # A refused or failed tool call reaches the reply path
                # directly -- no second brain call reworks it.
                return content_text
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": content_text})

    return _TOO_MANY_ROUNDS_REPLY


async def _speak(source: _AudioSource, tts: _TtsProvider, timings: TurnTimings, reply_text: str) -> None:
    async def _one_delta() -> AsyncIterator[str]:
        yield reply_text

    first_audio_marked = False
    async for chunk in tts.synthesize(_one_delta()):
        if not first_audio_marked:
            timings.mark_first_audio()
            first_audio_marked = True
        await source.send_audio(chunk)

    await _emit_event(source, {"type": "reply.text", "text": reply_text})


async def _emit_event(source: _AudioSource, event: dict[str, Any]) -> None:
    """Send an event if `source` supports it.

    The real `AudioSource` protocol always implements `send_event` (see
    `transports/base.py`); a test's fake audio source may not, since the
    events it renders are not load-bearing for what that test proves.
    """
    send_event = getattr(source, "send_event", None)
    if send_event is not None:
        await send_event(event)


def _is_error(result: Any) -> bool:
    return bool(getattr(result, "isError", getattr(result, "is_error", False)))


def _result_text(result: Any) -> str:
    content = getattr(result, "content", None) or []
    if content:
        text = getattr(content[0], "text", None)
        if text is not None:
            return text
    return ""
