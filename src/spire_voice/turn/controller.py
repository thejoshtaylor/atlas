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

import asyncio
import contextlib
import json
import logging
import time as _time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol

from spire_voice.timing import TurnTimings

logger = logging.getLogger("spire_voice.turn.controller")

_NO_SPEECH_REPLY = "sorry, i didn't catch that"
_TOO_MANY_ROUNDS_REPLY = "that needs more steps than i can take at once"

# How often the silence-timeout guard rechecks its deadline while waiting on
# an STT event that may never arrive. Real events short-circuit this --
# `asyncio.wait` returns the instant the event lands, so this interval only
# bounds how quickly a *stuck* session notices the deadline passed, not the
# latency of an ordinary turn.
_DEFAULT_POLL_INTERVAL_S = 0.05


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
    max_utterance_s: float = 15.0

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
            self.max_utterance_s,
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
    max_utterance_s: float = 15.0,
    *,
    clock: Callable[[], float] = _time.monotonic,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
) -> None:
    """Drive one turn end to end: frames -> transcript -> tool calls -> speech.

    `max_utterance_s`, `clock`, and `poll_interval_s` all default to their
    production values, so every caller that predates this plan (the
    WebSocket/WebRTC routes in `app.py`, the safety-integration test) keeps
    working unmodified. `clock` exists so a test can drive the silence-timeout
    guard without waiting out the real configured duration -- see
    `tests/test_turn_controller.py::test_silence_timeout_closes_turn`.
    """
    timings.mark_turn_started()
    timings.turn_outcome = "completed"

    final = await _drain_to_final_transcript(
        source, stt, max_utterance_s, timings, clock=clock, poll_interval_s=poll_interval_s
    )
    timings.mark_stt_final()

    final_text = getattr(final, "text", "") if final is not None else ""
    if not final_text:
        # VOICE-08's two cases end the turn the same way, with no language
        # model call and no text-to-speech call: `final is None` is
        # RESEARCH.md Pitfall 3's second case (the provider never sent
        # anything at all, closed here by the client-side timeout); a
        # `FinalTranscript` whose text is empty is the first case (something
        # arrived and decoded to nothing). `turn_outcome` keeps the two
        # distinguishable in the log even though the reply path is shared.
        timings.turn_outcome = "timeout" if final is None else "empty_transcript"
        await _emit_event(source, {"type": "reply.text", "text": _NO_SPEECH_REPLY})
        await _emit_event(source, timings.to_event())
        timings.log()
        return

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_text},
    ]

    reply_text = await _run_tool_rounds(brain, tool_host, tools_schema, messages, max_tool_rounds, timings)
    timings.mark_tool_rounds_done()

    await _speak(source, tts, timings, reply_text)
    await _emit_event(source, timings.to_event())
    timings.log()


async def _drain_to_final_transcript(
    source: _AudioSource,
    stt: _SttProvider,
    max_utterance_s: float,
    timings: TurnTimings,
    *,
    clock: Callable[[], float],
    poll_interval_s: float,
) -> Any | None:
    """Forward every event but the last as a partial; return the last, if any.

    RESEARCH.md Pitfall 3: xAI's STT sends no dedicated no-speech event -- a
    turn where nothing is ever said can otherwise leave this function
    suspended forever on a socket that will never send anything. Bounding
    the wait from this side, rather than trusting a provider event that may
    not exist, is the mechanism behind VOICE-08's "closes itself": once
    `max_utterance_s` elapses with no final transcript ever received, the
    stream is closed through its own `aclose()` (releasing the socket, not
    abandoning it) and the turn ends exactly like an empty transcript would.

    A real event short-circuits the wait immediately -- `asyncio.wait`
    returns the instant the pending `__anext__()` task completes, so
    `poll_interval_s` only bounds how quickly a *stuck* session notices its
    deadline passed, never the latency of an ordinary turn.
    """
    timings.mark_stt_socket_open()
    stream = stt.stream(source.frames())
    deadline = clock() + max_utterance_s
    pending: Any | None = None
    next_event_task = asyncio.ensure_future(stream.__anext__())
    while True:
        if clock() >= deadline:
            logger.info(
                "stt session exceeded max_utterance_s=%s with no final transcript; closing it",
                max_utterance_s,
            )
            next_event_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await next_event_task
            await stream.aclose()
            return None

        done, _pending_tasks = await asyncio.wait({next_event_task}, timeout=poll_interval_s)
        if next_event_task not in done:
            continue

        try:
            event = next_event_task.result()
        except StopAsyncIteration:
            return pending

        if pending is not None:
            timings.mark_first_partial()
            await _emit_event(source, {"type": "transcript.partial", "text": getattr(pending, "text", "")})
        pending = event
        next_event_task = asyncio.ensure_future(stream.__anext__())


async def _run_tool_rounds(
    brain: _BrainProvider,
    tool_host: _ToolHost,
    tools_schema: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    max_tool_rounds: int,
    timings: TurnTimings,
) -> str:
    """Run up to `max_tool_rounds` brain calls, executing any tool calls in between.

    A command needing more rounds than the configured cap is a
    misunderstanding, not a complex request (per `BrainConfig.max_tool_rounds`'s
    own doctrine) -- the turn ends by saying so, never with a success
    confirmation, per CMD-01's transparency prohibition.
    """
    for _round_num in range(max_tool_rounds):
        reply = await brain.chat(messages, tools=tools_schema)
        timings.mark_brain_first_token()
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
                # An error-shaped result short-circuits straight to speech --
                # no second brain call touches it. For a refusal this is the
                # boundary's own doctrine: `spire_mcp.safety.Denied.reason`
                # crosses the MCP boundary as `content_text` unchanged (see
                # `mcp_client.py`'s module docstring), so `content_text` here
                # IS `Denied.reason`, verbatim, with nothing in between to
                # reword it. This branch is deliberately the only place a
                # tool result can end a turn without another round -- it
                # covers both a refusal and an ordinary tool-level failure,
                # and neither one reaches the caller as a confirmation.
                return content_text
            # A non-2xx Home Assistant response (see `handle_call_service`)
            # is not a refusal -- it comes back as an ordinary, non-error
            # result whose content names the failure, so the next brain call
            # reports it rather than confirming success.
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": content_text})

    logger.warning("turn hit max_tool_rounds=%d without settling on a reply", max_tool_rounds)
    timings.turn_outcome = "round_cap"
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
