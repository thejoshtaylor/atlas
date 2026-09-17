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

Plan 01.1-04 replaces the single `_run_tool_rounds` call with a raced tier
list (`turn/brain_race.py`): every tier is dispatched concurrently, the
first confident reply wins, and a holding phrase from the startup cache
covers the wait past `filler_after_ms` if the race is still running. `tiers`
defaults to `None`, in which case `run_turn` wraps its `brain` positional
argument in a one-element tier list and races that -- there is one code
path, and a single tier is its degenerate case, not a bypass. This module
imports `spire_voice.turn.brain_race` at load time; `brain_race.py` imports
`_run_tool_rounds` back from this module, but only inside a function body
(deferred past both modules' load), so the two import in either order with
no cycle.

Plan 01.1-05 inserts a macro check between the empty-transcript guard and
the tier dispatch: a transcript matching a configured macro (`turn/macros.py`)
never reaches the tier race at all (MACRO-02), and the macro's actions run
through the same `tool_host.call_tool` / `allow_call` path a model-issued
call uses. `turn/macros.py` imports `_is_error`/`_result_text` back from
this module the same deferred, function-body way `brain_race.py` already
does, so this module can import `turn.macros` at load time with no cycle.

Plan 01.1-06 starts a caller-supplied `state_fetch` awaitable as a task
immediately after the turn starts -- before `_drain_to_final_transcript` is
ever awaited -- so it overlaps the operator still speaking and has usually
finished before the transcript is final (D-15). Its result becomes the
second of three messages a tier sees: a stable catalog system message (the
`system_prompt` positional, unchanged in shape), a volatile state system
message built from the fetch, and the user message. `_state_message` lives
in `app.py` alongside `_catalog_prompt` (D-14); this module reaches it with
the same deferred, function-body import `brain_race.py` already uses for
`_run_tool_rounds`, since `app.py` imports `run_turn` back from this module
at load time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time as _time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Literal, Mapping, Protocol

from spire_voice.config import MacroConfig
from spire_voice.providers.tier_reply import DEFAULT_FILLER, FILLER_TEXT, FillerPhrase, TierReply
from spire_voice.providers.tts_cache import CachedTts
from spire_voice.timing import TurnTimings
from spire_voice.turn import brain_race
from spire_voice.turn.macros import fire_macro, match as match_macro

logger = logging.getLogger("spire_voice.turn.controller")

_NO_SPEECH_REPLY = "sorry, i didn't catch that"
_TOO_MANY_ROUNDS_REPLY = "that needs more steps than i can take at once"
_EMPTY_REPLY = "sorry, i don't have anything to say to that"
# Spoken whenever an error-shaped tool result's extracted text is empty --
# in both `_run_tool_rounds`'s short-circuit below and the macro path's
# failure branch above. Names the case without asserting an outcome: an
# operator hearing this reads it as "that was refused and i cannot tell you
# more," never as a confirmation. Silence after a spoken command is
# indistinguishable from a dropped turn, and an operator who cannot tell a
# refusal from a crash will stop trusting the refusals.
_DENIED_FALLBACK_REPLY = "that was refused, and i don't have anything more to tell you about it"

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
    tiers: list[brain_race.TierBrain] | None = None,
    filler_after_ms: float = 600.0,
    filler_cache: Mapping[str, bytes] | None = None,
    macros: tuple[MacroConfig, ...] = (),
    state_fetch: Callable[[], Any] | None = None,
) -> None:
    """Drive one turn end to end: frames -> transcript -> macro/tier -> speech.

    `max_utterance_s`, `clock`, `poll_interval_s`, `tiers`, `filler_after_ms`,
    `filler_cache`, `macros`, and `state_fetch` all default to their Phase 01
    (or Phase 01 shape-only) behavior, so every caller that predates this
    plan (the WebSocket/WebRTC routes in `app.py`, the safety-integration
    test) keeps working unmodified. `clock` exists so a test can drive the
    silence-timeout guard and the filler deadline without waiting out the
    real configured duration -- see
    `tests/test_turn_controller.py::test_silence_timeout_closes_turn`.

    `tiers=None` wraps `brain` in a one-element tier list and races that --
    there is one code path, always the list; a one-entry list is the
    degenerate case, not a bypass.

    `filler_cache` doubles as the macro-reply cache: `app.py`'s `lifespan`
    (plan 01.1-05) precaches every macro's `reply` into the same dict it
    builds every holding phrase into, so a macro's confirmation is served
    through the exact `CachedTts` adapter the filler already uses -- no
    second cache, no second miss-handling path.

    `state_fetch`, when given, is an awaitable factory the caller supplies
    -- never a tool-host call this function constructs itself, so this
    module keeps knowing nothing about which tool provides state, the same
    defaulting discipline `tiers`/`filler_cache`/`macros` already follow.
    `state_fetch=None` (the default) produces the exact two-message list
    (`system_prompt`, user text) this function always has; supplying it adds
    a second, volatile system message built from the fetch's result, so a
    tier sees three messages in catalog-state-user order (D-14).
    """
    timings.mark_turn_started()
    timings.turn_outcome = "completed"

    # D-15: started here, before the drain below is ever awaited, so the
    # fetch overlaps the operator still speaking and has usually finished by
    # the time the transcript is final -- costing about nothing inside the
    # measured budget. Started after the drain returns instead, it would pay
    # its full latency inside that exact window. `state_fetch is None` skips
    # the task entirely: no fetch, no cost, no behavior change for a caller
    # that predates this plan.
    state_task: asyncio.Task[Any] | None = asyncio.create_task(state_fetch()) if state_fetch is not None else None

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
        await _cancel_state_task(state_task)
        await _emit_event(source, {"type": "reply.text", "text": _NO_SPEECH_REPLY})
        await _emit_event(source, timings.to_event())
        timings.log()
        return

    # The macro check runs here, before a single tier task is created:
    # placed after the dispatch below, the model round trip would already
    # have been paid and MACRO-02 would be false while every other
    # behavioral test still passed. `match_macro` on an empty `macros` tuple
    # (the default) always returns `None`, so a caller that predates this
    # plan reaches the tier race exactly as before, at no observable cost.
    matched_macro = match_macro(macros, final_text)
    if matched_macro is not None:
        # A macro turn never builds the message list and never consumes the
        # state fetch's result -- but the fetch was already started above
        # (state_fetch does not know yet whether this turn will match a
        # macro), so it must be cancelled and its cancellation awaited here,
        # never left dangling. Same cancel-then-await shape `race_tiers`
        # already uses for a losing tier -- not a second mechanism.
        await _cancel_state_task(state_task)
        outcome = await fire_macro(matched_macro, tool_host)
        timings.turn_outcome = "macro" if outcome.succeeded else "macro_failed"
        # A macro reply is an answer, not a holding phrase -- it is the one
        # utterance in this system that is both the answer and instant. On
        # failure `outcome.cacheable` is always False (MacroOutcome's own
        # doctrine): the failure text is composed at turn time from whichever
        # action actually failed, so it was never precached, and this turn
        # deliberately pays the live synthesis cost and loses the sub-1.5
        # second claim. Losing it here is correct: CMD-07 forbids a cached
        # confirmation of something that did not happen, and the latency
        # cost is the honest price of not lying.
        speaking_tts = CachedTts(filler_cache or {}) if outcome.cacheable else tts
        # `outcome.text` is the boundary's own words verbatim (CMD-08) --
        # nothing prepended, appended, or reworded here, and no length check
        # or truncation either: a refusal never passes through a model, so
        # `brain.max_tokens` does not apply to it and there is nothing to
        # truncate against. `_DENIED_FALLBACK_REPLY` covers only the one
        # case where there are no words at all.
        await _speak(source, speaking_tts, timings, outcome.text or _DENIED_FALLBACK_REPLY, kind="answer")
        await _emit_event(source, timings.to_event())
        timings.log()
        return

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if state_task is not None:
        # `state_task` was started before the drain above, so by now it has
        # usually already finished -- this await rarely actually waits. A
        # fetch that raised is logged and treated as "nothing known" rather
        # than ending the turn: an assistant that cannot read current state
        # can still take a command, and failing the whole turn over a
        # stale-state fetch would be a worse outcome than answering without
        # it (T-01.1-17).
        try:
            states_payload = await state_task
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("state fetch raised; continuing turn with no known state")
            states_payload = []
        states = (
            {entity["entity_id"]: entity["state"] for entity in states_payload}
            if isinstance(states_payload, list)
            else {}
        )
        # Deferred, not module-level: `app.py` imports `run_turn` from this
        # module at load time, so a module-level import here of anything
        # from `app.py` would deadlock the two modules' load order. By the
        # time this line actually runs, this module has always finished
        # loading -- there is no way to call `run_turn` without importing
        # `spire_voice.turn.controller` first -- so importing `app.py` here,
        # even the first time, only ever fetches or finishes a module that
        # cannot be mid-load on this side. Same deferred-import shape
        # `brain_race.py` already uses for `_run_tool_rounds`.
        from spire_voice.app import _state_message

        messages.append({"role": "system", "content": _state_message(states)})
    messages.append({"role": "user", "content": final_text})

    if tiers is None:
        # The degenerate one-tier case: no instructor client exists, so
        # `run_top_tier` wraps its settled text locally as a confident
        # `TierReply` instead of calling out. This is the seam that keeps
        # every Phase 01 test -- which drives `run_turn` with a `FakeBrain`
        # and no instructor client -- working unchanged.
        tiers = [brain_race.TierBrain(index=0, model="", brain=brain, envelope_client=None, calls_tools=True)]

    # CR-01: one instance per turn, shared between the top tier's tool
    # round and the race -- set True the instant a real tool call is made,
    # so a triage tier's confident reply can no longer end the race in the
    # top tier's place once its action is no longer cancellable.
    commitment = brain_race.ToolCommitment()

    tier_tasks: dict[int, asyncio.Task[TierReply]] = {}
    for tier in tiers:
        tier_messages = list(messages)
        if tier.calls_tools:
            coro = brain_race.run_top_tier(
                tier, tool_host, tools_schema, tier_messages, max_tool_rounds, timings, commitment
            )
        else:
            coro = brain_race.run_triage_tier(tier, tier_messages)
        tier_tasks[tier.index] = asyncio.create_task(coro)

    race_task = asyncio.create_task(brain_race.race_tiers(tier_tasks, commitment))

    filler_deadline = clock() + filler_after_ms / 1000
    first_filler: FillerPhrase | None = None
    inspected: set[asyncio.Task[TierReply]] = set()
    while not race_task.done() and clock() < filler_deadline:
        await asyncio.sleep(poll_interval_s)
        for task in tier_tasks.values():
            if task.done() and task not in inspected:
                inspected.add(task)
                if task.cancelled() or task.exception() is not None:
                    continue
                reply = task.result()
                if not reply.confident and first_filler is None:
                    first_filler = reply.filler

    if not race_task.done() and filler_cache:
        # D-08/D-09: the holding phrase plays on a deadline, from the
        # startup cache only, and is awaited to completion before the
        # answer -- that sequential await is what makes D-09 true (the
        # filler finishes before the answer can start) with no second
        # mechanism. When `filler_cache` is empty or `None`, this branch is
        # skipped entirely: the turn waits in silence rather than
        # synthesizing at turn time.
        phrase = first_filler or DEFAULT_FILLER
        await _speak(source, CachedTts(filler_cache), timings, FILLER_TEXT[phrase], kind="filler")

    winner = await race_task
    timings.mark_tool_rounds_done()

    await _speak(source, tts, timings, winner.answer, kind="answer")
    await _emit_event(source, timings.to_event())
    timings.log()


async def _cancel_state_task(state_task: "asyncio.Task[Any] | None") -> None:
    """Cancel a still-pending state-fetch task and await its unwind.

    Same cancel-then-gather shape `race_tiers` already uses for a tier that
    lost the race -- reused rather than inventing a second mechanism for the
    same kind of cleanup. A no-op if `state_task` is `None` or already done,
    so every early-return call site can call this unconditionally.
    """
    if state_task is None or state_task.done():
        return
    state_task.cancel()
    await asyncio.gather(state_task, return_exceptions=True)


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
    commitment: "brain_race.ToolCommitment | None" = None,
) -> str:
    """Run up to `max_tool_rounds` brain calls, executing any tool calls in between.

    A command needing more rounds than the configured cap is a
    misunderstanding, not a complex request (per `BrainConfig.max_tool_rounds`'s
    own doctrine) -- the turn ends by saying so, never with a success
    confirmation, per CMD-01's transparency prohibition.

    `commitment`, when given, is set the instant a call to `tool_host.call_tool`
    returns -- whichever way it resolved (CR-01). A stdio round trip to the
    MCP child cannot be un-sent once dispatched, so from that point on
    `race_tiers` must not let a triage tier's confident reply end the race
    in this tier's place; only this tier's own settled outcome may describe
    what actually happened.
    """
    for _round_num in range(max_tool_rounds):
        reply = await brain.chat(messages, tools=tools_schema)
        timings.mark_brain_first_token()
        if not reply.tool_calls:
            if not reply.text:
                # A tool-call-free reply with no text is a valid, reachable
                # chat-completions shape (the model stops early against
                # `max_tokens`, or simply has nothing to add) -- not an
                # error, but left unhandled it reaches `_speak` as an empty
                # string and the operator hears silence with no indication
                # the turn ended. Handled the same way VOICE-08's
                # empty-transcript case is: a fixed fallback, and a
                # `turn_outcome` that keeps this distinguishable in the log.
                timings.turn_outcome = "empty_reply"
                return _EMPTY_REPLY
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
            if commitment is not None:
                commitment.committed = True
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
                # No length check or truncation applies: a refusal never
                # passes through a model, so `brain.max_tokens` has nothing
                # to say about it. `_DENIED_FALLBACK_REPLY` covers only the
                # case where `content_text` itself is empty, so a refusal is
                # never indistinguishable from a dropped turn.
                return content_text or _DENIED_FALLBACK_REPLY
            # A non-2xx Home Assistant response (see `handle_call_service`)
            # is not a refusal -- it comes back as an ordinary, non-error
            # result whose content names the failure, so the next brain call
            # reports it rather than confirming success.
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": content_text})

    logger.warning("turn hit max_tool_rounds=%d without settling on a reply", max_tool_rounds)
    timings.turn_outcome = "round_cap"
    return _TOO_MANY_ROUNDS_REPLY


async def _speak(
    source: _AudioSource,
    tts: _TtsProvider,
    timings: TurnTimings,
    reply_text: str,
    *,
    kind: Literal["filler", "answer"],
) -> None:
    """Speak one utterance, and mark whichever timing(s) `kind` calls for.

    `kind` has no default on purpose: it is the one thing keeping a filler
    from setting the answer mark. Both kinds mark `first_audio` on the first
    chunk (criterion 6 wants "any audio, filler included" for that mark);
    only the answer utterance also marks `answer_audio`.

    Both marks are set from one captured `time.monotonic()` reading, not two
    separate calls to `timings.mark_first_audio()`/`mark_answer_audio()`: a
    turn whose race finished before the filler deadline reaches this branch
    with `kind="answer"` on its very first chunk, and `first_audio_at`/
    `answer_audio_at` describe the exact same event then -- two independent
    `time.monotonic()` reads would leave them off by a fraction of a
    microsecond forever, which is not "equal" by any test that checks it.
    Assignment (guarded exactly like the two mark methods) rather than
    calling them is what makes a shared reading possible; `turn_outcome` is
    already assigned directly from this module the same way.
    """

    async def _one_delta() -> AsyncIterator[str]:
        yield reply_text

    first_audio_marked = False
    async for chunk in tts.synthesize(_one_delta()):
        if not first_audio_marked:
            first_audio_marked = True
            now = _time.monotonic()
            if timings.first_audio_at is None:
                timings.first_audio_at = now
            if kind == "answer" and timings.answer_audio_at is None:
                timings.answer_audio_at = now
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
