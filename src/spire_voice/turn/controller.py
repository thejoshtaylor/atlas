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

Plan 04-04 adds `needs_clarification` as a third, exclusive outcome a
triage tier's reply can win the race with (CMD-09, D-07): a spoken name
matching more than one known entity asks which one was meant instead of
guessing. `run_turn`'s post-race branch speaks a question composed in code
from `winner.candidates`, preferring a friendly name from this turn's own
injected state fetch over the bare entity id, and returns -- no macro
follow-up, no further tool round, and nothing that holds the audio source
open past the question (D-08; the operator answers by waking the assistant
again, and VOICE-21 is deliberately out of scope). `turn_outcome` gains its
own `"needs_clarification"` value, distinct from an ordinary answer, an
empty reply, and a round cap.

Plan 06-05 (D-11) extends `winner.candidates` to a third kind of value:
two plugins' own display names, when a spoken command could reach either
plugin's version of the same capability. Nothing below this comment
changed to add that case -- `_compose_clarifying_question` already speaks
whatever `candidates` carries, falling back to the literal string when
`friendly_names` (built from live entity state) has no entry for it,
which a plugin's display name never has. Same envelope, same validators,
same composer; only the meaning of the strings inside `candidates` grew a
third case.
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
from spire_voice.providers.base import BrainError
from spire_voice.transports.base import SourceFormat
from spire_voice.providers.tier_reply import DEFAULT_FILLER, FILLER_TEXT, FillerPhrase, TierReply
from spire_voice.providers.tts_cache import CachedTts
from spire_voice.providers.tts_xai import SinkFormat
from spire_voice.session.recorder import SessionRecorder
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

# The fixed phrase `_compose_mixed_outcome_reply` speaks for one action that
# succeeded, in a batch where at least one other action did not (CMD-07,
# D-14). Never a restatement of what the model asked for -- the point is
# only to mark the action as done, the same way a successful action's tool
# result carries no special wording either.
_ACTION_SUCCEEDED_CLAUSE = "succeeded"
# Spoken for an action whose `tool_host.call_tool` awaitable raised instead
# of returning a result -- the call itself never reached a verdict, refusal
# or otherwise. Deliberately a different fixed phrase from a boundary
# refusal (`_result_text`'s reason text): "it never even ran" and "it ran
# and was refused" are different facts about the house, and an operator
# hearing this reply must be able to tell them apart.
_ACTION_DID_NOT_COMPLETE_CLAUSE = "the call itself did not complete"

# The fixed carrier phrase `_compose_clarifying_question` speaks before every
# candidate name, for a turn a triage tier's `needs_clarification` reply won
# (CMD-09, D-07, D-14). Fixed and asserted-on, never model-composed: the
# candidates are the only variable part of the sentence, joined by code, so
# the same candidate list always produces the same question -- the same
# discipline `_compose_mixed_outcome_reply` already applies to a mixed
# tool-round summary.
_CLARIFYING_QUESTION_CARRIER = "i'm not sure which one you mean --"

# How often the silence-timeout guard rechecks its deadline while waiting on
# an STT event that may never arrive. Real events short-circuit this --
# `asyncio.wait` returns the instant the event lands, so this interval only
# bounds how quickly a *stuck* session notices the deadline passed, not the
# latency of an ordinary turn.
_DEFAULT_POLL_INTERVAL_S = 0.05


class _AudioSource(Protocol):
    def frames(self) -> AsyncIterator[bytes]: ...
    async def send_audio(self, chunk: bytes) -> None: ...
    def source_format(self) -> SourceFormat: ...


class _SttProvider(Protocol):
    def stream(self, frames: AsyncIterator[bytes], source_format: SourceFormat) -> AsyncIterator[Any]: ...


class _BrainProvider(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any: ...


class _TtsProvider(Protocol):
    def synthesize(
        self, text_deltas: AsyncIterator[str], sink: "SinkFormat | None" = None
    ) -> AsyncIterator[bytes]: ...


class _ToolHost(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class _BargeInMonitor(Protocol):
    """The shape `sources/runner.py`'s `BargeInMonitor` satisfies (VOICE-07,
    plan 02-06) -- a structural Protocol, not an import, the same
    discipline every other provider Protocol in this module already
    follows: `run_turn` never imports a concrete source-runner module, the
    same way it never imports a concrete transport (D-02).

    `source.barge_in` (`getattr(source, "barge_in", None)`) is duck-typed,
    the same way `_emit_event` already reaches for `send_event` below --
    `None` when the source carries none (every caller that predates this
    plan, and any source `SourceRunner` did not attach one to), in which
    case every one of `_speak`'s new checks is a no-op and behavior is
    byte-for-byte what it was before this plan.

    `trace` (plan 02-12, D-18) is the same kind of optional attribute,
    read the same way (`getattr(barge_in, "trace", None)`) rather than
    declared here as a required field -- a `BargeInMonitor` with
    correlation disabled, and every fake that predates plan 02-12, carries
    no `trace` at all.
    """

    enabled: bool
    interrupt_requested: bool

    def mark_playback_started(self, now: float) -> None: ...
    def mark_transcript_done(self) -> None: ...


class _RecordingAudioSource:
    """Wraps one `AudioSource`, taping the two things Task 3 taps and only
    those two: the frame iterator `run_turn` already drains, and the
    events it already emits through `_emit_event`.

    `send_audio()`/`source_format()` delegate straight through -- this
    wrapper adds no behavior to either, and reply audio going out never
    reaches a session recording (D-13 only names the audio a turn heard).
    Constructing this wrapper is the only tap this plan opens: `frames()`
    yields exactly what `self._wrapped.frames()` yields, one chunk handed
    to `self._recorder` before it is handed onward, so the recorder can
    hold no more than the turn actually sent to the transcriber (D-15).
    `send_event()` records every event before forwarding it, so the
    session's JSONL is the same stream the browser sees, not a second one
    invented for disk.
    """

    def __init__(self, wrapped: Any, recorder: SessionRecorder) -> None:
        self._wrapped = wrapped
        self._recorder = recorder

    async def frames(self) -> AsyncIterator[bytes]:
        async for chunk in self._wrapped.frames():
            self._recorder.record_audio_chunk(chunk)
            yield chunk

    async def send_audio(self, chunk: bytes) -> None:
        await self._wrapped.send_audio(chunk)

    async def send_event(self, event: dict[str, Any]) -> None:
        self._recorder.record_event(event)
        wrapped_send_event = getattr(self._wrapped, "send_event", None)
        if wrapped_send_event is not None:
            await wrapped_send_event(event)

    def source_format(self) -> SourceFormat:
        return self._wrapped.source_format()


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
    filler_cache: "Mapping[tuple[str, int] | None, Mapping[str, bytes]] | None" = None,
    macros: tuple[MacroConfig, ...] = (),
    state_fetch: Callable[[], Any] | None = None,
    pending_runs_fetch: Callable[[], Any] | None = None,
    session_recorder: SessionRecorder | None = None,
    speech_lock: "asyncio.Lock | None" = None,
    workflow_tool_host: Any | None = None,
    tool_owners: "Callable[[str], tuple[str, ...]] | None" = None,
) -> None:
    """Drive one turn end to end: frames -> transcript -> macro/tier -> speech.

    `max_utterance_s`, `clock`, `poll_interval_s`, `tiers`, `filler_after_ms`,
    `filler_cache`, `macros`, `state_fetch`, and `session_recorder` all
    default to their Phase 01 (or Phase 01 shape-only) behavior, so every
    caller that predates this plan (the WebSocket/WebRTC routes in
    `app.py`, the safety-integration test) keeps working unmodified.
    `clock` exists so a test can drive the silence-timeout guard and the
    filler deadline without waiting out the real configured duration --
    see `tests/test_turn_controller.py::test_silence_timeout_closes_turn`.

    `session_recorder=None` (the default, plan 02-05) means no on-disk
    session is written and `source` is used exactly as handed in --
    every test that predates this plan exercises precisely the code path
    it always has. When given, `source` is wrapped in
    `_RecordingAudioSource` before the drain below ever runs, so the
    recorder can only ever see what this turn actually sent to the
    transcriber, and it is closed from a `finally` block that covers every
    exit path this function has, including the two early returns below.

    `tiers=None` wraps `brain` in a one-element tier list and races that --
    there is one code path, always the list; a one-entry list is the
    degenerate case, not a bypass.

    `filler_cache` doubles as the macro-reply cache: `app.py`'s `lifespan`
    (plan 01.1-05) precaches every macro's `reply` into the same dict it
    builds every holding phrase into, so a macro's confirmation is served
    through the exact `CachedTts` adapter the filler already uses -- no
    second cache, no second miss-handling path. 260922-cts: it is keyed by
    `(codec, sample_rate)` -- one entry per sink format `app.py` actually
    precached, plus a `None` entry for the browser default -- rather than
    a single flat `{text: bytes}` dict, so `CachedTts.synthesize`'s own
    `sink` argument (this turn's resolved `sink`, above) reads the entry
    built for the source that will actually play it back, never whichever
    one happened to be built.

    `state_fetch`, when given, is an awaitable factory the caller supplies
    -- never a tool-host call this function constructs itself, so this
    module keeps knowing nothing about which tool provides state, the same
    defaulting discipline `tiers`/`filler_cache`/`macros` already follow.
    `state_fetch=None` (the default) produces the exact two-message list
    (`system_prompt`, user text) this function always has; supplying it adds
    a second, volatile system message built from the fetch's result, so a
    tier sees three messages in catalog-state-user order (D-14).

    `speech_lock=None` (the default, and every caller that predates plan
    05-03) is threaded unchanged into every `_speak` call this function
    makes; every pre-existing caller keeps sharing no lock with anything.
    Given a lock, this turn's filler and answer utterances -- and a
    scheduled step's own utterance sharing the same lock elsewhere -- can
    never interleave on the one speaker the room has.

    `pending_runs_fetch` (plan 05-05 Task 2, D-09) is `state_fetch`'s own
    sibling: an awaitable factory the caller supplies, started as its own
    task alongside `state_task` before the transcript drain below is ever
    awaited, so it overlaps the operator still speaking exactly the way
    `state_fetch` already does. `pending_runs_fetch=None` (the default,
    and every caller that predates this plan) produces byte-identical
    behavior -- no second task, no cost, `_state_message` called with its
    own `pending_runs=()` default. Given a factory, its result is threaded
    into `_state_message` alongside `states` so a tier sees the pending-run
    block in the same volatile system message live entity state already
    occupies. A fetch that raised is logged and treated as "nothing
    scheduled" rather than ending the turn, the identical T-01.1-17
    posture `state_fetch` already carries; cancelled and its cancellation
    awaited on the macro path and the empty-transcript early return,
    exactly where `state_task` already is, never left dangling.

    `workflow_tool_host` (WR-01 code-review fix) is given
    `set_current_turn_run_ids(...)` -- exactly the ids `pending_runs`
    resolved to above, the same set this turn's own pending-run context
    block already showed the model -- once per turn, before the user
    message and any tool round: `WorkflowToolHost.cancel_workflow_run`/
    `append_workflow_steps` refuse any `run_id` outside that set. Typed
    loosely (`Any`, duck-typed to `set_current_turn_run_ids`) rather than
    importing `spire_voice.workflow.tool.WorkflowToolHost` at module level
    for the same load-order reason this function already defers its
    `spire_voice.app` import below. `None` (the default, and every caller
    that predates this fix) skips the call -- no behavior change for a
    caller with no workflow tool host to scope.

    `tool_owners` (06-CONTEXT.md D-12, plan 06-05) is forwarded unchanged
    to `fire_macro` on the macro path below -- this function has no
    opinion of its own about plugin ownership, only a caller-supplied
    answer to "how many plugins currently publish this bare tool name" it
    passes through. `None` (the default, and every caller that predates
    this plan) reproduces `fire_macro`'s own pre-existing behavior exactly.
    """
    timings.mark_turn_started()
    timings.turn_outcome = "completed"

    # VOICE-07: captured from the *unwrapped* source, before the
    # `_RecordingAudioSource` wrapping below -- that wrapper delegates
    # `frames()`/`send_audio()` but does not forward arbitrary attributes,
    # and `barge_in` is `SourceRunner`'s own duck-typed attachment
    # (`sources/runner.py`'s module docstring), not part of the `AudioSource`
    # protocol. `None` for every caller that predates this plan, or any
    # source `SourceRunner` did not attach one to -- `_speak`'s own checks
    # below are then no-ops (`_BargeInMonitor`'s docstring).
    barge_in: _BargeInMonitor | None = getattr(source, "barge_in", None)

    # 260922-cts: captured the same way, and for the same reason, as
    # `barge_in` immediately above -- off the *unwrapped* source, before
    # `_RecordingAudioSource` wraps it below. `sink_format`, when the
    # source declares one (a `CameraAudioSource`, or a wrapper that
    # forwards it), is this turn's own playback pair, resolved once and
    # threaded into every `_speak` call below so a synthesis call -- and a
    # `CachedTts` lookup -- asks for the format this source's speaker
    # actually plays, never the browser default every caller that
    # predates this plan implicitly asked for (module docstring's Bug).
    # `None` for every source with no `sink_format` at all (every browser
    # and WebRTC source today), which is `_speak`'s own pre-fix default.
    _sink_format_fn = getattr(source, "sink_format", None)
    sink: SinkFormat | None = _sink_format_fn() if _sink_format_fn is not None else None

    # D-15: started here, before the drain below is ever awaited, so the
    # fetch overlaps the operator still speaking and has usually finished by
    # the time the transcript is final -- costing about nothing inside the
    # measured budget. Started after the drain returns instead, it would pay
    # its full latency inside that exact window. `state_fetch is None` skips
    # the task entirely: no fetch, no cost, no behavior change for a caller
    # that predates this plan.
    state_task: asyncio.Task[Any] | None = asyncio.create_task(state_fetch()) if state_fetch is not None else None
    # D-09: the same overlap-the-operator-still-speaking discipline
    # `state_task` above already establishes, for the pending-run block
    # rather than live entity state -- started here, not after, so a slow
    # workflow-repository read costs nothing inside the measured budget
    # either.
    pending_runs_task: asyncio.Task[Any] | None = (
        asyncio.create_task(pending_runs_fetch()) if pending_runs_fetch is not None else None
    )

    if session_recorder is not None:
        # Resolved from the source's own declaration, never assumed (D-13),
        # and wrapped before `_drain_to_final_transcript` is ever awaited
        # below -- there is no earlier point at which `source.frames()`
        # could be read, so this is the only tap this function opens.
        fmt = source.source_format()
        session_recorder.set_audio_format(fmt.encoding, fmt.sample_rate)
        # Plan 08-11: read off the same still-unwrapped `source` the
        # `barge_in` attachment above already reads off, and for the same
        # reason -- `_RecordingAudioSource` delegates `frames()`/
        # `send_audio()`/`send_event()` but forwards no arbitrary
        # attribute, so after the wrap below this value is unreachable.
        # `0` is the default for every source that is not a
        # `PrerollReplayingSource` -- the browser microphone and WebRTC
        # paths build no pre-roll buffer, so they pass zero and nothing
        # about them moves (DBG-03).
        session_recorder.set_preroll_bytes(getattr(source, "preroll_bytes", 0))
        source = _RecordingAudioSource(source, session_recorder)

    try:
        final = await _drain_to_final_transcript(
            source, stt, max_utterance_s, timings, clock=clock, poll_interval_s=poll_interval_s
        )
        timings.mark_stt_final()

        if barge_in is not None:
            # The exact moment `sources/runner.py`'s own listener may safely
            # become the sole reader of the source's raw frames -- `stt`'s
            # own frame-reading task has, by construction, already been
            # cancelled and awaited by the time `_drain_to_final_transcript`
            # returns (whether by a real final transcript or by the
            # timeout branch's `stream.aclose()`), so there is no window
            # here in which two readers are ever active (`sources/runner.py`
            # module docstring, D-09).
            barge_in.mark_transcript_done()

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
            await _cancel_state_task(pending_runs_task)
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
            # already uses for a losing tier -- not a second mechanism. The
            # pending-runs task (D-09) shares the identical fate for the
            # identical reason: a macro turn never builds the message list
            # its result would have joined either.
            await _cancel_state_task(state_task)
            await _cancel_state_task(pending_runs_task)
            outcome = await fire_macro(matched_macro, tool_host, tool_owners=tool_owners)
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
            await _speak(
                source,
                speaking_tts,
                timings,
                outcome.text or _DENIED_FALLBACK_REPLY,
                kind="answer",
                barge_in=barge_in,
                speech_lock=speech_lock,
                sink=sink,
            )
            await _emit_event(source, timings.to_event())
            timings.log()
            return

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        # CMD-09/D-07: a candidate id is what `TierReply.candidates` carries
        # and what must not be guessed between, but it is not what a person
        # says out loud. Populated from this turn's own injected live-state
        # fetch below when one ran -- never a second lookup -- and left empty
        # otherwise, in which case `_compose_clarifying_question` falls back
        # to speaking the id itself.
        friendly_names: dict[str, str] = {}
        states: dict[str, str] = {}
        # D-09: the pending-run block's own payload, threaded into
        # `_state_message` alongside `states` below -- `()` (the default)
        # when `pending_runs_fetch` was never given, matching `states`' own
        # `{}` default for the identical reason.
        pending_runs: "tuple[Any, ...]" = ()
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
            friendly_names = (
                {
                    entity["entity_id"]: entity["friendly_name"]
                    for entity in states_payload
                    if isinstance(entity, dict) and "friendly_name" in entity
                }
                if isinstance(states_payload, list)
                else {}
            )
        if pending_runs_task is not None:
            # Same T-01.1-17 posture `state_task` above already carries,
            # applied to D-09's own fetch: a workflow-repository read that
            # raised is logged and treated as "nothing scheduled known"
            # rather than ending the turn -- the operator can still cancel
            # or schedule a run this turn even when this particular read
            # failed, and failing the whole turn over it would be a worse
            # outcome than answering with an empty pending-run block.
            try:
                pending_runs_payload = await pending_runs_task
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "pending-runs fetch raised; continuing turn with nothing scheduled known"
                )
                pending_runs_payload = ()
            pending_runs = tuple(pending_runs_payload) if pending_runs_payload else ()
        if workflow_tool_host is not None:
            # WR-01 fix: scope this turn's `cancel_workflow_run`/
            # `append_workflow_steps` to exactly the ids `pending_runs`
            # resolved to, whether or not `pending_runs_fetch` actually
            # produced any (a fetch that raised, or was never given,
            # leaves `pending_runs = ()` above, which correctly clears
            # last turn's set rather than leaving it stale).
            workflow_tool_host.set_current_turn_run_ids(
                frozenset(run.id for run in pending_runs)
            )
        if state_task is not None or pending_runs_task is not None:
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

            messages.append(
                {"role": "system", "content": _state_message(states, pending_runs)}
            )
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
        _validate_tiers(tiers)

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
            await _speak(
                source,
                CachedTts(filler_cache),
                timings,
                FILLER_TEXT[phrase],
                kind="filler",
                barge_in=barge_in,
                speech_lock=speech_lock,
                sink=sink,
            )

        winner = await race_task
        timings.mark_tool_rounds_done()

        if winner.needs_clarification:
            # CMD-09/D-07: a third, exclusive outcome -- speaks a question
            # naming every candidate and stops there. No macro follow-up (the
            # macro check already ran, above, before a single tier task was
            # created) and no further tool round: this turn ends exactly the
            # way an ordinary answered turn does, through the same `_speak`/
            # `_emit_event`/`timings.log()` sequence, so nothing here keeps
            # the audio source open. D-08: the operator answers by waking the
            # assistant again -- VOICE-21 (holding the microphone open) is
            # out of scope and deliberately not built by this branch.
            timings.turn_outcome = "needs_clarification"
            question = _compose_clarifying_question(winner.candidates, friendly_names)
            await _speak(
                source, tts, timings, question, kind="answer", barge_in=barge_in, speech_lock=speech_lock, sink=sink
            )
            await _emit_event(source, timings.to_event())
            timings.log()
            return

        await _speak(
            source, tts, timings, winner.answer, kind="answer", barge_in=barge_in, speech_lock=speech_lock, sink=sink
        )
        await _emit_event(source, timings.to_event())
        timings.log()
    finally:
        # Covers every exit path above, including the two early returns --
        # exactly the turns whose folders an operator will want, and the
        # easiest ones to leak (D-13, T-02-22). A no-op when
        # `session_recorder` is `None`.
        if session_recorder is not None:
            session_recorder.close(timings)


def _validate_tiers(tiers: "list[brain_race.TierBrain]") -> None:
    """D-05, enforced here rather than only by convention (WR-01).

    `build_tiers` is the only production construction path and correctly
    derives `calls_tools` from position, but `run_turn` accepts a `tiers`
    list directly from any caller. `race_tiers` independently derives "top"
    as `max(tasks_by_index)` (`brain_race.py`); this checks that at most one
    tier is flagged `calls_tools=True` -- the actual danger WR-01 names, two
    racing tiers both able to reach the tool host -- and that whichever one
    is flagged (if any) is that same highest-index tier, so the two
    derivations of "top" can never disagree. Zero flagged tiers is accepted
    deliberately: it is strictly more conservative (nothing can call a
    tool), and `tests/test_turn_controller.py`'s own
    `test_criterion_4_a_confident_triage_tier_answers_with_zero_tool_calls`
    exercises exactly that shape to isolate `run_triage_tier`'s behavior.
    Raises rather than silently proceeding, the same raise-not-return
    doctrine `config.py` already uses everywhere else.
    """
    calls_tools_tiers = [tier for tier in tiers if tier.calls_tools]
    if len(calls_tools_tiers) > 1:
        raise BrainError(
            f"at most one tier may have calls_tools=True (D-05); got {len(calls_tools_tiers)} "
            f"across tier indices {[tier.index for tier in tiers]!r}"
        )
    if not calls_tools_tiers:
        return
    top_by_flag = calls_tools_tiers[0]
    top_by_index = max(tiers, key=lambda tier: tier.index)
    if top_by_flag is not top_by_index:
        raise BrainError(
            "the tier with calls_tools=True must be the highest-index tier (D-05); got "
            f"calls_tools=True on index {top_by_flag.index}, but the highest index present is "
            f"{top_by_index.index}"
        )


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
    stream = stt.stream(source.frames(), source.source_format())
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


def _compose_mixed_outcome_reply(pairs: "list[tuple[Any, Any]]") -> str:
    """One clause per action, in `reply.tool_calls` order, for a round where
    at least one result is error-shaped or a raised exception (CMD-07, D-14).

    `pairs` is the ordered `(tool_call, result)` list for one round -- exactly
    `zip(reply.tool_calls, results)`, so clause order is `reply.tool_calls`
    order by construction, never completion order (D-05); `asyncio.gather`
    already guarantees `results[i]` answers `reply.tool_calls[i]`.

    Composed here, in code, and never through a second `brain.chat` round:
    a second inference pass over a mixed-outcome batch could paraphrase a
    refusal reason, which would break the verbatim-refusal invariant
    `test_denied_reason_reaches_the_reply_verbatim` already guards for the
    single-action path (this module's own docstring, D-14). A failed
    action's clause carries `_result_text(result)` exactly as the boundary
    wrote it -- no prefix, no suffix, no rewording, no truncation of the
    reason text itself, the same rule `fire_macro`'s own docstring states for
    the macro path: a refusal never passes through a model, so there is
    nothing to summarise and nothing to shorten against. `_DENIED_FALLBACK_REPLY`
    covers only the case where the reason text itself is empty. This is what
    keeps a single-action batch's composed sentence byte-identical to the
    single-action path's own reply: one call, one error-shaped result,
    one clause -- exactly the boundary's own reason text, nothing joined
    around it.

    A successful action's clause is the fixed `_ACTION_SUCCEEDED_CLAUSE`
    phrase, carrying no name either -- attached to its action by its
    position in the sentence, the same "named by written order" doctrine
    `fire_macro`'s own docstring already uses ("the reason names the one
    written first"), not by a literal label repeated on every clause.

    An entry the gather returned as a raised exception is the one case named
    explicitly by this task: the call never reached a verdict at all, so its
    clause names the action's own tool and carries `_ACTION_DID_NOT_COMPLETE_CLAUSE`
    instead of a reason -- deliberately worded differently from a boundary
    refusal, since "it never even ran" and "it ran and was refused" are
    different facts about the house. Logged at `logger.exception` grade
    (full traceback) so a raised tool call is diagnosable from the log even
    though the spoken reply, by design, says only that it did not complete.
    """
    clauses: list[str] = []
    for tool_call, result in pairs:
        if isinstance(result, BaseException):
            logger.exception(
                "tool call %r raised instead of returning a result", tool_call.name, exc_info=result
            )
            clauses.append(f"{tool_call.name}: {_ACTION_DID_NOT_COMPLETE_CLAUSE}")
        elif _is_error(result):
            clauses.append(_result_text(result) or _DENIED_FALLBACK_REPLY)
        else:
            clauses.append(_ACTION_SUCCEEDED_CLAUSE)
    return "; ".join(clauses)


def _compose_clarifying_question(candidates: "tuple[str, ...]", friendly_names: "Mapping[str, str]") -> str:
    """The spoken question for a turn a triage tier's `needs_clarification`
    reply won (CMD-09, D-07).

    Composed here, in code, from the reply's own candidates -- never through
    a second `brain.chat` round (D-14): a second inference pass over the
    candidate list is exactly how a question about two entities becomes a
    model's own paraphrase of one, the same failure `_compose_mixed_outcome_reply`
    already exists to close off for a tool-round summary. The same candidate
    list, in the same order, always produces the same sentence -- nothing
    here reads a clock, a random source, or any per-call state.

    An entity id is what the code holds and what must not be guessed
    between, but it is not what a person says out loud: `friendly_names`,
    when it carries an entry for a candidate, is preferred and the id is
    kept out of the sentence entirely for that candidate. `friendly_names`
    is built by `run_turn` from this turn's own injected live-state fetch
    (the same payload `_state_message` renders from) -- never a second
    lookup this function performs itself. When no friendly name is known
    for a candidate, the id is spoken instead: worse than a name, better
    than silence. Every candidate is named regardless -- naming two of
    three is a worse failure than naming none.
    """
    named = [friendly_names.get(candidate, candidate) for candidate in candidates]
    return f"{_CLARIFYING_QUESTION_CARRIER} {', '.join(named)}?"


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

    `commitment`, when given, is set the instant one round's dispatch begins --
    before `asyncio.gather` is awaited, not after any one call returns (CR-01,
    widened by Task 1 of plan 04-01). A stdio round trip to the MCP child
    cannot be un-sent once dispatched, and with three or more calls genuinely
    in flight at once, waiting for the first result to land would leave a
    window in which a racing triage tier's confident reply could still end
    the race while uncancellable service calls are running -- exactly the
    hazard CR-01 closed for the single-call case. From the moment dispatch
    begins, `race_tiers` must not let a triage tier's confident reply end the
    race in this tier's place; only this tier's own settled outcome may
    describe what actually happened.
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

        # CMD-06: every call in this round is dispatched together and run to
        # completion -- `asyncio.gather(..., return_exceptions=True)`, never
        # `asyncio.TaskGroup`, whose cancel-every-sibling-on-first-exception
        # semantics would abort actions two and three the instant action one
        # raised, which is the exact behaviour this task removes (the same
        # reasoning `turn/brain_race.py`'s own module docstring already
        # records for its own choice not to use a `TaskGroup`). `commitment`
        # is set the instant dispatch begins, above, not after a result
        # returns -- see this function's own docstring for why. `gather`
        # always returns results positionally aligned with the awaitables it
        # was given, so `results[i]` is `reply.tool_calls[i]`'s own result
        # regardless of which call happens to finish first (D-05) -- there is
        # no separate correlation step to get wrong.
        if commitment is not None:
            commitment.committed = True
        results = await asyncio.gather(
            *(tool_host.call_tool(tc.name, tc.arguments) for tc in reply.tool_calls),
            return_exceptions=True,
        )

        # Collect every result before deciding anything (D-06): a batch
        # containing any error-shaped or raised entry is composed in code,
        # below, and returned directly -- never through a second `brain.chat`
        # round (D-14). A batch where every call succeeded falls through to
        # the loop-to-the-next-round behaviour, byte-identical to before this
        # plan.
        if any(isinstance(result, BaseException) or _is_error(result) for result in results):
            return _compose_mixed_outcome_reply(list(zip(reply.tool_calls, results)))

        # Every call in this round succeeded: byte-identical to the
        # pre-concurrency behaviour -- one `role: tool` message per call, in
        # `reply.tool_calls` index order, and the loop continues to the next
        # round where the next `brain.chat` call composes the confirmation.
        for i, result in enumerate(results):
            # A non-2xx Home Assistant response (see `handle_call_service`)
            # is not a refusal -- it comes back as an ordinary, non-error
            # result whose content names the failure, so the next brain call
            # reports it rather than confirming success.
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": _result_text(result)})

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
    barge_in: "_BargeInMonitor | None" = None,
    speech_lock: "asyncio.Lock | None" = None,
    sink: "SinkFormat | None" = None,
) -> None:
    """Speak one utterance, and mark whichever timing(s) `kind` calls for.

    `sink=None` -- the default, and every caller that predates 260922-cts
    -- reproduces the exact browser-PCM request `tts.synthesize` always
    made before this fix. `run_turn` resolves the real value once, from
    the turn's own source, and passes it to every `_speak` call it makes
    (filler, clarifying question, and answer alike) so a camera turn's
    synthesis call -- and a `CachedTts` lookup -- asks for the sink the
    camera speaker actually plays.

    `speech_lock=None` -- the default, and every caller that predates
    plan 05-03 -- runs this function exactly as it always has: nothing
    below changes for a caller that never heard of a second speaker.
    Given a lock, the whole synthesize-and-write loop below runs inside
    it, not around a single `source.send_audio()` call: this phase puts a
    second speaker in the room for the first time (a scheduled step's own
    utterance, `app.py`'s scheduled-speech closure), and a scheduled
    utterance's chunks interleaved with a live reply's chunks on the same
    FIFO is garbled audio, not two sentences (05-CONTEXT.md's own opening
    failure class, applied to the speaker rather than a light). Held for
    the whole loop, not acquired per chunk, because the point is one
    utterance finishes before the other's first chunk is written -- a
    per-chunk lock would still let the two interleave, just at chunk
    granularity instead of byte granularity.

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

    VOICE-07 (D-11): `barge_in=None` (the default, and every caller that
    predates this plan) skips every check below -- byte-for-byte the same
    loop this function has always run. When given, `tts_xai.py`'s own
    module docstring already states text-to-speech is a single REST
    response holding the whole utterance -- there is nothing left to cancel
    by the time an operator would plausibly interrupt (RESEARCH.md Pitfall
    6), so the interrupt lands here, on the write loop, between chunks,
    rather than on `tts.synthesize()`. Once requested, this loop stops
    calling `source.send_audio()` -- CONTEXT.md's "stops the FIFO write
    immediately" -- but keeps iterating the (already-downloaded, already
    fully paid for) remaining chunks with no further cost, only to learn
    how far the cut reply would have gone (D-11's "how far it got"). This
    never starts anything: stopping playback and starting a turn are
    separate paths, and only the wake path may ever start one (T-02-24) --
    nothing below calls `run_turn`, matches a macro, or opens a
    transcription stream.

    Plan 02-12 (D-18): every chunk actually written to `source.send_audio()`
    is also appended to `barge_in.trace` -- the emitted-output record
    `sources/runner.py`'s correlation compares microphone energy against.
    `trace` is read with `getattr`, never assumed present, so a `barge_in`
    double that carries no `trace` attribute (every test predating this
    plan) is untouched: the same duck-typed-optional discipline this
    module already applies to `source.barge_in` and `source.send_event`.
    Appended only for a chunk that is actually sent -- never one already
    skipped by the `interrupted` branch above, matching CONTEXT.md's "stops
    the FIFO write immediately": nothing is appended for bytes that were
    never written.
    """

    async def _one_delta() -> AsyncIterator[str]:
        yield reply_text

    async def _synthesize_and_write() -> None:
        first_audio_marked = False
        interrupted = False
        chunks_sent = 0
        chunks_total = 0
        async for chunk in tts.synthesize(_one_delta(), sink=sink):
            chunks_total += 1
            if interrupted:
                # Nothing left to cancel (module docstring) -- the rest of this
                # loop only counts how many chunks the cut reply would have
                # held, without writing any more of them to the speaker.
                continue
            if not first_audio_marked:
                first_audio_marked = True
                now = _time.monotonic()
                if timings.first_audio_at is None:
                    timings.first_audio_at = now
                if kind == "answer" and timings.answer_audio_at is None:
                    timings.answer_audio_at = now
                if barge_in is not None and barge_in.enabled:
                    barge_in.mark_playback_started(now)
            if barge_in is not None and barge_in.enabled and barge_in.interrupt_requested:
                interrupted = True
                timings.turn_outcome = "barged_in"
                continue
            await source.send_audio(chunk)
            chunks_sent += 1
            if barge_in is not None and barge_in.enabled:
                trace = getattr(barge_in, "trace", None)
                if trace is not None:
                    trace.append(chunk)

        if interrupted:
            await _emit_event(
                source,
                {"type": "reply.interrupted", "chunks_sent": chunks_sent, "chunks_total": chunks_total},
            )
        await _emit_event(source, {"type": "reply.text", "text": reply_text})

    if speech_lock is not None:
        async with speech_lock:
            await _synthesize_and_write()
    else:
        await _synthesize_and_write()


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
