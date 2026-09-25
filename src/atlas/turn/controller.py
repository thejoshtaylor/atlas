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
like the real ones from `atlas.providers.base`.

`Denied`, when a tool call is refused, reaches the reply path directly: its
`reason` becomes the text handed to text-to-speech, with no second language
model call in between to reword it. That is the one thing this path must
not do, per `atlas_mcp.safety.Denied`'s own doctrine.

Plan 01.1-04 replaces the single `_run_tool_rounds` call with a raced tier
list (`turn/brain_race.py`): every tier is dispatched concurrently, the
first confident reply wins, and a holding phrase from the startup cache
covers the wait past `filler_after_ms` if the race is still running. `tiers`
defaults to `None`, in which case `run_turn` wraps its `brain` positional
argument in a one-element tier list and races that -- there is one code
path, and a single tier is its degenerate case, not a bypass. This module
imports `atlas.turn.brain_race` at load time; `brain_race.py` imports
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
open past the question. `turn_outcome` gains its own `"needs_clarification"`
value, distinct from an ordinary answer, an empty reply, and a round cap.

Plan 09-07 (D-06) replaces phase 4's own D-08 for this question on a
source that has a follow-up channel attached: the operator now answers
inside the same brief no-wake-word window a calendar confirmation opens
(plan 09-06), and the answer continues the original request through the
ordinary pipeline rather than requiring a fresh wake. A source with no
channel attached is unchanged -- the operator still answers by waking the
assistant again, and VOICE-21 is deliberately out of scope there.

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
import re
import time as _time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Literal, Mapping, Protocol

from atlas_mcp.google_tools import UNREACHABLE_KEY

from atlas.audio.channels import stt_view
from atlas.config import MacroConfig
from atlas.providers.base import BrainError
from atlas.transports.base import SourceFormat
from atlas.providers.tier_reply import DEFAULT_FILLER, FILLER_TEXT, FillerPhrase, TierReply
from atlas.providers.tts_cache import CachedTts
from atlas.audio.cue import wake_cue as wake_cue_audio
from atlas.speaker.fifo_writer import SpeakerError
from atlas.providers.tts_xai import SinkFormat
from atlas.session.recorder import SessionRecorder
from atlas.timing import TurnTimings
from atlas.turn import brain_race
from atlas.turn.early_finalize import wait_for_end_of_speech
from atlas.turn.follow_up import (
    HA_CALL_SERVICE_TOOL,
    MAX_CHAINED_FOLLOW_UPS,
    AnswerScope,
    FollowUpChannel,
    FollowUpRequest,
    estimate_playback_end,
)
from atlas.turn.handoff import (
    AMENDED_CONTINUATION_REFUSAL,
    CODE_ONLY_REFUSAL,
    HandoffContext,
    HandoffSlot,
    dispatch_handoff,
    is_code_only_tool,
    parse_handoff,
)
from atlas.turn.local_intent import match_on_off
from atlas.turn.macros import fire_macro, match as match_macro, normalize
from atlas.turn.pending_action import (
    BULK_REFUSAL_REPLY,
    CANCELLED_REPLY,
    HANDOFF_NOT_ALONE_REPLY,
    handle_confirmation_reply,
)
from atlas.turn.transcript_guard import asks_for_information, is_no_command
from atlas.turn.wake_echo import is_wake_only

logger = logging.getLogger("atlas.turn.controller")

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

# 260922-woc: spoken whenever a turn would otherwise end silent past the
# point a language model was ever consulted -- a winning `TierReply` whose
# `answer` is empty or whitespace-only (a top tier can win the race with
# `confident=False` and no fallback text at all; `TierReply`'s own
# validator only requires a non-empty answer when `confident` is True), and
# the brain-timeout path below. Precached at startup (`tts.precache` in
# `config.example.yaml`) specifically so this phrase is available from
# every sink's cache, never a live synthesis call on top of a turn that has
# already run long or failed to answer.
_CANNOT_DO_REPLY = "i can't do that one"

# 260922-lim: spoken after a local on/off intent's tool call succeeds --
# "done" is the operator's own confirmation, not a restatement of what was
# asked for, and it is precached at startup (`tts.precache` in
# `config.example.yaml`) for exactly this fast path.
_DONE_REPLY = "done"

# 260924-4it: the tool names `_round_settles_as_done` may speak "done" for,
# with no second model round -- an exact match on the name the model
# called (the offered name), never `readOnlyHint`. atlas-ha declares no
# tool annotations, the MCP spec defaults `readOnlyHint` to False (so an
# unannotated read tool would count as an action), and a plugin process
# must not be able to opt itself into a canned confirmation. A tool not on
# this list -- a read, a workflow tool, a plugin tool, or a
# collision-prefixed `{slug}__ha_call_service` -- always takes the second
# round. The cost of missing this list is only latency.
_DONE_SHORTCUT_TOOLS: frozenset[str] = frozenset({"ha_call_service"})

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

# R3-IN-05 (D-24): a Home Assistant entity id, `domain.object_id`. A
# triage clarification whose candidates all have this shape scopes its
# answer to those entities (`_triage_clarification_scope`). A plugin
# display name or a scheduled-run summary never matches: both are spoken
# words, with capitals or spaces and no dot.
_ENTITY_ID_SHAPE = re.compile(r"[a-z0-9_]+\.[a-z0-9_]+")

# GOOG-12, plan 09-05 Task 3: appended, by code, to an ordinary answer for
# every account this turn's own tool round could not reach (`handoff_slot.
# unreachable_accounts`) that the answer does not already name -- an
# unreachable account must never be silently left out of a spoken calendar
# answer. Fixed and asserted-on, the same "never model-composed" discipline
# `_CLARIFYING_QUESTION_CARRIER` already follows; only `label` varies.
_UNREACHABLE_ACCOUNT_NOTE = " i can't reach your {label} account right now."

# How often the silence-timeout guard rechecks its deadline while waiting on
# an STT event that may never arrive. Real events short-circuit this --
# `asyncio.wait` returns the instant the event lands, so this interval only
# bounds how quickly a *stuck* session notices the deadline passed, not the
# latency of an ordinary turn.
_DEFAULT_POLL_INTERVAL_S = 0.05

# 10-07-PLAN.md (D-17): the prefix every edge `speech_signals` event is
# recorded under -- `edge.vad.start`, `edge.vad.end`, `edge.doa`,
# `edge.latency` -- so a reader of `events.jsonl` can tell an edge-source
# event apart from a browser/camera one without depending on the exact
# type strings `transports/edge.py` happens to use today. A module
# constant, not an import from `transports/`: this module has no concrete
# transport dependency (D-02), and the literal string is all this prefix
# needs to be.
_EDGE_EVENT_PREFIX = "edge."

# 260924-4iv (item a): the one value `_read_prefetched` returns for "this
# fetch was never given, never finished in time, or raised" -- distinct
# from every real fetch result (`None` is itself a legitimate scripted
# return in a few existing tests), so a caller can tell "genuinely
# unavailable" apart from "read successfully, and it was empty/None."
_UNAVAILABLE = object()


class _AudioSource(Protocol):
    def frames(self) -> AsyncIterator[bytes]: ...
    async def send_audio(self, chunk: bytes) -> None: ...
    def source_format(self) -> SourceFormat: ...


class _SttProvider(Protocol):
    def stream(
        self,
        frames: AsyncIterator[bytes],
        source_format: SourceFormat,
        *,
        finalize: "asyncio.Event | None" = None,
    ) -> AsyncIterator[Any]: ...


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


def _continuation_messages(incoming: FollowUpRequest) -> "list[dict[str, Any]]":
    """The messages a continuation turn -- an amendment's own reply, or a
    clarification's own answer -- runs with, ahead of its own new user
    message (plan 09-07, D-06): every earlier exchange this follow-up
    chain already carries (`incoming.prior_messages`, empty for the first
    link), then the original request and the question it was asked.
    Inserted before the new user message (`run_turn`'s own message-
    building section below), so a tier sees
    catalog/state/prior-exchange/user in that order -- the same shape
    plan 09-06 already established for an amendment, extended here to a
    clarification's own answer and to a chain of more than one link.
    """
    return [
        *incoming.prior_messages,
        {"role": "user", "content": incoming.original_transcript},
        {"role": "assistant", "content": incoming.question},
    ]


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
    wake_phrase: str | None = None,
    wake_cue: bool = False,
    brain_turn_timeout_s: float = 25.0,
    local_intents: bool = False,
    state_timeout_ms: float = 500.0,
    state_domains: frozenset[str] | None = None,
    handoff_context: "HandoffContext | None" = None,
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
    importing `atlas.workflow.tool.WorkflowToolHost` at module level
    for the same load-order reason this function already defers its
    `atlas.app` import below. `None` (the default, and every caller
    that predates this fix) skips the call -- no behavior change for a
    caller with no workflow tool host to scope.

    `tool_owners` (06-CONTEXT.md D-12, plan 06-05) is forwarded unchanged
    to `fire_macro` on the macro path below -- this function has no
    opinion of its own about plugin ownership, only a caller-supplied
    answer to "how many plugins currently publish this bare tool name" it
    passes through. `None` (the default, and every caller that predates
    this plan) reproduces `fire_macro`'s own pre-existing behavior exactly.

    `wake_phrase` (260922-woc): the configured wake phrase
    (`config.wake.phrase`), threaded through only by the camera's own
    wake-word caller (`app.py`'s `_make_run_turn_for_source`) -- the
    browser and WebRTC routes pass nothing, which is what `None` (the
    default) means: skip `is_wake_only` entirely and behave exactly as
    every caller that predates this plan does. When given, a first final
    transcript that is nothing but the wake phrase (or a mangled tail of
    it -- see `turn/wake_echo.py`) does not close the turn; it drains the
    STT stream a second time, once only, for the command the operator
    actually spoke, with whatever's left of `max_utterance_s`'s own
    deadline. `stt_final`/`barge_in.mark_transcript_done()` are held back
    until whichever drain turns out to be the last one, so a source's
    barge-in listener never starts reading `frames()` while this
    function's own second drain still needs to be its sole reader.

    `brain_turn_timeout_s` (260922-woc, default 25.0) bounds the wait on
    the tier race as a whole, on top of whatever `filler_after_ms` already
    covers: a race that is still running past this deadline is cancelled
    -- every tier task cancelled and awaited, since `race_tiers`'s own
    cleanup never runs when the cancellation reaches it from outside
    (`asyncio.wait_for` raises `CancelledError` into the task, which
    `race_tiers`' `except Exception` does not catch) -- and the turn
    speaks the cached "i can't do that one" phrase rather than sitting
    silent forever, with `turn_outcome = "brain_timeout"`.

    `local_intents` (260922-lim, default `False`) gates the local on/off
    matcher (`turn/local_intent.py`): when `True` and a `tool_host` is
    given, a plain on/off command is matched against this turn's own live
    entity fetch and, on a match, calls the tool directly -- no tier race,
    no language model round trip at all. `False` for every caller that
    predates this plan, so nothing about an existing caller's behavior
    changes until it opts in (`app.py`'s camera `run_turn` call passes
    `config.brain.local_intents`).

    `state_timeout_ms`/`state_domains` (260924-4iv, items a/b): the one
    deadline, past `stt_final_at`, this turn may spend waiting on
    `state_task`/`pending_runs_task` together, and the domain filter
    `_state_message` applies to the live-state block. `state_timeout_ms`
    defaults to 500 (`BrainConfig.state_timeout_ms`'s own default);
    `state_domains=None` (the default) keeps every domain, matching every
    caller that predates this plan. `state_task` is read at most once,
    through `_read_prefetched`: the local on/off block above reads it
    first when that block runs at all, and the message-building section
    below reuses that same result rather than reading a second time. A
    read that misses the deadline is cancelled and awaited, never left
    running, and this turn's brain sees an explicit "not available" line
    (`app.py::_STATE_UNAVAILABLE_LINE`/`_PENDING_RUNS_UNAVAILABLE_LINE`)
    rather than an empty list that could be misread as "nothing is on" or
    "nothing is scheduled." Degraded behavior, stated plainly: the cached
    catalog prompt still names every entity id regardless of this
    timeout, so the brain is never blind to what exists, only to what
    state it is currently in; the local matcher does not run without a
    real state list; a clarifying question falls back to speaking entity
    ids; and the MCP child's own safety policy still runs on every
    `call_service` whatever this turn saw, so a turn with no live state
    can never reach an entity the operator marked off limits.
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

    # Plan 09-04 (D-06): read off the *unwrapped* source, the same way and
    # for the same reason `barge_in` is above -- `_RecordingAudioSource`
    # forwards no arbitrary attribute. `None` for every source with no
    # follow-up channel attached (every caller today; `SourceRunner`
    # attaches one starting plan 09-06), which is what makes
    # `atlas.turn.pending_action.CONFIRMATION_UNAVAILABLE_REPLY` the honest
    # answer to every proposal until then.
    follow_up: "FollowUpChannel | None" = getattr(source, "follow_up", None)

    # Plan 09-06 (D-06): the request the *previous* turn left on this
    # source's own channel for this turn to answer, or `None` for an
    # ordinary wake turn -- read once, here, and reused for every branch
    # below that needs to know whether this is a follow-up turn. Never
    # cleared by this function: `SourceRunner._run_follow_ups` (the only
    # caller that ever sets it) is also what decides whether to attach
    # another one for the next turn in the chain.
    incoming: "FollowUpRequest | None" = follow_up.incoming if follow_up is not None else None

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

    # 260922-cue: a chime tells the operator when to speak. Only a source
    # that declares its own sink (the camera) has a speaker to play it on.
    # Plan 09-06: skipped for a follow-up turn -- there is no fresh wake
    # hit to cue, and the window already opened silently after the
    # readback's own echo tail (D-09).
    if wake_cue and sink is not None and incoming is None:
        await _play_wake_cue(source, sink, speech_lock)

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

    # 10-05-PLAN.md (D-09 through D-13): read off the same still-unwrapped
    # `source` `preroll_bytes` below is read off, and for the same reason --
    # `_RecordingAudioSource` forwards no arbitrary attribute, so after that
    # wrap this value would be unreachable. Read unconditionally, not inside
    # `if session_recorder is not None:` below, so a deployment with session
    # recording off still finalizes an edge turn's speech-to-text early.
    # `None` for every source that predates this plan (every browser/camera
    # source, and every test double that carries no `speech_signals`), in
    # which case `_drain_to_final_transcript` behaves exactly as it did
    # before this plan.
    speech_signals = getattr(source, "speech_signals", None)

    # 10-07-PLAN.md (D-17): a Pi's own vad.start/vad.end/doa/latency events
    # land in this turn's session, prefixed `edge.`, for the turn's whole
    # life -- never through `_emit_event` (which reaches the observer feed
    # and the browser), because DoA is recorded only, never shown. Only
    # when both `speech_signals` and `session_recorder` exist: no
    # `speech_signals` means no edge source; no `session_recorder` means
    # no session to record into. `replay_segment=True` (the default)
    # means a turn that subscribes mid-segment still gets that segment's
    # own vad.start and every DoA reading already published, not only
    # what arrives from this instant forward.
    _edge_event_unsubscribe: "Callable[[], None] | None" = None
    if speech_signals is not None and session_recorder is not None:

        def _record_edge_event(event: dict[str, Any]) -> None:
            session_recorder.record_event({**event, "type": _EDGE_EVENT_PREFIX + event["type"]})

        _edge_event_unsubscribe = speech_signals.subscribe(_record_edge_event, replay_segment=True)

    if session_recorder is not None:
        # Resolved from the source's own declaration, never assumed (D-13),
        # and wrapped before `_drain_to_final_transcript` is ever awaited
        # below -- there is no earlier point at which `source.frames()`
        # could be read, so this is the only tap this function opens.
        fmt = source.source_format()
        session_recorder.set_audio_format(fmt.encoding, fmt.sample_rate, fmt.channels)
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
        turn_deadline = clock() + max_utterance_s
        # Plan 09-06: a follow-up turn's own onset deadline -- how long
        # `_drain_to_final_transcript` waits for the *first* event before
        # giving up on the window entirely (D-06, D-09). `None` for an
        # ordinary wake turn (every caller that predates this plan), and
        # also `None` for a follow-up turn whose channel carries no window
        # (a test double, or a source with no timing attached) -- in
        # which case this behaves exactly like an ordinary turn's own
        # `max_utterance_s` bound.
        onset_deadline: float | None = None
        if incoming is not None and follow_up is not None:
            if follow_up.window_opens_at is not None and follow_up.window_s is not None:
                onset_deadline = follow_up.window_opens_at + follow_up.window_s
        final = await _drain_to_final_transcript(
            source,
            stt,
            max_utterance_s,
            timings,
            clock=clock,
            poll_interval_s=poll_interval_s,
            onset_deadline=onset_deadline,
            speech_signals=speech_signals,
            # 10-05-PLAN.md: an ordinary wake turn (`incoming is None`)
            # whose segment already ended before speech-to-text even opened
            # holds only the wake phrase -- finalizing at once sends it
            # straight to the wake-only path below. A follow-up turn
            # (`incoming is not None`) must NOT finalize on that same
            # already-ended state; it waits for a genuinely new segment.
            finalize_if_already_ended=incoming is None,
        )
        final_text = getattr(final, "text", "") if final is not None else ""

        if incoming is None and wake_phrase and final_text and is_wake_only(final_text, wake_phrase):
            # 260922-woc: the operator paused after the wake phrase, and
            # `stt.endpointing_ms` ended the utterance there -- the command
            # is still coming. Drain again, once (no loop): a second
            # wake-only reply here is treated as VOICE-08's empty-transcript
            # case below, never a third attempt. `remaining_s` is whatever is
            # left of this turn's own `max_utterance_s` budget, not a fresh
            # allowance -- an operator who pauses twice as long still cannot
            # hold this turn open past its original deadline.
            remaining_s = max(0.0, turn_deadline - clock())
            final = await _drain_to_final_transcript(
                source,
                stt,
                remaining_s,
                timings,
                clock=clock,
                poll_interval_s=poll_interval_s,
                speech_signals=speech_signals,
                # 10-05-PLAN.md: the command is still coming -- this second
                # drain must wait for the NEXT segment's own vad.end, never
                # finalize immediately on the already-ended state the first
                # drain's wake-only segment left behind.
                finalize_if_already_ended=False,
            )
            final_text = getattr(final, "text", "") if final is not None else ""

        timings.mark_stt_final()
        # 260924-4iv (item a): one absolute deadline, in `_time.monotonic()`'s
        # own domain (real time, since `asyncio.wait_for` waits in real
        # time) -- every bounded read below shares this one instant rather
        # than each computing its own from a possibly-later "now".
        state_deadline = _time.monotonic() + state_timeout_ms / 1000
        # The boundary between hearing and thinking, for a page that shows it.
        await _emit_event(source, {"type": "transcript.final", "text": final_text})

        if barge_in is not None:
            # The exact moment `sources/runner.py`'s own listener may safely
            # become the sole reader of the source's raw frames -- `stt`'s
            # own frame-reading task has, by construction, already been
            # cancelled and awaited by the time `_drain_to_final_transcript`
            # returns (whether by a real final transcript or by the
            # timeout branch's `stream.aclose()`), so there is no window
            # here in which two readers are ever active (`sources/runner.py`
            # module docstring, D-09). Held back until whichever drain above
            # is the last one (260922-woc): a wake-only first drain leaves
            # `source.frames()` still needed by this function's own second
            # drain, and starting the barge-in listener before that second
            # drain finishes would give `frames()` two concurrent readers.
            barge_in.mark_transcript_done()

        # A-CR-02: True for the turn that continues an `amended`
        # confirmation reply, and for every later turn in the same
        # follow-up chain (`incoming.proposals_only`, set below). Never
        # True for an ordinary wake turn. Read by the tier-dispatch loop
        # further down to narrow both the offered tool schema and, as the
        # structural backstop, which tool names `_run_tool_rounds` will
        # actually dispatch (`HandoffContext.proposal_tool_names`). The same flag
        # keeps every triage tier out of the race, and is copied onto every
        # follow-up this turn requests. Any change the operator describes
        # in a no-wake-word window can therefore only ever become a NEW
        # pending_action, never an action that runs with no confirmation
        # step at all, however many links the chain has.
        restrict_tools_to_proposals = False
        # R3-IN-05 (D-24): what this turn may reach when it answers a
        # follow-up -- `None` for an ordinary wake turn, and for an
        # amendment whose chain began with a wake turn's own proposal.
        # Narrows both the offered schema and dispatch below, next to
        # `restrict_tools_to_proposals`, and is copied (narrowed, never
        # widened) onto every follow-up this turn requests.
        answer_scope: "AnswerScope | None" = None

        # Plan 09-06 (D-08, D-09): a follow-up turn's own reply to a
        # stored confirmation -- this entirely replaces the ordinary
        # empty-transcript branch below for this turn (silence speaks
        # `CANCELLED_REPLY`, never `_NO_SPEECH_REPLY`), and never reaches
        # a macro or the local on/off matcher. Plan 09-07 adds a sibling
        # branch immediately below for `incoming.kind == "clarification"`
        # -- an `incoming` of any other kind falls through and is treated
        # as an ordinary turn, exactly as if no follow-up channel existed.
        if incoming is not None and incoming.kind == "confirmation":
            await _cancel_state_task(state_task)
            await _cancel_state_task(pending_runs_task)
            confirmation_outcome = await handle_confirmation_reply(
                handoff_context, incoming, final_text, timeout_s=brain_turn_timeout_s
            )
            if not confirmation_outcome.amended:
                timings.turn_outcome = confirmation_outcome.turn_outcome
                confirmation_tts = _tts_for_precached_fallback(
                    filler_cache, sink, confirmation_outcome.reply_text, tts
                )
                await _speak(
                    source,
                    confirmation_tts,
                    timings,
                    confirmation_outcome.reply_text,
                    kind="answer",
                    barge_in=barge_in,
                    speech_lock=speech_lock,
                    sink=sink,
                )
                await _emit_event(source, timings.to_event())
                timings.log()
                return
            # D-08: an amendment supersedes the stored action (already
            # resolved `superseded` by `handle_confirmation_reply` above)
            # and runs the ordinary pipeline once more, with every
            # earlier exchange this chain already carries inserted ahead
            # of the operator's own amendment -- never a second call to
            # `handle_confirmation_reply`, and never the macro or
            # local-intent blocks below (an amendment is never a macro
            # phrase or an on/off command).
            prior_exchange: "list[dict[str, Any]] | None" = _continuation_messages(incoming)
            restrict_tools_to_proposals = True
            answer_scope = incoming.answer_scope
        elif incoming is not None and incoming.kind == "clarification":
            # Plan 09-07 (D-06): a clarification's own answer -- "home"
            # to "which account -- home, work?" -- runs the ordinary
            # pipeline once, the same way an amendment does immediately
            # above, with the whole chain inserted ahead of the answer
            # itself, skipping the macro and local-intent blocks below.
            # There is no stored pending action to resolve here (a
            # clarification is never a `pending_action` handoff), so
            # silence or an unreadable reply is handled directly, right
            # here, rather than through `handle_confirmation_reply` --
            # the same `CANCELLED_REPLY`/`"follow_up_silence"` shape a
            # stored confirmation's own silence already uses (D-09), and
            # nothing is stored either way.
            if not final_text or is_no_command(final_text, wake_phrase):
                await _cancel_state_task(state_task)
                await _cancel_state_task(pending_runs_task)
                timings.turn_outcome = "follow_up_silence"
                cancelled_tts = _tts_for_precached_fallback(filler_cache, sink, CANCELLED_REPLY, tts)
                await _speak(
                    source,
                    cancelled_tts,
                    timings,
                    CANCELLED_REPLY,
                    kind="answer",
                    barge_in=barge_in,
                    speech_lock=speech_lock,
                    sink=sink,
                )
                await _emit_event(source, timings.to_event())
                timings.log()
                return
            prior_exchange = _continuation_messages(incoming)
            # A-CR-02: a clarification that a restricted turn asked keeps
            # that turn's restriction -- the answer is still spoken in a
            # no-wake-word window.
            restrict_tools_to_proposals = incoming.proposals_only
            # R3-IN-05 (D-24): the answer reaches only what the clarifying
            # turn asked about. A request with no recorded scope fails
            # closed (no tool) -- unless `proposals_only` already governs
            # it, which keeps that chain's own behavior unchanged.
            answer_scope = incoming.answer_scope
            if answer_scope is None and not incoming.proposals_only:
                answer_scope = AnswerScope(tool_names=frozenset())
        else:
            prior_exchange = None

        # 260923-kao: a wake-only or filler-only final transcript ends the
        # turn the same way VOICE-08's empty-transcript case does below --
        # it never reaches a macro, the local on/off matcher, or the tier
        # race, and it shows as `turn_outcome = "no_command"` in
        # `timing.json`. This is what makes the second-drain comment above
        # true for a second wake-only reply: without this guard, that
        # second drain's own transcript was never checked, and a repeated
        # wake phrase or a bare "It's" fell straight through to the brain.
        # Plan 09-06/09-07: `prior_exchange is not None` means this turn
        # is an amendment or a clarification's own answer continuing past
        # the branches above, both of which already proved `final_text`
        # is real content -- this check is therefore always False on
        # that path and never revisits it.
        if prior_exchange is None and (not final_text or is_no_command(final_text, wake_phrase)):
            # VOICE-08's two cases end the turn the same way, with no language
            # model call: `final is None` is RESEARCH.md Pitfall 3's second
            # case (the provider never sent anything at all, closed here by
            # the client-side timeout); a `FinalTranscript` whose text is
            # empty is the first case (something arrived and decoded to
            # nothing). `turn_outcome` keeps the two distinguishable in the
            # log even though the reply path is shared.
            #
            # 260922-woc: this used to emit `_NO_SPEECH_REPLY` as a
            # `reply.text` event only, with no audio ever sent -- a camera
            # operator who paused, then said nothing intelligible, heard
            # silence with no indication the turn had ended at all. `_speak`
            # both plays the cached phrase and emits the identical event
            # itself, so this is one call doing what used to be two, not an
            # added step. The no-language-model-call guarantee above is
            # untouched: nothing here reaches `brain`.
            if final is None:
                timings.turn_outcome = "timeout"
            elif not final_text:
                timings.turn_outcome = "empty_transcript"
            else:
                timings.turn_outcome = "no_command"
            await _cancel_state_task(state_task)
            await _cancel_state_task(pending_runs_task)
            no_speech_tts = _tts_for_precached_fallback(filler_cache, sink, _NO_SPEECH_REPLY, tts)
            await _speak(
                source,
                no_speech_tts,
                timings,
                _NO_SPEECH_REPLY,
                kind="answer",
                barge_in=barge_in,
                speech_lock=speech_lock,
                sink=sink,
            )
            await _emit_event(source, timings.to_event())
            timings.log()
            return

        # The macro check runs here, before a single tier task is created:
        # placed after the dispatch below, the model round trip would already
        # have been paid and MACRO-02 would be false while every other
        # behavioral test still passed. `match_macro` on an empty `macros` tuple
        # (the default) always returns `None`, so a caller that predates this
        # plan reaches the tier race exactly as before, at no observable cost.
        # Plan 09-06/09-07: `prior_exchange is not None` means this is an
        # amendment or a clarification's own answer continuing past the
        # branches above -- it never matches a macro (09-07's own
        # `<behavior>`: "skipping macros and the local on/off intent").
        matched_macro = match_macro(macros, final_text) if prior_exchange is None else None
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

        # 260922-lim: after the macro check (macros win) and before the tier
        # race -- a plain on/off command needs neither a tier race nor a
        # language model round trip at all. `local_intents=False` (the
        # default, and every caller that predates this plan) skips this
        # block entirely, at no observable cost. `tool_host is None` skips
        # it too: there would be nowhere to send the matched call.
        # 260924-4iv (item a): `state_task` is read at most once per turn,
        # through `_read_prefetched` and the one shared `state_deadline`
        # above. `state_read` is this turn's own flag for "already read" --
        # the local on/off block immediately below reads it first when
        # that block runs at all; the message-building section further
        # down reuses `state_result` rather than reading a second time.
        # This folds the separate, longer 1.5 s bound the local-intent
        # path used to apply on top of the tier path's own un-timed await
        # into the one shared deadline: `local_intents` is on by default
        # on the camera path, so a separate longer bound here would make
        # item (a) a no-op on the main path.
        state_read = False
        state_result: Any = _UNAVAILABLE

        # Plan 09-06/09-07: skipped on the amendment or clarification-
        # answer path for the identical reason the macro check above is
        # -- neither is ever an on/off command.
        if prior_exchange is None and local_intents and tool_host is not None:
            entities: list[dict[str, Any]] = []
            if state_task is not None:
                state_result = await _read_prefetched(
                    state_task, state_deadline, "state fetch", timings.turn_id, state_timeout_ms
                )
                state_read = True
                if isinstance(state_result, list):
                    entities = state_result

            local_intent = match_on_off(final_text, entities) if entities else None
            if local_intent is not None:
                # A local-intent turn never builds the message list and
                # never consumes the pending-runs fetch's result -- same
                # cancel-then-await cleanup the macro path above already
                # uses for the identical reason. `state_task` is left
                # alone: it is either already done (the await above
                # resolved it) or still needed by nothing here, and
                # `_cancel_state_task` is a no-op on an already-done task.
                await _cancel_state_task(pending_runs_task)
                try:
                    tool_result = await tool_host.call_tool(
                        "ha_call_service",
                        {
                            "domain": local_intent.domain,
                            "service": local_intent.service,
                            "entity_id": local_intent.entity_id,
                        },
                    )
                except Exception:
                    # The call itself never reached a verdict -- logged and
                    # treated as a failure, never a silent fall-through to
                    # the brain: a tool call already in flight is not
                    # cancellable, and retrying it through a second path
                    # risks a double action (D-05's whole reasoning, one
                    # level up from the tier race this path bypasses).
                    logger.exception(
                        "local intent tool call raised for %s", local_intent.entity_id
                    )
                    intent_failed = True
                else:
                    intent_failed = _is_error(tool_result)

                if intent_failed:
                    timings.turn_outcome = "local_intent_failed"
                    reply_text = _CANNOT_DO_REPLY
                else:
                    timings.turn_outcome = "local_intent"
                    reply_text = _DONE_REPLY
                # Never falls back to the brain after a tool call was
                # attempted -- a policy denial may have been on purpose
                # (CMD-08's own doctrine, applied here the same way the
                # macro path already applies it above).
                speaking_tts = _tts_for_precached_fallback(filler_cache, sink, reply_text, tts)
                await _speak(
                    source,
                    speaking_tts,
                    timings,
                    reply_text,
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
        # 260924-4iv (item a): `None` means "unavailable this turn" to
        # `_state_message` -- distinct from `{}`/`()`, which mean "read
        # successfully, and it was empty" (or "no fetch was ever given").
        states_or_none: dict[str, str] | None = {}
        # D-09: the pending-run block's own payload, threaded into
        # `_state_message` alongside `states_or_none` below -- `()` (the
        # default) when `pending_runs_fetch` was never given, matching
        # `states_or_none`'s own `{}` default for the identical reason.
        pending_runs_or_none: tuple[Any, ...] | None = ()
        if state_task is not None:
            # Read once: the local on/off block above already read this
            # turn's `state_task` when it ran at all (`state_read=True`);
            # otherwise it is read here, for the first and only time, bound
            # by the same `state_deadline` (T-01.1-17's own posture --
            # unavailable is treated as "nothing known", never as ending
            # the turn).
            if not state_read:
                state_result = await _read_prefetched(
                    state_task, state_deadline, "state fetch", timings.turn_id, state_timeout_ms
                )
                state_read = True
            if state_result is _UNAVAILABLE:
                states_or_none = None
            elif isinstance(state_result, list):
                states_or_none = {entity["entity_id"]: entity["state"] for entity in state_result}
                friendly_names = {
                    entity["entity_id"]: entity["friendly_name"]
                    for entity in state_result
                    if isinstance(entity, dict) and "friendly_name" in entity
                }
            else:
                states_or_none = {}
        if pending_runs_task is not None:
            # Same one-shared-deadline discipline as `state_task` above,
            # applied to D-09's own fetch -- read once, bound by
            # `state_deadline`, unavailable treated as "nothing scheduled
            # known" rather than ending the turn (T-01.1-17).
            pending_runs_result = await _read_prefetched(
                pending_runs_task, state_deadline, "pending-runs fetch", timings.turn_id, state_timeout_ms
            )
            if pending_runs_result is _UNAVAILABLE:
                pending_runs_or_none = None
            else:
                pending_runs_or_none = tuple(pending_runs_result) if pending_runs_result else ()
        if workflow_tool_host is not None:
            # WR-01 fix: scope this turn's `cancel_workflow_run`/
            # `append_workflow_steps` to exactly the ids `pending_runs_or_none`
            # resolved to. 260924-4iv: an unavailable read (`None`) and a
            # genuinely empty one (`()`) both scope to no run ids -- cancel
            # and append refuse every id this turn either way, which is the
            # safe choice when the list could not be read at all.
            workflow_tool_host.set_current_turn_run_ids(
                frozenset(run.id for run in pending_runs_or_none) if pending_runs_or_none else frozenset()
            )
        if state_task is not None or pending_runs_task is not None:
            # Deferred, not module-level: `app.py` imports `run_turn` from this
            # module at load time, so a module-level import here of anything
            # from `app.py` would deadlock the two modules' load order. By the
            # time this line actually runs, this module has always finished
            # loading -- there is no way to call `run_turn` without importing
            # `atlas.turn.controller` first -- so importing `app.py` here,
            # even the first time, only ever fetches or finishes a module that
            # cannot be mid-load on this side. Same deferred-import shape
            # `brain_race.py` already uses for `_run_tool_rounds`.
            from atlas.app import _state_message

            messages.append(
                {
                    "role": "system",
                    "content": _state_message(states_or_none, pending_runs_or_none, domains=state_domains),
                }
            )
        # Plan 09-06/09-07 (D-06, D-08): a continuation turn's own prior
        # exchange -- an amendment's original request and the readback it
        # is amending, or a clarification's original request and the
        # question it answers, including every earlier link in a longer
        # chain (`_continuation_messages`) -- inserted here, just before
        # the user message carrying the amendment or answer itself, so a
        # tier sees catalog/state/prior-exchange/user in that order.
        # `None` (every turn that is not a continuation) adds nothing,
        # byte-identical to before plan 09-06.
        if prior_exchange:
            messages.extend(prior_exchange)
        messages.append({"role": "user", "content": final_text})

        if tiers is None:
            # The degenerate one-tier case: `run_top_tier` always wraps its
            # settled text locally now (260924-4it), for every top tier, not
            # only this one -- this one-element list just names the
            # single-model default explicitly. This is the seam that keeps
            # every Phase 01 test -- which drives `run_turn` with a `FakeBrain`
            # and no instructor client -- working unchanged.
            tiers = [brain_race.TierBrain(index=0, model="", brain=brain, envelope_client=None, calls_tools=True)]

        if restrict_tools_to_proposals:
            # A-CR-02: a triage tier has no tools, but its own
            # `needs_clarification` reply can still win the race and open
            # another no-wake-word window. A restricted turn races the
            # tool-calling tier alone -- and falls back to the single-model
            # default when this deployment configured no tool-calling tier.
            tiers = [tier for tier in tiers if tier.calls_tools] or [
                brain_race.TierBrain(index=0, model="", brain=brain, envelope_client=None, calls_tools=True)
            ]

        # CR-01: one instance per turn, shared between the top tier's tool
        # round and the race -- set True the instant a real tool call is made,
        # so a triage tier's confident reply can no longer end the race in the
        # top tier's place once its action is no longer cancellable.
        _validate_tiers(tiers)

        commitment = brain_race.ToolCommitment()
        # Plan 09-04: one `HandoffSlot` per turn, mutated in place by
        # whichever tier's tool round produces a handoff -- only the top
        # tier ever calls tools, so only `run_top_tier` (below) ever writes
        # to it, mirroring `commitment`'s own "one instance per turn" shape.
        handoff_slot = HandoffSlot()

        # A-CR-02: the schema itself is narrowed too, not only the dispatch
        # check `_run_tool_rounds` applies below -- a model offered only the
        # proposal tools has nothing else to even attempt to call. This is
        # the belt; `restricted_to` (threaded into `run_top_tier` below) is
        # the suspenders, and the one that actually holds: it is checked by
        # exact tool name, at dispatch, regardless of what the model was
        # offered or asked to do. R2-WR-04: both checks read the one set,
        # `HandoffContext.proposal_tool_names` -- the Google plugin's own
        # offered names, resolved by ownership -- so they cannot disagree.
        restricted_to: "frozenset[str] | None" = None
        if restrict_tools_to_proposals:
            restricted_to = handoff_context.proposal_tool_names if handoff_context is not None else frozenset()
        # R3-IN-05 (D-24): a scoped answer turn is limited the same two
        # ways, by the intersection when both limits apply. The scope's
        # entity check has no schema form; `_run_tool_rounds` applies it
        # at dispatch only.
        if answer_scope is not None:
            restricted_to = (
                answer_scope.tool_names if restricted_to is None else restricted_to & answer_scope.tool_names
            )
        turn_tools_schema = (
            [entry for entry in tools_schema if entry.get("function", {}).get("name") in restricted_to]
            if restricted_to is not None
            else tools_schema
        )

        tier_tasks: dict[int, asyncio.Task[TierReply]] = {}
        for tier in tiers:
            tier_messages = list(messages)
            if tier.calls_tools:
                coro = brain_race.run_top_tier(
                    tier,
                    tool_host,
                    turn_tools_schema,
                    tier_messages,
                    max_tool_rounds,
                    timings,
                    commitment,
                    handoff_slot=handoff_slot,
                    restricted_to=restricted_to,
                    answer_scope=answer_scope,
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

        try:
            winner = await asyncio.wait_for(race_task, timeout=brain_turn_timeout_s)
        except asyncio.TimeoutError:
            # 260922-woc: `race_task` is cancelled by `wait_for` itself, but
            # `race_tiers`' own cleanup (module docstring: cancel every
            # still-pending tier and await it) never runs for a cancellation
            # that arrives from *outside* -- `asyncio.CancelledError` is not
            # an `Exception`, so `race_tiers`' `except Exception:` handler
            # does not catch it, and the `CancelledError` propagates straight
            # out of the suspended `asyncio.wait` inside it with no chance to
            # reach its own cleanup block. `tier_tasks` is this function's
            # own reference to the same tasks `race_tiers` was racing, so
            # cancelling and awaiting them here is not a second mechanism --
            # it is the one cleanup `race_tiers` would have run, performed
            # from the one place that still holds a reference once the
            # cancellation has already bypassed it.
            for task in tier_tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tier_tasks.values(), return_exceptions=True)
            timings.turn_outcome = "brain_timeout"
            timeout_tts = _tts_for_precached_fallback(filler_cache, sink, _CANNOT_DO_REPLY, tts)
            await _speak(
                source,
                timeout_tts,
                timings,
                _CANNOT_DO_REPLY,
                kind="answer",
                barge_in=barge_in,
                speech_lock=speech_lock,
                sink=sink,
            )
            await _emit_event(source, timings.to_event())
            timings.log()
            return
        timings.mark_tool_rounds_done()

        if handoff_slot.handoff is not None:
            # Plan 09-04 (D-08): the tool round stored a handoff instead of
            # ending with an ordinary reply -- code, not the model, decides
            # what happens next. `follow_up_available` is False whenever
            # there is no context to store a pending action through or no
            # channel on this source to ask a confirmation over (D-06);
            # `chain_depth` extends whatever depth the incoming follow-up
            # (if this turn is itself answering one) already reached.
            follow_up_available = follow_up is not None and handoff_context is not None
            incoming = follow_up.incoming if follow_up is not None else None
            chain_depth = (incoming.chain_depth if incoming is not None else 0) + 1
            outcome = await dispatch_handoff(
                handoff_slot.handoff,
                handoff_context,
                transcript=final_text,
                follow_up_available=follow_up_available,
                chain_depth=chain_depth,
            )
            timings.turn_outcome = outcome.turn_outcome
            speech_result = await _speak(
                source,
                tts,
                timings,
                outcome.reply_text,
                kind="answer",
                barge_in=barge_in,
                speech_lock=speech_lock,
                sink=sink,
            )
            if outcome.follow_up is not None and follow_up is not None:
                # Plan 09-06 (D-06, D-09): `playback_ends_at` is set from
                # this readback's own `SpeechResult` -- never a guessed
                # constant -- so `SourceRunner._run_follow_ups` can open
                # the next turn's microphone only after the readback's own
                # audio has actually finished playing, plus an echo tail.
                # Plan 09-07 (D-06): `prior_messages` carries this turn's
                # own continuation messages forward -- non-empty only when
                # this turn was itself answering an earlier follow-up
                # (`prior_exchange`, built above, covers both an amendment
                # and a clarification's own answer); `()` for an ordinary
                # wake turn's first proposal or clarification, so a chain
                # of exactly one link adds nothing new here.
                # `outcome.follow_up` is frozen (`FollowUpRequest`), so
                # this is a new instance, not a mutation of the original.
                #
                # R3-IN-05 (D-24): a tool's own clarification scopes its
                # answer to that one tool (`handoff_slot.tool_name`, the
                # exact offered name the model called). A confirmation
                # keeps this turn's own scope, so a later amendment stays
                # inside it too. Either way the chain's scope only narrows.
                if outcome.follow_up.kind == "clarification":
                    asked_by = frozenset({handoff_slot.tool_name}) if handoff_slot.tool_name else frozenset()
                    next_scope: "AnswerScope | None" = AnswerScope(tool_names=asked_by).narrowed_by(answer_scope)
                else:
                    next_scope = answer_scope
                follow_up.request(
                    replace(
                        outcome.follow_up,
                        playback_ends_at=estimate_playback_end(speech_result, sink),
                        prior_messages=tuple(prior_exchange) if prior_exchange else (),
                        # A-CR-02: the restriction belongs to the chain.
                        proposals_only=outcome.follow_up.proposals_only or restrict_tools_to_proposals,
                        answer_scope=next_scope,
                    )
                )
            await _emit_event(source, timings.to_event())
            timings.log()
            return

        if handoff_slot.bulk_refused:
            # D-10: the tool round already settled on `BULK_REFUSAL_REPLY`
            # as its own settled text (`_run_tool_rounds` below) -- this
            # only corrects `turn_outcome` before the ordinary answer path
            # further down speaks it.
            timings.turn_outcome = "bulk_refused"

        if winner.needs_clarification:
            # CMD-09/D-07: a third, exclusive outcome -- speaks a question
            # naming every candidate and stops there. No macro follow-up (the
            # macro check already ran, above, before a single tier task was
            # created) and no further tool round: this turn ends exactly the
            # way an ordinary answered turn does, through the same `_speak`/
            # `_emit_event`/`timings.log()` sequence, so nothing here keeps
            # the audio source open past the question itself.
            #
            # Plan 09-07 (D-06): on a source with a follow-up channel
            # attached, the same brief no-wake-word window a calendar
            # confirmation opens (plan 09-06) opens after this question
            # too -- requested below, respecting the identical
            # `MAX_CHAINED_FOLLOW_UPS` cap `dispatch_handoff`'s own
            # pending-action path already enforces. A source with no
            # channel (`follow_up is None`) is unchanged: the operator
            # answers by waking the assistant again, and VOICE-21
            # (holding the microphone open) stays out of scope there.
            timings.turn_outcome = "needs_clarification"
            question = _compose_clarifying_question(winner.candidates, friendly_names)
            clarification_speech = await _speak(
                source, tts, timings, question, kind="answer", barge_in=barge_in, speech_lock=speech_lock, sink=sink
            )
            if follow_up is not None:
                clarification_chain_depth = (incoming.chain_depth if incoming is not None else 0) + 1
                if clarification_chain_depth <= MAX_CHAINED_FOLLOW_UPS:
                    follow_up.request(
                        FollowUpRequest(
                            kind="clarification",
                            chain_depth=clarification_chain_depth,
                            original_transcript=final_text,
                            question=question,
                            prior_messages=tuple(prior_exchange) if prior_exchange else (),
                            playback_ends_at=estimate_playback_end(clarification_speech, sink),
                            # A-CR-02: the restriction belongs to the chain.
                            proposals_only=restrict_tools_to_proposals,
                            # R3-IN-05 (D-24): no tool asked this question.
                            answer_scope=_triage_clarification_scope(winner.candidates).narrowed_by(answer_scope),
                        )
                    )
            await _emit_event(source, timings.to_event())
            timings.log()
            return

        # 260922-woc: a winning `TierReply` can carry `confident=False` and
        # no answer at all -- `TierReply`'s own validator only requires a
        # non-empty `answer` when `confident` is True (`providers/
        # tier_reply.py`), and the top tier wins the race unconditionally
        # regardless of its own `confident` flag (`brain_race.py`'s own
        # docstring: "there is nothing above it to escalate to"). Before
        # this fix, that reply reached `_speak` verbatim and the turn ended
        # in silence with no indication anything had gone wrong -- one of
        # this task's own two live-house recordings. `.strip()` catches a
        # whitespace-only answer the same way, not just a bare `""`.
        answer_text = winner.answer
        # GOOG-12, plan 09-05 Task 3: an account this turn's tool round
        # could not reach is never silently left out of a spoken calendar
        # answer -- appended here, by code, for every label the model's
        # own answer does not already name as a whole word (case
        # insensitive: "Home" already names "home"). This runs on the
        # ordinary answer path only (never a macro reply, a local-intent
        # "done", a clarifying question, or a handoff readback, none of
        # which speak this turn's own tool-round results back), and always
        # speaks through the live `tts` -- an answer with an appended note
        # is never the exact phrase a precached entry holds.
        appended_unreachable_note = False
        if answer_text.strip() and handoff_slot.unreachable_accounts:
            for label in handoff_slot.unreachable_accounts:
                if re.search(rf"\b{re.escape(label)}\b", answer_text, re.IGNORECASE):
                    continue
                answer_text = f"{answer_text}{_UNREACHABLE_ACCOUNT_NOTE.format(label=label)}"
                appended_unreachable_note = True
        if not answer_text.strip():
            timings.turn_outcome = "empty_answer"
            # A-WR-02: GOOG-12's own "never silently omit an unreachable
            # account" doctrine covers every ordinary answer path except
            # this one until now -- the `appended_unreachable_note` branch
            # just above only ever runs when `answer_text.strip()` is
            # truthy. An empty winning answer with a non-empty
            # `unreachable_accounts` is a real, reachable shape (this
            # module's own `260922-woc` comment above), and it is exactly
            # the turn where the note matters most: the operator hears
            # "i can't do that one" with no hint that the real reason was
            # an unreachable account, unless it is named here too. Spoken
            # live (never the cached fallback, matching the existing
            # `appended_unreachable_note` branch below): an answer with an
            # appended note is never the exact phrase a precached entry
            # holds.
            if handoff_slot.unreachable_accounts:
                reply_text = _CANNOT_DO_REPLY
                for label in handoff_slot.unreachable_accounts:
                    reply_text = f"{reply_text}{_UNREACHABLE_ACCOUNT_NOTE.format(label=label)}"
                speaking_tts = tts
            else:
                reply_text = _CANNOT_DO_REPLY
                speaking_tts = _tts_for_precached_fallback(filler_cache, sink, _CANNOT_DO_REPLY, tts)
        elif appended_unreachable_note:
            reply_text = answer_text
            speaking_tts = tts
        else:
            # 260924-4iu (b): reply_text becomes the cached phrase, not the
            # model's own text, because reply_text is what the `reply.text`
            # event carries (`_speak` below) and it must name the audio
            # that is actually about to play. The match is exact after
            # normalize(), never fuzzy -- the same rule macros use (D-11).
            # Model text comes from an untrusted transcript; it can only
            # select audio for a phrase the operator already configured, and
            # it never builds a cache path or a cache key itself.
            cached_phrase = _precached_phrase_for(filler_cache, sink, answer_text)
            if cached_phrase is not None:
                reply_text = cached_phrase
                speaking_tts = CachedTts(filler_cache)
            else:
                reply_text = answer_text
                speaking_tts = tts
        await _speak(
            source,
            speaking_tts,
            timings,
            reply_text,
            kind="answer",
            barge_in=barge_in,
            speech_lock=speech_lock,
            sink=sink,
        )
        await _emit_event(source, timings.to_event())
        timings.log()
    finally:
        # 10-07-PLAN.md (D-17): unsubscribed before the recorder closes
        # below -- an edge event arriving after this point must never
        # reach a turn that has already finished writing its own
        # `events.jsonl` (this plan's own "an edge event arriving after
        # the turn ended is not written to that turn's session" case). A
        # no-op when no subscription was ever made.
        if _edge_event_unsubscribe is not None:
            _edge_event_unsubscribe()
        # Covers every exit path above, including the two early returns --
        # exactly the turns whose folders an operator will want, and the
        # easiest ones to leak (D-13, T-02-22). A no-op when
        # `session_recorder` is `None`. Quick task 260924-4is (D3): the
        # file writes inside `close()` block on disk, so they run on a
        # worker thread rather than the loop -- nothing else still holds a
        # reference to append to this recorder by the time this `finally`
        # starts (the barge-in listener reads the raw source, never the
        # `_RecordingAudioSource` wrapper this function built above).
        if session_recorder is not None:
            await asyncio.to_thread(session_recorder.close, timings)


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


def _tts_for_precached_fallback(
    filler_cache: "Mapping[tuple[str, int] | None, Mapping[str, bytes]] | None",
    sink: "SinkFormat | None",
    text: str,
    live_tts: "_TtsProvider",
) -> "_TtsProvider":
    """`CachedTts(filler_cache)` when `text` is actually in the cache entry
    built for `sink`; `live_tts` otherwise (260922-woc).

    Every call site of this function names a phrase this project's own
    `tts.precache` list always carries (`_NO_SPEECH_REPLY`,
    `_CANNOT_DO_REPLY`) -- in the shipped configuration, this always
    resolves to the cache. The membership check exists only because
    `get_cached` (`providers/tts_cache.py`) raises `TtsError` on a miss
    rather than falling back itself (that module's own doctrine, for the
    ordinary filler/macro-reply case where a miss means the startup
    precache failed and should be loud about it) -- and this function's own
    callers run on a path that must never itself raise past an already
    failing or empty turn. A misconfigured deployment that dropped one of
    these two phrases from `tts.precache` still gets *a* voice, through the
    live provider, rather than a second silent turn stacked on top of the
    first one this fix exists to close.
    """
    if filler_cache:
        key = (sink.codec, sink.sample_rate) if sink is not None else None
        cache = filler_cache.get(key) or {}
        if text in cache:
            return CachedTts(filler_cache)
    return live_tts


def _precached_phrase_for(
    filler_cache: Mapping[tuple[str, int] | None, Mapping[str, bytes]] | None,
    sink: SinkFormat | None,
    text: str,
) -> str | None:
    """The cache key for `sink`'s cache entry whose `normalize()` form
    equals `normalize(text)`, or `None` (260924-4iu, b).

    Selects the cache entry with the same rule `_tts_for_precached_fallback`
    above uses: the key is `(sink.codec, sink.sample_rate)`, or `None` when
    `sink` is `None`. Returns `None` when `filler_cache` is empty or `None`,
    when that sink has no entry, when `normalize(text)` is empty (a
    punctuation-only answer normalizes to nothing and can never match), or
    when nothing in the entry matches. The cache holds a few dozen phrases
    at most, so this normalizes each key on every call rather than building
    and maintaining a second, precomputed index.
    """
    if not filler_cache:
        return None
    key = (sink.codec, sink.sample_rate) if sink is not None else None
    cache = filler_cache.get(key)
    if not cache:
        return None
    target = normalize(text)
    if not target:
        return None
    for phrase in cache:
        if normalize(phrase) == target:
            return phrase
    return None


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


async def _read_prefetched(
    task: asyncio.Task[Any] | None,
    deadline: float,
    what: str,
    turn_id: str,
    bound_ms: float,
) -> Any:
    """Read `task`'s result without ever costing this turn more than
    `deadline` (a `_time.monotonic()`-domain instant) to do so (260924-4iv,
    item a) -- the one bounded read `run_turn` uses for both the
    prefetched state fetch and the prefetched pending-runs fetch, each
    read at most once per turn.

    Returns `_UNAVAILABLE` for every case that is not a genuine result:
    `task is None` (the caller never started a fetch at all); `task`
    already done but raised (logged, T-01.1-17's existing posture); no
    time left before `deadline` (the task is cancelled and awaited --
    `_cancel_state_task` -- never left running past the bound); or a
    `asyncio.wait_for` timeout on the remaining time (which itself
    cancels and awaits the task; logged as a warning naming `what`,
    `bound_ms`, and `turn_id`, so a skipped read is visible against this
    turn in the log -- T-4iv-06).

    `asyncio.CancelledError` always re-raises, never swallowed -- this
    function is not the place that decides whether the turn itself should
    stop.
    """
    if task is None:
        return _UNAVAILABLE
    if task.done():
        try:
            return task.result()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s raised; treating as unavailable", what)
            return _UNAVAILABLE
    remaining = deadline - _time.monotonic()
    if remaining <= 0:
        await _cancel_state_task(task)
        return _UNAVAILABLE
    try:
        return await asyncio.wait_for(task, timeout=remaining)
    except asyncio.TimeoutError:
        logger.warning(
            "%s did not complete within brain.state_timeout_ms=%sms (turn_id=%s)",
            what,
            bound_ms,
            turn_id,
        )
        return _UNAVAILABLE
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("%s raised; treating as unavailable", what)
        return _UNAVAILABLE


async def _drain_to_final_transcript(
    source: _AudioSource,
    stt: _SttProvider,
    max_utterance_s: float,
    timings: TurnTimings,
    *,
    clock: Callable[[], float],
    poll_interval_s: float,
    onset_deadline: "float | None" = None,
    speech_signals: "Any | None" = None,
    finalize_if_already_ended: bool = False,
) -> Any | None:
    """Forward every event but the last as a partial; return the last, if any.

    `speech_signals` (10-05-PLAN.md, D-09 through D-13), when not `None`,
    is the edge source's own `SpeechSignals` -- `run_turn` reads it off the
    still-unwrapped source next to the `preroll_bytes` read, before
    `_RecordingAudioSource` ever wraps it. When given, this function starts
    a watch task alongside the drain that awaits
    `turn.early_finalize.wait_for_end_of_speech`, marks `timings.vad_end_at`,
    and sets a per-call `asyncio.Event` passed to `stt.stream(...,
    finalize=...)` -- ending the utterance the instant the Pi reports the
    end of speech, rather than waiting for this provider's own endpointing
    or for `max_utterance_s` to elapse. The watch task is cancelled and
    awaited on every return path below, so it never outlives this call.
    `speech_signals=None` (the default, and every caller that predates this
    plan) makes `stt.stream` called with exactly today's two positional
    arguments -- no `finalize` keyword at all -- so every source and every
    STT double that carries no `speech_signals` is unaffected byte for
    byte.

    `finalize_if_already_ended` governs what "speech already ended before
    this call started watching" means to the watch above -- see
    `wait_for_end_of_speech`'s own docstring for the full rule; `run_turn`
    passes `incoming is None` for an ordinary wake turn's first drain and
    `False` for the 260922-woc second drain.

    `onset_deadline` (plan 09-06, in `clock()`'s own domain) is a second,
    independent deadline from `max_utterance_s`'s own `deadline` below --
    it bounds only how long this function waits for the *first* event to
    arrive at all, and it stops applying the instant one does (D-06,
    D-09): a follow-up turn's open-microphone window must give up and
    speak "cancelled" if nobody answers by the time the window closes, but
    once an answer starts arriving, the ordinary `max_utterance_s` bound
    governs the rest of it exactly as it always has. `None` (the default,
    and every caller that predates this plan) means there is no such
    window and this check never fires.

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

    260924-4iv (item e): each event's arrival time is recorded with
    `_time.monotonic()` -- the same clock `TurnTimings` itself uses, not
    the injectable `clock` argument, which exists only for the deadline
    above. `timings.mark_first_partial(at=...)` is called with the held-
    back event's own recorded arrival, not the arrival of the event that
    confirmed it a partial. `speech_end_at` is set, on every return path,
    to the arrival time of the last partial whose normalized words differ
    from the previous *counted* partial (`brain_race._normalize_for_echo_check`,
    already imported by this module) -- a partial that normalizes to empty
    is never counted, and the final event itself is never counted, since it
    is the endpointing decision this stage exists to measure against. Per
    T-4iv-03, only that float ever leaves this function; the normalized
    words themselves stay in these locals.
    """
    timings.mark_stt_socket_open()
    # D-09: select the ASR channel here, after the recorder's own tap
    # (`_RecordingAudioSource` already wraps `source` above, before this
    # function is ever called) and before speech-to-text -- the only point
    # where "both channels to the recorder, one to speech-to-text" holds
    # with no second reader of `source.frames()`'s own queue. A one-channel
    # format (every source before Phase 10) is unchanged by `stt_view`.
    frames, fmt = stt_view(source.frames(), source.source_format())

    finalize_event: "asyncio.Event | None" = None
    watch_task: "asyncio.Task[None] | None" = None
    if speech_signals is not None:
        finalize_event = asyncio.Event()

        async def _watch_end_of_speech() -> None:
            at = await wait_for_end_of_speech(
                speech_signals,
                hangover_s=speech_signals.hangover_s,
                already_ended_counts=finalize_if_already_ended,
            )
            timings.mark_vad_end(at)
            finalize_event.set()

        watch_task = asyncio.create_task(_watch_end_of_speech())
        stream = stt.stream(frames, fmt, finalize=finalize_event)
    else:
        stream = stt.stream(frames, fmt)

    async def _stop_watch() -> None:
        # Cancelled and awaited on every return path below, so a watch
        # that never saw its own `vad.end` (D-13: xAI's own endpointing or
        # `max_utterance_s` ended the turn first) never outlives this call.
        if watch_task is not None:
            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch_task

    deadline = clock() + max_utterance_s
    pending: Any | None = None
    pending_arrival: float | None = None
    last_counted_words: str | None = None
    last_word_change_arrival: float | None = None
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
            await _stop_watch()
            timings.mark_speech_end(last_word_change_arrival)
            return None

        # 09-06: no event has arrived at all yet, and this window's own
        # onset deadline has passed -- give up exactly like the
        # `max_utterance_s` branch above (same cancel/close/mark
        # sequence), never once `pending is not None` (docstring: the
        # ordinary bound governs from the first event onward).
        if onset_deadline is not None and pending is None and clock() >= onset_deadline:
            next_event_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await next_event_task
            await stream.aclose()
            await _stop_watch()
            timings.mark_speech_end(last_word_change_arrival)
            return None

        done, _pending_tasks = await asyncio.wait({next_event_task}, timeout=poll_interval_s)
        if next_event_task not in done:
            continue

        try:
            event = next_event_task.result()
        except StopAsyncIteration:
            await _stop_watch()
            timings.mark_speech_end(last_word_change_arrival)
            return pending

        arrival = _time.monotonic()

        if pending is not None:
            timings.mark_first_partial(at=pending_arrival)
            await _emit_event(source, {"type": "transcript.partial", "text": getattr(pending, "text", "")})
            normalized = brain_race._normalize_for_echo_check(getattr(pending, "text", ""))
            if normalized and normalized != last_counted_words:
                last_counted_words = normalized
                last_word_change_arrival = pending_arrival
        pending = event
        pending_arrival = arrival
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


def _round_settles_as_done(reply: Any, results: list[Any], messages: list[dict[str, Any]]) -> bool:
    """True when a tool round is plain action success and nothing more --
    the first-round done shortcut in `_run_tool_rounds` may speak the fixed
    `_DONE_REPLY` instead of paying a second `brain.chat` round only to
    phrase a confirmation.

    "Done" asserts an outcome, so every condition here errs toward taking
    the model round -- CMD-07 never lets a canned reply confirm something
    that did not actually happen.

    - `reply.text` must be blank: the model attached words of its own,
      which may carry information the operator needs to hear.
    - Every tool call name must be in `_DONE_SHORTCUT_TOOLS`: a read tool,
      a workflow tool, a plugin tool, or a collision-prefixed name always
      takes the model round instead.
    - Every result must be a real 2xx Home Assistant reply with nothing to
      read: not a raised exception, not `_is_error`, and its payload must
      be a dict whose key set is exactly `{"changed"}` -- a response-only
      service's `{"changed": [...], "response": {...}}` reply, an
      `{"error": ...}` payload, and an unparseable result all take the
      model round, since each carries something the operator may need to
      hear.
    - The transcript must not ask for information: `asks_for_information`
      errs toward True, so a miss here only costs latency, never a dropped
      answer.
    """
    if (reply.text or "").strip():
        return False

    for tool_call in reply.tool_calls:
        if tool_call.name not in _DONE_SHORTCUT_TOOLS:
            return False

    for result in results:
        if isinstance(result, BaseException) or _is_error(result):
            return False
        payload = _result_payload(result)
        if not (isinstance(payload, dict) and set(payload) == {"changed"}):
            return False

    transcript = brain_race._last_user_message(messages)
    return transcript is not None and not asks_for_information(transcript)


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


def _triage_clarification_scope(candidates: "tuple[str, ...]") -> AnswerScope:
    """R3-IN-05 (D-24): the scope for the answer to a clarification that a
    triage tier asked. No tool asked this question, so the scope comes from
    the candidates -- the smallest set that still finishes the request:

    - Every candidate is an entity id: `ha_call_service`, for those entity
      ids only. "Which light?" can still turn on the light that the answer
      names, and nothing else.
    - Otherwise (plugin display names, scheduled-run summaries, or a mix):
      no tool at all. These routes have no single tool and no target
      argument to check, so their answer can only be spoken, and an action
      needs the wake word again.
    """
    if candidates and all(_ENTITY_ID_SHAPE.fullmatch(candidate) for candidate in candidates):
        return AnswerScope(tool_names=frozenset({HA_CALL_SERVICE_TOOL}), entity_ids=frozenset(candidates))
    return AnswerScope(tool_names=frozenset())


async def _run_tool_rounds(
    brain: _BrainProvider,
    tool_host: _ToolHost,
    tools_schema: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    max_tool_rounds: int,
    timings: TurnTimings,
    commitment: "brain_race.ToolCommitment | None" = None,
    handoff_slot: "HandoffSlot | None" = None,
    restricted_to: "frozenset[str] | None" = None,
    answer_scope: "AnswerScope | None" = None,
) -> str:
    """Run up to `max_tool_rounds` brain calls, executing any tool calls in between.

    A command needing more rounds than the configured cap is a
    misunderstanding, not a complex request (per `BrainConfig.max_tool_rounds`'s
    own doctrine) -- the turn ends by saying so, never with a success
    confirmation, per CMD-01's transparency prohibition.

    `restricted_to` (A-CR-02, R2-WR-04, default `None` -- no restriction,
    every caller that predates this fix) is `run_turn`'s own allowlist for
    a proposal-restricted turn, threaded down through `run_top_tier`: when
    set, any tool call whose exact name is not in it is refused exactly
    like a code-only tool a model named anyway -- never dispatched to
    `tool_host` -- regardless of what `tools_schema` this round was
    actually offered.
    This is the structural backstop the schema restriction alone cannot
    be: a model that names a tool outside the schema it was given still
    never reaches `tool_host.call_tool` for it.

    `answer_scope` (R3-IN-05, D-24, default `None`) adds the scope's own
    target check at dispatch: a call whose arguments name an entity
    outside the scope is refused the same way.

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

    260924-4it: when the first round (`round_num == 0`) is plain action
    success -- every call is on `_DONE_SHORTCUT_TOOLS`, every result is a
    bare `{"changed": [...]}` Home Assistant reply, the model attached no
    text of its own, and the transcript does not ask for information
    (`_round_settles_as_done`) -- this function returns the fixed
    `_DONE_REPLY` at once, skipping the second `brain.chat` round that
    would otherwise only phrase a confirmation of what already happened.
    """
    for round_num in range(max_tool_rounds):
        reply = await brain.chat(messages, tools=tools_schema)
        timings.mark_brain_first_round()
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
        # T-09-27: a code-only tool (`atlas_mcp.google_tools.CODE_ONLY_TOOL_NAMES`)
        # is never dispatched to `tool_host`, regardless of what
        # `PluginManager.rebuild`'s own schema hiding already withheld --
        # this is the second, independent control, checked here by name
        # rather than trusted from the schema the model happened to be
        # offered. Refused calls never reach `tool_host.call_tool` at all;
        # each gets an error-shaped result carrying `CODE_ONLY_REFUSAL` at
        # its own position, so every composer downstream (the mixed-outcome
        # composer, the handoff detector) sees a normal, positionally
        # aligned result list.
        #
        # A-CR-02: `restricted_to` applies the identical control on a
        # proposal-restricted turn -- any tool call whose exact name is not
        # in it is refused the same way, checked here by name, never
        # trusted from the (already narrowed) schema this round was offered.
        #
        # R3-IN-05 (D-24): `answer_scope` adds its target check here too.
        def _refused(tc: Any) -> bool:
            if restricted_to is not None and tc.name not in restricted_to:
                return True
            return answer_scope is not None and not answer_scope.allows_targets(tc.arguments)

        dispatchable = [
            (index, tc)
            for index, tc in enumerate(reply.tool_calls)
            if not is_code_only_tool(tc.name) and not _refused(tc)
        ]
        dispatched_results = await asyncio.gather(
            *(tool_host.call_tool(tc.name, tc.arguments) for _, tc in dispatchable),
            return_exceptions=True,
        )
        results: list[Any] = [None] * len(reply.tool_calls)
        for (index, _tc), result in zip(dispatchable, dispatched_results):
            results[index] = result
        for index, tc in enumerate(reply.tool_calls):
            if is_code_only_tool(tc.name):
                results[index] = SimpleNamespace(
                    isError=True, content=[SimpleNamespace(text=CODE_ONLY_REFUSAL)]
                )
            elif _refused(tc):
                results[index] = SimpleNamespace(
                    isError=True, content=[SimpleNamespace(text=AMENDED_CONTINUATION_REFUSAL)]
                )

        # GOOG-12, plan 09-05 Task 3: every account this round's results
        # named unreachable (`atlas_mcp.google_tools.UNREACHABLE_KEY`,
        # e.g. `calendar_list_events`'s own `unreachable_accounts` list)
        # collected into `handoff_slot.unreachable_accounts`, deduplicated
        # in first-seen order -- read for every successful, non-error
        # result in every round, never only the round a handoff came from,
        # since an ordinary read is exactly where this matters most.
        # `run_turn`'s own ordinary-answer branch appends a fixed note for
        # each label the model's answer does not already name.
        if handoff_slot is not None:
            for result in results:
                if isinstance(result, BaseException) or _is_error(result):
                    continue
                payload = _result_payload(result)
                if not isinstance(payload, dict):
                    continue
                unreachable_entries = payload.get(UNREACHABLE_KEY)
                if not isinstance(unreachable_entries, list):
                    continue
                for entry in unreachable_entries:
                    if not isinstance(entry, dict):
                        continue
                    label = entry.get("account")
                    if isinstance(label, str) and label not in handoff_slot.unreachable_accounts:
                        handoff_slot.unreachable_accounts.append(label)

        # Plan 09-04 (D-08, D-10): detect any handoff among this round's
        # results before the existing mixed-outcome check below -- a
        # handoff never reaches a `role: tool` message and never reaches a
        # second `brain.chat` round.
        handoffs_present = [
            (index, handoff) for index, handoff in enumerate(parse_handoff(r) for r in results) if handoff is not None
        ]
        pending_action_count = sum(1 for _, h in handoffs_present if h.kind == "pending_action")

        if pending_action_count >= 2:
            # D-10: two change proposals in one round are a bulk change --
            # refused outright, nothing stored. `handoff_slot.bulk_refused`
            # lets `run_turn` correct `turn_outcome`; this function's own
            # settled-text contract (return the reply text) is unchanged.
            if handoff_slot is not None:
                handoff_slot.bulk_refused = True
            return BULK_REFUSAL_REPLY

        if handoffs_present and len(handoffs_present) == 1 and len(reply.tool_calls) == 1:
            # D-08: the round's only call produced a handoff -- store it and
            # end the round at once. No `role: tool` message is appended for
            # it, so the handoff payload never enters the message list, and
            # nothing here calls `brain.chat` a second time.
            if handoff_slot is not None:
                handoff_slot.handoff = handoffs_present[0][1]
                # R3-IN-05 (D-24): the exact offered name that asked, so a
                # clarification's answer can be scoped to it.
                handoff_slot.tool_name = reply.tool_calls[handoffs_present[0][0]].name
            return ""

        if handoffs_present:
            # A handoff sharing its round with other calls is never
            # honoured (D-08): replace each handoff result with an
            # error-shaped one carrying the fixed refusal, then fall
            # through to the existing mixed-outcome composer below, which
            # already knows how to report one failed call next to any
            # number of successful ones.
            results = list(results)
            for index, _handoff in handoffs_present:
                results[index] = SimpleNamespace(
                    isError=True, content=[SimpleNamespace(text=HANDOFF_NOT_ALONE_REPLY)]
                )

        # Collect every result before deciding anything (D-06): a batch
        # containing any error-shaped or raised entry is composed in code,
        # below, and returned directly -- never through a second `brain.chat`
        # round (D-14). A batch where every call succeeded falls through to
        # the loop-to-the-next-round behaviour, byte-identical to before this
        # plan.
        if any(isinstance(result, BaseException) or _is_error(result) for result in results):
            return _compose_mixed_outcome_reply(list(zip(reply.tool_calls, results)))

        # 260924-4it: the first round only, and only when nothing about it
        # needs a model's own words -- a later round may follow an earlier
        # read the operator is waiting to hear, so the shortcut never
        # applies past round 0 (the same reasoning `_round_settles_as_done`
        # uses for a question asked next to the action). The second
        # `brain.chat` call this replaces would only phrase a confirmation
        # of a plain action success; the local on/off intent path already
        # speaks "done" after one, with no model call at all.
        if round_num == 0 and _round_settles_as_done(reply, results, messages):
            logger.info(
                "first tool round was plain action success (%d call(s)); speaking %r with no phrasing round",
                len(reply.tool_calls),
                _DONE_REPLY,
            )
            return _DONE_REPLY

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


@dataclass(frozen=True)
class SpeechResult:
    """What `_speak` actually wrote to the speaker for one utterance --
    plan 09-06's own input to `turn/follow_up.py::estimate_playback_end`,
    which is how a follow-up window's own open-microphone timing is
    derived from real playback rather than a guessed constant (D-09).

    `bytes_sent` counts only chunks that actually reached
    `source.send_audio()` -- a chunk dropped by a barge-in interrupt or a
    dead speaker is never counted, matching `chunks_sent` inside
    `_speak`'s own write loop. `first_write_at`/`last_write_at` are
    `None` only when nothing was ever written at all (an empty synthesis,
    or every chunk dropped before the first write) -- every existing
    caller of `_speak` ignores this return value entirely, so nothing
    about their behavior changes.
    """

    bytes_sent: int
    first_write_at: "float | None"
    last_write_at: "float | None"


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
) -> SpeechResult:
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

    bytes_sent = 0
    first_write_at: float | None = None
    last_write_at: float | None = None

    async def _synthesize_and_write() -> None:
        nonlocal bytes_sent, first_write_at, last_write_at
        first_audio_marked = False
        interrupted = False
        speaker_failed = False
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
            if speaker_failed:
                continue
            try:
                await source.send_audio(chunk)
            except SpeakerError as exc:
                # A dead speaker must not kill the turn: the brain's tool
                # calls (the actual command) still run and the reply text
                # still reaches the event stream. Skip the rest of this
                # reply's audio rather than retrying chunk by chunk.
                speaker_failed = True
                logger.warning("speaker unavailable, dropping %s audio: %s", kind, exc)
                continue
            chunks_sent += 1
            # Plan 09-06: counted only for a chunk that actually reached
            # `source.send_audio()` above -- never one dropped by the
            # `interrupted`/`speaker_failed` branches, matching
            # `chunks_sent`'s own accounting exactly.
            now_write = _time.monotonic()
            if first_write_at is None:
                first_write_at = now_write
            last_write_at = now_write
            bytes_sent += len(chunk)
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

    return SpeechResult(bytes_sent=bytes_sent, first_write_at=first_write_at, last_write_at=last_write_at)


async def _play_wake_cue(source: _AudioSource, sink: SinkFormat, speech_lock: asyncio.Lock | None) -> None:
    cue = wake_cue_audio(sink)
    if not cue:
        return
    try:
        if speech_lock is not None:
            async with speech_lock:
                await source.send_audio(cue)
        else:
            await source.send_audio(cue)
    except SpeakerError as exc:
        logger.warning("speaker unavailable, skipping the wake cue: %s", exc)


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


def _result_payload(result: Any) -> Any:
    """A tool result's parsed JSON payload, or `None` when there isn't one.

    Prefers `structured_content`/`structuredContent` when present, the same
    way `app.py`'s own `_tool_result_json` does (not imported from there --
    that would be an import cycle) -- unwrapping the MCP SDK's `{"result":
    value}` wrapper for a non-object return. Falls back to parsing
    `_result_text(result)` as JSON, and returns `None` rather than raising
    when there is nothing to parse.
    """
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    try:
        return json.loads(_result_text(result))
    except ValueError:
        return None
