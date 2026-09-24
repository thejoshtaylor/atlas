"""Race the configured tiers; only the last one ever reaches Home Assistant.

D-05 is this module's whole reason to exist: two racing models must never
both switch the same switch, and a cancelled tool call has no rollback. The
guarantee is structural, not a runtime check -- only the last entry in
`brain.models` (`BrainConfig.top_tier`) is ever constructed with a tool
schema at all. `run_triage_tier` issues its `instructor` call with no
`tools=` parameter whatsoever, so a lower tier has no tool schema to call
even if it wanted to.

`race_tiers` uses `asyncio.wait(..., return_when=asyncio.FIRST_COMPLETED)`,
not `asyncio.TaskGroup`: `TaskGroup` cancels every sibling the instant one
task raises and re-raises through an `ExceptionGroup` -- built for "all must
succeed," not "keep some running while I inspect one that finished." A
non-confident `TierReply` is not an exception, so `TaskGroup` gives no
natural hook for "keep waiting."
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
from dataclasses import dataclass
from typing import Any

import instructor
from openai import AsyncOpenAI

from atlas.config import BrainConfig
from atlas.providers.base import BrainError
from atlas.providers.brain_xai import XaiBrain
from atlas.providers.tier_reply import DEFAULT_FILLER, TierReply

logger = logging.getLogger("atlas.turn.brain_race")


@dataclass
class TierBrain:
    """One racing candidate: a position in `brain.models`, and the two ways
    it can be asked to answer.

    `calls_tools` is derived once, in `build_tiers`, from position in the
    list -- never settable per entry (D-05). `brain` is exercised only by
    `run_top_tier`; a triage tier never touches it, since it never runs a
    tool round at all. `envelope_client` is `None` for the top tier
    (260924-4it) -- only a triage tier's `run_triage_tier` ever makes an
    envelope call; the top tier wraps its own settled text into a
    `TierReply` locally, in `run_top_tier`.
    """

    index: int
    model: str
    brain: Any
    envelope_client: Any | None
    calls_tools: bool


@dataclass
class ToolCommitment:
    """A mutable, per-turn flag: set True the instant the top tier's tool
    round makes a real call to the tool host.

    CR-01: a cancelled tool call has no rollback -- once this is True, a
    stdio round trip to the MCP child has already happened and cannot be
    un-sent. `race_tiers` must not let a triage tier's confident reply end
    the race in the top tier's place once this is True; only the top tier's
    own settled outcome may describe what actually happened. One instance
    is created per turn in `controller.run_turn` and threaded through both
    `run_top_tier` (which sets it) and `race_tiers` (which reads it) -- it
    is not shared across turns.
    """

    committed: bool = False


def build_tiers(brain_config: BrainConfig) -> tuple[TierBrain, ...]:
    """One `TierBrain` per `brain.models` entry, in list order.

    One shared `instructor`-wrapped client backs every triage tier's
    envelope call: `instructor.from_openai(AsyncOpenAI(...))`, never
    `from_provider`, which reads `XAI_API_KEY` from the environment directly
    and bypasses `BrainConfig.api_key`/`base_url` and config.py's
    env-expansion contract. The top tier gets no envelope client at all
    (260924-4it) -- `run_top_tier` wraps its own settled text locally, so a
    one-model config (only the top tier, no triage tiers) builds no
    `instructor` client whatsoever.

    `Mode.JSON` is fixed at client-construction time -- `instructor`'s own
    `.create()` call has no per-call `mode=` parameter, and `Mode.JSON` is
    the only mode any envelope call in this module ever needs: it is backed
    by `response_format`, a channel xAI's own docs describe as combinable
    with tool calling, unlike `Mode.TOOLS`, which re-enters the exact wire
    channel the top tier's tool rounds just used.
    """
    envelope_client = (
        instructor.from_openai(
            AsyncOpenAI(api_key=brain_config.api_key, base_url=brain_config.base_url),
            mode=instructor.Mode.JSON,
        )
        if brain_config.triage_tiers
        else None
    )
    return tuple(
        TierBrain(
            index=index,
            model=entry.model,
            brain=XaiBrain(brain_config, model=entry.model),
            envelope_client=None if entry is brain_config.top_tier else envelope_client,
            calls_tools=(entry is brain_config.top_tier),
        )
        for index, entry in enumerate(brain_config.models)
    )


_ECHO_PUNCTUATION_RE = re.compile(r"[^\w\s]")

# A live-house bug (see this task's plan): a triage tier answered a garbled
# transcript back verbatim as a CONFIDENT reply, and that echo won the race
# in under the top tier's own answer's time. 0.85 is loose enough to catch a
# reply that only drops or adds a trailing word, tight enough that a genuine
# short answer ("it is 3 pm") does not collide with a short question.
_ECHO_RATIO_THRESHOLD = 0.85


def _normalize_for_echo_check(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Shared normalization for both sides of the echo comparison, so
    "We're off the swamp core." and "were off the swamp core" compare on
    their words alone, not on case or punctuation noise transcription adds.
    """
    return " ".join(_ECHO_PUNCTUATION_RE.sub("", text.lower()).split())


def _last_user_message(messages: list[dict[str, Any]]) -> str | None:
    """The transcript a triage tier was actually asked about: the last
    `{"role": "user"}` message in its own messages list."""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


async def run_triage_tier(tier: TierBrain, messages: list[dict[str, Any]]) -> TierReply:
    """One `instructor` call, no `tools=` parameter at all.

    A triage tier never reaches the tool host -- it either answers what the
    injected live state already covers, or reports (through `TierReply`)
    that the request needs a tool call and hands off with a filler. Lower
    tiers only ever need this single-call, no-collision path.

    A confident reply that is really just the user's own transcript echoed
    back (a live-house bug: a garbled transcript answered as a CONFIDENT
    reply, word for word) is downgraded to non-confident here, before it
    ever reaches `race_tiers` -- it keeps its filler, so the race simply
    keeps waiting and the top tier answers instead.
    """
    reply = await tier.envelope_client.chat.completions.create(
        model=tier.model,
        messages=messages,
        response_model=TierReply,
    )
    if reply.confident:
        transcript = _last_user_message(messages)
        if transcript is not None:
            ratio = difflib.SequenceMatcher(
                None,
                _normalize_for_echo_check(reply.answer),
                _normalize_for_echo_check(transcript),
            ).ratio()
            if ratio >= _ECHO_RATIO_THRESHOLD:
                logger.info(
                    "triage tier %d confident answer echoes the user transcript "
                    "(ratio=%.2f); treating as not confident",
                    tier.index,
                    ratio,
                )
                return reply.model_copy(update={"confident": False})
    return reply


async def run_top_tier(
    tier: TierBrain,
    tool_host: Any,
    tools_schema: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    max_tool_rounds: int,
    timings: Any,
    commitment: ToolCommitment | None = None,
) -> TierReply:
    """The existing, unmodified tool-calling loop, wrapped locally into a
    `TierReply` -- no envelope call.

    `_run_tool_rounds`'s settled text is already the final reply. It used to
    be handed to a second `instructor` call whose only job was to wrap that
    same text into a `TierReply` -- one full model round trip that added no
    information, and could reword a refusal `_compose_mixed_outcome_reply`
    composed in code, breaking CMD-08's verbatim guarantee on a multi-tier
    turn (260924-4it). `confident` is `True` only when the settled text
    carries real content: a whitespace-only settled text (`reply.text` can
    be whitespace, though `_run_tool_rounds` never returns an empty string)
    gives `confident=False`, so `run_turn`'s empty-answer branch speaks the
    fallback instead of `TierReply`'s own validator raising on a
    `confident=True` empty answer. This function no longer reads
    `tier.envelope_client` at all -- see `build_tiers` below.
    """
    # Deferred, not module-level: `controller.py` imports this module at
    # load time to dispatch tiers, so a module-level import here of anything
    # from `controller.py` would deadlock the two modules' load order. By
    # the time this function actually runs, `controller.py` has always
    # finished loading (there is no way to call `run_top_tier` without
    # importing `atlas.turn.brain_race` first, and nothing in this
    # module imports `controller` before this point), so the import below
    # always succeeds.
    from atlas.turn.controller import _run_tool_rounds

    settled_text = await _run_tool_rounds(
        tier.brain, tool_host, tools_schema, messages, max_tool_rounds, timings, commitment=commitment
    )

    return TierReply(
        answer=settled_text,
        confident=bool(settled_text.strip()),
        needs_tool=False,
        filler=DEFAULT_FILLER,
    )


def _triage_tier_wins(reply: TierReply) -> bool:
    """The two ways a triage tier's own reply can end the race early.

    A confident answer and a `needs_clarification` question are both
    settled outcomes a triage tier can reach without ever touching a tool --
    CMD-09/D-07 gives the second the identical early-exit treatment the
    first already had, through this one predicate, so a third outcome added
    later cannot be guarded here and forgotten at the (single) call site
    below.
    """
    return reply.confident or reply.needs_clarification


async def race_tiers(
    tasks_by_index: dict[int, "asyncio.Task[TierReply]"],
    commitment: ToolCommitment | None = None,
) -> TierReply:
    """Resolve to the first settled triage reply, always breaking ties by index.

    A triage tier can win the race two ways: a confident answer, or (CMD-09,
    D-07) a `needs_clarification` reply asking which entity was meant --
    `_triage_tier_wins` above is the one predicate covering both, so this
    function's win condition never needs to distinguish which outcome an
    early triage reply carries. Inside every `asyncio.wait` batch that
    contains more than one done task, the tasks are inspected in ascending
    tier index -- never set iteration order -- so the same two completions
    always produce the same winner. The top tier's reply ends the race
    whichever way its flags read: there is nothing above it to escalate to.
    A triage tier that raises is logged and dropped from the race; the top
    tier raising propagates, because there is no fallback behind it.

    CR-01, and now D-07 by the identical reasoning: once `commitment.committed`
    is True -- the top tier has already made a real, uncancellable tool call
    this turn -- a triage tier's early-win reply (confident or
    `needs_clarification`) is dropped rather than ending the race. A real
    tool call cannot be un-sent, so a clarifying question arriving after one
    has already run would be a second kind of lying about the house, not a
    safer alternative to one -- continuing to wait for the top tier's own
    outcome is the only choice that cannot discard an already-executed side
    effect. `commitment=None` (the default) preserves the exact prior
    behavior for any caller that races tiers with no top-tier tool round at
    all.

    When a winner is found, every still-pending task is cancelled and its
    unwind is awaited (`return_exceptions=True`) before this function
    returns, rather than firing the cancellation and moving on.

    CR-02: that cancel-and-await cleanup runs on every exit path, including
    the one where the top tier's own task raises. Without the `try`/`except`
    below, a raise from inside the `while` loop's `for` loop propagated
    straight out of this function, skipping the cleanup block entirely: a
    triage tier still pending at that moment was neither cancelled nor
    awaited, its underlying request left running detached from the turn
    that started it.

    `needs_clarification` is exercised on the triage tier only. A top-tier
    reply still wins unconditionally at `index == top_index`, unchanged by
    this function. `run_top_tier` builds its `TierReply` locally, from its
    own settled tool-round text, and never sets `needs_clarification`
    itself (260924-4it) -- there is no longer an envelope call on the top
    tier's own path that could set it, so a top-tier clarification cannot
    arise.
    """
    top_index = max(tasks_by_index)
    task_index = {task: index for index, task in tasks_by_index.items()}
    pending = set(tasks_by_index.values())
    winner: TierReply | None = None

    try:
        while pending and winner is None:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=lambda t: task_index[t]):
                index = task_index[task]
                try:
                    reply = task.result()
                except Exception:
                    if index == top_index:
                        raise
                    logger.exception("triage tier %d raised; dropped from the race", index)
                    continue
                if index == top_index:
                    winner = reply
                    break
                if _triage_tier_wins(reply):
                    if commitment is not None and commitment.committed:
                        # The top tier already reached the tool host this
                        # turn -- its own outcome is the only one allowed to
                        # end the race now (CR-01, and D-07 by the identical
                        # reasoning). Drop this reply and keep waiting.
                        continue
                    winner = reply
                    break
    except Exception:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    if winner is None:  # pragma: no cover - the top tier always ends the race
        raise BrainError("race_tiers loop exited with no winner and no top-tier exception")
    return winner
