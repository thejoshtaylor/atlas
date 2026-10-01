"""The no-wake-word window after an ordinary answer (VOICE-21, Phase 13).

After the assistant answers on a source that opts in (the edge Pi), the
microphone stays open for a short time with no wake word, so the operator can
reply with "and the other one" or "turn it off". This module holds the pure
rules for that window. `turn/controller.py` calls them and stays short.

The window fails closed, because every word heard in it is untrusted. The
microphone hears the television, and a television can speak any sentence.
So the window turn may reach only the tools the answered turn itself
dispatched (D-10). After a conversation-only answer it reaches no tool at all
(D-11), except when that answer asked the operator a question. Then it reaches
the catalog the turn offered (261001-dlp, a user decision that amends D-11 for
this case only). Along a chain of windows the scope only narrows. A window turn never
runs a macro, a local on/off intent or a timer intent, because the continued
exchange skips those paths. A window that hears nothing speaks nothing (D-15).
After a question the model asked, "yeah" and "okay" are answers and reach the
brain (261001-dlp). After an ordinary answer they still end the window.
"""

from __future__ import annotations

import re
from typing import Any

from atlas_mcp.ha_names import HA_WRITE_TOOL_NAMES

from atlas.timers.ring_stop import is_stop_command
from atlas.turn.entity_claims import bare_tool_name
from atlas.turn.follow_up import (
    _EXPANDING_TARGET_ARGUMENTS,
    MAX_CHAINED_FOLLOW_UPS,
    AnswerScope,
    FollowUpRequest,
)
from atlas.turn.transcript_guard import is_no_command
from atlas.turn.wake_echo import strip_wake_phrase

ANSWER_WINDOW_SILENT = "answer_window_silent"

# WR-02: a wake hit counts as evidence for the window turn only when it lands
# within this many seconds of the first audio the window heard. The Pi sends
# voice-gated segments only, so that first audio is where the utterance the
# turn transcribes begins. A hit later in the window, or after another sound,
# cannot pair with a transcript that happens to open like the wake phrase.
WAKE_EVIDENCE_WINDOW_S = 1.5

# "turn it off" is a ring stop phrase, but in this window it is a real
# command about the thing the last answer touched. Any text with the word
# "off" goes to the brain, never to a silent stop.
_OFF_RE = re.compile(r"\boff\b", re.IGNORECASE)

# 261001-dlp: whole-utterance affirmatives. After a question the model asked,
# these are the operator's answer. "okay" and "yeah" would otherwise end the
# window silently as a stop word or as filler. The timer ring stop in
# `timers/ring_stop.py` is a separate path and does not change.
_AFFIRMATIVE_WORDS = frozenset({"yes", "yeah", "yep", "yup", "ok", "okay", "sure"})
_MAX_AFFIRMATIVE_TOKENS = 3
_PUNCT_RE = re.compile(r"[^\w\s]")


def is_affirmative(text: str) -> bool:
    """True when every word of `text` is an affirmative ("yeah", "Okay.").
    Case and punctuation do not matter."""
    tokens = _PUNCT_RE.sub("", text.lower()).split()
    return 1 <= len(tokens) <= _MAX_AFFIRMATIVE_TOKENS and all(t in _AFFIRMATIVE_WORDS for t in tokens)


def is_quiet_stop(text: str) -> bool:
    """True when `text` is a stop phrase that ends a window or an interrupt
    turn with no reply. Text with the word "off" never counts: it is a
    command for the brain."""
    return is_stop_command(text) and not _OFF_RE.search(text)


def _named_entities(arguments: Any) -> "frozenset[str]":
    """The entity ids a call names through `entity_id` alone. Empty when it
    names none, or when it also names an area, a device or a label, which
    expand to entities the call never listed."""
    if not isinstance(arguments, dict) or any(arguments.get(key) for key in _EXPANDING_TARGET_ARGUMENTS):
        return frozenset()
    value = arguments.get("entity_id")
    named = [value] if isinstance(value, str) else value
    if not isinstance(named, list) or not all(isinstance(item, str) and item for item in named):
        return frozenset()
    return frozenset(named)


def note_dispatched_call(slot: Any, name: str, arguments: Any) -> None:
    """Record one dispatched call on the turn's `HandoffSlot` (D-10, WR-01).
    A Home Assistant write tool also records the entities it named, or that
    it named none."""
    slot.called_tools.add(name)
    if bare_tool_name(name) not in HA_WRITE_TOOL_NAMES:
        return
    entities = _named_entities(arguments)
    if entities:
        slot.called_entities |= entities
    else:
        slot.untargeted_writes.add(name)


def dispatched_scope(slot: Any) -> AnswerScope:
    """What a window after this turn may reach: the tools it dispatched,
    and for a Home Assistant write tool only the entities it named (WR-01).
    A write tool that named no entity fails closed: it is left out. The
    entity limit applies to the write tools only (`allows_targets`)."""
    tool_names = frozenset(
        name
        for name in slot.called_tools
        if name not in slot.untargeted_writes
        and (bare_tool_name(name) not in HA_WRITE_TOOL_NAMES or slot.called_entities)
    )
    writes = any(bare_tool_name(name) in HA_WRITE_TOOL_NAMES for name in tool_names)
    return AnswerScope(tool_names=tool_names, entity_ids=frozenset(slot.called_entities) if writes else None)


def build_answer_request(
    *,
    incoming: "FollowUpRequest | None",
    final_text: str,
    reply_text: str,
    called_tools: "set[str] | frozenset[str]",
    answer_scope: "AnswerScope | None",
    proposals_only: bool,
    prior_exchange: "list[dict[str, Any]] | None",
    playback_ends_at: "float | None",
    answer_only_from: "str | None",
    expects_reply: bool = False,
    offered_tools: "frozenset[str]" = frozenset(),
    called_entities: "frozenset[str] | None" = None,
) -> "FollowUpRequest | None":
    """The request for the window after this answer, or `None` for no window.

    The scope is the exact set of tools this turn dispatched, narrowed by the
    scope this turn itself ran under, so a chain never widens (D-10). When the
    turn dispatched no tool and the model asked the operator a question
    (`expects_reply`), the scope is `offered_tools` instead, the catalog the
    turn offered. It is narrowed the same way. No window opens past
    `MAX_CHAINED_FOLLOW_UPS` links or after a blank reply.
    """
    chain_depth = (incoming.chain_depth if incoming is not None else 0) + 1
    if chain_depth > MAX_CHAINED_FOLLOW_UPS or not reply_text.strip():
        return None
    scope_tools = frozenset(called_tools)
    if not scope_tools and expects_reply:
        scope_tools = frozenset(offered_tools)
    return FollowUpRequest(
        kind="answer",
        chain_depth=chain_depth,
        original_transcript=final_text,
        question=reply_text,
        prior_messages=tuple(prior_exchange) if prior_exchange else (),
        playback_ends_at=playback_ends_at,
        proposals_only=proposals_only,
        answer_scope=AnswerScope(tool_names=scope_tools, entity_ids=called_entities).narrowed_by(answer_scope),
        answer_only_from=answer_only_from,
        expects_reply=expects_reply,
    )


def answer_turn_scope(incoming: FollowUpRequest) -> AnswerScope:
    """What the window turn may reach. A request with no recorded scope
    fails closed and offers no tool. This differs from a clarification,
    which defers to `proposals_only`."""
    if incoming.answer_scope is not None:
        return incoming.answer_scope
    return AnswerScope(tool_names=frozenset())


def wake_addressed_command(follow_up: Any, final_text: str, wake_phrase: "str | None") -> "str | None":
    """The command after the wake phrase when this window turn was addressed
    to Atlas, or `None` for an ordinary window turn (plan 13-06).

    The rule needs two facts together: the wake detector hit at the start of
    this window's audio (`follow_up.wake_heard`, set only for a hit inside
    `WAKE_EVIDENCE_WINDOW_S`), and a transcript that opens with the wake
    phrase. That is the same evidence an ordinary wake turn has with
    `verify_transcript`. Transcript text alone never qualifies, because a
    television can say "hey atlas" and any command. An empty string means
    only the phrase was heard.
    """
    if not getattr(follow_up, "wake_heard", False) or not wake_phrase:
        return None
    return strip_wake_phrase(final_text, wake_phrase)


def silent_answer_outcome(
    final: Any, final_text: str, wake_phrase: "str | None", *, expects_reply: bool = False
) -> "str | None":
    """The `turn_outcome` for a window turn that must end with no reply, or
    `None` when the turn has a command to run.

    Nothing heard ends as `ANSWER_WINDOW_SILENT`. A stop phrase ends as
    `"stopped"`, checked before the filler test, so "okay" and "thanks" end
    the window. Filler or a bare wake word ends as `"no_command"`.

    `expects_reply` is True when the reply that opened this window asked a
    question. A whole-utterance affirmative ("yeah", "okay") then answers it
    and runs as a turn. Every other stop phrase still ends the window.
    """
    if final is None or not final_text.strip():
        return ANSWER_WINDOW_SILENT
    if expects_reply and is_affirmative(final_text):
        return None
    if is_quiet_stop(final_text):
        return "stopped"
    if is_no_command(final_text, wake_phrase):
        return "no_command"
    return None


def answer_speaker_mismatch(
    incoming: "FollowUpRequest | None", *, speaker_event: dict, effective_mode: str
) -> bool:
    """True when an answer window was asked by one identified speaker and a
    different voice (or no identified voice) answers it, in enforce mode.

    This applies Phase 12 D-10 on the serial edge path, where `turn_context`
    is `None` and `follow_up_speaker_mismatch` cannot decide. Other modes
    never restrict, as D-12 requires.
    """
    if incoming is None or incoming.answer_only_from is None or effective_mode != "enforce":
        return False
    answering = speaker_event.get("speaker_id")
    return answering is None or str(answering) != incoming.answer_only_from
