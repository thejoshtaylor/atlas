"""The no-wake-word window after an ordinary answer (VOICE-21, Phase 13).

After the assistant answers on a source that opts in (the edge Pi), the
microphone stays open for a short time with no wake word, so the operator can
reply with "and the other one" or "turn it off". This module holds the pure
rules for that window. `turn/controller.py` calls them and stays short.

The window fails closed, because every word heard in it is untrusted. The
microphone hears the television, and a television can speak any sentence.
So the window turn may reach only the tools the answered turn itself
dispatched (D-10). After a conversation-only answer it reaches no tool at all
(D-11). Along a chain of windows the scope only narrows. A window turn never
runs a macro, a local on/off intent or a timer intent, because the continued
exchange skips those paths. A window that hears nothing speaks nothing (D-15).
"""

from __future__ import annotations

import re
from typing import Any

from atlas.timers.ring_stop import is_stop_command
from atlas.turn.follow_up import MAX_CHAINED_FOLLOW_UPS, AnswerScope, FollowUpRequest
from atlas.turn.transcript_guard import is_no_command

ANSWER_WINDOW_SILENT = "answer_window_silent"

# "turn it off" is a ring stop phrase, but in this window it is a real
# command about the thing the last answer touched. Any text with the word
# "off" goes to the brain, never to a silent stop.
_OFF_RE = re.compile(r"\boff\b", re.IGNORECASE)


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
) -> "FollowUpRequest | None":
    """The request for the window after this answer, or `None` for no window.

    The scope is the exact set of tools this turn dispatched, narrowed by the
    scope this turn itself ran under, so a chain never widens (D-10). No
    window opens past `MAX_CHAINED_FOLLOW_UPS` links or after a blank reply.
    """
    chain_depth = (incoming.chain_depth if incoming is not None else 0) + 1
    if chain_depth > MAX_CHAINED_FOLLOW_UPS or not reply_text.strip():
        return None
    return FollowUpRequest(
        kind="answer",
        chain_depth=chain_depth,
        original_transcript=final_text,
        question=reply_text,
        prior_messages=tuple(prior_exchange) if prior_exchange else (),
        playback_ends_at=playback_ends_at,
        proposals_only=proposals_only,
        answer_scope=AnswerScope(tool_names=frozenset(called_tools)).narrowed_by(answer_scope),
        answer_only_from=answer_only_from,
    )


def answer_turn_scope(incoming: FollowUpRequest) -> AnswerScope:
    """What the window turn may reach. A request with no recorded scope
    fails closed and offers no tool. This differs from a clarification,
    which defers to `proposals_only`."""
    if incoming.answer_scope is not None:
        return incoming.answer_scope
    return AnswerScope(tool_names=frozenset())


def silent_answer_outcome(final: Any, final_text: str, wake_phrase: "str | None") -> "str | None":
    """The `turn_outcome` for a window turn that must end with no reply, or
    `None` when the turn has a command to run.

    Nothing heard ends as `ANSWER_WINDOW_SILENT`. A stop phrase ends as
    `"stopped"`, checked before the filler test, so "okay" and "thanks" end
    the window. Filler or a bare wake word ends as `"no_command"`.
    """
    if final is None or not final_text.strip():
        return ANSWER_WINDOW_SILENT
    if is_stop_command(final_text) and not _OFF_RE.search(final_text):
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
