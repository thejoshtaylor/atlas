"""Handles the `email_list` and `email_read` handoffs `mcp/atlas_mcp/google.py`
produces for every Gmail read (D-14 .. D-18) -- the Gmail equivalent of
`turn/pending_action.py`'s own "composed in code, never a second model
round" discipline, extended with the one thing a calendar proposal never
needed: summarizing untrusted body text through `turn/quarantine.py`
before any of it can reach speech.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from atlas_mcp.mail_clean import cap_text

from atlas.turn.email_memory import EmailListItem, ResolveMiss
from atlas.turn.quarantine import SUMMARY_INSTRUCTION, QuarantineError, quarantine_round

if TYPE_CHECKING:
    from atlas.turn.handoff import Handoff, HandoffContext, HandoffOutcome

# GOOG-12's own fixed note, duplicated here rather than imported from
# `turn/controller.py` -- that module imports `turn/handoff.py` (which
# imports this module) at load time, so a module-level import back from
# here would deadlock the two modules' load order, the same reason
# `turn/pending_action.py` defers its own `controller.py` import into a
# function body.
_UNREACHABLE_ACCOUNT_NOTE = " i can't reach your {label} account right now."

# Fixed, spoken-word-for-word replies (D-14, D-15) -- never composed by a
# model, the same doctrine every other fixed reply in this turn pipeline
# already follows.
_NO_UNREAD_REPLY = "no new email."
_NO_SEARCH_RESULTS_REPLY = "i didn't find any email matching that."
_UNREAD_COUNT_CARRIER = "you have {count} new emails."
_FOUND_COUNT_CARRIER = "i found {count} emails."
_SUMMARY_FAILED_REPLY = "i couldn't summarize that one"

# D-15: two or fewer unread/found messages get a summary each; more than
# that get the count and the senders, grouped by account then sender.
_SUMMARIZE_UP_TO = 2

# Task 3 (D-16, GOOG-08): the fixed replies `handle_email_read` speaks for
# every way "read the second one" / "read Dana's" can fail to resolve, and
# for a message with nothing left to read once cleaned.
_NO_RECENT_LIST_REPLY = "i don't have a recent email list -- ask me what's new first."
_OUT_OF_RANGE_CARRIER = "there were only {count} in the last list."
_NO_SENDER_MATCH_CARRIER = "i don't see an email from {sender} in the last list."
_EMPTY_BODY_REPLY = "that email has no text i can read."
_FETCH_UNAVAILABLE_REPLY = "i can't check email right now"

# D-17: a word-for-word read uses the child's own cleaned text directly --
# `mail_clean.clean_body`'s quote/signature/footer removal already applied
# to it in `handle_gmail_fetch_body` -- capped at `VERBATIM_READ_CAP`
# characters, distinct from (and much smaller than) `mail_clean.MODEL_INPUT_CAP`.
VERBATIM_READ_CAP = 1500
_VERBATIM_CARRIER = "{sender} wrote: {text}"
_VERBATIM_TRUNCATED_SUFFIX = " ... that's where i stop reading."


def _sender_display(item: EmailListItem) -> str:
    """`from_name` when the header carried one, else the address's own
    local part -- never the bare address (an operator hears a name, not
    an email address, whenever one exists)."""
    return item.from_name or item.from_address.split("@", 1)[0]


def _account_order(items: "tuple[EmailListItem, ...]") -> "list[str]":
    """Every distinct account in `items`, in first-appearance order --
    `items` is already newest-first (the child's own sort), so this is
    "whichever account's most recent message is newest" order, not an
    alphabetical one."""
    seen: "list[str]" = []
    for item in items:
        if item.account not in seen:
            seen.append(item.account)
    return seen


def _grouped(
    items: "tuple[EmailListItem, ...]",
) -> "list[tuple[str, list[tuple[str, list[EmailListItem]]]]]":
    """`items` grouped by account (`_account_order`), then within each
    account by sender display name -- sender groups ordered by count
    descending, ties broken by first appearance (recency, since `items`
    is newest-first). The one grouping both `compose_email_list_reply`'s
    "more than two" branch and this module's own spoken-order flattening
    (`_order_for_speaking`) share, so the two can never silently drift
    apart."""
    result: "list[tuple[str, list[tuple[str, list[EmailListItem]]]]]" = []
    for account in _account_order(items):
        account_items = [item for item in items if item.account == account]
        groups: "dict[str, list[EmailListItem]]" = {}
        first_index: "dict[str, int]" = {}
        for index, item in enumerate(account_items):
            key = _sender_display(item)
            groups.setdefault(key, []).append(item)
            first_index.setdefault(key, index)
        sender_keys = sorted(groups, key=lambda k: (-len(groups[k]), first_index[k]))
        result.append((account, [(key, groups[key]) for key in sender_keys]))
    return result


def _order_for_speaking(items: "tuple[EmailListItem, ...]") -> "tuple[EmailListItem, ...]":
    """The order items are spoken in, and therefore the order
    `EmailListMemory` stores them under (positions `1..N`): unchanged
    (newest-first, the child's own order) for `_SUMMARIZE_UP_TO` or fewer
    items; the flattened account/sender grouping for anything larger, so
    a later "read the second one" resolves against what was actually
    said, not the child's raw fetch order."""
    if len(items) <= _SUMMARIZE_UP_TO:
        return items
    ordered: "list[EmailListItem]" = []
    for _account, sender_groups in _grouped(items):
        for _sender, group_items in sender_groups:
            ordered.extend(group_items)
    return tuple(ordered)


def compose_email_list_reply(
    items: "tuple[EmailListItem, ...]",
    *,
    source_kind: str,
    has_more: "list[str]",
    unreachable: "list[dict[str, str]]",
    summaries: "dict[str, str]",
) -> str:
    """The one sentence `handle_email_list` speaks -- composed entirely
    from `items` and `summaries` (D-14, D-15), never from a model. Zero
    items speaks a source-specific fixed reply; up to `_SUMMARIZE_UP_TO`
    items names each one's own account, sender, and `summaries` entry
    (falling back to `_SUMMARY_FAILED_REPLY` for a message that has none);
    more than that names the count and the senders, grouped by account.

    An account in `unreachable` gets GOOG-12's own fixed note appended,
    one per account, regardless of which branch above produced the base
    reply -- the identical "append, by code, never silently left out"
    discipline `turn/controller.py`'s own ordinary-answer path already
    applies to a calendar read.
    """
    if not items:
        base = _NO_UNREAD_REPLY if source_kind != "search" else _NO_SEARCH_RESULTS_REPLY
    elif len(items) <= _SUMMARIZE_UP_TO:
        segments = [
            f"{item.account}: {_sender_display(item)}: {summaries.get(item.message_id, _SUMMARY_FAILED_REPLY)}"
            for item in items
        ]
        base = ". ".join(segments) + "."
    else:
        segments = []
        for account, sender_groups in _grouped(items):
            sender_clauses = [f"{len(group_items)} from {sender}" for sender, group_items in sender_groups]
            segments.append(f"{account}: {', '.join(sender_clauses)}")
        carrier = _FOUND_COUNT_CARRIER if source_kind == "search" else _UNREAD_COUNT_CARRIER
        base = f"{carrier.format(count=len(items))} " + ". ".join(segments) + "."

    for entry in unreachable:
        label = entry.get("account") if isinstance(entry, dict) else None
        if label:
            base = f"{base}{_UNREACHABLE_ACCOUNT_NOTE.format(label=label)}"
    return base


def _build_items(raw_items: "list[dict[str, Any]]") -> "tuple[EmailListItem, ...]":
    """`raw_items` (the child's own `email_list` payload shape, D-14) as
    `EmailListItem`s -- `position` is a placeholder here (`0`); the real,
    spoken-order position is assigned once, after `_order_for_speaking`
    settles the final order, so a memory read-back never has to guess
    which ordering a stale position number came from."""
    return tuple(
        EmailListItem(
            position=0,
            account=str(raw.get("account", "")),
            message_id=str(raw.get("message_id", "")),
            thread_id=str(raw.get("thread_id") or ""),
            from_name=str(raw.get("from_name") or ""),
            from_address=str(raw.get("from_address") or ""),
            subject=str(raw.get("subject") or ""),
        )
        for raw in raw_items
    )


def _positioned(items: "tuple[EmailListItem, ...]") -> "tuple[EmailListItem, ...]":
    return tuple(
        EmailListItem(
            position=index + 1,
            account=item.account,
            message_id=item.message_id,
            thread_id=item.thread_id,
            from_name=item.from_name,
            from_address=item.from_address,
            subject=item.subject,
        )
        for index, item in enumerate(items)
    )


async def _summarize(item: EmailListItem, ctx: "HandoffContext") -> "tuple[str, str]":
    """One item's own body, fetched through the code-only
    `gmail_fetch_body` tool and summarized in a quarantine round (D-17,
    D-18) -- `(message_id, summary_or_fallback)`, never raising: a fetch
    or summarization failure both fall back to `_SUMMARY_FAILED_REPLY`
    for this one item alone, never the whole turn."""
    # Deferred, not module-level: `turn/controller.py` imports
    # `turn/handoff.py` (which imports this module) at load time -- the
    # same load-order constraint `turn/pending_action.py`'s own deferred
    # `controller.py` import already documents.
    from atlas.turn.controller import _is_error, _result_payload

    try:
        result = await ctx.tool_host.call_tool(
            "gmail_fetch_body", {"account": item.account, "message_id": item.message_id}
        )
    except Exception:
        return item.message_id, _SUMMARY_FAILED_REPLY
    if _is_error(result):
        return item.message_id, _SUMMARY_FAILED_REPLY
    payload = _result_payload(result)
    body_text = payload.get("body", "") if isinstance(payload, dict) else ""
    try:
        summary = await quarantine_round(
            ctx.brain, instruction=SUMMARY_INSTRUCTION, content=body_text, timeout_s=ctx.quarantine_timeout_s
        )
    except QuarantineError:
        return item.message_id, _SUMMARY_FAILED_REPLY
    return item.message_id, summary


async def handle_email_list(handoff: "Handoff", ctx: "HandoffContext | None") -> "HandoffOutcome":
    """Resolve one `email_list` handoff into what to say (D-14, D-15) --
    stores the spoken list in `ctx.email_memory` (when given) so a later
    turn's `email_read` handoff can resolve "the second one" or "Dana's"
    against it (D-16).
    """
    from atlas.turn.handoff import HandoffOutcome

    payload = handoff.payload
    source_kind = str(payload.get("source") or "unread")
    unreachable = payload.get("unreachable_accounts") or []
    has_more = payload.get("has_more") or []
    raw_items = payload.get("items") or []

    ordered = _order_for_speaking(_build_items(raw_items))
    positioned = _positioned(ordered)

    summaries: "dict[str, str]" = {}
    if positioned and len(positioned) <= _SUMMARIZE_UP_TO and ctx is not None and ctx.tool_host is not None and ctx.brain is not None:
        results = await asyncio.gather(*(_summarize(item, ctx) for item in positioned))
        summaries = dict(results)
    elif positioned and len(positioned) <= _SUMMARIZE_UP_TO:
        summaries = {item.message_id: _SUMMARY_FAILED_REPLY for item in positioned}

    if ctx is not None and ctx.email_memory is not None:
        ctx.email_memory.store(ctx.source_name, positioned)

    reply = compose_email_list_reply(
        positioned, source_kind=source_kind, has_more=list(has_more), unreachable=list(unreachable), summaries=summaries
    )
    return HandoffOutcome(reply_text=reply, turn_outcome="email_list")


async def handle_email_read(handoff: "Handoff", ctx: "HandoffContext | None") -> "HandoffOutcome":
    """Resolve one `email_read` handoff -- "read the second one" or "read
    Dana's" -- against `ctx.email_memory`'s own last-spoken list for this
    source (D-16, GOOG-08). Makes no Gmail search of its own: the handoff
    carries only `position`/`sender`/`word_for_word` (`mcp/atlas_mcp/google.py::handle_gmail_read`
    made no request at all), so resolving "the second one" is entirely
    this function's own job.

    `turn_outcome` is `"email_read"` for every reply that actually names
    or reads a message (including the "no text" and resolve-miss
    fallbacks, which are correct answers, not failures) and
    `"email_read_failed"` only when the body fetch itself errored -- that
    one case speaks the failing tool's own error text verbatim (never
    "done", the same doctrine `pending_action.py::ExecutionResult`
    already follows for a failed calendar write).
    """
    from atlas.turn.handoff import HandoffOutcome

    payload = handoff.payload
    position = payload.get("position")
    sender = payload.get("sender")
    word_for_word = bool(payload.get("word_for_word", False))

    if ctx is None or ctx.email_memory is None:
        return HandoffOutcome(reply_text=_NO_RECENT_LIST_REPLY, turn_outcome="email_read")

    resolved = ctx.email_memory.resolve(ctx.source_name, position=position, sender=sender)
    if isinstance(resolved, ResolveMiss):
        if resolved.reason == "no_list":
            return HandoffOutcome(reply_text=_NO_RECENT_LIST_REPLY, turn_outcome="email_read")
        if resolved.reason == "out_of_range":
            return HandoffOutcome(
                reply_text=_OUT_OF_RANGE_CARRIER.format(count=resolved.count), turn_outcome="email_read"
            )
        # "no_sender" -- `sender` is the model's own tool argument, derived
        # from the transcript, not email content, so speaking it back is
        # safe (the plan's own action text).
        return HandoffOutcome(
            reply_text=_NO_SENDER_MATCH_CARRIER.format(sender=sender), turn_outcome="email_read"
        )

    item = resolved
    if ctx.tool_host is None:
        return HandoffOutcome(reply_text=_FETCH_UNAVAILABLE_REPLY, turn_outcome="email_read_failed")

    from atlas.turn.controller import _is_error, _result_payload, _result_text

    try:
        result = await ctx.tool_host.call_tool(
            "gmail_fetch_body", {"account": item.account, "message_id": item.message_id}
        )
    except Exception as exc:
        return HandoffOutcome(reply_text=str(exc), turn_outcome="email_read_failed")

    if _is_error(result):
        return HandoffOutcome(reply_text=_result_text(result), turn_outcome="email_read_failed")

    body_payload = _result_payload(result)
    body_text = str(body_payload.get("body", "")) if isinstance(body_payload, dict) else ""
    sender_display = _sender_display(item)

    if not body_text.strip():
        return HandoffOutcome(reply_text=_EMPTY_BODY_REPLY, turn_outcome="email_read")

    if word_for_word:
        capped, truncated = cap_text(body_text, VERBATIM_READ_CAP)
        text = capped + (_VERBATIM_TRUNCATED_SUFFIX if truncated else "")
        reply = _VERBATIM_CARRIER.format(sender=sender_display, text=text)
        return HandoffOutcome(reply_text=reply, turn_outcome="email_read")

    if ctx.brain is None:
        return HandoffOutcome(
            reply_text=f"{item.account}: {sender_display}: {_SUMMARY_FAILED_REPLY}", turn_outcome="email_read"
        )
    try:
        summary = await quarantine_round(
            ctx.brain, instruction=SUMMARY_INSTRUCTION, content=body_text, timeout_s=ctx.quarantine_timeout_s
        )
    except QuarantineError:
        summary = _SUMMARY_FAILED_REPLY

    reply = f"{item.account}: {sender_display}: {summary}"
    return HandoffOutcome(reply_text=reply, turn_outcome="email_read")
