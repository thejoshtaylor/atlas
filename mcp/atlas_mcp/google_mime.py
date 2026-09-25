"""Builds one reply MIME message -- standard library only
(`email.message.EmailMessage`), no hand-rolled MIME construction
(09-RESEARCH.md's own Don't Hand-Roll table). A thread needs matching
`Subject`, `In-Reply-To`, and `References`, not only Gmail's own
`threadId` (09-RESEARCH.md Pitfall 2) -- `build_reply_message` is the one
place all four are derived together, from the original message's own
headers, never from anything the model or the operator's instructions
supplied.

No `From` header is ever set: Gmail fills in the sending account's own
address, and setting one here would either be ignored or, worse, spoofed
if it ever diverged from the account actually creating the draft.
"""

from __future__ import annotations

import base64
import html
import re
from email.message import EmailMessage
from email.utils import formataddr

_RE_PREFIX_RE = re.compile(r"^re:\s", re.IGNORECASE)
_LINE_BREAK_CHARS = ("\n", "\r")


def _reject_line_breaks(label: str, value: str) -> None:
    if any(ch in value for ch in _LINE_BREAK_CHARS):
        raise ValueError(f"{label} contains a line break")


def build_reply_message(
    *,
    to_name: str,
    to_address: str,
    original_subject: str,
    original_message_id: str,
    original_references: str,
    body_text: str,
    signature_text: "str | None",
    signature_html: "str | None",
) -> EmailMessage:
    """One `multipart/alternative` reply, threaded on `original_message_id`
    (T-09-50): `In-Reply-To` is exactly `original_message_id`; `References`
    is `original_references` with `original_message_id` appended (or just
    `original_message_id` when the original carried no `References` of its
    own); `Subject` is `original_subject` prefixed `"Re: "` unless it
    already carries one (no double `"Re: Re: ..."`).

    Every header value here is checked for an embedded line break first
    (`ValueError` if one is found) -- a header-injection attempt in the
    original message's own `From`/`Subject`/`Message-Id`/`References`
    must never reach a real MIME header.

    The plain part is `body_text` plus `signature_text` (when given); the
    HTML part is `body_text`, HTML-escaped, plus `signature_html` verbatim
    (already HTML, never escaped a second time).
    """
    for label, value in (
        ("the sender's name", to_name),
        ("the sender's address", to_address),
        ("the original subject", original_subject),
        ("the original message id", original_message_id),
        ("the original references", original_references),
    ):
        _reject_line_breaks(label, value)

    subject = original_subject if _RE_PREFIX_RE.match(original_subject) else f"Re: {original_subject}"
    references = f"{original_references} {original_message_id}".strip() if original_references else original_message_id

    message = EmailMessage()
    message["To"] = formataddr((to_name, to_address)) if to_name else to_address
    message["Subject"] = subject
    message["In-Reply-To"] = original_message_id
    message["References"] = references

    plain_body = f"{body_text}\n\n{signature_text}" if signature_text else body_text
    message.set_content(plain_body)

    escaped_body = html.escape(body_text).replace("\n", "<br>\n")
    html_body = f"<div>{escaped_body}</div>"
    if signature_html:
        html_body += signature_html
    message.add_alternative(html_body, subtype="html")

    return message


def encode_raw(message: EmailMessage) -> str:
    """`message` as the URL-safe base64 string `drafts.create`'s own
    `raw` field expects."""
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
