"""The words a person wrote in a Gmail message, and nothing they quoted.

D-17 and D-20 both need this before a message body reaches speech, a summary,
or a style sample: no quoted history, no forwarded block, no signature, no
"Sent from my iPhone" footer. Plans 09-08 (reading and summarizing) and 09-09
(style learning and drafting) both call this module from the MCP child and
from the backend, so it carries no import from `atlas` -- the MCP child's
`PYTHONPATH` is the `mcp/` directory only.

09-RESEARCH.md's Don't Hand-Roll table named quote/signature stripping as a
parsing problem a library should solve, and recommended `talon`. The only
`talon` on PyPI is 1.4.4, a 2017 source-only build pulling in `cchardet`,
`scipy`, and `scikit-learn` -- not a safe bet on this project's Python 3.14.
09-02-PLAN.md's Task 1 replaced that choice with `mail-parser-reply`
(human-approved, gate="blocking-human"; see pyproject.toml).

Two layers do the cleaning (T-09-09): `mail-parser-reply`'s
`EmailReplyParser` when it is importable, or a small standard-library rule
set when it is not -- and in both cases a backstop afterwards drops any
quoted (`>`) line and any "Sent from my ..." footer that survived, so a body
that is nothing but quoted history cleans to an empty string, never to the
quote.
"""

from __future__ import annotations

import base64
import re
from html.parser import HTMLParser
from typing import Sequence

try:
    from mailparser_reply import EmailReplyParser as _EmailReplyParser

    _HAS_LIBRARY = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _EmailReplyParser = None  # type: ignore[assignment]
    _HAS_LIBRARY = False

#: The most cleaned body text any model round is ever given.
MODEL_INPUT_CAP = 8000

_CONTENT_TYPE_HEADER_RE = re.compile(r"^content-type$", re.IGNORECASE)
_CHARSET_RE = re.compile(r'charset="?([\w-]+)"?', re.IGNORECASE)

_QUOTE_LINE_RE = re.compile(r"^>")
_SENT_FROM_RE = re.compile(r"^sent from my ", re.IGNORECASE)
_GET_OUTLOOK_RE = re.compile(r"^get outlook for ", re.IGNORECASE)
_FORWARDED_BANNER_RE = re.compile(r"^-{2,}\s*forwarded message\s*-{2,}$", re.IGNORECASE)
_ORIGINAL_MESSAGE_RE = re.compile(r"^-{3,}\s*original message\s*-{3,}$", re.IGNORECASE)
_ON_LINE_RE = re.compile(r"^on\s", re.IGNORECASE)
_WROTE_TAIL_RE = re.compile(r"wrote:\s*$", re.IGNORECASE)
_FROM_LINE_RE = re.compile(r"^from:", re.IGNORECASE)
_SENT_OR_DATE_LINE_RE = re.compile(r"^(sent|date):", re.IGNORECASE)
_SIGNATURE_MARKER_RE = re.compile(r"^--\s?$")
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?=\s|$)")

_VOID_TAGS = frozenset({"br", "img", "hr", "input", "meta", "link"})
_SKIP_TAGS = frozenset({"style", "script", "head"})
_BLOCK_END_TAGS = frozenset({"p", "div", "li", "tr"})
_SKIP_CLASSES = frozenset({"gmail_quote", "gmail_signature"})


def extract_text(payload: dict) -> str:
    """The message text from a Gmail `format=full` payload.

    Walks `payload`'s parts depth-first for the first `text/plain` part; if
    none exists, the first `text/html` part through `html_to_text`. A part's
    `body.data` is base64url with the `=` padding already stripped by Gmail,
    so padding is restored before decoding; the part's own declared charset
    (default utf-8, `errors="replace"`) decodes the bytes.
    """
    plain_part = _find_part(payload, "text/plain")
    if plain_part is not None:
        return _decode_part(plain_part)

    html_part = _find_part(payload, "text/html")
    if html_part is not None:
        return html_to_text(_decode_part(html_part))

    return ""


def _find_part(payload: dict, mime_type: str) -> dict | None:
    if not payload:
        return None
    if payload.get("mimeType", "").lower() == mime_type and (payload.get("body") or {}).get("data"):
        return payload
    for part in payload.get("parts") or []:
        found = _find_part(part, mime_type)
        if found is not None:
            return found
    return None


def _decode_part(part: dict) -> str:
    data = (part.get("body") or {}).get("data") or ""
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
    except Exception:
        return ""
    charset = _get_charset(part)
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _get_charset(part: dict) -> str:
    for header in part.get("headers") or []:
        name = header.get("name", "")
        if _CONTENT_TYPE_HEADER_RE.match(name):
            match = _CHARSET_RE.search(header.get("value", ""))
            if match:
                return match.group(1)
    return "utf-8"


class _HtmlTextExtractor(HTMLParser):
    """Plain text from HTML, dropping quoted and signature blocks structurally."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_opened: list[bool] = []
        self._skip_depth = 0

    def _opens_skip(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag == "blockquote" or tag in _SKIP_TAGS:
            return True
        classes = (dict(attrs).get("class") or "").split()
        return bool(_SKIP_CLASSES.intersection(classes))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            if tag == "br" and self._skip_depth == 0:
                self._chunks.append("\n")
            return
        opens = self._skip_depth == 0 and self._opens_skip(tag, attrs)
        self._skip_opened.append(opens)
        if opens:
            self._skip_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "br" and self._skip_depth == 0:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            return
        if self._skip_opened:
            if self._skip_opened.pop():
                self._skip_depth = max(0, self._skip_depth - 1)
        if self._skip_depth == 0 and tag in _BLOCK_END_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def html_to_text(html: str) -> str:
    """Plain text from an HTML email body.

    Skips everything inside `blockquote`, any element whose class list
    contains `gmail_quote` or `gmail_signature`, and `style`/`script`/`head`.
    `br` and the end of `p`/`div`/`li`/`tr` become newlines; entities decode
    via `convert_charrefs=True`.
    """
    parser = _HtmlTextExtractor()
    parser.feed(html)
    parser.close()
    text = re.sub(r"\n{3,}", "\n\n", parser.text())
    return text.strip()


def clean_body(text: str, *, languages: Sequence[str] = ("en",)) -> str:
    """The reply text a person wrote, with quoted history, forwarded blocks,
    and signatures removed.

    Prefers `mail-parser-reply`'s `EmailReplyParser` when it is importable;
    falls back to a standard-library rule set otherwise (or when the library
    returns nothing). Either way, a backstop afterwards drops any surviving
    `>`-quoted line, "Sent from my ..." footer, and forwarded-message banner
    -- the library has no concept of that last banner text on its own -- then
    collapses long runs of blank lines and strips.
    """
    if not text:
        return ""

    cleaned = ""
    if _HAS_LIBRARY:
        try:
            message = _EmailReplyParser(languages=list(languages)).read(text=text)
            if message.replies:
                cleaned = message.replies[0].body or ""
        except Exception:
            cleaned = ""

    if not cleaned.strip():
        cleaned = _fallback_clean(text)

    return _apply_backstop(cleaned)


def _fallback_clean(text: str) -> str:
    """Standard-library rule set used when the library is absent or empty."""
    lines = text.split("\n")
    cut = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _looks_like_wrote_attribution(lines, i):
            cut = i
            break
        if _ORIGINAL_MESSAGE_RE.match(stripped):
            cut = i
            break
        if _FORWARDED_BANNER_RE.match(stripped):
            cut = i
            break
        if _FROM_LINE_RE.match(stripped) and _has_sent_or_date_within(lines, i):
            cut = i
            break
    lines = lines[:cut]

    for i, line in enumerate(lines):
        if _SIGNATURE_MARKER_RE.match(line):
            lines = lines[:i]
            break

    lines = [line for line in lines if not _GET_OUTLOOK_RE.match(line.strip())]
    return "\n".join(lines)


def _looks_like_wrote_attribution(lines: list[str], i: int) -> bool:
    line = lines[i].strip()
    if not _ON_LINE_RE.match(line):
        return False
    if _WROTE_TAIL_RE.search(line):
        return True
    # Allows the attribution to wrap onto a second line.
    if i + 1 < len(lines) and _WROTE_TAIL_RE.search(lines[i + 1].strip()):
        return True
    return False


def _has_sent_or_date_within(lines: list[str], i: int, window: int = 4) -> bool:
    for j in range(i + 1, min(i + 1 + window, len(lines))):
        if _SENT_OR_DATE_LINE_RE.match(lines[j].strip()):
            return True
    return False


def _apply_backstop(text: str) -> str:
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if _QUOTE_LINE_RE.match(stripped):
            continue
        if _SENT_FROM_RE.match(stripped):
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = _cut_at_forwarded_banner(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _cut_at_forwarded_banner(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if _FORWARDED_BANNER_RE.match(line.strip()):
            return "\n".join(lines[:i])
    return text


def cap_text(text: str, limit: int) -> tuple[str, bool]:
    """`text` cut to at most `limit` characters, at a sentence or word boundary.

    Returns `(text, False)` unchanged when already within `limit`, or
    `(capped, True)` otherwise.
    """
    if len(text) <= limit:
        return text, False

    window = text[:limit]
    boundaries = [m.end() for m in _SENTENCE_BOUNDARY_RE.finditer(window)]
    if boundaries:
        return text[: boundaries[-1]].rstrip(), True

    last_space = window.rfind(" ")
    if last_space > 0:
        return text[:last_space].rstrip(), True

    return window, True
