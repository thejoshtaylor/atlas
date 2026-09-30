"""Keep raw tool errors out of speech, and give the model a short retry line.

A tool host can return a pydantic `ValidationError` as its error text. That
text is long, has line breaks and holds a URL. Spoken aloud, it cost a live
turn about 37 s of TTS. This module holds the rules that stop that:

- `is_speakable_error` decides whether error text may be spoken verbatim. A
  short single-line refusal can. A validation dump, a URL, a traceback or a
  long text cannot. The caller speaks an apology instead and logs the full
  text.
- `is_argument_error` and `condense_argument_error` handle the other half. An
  argument error is a malformed call, not a refusal. It goes back to the model
  as one short line, so the model can correct the call.

The module imports only the standard library.
"""

from __future__ import annotations

import re

ARGUMENT_ERROR_PREFIX = "invalid arguments: "
MAX_SPOKEN_ERROR_CHARS = 300
MAX_MODEL_ERROR_CHARS = 500

_HEADER_RE = re.compile(r"\b\d+ validation errors? for \S+")
_VALUE_PREFIXES = ("Value error, ", "Assertion failed, ")
_TYPE_MARKER = " [type="
_INFO_MARKER = "For further information visit"


def is_argument_error(text: str) -> bool:
    """True when `text` is a malformed-call error, not a refusal."""
    return text.startswith(ARGUMENT_ERROR_PREFIX) or _HEADER_RE.search(text) is not None


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _cut(text: str) -> str:
    if len(text) <= MAX_MODEL_ERROR_CHARS:
        return text
    return text[: MAX_MODEL_ERROR_CHARS - 3].rstrip() + "..."


def _entries(text: str) -> list[str]:
    """Read the `str(ValidationError)` layout into "location: message" entries."""
    entries: list[str] = []
    location: str | None = None
    for line in text.splitlines():
        if not line.strip() or _HEADER_RE.search(line) or _INFO_MARKER in line:
            continue
        cut = line.find(_TYPE_MARKER)
        if cut != -1:
            line = line[:cut]
        body = line.strip()
        for prefix in _VALUE_PREFIXES:
            if body.startswith(prefix):
                body = body[len(prefix) :]
        if not line[:1].isspace():
            location = body
            continue
        entries.append(f"{location}: {body}" if location else body)
        location = None
    return entries


def condense_argument_error(text: str) -> str:
    """One short line for the model: the prefix, then each field and its message."""
    if text.startswith(ARGUMENT_ERROR_PREFIX):
        return _cut(_collapse(text))
    entries = _entries(text)
    if entries:
        body = "; ".join(entries)
    else:
        lines = text.strip().splitlines()
        body = lines[0] if lines else ""
    return _cut(_collapse(ARGUMENT_ERROR_PREFIX + body))


def is_speakable_error(text: str) -> bool:
    """True when `text` may be spoken verbatim as a tool refusal.

    Empty text is speakable, because the callers keep their own fallback for it.
    """
    if not text:
        return True
    if is_argument_error(text):
        return False
    stripped = text.strip()
    if "\n" in stripped or "\r" in stripped:
        return False
    if "http://" in text or "https://" in text or "Traceback" in text:
        return False
    return len(text) <= MAX_SPOKEN_ERROR_CHARS
