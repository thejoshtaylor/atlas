"""The one text sanitizer for every remote string a Mac shows, and the wire
caps for the panel frames (Phase 15, D-11, CARD-08, PANEL-03).

A reply, a transcript or a timer label can hold words from a tool, a mail or
a television. Nothing here trusts it. The Mac sanitizes again and does not
trust the server (`TextSanitizer.sanitize` in AtlasKit). Both sides read the
same vectors from `desktop/protocol/v1/hostile_text.json`.

The rules, over code points:

- U+0009, U+000A, U+000B, U+000C, U+000D, U+0085, U+2028 and U+2029 become a
  space.
- Every other code point in U+0000-U+001F and U+007F-U+009F is removed, and so
  are the bidi controls U+202A-U+202E and U+2066-U+2069, U+200E, U+200F and
  U+061C.
- A run of U+0020 becomes one, and both ends lose U+0020.
- Everything else stays, U+00A0 and Markdown characters included.

Do not use `str.split()` with no argument here. It also splits on U+00A0 and
other spaces, and the Swift twin must give the same output.
"""

from __future__ import annotations

import re
from typing import Literal

ID_MAX = 64
WORD_MAX = 32
TRANSCRIPT_TEXT_MAX = 1000
CARD_TEXT_MAX = 600
LABEL_MAX = 120
MS_MAX = 600_000
TIMER_ID_MAX = 2_147_483_647

_ELLIPSIS = "…"
_TO_SPACE = (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x85, 0x2028, 0x2029)
_REMOVED = (
    {*range(0x00, 0x20), *range(0x7F, 0xA0)}
    | {*range(0x202A, 0x202F), *range(0x2066, 0x206A), 0x200E, 0x200F, 0x061C}
) - set(_TO_SPACE)
_TABLE: dict[int, str | None] = {code: " " for code in _TO_SPACE}
_TABLE.update({code: None for code in _REMOVED})
_SPACE_RUN = re.compile(" {2,}")


def sanitize_display_text(
    value: str, max_len: int, *, keep: Literal["start", "end"] = "start"
) -> str:
    """`value` as plain text safe to show, at most `max_len` code points.

    Over the cap, `keep="start"` keeps the first `max_len - 1` code points and
    ends with U+2026, and `keep="end"` starts with U+2026 and keeps the last
    `max_len - 1`. Spaces at the cut are removed."""
    assert max_len >= 2, "max_len must leave room for the cut mark"
    text = _SPACE_RUN.sub(" ", value.translate(_TABLE)).strip(" ")
    if len(text) <= max_len:
        return text
    if keep == "start":
        return text[: max_len - 1].rstrip(" ") + _ELLIPSIS
    return _ELLIPSIS + text[len(text) - (max_len - 1) :].lstrip(" ")
