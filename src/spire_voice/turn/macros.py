"""Macros: a shortcut past the language model, never past the safety boundary.

A macro fires when a normalized transcript exactly matches a normalized phrase
or alias -- no fuzzy matching and no embeddings, because a fuzzy match that
fires the wrong macro switches real power (D-11). Every macro action still
passes `allow_call` (`mcp/spire_mcp/safety.py`) at fire time, exactly like a
model-issued call: a macro is a shortcut past the model, never past the
boundary.

Import direction, load-bearing: `spire_voice.config` imports `normalize` from
this module, so this module must never import `spire_voice.config` at
runtime -- doing so would deadlock the two modules on import. Any config type
this module needs only for annotations goes under `typing.TYPE_CHECKING`,
with `from __future__ import annotations` already in effect below.

`normalize()` is this task's whole contribution. `match()` and `fire_macro()`
belong to plan 01.1-05 and extend this file rather than replace it.
"""

from __future__ import annotations

import re
import unicodedata

# Stripped after NFKC + casefold, before whitespace collapse. A deliberate
# class of ASCII punctuation, not `\W`, so a non-ASCII letter (accented, CJK,
# etc.) is preserved rather than stripped as "not a word character" under
# some locales' interpretation of `\W`.
_PUNCT_RE = re.compile(r"[.,!?;:'\"()\[\]{}\-_/\\]")


def normalize(text: str) -> str:
    """Fold `text` to the one canonical form macro matching compares against.

    Deliberately lossy: two phrases differing only in case, punctuation, or
    whitespace collapse to the same string. That loss is what makes load-time
    duplicate detection (`Config.from_config`) meaningful -- two macros whose
    phrases normalize to the same key are a real collision, not two
    coincidentally similar strings, and the config loader rejects that pair
    at load rather than letting one silently shadow the other at runtime.

    Four steps, none hand-rolled (RESEARCH.md's Don't-Hand-Roll table): NFKC
    compatibility folding, `casefold()` (not `lower()` -- casefold is the one
    that handles non-ASCII correctly), punctuation stripping, then whitespace
    collapse. Comparison is by code point throughout, never by byte length.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _PUNCT_RE.sub("", text)
    return " ".join(text.split())
