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

`normalize()` was plan 01.1-02's contribution. `match()` and `fire_macro()`
below are plan 01.1-05's: the exact-match lookup against a transcript, and
the sequential, safety-gated action runner. Both stay in this file rather
than a new one, per the plan's own instruction to extend, not replace.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Sequence

if TYPE_CHECKING:
    # Annotation-only: `spire_voice.config` imports `normalize` from this
    # module (see above), so an import of `MacroConfig` here must never run
    # at module load, only under a type checker's eyes.
    from spire_voice.config import MacroConfig

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


class _ToolHost(Protocol):
    """Structural twin of `turn/controller.py`'s own `_ToolHost` protocol.

    Duplicated, not imported: importing `controller.py`'s protocol would
    import `controller.py` at module level, and `fire_macro` below already
    needs a *runtime* import from `controller.py` for `_is_error`/
    `_result_text` -- deferred to inside the function body for exactly that
    reason. A typing-only `Protocol` carries no runtime coupling either way,
    so duplicating this one small shape is cheaper than the alternative of
    restructuring either module around the other.
    """

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


def match(macros: Sequence["MacroConfig"], transcript: str) -> "MacroConfig | None":
    """Return the macro whose phrase or alias equals `transcript`, or `None`.

    Normalizes `transcript` once and returns `None` immediately if that
    normalized form is empty, whatever macros are configured -- an empty or
    whitespace-only transcript can never match. Otherwise compares the
    normalized transcript for exact equality against each macro's
    `normalized_keys` (`MacroConfig`'s own deduplicated set of `normalize()`
    applied to its phrase and every alias).

    Exact equality only -- no substring test, no prefix test, no edit
    distance, no embedding. A fuzzy match that fires the wrong macro
    switches real power (D-11); the operator deferred a looser matcher
    until Phase 2's recorded corpus shows how the 8 kHz camera path
    actually mis-transcribes known phrases. Because equality is the whole
    test, there is no length threshold to be off by one at, and no
    configuration can produce a cross-macro tie: `Config.from_config`'s own
    `_check_macros_do_not_collide` already rejected that pair before the
    process started.
    """
    normalized = normalize(transcript)
    if not normalized:
        return None
    for macro in macros:
        if normalized in macro.normalized_keys:
            return macro
    return None


@dataclass(frozen=True)
class MacroOutcome:
    """What one `fire_macro` call decided.

    Three things, not a bare tuple of loose values: whether every action
    returned success, the text to speak either way, and whether that text
    may be served from the startup precache rather than a live
    text-to-speech call. A tuple here would make the CMD-07 gate a
    convention a caller has to remember to honor; a typed result makes the
    caller read the flag instead.

    `cacheable` is `False` on failure by construction -- a failure reason is
    composed at fire time from whichever action actually failed, so it was
    never in the startup precache and cannot be served from it. Only a
    macro's own configured `reply`, spoken after every action succeeds, is
    ever `cacheable=True`.
    """

    succeeded: bool
    text: str
    cacheable: bool


async def fire_macro(macro: "MacroConfig", tool_host: _ToolHost) -> MacroOutcome:
    """Run `macro.actions` one at a time, in written order, through `tool_host`.

    Every action reaches `tool_host.call_tool` -- the exact entry a
    model-issued tool call uses -- so `allow_call` (`mcp/spire_mcp/safety.py`)
    gates it identically. A macro is a shortcut past the model, never past
    the boundary: there is no second, macro-specific path to Home Assistant,
    and no config-load validation this function trusts instead of re-asking
    the boundary at fire time.

    On the first error-shaped result, this stops -- the remaining actions
    never run -- and returns the boundary's own text unmodified as the
    reason: a refusal or a Home Assistant failure reaches the operator in
    the words the boundary wrote (CMD-08), with nothing prepended, appended,
    or reworded here. Actions run strictly in order, never concurrently,
    which is what makes "the action that actually failed" a well-defined,
    deterministic answer rather than a race between whichever call happens
    to finish first.

    The success/failure gate sits after the whole loop, not inside it:
    checking only the most recent result is the natural simplification and
    is wrong, because a macro whose first action fails and second action
    succeeds would still be confirmed. This mirrors `allow_call`'s own "one
    denied entity denies the whole call" doctrine, applied one level up --
    undoing it would let a partly-applied macro sound like a fully-applied
    one.
    """
    # Deferred, not module-level: `controller.py` imports this module at
    # load time to dispatch a macro hit (`match`/`fire_macro` above), so a
    # module-level import here of anything from `controller.py` would
    # deadlock the two modules on import. This mirrors `brain_race.py`'s own
    # deferred import of `controller._run_tool_rounds`, and reuses the same
    # two helpers rather than re-implementing MCP result introspection here.
    from spire_voice.turn.controller import _is_error, _result_text

    for action in macro.actions:
        result = await tool_host.call_tool(action.tool, action.arguments)
        if _is_error(result):
            return MacroOutcome(succeeded=False, text=_result_text(result), cacheable=False)

    return MacroOutcome(succeeded=True, text=macro.reply, cacheable=True)
