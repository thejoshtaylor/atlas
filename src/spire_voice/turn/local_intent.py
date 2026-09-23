"""A plain on/off command matched against live entity state, no brain call.

`turn/controller.py::run_turn` calls `match_on_off` after the macro check and
before the tier race: a live-house bug (this task's own evidence, kept out of
this public repo -- see the plan) showed the round trip to a language model
costing several seconds on a command as simple as "turn off the lamp," with
an 8 kHz camera transcript garbled enough that the model sometimes answered
wrong anyway. Nothing here needs a model at all: an on/off command names one
entity and one of two services, and both are already in the live entity list
this turn already fetched for the tier race (`turn/controller.py`'s own
`state_task`).

`match_on_off` returns `None` for anything it does not recognize -- no dim,
no set, no question, no ambiguous name -- and the brain handles it exactly
as it always has. This module is deliberately narrow: it is a fast path for
the common case, never a second, competing command parser.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Literal

_PUNCT_RE = re.compile(r"[^\w\s]")

# Dropped wherever they appear -- "can you"/"could you" fall out of this set
# as their two words individually, which has the identical effect on a
# transcript this narrow without needing a two-word phrase match.
_FILLER_WORDS = frozenset({"please", "the", "my", "can", "you", "could"})

# Never part of an entity name -- stripped once the on/off signal has been
# found. "were" covers the "we're off"/"were off" garbled-verb case; "shut"
# maps to "off" on its own, with no explicit on/off word required.
_VERB_NOISE_WORDS = frozenset({"turn", "switch", "shut", "were"})

_ON_OFF_WORDS = frozenset({"on", "off"})

# Stripped once, from the end of a candidate's name only -- these are the
# physical-object words a spoken command routinely omits ("the cooler" for
# "Example Cooler Socket"), never a word that could itself be a room or
# device name.
_GENERIC_TRAILING_WORDS = frozenset(
    {"socket", "switch", "plug", "outlet", "light", "lights", "lamp"}
)

_CANDIDATE_DOMAINS = frozenset({"light", "switch", "fan", "input_boolean"})

# Table-tested in `tests/test_local_intent.py` against an invented entity
# set -- these are not tuned against a live corpus (RESEARCH.md's own
# caveat for `turn/wake_echo.py`'s similar constant applies here too).
_MIN_SCORE = 0.75
_MIN_MARGIN = 0.10

_WAKE_PREFIXES = ("hey spire ", "spire ")


@dataclass(frozen=True)
class LocalIntent:
    """One matched command: the tool call `turn/controller.py` makes, and
    the friendly name to log/speak against if needed."""

    domain: str
    service: str
    entity_id: str
    friendly_name: str


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Stripping punctuation this way also turns "we're" into "were" (the
    apostrophe is not a word character) -- exactly the garbled-verb form
    `_VERB_NOISE_WORDS` already expects, with no separate contraction
    handling needed.
    """
    return " ".join(_PUNCT_RE.sub("", text.lower()).split())


def _strip_wake_prefix(text: str) -> str:
    for prefix in _WAKE_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _parse_command(text: str) -> "tuple[str, Literal['on', 'off']] | None":
    """The spoken entity name and the on/off signal, or `None` for anything
    that is not a plain on/off command at all.

    Recognizes "turn|switch|shut on|off <name>" and "turn <name> on|off"
    -- a leading or trailing on/off token plus whatever is left over once
    every verb-noise word is stripped. "shut" alone (no explicit on/off
    word) also signals off. A verb is required: a bare "<name> off" is
    exactly what speech-to-text keyterm biasing makes of unclear
    background speech, so it goes to the brain instead. Nothing else
    matches: no dim, no set, no question.
    """
    normalized = _strip_wake_prefix(_normalize(text))
    tokens = [word for word in normalized.split() if word not in _FILLER_WORDS]

    # The one accepted typo: "turn of the fan" for "turn off the fan" --
    # only right after a verb, never a bare "of" floating anywhere else in
    # the transcript, where it is far more likely to be the real word "of".
    for i in range(1, len(tokens)):
        if tokens[i] == "of" and tokens[i - 1] in ("turn", "switch", "shut"):
            tokens[i] = "off"

    if not _VERB_NOISE_WORDS.intersection(tokens):
        return None

    on_off_index = next((i for i, word in enumerate(tokens) if word in _ON_OFF_WORDS), None)

    if on_off_index is None:
        if "shut" not in tokens:
            return None
        signal: "Literal['on', 'off']" = "off"
        name_tokens = [word for word in tokens if word not in _VERB_NOISE_WORDS]
    else:
        signal = "on" if tokens[on_off_index] == "on" else "off"
        name_tokens = [
            word
            for i, word in enumerate(tokens)
            if i != on_off_index and word not in _VERB_NOISE_WORDS
        ]

    spoken_name = " ".join(name_tokens)
    if not spoken_name:
        return None
    return spoken_name, signal


def _forms(entity: "dict[str, Any]") -> tuple[str, str]:
    """A candidate's name, two ways: as given, and with one trailing generic
    physical-object word removed. `friendly_name` when the entity has one;
    the object id with underscores as spaces otherwise -- the same fallback
    `turn/controller.py`'s own `friendly_names` dict declines to make,
    since a candidate here always needs some name to score against.
    """
    friendly_name = entity.get("friendly_name")
    if not friendly_name:
        object_id = entity["entity_id"].split(".", 1)[1]
        friendly_name = object_id.replace("_", " ")
    primary = friendly_name.lower()
    words = primary.split()
    if words and words[-1] in _GENERIC_TRAILING_WORDS:
        secondary = " ".join(words[:-1])
    else:
        secondary = primary
    return primary, secondary


def match_on_off(text: str, entities: "list[dict[str, Any]]") -> LocalIntent | None:
    """The one entry point: a matched on/off command, or `None`.

    `entities` is the `ha_list_entities` shape (`entity_id`, `friendly_name`,
    `state`) -- exactly what `turn/controller.py`'s own `state_task` already
    fetches for the tier race, passed through unchanged. Only `light`,
    `switch`, `fan`, and `input_boolean` entities are ever candidates, and an
    `unavailable` entity is never one -- calling a service on an entity that
    is not there to answer would be a worse failure than falling back to the
    brain.

    Confident only when the best-scoring candidate reaches `_MIN_SCORE` and
    beats the next-best *different* entity by at least `_MIN_MARGIN` --
    otherwise `None`, same as no match at all: an ambiguous or low-confidence
    guess is exactly the kind of silent misrouting a fast path must not
    introduce, and `turn/controller.py` falls back to the brain, which can
    still ask a clarifying question if it needs to.
    """
    parsed = _parse_command(text)
    if parsed is None:
        return None
    spoken_name, signal = parsed

    scored: list[tuple[float, "dict[str, Any]"]] = []
    for entity in entities:
        entity_id = entity.get("entity_id", "")
        domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
        if domain not in _CANDIDATE_DOMAINS:
            continue
        if entity.get("state") == "unavailable":
            continue
        primary, secondary = _forms(entity)
        score = max(
            difflib.SequenceMatcher(None, spoken_name, primary).ratio(),
            difflib.SequenceMatcher(None, spoken_name, secondary).ratio(),
        )
        scored.append((score, entity))

    if not scored:
        return None

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_entity = scored[0]
    if best_score < _MIN_SCORE:
        return None

    second_best_score = next(
        (
            score
            for score, entity in scored[1:]
            if entity.get("entity_id") != best_entity.get("entity_id")
        ),
        None,
    )
    if second_best_score is not None and (best_score - second_best_score) < _MIN_MARGIN:
        return None

    domain = best_entity["entity_id"].split(".", 1)[0]
    return LocalIntent(
        domain=domain,
        service="turn_on" if signal == "on" else "turn_off",
        entity_id=best_entity["entity_id"],
        friendly_name=best_entity.get("friendly_name", best_entity["entity_id"]),
    )
