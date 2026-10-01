"""Is a final transcript just the wake phrase echoing back, with no command?

The camera's `stt.endpointing_ms` (200 ms by default) is tuned for a snappy
end-of-speech, and an operator who pauses for even a beat after "hey atlas"
gets exactly that pause read as the end of the utterance -- the wake phrase
itself, or a mangled echo of it ("at last" for "hey atlas"), becomes the whole
final transcript and the turn ends there, one word short of the actual
command (260922-woc's own evidence: two live turns each closed on 2.1 s of
audio, one of them nothing but the wake word's tail).

`is_wake_only` is the pure predicate `turn/controller.py` checks before
deciding whether to drain the STT stream a second time for the command that
follows. Two checks, both against a word-suffix of the configured phrase
("hey atlas", "atlas" for the default two-word phrase), never against the
phrase as a whole alone -- a short reply that only echoes the tail ("at less")
must count as wake-only exactly as readily as one that echoes the whole
phrase ("hey atlas").

Word-count first: `text` must have at most `len(phrase.split()) + 1` words,
so a real command that happens to share a fuzzy-matchable prefix ("at last
the lights" against "atlas") is never misread as wake-only by length alone.
Both checks run on lowercased text -- the word-count check additionally
strips punctuation and collapses whitespace first (`"hey, atlas"` splits
into two words, not `["hey,", "atlas"]` read as if the comma mattered to
the count); the similarity check does not strip punctuation, only
lowercases and collapses whitespace, because `difflib.SequenceMatcher`'s
ratio is sensitive to exactly this kind of near-miss and a comma-heavy STT
transcript ("hey, atlas") should still read as extremely close to the
clean phrase, not artificially perfect once its punctuation is scrubbed
away.

`difflib.SequenceMatcher` rather than a hand-rolled edit distance: it is
already in the standard library, already used the same way `turn/macros.py`
documents this project avoiding hand-rolled string comparisons in general.
`0.55` is not a tuned constant from real data -- it is the value that
separated the wake-only samples this task's own evidence produced under the
old "hey spire" phrase. Re-measured for "hey atlas": wake-only ("at last"
0.83, "at less" 0.67, "hey, atlas" 0.95, "a atlas" 0.83), genuine commands
that must never trip this check ("stop" 0.22, "turn on the lights" 0.37,
"at last the lights" 0.43) -- provisional until a real
corpus of camera turns says otherwise.

`strip_wake_phrase` (260929-icf) is the second function: it checks that a
transcript opens with the wake phrase and returns the text after it.

`is_wake_without_command` and `WakeHold` (261001-glr) use both to keep one
stream open: a provider that can hold a final asks `WakeHold` whether a final
is only the wake phrase, and holds it when so.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_PUNCT_RE = re.compile(r"[.,!?;:'\"()\[\]{}\-_/\\]")

# Below this, a transcript reads as a real command rather than an echo of
# the wake phrase -- see the module docstring for how this value was picked.
_SIMILARITY_THRESHOLD = 0.55

# The lowest similarity between the phrase keyword (its last word) and the
# words at the start of a transcript that still counts as the wake phrase.
_KEYWORD_SIMILARITY = 0.7


# A lead-in word before the keyword must be one of these, or look like one of
# the phrase's own lead words at this similarity or better.
_LEAD_FILLERS = frozenset({"a", "the", "oh", "ok", "okay", "hi"})
_LEAD_SIMILARITY = 0.6


def _similar(candidate: str, keyword: str) -> bool:
    return difflib.SequenceMatcher(None, candidate, keyword).ratio() >= _KEYWORD_SIMILARITY


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def is_wake_only(text: str, phrase: str) -> bool:
    """True when `text` is nothing but the wake phrase (or a mangled tail
    of it) and not a command.

    `text == ""` returns False unconditionally -- the existing
    empty-transcript path in `turn/controller.py` already closes that turn
    on its own, and this function has nothing to add there.
    """
    if not text:
        return False

    word_count_text = _collapse_whitespace(_PUNCT_RE.sub("", text.lower()))
    if not word_count_text:
        return False

    phrase_words = phrase.lower().split()
    if not phrase_words:
        return False

    words = word_count_text.split()
    if len(words) > len(phrase_words) + 1:
        return False

    similarity_text = _collapse_whitespace(text.lower())
    for start in range(len(phrase_words)):
        suffix = " ".join(phrase_words[start:])
        ratio = difflib.SequenceMatcher(None, similarity_text, suffix).ratio()
        if ratio >= _SIMILARITY_THRESHOLD:
            return True
    return False


def _may_lead(word: str, lead_words: list[str]) -> bool:
    """True when `word` may stand before the keyword in a wake phrase."""
    if not word or word in _LEAD_FILLERS:
        return True
    return any(difflib.SequenceMatcher(None, word, lead).ratio() >= _LEAD_SIMILARITY for lead in lead_words)


def strip_wake_phrase(text: str, phrase: str) -> str | None:
    """Return the text after a wake phrase at the start of `text`.

    Return `None` when `text` does not open with the phrase. The match is on
    the last word of the phrase (the keyword, "atlas" for "hey atlas"). It
    looks at the first `len(phrase words)` positions, so one lead-in word is
    allowed. At each position it tries one word, then two words joined
    ("at last" for "atlas").

    A lead-in word must look like the phrase's own lead word ("hey": "they",
    "he", "hay" pass) or be one of a few fixed fillers ("a", "the", "oh",
    "ok", "okay", "hi"). "You always say AM in the morning." fails because
    "you" is not a lead word, even though "always" is close to "atlas".

    A two-word join counts only as a split keyword ("at last", "at less").
    It is skipped when the second word alone already matches the keyword,
    because then the first word is a lead-in and the lead-in rule judges it.

    Keyword similarity that passes: "atlas" 1.0, "atlast" 0.91, "aatlas"
    0.91, "heyatlas" 0.77, "atless" 0.73. Keyword similarity that fails:
    "atlanta" 0.67, "whats" 0.60, "thats" 0.60, "alice" 0.40.

    Accepted misses: a sentence that opens with "the atlas", "at least", or
    "alas" passes, because its keyword similarity is high. This check
    narrows an ambient trigger. It is not a safety boundary: every action
    still goes through `mcp.atlas_mcp.safety.allow_call`.
    """
    phrase_words = _PUNCT_RE.sub("", phrase.lower()).split()
    raw = text.split()
    if not phrase_words or not raw:
        return None

    keyword = phrase_words[-1]
    lead_words = phrase_words[:-1]
    norm = [_PUNCT_RE.sub("", word.lower()) for word in raw]
    for start in range(len(phrase_words)):
        if not all(_may_lead(word, lead_words) for word in norm[:start]):
            continue
        for width in (1, 2):
            end = start + width
            if end > len(norm):
                continue
            if width == 2 and _similar(norm[end - 1], keyword):
                continue
            candidate = "".join(norm[start:end])
            if not candidate:
                continue
            if _similar(candidate, keyword):
                return " ".join(raw[end:]).lstrip(" ,.;:!?-")
    return None


def is_wake_without_command(text: str, phrase: str, *, verify: bool) -> bool:
    """True when `text` is the wake phrase and no command follows it.

    With `verify` on, text that does not open with the phrase is not a wake
    hit, so it is never "wake without command". With `verify` off, the text
    is judged as it is.
    """
    if not text or not phrase:
        return False
    command = strip_wake_phrase(text, phrase)
    if command is None:
        if verify:
            return False
        command = text
    return not command or is_wake_only(command, phrase)


@dataclass
class WakeHold:
    """A `hold_final` predicate for `XaiStt.stream`.

    It holds a final that is only the wake phrase, so the command is read on
    the same socket. `heard_text` puts the held phrase back in front of the
    command, so wake verification still sees it.
    """

    phrase: str
    verify: bool
    held: list[str] = field(default_factory=list)

    def __call__(self, text: str) -> bool:
        if is_wake_without_command(text, self.phrase, verify=self.verify):
            self.held.append(text)
            return True
        # An empty final after the wake phrase is still no command.
        if not text.strip() and self._first_held():
            self.held.append(text)
            return True
        return False

    def _first_held(self) -> str:
        return next((text for text in self.held if text.strip()), "")

    def heard_text(self, final_text: str) -> str:
        first = self._first_held()
        if not first or strip_wake_phrase(final_text, self.phrase) is not None:
            return final_text
        return f"{first} {final_text}".strip()
