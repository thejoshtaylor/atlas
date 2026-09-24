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
"""

from __future__ import annotations

import difflib
import re

_PUNCT_RE = re.compile(r"[.,!?;:'\"()\[\]{}\-_/\\]")

# Below this, a transcript reads as a real command rather than an echo of
# the wake phrase -- see the module docstring for how this value was picked.
_SIMILARITY_THRESHOLD = 0.55


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
