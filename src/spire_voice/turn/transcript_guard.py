"""Is a final transcript nothing but the wake word, or nothing but filler?

Transcribed text is untrusted (CLAUDE.md, Security): a television, a
mishearing, or a repeated wake phrase can put any words in front of the
brain. Two real turns showed the shape of the problem this guard closes.
A repeated wake phrase came back from STT as "Space Spire", and the brain
called two tool actions on it. A second turn's whole transcript was a bare
"It's", and the brain read state and answered. Neither turn said a command;
both should have ended in silence-with-a-nudge, not a tool call.

`is_no_command` is the pure predicate `turn/controller.py` checks before a
final transcript reaches a macro, the local on/off matcher, or the tier
race. It drops two shapes of transcript:

- Filler-only text: nothing left once stopwords like "uh", "um", "it's",
  "okay" are stripped out.
- Wake-only text: three words or fewer, whose last word sounds like the
  configured wake keyword.

The rule checks only the *last* word against the keyword, not every word,
because STT does not mishear the wake word consistently -- it mishears the
lead-in. "Hey spire" can come back as "stay spire", and "stay" has a
`difflib` ratio of only about 0.22 against "spire". The keyword itself
("spire") is the one token STT renders reliably, so it is the one position
this rule can trust.

This has one accepted false positive: a real command of three words or
fewer whose last word happens to sound like the keyword is dropped too (a
name that rhymes with the wake word, said as a short reply). The operator
hears the same "sorry, i didn't catch that" reply and can say it again.
`turn_outcome = "no_command"` makes this visible in `timing.json`, so a
real corpus can tell the difference between a mishearing and a mistuned
threshold later.
"""

from __future__ import annotations

import difflib
import re

# The filler set an operator's own speech leaves behind after the wake
# word, plus "hay" -- a mishearing of the wake word's own lead-in ("hey"),
# not filler an operator would say on purpose.
_NOISE_WORDS: frozenset[str] = frozenset(
    {
        "it's",
        "its",
        "it",
        "uh",
        "um",
        "hmm",
        "the",
        "a",
        "oh",
        "okay",
        "ok",
        "yeah",
        "so",
        "and",
        "hey",
        "hi",
        "hay",
    }
)

_DEFAULT_KEYWORD = "spire"
_KEYWORD_SIMILARITY = 0.6
_MAX_WAKE_ONLY_TOKENS = 3
_NON_WORD_RE = re.compile(r"[^\w\s]")


def _tokens(text: str) -> list[str]:
    """Lowercase, drop every punctuation mark (including a curly
    apostrophe), and split on whitespace."""
    return _NON_WORD_RE.sub("", text.casefold()).split()


def is_no_command(text: str, wake_phrase: str | None = None) -> bool:
    """True when `text` is filler-only, or three words or fewer ending in
    something that sounds like the wake keyword -- never a real command.

    `wake_phrase` is the phrase `run_turn` received for this turn
    (`config.wake.phrase` on the camera path). Its last word is the
    keyword this rule matches against; a blank or missing phrase falls
    back to the default keyword, "spire".
    """
    tokens = _tokens(text)
    content = [token for token in tokens if token not in _NOISE_WORDS]

    if not content:
        return True

    if len(tokens) > _MAX_WAKE_ONLY_TOKENS:
        return False

    phrase_tokens = _tokens(wake_phrase or "")
    keyword = phrase_tokens[-1] if phrase_tokens else _DEFAULT_KEYWORD

    return difflib.SequenceMatcher(None, content[-1], keyword).ratio() >= _KEYWORD_SIMILARITY
