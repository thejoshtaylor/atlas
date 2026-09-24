"""Is a final transcript nothing but the wake word, or nothing but filler?

Transcribed text is untrusted (CLAUDE.md, Security): a television, a
mishearing, or a repeated wake phrase can put any words in front of the
brain. Two real turns showed the shape of the problem this guard closes.
A repeated wake phrase (the old "hey spire") came back from STT as "Space Spire", and the brain
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
lead-in. Under the old phrase, "Hey spire" came back as "stay spire", and
"stay" has a `difflib` ratio of only about 0.22 against the keyword. The keyword itself
("atlas") is the one token STT renders reliably, so it is the one position
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

_DEFAULT_KEYWORD = "atlas"
_KEYWORD_SIMILARITY = 0.6
_MAX_WAKE_ONLY_TOKENS = 3
_NON_WORD_RE = re.compile(r"[^\w\s]")

# 260924-4it: words that open a genuine information request -- the words
# themselves ("what", "how", "check", ...), not question marks alone, since
# a spoken command often carries no punctuation at all. Apostrophe-free
# ("whats", not "what's") because `_tokens` drops the apostrophe.
_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "whats",
        "how",
        "hows",
        "when",
        "where",
        "wheres",
        "which",
        "who",
        "whos",
        "whose",
        "why",
        "tell",
        "check",
        "read",
    }
)

# An auxiliary verb that opens a yes/no question ("is the door locked").
# "can", "could", "would", and "will" are deliberately absent: they open a
# polite command ("could you turn off the lamp"), not a question.
_AUX_WORDS: frozenset[str] = frozenset(
    {
        "is",
        "are",
        "am",
        "was",
        "were",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "dont",
        "doesnt",
        "didnt",
        "hasnt",
        "havent",
    }
)

# Words that precede a real sentence without carrying meaning of their own
# -- dropped from the front before checking for a leading auxiliary.
_LEAD_IN_WORDS: frozenset[str] = frozenset(
    {"hey", "hi", "hay", "ok", "okay", "so", "um", "uh", "oh", "please", "atlas"}
)


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
    back to the default keyword, "atlas".
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


def asks_for_information(text: str) -> bool:
    """True when `text` may be asking to hear something, not just asking
    for an action.

    The done shortcut in `turn/controller.py`'s `_run_tool_rounds` calls
    this predicate. True means the operator may be waiting to hear spoken
    information, so the model must phrase the reply -- False lets the
    shortcut speak the canned "done" reply after a plain action succeeds,
    with no second model round.

    The predicate errs toward True on purpose. A false True costs one
    model round of latency. A false False drops information the operator
    asked for.

    Accepted miss: a custom wake word that is not in `_LEAD_IN_WORDS`,
    followed by an unpunctuated yes/no question, is not caught. xAI STT
    usually adds a question mark, and rule 1 below then catches it anyway.
    """
    if "?" in text:
        return True

    tokens = _tokens(text)
    if any(token in _QUESTION_WORDS for token in tokens):
        return True

    lead = 0
    while lead < len(tokens) and tokens[lead] in _LEAD_IN_WORDS:
        lead += 1
    if lead < len(tokens) and tokens[lead] in _AUX_WORDS:
        return True

    for index, token in enumerate(tokens):
        if index > 0 and tokens[index - 1] in ("and", "also", "then") and token in _AUX_WORDS:
            return True

    return False
