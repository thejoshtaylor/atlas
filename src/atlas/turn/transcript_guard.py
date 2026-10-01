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

`missing_target_question` is a second guard, for a command that is a verb
with no device. Two live turns on the deployed edge (2026-10-01) showed why.
The operator paused after "Turn off." and, in another turn, after "Turn on
the.". Each fragment reached the brain, the brain guessed a target, and it
switched real devices. The check is in code, and not only in a prompt, for
the same reason as rule 2 in `atlas_mcp/safety.py`: a prompt is advice, and
a television can speak any sentence. The prompt line in `app.py` is only a
second, weaker layer.
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
# 0.75, not 0.6: "alarm" (0.6) and "alarms" (0.73) sound enough like
# "atlas" at 0.6 that "cancel my alarm" was dropped as wake-only.
_KEYWORD_SIMILARITY = 0.75
_MAX_WAKE_ONLY_TOKENS = 3
# A wake-only echo is a misheard lead-in plus the keyword, so at most two
# content words. "cancel my alarm" has three and is always a command.
_MAX_WAKE_ONLY_CONTENT = 2
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


# The verbs of a control command. Same three as `local_intent`. "power" is
# not here because it is often a noun ("is the power on").
_CONTROL_VERBS: frozenset[str] = frozenset({"turn", "switch", "shut"})
_PARTICLES: frozenset[str] = frozenset({"on", "off"})
# Words that add no device to a command.
_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "please",
        "can",
        "could",
        "would",
        "will",
        "you",
        "just",
        "now",
        "uh",
        "um",
        "hey",
        "ok",
        "okay",
        "back",
        "again",
    }
)
_FUNCTION_WORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "to", "and", "of", "or", "my", "your", "our", "for", "with", "in", "at", "into"}
)
# A pronoun names a device only when something earlier in the talk does.
_PRONOUNS: frozenset[str] = frozenset({"it", "that", "this", "them", "those", "these", "one"})
# A last word that cannot end a sentence: the speaker has more to say.
_DANGLING_WORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "to", "and", "of", "or", "my", "your", "for", "with", "in", "at", "into"}
)
_ELLIPSES = ("...", "\u2026")


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

    if len(tokens) > _MAX_WAKE_ONLY_TOKENS or len(content) > _MAX_WAKE_ONLY_CONTENT:
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


def _command_remainder(text: str) -> tuple[str, str, list[str]] | None:
    """The verb, the particle and the words left over, or None.

    None means `text` is not a control command with an on/off particle.
    "shut" with no particle reads as "off". The words left over are what is
    still there once the verb, the particle, filler and function words go.
    """
    tokens = _tokens(text)
    verb_index = next((i for i, token in enumerate(tokens) if token in _CONTROL_VERBS), None)
    if verb_index is None:
        return None
    verb = tokens[verb_index]
    # The accepted typo: "turn of the" for "turn off the". Only right after a verb.
    if verb_index + 1 < len(tokens) and tokens[verb_index + 1] == "of":
        tokens[verb_index + 1] = "off"
    particle_index = next((i for i, token in enumerate(tokens) if token in _PARTICLES), None)
    if particle_index is None:
        if verb != "shut":
            return None
        particle = "off"
    else:
        particle = tokens[particle_index]
    rest = [
        token
        for i, token in enumerate(tokens)
        if i != verb_index
        and i != particle_index
        and token not in _FILLER_WORDS
        and token not in _FUNCTION_WORDS
    ]
    return verb, particle, rest


def missing_target_question(text: str, *, has_referent: bool) -> str | None:
    """The question to ask when `text` is a control command that names no
    device, or None.

    A bare verb ("Turn off.", "Turn on the.") always gets the question.
    A command with only a pronoun ("turn it on") gets it when nothing
    earlier in the talk gives the pronoun a meaning (`has_referent` False).
    Every other text returns None, "everything" included: it is a target.
    """
    parts = _command_remainder(text)
    if parts is None:
        return None
    verb, particle, rest = parts
    if rest and not (has_referent is False and all(token in _PRONOUNS for token in rest)):
        return None
    return f"{verb} {particle} what?"


def looks_unfinished(text: str) -> bool:
    """True when `text` reads as a command the speaker has not finished.

    The cases are an ellipsis at the end, a last word that cannot end a
    sentence ("turn on the"), and a bare control verb ("Turn off."). A
    pronoun counts as a target here, so "turn it off" is finished.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(_ELLIPSES):
        return True
    tokens = _tokens(stripped)
    if tokens and tokens[-1] in _DANGLING_WORDS:
        return True
    return missing_target_question(stripped, has_referent=True) is not None
