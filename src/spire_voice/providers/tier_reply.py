"""The structured tier reply, and the closed-enum filler CMD-07 requires.

D-03 makes a tier's reply a validated object rather than free prose: the
answer, a confidence flag, a needs-a-tool flag, and a holding line. D-06
makes that holding line a closed enum rather than a string, so an
outcome-asserting filler ("turning that off") is unrepresentable rather
than merely discouraged -- CMD-07's "impossible by construction" is
literally true only because Pydantic rejects any `filler` value outside
`FillerPhrase` before it can reach `_speak`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class FillerPhrase(str, Enum):
    """The closed set of holding lines a tier may select while it is not confident.

    Every member was chosen so that no word in it can be heard as a
    confirmation that something happened -- see `OUTCOME_ASSERTING_WORDS`
    below, and `tests/test_filler_enum.py`, which proves it.
    """

    LET_ME_CHECK = "let_me_check"
    ONE_MOMENT = "one_moment"
    STILL_LOOKING = "still_looking"
    GIVE_ME_A_SECOND = "give_me_a_second"


FILLER_TEXT: dict[FillerPhrase, str] = {
    FillerPhrase.LET_ME_CHECK: "let me check",
    FillerPhrase.ONE_MOMENT: "one moment",
    FillerPhrase.STILL_LOOKING: "still looking",
    FillerPhrase.GIVE_ME_A_SECOND: "give me a second",
}
"""The exact sentence spoken for each member.

This mapping is the single source the startup precache (plan 01.1-04)
synthesizes from, so the enum and the cache key cannot drift apart --
`tests/test_filler_enum.py` asserts `set(FILLER_TEXT) == set(FillerPhrase)`.
"""

# D-08's filler deadline can fire before any tier has replied at all, so
# there is sometimes no tier-selected phrase to fall back to yet. Resolution:
# the deadline plays the phrase selected by the first non-confident reply
# received so far, and this default when none has arrived. Both are members
# of the closed set, so CMD-07 holds either way.
DEFAULT_FILLER = FillerPhrase.LET_ME_CHECK


OUTCOME_ASSERTING_WORDS: frozenset[str] = frozenset(
    {
        # switching verbs -- past, present, and gerund forms
        "turn", "turns", "turned", "turning",
        "switch", "switches", "switched", "switching",
        "toggle", "toggles", "toggled", "toggling",
        "set", "sets", "setting",
        "start", "starts", "started", "starting",
        "stop", "stops", "stopped", "stopping",
        "open", "opens", "opened", "opening",
        "close", "closes", "closed", "closing",
        "lock", "locks", "locked", "locking",
        "unlock", "unlocks", "unlocked", "unlocking",
        "enable", "enables", "enabled", "enabling",
        "disable", "disables", "disabled", "disabling",
        "activate", "activates", "activated", "activating",
        "deactivate", "deactivates", "deactivated", "deactivating",
        "raise", "raises", "raised", "raising",
        "lower", "lowers", "lowered", "lowering",
        "dim", "dims", "dimmed", "dimming",
        "adjust", "adjusts", "adjusted", "adjusting",
        "change", "changes", "changed", "changing",
        # completion adjectives
        "done", "complete", "completed", "finished", "finishing",
        # affirmations
        "yes", "yep", "sure", "ok", "okay",
        # the two state words a switch reports
        "on", "off",
    }
)
"""Words that would make a holding line sound like a confirmation.

Matched on word boundaries, not substrings -- a substring check would
reject "one moment" on the two letters inside "one", which is exactly the
kind of false positive that gets a safety check quietly deleted.
"""


class TierReply(BaseModel):
    """One tier's structured reply -- validated, never free prose (D-03)."""

    answer: str = Field(description="What to say aloud if `confident` is true.")
    confident: bool = Field(
        description="True only when `answer` fully answers the request with no tool call needed."
    )
    needs_tool: bool = Field(description="True when fulfilling the request requires a tool call.")
    filler: FillerPhrase = Field(
        description="What to say while a higher tier works, when `confident` is false."
    )

    @model_validator(mode="after")
    def _confident_reply_must_have_an_answer(self) -> "TierReply":
        if self.confident and not self.answer.strip():
            raise ValueError(
                "a confident TierReply must carry a non-empty answer -- an empty confident "
                "reply is the empty-reply case controller.py already handles separately, and "
                "it must not arrive disguised as a confident answer"
            )
        return self

    @model_validator(mode="after")
    def _confident_and_needs_tool_are_mutually_exclusive(self) -> "TierReply":
        # WR-02: the field docstrings already claim these are mutually
        # exclusive outcomes -- `confident` means the answer is complete with
        # no tool call needed, `needs_tool` means one is required. Nothing
        # enforced that until now, and `race_tiers` only ever reads
        # `.confident`, so a reply carrying both flags `True` would win the
        # race exactly like any ordinary confident reply. That is the direct
        # mechanism feeding CR-01: a triage tier that has no tool schema and
        # no code-level penalty for asserting confidence on an action
        # request. Same impossible-by-construction treatment already given
        # to the filler enum in this file.
        if self.confident and self.needs_tool:
            raise ValueError(
                "a TierReply cannot be both confident and needs_tool -- confident means no "
                "tool call is needed, needs_tool means one is required; a reply can claim at "
                "most one of the two"
            )
        return self
