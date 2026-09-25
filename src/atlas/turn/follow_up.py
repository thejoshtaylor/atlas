"""The open-microphone follow-up channel a source may attach (plan 09-06)
and this plan's `dispatch_handoff` requests through.

`FollowUpChannel` is a per-source, long-lived object -- `SourceRunner`
(plan 09-06) attaches one to a source and, after speaking a confirmation
readback, opens the microphone once for exactly the answer this channel's
`requested` field names. Until then, every proposal on every source finds
no channel attached at all (`getattr(source, "follow_up", None)` in
`turn/controller.py`), which is what makes
`atlas.turn.pending_action.CONFIRMATION_UNAVAILABLE_REPLY` the honest
answer everywhere in this plan and the next: nothing here yet actually
listens for a confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# D-06: a source that cannot hear an answer must never be left holding a
# pending action open forever. `chain_depth` above this many chained
# follow-ups (a clarification answered with another clarification, and so
# on) ends the exchange with `FOLLOW_UP_LIMIT_REPLY` rather than looping.
MAX_CHAINED_FOLLOW_UPS = 3


@dataclass(frozen=True)
class FollowUpRequest:
    """One turn's own request for the next thing this source hears to be
    treated as an answer to a specific question, not a fresh command.

    `kind` distinguishes a yes/no confirmation from a which-one
    clarification -- the two shapes `dispatch_handoff` ever builds one for.
    `chain_depth` is this request's own position in a chain of follow-ups
    (1 for the first ask); `pending_action_id` is set only for a
    `"confirmation"` request, naming the exact row a "yes" resolves.
    `playback_ends_at` is reserved for a later plan's barge-in-aware
    open-microphone timing and is `None` here.
    """

    kind: Literal["confirmation", "clarification"]
    chain_depth: int
    original_transcript: str
    question: str
    pending_action_id: "int | None" = None
    playback_ends_at: "float | None" = None


@dataclass
class FollowUpChannel:
    """A source's own mutable follow-up state -- `incoming` is the request
    the *previous* turn left for this one to answer (set by plan 09-06's
    open-microphone listener before the next turn starts); `requested` is
    the request *this* turn is leaving for the next one, written by
    `request()`.

    Deliberately not a queue: at most one follow-up is ever pending on a
    source at a time (D-10's own "one action at a time" discipline extends
    to the confirmation asking for it).
    """

    incoming: "FollowUpRequest | None" = None
    requested: "FollowUpRequest | None" = None

    def request(self, req: FollowUpRequest) -> None:
        self.requested = req
