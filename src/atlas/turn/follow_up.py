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
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    # Annotation-only: `turn/controller.py` imports this module at load
    # time (its own module docstring), so an import of `SpeechResult` here
    # must never run at module load -- only under a type checker's eyes,
    # the same `TYPE_CHECKING` discipline `turn/pending_action.py` already
    # establishes for `HandoffContext`.
    from atlas.providers.tts_xai import SinkFormat
    from atlas.turn.controller import SpeechResult

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
    # Plan 09-07 (D-06): every earlier exchange this follow-up chain
    # already carries, oldest first -- empty for the first link in a
    # chain. `turn/controller.py`'s own continuation-message helper
    # prepends this ahead of `original_transcript`/`question` themselves,
    # so a third link in a chain (a clarification answered with another
    # clarification, or an amendment on top of an amendment) still shows
    # a tier the whole conversation, not just the last hop.
    prior_messages: tuple[dict, ...] = ()
    # A-CR-02: True when the turn that asked this question was itself
    # proposal-restricted (it continued an `amended` confirmation reply,
    # or answered a follow-up that already carried this flag). The turn
    # that answers this request runs with the same restriction, and every
    # follow-up that turn requests carries the flag on. The restriction
    # belongs to the chain, not to one turn, so no later link of the chain
    # can reach a tool that is not a calendar proposal.
    proposals_only: bool = False


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
    # Plan 09-06: the instant, in `_speak`'s own `time.monotonic()` domain,
    # after which the next turn's frames may reach its own STT -- set by
    # `SourceRunner._run_follow_ups` from `estimate_playback_end` below plus
    # the configured echo tail, never by `run_turn` itself. `window_s` is
    # this same request's own configured window length
    # (`FollowUpConfig.window_s`), so `_drain_to_final_transcript`'s
    # `onset_deadline` can be computed as `window_opens_at + window_s` at
    # the one call site that needs it, with no second source of the
    # number.
    window_opens_at: "float | None" = None
    window_s: "float | None" = None

    def request(self, req: FollowUpRequest) -> None:
        self.requested = req


def estimate_playback_end(result: "SpeechResult", sink: "SinkFormat | None") -> "float | None":
    """When a spoken utterance's own audio actually finished arriving in
    the room, in the same `time.monotonic()` domain `_speak` already
    stamps `result.first_write_at`/`result.last_write_at` in.

    `sink is None` (every browser source today, `turn/controller.py`'s own
    pre-260922-cts default) means there is no format to compute playback
    duration from -- `result.last_write_at` (the moment the last chunk was
    *written*) is the best available estimate, and it is what every
    caller predating this plan implicitly assumed playback finishing meant.

    With a sink, the write-finish time can understate how long the room
    actually keeps hearing audio -- FIFO buffering plays the last written
    bytes out over real time, not instantaneously -- so this instead
    estimates the moment the *last byte written* would finish playing:
    `first_write_at + bytes_sent / (sample_rate * bytes_per_sample)`,
    never earlier than `last_write_at` itself (a burst of writes that all
    land faster than the audio duration they encode must not estimate a
    playback end in the past). A-law and mu-law are one byte per sample;
    `pcm` (16-bit signed) is two.
    """
    if sink is None:
        return result.last_write_at
    if result.first_write_at is None or result.last_write_at is None:
        return result.last_write_at
    bytes_per_sample = 2 if sink.codec == "pcm" else 1
    playback_duration_s = result.bytes_sent / (sink.sample_rate * bytes_per_sample)
    return max(result.last_write_at, result.first_write_at + playback_duration_s)
