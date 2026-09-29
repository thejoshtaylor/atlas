"""What `TurnRun` hands `run_turn` about the turn's place in a group (Phase 12).

One `TurnContext` per turn, built by `sources/turn_run.py` and read by
`run_turn` off the source it is given (`getattr(source, "turn_context", None)`).
A turn that runs on a serial source has none.

This module imports nothing from `atlas.sources`: `turn/controller.py` imports
it, and the other direction would be a cycle. The fields that come from other
modules (the speaker span, the reply handle, the claims registry) are typed as
`Any` for the same reason.

The speaker label is untrusted (Phase 11 D-14, D-15). It is stored here for the
reply handle, which composes reply text in code after the model has run. It
never enters the `turn.group` event, and no claim key uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

TURN_GROUP_EVENT = "turn.group"


@dataclass(eq=False)
class TurnContext:
    """One turn's identity in its group.

    `turn_key` is unique per run on one source (`<source name>:<n>`) and is the
    only key a claim may use. `group_id` is shared by every turn that overlaps.
    `order_frame` is the turn's first frame index, which sets speech order in a
    merged reply. A follow-up turn keeps the asking turn's `turn_key`,
    `group_id`, `claims`, and `reply_group`, and sets `follow_up=True` and
    `answer_only_from`.
    """

    turn_key: str
    group_id: str
    order_frame: int
    split_part: bool = False
    follow_up: bool = False
    speaker_span: Any | None = None
    reply_group: Any | None = None
    claims: Any | None = None
    answer_only_from: str | None = None
    replay_until: Callable[[int], list[bytes] | None] | None = None
    speaker_id: str | None = None
    speaker_label: str | None = None
    friendly_names: dict[str, str] = field(default_factory=dict)
    _record_event: Callable[[dict[str, Any]], None] | None = field(default=None, init=False, repr=False)

    def note_speaker(self, speaker_id: str | None, speaker_label: str | None) -> None:
        """Record who this turn's speaker is, and tell the reply handle the
        label so the merged reply can address the speaker by name."""
        self.speaker_id = speaker_id
        self.speaker_label = speaker_label
        set_label = getattr(self.reply_group, "set_label", None)
        if set_label is not None:
            set_label(speaker_label)

    def note_friendly_names(self, names: dict[str, str]) -> None:
        """Replace the entity id to friendly name map this turn's tool calls
        have seen, for a refused-claim reply."""
        self.friendly_names = dict(names)

    def bind_recorder(self, record_event: Callable[[dict[str, Any]], None]) -> None:
        """Store the session recorder's `record_event` and write the one
        `turn.group` event. The event carries no speaker label."""
        self._record_event = record_event
        record_event(
            {
                "type": TURN_GROUP_EVENT,
                "group_id": self.group_id,
                "turn_key": self.turn_key,
                "order_frame": self.order_frame,
                "split_part": self.split_part,
                "follow_up": self.follow_up,
            }
        )

    def record_event(self, event: dict[str, Any]) -> None:
        """Forward `event` to the bound recorder. A no-op before binding."""
        if self._record_event is not None:
            self._record_event(event)
