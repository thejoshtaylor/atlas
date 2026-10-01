"""A live-feed publisher that stays quiet until a window hears speech.

An answer window (VOICE-21, Phase 13) opens after every edge answer, and most
of them hear nothing. A card on the live page for each silent window would
fill the feed with empty turns. `QuietStartPublisher` stands in for the
`ObserverRegistry` for such a turn. It holds the `turn.started` event back and
drops content-free events (an empty `transcript.final`, `turn.timing`,
`reply.interrupted`). When the first content arrives, it publishes the held
start and then that event. After that, it forwards everything.

It is synchronous and never awaits, the same rule as `ObserverRegistry.publish`.
`ObserverPublishingSource` calls only `publish`.
"""

from __future__ import annotations

from typing import Any

_TRANSCRIPT_TYPES = frozenset({"transcript.partial", "transcript.final"})


class QuietStartPublisher:
    def __init__(self, registry: Any, started_event: dict[str, Any]) -> None:
        self._registry = registry
        self._started_event = started_event
        self._started = False

    @property
    def started(self) -> bool:
        """True once the held `turn.started` event has been published."""
        return self._started

    def publish(self, event: dict[str, Any]) -> None:
        if not self._started:
            if not _is_content(event):
                return
            self._started = True
            self._registry.publish(self._started_event)
        self._registry.publish(event)


def _is_content(event: dict[str, Any]) -> bool:
    kind = event.get("type")
    if kind == "reply.text":
        return True
    if kind in _TRANSCRIPT_TYPES:
        text = event.get("text")
        return isinstance(text, str) and bool(text.strip())
    return False
