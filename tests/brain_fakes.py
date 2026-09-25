"""`RecordingFakeBrain`: a scripted language-model double that records
every call's own `messages` and `tools` -- lets a test prove which brain
calls in a turn carried `tools=None` (a quarantine round, T-09-42) versus
the real tool schema (the top tier's own tool round). The direct sibling
of `tests/conftest.py::FakeBrain`, extended with per-call recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from atlas.providers.base import BrainReply


@dataclass(frozen=True)
class RecordedCall:
    """One `chat()` call, exactly as it was made -- `messages` is a copy
    (the caller's own list may be mutated afterward, e.g. `run_turn`
    appending further messages to the same list object) and `tools` is
    whatever the caller passed, `None` included."""

    messages: "list[dict[str, Any]]"
    tools: "list[dict[str, Any]] | None"


class RecordingFakeBrain:
    """Each call to `chat()` consumes and returns the next scripted
    `BrainReply`, in order -- the same "calling `chat()` more times than
    scripted raises" contract `tests/conftest.py::FakeBrain` already
    establishes."""

    def __init__(self, replies: Sequence[BrainReply] = ()) -> None:
        self._replies = list(replies)
        self.calls: "list[RecordedCall]" = []

    async def chat(self, messages: "list[dict[str, Any]]", tools: "list[dict[str, Any]] | None" = None) -> BrainReply:
        call_index = len(self.calls)
        self.calls.append(RecordedCall(messages=list(messages), tools=tools))
        if call_index >= len(self._replies):
            raise AssertionError("RecordingFakeBrain.chat called more times than scripted")
        return self._replies[call_index]

    @property
    def call_count(self) -> int:
        return len(self.calls)
