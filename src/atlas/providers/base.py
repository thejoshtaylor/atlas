"""Thin provider protocols for speech-to-text, the language model, and text-to-speech.

Three separate protocols, not one fat provider object: Phase 7 swaps each
independently -- a local speech-to-text model could replace xAI's while the
language model call stays remote -- and that swap stays cheap only if
nothing downstream depends on a single combined interface. Each error type
is raised, not returned, mirroring `atlas_mcp.safety.Denied`'s doctrine: a
caller cannot silently continue past a failed provider call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol


@dataclass
class PartialTranscript:
    """An in-progress transcript the page renders live as the operator speaks."""

    text: str


@dataclass
class FinalTranscript:
    """The finished transcript. `text == ""` is the empty-utterance case (VOICE-08)."""

    text: str


@dataclass
class ToolCall:
    """One tool call the language model asked for, already parsed from JSON."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrainReply:
    """One `chat()` result: zero or more tool calls, and/or reply text."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""


class SttError(Exception):
    """Raised on the speech-to-text provider's `error` event."""


class BrainError(Exception):
    """Raised when the language model provider call fails."""


class TtsError(Exception):
    """Raised on the text-to-speech provider's `error` event."""


class SttProvider(Protocol):
    def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[PartialTranscript | FinalTranscript]:
        """Stream transcript events for one turn's worth of audio frames."""
        ...


class BrainProvider(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> BrainReply:
        """Run one chat-completion round, possibly returning tool calls."""
        ...


class TtsProvider(Protocol):
    def synthesize(
        self, text_deltas: AsyncIterator[str], sink: "Any | None" = None
    ) -> AsyncIterator[bytes]:
        """Stream audio chunks as reply text arrives.

        260922-cts: `sink`, when given, is the playback format (codec,
        sample rate) to render against -- every real implementation
        (`BatchTtsAdapter`, `CachedTts`) already reads it; `None` (the
        default) is today's browser-PCM request, unchanged for a caller
        that predates this fix.
        """
        ...
