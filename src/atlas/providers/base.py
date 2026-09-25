"""Thin provider protocols for speech-to-text, the language model, and text-to-speech.

Three separate protocols, not one fat provider object: Phase 7 swaps each
independently -- a local speech-to-text model could replace xAI's while the
language model call stays remote -- and that swap stays cheap only if
nothing downstream depends on a single combined interface. Each error type
is raised, not returned, mirroring `atlas_mcp.safety.Denied`'s doctrine: a
caller cannot silently continue past a failed provider call.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

import httpx

from atlas.transports.base import SourceFormat

# 260924-4iv (item d): httpx's own default `keepalive_expiry` is 5 s. A
# wake-time warm call opens a connection this many seconds before a
# typical camera turn (about 5 s, wake to final transcript) even reaches
# the provider it warmed -- at httpx's default, the connection is already
# closed again by the time the turn would reuse it, making the warm call
# pure waste. 60 s covers a full turn's own latency budget with room to
# spare, and keeps a second, closely-following turn on the same pooled
# connection too.
PROVIDER_KEEPALIVE_EXPIRY_S = 60.0


def provider_http_limits() -> httpx.Limits:
    """The one `httpx.Limits` every warmed provider pool shares (260924-4iv,
    item d). `max_connections`/`max_keepalive_connections` are given
    explicitly because `httpx.Limits`' own defaults for both are
    unlimited -- an explicit, bounded pool here rather than inheriting
    that default.
    """
    return httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=PROVIDER_KEEPALIVE_EXPIRY_S)


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
    def stream(
        self,
        frames: AsyncIterator[bytes],
        source_format: SourceFormat,
        *,
        finalize: "asyncio.Event | None" = None,
    ) -> AsyncIterator[PartialTranscript | FinalTranscript]:
        """Stream transcript events for one turn's worth of audio frames.

        `source_format` is restated here to match every real implementation
        (`XaiStt.stream`, `FasterWhisperStt.stream`), which already take it --
        the Protocol had lagged one argument behind them.

        `finalize` (D-12), when given and set, ends the current utterance at
        once instead of waiting for this provider's own endpointing or for
        `frames` to exhaust. `None` (the default, and every caller that
        predates this plan) is today's behavior, unchanged. The event is per
        *stream call*, never state held on the provider instance: one
        provider instance serves every source (SRC-03), and a method on the
        instance instead (`async def finalize(self)`, 10-PATTERNS.md's
        original sketch) would let one source's VAD end finalize a different
        source's concurrent turn.
        """
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
