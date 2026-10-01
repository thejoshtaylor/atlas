"""The server's own confirmation of a wake, one time per turn (Phase 15, D-05).

The wake detector on the camera or on the Pi can fire on a television. The
Mac panel must not open on that. A wake counts as confirmed only when the
transcript the server itself got from speech-to-text opens with the wake
phrase, judged by `strip_wake_phrase`. `WakeConfirmation` watches the STT
events of one turn, as each one arrives, and awaits `emit` the first time
that test passes.

A bare "hey atlas" confirms: `strip_wake_phrase` returns an empty string for
it, and an empty string is a match. "hey" alone does not confirm, and neither
does "You always say AM in the morning."

This check narrows the television risk. It is not a safety boundary:
`strip_wake_phrase` has accepted misses (a sentence that opens with "the
atlas" or "at least" passes), and every action still goes through
`mcp.atlas_mcp.safety.allow_call`.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from atlas.turn.wake_echo import strip_wake_phrase


class WakeConfirmation:
    """Idempotent: `observe` awaits `emit` at most one time."""

    def __init__(self, phrase: str, emit: Callable[[], Awaitable[None]]) -> None:
        self._phrase = phrase
        self._emit = emit
        self._confirmed = False

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    async def observe(self, text: str) -> None:
        """Look at one STT event's text. Confirm on the first text that
        opens with the wake phrase; ignore every text after that."""
        if self._confirmed:
            return
        if strip_wake_phrase(text, self._phrase) is None:
            return
        # Set before the await, so a second event that arrives while `emit`
        # is still running cannot confirm a second time.
        self._confirmed = True
        await self._emit()
