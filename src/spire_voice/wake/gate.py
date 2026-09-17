"""The wake-hit gate: a narrowing control on an ambient trigger, never the
safety boundary. Every action a turn reaches still passes the same
`mcp.spire_mcp.safety.allow_call` check a browser turn does, unconditionally
-- no decision from this module may be read as authorization for anything.

This module is pure: no I/O, no imports from `turn/` or `providers/`. It
decides; it never acts. Media-player state -- the one thing this gate needs
that lives outside it -- arrives as an injected callable, so a house that
ships `mute_when_playing` empty (D-02) never reaches for one at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

BlockReason = Literal["below_threshold", "refractory", "media_playing"]


@dataclass(frozen=True)
class GateDecision:
    """Whether a wake hit may start a turn, and why not when it may not.

    A block always carries its reason (D-03): the reason is the data that
    tunes the gate, and a block with no reason is indistinguishable from
    the silent drop CONTEXT.md rejects by name.
    """

    allowed: bool
    reason: BlockReason | None = None

    @classmethod
    def allow(cls) -> "GateDecision":
        return cls(allowed=True)

    @classmethod
    def block(cls, reason: BlockReason) -> "GateDecision":
        return cls(allowed=False, reason=reason)


class WakeGate:
    """May this wake hit start a turn -- and if not, why not.

    Three reasons block a hit, checked in this order: the score never
    reached the threshold, the hit fell inside the refractory window, or a
    configured media player is currently playing. The first that applies
    wins; no hit is ever blocked for more than one stated reason.
    """

    def __init__(
        self,
        threshold: float,
        refractory_s: float,
        mute_when_playing: tuple[str, ...],
        is_media_playing: Callable[[tuple[str, ...]], bool] | None = None,
    ) -> None:
        self._threshold = threshold
        self._refractory_s = refractory_s
        self._mute_when_playing = mute_when_playing
        self._is_media_playing = is_media_playing or (lambda _players: False)

    def evaluate(self, score: float, now: float, last_hit_at: float | None) -> GateDecision:
        """Decide one hit. `score` is compared as the float it arrived as,
        with no rounding applied first -- a threshold change of one
        hundredth must change behaviour by exactly that much, not by a
        rounding step. A score exactly equal to `threshold` is a hit;
        strictly below it is not.
        """
        if score < self._threshold:
            return GateDecision.block("below_threshold")

        # A hit exactly at the end of the refractory window counts; one
        # arriving inside it is suppressed -- `<` here, not `<=`, is the
        # boundary this comparison exists to get right.
        if last_hit_at is not None and (now - last_hit_at) < self._refractory_s:
            return GateDecision.block("refractory")

        # An empty list never reaches the injected callable at all -- a
        # house with `mute_when_playing` empty pays nothing for this check.
        if self._mute_when_playing and self._is_media_playing(self._mute_when_playing):
            return GateDecision.block("media_playing")

        return GateDecision.allow()
