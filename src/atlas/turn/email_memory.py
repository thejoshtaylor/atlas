"""`EmailListMemory`: what "read the second one" or "read Dana's" resolves
against -- the last spoken email list, per source, held in process for a
few minutes (D-16, GOOG-08). Never the Postgres `last_email_list` table
09-RESEARCH.md sketched: keeping email metadata off disk means a restart
simply forgets the list ("i don't have a recent email list") rather than
persisting senders and subjects. Per source, per instance -- never module
state (SRC-03), matching `HandoffSlot`'s own "one instance per turn"
discipline extended to "one instance per application, one entry per
source".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal, Sequence


@dataclass(frozen=True)
class EmailListItem:
    """One item of a spoken email list -- no body field, by construction
    (D-16): this is metadata only, exactly what `EmailListMemory` may
    ever hold onto past the turn that spoke it."""

    position: int
    account: str
    message_id: str
    thread_id: str
    from_name: str
    from_address: str
    subject: str


@dataclass(frozen=True)
class ResolveMiss:
    """Why `EmailListMemory.resolve` could not find a match --
    `"no_list"` (nothing stored for this source, or it expired),
    `"out_of_range"` (a position past the end of the stored list; `count`
    is how many there were), or `"no_sender"` (no item's sender matched)."""

    reason: Literal["no_list", "out_of_range", "no_sender"]
    count: int = 0


class EmailListMemory:
    """The last spoken email list, per source, expiring after
    `lifetime_s` (default ten minutes, D-16) -- `clock` is injectable so a
    test can prove the expiry without a real sleep."""

    def __init__(self, lifetime_s: float = 600.0, clock: "Callable[[], float]" = time.monotonic) -> None:
        self._lifetime_s = lifetime_s
        self._clock = clock
        self._by_source: "dict[str, tuple[float, tuple[EmailListItem, ...]]]" = {}

    def store(self, source: str, items: "Sequence[EmailListItem]") -> None:
        """Replace `source`'s own stored list -- a later `store` for the
        same source (a fresh "any new email?") supersedes whatever was
        there, never appends to it."""
        self._by_source[source] = (self._clock(), tuple(items))

    def get(self, source: str) -> "tuple[EmailListItem, ...] | None":
        """`source`'s own stored list, or `None` when nothing was ever
        stored for it, or the stored entry is older than `lifetime_s`."""
        entry = self._by_source.get(source)
        if entry is None:
            return None
        stored_at, items = entry
        if self._clock() - stored_at > self._lifetime_s:
            return None
        return items

    def resolve(
        self, source: str, *, position: "int | None" = None, sender: "str | None" = None
    ) -> "EmailListItem | ResolveMiss":
        """One item from `source`'s own stored list, by 1-based `position`
        or by `sender` (never both -- the caller, `handle_email_read`,
        already refused that combination before this is ever reached).

        Sender matching is case-insensitive against each word of the
        item's own `from_name`, the address's local part, and the whole
        address -- the lowest position wins when more than one item
        matches (the stored list is already in the order it was spoken)."""
        items = self.get(source)
        if items is None:
            return ResolveMiss(reason="no_list")
        if position is not None:
            if 1 <= position <= len(items):
                return items[position - 1]
            return ResolveMiss(reason="out_of_range", count=len(items))

        normalized = (sender or "").strip().casefold()
        for item in items:
            words = item.from_name.casefold().split()
            local_part = item.from_address.split("@", 1)[0].casefold()
            address = item.from_address.casefold()
            if normalized and (normalized in words or normalized == local_part or normalized == address):
                return item
        return ResolveMiss(reason="no_sender")
