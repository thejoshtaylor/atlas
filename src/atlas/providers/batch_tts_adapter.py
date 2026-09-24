"""Adapts a batch text-to-speech provider to the streaming `TtsProvider`
protocol (D-05, D-06, D-07, D-08).

xAI's text-to-speech endpoint and Piper both render a whole utterance in
one call -- there is no partial audio to stream out as it arrives. This
adapter is the one place that turns "one call, one buffer" into the async
iterator of bytes every turn already consumes: `turn/controller.py` calls
`tts.synthesize()` and iterates the result, and it cannot tell a wrapped
provider from a real streaming one, because it is never given a reason to
look (D-07 -- one shared adapter, not a per-provider special case).

D-06 is the reason `synthesize()`'s own order matters more than anything
else in this file. First-audio latency is this project's load-bearing
measured number (Phase 01.1 exists because of it). The clock starts right
before the one provider call and stops right after it returns -- before a
single chunk reaches the caller. Reading the clock any earlier would time
work that has not started; reading it any later (say, once the first
chunk has been handed back) would credit the wrapper with speed the
provider never had, and that is exactly the corruption D-06 exists to
rule out.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Protocol

from atlas.providers.tts_xai import CHUNK_BYTES, SinkFormat

__all__ = ["BatchTts", "BatchTtsAdapter", "CHUNK_BYTES"]


class BatchTts(Protocol):
    async def synthesize_once(self, text: str, sink: "SinkFormat | None" = None) -> bytes:
        """Render the whole utterance in one call and return it as one buffer."""
        ...


class BatchTtsAdapter:
    """Wraps one `BatchTts` provider so it satisfies `TtsProvider.synthesize`.

    `wrapped` is a class attribute, not a property or a method: D-08 needs
    a caller (`boot.py::resolve_slot`, then `routes/providers.py`) to read
    it with no call and no network, off either the class or an instance.

    Any attribute this class does not define itself -- `browser_sink()`,
    `build_session_update()` -- is forwarded to the wrapped provider. The
    adapter does not need to know those methods exist to pass them
    through; it only owns the one method the streaming protocol requires.

    IN-01 (code review): the forwarding used to include
    `synthesize_once`, which made an adapter satisfy `BatchTts` itself --
    so `BatchTtsAdapter(BatchTtsAdapter(provider))` type-checked and ran,
    double-chunking the audio and, worse, leaving the outer wrapper's
    `last_synthesis_ms` measuring the inner wrapper's full drain rather
    than the provider call. That is precisely the corruption of the
    measured figure D-06 exists to rule out, and it would have looked
    like a slower provider rather than a bug. `boot.py:120` is the only
    wrap site today, so it could not happen -- but nothing said so.
    """

    wrapped = True

    # The methods that define the batch side of this adapter's own
    # boundary. Forwarding either one would make an adapter indisting-
    # uishable from the provider it wraps.
    _NEVER_FORWARDED = frozenset({"synthesize_once", "synthesize"})

    def __init__(self, provider: BatchTts, chunk_bytes: int = CHUNK_BYTES) -> None:
        if isinstance(provider, BatchTtsAdapter):
            raise TypeError(
                "BatchTtsAdapter cannot wrap another BatchTtsAdapter: the audio would "
                "be chunked twice and last_synthesis_ms would measure the inner "
                "wrapper's drain rather than the provider call (D-06)"
            )
        self._provider = provider
        self._chunk_bytes = chunk_bytes
        # None until the first real call -- D-08's own honest state for
        # "nothing has been synthesized since this process started."
        self.last_synthesis_ms: "float | None" = None

    def __getattr__(self, name: str) -> Any:
        # `__getattr__` runs for `_provider` itself on a half-constructed
        # instance (one whose `__init__` raised before the assignment), and
        # forwarding it would recurse until the stack ran out. A plain
        # AttributeError is what a reader can act on.
        if name == "_provider":
            raise AttributeError(name)
        if name in self._NEVER_FORWARDED:
            raise AttributeError(
                f"{name!r} is not forwarded: a BatchTtsAdapter is a streaming "
                "provider, and answering the batch protocol as well would make it "
                "legal to wrap one in another"
            )
        return getattr(self._provider, name)

    async def synthesize(
        self, text_deltas: AsyncIterator[str], sink: "SinkFormat | None" = None
    ) -> AsyncIterator[bytes]:
        """Join `text_deltas`, call the batch provider once, then yield
        the result in `chunk_bytes`-sized pieces.

        The steps below run in this exact order because the order is
        D-06 itself, not an implementation detail: join the text, return
        early on nothing to say, read the clock, await the one provider
        call, read the clock again and store the elapsed time, and only
        then start yielding chunks. Nothing before the provider's own
        return produces one real byte of audio, so the stored figure
        covers exactly the span the provider took -- never the time a
        slow caller spends consuming what was already rendered.

        A provider that raises propagates the error rather than this
        method yielding silence in its place (`providers/base.py`'s
        raise-do-not-return rule) -- a caller must never read a failed
        synthesis as an empty reply.
        """
        text = "".join([delta async for delta in text_deltas])
        if not text.strip():
            return

        start = time.monotonic()
        audio = await self._provider.synthesize_once(text, sink=sink)
        self.last_synthesis_ms = (time.monotonic() - start) * 1000

        for chunk_start in range(0, len(audio), self._chunk_bytes):
            yield audio[chunk_start : chunk_start + self._chunk_bytes]
