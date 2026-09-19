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

from spire_voice.providers.tts_xai import CHUNK_BYTES, SinkFormat

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
    """

    wrapped = True

    def __init__(self, provider: BatchTts, chunk_bytes: int = CHUNK_BYTES) -> None:
        self._provider = provider
        self._chunk_bytes = chunk_bytes
        # None until the first real call -- D-08's own honest state for
        # "nothing has been synthesized since this process started."
        self.last_synthesis_ms: "float | None" = None

    def __getattr__(self, name: str) -> Any:
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
