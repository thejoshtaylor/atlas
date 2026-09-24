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

Quick task 260924-4iu (c): `synthesize` used to join the whole reply and
make one provider call for it, so a long reply's audio started only
after every sentence had rendered. A reply of two sentences or more now
renders as a head (its first sentence) and a tail (the rest): the head
call is the one D-06 times and the one whose audio starts playing first;
the tail call starts only after the head call returns -- never
concurrently with it, since Piper renders in a worker thread and its
phonemizer is not safe to call twice at once -- and its audio follows the
head's. A single-sentence reply, or one whose first sentence is shorter
than `_MIN_HEAD_CHARS`, still makes exactly one provider call, unchanged.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, AsyncIterator, Protocol

from atlas.providers.tts_xai import CHUNK_BYTES, SinkFormat

__all__ = ["BatchTts", "BatchTtsAdapter", "CHUNK_BYTES"]

# A sentence boundary: `.`, `!`, or `?`, an optional closing quote or
# parenthesis, then at least one whitespace character. A decimal point
# ("3.5") has no whitespace after it and is never a boundary.
_SENTENCE_BOUNDARY_RE = re.compile(r"""[.!?]['")\]]?(\s+)""")

# Speech runs at roughly 15 characters per second, so a 20-character head
# plays for about 1.3 s -- enough to cover most of a tail render
# (production p50 is 1.25 s) without an audible gap after a short head
# such as "Sure." (5 characters, below this floor on its own).
_MIN_HEAD_CHARS = 20


def _split_first_sentence(text: str) -> tuple[str, str]:
    """Split `text` at the first sentence boundary whose head reaches
    `_MIN_HEAD_CHARS` and whose tail is not only whitespace.

    Returns `(text, "")` -- the whole text as the head, no tail -- when no
    boundary qualifies: a single-sentence reply, a reply whose first
    sentence is too short to be worth a head/tail split on its own, or a
    reply with a trailing boundary and nothing but whitespace after it.
    """
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        head_end = match.start(1)
        tail_start = match.end(1)
        head = text[:head_end]
        tail = text[tail_start:]
        if len(head.strip()) >= _MIN_HEAD_CHARS and tail.strip():
            return head, tail
    return text, ""


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
        """Join `text_deltas`, call the batch provider for the head (and,
        for a longer reply, the tail), then yield each in `chunk_bytes`
        pieces -- one provider call for a single-sentence reply, two
        (head, then tail) for a longer one (260924-4iu, c).

        The steps below run in this exact order because the order is
        D-06 itself, not an implementation detail: join the text, return
        early on nothing to say, split it into a head and a tail, read
        the clock, await the head call, read the clock again and store
        the elapsed time -- `last_synthesis_ms` is the head call only,
        the time a listener actually waits before the first audio, which
        is what D-06 protects -- and only then start yielding chunks.
        Nothing before the head call's own return produces one real byte
        of audio, so the stored figure covers exactly that call's span,
        never the time a slow caller spends consuming what was already
        rendered.

        The tail call, when there is a tail, starts as a task right after
        the head call returns -- never before, and never concurrently
        with it, so at most one provider call is in flight at any time
        (Piper's phonemizer is not safe to call twice at once). Its audio
        overlaps the head's playback rather than the head's own render:
        the caller is already consuming head chunks while the tail
        renders in the background.

        A provider that raises propagates the error rather than this
        method yielding silence in its place (`providers/base.py`'s
        raise-do-not-return rule) -- a caller must never read a failed
        synthesis as an empty reply. A tail failure surfaces the same way,
        after every head chunk has already been yielded.

        `finally` cancels a tail task the caller never waited out (the
        consumer stopped early, or was itself cancelled) and retrieves the
        exception of one that finished but was never awaited, so asyncio
        never logs an unretrieved exception either way. It does not await
        the cancelled task -- that could hold the generator's own close
        open on however long the task takes to unwind.
        """
        text = "".join([delta async for delta in text_deltas])
        if not text.strip():
            return

        head, tail = _split_first_sentence(text)

        start = time.monotonic()
        head_audio = await self._provider.synthesize_once(head, sink=sink)
        self.last_synthesis_ms = (time.monotonic() - start) * 1000

        tail_task: asyncio.Task[bytes] | None = None
        if tail:
            tail_task = asyncio.create_task(self._provider.synthesize_once(tail, sink=sink))

        try:
            for chunk_start in range(0, len(head_audio), self._chunk_bytes):
                yield head_audio[chunk_start : chunk_start + self._chunk_bytes]

            if tail_task is not None:
                tail_audio = await tail_task
                for chunk_start in range(0, len(tail_audio), self._chunk_bytes):
                    yield tail_audio[chunk_start : chunk_start + self._chunk_bytes]
        finally:
            if tail_task is not None:
                if not tail_task.done():
                    tail_task.cancel()
                elif not tail_task.cancelled():
                    # Already resolved (result or exception) but never
                    # awaited above -- an early-closed consumer skips the
                    # `await tail_task` entirely. Reading the exception
                    # here (and discarding it) is what keeps asyncio from
                    # logging "Task exception was never retrieved".
                    tail_task.exception()
