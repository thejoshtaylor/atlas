"""BatchTtsAdapter: one adapter over any provider that renders a whole
utterance in one call (D-05, D-06, D-07, D-08).

D-06 is the test that matters most here: the clock must cover only the
provider's own call, never the time a caller spends consuming chunks
afterward. A fake provider with a known, deliberate delay is what proves
that -- not a real network call, which would make the timing assertion
flaky by construction.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from atlas.providers.base import TtsError
from atlas.providers.batch_tts_adapter import (
    CHUNK_BYTES,
    BatchTtsAdapter,
    _split_first_sentence,
)


class _FakeBatchProvider:
    """A `BatchTts` double. Records every call it receives; can sleep a
    fixed interval before returning, or raise instead of returning."""

    def __init__(self, audio: bytes = b"", delay_s: float = 0.0, error: "Exception | None" = None) -> None:
        self.audio = audio
        self.delay_s = delay_s
        self.error = error
        self.calls: list[str] = []

    async def synthesize_once(self, text: str, sink=None) -> bytes:
        self.calls.append(text)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return self.audio


async def _deltas(*parts: str) -> AsyncIterator[str]:
    for part in parts:
        yield part


@pytest.mark.asyncio
async def test_chunks_at_the_repo_wide_chunk_size():
    provider = _FakeBatchProvider(audio=b"\x01" * 1600)
    adapter = BatchTtsAdapter(provider)

    chunks = [c async for c in adapter.synthesize(_deltas("hello"))]

    assert [len(c) for c in chunks] == [640, 640, 320]
    assert b"".join(chunks) == b"\x01" * 1600


@pytest.mark.asyncio
async def test_a_custom_chunk_size_overrides_the_default():
    provider = _FakeBatchProvider(audio=b"\x01" * 10)
    adapter = BatchTtsAdapter(provider, chunk_bytes=4)

    chunks = [c async for c in adapter.synthesize(_deltas("hi"))]

    assert [len(c) for c in chunks] == [4, 4, 2]


@pytest.mark.asyncio
async def test_deltas_are_joined_and_the_provider_is_called_exactly_once():
    provider = _FakeBatchProvider(audio=b"\x01" * 10)
    adapter = BatchTtsAdapter(provider)

    [_ async for _ in adapter.synthesize(_deltas("the light ", "is off"))]

    assert provider.calls == ["the light is off"]


@pytest.mark.asyncio
async def test_whitespace_only_text_calls_the_provider_zero_times_and_yields_nothing():
    provider = _FakeBatchProvider(audio=b"\x01" * 10)
    adapter = BatchTtsAdapter(provider)

    chunks = [c async for c in adapter.synthesize(_deltas("   "))]

    assert chunks == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_empty_text_calls_the_provider_zero_times_and_yields_nothing():
    provider = _FakeBatchProvider(audio=b"\x01" * 10)
    adapter = BatchTtsAdapter(provider)

    chunks = [c async for c in adapter.synthesize(_deltas())]

    assert chunks == []
    assert provider.calls == []


def test_last_synthesis_ms_is_none_before_any_call():
    adapter = BatchTtsAdapter(_FakeBatchProvider(audio=b"\x01" * 10))

    assert adapter.last_synthesis_ms is None


@pytest.mark.asyncio
async def test_last_synthesis_ms_covers_only_the_providers_own_span():
    """D-06: the clock stops when the provider returns, before the first
    chunk is yielded. A caller that is slow to keep consuming the
    iterator must never see the figure grow afterward."""
    provider = _FakeBatchProvider(audio=b"\x01" * 20, delay_s=0.05)
    adapter = BatchTtsAdapter(provider)

    chunks_iter = adapter.synthesize(_deltas("hi"))
    first_chunk = await chunks_iter.__anext__()
    measured_at_first_chunk = adapter.last_synthesis_ms

    assert measured_at_first_chunk is not None
    assert 40 <= measured_at_first_chunk < 2000, (
        f"expected the figure to fall inside the provider's own delay, got {measured_at_first_chunk}"
    )

    # A slow caller: keep the iterator open a while before finishing it.
    await asyncio.sleep(0.1)
    remaining = [c async for c in chunks_iter]

    assert [first_chunk, *remaining]
    assert adapter.last_synthesis_ms == measured_at_first_chunk, (
        "the figure must not grow while the caller consumes chunks slowly"
    )


def test_wrapped_is_true_and_readable_from_the_class_with_no_call():
    assert BatchTtsAdapter.wrapped is True
    assert BatchTtsAdapter(_FakeBatchProvider()).wrapped is True


@pytest.mark.asyncio
async def test_a_raising_provider_propagates_the_error_rather_than_yielding_silence():
    provider = _FakeBatchProvider(error=TtsError("boom"))
    adapter = BatchTtsAdapter(provider)

    with pytest.raises(TtsError):
        [c async for c in adapter.synthesize(_deltas("hello"))]


@pytest.mark.asyncio
async def test_default_chunk_size_is_the_one_definition_tts_xai_owns():
    """The adapter imports the chunk size from `tts_xai` rather than
    restating the integer -- this pins the import, not a duplicated
    literal, as the source of truth."""
    from atlas.providers.tts_xai import CHUNK_BYTES as XAI_CHUNK_BYTES

    assert CHUNK_BYTES == XAI_CHUNK_BYTES


@pytest.mark.asyncio
async def test_precache_all_through_the_adapter_populates_the_measured_figure(tmp_path):
    """Must-have: one adapter instance serves both the startup precache
    and every live turn, and the precache is what populates D-08's
    figure the first time an uncached phrase is actually synthesized."""
    from atlas.providers.tts_cache import precache_all
    from atlas.providers.tts_xai import SinkFormat

    provider = _FakeBatchProvider(audio=b"\x01" * 100)
    adapter = BatchTtsAdapter(provider)
    assert adapter.last_synthesis_ms is None

    await precache_all(
        adapter, tmp_path / "cache", ["hello"], "eve", SinkFormat(codec="pcm", sample_rate=24000)
    )

    assert adapter.last_synthesis_ms is not None
    assert provider.calls == ["hello"]


@pytest.mark.asyncio
async def test_the_adapter_forwards_unknown_attributes_to_the_wrapped_provider():
    """`browser_sink()`/`build_session_update()` stay on the raw client
    (Task 1's own action); the adapter forwards to them so a caller
    holding the wrapped object -- `app.state.tts`, once xAI's tts entry
    is batch -- can still reach them with no special case."""
    from atlas.config import TtsConfig
    from atlas.providers.tts_xai import SinkFormat, XaiTts

    xai = XaiTts(TtsConfig(browser_codec="pcm", browser_sample_rate=24000))
    adapter = BatchTtsAdapter(xai)

    sink = adapter.browser_sink()

    assert sink == SinkFormat(codec="pcm", sample_rate=24000)


# --- IN-01 (code review): a double wrap must not be legal --------------


def test_wrapping_an_adapter_in_another_adapter_is_refused():
    """IN-01. The blanket `__getattr__` forwarded `synthesize_once`, so an
    adapter satisfied `BatchTts` itself and a double wrap type-checked and
    ran. It double-chunks the audio, and the outer wrapper's
    `last_synthesis_ms` measures the inner wrapper's full drain rather
    than the provider call -- a corrupted figure that reads as a slow
    provider, which is the exact failure D-06 exists to rule out.

    `boot.py` is the only wrap site today so this could not happen; the
    point is that nothing forbade it."""
    inner = BatchTtsAdapter(_FakeBatchProvider(audio=b"audio"))

    with pytest.raises(TypeError) as excinfo:
        BatchTtsAdapter(inner)

    assert "BatchTtsAdapter" in str(excinfo.value)


def test_an_adapter_does_not_answer_the_batch_protocol_it_consumes():
    """The property that makes the refusal above unnecessary in the first
    place: an adapter is a streaming provider, and does not pretend to be
    a batch one."""
    adapter = BatchTtsAdapter(_FakeBatchProvider(audio=b"audio"))

    assert hasattr(adapter, "synthesize")
    with pytest.raises(AttributeError):
        adapter.synthesize_once


def test_a_half_constructed_adapter_raises_attribute_error_not_recursion_error():
    """`__getattr__` runs for `_provider` on an instance whose `__init__`
    never assigned it, and forwarding that name recursed until the stack
    ran out. A reader can act on an AttributeError; a RecursionError
    names nothing."""
    orphan = BatchTtsAdapter.__new__(BatchTtsAdapter)

    with pytest.raises(AttributeError):
        orphan.browser_sink


# --- 260924-4iu (c): head-then-tail sentence rendering ---------------------


def test_split_first_sentence_cases():
    """`_split_first_sentence`'s own boundary rule, table-tested rather
    than asserted one case at a time -- every case is a distinct reason a
    split does, or does not, happen."""
    cases = [
        (
            "It is eighteen degrees and sunny right now. Tomorrow brings rain after noon.",
            "It is eighteen degrees and sunny right now.",
            "Tomorrow brings rain after noon.",
        ),
        ("the light is off", "the light is off", ""),
        # The head ("Sure.") is below _MIN_HEAD_CHARS -- no boundary qualifies.
        ("Sure. The kitchen light is now off.", "Sure. The kitchen light is now off.", ""),
        # "3.5" has no whitespace after the decimal point -- not a boundary.
        (
            "The temperature outside is 3.5 degrees right now",
            "The temperature outside is 3.5 degrees right now",
            "",
        ),
        # The only boundary's tail is whitespace only -- disqualified.
        (
            "It is eighteen degrees and sunny right now.   ",
            "It is eighteen degrees and sunny right now.   ",
            "",
        ),
        # The first boundary ("Short.") is too short; the second qualifies.
        (
            "Short. It is eighteen degrees and sunny today! Rain comes later.",
            "Short. It is eighteen degrees and sunny today!",
            "Rain comes later.",
        ),
    ]
    for text, expected_head, expected_tail in cases:
        assert _split_first_sentence(text) == (expected_head, expected_tail), text


class _FakeSplitProvider:
    """A `BatchTts` double shaped for head/tail rendering: per-text audio,
    an optional `asyncio.Event` a call waits on before returning, an
    optional error one text raises instead of returning, an in-flight
    counter that records its own peak (for proving at most one call is
    ever concurrent), and cancellation recording.
    """

    def __init__(
        self,
        audio_by_text: dict,
        *,
        wait_event_by_text: dict | None = None,
        delay_by_text: dict | None = None,
        error_by_text: dict | None = None,
    ) -> None:
        self.audio_by_text = audio_by_text
        self.wait_event_by_text = wait_event_by_text or {}
        self.delay_by_text = delay_by_text or {}
        self.error_by_text = error_by_text or {}
        self.calls: list[str] = []
        self.completed_texts: list[str] = []
        self.cancelled_texts: list[str] = []
        self._in_flight = 0
        self.peak_in_flight = 0

    async def synthesize_once(self, text: str, sink=None) -> bytes:
        self.calls.append(text)
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            event = self.wait_event_by_text.get(text)
            if event is not None:
                await event.wait()
            delay = self.delay_by_text.get(text, 0.0)
            if delay:
                await asyncio.sleep(delay)
            error = self.error_by_text.get(text)
            if error is not None:
                raise error
            self.completed_texts.append(text)
            return self.audio_by_text.get(text, b"")
        except asyncio.CancelledError:
            self.cancelled_texts.append(text)
            raise
        finally:
            self._in_flight -= 1


_TWO_SENTENCE_REPLY = (
    "It is eighteen degrees and sunny right now. Tomorrow brings rain after noon."
)
_HEAD_TEXT, _TAIL_TEXT = _split_first_sentence(_TWO_SENTENCE_REPLY)


@pytest.mark.asyncio
async def test_multi_sentence_reply_plays_the_head_while_the_tail_renders():
    tail_event = asyncio.Event()
    head_audio = b"\x01" * 10
    tail_audio = b"\x02" * 10
    provider = _FakeSplitProvider(
        {_HEAD_TEXT: head_audio, _TAIL_TEXT: tail_audio},
        wait_event_by_text={_TAIL_TEXT: tail_event},
    )
    adapter = BatchTtsAdapter(provider)

    chunks_iter = adapter.synthesize(_deltas(_TWO_SENTENCE_REPLY))
    first_chunk = await chunks_iter.__anext__()

    await asyncio.sleep(0)
    assert provider.calls == [_HEAD_TEXT, _TAIL_TEXT]
    assert _TAIL_TEXT not in provider.completed_texts

    tail_event.set()
    tail_chunks = [c async for c in chunks_iter]

    assert first_chunk == head_audio
    assert b"".join(tail_chunks) == tail_audio


@pytest.mark.asyncio
async def test_at_most_one_provider_call_is_in_flight():
    provider = _FakeSplitProvider({_HEAD_TEXT: b"\x01" * 10, _TAIL_TEXT: b"\x02" * 10})
    adapter = BatchTtsAdapter(provider)

    [c async for c in adapter.synthesize(_deltas(_TWO_SENTENCE_REPLY))]

    assert provider.peak_in_flight == 1


@pytest.mark.asyncio
async def test_last_synthesis_ms_measures_only_the_head_call():
    provider = _FakeSplitProvider(
        {_HEAD_TEXT: b"\x01" * 10, _TAIL_TEXT: b"\x02" * 10},
        delay_by_text={_HEAD_TEXT: 0.05, _TAIL_TEXT: 0.3},
    )
    adapter = BatchTtsAdapter(provider)

    chunks_iter = adapter.synthesize(_deltas(_TWO_SENTENCE_REPLY))
    first_chunk = await chunks_iter.__anext__()
    measured_at_first_chunk = adapter.last_synthesis_ms

    assert measured_at_first_chunk is not None
    assert 40 <= measured_at_first_chunk < 250, (
        f"expected the figure to fall inside the head call's own delay, got {measured_at_first_chunk}"
    )

    remaining = [c async for c in chunks_iter]

    assert [first_chunk, *remaining]
    assert adapter.last_synthesis_ms == measured_at_first_chunk, (
        "the figure must not change once the tail finishes"
    )


@pytest.mark.asyncio
async def test_a_failing_tail_raises_after_the_head_audio():
    head_audio = b"\x01" * 10
    provider = _FakeSplitProvider(
        {_HEAD_TEXT: head_audio}, error_by_text={_TAIL_TEXT: TtsError("tail boom")}
    )
    adapter = BatchTtsAdapter(provider)

    received: list[bytes] = []
    with pytest.raises(TtsError):
        async for chunk in adapter.synthesize(_deltas(_TWO_SENTENCE_REPLY)):
            received.append(chunk)

    assert b"".join(received) == head_audio, "every head chunk must reach the caller before the raise"


@pytest.mark.asyncio
async def test_closing_the_iterator_early_cancels_the_tail_render():
    tail_event = asyncio.Event()
    provider = _FakeSplitProvider(
        {_HEAD_TEXT: b"\x01" * 10, _TAIL_TEXT: b"\x02" * 10},
        wait_event_by_text={_TAIL_TEXT: tail_event},
    )
    adapter = BatchTtsAdapter(provider)

    chunks_iter = adapter.synthesize(_deltas(_TWO_SENTENCE_REPLY))
    await chunks_iter.__anext__()
    await chunks_iter.aclose()

    tail_event.set()
    await asyncio.sleep(0.05)

    assert _TAIL_TEXT not in provider.completed_texts
    assert _TAIL_TEXT in provider.cancelled_texts or _TAIL_TEXT not in provider.calls, (
        "a cancelled-before-its-first-step task never runs, so 'never called' is also valid"
    )
