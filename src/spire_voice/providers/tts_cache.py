"""Startup-built audio for the closed set of holding phrases and macro replies.

These phrases are a closed set known at startup -- every `FillerPhrase`
member and every configured macro reply -- so a lookup miss at turn time
means the startup precache itself failed, not a runtime condition to paper
over. This module raises on a miss rather than falling back to a live
synthesis call: a permissive miss-then-synthesize fallback is the natural
instinct for cache code in general, and it is wrong here specifically,
because it would put the ~1,467 ms REST text-to-speech cost back onto
exactly the path this phase built the cache to remove it from -- silently,
with no test failure to show for it (RESEARCH.md Pitfall 4).

The cache is keyed on the playback sink's `SinkFormat` -- its codec and
sample rate -- never on `TtsConfig.codec`/`TtsConfig.sample_rate` directly.
Those two fields describe the camera speaker's A-law-at-8kHz path; the
browser sink this phase's only consumer uses is PCM at 24 kHz. A cache built
for the wrong sink plays as audible noise in the browser rather than failing
cleanly -- the exact failure `tts_xai.py`'s own module docstring already
records. Taking `SinkFormat` as an explicit parameter, rather than reading
config directly, is what keeps the camera-sink precache (Phase 2) from
colliding with the browser one: the codec and sample rate are part of the
on-disk cache key, so both sinks can populate the same directory safely.

260922-cts: that same "the sink is a parameter, never assumed" discipline
was true of `precache_all`'s on-disk key from Phase 2 onward, but not of
`CachedTts.synthesize` itself, which ignored the sink it was handed and
always read the one flat, in-memory cache it was constructed with -- so a
turn on the camera played back whatever was in the browser's in-memory
cache, in browser PCM, through the camera's A-law speaker. `CachedTts` now
takes the same `{(codec, sample_rate): cache}` shape `app.py`'s own
precache step builds, and its `synthesize` reads the entry matching the
`sink` it is given, closing the in-memory half of this gap.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

from spire_voice.providers.base import TtsError, TtsProvider
from spire_voice.providers.tts_xai import SinkFormat

# Matches tts_xai.py's own chunk size, so a cached-audio consumer sees the
# same chunk granularity a live synthesis call would have produced.
_CHUNK_BYTES = 640


def cache_key(voice_id: str, codec: str, sample_rate: int, text: str) -> str:
    """A SHA-256 hex digest over voice, codec, sample rate, and text.

    The four fields are joined with a separator that cannot appear inside
    any of them by coincidence -- a pipe is not a legal character in a
    voice id, a codec name, or an integer sample rate, so only a change in
    `text` itself (which could contain a pipe) could theoretically produce a
    collision, and a hash collision on top of that is what makes SHA-256 the
    right primitive here rather than a plain string join.
    """
    raw = "|".join((voice_id, codec, str(sample_rate), text)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_path(cache_dir: Path, key: str, codec: str, sample_rate: int) -> Path:
    """The on-disk file for `key`.

    The codec and sample rate are embedded in the filename for human
    debugging only -- the hash alone is already collision-safe across a
    voice, codec, sample-rate, or text change.
    """
    return cache_dir / f"{key}_{codec}_{sample_rate}.raw"


async def _one_delta(text: str) -> AsyncIterator[str]:
    yield text


async def precache_all(
    tts: TtsProvider,
    cache_dir: Path,
    texts: list[str],
    voice_id: str,
    sink: SinkFormat,
) -> dict[str, bytes]:
    """Build (or read) the on-disk cache for every phrase in `texts`.

    Creates `cache_dir` if it does not exist. A phrase whose cache file
    already exists is read from disk rather than re-synthesized -- the point
    of a startup precache is that a restart does not re-pay the ~1,467 ms
    REST cost for a phrase that has not changed. Returns a `dict[str, bytes]`
    keyed by the phrase text itself, which is exactly the shape `get_cached`
    and `CachedTts` consume.

    `tts` is typed on the streaming `TtsProvider` protocol (plan 07-02),
    not a concrete provider class -- this function calls only
    `synthesize()` with a `sink` keyword, which every provider this
    process ever hands it (a raw streaming client, or a
    `BatchTtsAdapter`) already satisfies. `lifespan` (`app.py`) passes
    the exact same wrapped object it assigns to `app.state.tts`, never a
    second, separately-built one: this is what lets D-08's measured
    figure be populated at boot by the first phrase this call actually
    synthesizes, and stay honestly unset when every phrase was already on
    disk.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, bytes] = {}
    for text in texts:
        key = cache_key(voice_id, sink.codec, sink.sample_rate, text)
        path = cache_path(cache_dir, key, sink.codec, sink.sample_rate)
        if path.exists():
            cache[text] = path.read_bytes()
            continue
        audio = b"".join([chunk async for chunk in tts.synthesize(_one_delta(text), sink=sink)])
        path.write_bytes(audio)
        cache[text] = audio
    return cache


async def precache_other_sinks(
    tts: TtsProvider,
    cache_dir: Path,
    texts: list[str],
    voice_id: str,
    caches: Mapping[Any, dict[str, bytes]],
) -> None:
    """Add `texts` to every non-browser entry of `app.state.filler_caches`
    (keyed `(codec, sample_rate)`; the `None` key is the browser cache,
    which the caller already filled). Without this, a macro or workflow
    saved after boot is cached for the browser only, and a camera turn
    that speaks it raises `TtsError`."""
    for key, cache in caches.items():
        if key is None:
            continue
        codec, sample_rate = key
        cache.update(
            await precache_all(tts, cache_dir, texts, voice_id, SinkFormat(codec=codec, sample_rate=sample_rate))
        )


def get_cached(cache: Mapping[str, bytes], text: str) -> bytes:
    """Look up `text` in a precache built by `precache_all`.

    Raises `TtsError` naming the missing phrase on a miss -- never a
    default, and never a fallback to live synthesis. A miss means the
    startup precache itself failed to cover this phrase, which is a
    startup-time bug to fix, not a runtime condition this function should
    paper over.
    """
    try:
        return cache[text]
    except KeyError:
        raise TtsError(f"phrase not precached at startup: {text!r}") from None


class CachedTts:
    """Adapts a startup-built cache to the same `synthesize()` contract
    `XaiTts` exposes, so `_speak` can play a cached phrase through the exact
    code path it already uses for a live utterance -- no second
    audio-sending loop.

    260922-cts: `caches` is a mapping keyed by `(codec, sample_rate)` --
    one entry per sink format this process actually precached a cache for
    -- plus a `None` entry for the browser default, the same "no sink
    declared" meaning `turn/controller.py`'s `_speak` already gives a
    `None` sink. `synthesize`'s own `sink` argument selects which entry to
    read from; before this fix, this class ignored the sink entirely and
    always read the one flat cache it was constructed with, which played
    browser-format PCM through the camera speaker for every cached phrase
    (module docstring, RESEARCH.md Pitfall 4).
    """

    def __init__(self, caches: "Mapping[tuple[str, int] | None, Mapping[str, bytes]]") -> None:
        self._caches = caches

    async def synthesize(
        self, text_deltas: AsyncIterator[str], sink: "SinkFormat | None" = None
    ) -> AsyncIterator[bytes]:
        text = "".join([delta async for delta in text_deltas])
        key = (sink.codec, sink.sample_rate) if sink is not None else None
        cache = self._caches.get(key) or {}
        audio = get_cached(cache, text)
        for start in range(0, len(audio), _CHUNK_BYTES):
            yield audio[start : start + _CHUNK_BYTES]
