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
config directly, is what keeps a future camera-sink precache (Phase 2) from
colliding with this one: the codec and sample rate are part of the cache
key, so both sinks can populate the same directory safely.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import AsyncIterator, Mapping

from spire_voice.providers.base import TtsError
from spire_voice.providers.tts_xai import SinkFormat, XaiTts

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
    tts: XaiTts,
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
    """

    def __init__(self, cache: Mapping[str, bytes]) -> None:
        self._cache = cache

    async def synthesize(self, text_deltas: AsyncIterator[str]) -> AsyncIterator[bytes]:
        text = "".join([delta async for delta in text_deltas])
        audio = get_cached(self._cache, text)
        for start in range(0, len(audio), _CHUNK_BYTES):
            yield audio[start : start + _CHUNK_BYTES]
