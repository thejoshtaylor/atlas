"""Pure G.711 A-law conversion -- the codec, and nothing else.

Python 3.14 removed `audioop`, and this codebase's only remaining A-law
decode is `CameraAudioSource.decode_for_detector`'s PyAV codec context
(`transports/camera.py`), which is stateful and belongs to a source. Neither
the probe generator (`audio/probe.py`) nor plan 02-12's emitted-output trace
has a source to borrow one from, so this module supplies conversion as pure
functions over `bytes` instead. It adds no dependency: `numpy` is already
declared in `pyproject.toml`.

The one governing move, mirroring `audio/energy.py`'s own habit of stating
its unit once at the top: the 256-entry A-law decode table is built once at
import, from the standard ITU-T G.711 reference algorithm, and every code's
reconstructed PCM16 value in that table is distinct (verified by
`tests/test_echo_path.py`'s full-range round-trip case). Encoding is then
nearest-value lookup against a sorted copy of that same table via
`numpy.searchsorted` -- encode and decode share one table and can never
disagree about what a code means, which is what makes the 256-code
round-trip exact rather than merely close.

This module rejects an input it cannot honestly interpret rather than
misreading it: an odd-length PCM16 buffer is not a truncation to tolerate,
it is a caller error naming a wrong sample width.
"""

from __future__ import annotations

import numpy as np

_SIGN_BIT = 0x80
_QUANT_MASK = 0x0F
_SEG_SHIFT = 4
_SEG_MASK = 0x70

# Bytes per sample for the two encodings `transports/base.py`'s
# `SourceFormat` declares -- 16-bit signed PCM and 8-bit companded A-law.
_BYTES_PER_SAMPLE = {"pcm": 2, "alaw": 1}


class AlawError(ValueError):
    """Raised for an input this module cannot honestly interpret, rather
    than silently truncating or misreading it."""


def _alaw_code_to_pcm16(a_val: int) -> int:
    """One A-law code to its linear PCM16 reconstruction level, by the
    standard ITU-T G.711 reference algorithm (the same bit-inversion and
    segment-shift form used by the reference `g711.c` this format
    originates from)."""
    a_val ^= 0x55
    t = (a_val & _QUANT_MASK) << 4
    seg = (a_val & _SEG_MASK) >> _SEG_SHIFT
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    return t if (a_val & _SIGN_BIT) else -t


# Built once at import: index i holds the PCM16 value A-law code i decodes
# to. All 256 entries are distinct (asserted by test), which is exactly what
# makes nearest-value encoding below unambiguous for every one of them.
_DECODE_TABLE = np.array([_alaw_code_to_pcm16(code) for code in range(256)], dtype=np.int16)

# A sorted view of the same table, plus the codes in that sorted order --
# `numpy.searchsorted` needs an ascending array to search against, and this
# is a sorted *copy*, never a second, independently-computed table that
# could drift from `_DECODE_TABLE`.
_SORT_ORDER = np.argsort(_DECODE_TABLE)
_SORTED_VALUES = _DECODE_TABLE[_SORT_ORDER].astype(np.int64)
_SORTED_CODES = _SORT_ORDER.astype(np.uint8)


def bytes_per_sample(encoding: str) -> int:
    """Bytes per sample for one of the two encodings `SourceFormat` names:
    `"pcm"` (16-bit signed PCM) or `"alaw"` (8-bit companded). Any other
    value is rejected by name -- this codebase carries no third encoding
    end to end."""
    try:
        return _BYTES_PER_SAMPLE[encoding]
    except KeyError:
        raise AlawError(
            f"unknown encoding {encoding!r} -- supported values are {sorted(_BYTES_PER_SAMPLE)!r}"
        ) from None


def alaw_to_pcm16(data: bytes) -> bytes:
    """Decode A-law bytes to 16-bit signed PCM, one code per input byte."""
    if not data:
        return b""
    codes = np.frombuffer(data, dtype=np.uint8)
    return _DECODE_TABLE[codes].tobytes()


def pcm16_to_alaw(data: bytes) -> bytes:
    """Encode 16-bit signed PCM to A-law, by nearest-value lookup against
    the same table `alaw_to_pcm16` decodes from.

    An odd-length `data` is rejected by name: PCM16 is two bytes per
    sample, and truncating the trailing byte would silently drop a sample
    instead of reporting the caller's mistake.
    """
    if not data:
        return b""
    if len(data) % 2 != 0:
        raise AlawError(f"pcm16_to_alaw requires an even-length buffer, got {len(data)} bytes")
    samples = np.frombuffer(data, dtype="<i2").astype(np.int64)
    # searchsorted gives the insertion point; the nearest reconstruction
    # level is whichever of the two neighboring sorted entries is closer,
    # so a value that falls exactly between two levels does not always
    # round the same direction as a plain floor/ceil search would.
    idx = np.searchsorted(_SORTED_VALUES, samples)
    idx = np.clip(idx, 1, len(_SORTED_VALUES) - 1)
    left = _SORTED_VALUES[idx - 1]
    right = _SORTED_VALUES[idx]
    use_left = (samples - left) <= (right - samples)
    nearest_idx = np.where(use_left, idx - 1, idx)
    codes = _SORTED_CODES[nearest_idx]
    return codes.astype(np.uint8).tobytes()
