"""A deterministic broadband probe, in the camera's own wire format.

`build_ffmpeg_argv` (`speaker/ffmpeg_supervisor.py`) reads the speaker FIFO
as `-f alaw -ar 8000 -ac 1` and pushes it with `-c copy` -- no re-encode.
A probe generated in any other format would travel a path the reply audio
never takes, so this module emits A-law bytes directly, alongside the
PCM16 reference `audio/echo_path.py`'s correlator compares the recording
against.

Why not a tone: a periodic signal's autocorrelation is periodic too, so the
delay a correlator reads back from it is ambiguous by whole periods -- a
number that looks measured and is not. This module generates band-limited
noise instead, shaped to the band an 8 kHz telephony path actually carries
(roughly 300-3400 Hz, the classic G.711 voice band), by zeroing every
frequency bin outside that band in the frequency domain and transforming
back -- no filter design, no new dependency, just the FFT the correlator
already needs.

`PROBE_FORMAT_VERSION` and `DEFAULT_PROBE_SEED` are named constants, not
buried literals, because a later change to the waveform -- a different
band, a different fade, a different default seed -- must be visible in
every `EchoCalibration` record that carries it (`calibration/record.py`),
never a silent change to what "the probe" means.
"""

from __future__ import annotations

import numpy as np

from spire_voice.audio.alaw import pcm16_to_alaw

# Bumped whenever the waveform itself changes shape (band, fade, or
# normalization) -- `calibration/record.py` stores this alongside the seed
# so a stale record's provenance says plainly which probe produced it.
PROBE_FORMAT_VERSION = 1

# The default seed real callers (plan 02-11's calibration runner) use, so
# calibrations taken weeks apart are comparable rather than each drawing an
# unrelated noise burst. A caller may still pass a different seed --
# `tests/test_echo_path.py` does, to prove determinism holds for more than
# one value -- but this is the one a real run should reach for.
DEFAULT_PROBE_SEED = 20260917

# The G.711 voice band this probe is shaped to: narrower than 8 kHz's
# theoretical 4 kHz Nyquist ceiling, matching what a telephony-descended
# path (this camera's codec) actually carries end to end.
_BAND_LOW_HZ = 300.0
_BAND_HIGH_HZ = 3400.0

# A burst shorter than this cannot give the correlator (`echo_path.py`)
# enough independent samples to resolve a single delay from -- rejected by
# name rather than silently correlated against anyway.
MIN_DURATION_S = 1.0

# Each edge fades in/out over this long, so the burst does not start or end
# on a click the correlator could lock onto instead of the signal's body.
_FADE_S = 0.02

# Normalized to this fraction of full scale, not to full scale itself, so
# the camera speaker is never driven into clipping and the recorded level
# stays a linear function of what was sent -- the assumption
# `echo_path.py`'s gain measurement rests on.
_PEAK_FRACTION = 0.5

_INT16_FULL_SCALE = 32767.0


class ProbeError(Exception):
    """Raised for a probe request this module cannot honestly satisfy,
    rather than returning a signal too short to correlate against."""


def _raised_cosine_fade(num_samples: int, fade_samples: int) -> np.ndarray:
    """A `[0, 1]` envelope: raised-cosine up, flat, raised-cosine down."""
    envelope = np.ones(num_samples, dtype=np.float64)
    if fade_samples <= 0:
        return envelope
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_samples, endpoint=True)))
    envelope[:fade_samples] = ramp
    envelope[-fade_samples:] = ramp[::-1]
    return envelope


def build_probe(sample_rate: int, duration_s: float, seed: int = DEFAULT_PROBE_SEED) -> tuple[bytes, bytes]:
    """A deterministic band-limited noise burst: the A-law bytes to write
    toward the speaker, and the PCM16 reference `measure_echo_path`
    compares a recording against.

    Deterministic in `seed`, `sample_rate`, and `duration_s` alone -- two
    calls with the same three arguments return byte-identical output,
    which is what lets a test regenerate the probe and lets two
    calibrations taken weeks apart be compared against the same signal.
    """
    if duration_s < MIN_DURATION_S:
        raise ProbeError(
            f"duration_s={duration_s!r} is below the minimum {MIN_DURATION_S}s the "
            "correlator needs to resolve a single delay from"
        )

    num_samples = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(num_samples)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    band_mask = (freqs >= _BAND_LOW_HZ) & (freqs <= _BAND_HIGH_HZ)
    spectrum[~band_mask] = 0.0
    banded = np.fft.irfft(spectrum, n=num_samples)

    fade_samples = min(int(round(_FADE_S * sample_rate)), num_samples // 2)
    banded *= _raised_cosine_fade(num_samples, fade_samples)

    peak = np.max(np.abs(banded))
    if peak > 0:
        banded = banded * (_PEAK_FRACTION * _INT16_FULL_SCALE / peak)

    pcm16 = np.clip(np.round(banded), -32768, 32767).astype(np.int16)
    pcm16_bytes = pcm16.tobytes()
    alaw_bytes = pcm16_to_alaw(pcm16_bytes)
    return alaw_bytes, pcm16_bytes
