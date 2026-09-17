"""One measure -- root-mean-square amplitude -- and nothing else.

Mirroring `timing.py`'s own habit of stating one governing boundary up top:
this module measures energy and decides nothing about barge-in policy. The
floor comparison, the sustained-duration accumulation, and the post-playback
guard window all belong to `sources/runner.py`'s `BargeInMonitor` -- a
measure that also decided would be a measure nobody could test in isolation.

The unit is fixed here, once, and stated on the one function that produces
it: a normalized fraction of full-scale amplitude for 16-bit signed PCM,
in `[0.0, 1.0]`. Nothing on the path from a configured floor
(`BargeInConfig.energy_floor`, `config.py`) to the comparison against this
value may convert it to or from a decibel scale -- a configured floor means
one thing only. A decibel view, if ever wanted for a display, would be a
separate, explicitly named function that converts once; it is not this one,
and it does not exist here.

Sample rate is a parameter this module is told, per `SourceFormat`
(`transports/base.py`) -- never assumed. It is deliberately not read by
`rms_amplitude` at all: RMS amplitude of a constant-amplitude signal does
not depend on how many samples represent one second of it, so accepting a
rate here and not using it would invite a caller to believe rate matters to
this function when it does not.
"""

from __future__ import annotations

import numpy as np

# Signed 16-bit PCM's maximum magnitude -- `-32768..32767`, but the positive
# bound is what a full-scale RMS reading is normalized against, matching the
# convention `int16`'s own type range implies. Fixed here, the one place the
# unit is defined, per the module docstring.
_INT16_FULL_SCALE = 32768.0


def rms_amplitude(samples: bytes, *, sample_width: int = 2) -> float:
    """The root-mean-square amplitude of `samples`, normalized to
    `[0.0, 1.0]` against 16-bit signed PCM full scale.

    `samples` must already be linear PCM at `sample_width` bytes per sample
    -- this function never decodes A-law or any other companded encoding;
    that decode (`CameraAudioSource.decode_for_detector`, `transports/
    camera.py`) is a separate, earlier step the same way it already is for
    the wake detector. `sample_width` other than 2 is rejected rather than
    silently misread, since this module's only documented unit is 16-bit
    signed PCM's full scale.

    Independent of both chunk length and sample rate: a constant-amplitude
    signal measures the same whether handed one long chunk or many short
    ones, and the same signal resampled to a different rate (same waveform,
    more or fewer samples per second) measures the same too. Both are
    exactly what makes accumulating *duration* against this value, later,
    meaningful -- a measure that drifted with chunk size would make
    "sustained for the configured minimum duration" mean nothing.

    Silence (an all-zero chunk) measures `0.0` exactly, never a residual
    noise floor this function invents. An empty `samples` also measures
    `0.0` -- there is no signal to have any amplitude at all.
    """
    if sample_width != 2:
        raise ValueError(f"rms_amplitude only supports 16-bit PCM (sample_width=2), got {sample_width!r}")
    if not samples:
        return 0.0
    values = np.frombuffer(samples, dtype="<i2").astype(np.float64)
    rms = float(np.sqrt(np.mean(np.square(values))))
    return rms / _INT16_FULL_SCALE
