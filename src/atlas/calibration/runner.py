"""`run_echo_calibration`: the one implementation. Everything else that can
trigger a calibration -- `scripts/calibrate_echo_path.py`'s command line and
`app.py`'s two HTTP routes (both this plan's remaining tasks) -- calls this
coroutine and nothing else. Neither caller re-derives a measurement of its
own; that is the whole point of D-20, the seam Phase 3's wizard needs.

The sequence: build a probe at the source's own declared format; drain and
discard whatever the source has queued for `settle_s`, so the recording
that follows starts on room tone rather than whatever was mid-flight when
the source opened; write the probe to the speaker, converted to the
speaker's own sink format when one is given (260923-pyj, D4) -- the
reference used to correlate the recording always stays at the source's
own rate, regardless of what format the probe travelled in; drain and keep
`probe_duration_s + tail_s` worth of audio; decode it back to PCM16 at the
source's own native sample rate; hand both to `measure_echo_path`
(`audio/echo_path.py`). A measurement that fails to correlate returns that
failure by name and this module writes nothing -- a calibration that
measured nothing must never leave a record that looks like one (T-02-52).

**Decoding the recording never goes through `CameraAudioSource.
decode_for_detector`.** That method resamples unconditionally to 16 kHz
mono for the wake engine (`transports/camera.py`'s own
`_DETECTOR_SAMPLE_RATE`), regardless of the source's native rate. The probe
reference this module builds is generated at the source's *own* declared
rate (`source_format().sample_rate`, 8 kHz for the shipped camera), and
`build_probe` is deterministic in that rate: a second probe drawn at a
different rate from the same seed is not a resampled copy of the first, it
is an independent noise realization that happens to share a frequency
band. Correlating one against the other would measure nothing, every time,
against a real camera -- exactly the silent, permanent failure `audio/
alaw.py`'s own module docstring already anticipates for plan 02-12's
identical need ("neither the probe generator nor plan 02-12's emitted-
output trace has a source to borrow one from"). This module decodes the
drained bytes with `atlas.audio.alaw` directly, at the source's own
rate, for the same reason.

Two objects are handed to `run_echo_calibration`, not one: an audio source
(`frames()`, `source_format()`) and a speaker sink (`write()`), matching
exactly what `app.py`'s `lifespan` already holds apart as `camera_source`
and `speaker_writer` on `app.state` -- Task 3 hands both straight through,
never opening a second RTSP connection or a second FIFO writer alongside
the ones already running.

No field on the source's own configuration that carries a URL is ever
read here. The camera's credential-bearing address is a concern this
module has no reason to touch -- it operates on an already-open source and
sink, the same discipline `transports/camera.py`'s own module docstring
states for itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

import numpy as np

from atlas.audio.alaw import alaw_to_pcm16, pcm16_to_alaw
from atlas.audio.echo_path import AGC_ABSENT, measure_echo_path
from atlas.audio.energy import rms_amplitude
from atlas.audio.probe import DEFAULT_PROBE_SEED, PROBE_FORMAT_VERSION, build_probe
from atlas.calibration.record import SCHEMA_VERSION, CalibrationError, EchoCalibration
from atlas.config import CalibrationConfig, CameraConfig
from atlas.providers.tts_piper import _resample_pcm16
from atlas.providers.tts_xai import SinkFormat
from atlas.transports.base import SourceFormat

logger = logging.getLogger("atlas.calibration.runner")

# The one source name this system has today -- there is exactly one
# `AudioSource` a calibration can run against (the camera). Recorded in
# `EchoCalibration.source` rather than left for a caller to invent a label.
DEFAULT_SOURCE_NAME = "camera"

# Every calibration record this module writes shares this filename prefix,
# followed by a fixed-width UTC timestamp that sorts lexicographically in
# chronological order -- `find_latest_calibration` relies on exactly that
# to resolve "the current record" without a second, separately-maintained
# pointer file that could drift from what was actually written last.
_FILENAME_PREFIX = "echo_path-"
_FILENAME_GLOB = f"{_FILENAME_PREFIX}*.json"

# A written record is a few hundred bytes of JSON numbers and short
# strings -- never audio (module docstring). This bounds every file this
# module writes, so a test can assert the "no audio persists" claim
# structurally rather than by inspecting content.
MAX_RECORD_FILE_SIZE_BYTES = 8192

# The camera's A-law codec (`audio/alaw.py`) has no code that decodes to
# zero: its smallest reconstruction level is +/-8 (int16). Digital silence
# therefore decodes to samples that are all +/-8. An rms threshold cannot
# separate that from a real room: a quiet room on a camera that gates its
# own microphone sits at that same floor (measured live: rms 8 of 32768,
# and only about 20 while the probe played, well under any useful rms
# cut-off). The test is whether ANY sample rose above the floor. Pure
# digital silence never does, and a live room with even a faint residual
# probe does.
_ALAW_FLOOR_MAGNITUDE = 8


def _has_signal_above_codec_floor(pcm16: bytes) -> bool:
    samples = np.frombuffer(pcm16, dtype="<i2")
    return bool(samples.size) and int(np.max(np.abs(samples.astype(np.int32)))) > _ALAW_FLOOR_MAGNITUDE


class CalibrationRunnerError(CalibrationError):
    """Raised for a condition this module cannot honestly proceed past --
    reusing `CalibrationError`'s name rather than inventing a sibling,
    since both mean the same thing to a caller: do not trust what was
    about to be persisted."""


@dataclass(frozen=True)
class CalibrationRunResult:
    """Either `calibration` is set and `failure_reason` is `None`, or the
    reverse -- mirroring `EchoPathMeasurement`'s own discipline (`audio/
    echo_path.py`): a run that measured nothing must report that plainly,
    never a calibration that looks measured and is not.
    """

    calibration: EchoCalibration | None
    failure_reason: str | None


class _CalibrationSource(Protocol):
    """The two `AudioSource` members this module actually uses --
    `frames()` and `source_format()`. Structural, not `CameraAudioSource`
    imported by name, so a loopback fake drives the whole path in a unit
    test with no camera and no hardware at all."""

    def frames(self) -> AsyncIterator[bytes]: ...

    def source_format(self) -> SourceFormat: ...


class _SpeakerSink(Protocol):
    """The one thing the probe is written through -- the same structural
    shape `transports/camera.py`'s own `_SpeakerSink` protocol already
    uses, so `speaker.fifo_writer.FifoWriter` satisfies this with no
    adapter."""

    async def write(self, chunk: bytes) -> None: ...


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamped_filename(taken_at: datetime) -> str:
    stamp = taken_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{_FILENAME_PREFIX}{stamp}Z.json"


def _decode_native(raw: bytes, encoding: str) -> bytes:
    """Decode `raw` to PCM16 at whatever sample rate it already carries --
    never resampled, per this module's own docstring. `"alaw"` is decoded
    through `audio/alaw.py`'s pure codec; `"pcm"` is already PCM16 and
    passes through unchanged. `CameraConfig.from_config` accepts no third
    encoding (`config.py`), so anything else reaching here is a caller
    error, not a format this pipeline was ever meant to carry.
    """
    if encoding == "alaw":
        return alaw_to_pcm16(raw)
    if encoding == "pcm":
        return raw
    raise CalibrationRunnerError(
        f"unsupported source encoding {encoding!r} for echo-path calibration -- "
        "only 'alaw' and 'pcm' are ever produced by this pipeline"
    )


async def _collect_window(
    source: _CalibrationSource, seconds: float, sleep: Callable[[float], Awaitable[None]]
) -> bytes:
    """Drain `source.frames()` for `seconds`, via a background task racing
    an injected `sleep`.

    A fresh `frames()` iterator every call, never one held open across two
    windows: the real camera source's `frames()` reads from one shared
    queue underneath (`transports/camera.py`), so a second call after the
    first window's task has been cancelled or run to completion simply
    keeps draining that same queue from wherever it left off -- and
    cancelling a task mid-iteration on an async generator's own `__anext__`
    exhausts that particular generator object for good (Python's ordinary
    generator-close semantics), so reusing one iterator across windows
    would silently go silent on the second call. Two fresh iterators avoid
    that trap entirely.

    In production `sleep` is real `asyncio.sleep`, so this window is real
    wall-clock time. A test's injected `sleep` can instead pump the event
    loop without any real delay, letting a finite fixture source's whole
    output flow through near-instantly while still exercising the same
    code path.
    """
    buf = bytearray()

    async def _pull() -> None:
        async for chunk in source.frames():
            buf.extend(chunk)

    task: asyncio.Task[None] = asyncio.ensure_future(_pull())
    try:
        await sleep(seconds)
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
            await task
    return bytes(buf)


def _probe_for_sink(
    alaw_bytes: bytes, reference_pcm16: bytes, source_rate: int, sink: SinkFormat | None
) -> bytes:
    """The bytes actually written to the speaker: `alaw_bytes` unchanged
    when `sink` is `None` (the camera case of today), otherwise
    `reference_pcm16` resampled from `source_rate` to `sink.sample_rate`
    and encoded in `sink.codec`.

    Resampling from `reference_pcm16` rather than re-decoding `alaw_bytes`
    matters only in that both already carry identical audio
    (`audio/probe.py`'s own module docstring: `pcm16_to_alaw(reference_
    pcm16) == alaw_bytes` exactly) -- starting from the PCM16 reference
    avoids an extra decode step for the common `sink.codec == "pcm"` case.
    """
    if sink is None:
        return alaw_bytes
    resampled = _resample_pcm16(reference_pcm16, source_rate, sink.sample_rate)
    if sink.codec == "alaw":
        return pcm16_to_alaw(resampled)
    if sink.codec == "pcm":
        return resampled
    raise CalibrationRunnerError(
        f"unsupported sink codec {sink.codec!r} for the echo-path calibration probe -- "
        "only 'alaw' and 'pcm' are ever written to a speaker FIFO"
    )


async def run_echo_calibration(
    source: _CalibrationSource,
    speaker: _SpeakerSink,
    camera_config: CameraConfig,
    calibration_config: CalibrationConfig,
    placement_note: str,
    *,
    now: Callable[[], datetime] = _default_now,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    probe_seed: int = DEFAULT_PROBE_SEED,
    sink: SinkFormat | None = None,
    speaker_has_aec: bool = False,
) -> CalibrationRunResult:
    """Measure the echo path once, end to end, and persist the result --
    or return the reason it could not be measured, writing nothing.

    `source` and `speaker` are already open; this coroutine never connects
    or closes either (module docstring). `placement_note` is passed
    straight through to `EchoCalibration`, whose own construction rejects
    a value shaped like a URL (`calibration/record.py`) -- a caller that
    wants to fail before opening anything (`scripts/calibrate_echo_path.py`,
    this plan's Task 2) validates it a second time up front for that
    reason, but this coroutine adds no validation of its own beyond what
    the record already enforces.

    `sink` (260923-pyj, D4) is the format the speaker FIFO reader actually
    expects -- the TTS sink `app.py`'s `lifespan` builds as `camera_tts_
    sink`. `None` means A-law at the source's own rate, the camera case of
    today: the probe is written unconverted, byte for byte, the same as
    before this parameter existed.

    `speaker_has_aec` (260923-sfi, D2) gates the echo_cancelled fallback
    below. It defaults to `False`, the safe side: only `SpeakerConfig.
    cancels_own_echo` (true for `tapo_talk` alone) is trusted to explain
    "no echo came back" as the camera cancelling its own speaker output.
    On every other backend the same recording shape is refused as a plain
    failure instead, and nothing is saved.
    """
    fmt = source.source_format()
    alaw_bytes, reference_pcm16 = build_probe(
        fmt.sample_rate, calibration_config.probe_duration_s, seed=probe_seed
    )
    # Computed before the settle window, not after: an unsupported sink
    # codec must raise before anything plays, never after the probe has
    # already gone out over a real speaker.
    probe_bytes = _probe_for_sink(alaw_bytes, reference_pcm16, fmt.sample_rate, sink)

    # Drained and discarded: whatever the source queued before this run
    # ever wrote a probe is not room tone worth analyzing, it is leftover
    # from however long ago the source actually opened.
    await _collect_window(source, calibration_config.settle_s, sleep)

    await speaker.write(probe_bytes)

    recorded_raw = await _collect_window(
        source, calibration_config.probe_duration_s + calibration_config.tail_s, sleep
    )
    recorded_pcm16 = _decode_native(recorded_raw, fmt.encoding)

    measurement = measure_echo_path(reference_pcm16, recorded_pcm16, fmt.sample_rate)

    logger.info(
        "echo calibration measured: source=%s delay_s=%s confidence=%.3f "
        "gain=%s agc_verdict=%s failure_reason=%s",
        DEFAULT_SOURCE_NAME,
        measurement.delay_s,
        measurement.confidence,
        measurement.gain,
        measurement.agc_verdict,
        measurement.failure_reason,
    )

    if measurement.failure_reason is not None:
        # A camera that cancels its own echo (`aec` mode) produces exactly
        # this same "no correlation peak" failure a dead microphone would
        # -- `measurement.no_echo` only narrows it to that one failure
        # branch, never the "reference or recording is empty" branch. The
        # recording's own energy is what tells the two apart: a cancelled
        # echo still recorded *something* (room tone, at minimum), while a
        # dead microphone recorded digital silence. Checked here, not by
        # matching `failure_reason`'s text (T-260922-eca).
        #
        # 260923-sfi (D2): `speaker_has_aec` gates this fallback too -- a
        # speaker on another device, or one that never played at all, can
        # produce this exact same "something came back, none of it is the
        # probe" shape with no camera echo cancellation involved.
        if speaker_has_aec and measurement.no_echo and _has_signal_above_codec_floor(recorded_pcm16):
            logger.warning(
                "no echo came back for source=%s (confidence=%.3f) -- "
                "assuming the camera cancels its own speaker output from its microphone",
                DEFAULT_SOURCE_NAME,
                measurement.confidence,
            )
            taken_at = now()
            calibration = EchoCalibration(
                schema_version=SCHEMA_VERSION,
                probe_format_version=PROBE_FORMAT_VERSION,
                probe_seed=probe_seed,
                source=DEFAULT_SOURCE_NAME,
                delay_s=0.0,
                confidence=measurement.confidence,
                echo_level=rms_amplitude(recorded_pcm16),
                gain=0.0,
                agc_verdict=AGC_ABSENT,
                segment_levels=(),
                encoding=fmt.encoding,
                sample_rate=fmt.sample_rate,
                channels=camera_config.channels,
                placement_note=placement_note,
                taken_at=taken_at,
                echo_cancelled=True,
            )
            target = Path(calibration_config.dir) / _timestamped_filename(taken_at)
            calibration.save(target)
            return CalibrationRunResult(calibration=calibration, failure_reason=None)
        if measurement.no_echo and not speaker_has_aec:
            logger.warning(
                "no echo came back for source=%s (confidence=%.3f) from a speaker with no "
                "echo cancellation -- the probe did not reach the microphone, no record saved",
                DEFAULT_SOURCE_NAME,
                measurement.confidence,
            )
        return CalibrationRunResult(calibration=None, failure_reason=measurement.failure_reason)

    taken_at = now()
    calibration = EchoCalibration(
        schema_version=SCHEMA_VERSION,
        probe_format_version=PROBE_FORMAT_VERSION,
        probe_seed=probe_seed,
        source=DEFAULT_SOURCE_NAME,
        delay_s=measurement.delay_s,
        confidence=measurement.confidence,
        echo_level=measurement.echo_level,
        gain=measurement.gain,
        agc_verdict=measurement.agc_verdict,
        segment_levels=measurement.segment_levels,
        encoding=fmt.encoding,
        sample_rate=fmt.sample_rate,
        channels=camera_config.channels,
        placement_note=placement_note,
        taken_at=taken_at,
    )

    target = Path(calibration_config.dir) / _timestamped_filename(taken_at)
    calibration.save(target)
    return CalibrationRunResult(calibration=calibration, failure_reason=None)


def find_latest_calibration(dir_path: str | Path) -> EchoCalibration | None:
    """The most recently taken calibration on disk, or `None` if the
    directory holds none yet.

    "Current" is never a second, separately-written pointer file -- it is
    resolved here, on read, as the lexicographically greatest filename
    under `_FILENAME_GLOB`, which sorts chronologically because
    `_timestamped_filename` always writes the same fixed-width UTC stamp.
    A second successful run therefore "replaces the current record"
    (plan's own words) simply by being the new greatest name; the previous
    file is untouched and still loadable by its own path.
    """
    directory = Path(dir_path)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob(_FILENAME_GLOB))
    if not candidates:
        return None
    return EchoCalibration.load(candidates[-1])
