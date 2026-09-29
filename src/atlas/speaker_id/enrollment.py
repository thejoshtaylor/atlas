"""Admin enrollment: one prompted phrase captured through the real edge
microphone, written to a clip on disk, embedded, and stored (D-02, D-03,
Phase 11, plan 11-07).

`EnrollmentCapture` implements `EdgeAudioListener`
(`transports/edge.py`) -- it never opens a second reader of
`EdgeAudioSource.frames()` (11-RESEARCH.md Pitfall 2, the same rule
`speaker_id/tracker.py`'s `SpeakerTracker` already follows). It listens
through `add_listener` for exactly one capture's lifetime, then detaches.

`ClipStore` is the one place an enrollment clip's path is built -- always
from an integer speaker id and an integer phrase index, resolved and
checked against the enrollment root, never from a display name or any
text a request carries (T-11-23).

`enroll_phrase` is the one function `routes/speakers.py`'s enrollment
route calls: it checks the device, guards against a second concurrent
capture, suppresses wake detection for the capture's duration, writes the
clip, embeds it, stores the embedding, and reloads the member's vectors
into the live `ReferenceSet` so the very next turn can recognize the
newly-recorded phrase.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atlas.audio.channels import select_channel
from atlas.speaker_id.embedding import EmbeddingWorker, SAMPLE_RATE, SpeakerAudioTooShort, SpeakerModelError
from atlas.speaker_id.wiring import ensure_embedding_worker, refresh_live_reference

if TYPE_CHECKING:
    from atlas.config import SpeakerIdConfig
    from atlas.db.speaker_repository import Speaker, SpeakerRepository

logger = logging.getLogger("atlas.speaker_id.enrollment")


class EnrollmentError(Exception):
    """Raised for anything `enroll_phrase` maps directly to an HTTP error
    -- `detail`/`status_code` are exactly what `routes/speakers.py` puts
    in the response, so the route never re-derives wording per case."""

    def __init__(self, detail: str, *, status_code: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class EnrollmentTimeout(EnrollmentError):
    """No `vad.start` arrived within `enrollment_start_timeout_s` of
    attaching -- the admin never started speaking (or the device never
    sent anything)."""

    def __init__(self, detail: str = "no speech was heard in time") -> None:
        super().__init__(detail, status_code=422)


def _too_short_error() -> EnrollmentError:
    return EnrollmentError("the recording was too short", status_code=422)


class EnrollmentCapture:
    """Captures one enrollment phrase from a live `EdgeAudioSource`
    (D-02), following the capture rules `11-07-PLAN.md`'s `<interfaces>`
    states:

    - Ignores a segment already in progress when it attaches -- it starts
      collecting only at the first `vad.start` it sees after that.
    - Keeps the ASR-channel frames from each `vad.start` to its
      `vad.end`. After a `vad.end` it waits `enrollment_gap_ms`; a new
      `vad.start` inside that window continues the same phrase (a pause
      mid-sentence), otherwise the phrase is complete.
    - No `vad.start` within `enrollment_start_timeout_s` of attaching
      raises `EnrollmentTimeout`. Speech longer than
      `enrollment_max_phrase_s` completes the phrase at that length.
    - `speech_ms` is the length of the kept frames; less than
      `min_enrollment_speech_ms` fails with `EnrollmentError` naming the
      recording too short.

    Every `on_*` callback fires synchronously, on the same loop and
    thread `EdgeAudioSource.serve()` runs on (the `EdgeAudioListener`
    contract) -- the same loop `wait()` is awaited from in real use, so
    resolving `self._result` here is always same-thread, never a
    cross-loop hazard.
    """

    def __init__(
        self,
        *,
        channels: int,
        asr_channel: int,
        sample_rate: int,
        enrollment_gap_ms: int,
        enrollment_start_timeout_s: float,
        enrollment_max_phrase_s: float,
        min_enrollment_speech_ms: int,
    ) -> None:
        self._channels = channels
        self._asr_channel = asr_channel
        self._sample_rate = sample_rate
        self._gap_s = enrollment_gap_ms / 1000.0
        self._start_timeout_s = enrollment_start_timeout_s
        self._max_phrase_s = enrollment_max_phrase_s
        self._min_speech_ms = min_enrollment_speech_ms

        self._loop = asyncio.get_running_loop()
        self._result: "asyncio.Future[tuple[bytes, float]]" = self._loop.create_future()
        self._started = False
        self._in_target_segment = False
        self._buffer = bytearray()
        self._start_timeout_handle: "asyncio.TimerHandle | None" = self._loop.call_later(
            self._start_timeout_s, self._on_start_timeout
        )
        self._gap_handle: "asyncio.TimerHandle | None" = None

    # -- internal state machine -----------------------------------------

    def _speech_ms(self) -> float:
        samples = len(self._buffer) // 2  # PCM16, 2 bytes per sample.
        return samples / self._sample_rate * 1000.0

    def _on_start_timeout(self) -> None:
        if not self._result.done():
            self._result.set_exception(EnrollmentTimeout())

    def _on_gap_elapsed(self) -> None:
        self._gap_handle = None
        if not self._result.done():
            self._complete()

    def _complete(self) -> None:
        if self._result.done():
            return
        speech_ms = self._speech_ms()
        if speech_ms < self._min_speech_ms:
            self._result.set_exception(_too_short_error())
            return
        self._result.set_result((bytes(self._buffer), speech_ms))

    # -- EdgeAudioListener protocol ---------------------------------------

    def on_vad_start(self, seq: int, at: float) -> None:
        del seq, at
        if self._result.done() or self._in_target_segment:
            return
        if self._gap_handle is not None:
            self._gap_handle.cancel()
            self._gap_handle = None
        if not self._started:
            self._started = True
            if self._start_timeout_handle is not None:
                self._start_timeout_handle.cancel()
                self._start_timeout_handle = None
        self._in_target_segment = True

    def on_frame(self, chunk: bytes, frame_index: int, at: float) -> None:
        del frame_index, at
        if self._result.done() or not self._in_target_segment:
            return
        asr_chunk = select_channel(chunk, self._channels, self._asr_channel)
        self._buffer.extend(asr_chunk)
        if self._speech_ms() >= self._max_phrase_s * 1000.0:
            self._in_target_segment = False
            self._complete()

    def on_vad_end(self, seq: int, at: float) -> None:
        del seq, at
        if self._result.done() or not self._in_target_segment:
            return
        self._in_target_segment = False
        self._gap_handle = self._loop.call_later(self._gap_s, self._on_gap_elapsed)

    def on_wake_hit(self, frame_index: int) -> None:
        del frame_index  # unused -- enrollment never involves a wake hit.

    # -- public API -------------------------------------------------------

    async def wait(self) -> "tuple[bytes, float]":
        """Await the completed phrase -- `(pcm16_mono, speech_ms)`. Raises
        `EnrollmentTimeout` or `EnrollmentError` (too short) for the
        failure cases above."""
        try:
            return await self._result
        finally:
            self.close()

    def close(self) -> None:
        """Cancel any pending timer -- idempotent, safe to call more than
        once (a `TimerHandle.cancel()` on an already-fired or
        already-cancelled handle is a no-op)."""
        if self._start_timeout_handle is not None:
            self._start_timeout_handle.cancel()
            self._start_timeout_handle = None
        if self._gap_handle is not None:
            self._gap_handle.cancel()
            self._gap_handle = None


class ClipStore:
    """Enrollment clips on disk, under `root` (D-03): `<root>/<speaker_id>/
    phrase-<index>.wav`, mono PCM16. Every path is built only from an
    integer speaker id and an integer phrase index, resolved, and refused
    unless it sits under the resolved root (T-11-23) -- never from a
    display name or any text a request carries."""

    def __init__(self, root: "str | Path") -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def speaker_dir(self, speaker_id: int) -> Path:
        if speaker_id < 0:
            raise ValueError(f"speaker_id must be >= 0, got {speaker_id}")
        candidate = (self._root / str(speaker_id)).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise ValueError(
                f"resolved speaker directory {candidate} escapes the enrollment root {self._root}"
            )
        return candidate

    def clip_path(self, speaker_id: int, phrase_index: int) -> Path:
        if phrase_index < 0:
            raise ValueError(f"phrase_index must be >= 0, got {phrase_index}")
        return self.speaker_dir(speaker_id) / f"phrase-{phrase_index}.wav"

    def write_clip(
        self, speaker_id: int, phrase_index: int, pcm16_mono: bytes, *, sample_rate: int = SAMPLE_RATE
    ) -> None:
        """Write `pcm16_mono` as a mono PCM16 WAV file, via a temporary
        file in the same directory and then an atomic rename (`<interfaces>`)
        -- a reader of `clip_path` never observes a partially-written
        file."""
        speaker_dir = self.speaker_dir(speaker_id)
        speaker_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.clip_path(speaker_id, phrase_index)
        fd, tmp_name = tempfile.mkstemp(dir=speaker_dir, prefix=".tmp-phrase-", suffix=".wav")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with wave.open(str(tmp_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm16_mono)
            tmp_path.replace(final_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def read_clip(self, speaker_id: int, phrase_index: int) -> bytes:
        path = self.clip_path(speaker_id, phrase_index)
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes())

    def list_clips(self, speaker_id: int) -> "list[int]":
        """Every phrase index this member has a clip for, sorted --
        `[]` for a member with no clip directory at all."""
        speaker_dir = self.speaker_dir(speaker_id)
        if not speaker_dir.is_dir():
            return []
        indices: "list[int]" = []
        for path in speaker_dir.glob("phrase-*.wav"):
            try:
                indices.append(int(path.stem.split("-", 1)[1]))
            except (IndexError, ValueError):
                continue
        return sorted(indices)

    def delete_speaker_dir(self, speaker_id: int) -> None:
        """Remove every clip this member has, and the directory itself --
        a no-op for a member with no clip directory. Raises `OSError` if
        removal fails partway (the caller decides what that means, D-03's
        "removed at the next start" contract)."""
        import shutil

        speaker_dir = self.speaker_dir(speaker_id)
        if speaker_dir.is_dir():
            shutil.rmtree(speaker_dir)


@dataclass(frozen=True)
class EnrollmentResult:
    speaker_id: int
    phrase_index: int
    speech_ms: float


def _edge_not_configured_error() -> EnrollmentError:
    return EnrollmentError("the edge microphone is not the configured audio source", status_code=409)


def _edge_device_not_connected_error() -> EnrollmentError:
    return EnrollmentError("that edge device is not connected", status_code=409)


def _model_not_set_error() -> EnrollmentError:
    return EnrollmentError("speaker_id.model is not set", status_code=409)


def _model_missing_error(exc: "SpeakerModelError") -> EnrollmentError:
    return EnrollmentError(f"the speaker embedding model file is missing: {exc}", status_code=409)


def _already_recording_error() -> EnrollmentError:
    return EnrollmentError("another enrollment is already recording", status_code=409)


async def enroll_phrase(
    *,
    state: Any,
    speaker: "Speaker",
    phrase_index: int,
    device_id: int,
    speaker_repo: "SpeakerRepository",
    speaker_config: "SpeakerIdConfig",
) -> EnrollmentResult:
    """Capture, store, and embed one enrollment phrase (D-02, D-03).

    `state` is `request.app.state` -- duck-typed, the same "read app.state
    directly" convention `app.py`'s own `run_echo_path_calibration` uses
    for its own single in-flight flag. Guards, in order: the edge source
    exists at all (409), `speaker_config.model` is set (409), no other
    enrollment is already recording (409), the named device is the one
    connected (409), and an embedding worker can actually be built (409
    naming a missing model file). Only after every guard passes does this
    function set the in-flight flag and `edge_source.wake_suppressed` --
    both cleared in `finally`, together with detaching the capture
    listener, so a raised guard never leaves either set.
    """
    edge_source = getattr(state, "edge_source", None)
    if edge_source is None:
        raise _edge_not_configured_error()
    if speaker_config.model is None:
        raise _model_not_set_error()
    if getattr(state, "speaker_enrollment_in_progress", False):
        raise _already_recording_error()
    if edge_source.connected_device_id != device_id:
        raise _edge_device_not_connected_error()

    try:
        worker = ensure_embedding_worker(state, speaker_config)
    except SpeakerModelError as exc:
        raise _model_missing_error(exc) from None

    source_format = edge_source.source_format()
    capture = EnrollmentCapture(
        channels=source_format.channels,
        asr_channel=source_format.asr_channel,
        sample_rate=source_format.sample_rate,
        enrollment_gap_ms=speaker_config.enrollment_gap_ms,
        enrollment_start_timeout_s=speaker_config.enrollment_start_timeout_s,
        enrollment_max_phrase_s=speaker_config.enrollment_max_phrase_s,
        min_enrollment_speech_ms=speaker_config.min_enrollment_speech_ms,
    )

    state.speaker_enrollment_in_progress = True
    edge_source.wake_suppressed = True
    unsubscribe = edge_source.add_listener(capture)
    try:
        pcm, speech_ms = await capture.wait()
    finally:
        unsubscribe()
        edge_source.wake_suppressed = False
        state.speaker_enrollment_in_progress = False

    clip_store: "ClipStore" = state.speaker_clip_store
    await asyncio.to_thread(
        clip_store.write_clip, speaker.id, phrase_index, pcm, sample_rate=source_format.sample_rate
    )

    try:
        vector = await worker.embed(pcm)
    except SpeakerAudioTooShort:
        raise _too_short_error() from None

    model_id = speaker_config.model_id
    assert model_id is not None  # guarded above: speaker_config.model is not None.
    await speaker_repo.upsert_embedding(
        speaker_id=speaker.id,
        phrase_index=phrase_index,
        model_id=model_id,
        vector=list(vector),
        created_at=datetime.now(timezone.utc),
    )

    await refresh_live_reference(
        state, speaker_repo, speaker_id=speaker.id, display_name=speaker.display_name, model_id=model_id
    )

    return EnrollmentResult(speaker_id=speaker.id, phrase_index=phrase_index, speech_ms=speech_ms)
