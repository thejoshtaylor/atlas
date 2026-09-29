"""Retroactive clips from recorded edge turns (quick task 260929-j08).

An operator can add a voice the assistant did not recognize to a member,
from the recording of a turn that already happened. The clip is a normal
`ClipStore` clip at phrase index 100 or more. The prompted phrases keep
the indices 0 to 4. Clips on disk are the truth (D-03): the embedding rows
and the live `ReferenceSet` follow them.

A sidecar `phrase-<n>.json` next to each WAV holds the source session id.
The set of ids in all sidecars is the set of turns already assigned, so
no database column is needed and a model change cannot lose it.

A speaker label never grants permission (D-15). Nothing here reaches
policy, the denylist, or the brain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atlas.audio.channels import select_channel
from atlas.audio.energy import rms_amplitude
from atlas.speaker_id.embedding import (
    SAMPLE_RATE,
    EmbeddingWorker,
    SpeakerAudioTooShort,
    SpeakerModelError,
)
from atlas.speaker_id.enrollment import (
    ClipStore,
    _model_missing_error,
    _model_not_set_error,
    _too_short_error,
)
from atlas.speaker_id.wiring import ensure_embedding_worker, refresh_live_reference

if TYPE_CHECKING:
    from atlas.config import SpeakerIdConfig
    from atlas.db.speaker_repository import Speaker, SpeakerRepository

logger = logging.getLogger("atlas.speaker_id.retroactive")

# Prompted phrases use the indices 0 to 4. A retroactive clip starts here.
RETROACTIVE_MIN_INDEX = 100
# A member keeps at most this many retroactive clips. The next one drops the oldest.
RETROACTIVE_CAP = 20

_BYTES_PER_SAMPLE = 2


def trim_turn_speech(
    raw: bytes,
    *,
    channels: int,
    asr_channel: int,
    frame_samples: int,
    sample_rate: int,
    speech_rms_floor: "float | None",
    max_seconds: float,
) -> "tuple[bytes, float]":
    """Cut the ASR channel of a recorded edge turn to its speech.

    Session events do not map to audio bytes, so this does not use them.
    It walks the audio in slices of `frame_samples` frames. It keeps a
    slice when its RMS reaches `speech_rms_floor`, the same rule as the
    live accumulator. It keeps every slice when the floor is `None`. It
    then cuts the result to `max_seconds`.

    The clip keeps the wake phrase. The live gate embeds the wake-hit
    segment, so the clip then matches what the gate hears. The operator
    listens to the clip before an assign.

    Returns mono PCM16 and its length in milliseconds.
    """
    frame_bytes = channels * _BYTES_PER_SAMPLE
    usable = raw[: len(raw) - len(raw) % frame_bytes]
    step = frame_samples * frame_bytes
    kept: "list[bytes]" = []
    for start in range(0, len(usable), step):
        mono = select_channel(usable[start : start + step], channels, asr_channel)
        if speech_rms_floor is None or rms_amplitude(mono) >= speech_rms_floor:
            kept.append(mono)
    pcm = b"".join(kept)[: int(max_seconds * sample_rate) * _BYTES_PER_SAMPLE]
    return pcm, len(pcm) / _BYTES_PER_SAMPLE / sample_rate * 1000.0


def is_edge_turn(audio_format: Any, speaker_event: "dict | None", *, edge_channels: int) -> bool:
    """True when the turn came from the edge microphone at 16 kHz.

    The camera records A-law at 8 kHz. A browser turn has one channel.
    Neither can join a reference set.
    """
    if not isinstance(audio_format, dict):
        return False
    if audio_format.get("encoding") != "pcm":
        return False
    if audio_format.get("sample_rate") != SAMPLE_RATE:
        return False
    if audio_format.get("channels") != edge_channels:
        return False
    return speaker_event is None or speaker_event.get("detail") != "not_edge_source"


# -- sidecar files ----------------------------------------------------------


def _sidecar_path(clip_store: ClipStore, speaker_id: int, index: int) -> Path:
    # Built from the clip path, so it stays under the store root.
    return clip_store.clip_path(speaker_id, index).with_suffix(".json")


def write_sidecar(clip_store: ClipStore, speaker_id: int, index: int, *, session_id: str, speech_ms: float) -> None:
    target = _sidecar_path(clip_store, speaker_id, index)
    payload = {
        "session_id": session_id,
        "speech_ms": speech_ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-phrase-", suffix=".json")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        tmp_path.replace(target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_sidecar(clip_store: ClipStore, speaker_id: int, index: int) -> "dict | None":
    try:
        payload = json.loads(_sidecar_path(clip_store, speaker_id, index).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def remove_clip_files(clip_store: ClipStore, speaker_id: int, index: int) -> None:
    clip_store.clip_path(speaker_id, index).unlink(missing_ok=True)
    _sidecar_path(clip_store, speaker_id, index).unlink(missing_ok=True)


def assigned_session_ids(clip_store: ClipStore) -> "set[str]":
    """Every session id that a sidecar names, across all members."""
    if not clip_store.root.is_dir():
        return set()
    session_ids: "set[str]" = set()
    for path in clip_store.root.glob("*/phrase-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("session_id"), str):
            session_ids.add(payload["session_id"])
    return session_ids


def retroactive_indices(clip_store: ClipStore, speaker_id: int) -> "list[int]":
    return [index for index in clip_store.list_clips(speaker_id) if index >= RETROACTIVE_MIN_INDEX]


def next_retroactive_index(indices: "list[int]") -> int:
    return max(indices, default=RETROACTIVE_MIN_INDEX - 1) + 1


# -- store ------------------------------------------------------------------


def resolve_embedding_worker(state: Any, speaker_config: "SpeakerIdConfig") -> EmbeddingWorker:
    """The worker that embeds a clip. Raises an `EnrollmentError` (409)
    when `speaker_id.model` is not set or its model file is missing."""
    if speaker_config.model is None:
        raise _model_not_set_error()
    try:
        return ensure_embedding_worker(state, speaker_config)
    except SpeakerModelError as exc:
        raise _model_missing_error(exc) from None


def retroactive_lock(state: Any) -> asyncio.Lock:
    """One lock for every write to retroactive clips. It stops two assigns
    from taking the same index or the same turn."""
    lock = getattr(state, "speaker_retroactive_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        state.speaker_retroactive_lock = lock
    return lock


@dataclass(frozen=True)
class RetroactiveAssignResult:
    speaker_id: int
    phrase_index: int
    speech_ms: float
    dropped_phrase_indices: "tuple[int, ...]" = ()


@dataclass(frozen=True)
class RetroactiveClip:
    """One retroactive clip. Each field but the index is `None` when the
    sidecar is missing or unreadable."""

    phrase_index: int
    session_id: "str | None"
    speech_ms: "float | None"
    created_at: "str | None"


def list_retroactive_clips(clip_store: ClipStore, speaker_id: int) -> "list[RetroactiveClip]":
    """The retroactive clips of one member, oldest index first."""
    clips: "list[RetroactiveClip]" = []
    for index in retroactive_indices(clip_store, speaker_id):
        sidecar = read_sidecar(clip_store, speaker_id, index) or {}
        session_id = sidecar.get("session_id")
        speech_ms = sidecar.get("speech_ms")
        created_at = sidecar.get("created_at")
        clips.append(
            RetroactiveClip(
                phrase_index=index,
                session_id=session_id if isinstance(session_id, str) else None,
                speech_ms=float(speech_ms) if isinstance(speech_ms, (int, float)) else None,
                created_at=created_at if isinstance(created_at, str) else None,
            )
        )
    return clips


async def store_retroactive_clip(
    *,
    state: Any,
    clip_store: ClipStore,
    speaker: "Speaker",
    session_id: str,
    pcm16_mono: bytes,
    speech_ms: float,
    worker: Any,
    speaker_repo: "SpeakerRepository",
    model_id: str,
) -> RetroactiveAssignResult:
    """Write, embed, and store one retroactive clip. The caller holds
    `retroactive_lock`.

    Order: WAV, embedding row, sidecar, cap drop, live reference. When the
    embed or the row fails, the new files and row go away. The cap drop
    removes the oldest clips beyond `RETROACTIVE_CAP` in this order: row,
    live reference, files. A crash at worst leaves an orphan WAV that the
    clip list shows and the operator can remove. This uses no microphone,
    so it does not touch `speaker_enrollment_in_progress` or
    `wake_suppressed`.
    """
    index = next_retroactive_index(await asyncio.to_thread(retroactive_indices, clip_store, speaker.id))
    await asyncio.to_thread(clip_store.write_clip, speaker.id, index, pcm16_mono, sample_rate=SAMPLE_RATE)
    try:
        vector = await worker.embed(pcm16_mono)
        await speaker_repo.upsert_embedding(
            speaker_id=speaker.id,
            phrase_index=index,
            model_id=model_id,
            vector=list(vector),
            created_at=datetime.now(timezone.utc),
        )
        await asyncio.to_thread(
            write_sidecar, clip_store, speaker.id, index, session_id=session_id, speech_ms=speech_ms
        )
    except SpeakerAudioTooShort:
        await _discard_new_clip(clip_store, speaker_repo, speaker.id, index)
        raise _too_short_error() from None
    except Exception:
        await _discard_new_clip(clip_store, speaker_repo, speaker.id, index)
        raise

    # Only indices from RETROACTIVE_MIN_INDEX up are ever in this list, so a
    # prompted phrase can never be dropped.
    kept = sorted({*await asyncio.to_thread(retroactive_indices, clip_store, speaker.id), index})
    dropped = tuple(kept[: max(0, len(kept) - RETROACTIVE_CAP)])
    for dropped_index in dropped:
        await speaker_repo.delete_embedding(speaker_id=speaker.id, phrase_index=dropped_index)
    await refresh_live_reference(
        state, speaker_repo, speaker_id=speaker.id, display_name=speaker.display_name, model_id=model_id
    )
    for dropped_index in dropped:
        await asyncio.to_thread(remove_clip_files, clip_store, speaker.id, dropped_index)
    logger.info(
        "retroactive clip stored: speaker %s, phrase %s, session %s, dropped %s",
        speaker.id,
        index,
        session_id,
        len(dropped),
    )
    return RetroactiveAssignResult(
        speaker_id=speaker.id, phrase_index=index, speech_ms=speech_ms, dropped_phrase_indices=dropped
    )


async def _discard_new_clip(
    clip_store: ClipStore, speaker_repo: "SpeakerRepository", speaker_id: int, index: int
) -> None:
    """Undo a failed store: the files first, then the row when it exists."""
    await asyncio.to_thread(remove_clip_files, clip_store, speaker_id, index)
    try:
        await speaker_repo.delete_embedding(speaker_id=speaker_id, phrase_index=index)
    except Exception:
        logger.warning("retroactive clip: could not remove the row for speaker %s, phrase %s", speaker_id, index)


async def delete_retroactive_clip(
    *,
    state: Any,
    clip_store: ClipStore,
    speaker: "Speaker",
    phrase_index: int,
    speaker_repo: "SpeakerRepository",
    model_id: "str | None",
) -> bool:
    """Remove one retroactive clip: the row, the live reference, then the
    files. Return `False` when neither the WAV nor the sidecar exists. The
    caller holds `retroactive_lock`. The caller must also refuse an index
    under `RETROACTIVE_MIN_INDEX`."""

    def _exists() -> bool:
        return (
            clip_store.clip_path(speaker.id, phrase_index).exists()
            or _sidecar_path(clip_store, speaker.id, phrase_index).exists()
        )

    if not await asyncio.to_thread(_exists):
        return False
    await speaker_repo.delete_embedding(speaker_id=speaker.id, phrase_index=phrase_index)
    if model_id is not None:
        await refresh_live_reference(
            state, speaker_repo, speaker_id=speaker.id, display_name=speaker.display_name, model_id=model_id
        )
    await asyncio.to_thread(remove_clip_files, clip_store, speaker.id, phrase_index)
    logger.info("retroactive clip removed: speaker %s, phrase %s", speaker.id, phrase_index)
    return True
