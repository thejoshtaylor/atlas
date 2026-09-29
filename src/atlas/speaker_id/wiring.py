"""Boots the speaker-id subsystem once, at `lifespan` (`app.py`'s edge
branch), and hands `run_turn` the one `SpeakerIdTurnContext` every edge
turn reads.

Mode `"off"` loads nothing at all -- no model, no worker, no reference
set -- so a fresh install with `speaker_id.mode: "off"` (D-10's own
default) pays zero extra cost at boot. Everything else (`"record"`/
`"enforce"`) requires the Phase 11 spike's measured fields
(`speaker_id.require_measured()`) and a 16 kHz edge source (both candidate
models are 16 kHz-only, 11-RESEARCH.md Pitfall 4).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from atlas.config import Config, ConfigError, SpeakerIdConfig
from atlas.speaker_id.embedding import EmbeddingWorker, SpeakerEmbedder, SpeakerModelError
from atlas.speaker_id.matching import ReferenceSet
from atlas.speaker_id.tracker import SpeakerTracker
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext

if TYPE_CHECKING:
    from atlas.db.speaker_repository import SpeakerRepository
    from atlas.speaker_id.enrollment import ClipStore
    from atlas.transports.edge import EdgeAudioSource

logger = logging.getLogger("atlas.speaker_id.wiring")


def ensure_embedding_worker(state: Any, speaker_config: SpeakerIdConfig) -> EmbeddingWorker:
    """Return a worker to embed enrollment clips with (plan 11-07).

    The running context's own worker when `speaker_id.mode` already built
    one (`"record"`/`"enforce"`) -- never a second, redundant worker
    thread. Otherwise a lazily-built worker, cached on `state.
    speaker_enrollment_worker` on first use: enrollment must work in mode
    `"off"` too (D-10 -- an operator enrolls members before switching
    modes on), and mode `"off"` builds no worker at all
    (`build_speaker_context` above).
    """
    context = getattr(state, "speaker_id_context", None)
    if context is not None and context.worker is not None:
        return context.worker
    cached = getattr(state, "speaker_enrollment_worker", None)
    if cached is not None:
        return cached
    embedder = state.speaker_embedder_factory(speaker_config)
    worker = EmbeddingWorker(embedder)
    state.speaker_enrollment_worker = worker
    return worker


async def build_speaker_context(
    config: Config,
    *,
    edge_source: "EdgeAudioSource",
    speaker_repo: "SpeakerRepository | None",
    embedder_factory: "Callable[[SpeakerIdConfig], SpeakerEmbedder]",
    clock: "Callable[[], float]" = time.monotonic,
) -> SpeakerIdTurnContext:
    speaker_config = config.speaker_id

    if speaker_config.mode == "off":
        return SpeakerIdTurnContext(
            tracker=None, references=None, mode="off", threshold=0.0, model_id=None, worker=None,
        )

    # D-05/D-18: the same "stop by name, never guess" discipline
    # `EdgeSourceConfig.require_measured()` already establishes -- a live
    # gate must not run against an untuned threshold or window.
    speaker_config.require_measured()

    if config.edge.sample_rate != 16000:
        raise ConfigError(
            f"speaker_id.mode is {speaker_config.mode!r}, but edge.sample_rate is "
            f"{config.edge.sample_rate} -- both speaker embedding models (CAM++, TitaNet-small) "
            "are 16 kHz-only (11-RESEARCH.md Pitfall 4); set edge.sample_rate: 16000, or "
            "speaker_id.mode: \"off\" to disable speaker identification"
        )

    try:
        embedder = embedder_factory(speaker_config)
    except SpeakerModelError as exc:
        raise ConfigError(
            f"speaker_id.mode is {speaker_config.mode!r}, but {exc} -- run "
            "`scripts/fetch_models.py --only speaker-id` before starting with speaker "
            "identification enabled"
        ) from exc

    worker = EmbeddingWorker(embedder)

    references = ReferenceSet()
    if speaker_repo is not None:
        rows = await speaker_repo.list_reference_embeddings(speaker_config.model_id)
        references = ReferenceSet.load(
            [
                {"speaker_id": row.speaker_id, "display_name": row.display_name, "vector": row.vector}
                for row in rows
            ]
        )

    tracker = SpeakerTracker(
        channels=config.edge.channels,
        asr_channel=config.edge.asr_channel,
        sample_rate=config.edge.sample_rate,
        window_ms=speaker_config.window_ms,
        min_window_ms=speaker_config.min_window_ms,
        speech_rms_floor=speaker_config.speech_rms_floor,
        change_similarity_floor=speaker_config.change_similarity_floor,
        worker=worker,
        clock=clock,
    )
    edge_source.add_listener(tracker)

    if speaker_config.mode == "enforce" and references.enrolled_count == 0:
        # D-10: a fresh house with nobody enrolled must not silently block
        # every turn -- `speaker_id/gate.py::evaluate_speaker_gate` degrades
        # this exact condition to "record" turn by turn; this warning is
        # the one-time, by-name startup notice an operator reads once,
        # not a per-turn log line.
        logger.warning(
            "speaker_id.mode is 'enforce' but no household member is enrolled -- acting as "
            "'record' (no turn is blocked) until at least one member is enrolled"
        )

    return SpeakerIdTurnContext(
        tracker=tracker,
        references=references,
        mode=speaker_config.mode,
        threshold=speaker_config.threshold,
        model_id=speaker_config.model_id,
        worker=worker,
    )


@dataclass(frozen=True)
class ReconcileReport:
    """One `reconcile_enrollment` run's outcome (D-03) -- logged as two
    counts only, never a member id or a directory name (`app.py`'s own
    `lifespan` call site)."""

    orphans_removed: int
    phrases_reembedded: int


async def reconcile_enrollment(
    *,
    repo: "SpeakerRepository",
    clip_store: "ClipStore",
    speaker_config: SpeakerIdConfig,
    embedder_factory: "Callable[[SpeakerIdConfig], SpeakerEmbedder]",
) -> ReconcileReport:
    """Run at every start, before either audio source branch (plan
    11-07, D-03): remove every enrollment clip directory whose member no
    longer has a row (an interrupted delete, T-11-26), then re-embed every
    remaining member's stored clips under the currently configured model
    if they are not already embedded under it -- a model change needs no
    new recording.

    A missing enrollment root is not an error, and this function never
    creates one -- there is nothing to reconcile until the first clip is
    ever written. A directory whose name does not parse as an integer is
    left alone and logged, never touched (a stray file an operator placed
    there is not this function's business). With `speaker_config.model`
    unset, or the model file missing, the orphan sweep still runs in
    full -- only the re-embedding half is skipped, with one info line
    naming why.
    """
    root = clip_store.root
    if not root.is_dir():
        return ReconcileReport(orphans_removed=0, phrases_reembedded=0)

    orphans_removed = 0
    remaining_speaker_ids: "list[int]" = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            speaker_id = int(entry.name)
        except ValueError:
            logger.info(
                "speaker enrollment reconcile: leaving non-numeric directory %r under the "
                "enrollment root alone",
                entry.name,
            )
            continue
        speaker = await repo.get_speaker(speaker_id)
        if speaker is None:
            clip_store.delete_speaker_dir(speaker_id)
            orphans_removed += 1
            continue
        remaining_speaker_ids.append(speaker_id)

    if speaker_config.model is None:
        logger.info("speaker enrollment reconcile: speaker_id.model is not set -- skipping re-embedding")
        return ReconcileReport(orphans_removed=orphans_removed, phrases_reembedded=0)

    try:
        embedder = embedder_factory(speaker_config)
    except SpeakerModelError as exc:
        logger.info("speaker enrollment reconcile: %s -- skipping re-embedding", exc)
        return ReconcileReport(orphans_removed=orphans_removed, phrases_reembedded=0)

    model_id = speaker_config.model_id
    assert model_id is not None  # guarded above: speaker_config.model is not None.
    phrases_reembedded = 0
    worker = EmbeddingWorker(embedder)
    try:
        existing_counts = await repo.count_embeddings(model_id)
        for speaker_id in remaining_speaker_ids:
            if existing_counts.get(speaker_id, 0) > 0:
                continue
            for phrase_index in clip_store.list_clips(speaker_id):
                pcm = clip_store.read_clip(speaker_id, phrase_index)
                vector = await worker.embed(pcm)
                await repo.upsert_embedding(
                    speaker_id=speaker_id,
                    phrase_index=phrase_index,
                    model_id=model_id,
                    vector=list(vector),
                    created_at=datetime.now(timezone.utc),
                )
                phrases_reembedded += 1
    finally:
        worker.close()

    return ReconcileReport(orphans_removed=orphans_removed, phrases_reembedded=phrases_reembedded)
