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
from typing import TYPE_CHECKING, Callable

from atlas.config import Config, ConfigError, SpeakerIdConfig
from atlas.speaker_id.embedding import EmbeddingWorker, SpeakerEmbedder, SpeakerModelError
from atlas.speaker_id.matching import ReferenceSet
from atlas.speaker_id.tracker import SpeakerTracker
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext

if TYPE_CHECKING:
    from atlas.db.speaker_repository import SpeakerRepository
    from atlas.transports.edge import EdgeAudioSource

logger = logging.getLogger("atlas.speaker_id.wiring")


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
