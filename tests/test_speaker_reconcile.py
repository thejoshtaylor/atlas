"""The startup reconcile: orphan clip directories go, and a model change
re-embeds from the stored clips with no new recording (D-03, Phase 11,
plan 11-07).

Most cases here drive `reconcile_enrollment` directly against a `ClipStore`
under `tmp_path` and a `FakeSpeakerRepository` -- no application boot
needed. The one ordering case (`reconcile_enrollment` runs before
`build_speaker_context` loads the live `ReferenceSet`) boots the real
application through `lifespan`.
"""

from __future__ import annotations

import logging
import struct
from datetime import datetime, timezone

import pytest

from atlas.config import SpeakerIdConfig
from atlas.speaker_id.embedding import SpeakerModelError
from atlas.speaker_id.enrollment import ClipStore
from atlas.speaker_id.wiring import reconcile_enrollment

from tests.speaker_fakes import FakeEmbedder
from tests.speaker_repo_fakes import FakeSpeakerRepository, fake_speaker

_CAMPPLUS_MODEL_ID = "3dspeaker_speech_campplus_sv_en_voxceleb_16k"


def _speaker_config(**overrides: object) -> SpeakerIdConfig:
    base = dict(
        mode="record",
        model="campplus",
        threshold=0.5,
        window_ms=500,
        speech_rms_floor=0.01,
        change_similarity_floor=0.3,
    )
    base.update(overrides)
    return SpeakerIdConfig(**base)


def _fake_embedder_factory(config: object) -> FakeEmbedder:
    return FakeEmbedder()


def _pcm_for(value: int, num_samples: int = 250) -> bytes:
    return struct.pack(f"<{num_samples}h", *([value] * num_samples))


async def test_a_clip_directory_with_no_row_is_removed_and_counted(tmp_path):
    clip_store = ClipStore(tmp_path / "speakers")
    clip_store.write_clip(1, 0, _pcm_for(-2000), sample_rate=16000)
    speaker_repo = FakeSpeakerRepository()  # no speaker row 1 at all.

    report = await reconcile_enrollment(
        repo=speaker_repo,
        clip_store=clip_store,
        speaker_config=_speaker_config(),
        embedder_factory=_fake_embedder_factory,
    )

    assert report.orphans_removed == 1
    assert not clip_store.speaker_dir(1).exists()


async def test_a_member_with_clips_and_only_old_model_rows_gets_reembedded_and_old_rows_stay(tmp_path):
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2

    for phrase_index in range(5):
        clip_store.write_clip(1, phrase_index, _pcm_for(-2000 - phrase_index), sample_rate=16000)
        await speaker_repo.upsert_embedding(
            speaker_id=1,
            phrase_index=phrase_index,
            model_id="old-model",
            vector=[0.1, 0.2, 0.3],
            created_at=datetime.now(timezone.utc),
        )

    report = await reconcile_enrollment(
        repo=speaker_repo,
        clip_store=clip_store,
        speaker_config=_speaker_config(model="campplus"),
        embedder_factory=_fake_embedder_factory,
    )

    assert report.orphans_removed == 0
    assert report.phrases_reembedded == 5

    new_rows = await speaker_repo.list_reference_embeddings(_CAMPPLUS_MODEL_ID)
    assert len(new_rows) == 5
    old_rows = await speaker_repo.list_reference_embeddings("old-model")
    assert len(old_rows) == 5  # the old rows stay untouched.


async def test_a_member_already_embedded_under_the_configured_model_is_not_reembedded_again(tmp_path):
    clip_store = ClipStore(tmp_path / "speakers")
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1)
    speaker_repo._next_id = 2
    clip_store.write_clip(1, 0, _pcm_for(-2000), sample_rate=16000)
    await speaker_repo.upsert_embedding(
        speaker_id=1, phrase_index=0, model_id=_CAMPPLUS_MODEL_ID, vector=[0.1, 0.2],
        created_at=datetime.now(timezone.utc),
    )

    report = await reconcile_enrollment(
        repo=speaker_repo,
        clip_store=clip_store,
        speaker_config=_speaker_config(model="campplus"),
        embedder_factory=_fake_embedder_factory,
    )

    assert report.phrases_reembedded == 0
    [row] = await speaker_repo.list_reference_embeddings(_CAMPPLUS_MODEL_ID)
    assert row.vector == [0.1, 0.2]  # untouched -- not overwritten with a fresh embed.


async def test_speaker_id_model_null_still_sweeps_orphans_but_skips_reembedding(tmp_path, caplog):
    clip_store = ClipStore(tmp_path / "speakers")
    clip_store.write_clip(1, 0, _pcm_for(-2000), sample_rate=16000)  # orphan -- no row.
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[2] = fake_speaker(speaker_id=2)  # a real member, no clips.
    speaker_repo._next_id = 3

    with caplog.at_level(logging.INFO, logger="atlas.speaker_id.wiring"):
        report = await reconcile_enrollment(
            repo=speaker_repo,
            clip_store=clip_store,
            speaker_config=_speaker_config(mode="off", model=None),
            embedder_factory=_fake_embedder_factory,
        )

    assert report.orphans_removed == 1
    assert report.phrases_reembedded == 0
    assert any("model is not set" in r.getMessage() for r in caplog.records)


async def test_a_missing_model_file_still_sweeps_orphans_but_skips_reembedding(tmp_path, caplog):
    clip_store = ClipStore(tmp_path / "speakers")
    clip_store.write_clip(1, 0, _pcm_for(-2000), sample_rate=16000)
    speaker_repo = FakeSpeakerRepository()  # no row 1 -- an orphan.
    speaker_repo._next_id = 1

    def _raising_factory(config: object) -> FakeEmbedder:
        raise SpeakerModelError("no speaker embedding model at /models/speaker-id/missing.onnx")

    with caplog.at_level(logging.INFO, logger="atlas.speaker_id.wiring"):
        report = await reconcile_enrollment(
            repo=speaker_repo,
            clip_store=clip_store,
            speaker_config=_speaker_config(),
            embedder_factory=_raising_factory,
        )

    assert report.orphans_removed == 1
    assert report.phrases_reembedded == 0
    assert any("skipping re-embedding" in r.getMessage() for r in caplog.records)


async def test_a_directory_whose_name_is_not_an_integer_is_left_alone_and_logged(tmp_path, caplog):
    root = tmp_path / "speakers"
    root.mkdir()
    stray = root / "not-a-number"
    stray.mkdir()
    (stray / "some-file.txt").write_text("hello", encoding="utf-8")
    clip_store = ClipStore(root)
    speaker_repo = FakeSpeakerRepository()

    with caplog.at_level(logging.INFO, logger="atlas.speaker_id.wiring"):
        report = await reconcile_enrollment(
            repo=speaker_repo,
            clip_store=clip_store,
            speaker_config=_speaker_config(),
            embedder_factory=_fake_embedder_factory,
        )

    assert report.orphans_removed == 0
    assert stray.is_dir()  # left alone.
    assert any("non-numeric" in r.getMessage() for r in caplog.records)


async def test_a_missing_enrollment_root_is_not_an_error_and_is_never_created(tmp_path):
    root = tmp_path / "speakers"  # never created.
    clip_store = ClipStore(root)
    speaker_repo = FakeSpeakerRepository()

    report = await reconcile_enrollment(
        repo=speaker_repo,
        clip_store=clip_store,
        speaker_config=_speaker_config(),
        embedder_factory=_fake_embedder_factory,
    )

    assert report.orphans_removed == 0
    assert report.phrases_reembedded == 0
    assert not root.exists()


# --- Ordering: reconcile runs before build_speaker_context loads the ------
# --- live ReferenceSet -----------------------------------------------------


def test_the_lifespan_boot_reembeds_before_building_the_live_reference_set(tmp_path, monkeypatch):
    """A member with clips but only stale-model rows: if `reconcile_
    enrollment` ran AFTER `build_speaker_context`'s own `list_reference_
    embeddings` read, the live `ReferenceSet` would come up empty for this
    member despite the reconcile succeeding moments later. This boots the
    real application to prove the actual call order, not just that both
    functions individually work."""
    import test_auth_setup
    import test_startup_smoke as smoke
    import atlas.app as app_module
    from atlas.db.repository import Setting

    client = test_auth_setup._boot_with_empty_accounts(tmp_path, monkeypatch)

    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1, display_name="Member A")
    speaker_repo._next_id = 2
    original_build_repositories = app_module._build_repositories

    def _repositories(config: object, engine: object) -> dict:
        repositories = original_build_repositories(config, engine)
        repositories["settings_repo"].settings["audio_source"] = Setting(
            id=1,
            key="audio_source",
            value="edge",
            updated_at=datetime.now(timezone.utc),
            updated_by_user_id=None,
        )
        repositories["speaker_repo"] = speaker_repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _repositories)
    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(
            smoke._write_fake_config(
                tmp_path,
                extra={
                    "edge": {"asr_channel": 1, "pre_roll_ms": 200, "tail_ms": 300},
                    "speaker_id": {
                        "mode": "record",
                        "model": "campplus",
                        "threshold": 0.5,
                        "window_ms": 500,
                        "min_window_ms": 250,
                        "speech_rms_floor": 0.01,
                        "change_similarity_floor": 0.3,
                        "enrollment_dir": str(tmp_path / "speakers"),
                    },
                },
            )
        ),
    )
    monkeypatch.setattr(smoke.plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)

    class _NoOpWakeDetector:
        def process(self, chunk: bytes) -> None:
            return None

        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(app_module, "_build_wake_detector", lambda wake_config: _NoOpWakeDetector())
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "_build_speaker_embedder", _fake_embedder_factory)

    # A clip on disk, but only a stale-model row in Postgres -- reconcile
    # must re-embed it under "campplus" before `build_speaker_context`
    # ever reads `list_reference_embeddings(campplus)`.
    clip_store = ClipStore(tmp_path / "speakers")
    clip_store.write_clip(1, 0, _pcm_for(-2000), sample_rate=16000)
    import asyncio

    asyncio.run(
        speaker_repo.upsert_embedding(
            speaker_id=1, phrase_index=0, model_id="old-model", vector=[0.1, 0.2],
            created_at=datetime.now(timezone.utc),
        )
    )

    with client:
        speaker_id_context = app_module.app.state.speaker_id_context
        assert speaker_id_context is not None
        assert speaker_id_context.references.enrolled_count == 1
