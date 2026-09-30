"""`build_speaker_context` (11-04-PLAN.md Task 3): mode `"off"` builds
nothing, a missing model or an unmeasured field or an 8kHz edge source
each refuse the boot by name, and `"enforce"` with zero enrolled members
logs exactly one startup warning.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from atlas.speaker_id.embedding import SpeakerModelError
from atlas.speaker_id.wiring import build_speaker_context

from tests.speaker_repo_fakes import FakeSpeakerRepository, fake_speaker
from tests.test_config import _minimal_raw_config


class _FakeEdgeSource:
    """The one method `build_speaker_context` needs from a real
    `EdgeAudioSource`: `add_listener`."""

    def __init__(self) -> None:
        self.listeners: "list[object]" = []

    def add_listener(self, listener: object):
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)


class _DummyEmbedder:
    dim = 4

    def embed(self, pcm16_mono: bytes):
        raise NotImplementedError("this test never actually embeds anything")


def _config(*, speaker_id: dict, edge: "dict | None" = None) -> "Config":
    """Imports `Config` fresh, at call time, rather than at module load --
    `tests/test_config.py::test_config_and_turn_macros_import_in_either_order`
    (collected before this file, alphabetically) does a real
    `importlib.reload(atlas.config)` mid-suite, which replaces every class
    `atlas.config` defines with a new object of the same name. A name bound
    once at this file's own import time would go stale the moment that
    reload runs, so any `isinstance`/`pytest.raises` check against it later
    in the same session would silently stop matching. A fresh `from ...
    import` here always resolves against whichever class object is
    currently live."""
    from atlas.config import Config

    raw = _minimal_raw_config()
    raw["speaker_id"] = speaker_id
    if edge is not None:
        raw["edge"] = edge
    return Config.from_config(raw)


def _assert_is_config_error(exc: BaseException) -> None:
    """Checked by class NAME, not `isinstance`: `atlas.config`'s own
    `ConfigError` and `atlas.speaker_id.wiring`'s imported reference to it
    can be two DIFFERENT class objects of the same name in this same
    session -- `tests/test_config.py::
    test_config_and_turn_macros_import_in_either_order` (collected before
    this file) does a real `importlib.reload(atlas.config)`, which rebinds
    every class `atlas.config` defines to a new object; a module that
    imported the old one before that reload (this file's own
    `build_speaker_context` import, at collection time, unavoidably before
    any test -- including that reload -- ever runs) keeps raising with the
    old one forever, while `atlas.config`'s own methods raise with
    whichever is current. Neither `isinstance` check can see both; a name
    check needs no identity at all."""
    assert type(exc).__name__ == "ConfigError", f"expected a ConfigError, got {type(exc)!r}: {exc}"


_MEASURED_SPEAKER_ID = {
    "mode": "record",
    "model": "campplus",
    "threshold": 0.5,
    "window_ms": 500,
    "speech_rms_floor": 0.01,
    "change_similarity_floor": 0.3,
}


async def test_mode_off_builds_no_embedder_and_attaches_no_listener():
    config = _config(speaker_id={"mode": "off"})
    edge_source = _FakeEdgeSource()
    factory_calls: "list[object]" = []

    def factory(speaker_config):
        factory_calls.append(speaker_config)
        raise AssertionError("embedder_factory must never be called for mode off")

    context = await build_speaker_context(config, edge_source=edge_source, speaker_repo=None, embedder_factory=factory)

    assert context.tracker is None
    assert context.worker is None
    assert context.references is None
    assert context.mode == "off"
    assert factory_calls == []
    assert edge_source.listeners == []


async def test_a_missing_model_file_raises_config_error_naming_the_path_and_the_fetch_command():
    config = _config(speaker_id=_MEASURED_SPEAKER_ID)
    edge_source = _FakeEdgeSource()

    def factory(speaker_config):
        raise SpeakerModelError(f"no speaker embedding model at {speaker_config.model_path}")

    with pytest.raises(Exception) as exc_info:
        await build_speaker_context(config, edge_source=edge_source, speaker_repo=None, embedder_factory=factory)

    _assert_is_config_error(exc_info.value)
    message = str(exc_info.value)
    assert config.speaker_id.model_path in message
    assert "--only speaker-id" in message


async def test_a_null_measured_key_raises_the_require_measured_error_before_building_anything():
    config = _config(speaker_id={"mode": "record", "model": "campplus"})  # threshold/window_ms/etc left null.
    edge_source = _FakeEdgeSource()
    factory_calls: "list[object]" = []

    def factory(speaker_config):
        factory_calls.append(speaker_config)
        return _DummyEmbedder()

    with pytest.raises(Exception) as exc_info:
        await build_speaker_context(config, edge_source=edge_source, speaker_repo=None, embedder_factory=factory)

    _assert_is_config_error(exc_info.value)
    assert "threshold" in str(exc_info.value)
    assert factory_calls == []
    assert edge_source.listeners == []


async def test_edge_sample_rate_8000_raises_config_error_before_building_anything():
    config = _config(speaker_id=_MEASURED_SPEAKER_ID, edge={"sample_rate": 8000})
    edge_source = _FakeEdgeSource()
    factory_calls: "list[object]" = []

    def factory(speaker_config):
        factory_calls.append(speaker_config)
        return _DummyEmbedder()

    with pytest.raises(Exception) as exc_info:
        await build_speaker_context(config, edge_source=edge_source, speaker_repo=None, embedder_factory=factory)

    _assert_is_config_error(exc_info.value)
    assert "16" in str(exc_info.value)  # names the 16 kHz requirement.
    assert factory_calls == []
    assert edge_source.listeners == []


async def test_enforce_with_zero_enrolled_members_logs_exactly_one_warning(caplog):
    config = _config(speaker_id={**_MEASURED_SPEAKER_ID, "mode": "enforce"})
    edge_source = _FakeEdgeSource()

    def factory(speaker_config):
        return _DummyEmbedder()

    with caplog.at_level(logging.WARNING, logger="atlas.speaker_id.wiring"):
        context = await build_speaker_context(
            config, edge_source=edge_source, speaker_repo=None, embedder_factory=factory
        )

    assert context.tracker is not None
    assert context.references.enrolled_count == 0
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "enforce" in warnings[0].message
    assert edge_source.listeners == [context.tracker]


async def test_enforce_with_an_enrolled_member_logs_no_warning_and_loads_references(caplog):
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1, display_name="Member A")
    speaker_repo._next_id = 2
    model_id = "3dspeaker_speech_campplus_sv_en_voxceleb_16k"
    await speaker_repo.upsert_embedding(
        speaker_id=1, phrase_index=0, model_id=model_id, vector=[1.0, 0.0], created_at=datetime.now(timezone.utc),
    )

    config = _config(speaker_id={**_MEASURED_SPEAKER_ID, "mode": "enforce"})
    edge_source = _FakeEdgeSource()

    def factory(speaker_config):
        return _DummyEmbedder()

    with caplog.at_level(logging.WARNING, logger="atlas.speaker_id.wiring"):
        context = await build_speaker_context(
            config, edge_source=edge_source, speaker_repo=speaker_repo, embedder_factory=factory
        )

    assert context.references.enrolled_count == 1
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []


@pytest.mark.parametrize(("mode", "expected"), [("enforce", {1}), ("record", {1}), ("off", set())])
async def test_the_context_loads_the_members_without_home_control(mode, expected):
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1, display_name="Member A", can_control_home=False)
    speaker_repo._speakers[2] = fake_speaker(speaker_id=2, display_name="Member B")
    speaker_repo._next_id = 3
    config = _config(speaker_id={**_MEASURED_SPEAKER_ID, "mode": mode})

    context = await build_speaker_context(
        config,
        edge_source=_FakeEdgeSource(),
        speaker_repo=speaker_repo,
        embedder_factory=lambda speaker_config: _DummyEmbedder(),
    )

    assert context.home_control_denied == expected
