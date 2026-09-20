"""The provider registries and the boot resolver, against fakes -- no
network, no real credential, no real Postgres (D-01, D-03, D-04, PROV-01).
"""

from __future__ import annotations

import pytest

from spire_voice.config import BrainConfig, BrainTierConfig, SttConfig, TtsConfig
from spire_voice.providers import registry
from spire_voice.providers.batch_tts_adapter import BatchTtsAdapter
from spire_voice.providers.boot import ProviderSlotStatus, ProviderUnavailable, resolve_slot


def _stt_config() -> SttConfig:
    return SttConfig(url="wss://stt.invalid/v1/stt")


def _tts_config() -> TtsConfig:
    return TtsConfig(url="https://tts.invalid/v1/tts")


def _brain_config() -> BrainConfig:
    return BrainConfig(models=(BrainTierConfig(model="fake-model"),))


def test_build_stt_builds_the_registered_xai_provider():
    from spire_voice.providers.stt_xai import XaiStt

    client = registry.build_stt("xai", _stt_config(), "test-key")

    assert isinstance(client, XaiStt)


def test_build_stt_raises_naming_the_value_and_the_known_set():
    with pytest.raises(registry.ConfigError) as excinfo:
        registry.build_stt("does-not-exist", _stt_config(), "test-key")

    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "xai" in message


def test_build_stt_builds_the_registered_local_provider_with_no_credential(tmp_path, monkeypatch):
    """D-04's needs-a-credential gate is opt-in per entry (`requires_credential`)
    -- the local entry must build with an empty api_key, never raise
    `ProviderUnavailable` for a "missing" credential it never needed."""
    from spire_voice.providers.stt_faster_whisper import FasterWhisperStt

    model_dir = tmp_path / "faster-whisper"
    model_dir.mkdir()
    config = SttConfig(local_model_dir=str(model_dir))
    # `registry.build_stt` calls the entry's own `build(config, api_key)`
    # unmodified, which always goes through the real (module-scope-default)
    # loader -- substitute `faster_whisper.WhisperModel` itself so no real
    # model weights need to exist on disk for this test.
    monkeypatch.setattr("faster_whisper.WhisperModel", lambda *a, **kw: object())

    client = registry.build_stt("faster-whisper", config, "")

    assert isinstance(client, FasterWhisperStt)


def test_known_entries_returns_every_registered_option_for_the_slot():
    """Sorted by name (07-03-PLAN.md Task 2 adds `faster-whisper`
    alongside `xai`) -- `known_entries` never returns registration order."""
    entries = registry.known_entries("stt")

    assert [entry.name for entry in entries] == ["faster-whisper", "xai"]
    by_name = {entry.name: entry for entry in entries}
    assert by_name["xai"].requires_credential is True
    assert by_name["xai"].batch is False
    assert by_name["faster-whisper"].requires_credential is False
    assert by_name["faster-whisper"].batch is False


def test_build_tts_builds_the_registered_xai_provider():
    from spire_voice.providers.tts_xai import XaiTts

    client = registry.build_tts("xai", _tts_config(), "test-key")

    assert isinstance(client, XaiTts)


def test_tts_entry_declares_itself_batch():
    """D-05/D-09: xAI's text-to-speech and the local Piper entry are both
    a single call, not a stream -- batch -- and both registry entries must
    say so, since `boot.py::resolve_slot` reads exactly this field to
    decide whether to wrap."""
    entries = registry.known_entries("tts")

    assert {entry.name for entry in entries} == {"xai", "piper"}
    assert all(entry.batch is True for entry in entries)


def test_piper_entry_requires_no_credential_and_carries_the_licence_note():
    """D-10: the local entry's needs-a-credential hint must never appear,
    and its GPL-3.0 disclosure is data on the entry, not a special case
    the route or the component would otherwise have to hardcode."""
    entry = registry.TTS_REGISTRY["piper"]

    assert entry.requires_credential is False
    assert entry.licence_note is not None
    assert "GPL-3.0" in entry.licence_note
    # xAI carries no licence obligation -- only Piper's entry states one.
    assert registry.TTS_REGISTRY["xai"].licence_note is None


def test_build_tts_raises_provider_unavailable_when_the_credential_is_missing():
    with pytest.raises(ProviderUnavailable) as excinfo:
        registry.build_tts("xai", _tts_config(), "")

    message = str(excinfo.value)
    assert "xAI" in message
    assert "Settings" in message


def test_build_brain_builds_the_tier_tuple_build_tiers_produces():
    """D-01, D-03: the language-model entry's factory is `brain_race.
    build_tiers` itself, called with the credential-substituted config --
    not a second, parallel construction of the same tiers."""
    from spire_voice.turn.brain_race import TierBrain

    tiers = registry.build_brain("xai", _brain_config(), "test-key")

    assert len(tiers) == 1
    assert isinstance(tiers[0], TierBrain)
    assert tiers[0].model == "fake-model"


def test_build_brain_raises_provider_unavailable_when_the_credential_is_missing():
    with pytest.raises(ProviderUnavailable) as excinfo:
        registry.build_brain("xai", _brain_config(), "")

    message = str(excinfo.value)
    assert "xAI" in message
    assert "Settings" in message


def test_stt_and_brain_entries_are_not_batch():
    stt_entries = registry.known_entries("stt")
    [brain_entry] = registry.known_entries("brain")

    assert all(entry.batch is False for entry in stt_entries)
    assert brain_entry.batch is False


def test_known_names_is_sorted():
    assert registry.known_names("stt") == sorted(registry.known_names("stt"))


def test_build_stt_raises_provider_unavailable_when_the_credential_is_missing():
    """D-04: a registered provider that needs a credential nobody has set
    must not construct a client at all -- the boot resolver's own degraded
    path (`resolve_slot`) depends on this raising `ProviderUnavailable`,
    never on a client silently built with an empty API key."""
    with pytest.raises(ProviderUnavailable) as excinfo:
        registry.build_stt("xai", _stt_config(), "")

    message = str(excinfo.value)
    assert "xAI" in message
    assert "Settings" in message


def test_build_stt_with_a_credential_present_builds_normally():
    from spire_voice.providers.stt_xai import XaiStt

    client = registry.build_stt("xai", _stt_config(), "a-real-looking-key")
    assert isinstance(client, XaiStt)


class _FakeSelectionRepo:
    def __init__(self, selection=None) -> None:
        self._selection = selection

    async def get_selection(self, slot: str):
        return self._selection


class _Selection:
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name


@pytest.mark.asyncio
async def test_resolve_slot_falls_back_to_the_default_name_when_no_row_exists():
    repo = _FakeSelectionRepo(selection=None)

    status, client = await resolve_slot(
        "stt", repo, "xai", registry.build_stt, _stt_config(), "test-key"
    )

    assert status == ProviderSlotStatus(
        slot="stt", selected="xai", active="xai", state="running", reason=None, wrapped=False
    )
    assert client is not None


@pytest.mark.asyncio
async def test_resolve_slot_reads_the_stored_selection_when_one_exists():
    repo = _FakeSelectionRepo(selection=_Selection(provider_name="xai"))

    status, client = await resolve_slot(
        "stt", repo, "some-other-default", registry.build_stt, _stt_config(), "test-key"
    )

    assert status.selected == "xai"
    assert status.active == "xai"
    assert status.state == "running"
    assert client is not None


@pytest.mark.asyncio
async def test_resolve_slot_reports_degraded_when_the_builder_raises_provider_unavailable():
    repo = _FakeSelectionRepo(selection=_Selection(provider_name="xai"))

    def _always_unavailable(name, config, api_key):
        raise ProviderUnavailable("xAI needs an API key -- add one in Settings.")

    status, client = await resolve_slot(
        "stt", repo, "xai", _always_unavailable, _stt_config(), ""
    )

    assert status.state == "degraded"
    assert status.active is None
    assert status.reason == "xAI needs an API key -- add one in Settings."
    assert client is None


@pytest.mark.asyncio
async def test_resolve_slot_wraps_the_client_when_the_resolved_entry_is_batch():
    """Task 2's own must-have: resolving the text-to-speech slot returns a
    client already wrapped, and the status reports `wrapped=True` --
    `boot.py::resolve_slot` is the one place this wrap happens, never a
    second wrap inside `registry.py` or `tts_xai.py`."""
    repo = _FakeSelectionRepo(selection=None)

    status, client = await resolve_slot(
        "tts",
        repo,
        "xai",
        registry.build_tts,
        _tts_config(),
        "test-key",
        is_batch=lambda name: registry.TTS_REGISTRY[name].batch,
    )

    assert isinstance(client, BatchTtsAdapter)
    assert status.wrapped is True


@pytest.mark.asyncio
async def test_resolve_slot_does_not_wrap_when_the_resolved_entry_is_not_batch():
    repo = _FakeSelectionRepo(selection=None)

    status, client = await resolve_slot(
        "stt",
        repo,
        "xai",
        registry.build_stt,
        _stt_config(),
        "test-key",
        is_batch=lambda name: registry.STT_REGISTRY[name].batch,
    )

    assert not isinstance(client, BatchTtsAdapter)
    assert status.wrapped is False


@pytest.mark.asyncio
async def test_resolve_slot_wrapped_defaults_false_with_no_is_batch_argument():
    """Every caller that predates `is_batch` (the stt call site) keeps
    reading `wrapped` off the client itself, unchanged."""
    repo = _FakeSelectionRepo(selection=_Selection(provider_name="xai"))

    status, client = await resolve_slot(
        "stt", repo, "xai", registry.build_stt, _stt_config(), "test-key"
    )

    assert status.wrapped is False
    assert not isinstance(client, BatchTtsAdapter)


@pytest.mark.asyncio
async def test_resolve_slot_propagates_any_other_exception_and_stops_the_boot():
    repo = _FakeSelectionRepo(selection=_Selection(provider_name="not-a-real-provider"))

    with pytest.raises(registry.ConfigError):
        await resolve_slot(
            "stt", repo, "xai", registry.build_stt, _stt_config(), "test-key"
        )


def test_every_slot_registry_is_non_empty():
    """A module-level fact, not merely a test-time one: `07-UI-SPEC.md`'s
    no-empty-state screen assumption depends on this holding for real --
    an empty registry is an authoring mistake that would otherwise surface
    as a blank screen with no error at all."""
    for slot in ("stt", "tts", "brain"):
        assert registry.known_names(slot), f"{slot} registry must not be empty"


def test_registries_non_empty_check_raises_on_an_empty_registry(monkeypatch):
    """The self-check itself, proven to actually fire rather than being a
    no-op that always passes -- monkeypatch one registry empty and confirm
    the check catches it."""
    monkeypatch.setitem(registry._REGISTRIES, "stt", {})

    with pytest.raises(AssertionError):
        registry.check_registries_non_empty()
