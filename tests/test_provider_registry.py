"""The provider registries and the boot resolver, against fakes -- no
network, no real credential, no real Postgres (D-01, D-03, D-04, PROV-01).
"""

from __future__ import annotations

import pytest

from spire_voice.config import SttConfig
from spire_voice.providers import registry
from spire_voice.providers.boot import ProviderSlotStatus, ProviderUnavailable, resolve_slot


def _stt_config() -> SttConfig:
    return SttConfig(url="wss://stt.invalid/v1/stt")


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


def test_known_entries_returns_every_registered_option_for_the_slot():
    entries = registry.known_entries("stt")

    assert [entry.name for entry in entries] == ["xai"]
    assert entries[0].requires_credential is True
    assert entries[0].batch is False


def test_known_names_is_sorted():
    assert registry.known_names("stt") == sorted(registry.known_names("stt"))


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
async def test_resolve_slot_propagates_any_other_exception_and_stops_the_boot():
    repo = _FakeSelectionRepo(selection=_Selection(provider_name="not-a-real-provider"))

    with pytest.raises(registry.ConfigError):
        await resolve_slot(
            "stt", repo, "xai", registry.build_stt, _stt_config(), "test-key"
        )
