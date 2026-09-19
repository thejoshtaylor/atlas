"""Name-to-factory registries for the three provider slots (D-03).

Providers are code, not plugins (07-CONTEXT.md D-03): routing a speech
provider through Phase 6's MCP plugin machinery would mean a subprocess
boundary and a tool-call round trip in the latency path this project is
already 7x over on. A registry keyed by name, validated at boot so an
unknown name fails loudly with the list of known ones, is the whole
mechanism -- the same fail-loud-by-name shape `app.py::_build_wake_detector`
already uses for the wake engine.

The registries are plain dict literals of in-code callables -- never
`importlib`, never `getattr`, never a name-to-module-path string resolved
at runtime. That is the injection shape D-03 forecloses (T-07-01): an
unknown name raises here rather than being passed through to anything that
could import or exec it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from spire_voice.config import BrainConfig, ConfigError, SttConfig, TtsConfig
from spire_voice.providers.base import BrainProvider, SttProvider, TtsProvider
from spire_voice.providers.brain_xai import XaiBrain
from spire_voice.providers.stt_xai import XaiStt
from spire_voice.providers.tts_xai import XaiTts

__all__ = [
    "ConfigError",
    "ProviderEntry",
    "STT_REGISTRY",
    "TTS_REGISTRY",
    "BRAIN_REGISTRY",
    "build_stt",
    "build_tts",
    "build_brain",
    "known_entries",
    "known_names",
]


@dataclass(frozen=True)
class ProviderEntry:
    """One registry entry: everything a slot needs to know about one named
    provider, whether or not it is the currently selected one.

    `build` takes exactly `(config, api_key)` -- the same narrow,
    single-config-argument shape every provider implementation in this
    codebase already takes (`XaiStt(config)`, `XaiTts(config)`), with the
    resolved credential threaded in via `dataclasses.replace` rather than a
    second constructor parameter. `batch` is `False` for every entry this
    plan ships; plan 07-02's `BatchTtsAdapter` is what a batch-shaped entry
    (Piper, xAI TTS) sets it for (D-05, D-08).
    """

    name: str
    label: str
    build: "Callable[[Any, str], Any]"
    requires_credential: bool
    batch: bool
    licence_note: "str | None" = None


STT_REGISTRY: "dict[str, ProviderEntry]" = {
    "xai": ProviderEntry(
        name="xai",
        label="xAI",
        build=lambda config, api_key: XaiStt(replace(config, api_key=api_key)),
        requires_credential=True,
        batch=False,
        licence_note=None,
    ),
}

TTS_REGISTRY: "dict[str, ProviderEntry]" = {
    "xai": ProviderEntry(
        name="xai",
        label="xAI",
        build=lambda config, api_key: XaiTts(replace(config, api_key=api_key)),
        requires_credential=True,
        batch=False,
        licence_note=None,
    ),
}

BRAIN_REGISTRY: "dict[str, ProviderEntry]" = {
    "xai": ProviderEntry(
        name="xai",
        label="xAI",
        build=lambda config, api_key: XaiBrain(replace(config, api_key=api_key)),
        requires_credential=True,
        batch=False,
        licence_note=None,
    ),
}

# One tuple/dict drives every slot's own "what does 'unknown' mean" check --
# `app.py::_LEGACY_CONFIG_KEYS`'s own generalization rule, applied here to
# slot names rather than legacy config keys.
_REGISTRIES: "dict[str, dict[str, ProviderEntry]]" = {
    "stt": STT_REGISTRY,
    "tts": TTS_REGISTRY,
    "brain": BRAIN_REGISTRY,
}

_SLOT_DISPLAY_NAMES: "dict[str, str]" = {
    "stt": "stt provider",
    "tts": "tts provider",
    "brain": "brain provider",
}


def _build(
    registry: "dict[str, ProviderEntry]",
    slot_display_name: str,
    name: str,
    config: Any,
    api_key: str,
) -> Any:
    """The one place every `build_*` function below looks a name up and
    raises -- naming the rejected value first and the sorted known set
    second, the exact phrasing `_build_wake_detector` already uses for the
    wake engine. Adding a fourth slot later is a new registry passed to
    this helper, never a fourth copy of this sequence."""
    try:
        entry = registry[name]
    except KeyError:
        raise ConfigError(
            f"{slot_display_name} {name!r} is not registered -- known providers: {sorted(registry)}"
        ) from None
    return entry.build(config, api_key)


def build_stt(name: str, config: SttConfig, api_key: str) -> SttProvider:
    return _build(STT_REGISTRY, _SLOT_DISPLAY_NAMES["stt"], name, config, api_key)


def build_tts(name: str, config: TtsConfig, api_key: str) -> TtsProvider:
    return _build(TTS_REGISTRY, _SLOT_DISPLAY_NAMES["tts"], name, config, api_key)


def build_brain(name: str, config: BrainConfig, api_key: str) -> BrainProvider:
    return _build(BRAIN_REGISTRY, _SLOT_DISPLAY_NAMES["brain"], name, config, api_key)


def known_entries(slot: str) -> "list[ProviderEntry]":
    """Every entry registered for `slot`, sorted by name -- the option
    list `routes/providers.py` serves for that slot's `RadioGroup`."""
    return sorted(_REGISTRIES[slot].values(), key=lambda entry: entry.name)


def known_names(slot: str) -> "list[str]":
    """The sorted name set for `slot` -- what a route validates a
    submitted `provider_name` against before ever writing it (T-07-02)."""
    return sorted(_REGISTRIES[slot])
