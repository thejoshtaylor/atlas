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
from spire_voice.providers.base import SttProvider, TtsProvider
from spire_voice.providers.boot import ProviderUnavailable
from spire_voice.providers.stt_faster_whisper import FasterWhisperStt
from spire_voice.providers.stt_xai import XaiStt
from spire_voice.providers.tts_piper import PiperTts
from spire_voice.providers.tts_xai import XaiTts

# 07-UI-SPEC.md's Piper licence line, verbatim -- the registry entry
# carries the contract's exact string as data, so the screen renders it
# rather than a component-side paraphrase of it (D-10).
_PIPER_LICENCE_NOTE = (
    "Piper is licensed under GPL-3.0, not the MIT licence most of this project "
    "uses. See the README before you redistribute a build that includes it."
)
from spire_voice.turn import brain_race

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
    "check_registries_non_empty",
]


@dataclass(frozen=True)
class ProviderEntry:
    """One registry entry: everything a slot needs to know about one named
    provider, whether or not it is the currently selected one.

    `build` takes exactly `(config, api_key)` -- the same narrow,
    single-config-argument shape every provider implementation in this
    codebase already takes (`XaiStt(config)`, `XaiTts(config)`), with the
    resolved credential threaded in via `dataclasses.replace` rather than a
    second constructor parameter. `batch` (plan 07-02, D-05, D-08) is
    `True` only for xAI's text-to-speech entry today -- the one slot
    that cannot stream. `boot.py::resolve_slot` reads it to decide
    whether to wrap the built client in `BatchTtsAdapter`; the wrap
    itself never happens here or inside a provider module.
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
    "faster-whisper": ProviderEntry(
        name="faster-whisper",
        label="faster-whisper (local)",
        # No credential to thread through -- `config` is passed unchanged.
        # `requires_credential=False` is the whole point of the local set
        # (07-03-PLAN.md Task 2): the needs-a-credential hint must never
        # appear on this entry.
        build=lambda config, api_key: FasterWhisperStt(config),
        requires_credential=False,
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
        # D-05: xAI's text-to-speech is a single REST call, not a stream --
        # `boot.py::resolve_slot` reads this field to decide whether to
        # wrap the built client in `BatchTtsAdapter`. The wrap happens
        # there, not here: this entry only states the fact.
        batch=True,
        licence_note=None,
    ),
    "piper": ProviderEntry(
        name="piper",
        label="Piper (local)",
        # No credential to thread through -- `config` is passed unchanged,
        # the same shape the local speech-to-text entry uses.
        build=lambda config, api_key: PiperTts(config),
        requires_credential=False,
        # Piper also renders a whole utterance in one call -- batch, like
        # xAI's REST endpoint -- so `boot.py::resolve_slot` wraps it in the
        # same `BatchTtsAdapter`, never a second wrapper class.
        batch=True,
        licence_note=_PIPER_LICENCE_NOTE,
    ),
}

BRAIN_REGISTRY: "dict[str, ProviderEntry]" = {
    "xai": ProviderEntry(
        name="xai",
        label="xAI",
        # `brain_race.build_tiers` is the factory itself, not `XaiBrain`
        # directly -- it builds the whole tier tuple `app.py` needs
        # (one shared `instructor` client, one `TierBrain` per configured
        # model), keeping that function's existing shape rather than
        # threading extra arguments through it.
        build=lambda config, api_key: brain_race.build_tiers(replace(config, api_key=api_key)),
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
    this helper, never a fourth copy of this sequence.

    D-04: a registered entry that needs a credential nobody has set raises
    `ProviderUnavailable` here, before `entry.build` ever constructs a
    client with an empty API key -- `resolve_slot` (`boot.py`) is what
    turns this into a degraded slot rather than a boot failure. The
    message names the provider's own operator-facing label and where an
    admin fixes it, the exact phrasing 07-UI-SPEC.md's Copywriting
    Contract gives as its own example."""
    try:
        entry = registry[name]
    except KeyError:
        raise ConfigError(
            f"{slot_display_name} {name!r} is not registered -- known providers: {sorted(registry)}"
        ) from None
    if entry.requires_credential and not api_key:
        raise ProviderUnavailable(f"Missing an API key for {entry.label}. Add one in Settings.")
    return entry.build(config, api_key)


def build_stt(name: str, config: SttConfig, api_key: str) -> SttProvider:
    return _build(STT_REGISTRY, _SLOT_DISPLAY_NAMES["stt"], name, config, api_key)


def build_tts(name: str, config: TtsConfig, api_key: str) -> TtsProvider:
    return _build(TTS_REGISTRY, _SLOT_DISPLAY_NAMES["tts"], name, config, api_key)


def build_brain(name: str, config: BrainConfig, api_key: str) -> "tuple[Any, ...]":
    """Returns the tier tuple `brain_race.build_tiers` produces, not a
    single `BrainProvider` -- the language-model slot is a race across
    tiers (D-05, `turn/brain_race.py`), never one client."""
    return _build(BRAIN_REGISTRY, _SLOT_DISPLAY_NAMES["brain"], name, config, api_key)


def known_entries(slot: str) -> "list[ProviderEntry]":
    """Every entry registered for `slot`, sorted by name -- the option
    list `routes/providers.py` serves for that slot's `RadioGroup`."""
    return sorted(_REGISTRIES[slot].values(), key=lambda entry: entry.name)


def known_names(slot: str) -> "list[str]":
    """The sorted name set for `slot` -- what a route validates a
    submitted `provider_name` against before ever writing it (T-07-02)."""
    return sorted(_REGISTRIES[slot])


def check_registries_non_empty() -> None:
    """Every one of the three registries holds at least one entry --
    07-UI-SPEC.md's screen has no empty state at all (D-03's boot-time
    validation is what makes that a fact rather than a hope). Run
    unconditionally at import, below, so an empty registry is an authoring
    mistake caught at process start, never a blank screen with no error."""
    for slot, entries in _REGISTRIES.items():
        assert entries, f"providers/registry.py: the {slot!r} registry must not be empty"


check_registries_non_empty()
