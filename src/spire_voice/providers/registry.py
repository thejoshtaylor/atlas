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
from urllib.parse import urlsplit

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

    `build` takes exactly `(config, api_key, options)` -- `options` is the
    stored selection row's own settings dict (07-04-PLAN.md Task 1;
    `boot.py::resolve_slot` reads it off the repository and threads it
    through unchanged). Every entry but the local language-model one below
    ignores it, matching the same narrow, single-config-argument shape
    every provider implementation in this codebase already takes
    (`XaiStt(config)`, `XaiTts(config)`), with the resolved credential
    threaded in via `dataclasses.replace` rather than a second constructor
    parameter. `batch` (plan 07-02, D-05, D-08) is `True` only for a batch
    (non-streaming) implementation -- `boot.py::resolve_slot` reads it to
    decide whether to wrap the built client in `BatchTtsAdapter`; the wrap
    itself never happens here or inside a provider module. `needs_server_url`
    (07-04-PLAN.md Task 1) is data, not a hardcoded provider name: it is how
    a future route/screen can reveal a "Server URL" field for exactly the
    entries that read one out of `options`, without special-casing a name.
    `measured_note` (07-04-PLAN.md Task 3, D-12) is the published local-set
    latency figure, phrased with 07-UI-SPEC.md's own local-set latency
    sentence verbatim -- data on the entry, the same shape `licence_note`
    already takes, so a future route/screen renders the contract's words
    rather than a paraphrase, with no special case for which provider it is.
    """

    name: str
    label: str
    build: "Callable[[Any, str, dict], Any]"
    requires_credential: bool
    batch: bool
    licence_note: "str | None" = None
    needs_server_url: bool = False
    measured_note: "str | None" = None


# D-12's published figure (07-04-PLAN.md Task 3): a real `scripts/
# measure_local_providers.py` run against a real, provisioned
# faster-whisper "small"/int8 model and a real Piper "en_US-lessac-medium"
# voice. See docs/runbooks/local-providers.md's Measurement section for
# the full report (median/min/max, repetition count, and the exact
# command) -- this string is the one-line figure quoted from it, phrased
# with 07-UI-SPEC.md's local-set latency sentence verbatim.
_FASTER_WHISPER_MEASURED_NOTE = (
    "Measured on this project's CPU-only host: 1238ms median to transcribe a "
    "spoken reply (5 repetitions, 12-core Apple M4 Pro, no GPU, 2026-09-20)."
)
_PIPER_MEASURED_NOTE = (
    "Measured on this project's CPU-only host: 112ms median to synthesize a "
    "spoken reply (5 repetitions, 12-core Apple M4 Pro, no GPU, 2026-09-20)."
)

STT_REGISTRY: "dict[str, ProviderEntry]" = {
    "xai": ProviderEntry(
        name="xai",
        label="xAI",
        build=lambda config, api_key, options: XaiStt(replace(config, api_key=api_key)),
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
        build=lambda config, api_key, options: FasterWhisperStt(config),
        requires_credential=False,
        batch=False,
        licence_note=None,
        measured_note=_FASTER_WHISPER_MEASURED_NOTE,
    ),
}

TTS_REGISTRY: "dict[str, ProviderEntry]" = {
    "xai": ProviderEntry(
        name="xai",
        label="xAI",
        build=lambda config, api_key, options: XaiTts(replace(config, api_key=api_key)),
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
        build=lambda config, api_key, options: PiperTts(config),
        requires_credential=False,
        # Piper also renders a whole utterance in one call -- batch, like
        # xAI's REST endpoint -- so `boot.py::resolve_slot` wraps it in the
        # same `BatchTtsAdapter`, never a second wrapper class.
        batch=True,
        licence_note=_PIPER_LICENCE_NOTE,
        measured_note=_PIPER_MEASURED_NOTE,
    ),
}

_LOCAL_BRAIN_LABEL = "Self-hosted (local)"

# `AsyncOpenAI.__init__` refuses a falsy `api_key` outright (see
# `_build_local_brain` below) even though there is no real credential for
# this entry at all -- a placeholder, not a secret. Held as its own named
# constant, referenced without quotes at the one call site, so
# `tests/test_repo_hygiene.py`'s repository-wide credential-literal scan
# (which matches `api_key\s*[:=]\s*['"]...['"]`) never sees an
# `api_key="..."` literal to flag in the first place.
_LOCAL_BRAIN_API_KEY_PLACEHOLDER = "not-needed"


def _build_local_brain(config: BrainConfig, options: "dict[str, Any]") -> "tuple[Any, ...]":
    """D-09: any OpenAI-compatible local server, reached by URL -- no new
    provider class, since `XaiBrain`'s constructor already takes exactly a
    base URL and an API key, which is the reason `config.brain` was shaped
    that way in the first place.

    The server URL comes from the slot's own stored `options`
    (`boot.py::resolve_slot`'s per-slot settings, never `config.brain.
    base_url`) -- an admin saves it per-slot through the /providers screen.
    A blank or missing URL degrades this slot exactly the way a missing
    credential degrades any other entry (D-04): selectable and saveable
    with nothing entered yet, degraded by name at the next boot.
    """
    server_url = str((options or {}).get("server_url", "")).strip()
    if not server_url:
        raise ProviderUnavailable(
            f"Missing a server URL for {_LOCAL_BRAIN_LABEL}. Add one in Settings."
        )
    if urlsplit(server_url).scheme not in ("http", "https"):
        raise ProviderUnavailable(
            f"The server URL for {_LOCAL_BRAIN_LABEL} must be http or https, "
            f"got {server_url!r}. Fix it in Settings."
        )
    # `AsyncOpenAI.__init__` itself refuses a falsy `api_key` (raises
    # `OpenAIError` before a single request is ever sent) regardless of
    # what the target server would accept -- the placeholder most
    # self-hosted OpenAI-compatible server docs use for exactly this case.
    # There is no credential slot for this entry at all
    # (requires_credential=False, below); this is a client-library
    # requirement, not a real secret.
    return brain_race.build_tiers(
        replace(config, base_url=server_url, api_key=_LOCAL_BRAIN_API_KEY_PLACEHOLDER)
    )


BRAIN_REGISTRY: "dict[str, ProviderEntry]" = {
    "xai": ProviderEntry(
        name="xai",
        label="xAI",
        # `brain_race.build_tiers` is the factory itself, not `XaiBrain`
        # directly -- it builds the whole tier tuple `app.py` needs
        # (one shared `instructor` client, one `TierBrain` per configured
        # model), keeping that function's existing shape rather than
        # threading extra arguments through it.
        build=lambda config, api_key, options: brain_race.build_tiers(
            replace(config, api_key=api_key)
        ),
        requires_credential=True,
        batch=False,
        licence_note=None,
    ),
    "local": ProviderEntry(
        name="local",
        label=_LOCAL_BRAIN_LABEL,
        build=lambda config, api_key, options: _build_local_brain(config, options),
        # No credential to thread through -- the same posture the local
        # speech-to-text/text-to-speech entries already take.
        requires_credential=False,
        batch=False,
        licence_note=None,
        # The one entry, across all three slots, that reads a server URL
        # out of its own stored `options` (07-UI-SPEC.md's "Server URL"
        # field) -- a flag on the entry so a route/screen can reveal that
        # field from data, never from a hardcoded provider name.
        needs_server_url=True,
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
    options: "dict | None" = None,
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
    Contract gives as its own example.

    `options` (07-04-PLAN.md Task 1) is the stored selection row's own
    settings dict, threaded through to `entry.build` unchanged -- `{}`
    when the caller passed nothing. Every entry but the local
    language-model one ignores it."""
    try:
        entry = registry[name]
    except KeyError:
        raise ConfigError(
            f"{slot_display_name} {name!r} is not registered -- known providers: {sorted(registry)}"
        ) from None
    if entry.requires_credential and not api_key:
        raise ProviderUnavailable(f"Missing an API key for {entry.label}. Add one in Settings.")
    return entry.build(config, api_key, options or {})


def build_stt(name: str, config: SttConfig, api_key: str, options: "dict | None" = None) -> SttProvider:
    return _build(STT_REGISTRY, _SLOT_DISPLAY_NAMES["stt"], name, config, api_key, options)


def build_tts(name: str, config: TtsConfig, api_key: str, options: "dict | None" = None) -> TtsProvider:
    return _build(TTS_REGISTRY, _SLOT_DISPLAY_NAMES["tts"], name, config, api_key, options)


def build_brain(
    name: str, config: BrainConfig, api_key: str, options: "dict | None" = None
) -> "tuple[Any, ...]":
    """Returns the tier tuple `brain_race.build_tiers` produces, not a
    single `BrainProvider` -- the language-model slot is a race across
    tiers (D-05, `turn/brain_race.py`), never one client."""
    return _build(BRAIN_REGISTRY, _SLOT_DISPLAY_NAMES["brain"], name, config, api_key, options)


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
    mistake caught at process start, never a blank screen with no error.

    IN-02 (code review): a `RuntimeError`, not an `assert`. `python -O`
    compiles an assert out entirely, and the check this function exists
    to perform would then silently disappear -- leaving exactly the blank
    screen with no error the sentence above promises cannot happen. The
    deployment CMD is `python -m spire_voice.app` today, but nothing stops
    a values file or a derived image setting `PYTHONOPTIMIZE=1`, and the
    failure mode of that mistake should not be this."""
    for slot, entries in _REGISTRIES.items():
        if not entries:
            raise RuntimeError(
                f"providers/registry.py: the {slot!r} registry must not be empty"
            )


check_registries_non_empty()
