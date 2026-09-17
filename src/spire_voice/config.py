"""Load, expand, and validate spire-voice's configuration.

Two rules that are easy to get backwards:

1. Environment expansion happens once, over the raw file text, before the
   YAML parse. A `${NAME}` with no value in the environment is a startup
   error naming that variable, never an empty string -- an empty API key
   or an empty Home Assistant token would otherwise fail much later,
   inside a provider call, in a way that looks like the provider's fault.
2. Every config-shaped section builds itself through a `from_config`
   classmethod that raises `ConfigError` on an invalid value, mirroring
   `spire_mcp.safety.Policy`'s own doctrine: raised, not returned, so a
   caller cannot silently continue with a bad configuration. The
   `safety:` block itself is handed to `Policy.from_config` unchanged --
   this module does not reinterpret, filter, or default any key inside
   it. That boundary belongs to `safety.py` alone.
"""

from __future__ import annotations

import os
import re
from dataclasses import MISSING, dataclass, field, fields, replace

import yaml

from spire_mcp.safety import Policy
from spire_voice.turn.macros import normalize

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(Exception):
    """Raised instead of returned, so a caller cannot ignore a bad config."""


def expand_env(raw_text: str) -> str:
    """Replace every `${NAME}` in `raw_text` with its environment value.

    One pass, over the raw text, before `yaml.safe_load` ever runs -- an
    expanded value that itself contains `${` is not expanded a second
    time. A `${NAME}` naming a variable absent from the environment raises
    `ConfigError` naming that variable, rather than expanding to `""`.

    A whole-line YAML comment is left untouched. This file's own header
    documents the `${NAME}` contract in prose (see the top of
    config.example.yaml), and that documentation is not itself a
    placeholder to resolve -- expanding it would turn every load of the
    example file into a startup error naming the literal word "NAME".
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        try:
            return os.environ[name]
        except KeyError:
            raise ConfigError(f"missing required environment variable: {name}") from None

    lines = raw_text.splitlines(keepends=True)
    expanded_lines = [
        line if line.lstrip().startswith("#") else _PLACEHOLDER_RE.sub(_replace, line)
        for line in lines
    ]
    return "".join(expanded_lines)


@dataclass(frozen=True)
class ServerConfig:
    """The dev harness's own bind address, port, and transport selector.

    `transport` is the D-01 selector: the operator picks `websocket` or
    `webrtc`, and Phase 3's admin surface exposes this same key. There is
    no silent default for a value the operator typed wrong -- an
    unrecognized transport raises at load, the same way
    `Policy.from_config` raises on an unknown mode.
    """

    bind_host: str = "127.0.0.1"
    port: int = 8080
    transport: str = "websocket"

    @classmethod
    def from_config(cls, raw: dict | None) -> "ServerConfig":
        raw = raw or {}
        transport = raw.get("transport", "websocket")
        if transport not in ("websocket", "webrtc"):
            raise ConfigError(f"unknown server transport: {transport!r}")
        return cls(
            bind_host=raw.get("bind_host", "127.0.0.1"),
            port=raw.get("port", 8080),
            transport=transport,
        )


@dataclass(frozen=True)
class SttConfig:
    """The `stt:` block, carried under its own config key names.

    Translation to xAI's wire parameter names (`endpointing_ms ->
    endpointing`, etc.) happens in the speech-to-text provider, not here
    -- this class is a typed passthrough of what the config file says.
    """

    url: str = ""
    api_key: str = ""
    endpointing_ms: int = 400
    smart_turn: float = 0.7
    smart_turn_timeout_ms: int = 1200
    vad_threshold: float = 0.08
    interim_results: bool = True
    language: str = "en"
    max_utterance_s: int = 15

    @classmethod
    def from_config(cls, raw: dict | None) -> "SttConfig":
        raw = raw or {}
        return cls(
            url=raw.get("url", cls.url),
            api_key=raw.get("api_key", cls.api_key),
            endpointing_ms=raw.get("endpointing_ms", cls.endpointing_ms),
            smart_turn=raw.get("smart_turn", cls.smart_turn),
            smart_turn_timeout_ms=raw.get("smart_turn_timeout_ms", cls.smart_turn_timeout_ms),
            vad_threshold=raw.get("vad_threshold", cls.vad_threshold),
            interim_results=raw.get("interim_results", cls.interim_results),
            language=raw.get("language", cls.language),
            max_utterance_s=raw.get("max_utterance_s", cls.max_utterance_s),
        )


@dataclass(frozen=True)
class BrainTierConfig:
    """One entry in `brain.models` -- one candidate model id for the tier race.

    `calls_tools` is not a field here on purpose: which tier calls tools is
    decided by position (the last entry in `brain.models`, D-05), never by a
    per-entry flag. `from_config` raises if a config file tries to set one.
    """

    model: str = ""

    @classmethod
    def from_config(cls, raw: dict | None, index: int) -> "BrainTierConfig":
        raw = raw or {}
        if "calls_tools" in raw:
            raise ConfigError(
                f"brain.models[{index}] sets 'calls_tools': which tier calls "
                "tools is decided by position, not a per-entry flag -- the "
                "last entry in brain.models is the top tier and the only one "
                "that reaches Home Assistant (D-05). A per-entry flag would "
                "let a configuration produce two tool-calling tiers, which is "
                "the one thing D-05 exists to prevent. Reorder brain.models "
                "if you want a different tier to call tools."
            )
        model = raw.get("model", "")
        if not model:
            raise ConfigError(f"brain.models[{index}] is missing a 'model' id")
        return cls(model=model)


@dataclass(frozen=True)
class BrainConfig:
    """The `brain:` block: the language model endpoint, the ordered tier
    list, and the tool-round cap.

    `models` replaces the old single `model` key (D-01): an ordered list,
    fastest first, most capable last. `top_tier` (the last entry) is the only
    tier that calls tools (D-05); `triage_tiers` is every entry before it.
    There is no default tier list -- an empty or absent `models` key stops
    startup by name rather than quietly falling back to a hardcoded id,
    matching this module's own doctrine (see the module docstring).
    """

    base_url: str = "https://api.x.ai/v1"
    api_key: str = ""
    models: tuple[BrainTierConfig, ...] = ()
    # D-08's deadline: how long to wait, after the final transcript, before
    # playing the filler if no answer audio is ready yet. Provisional --
    # plan 01.1-08 measures the real number (RESEARCH.md Open Question 3).
    # 0 disables the filler entirely.
    filler_after_ms: int = 600
    cache_system_prompt: bool = True
    temperature: float = 0.0
    max_tokens: int = 400
    max_tool_rounds: int = 3

    @property
    def top_tier(self) -> BrainTierConfig:
        """The last entry in `models` -- the only tier that calls tools."""
        return self.models[-1]

    @property
    def triage_tiers(self) -> tuple[BrainTierConfig, ...]:
        """Every entry before `top_tier` -- triage plus voice, no tools."""
        return self.models[:-1]

    @classmethod
    def from_config(cls, raw: dict | None) -> "BrainConfig":
        raw = raw or {}
        if "model" in raw:
            raise ConfigError(
                "brain.model no longer exists: replaced by brain.models, an "
                "ordered tier list (D-01) -- see config.example.yaml. Wrap "
                "the single id in a one-element list, e.g. "
                'models: [{model: "grok-4.6"}]'
            )
        models_raw = raw.get("models", ())
        if isinstance(models_raw, str):
            raise ConfigError("brain.models must be a list, not a string")
        if not models_raw:
            raise ConfigError(
                "brain.models is empty: there is no default tier -- configure "
                "at least one model id"
            )
        models = tuple(
            BrainTierConfig.from_config(entry, index)
            for index, entry in enumerate(models_raw)
        )
        return cls(
            base_url=raw.get("base_url", cls.base_url),
            api_key=raw.get("api_key", cls.api_key),
            models=models,
            filler_after_ms=raw.get("filler_after_ms", cls.filler_after_ms),
            cache_system_prompt=raw.get("cache_system_prompt", cls.cache_system_prompt),
            temperature=raw.get("temperature", cls.temperature),
            max_tokens=raw.get("max_tokens", cls.max_tokens),
            max_tool_rounds=raw.get("max_tool_rounds", cls.max_tool_rounds),
        )


@dataclass(frozen=True)
class TtsConfig:
    """The `tts:` block: two codec pairs for two different sinks.

    `codec`/`sample_rate` matches the camera speaker path (Phase 2) and
    is left untouched here. `browser_codec`/`browser_sample_rate` is the
    Phase 1 addition: a browser's Web Audio API cannot decode the
    camera-facing A-law stream, so the browser sink requests raw PCM
    instead. Both live on the same `TtsConfig` because both describe the
    one TTS provider's two possible output requests, not two providers.
    """

    url: str = "https://api.x.ai/v1/tts"
    api_key: str = ""
    voice_id: str = "eve"
    language: str = "en"
    codec: str = "alaw"
    sample_rate: int = 8000
    browser_codec: str = "pcm"
    browser_sample_rate: int = 24000
    optimize_streaming_latency: int = 2
    # Synthesis is one REST call that returns the whole utterance, so this is
    # a ceiling on the entire reply, not on a first chunk. Generous enough for
    # a long sentence, short enough that a hung call cannot hold a turn open
    # past the point an operator would have given up and spoken again.
    request_timeout_s: float = 20.0
    # config.example.yaml has declared both of these since Phase 01; this is
    # wiring an already-declared key, not inventing one. Both are optional --
    # unlike brain.models, an absent tts: block should not stop startup.
    cache_dir: str = "/data/tts-cache"
    precache: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, raw: dict | None) -> "TtsConfig":
        raw = raw or {}
        return cls(
            url=raw.get("url", cls.url),
            api_key=raw.get("api_key", cls.api_key),
            voice_id=raw.get("voice_id", cls.voice_id),
            language=raw.get("language", cls.language),
            codec=raw.get("codec", cls.codec),
            sample_rate=raw.get("sample_rate", cls.sample_rate),
            browser_codec=raw.get("browser_codec", cls.browser_codec),
            browser_sample_rate=raw.get("browser_sample_rate", cls.browser_sample_rate),
            optimize_streaming_latency=raw.get(
                "optimize_streaming_latency", cls.optimize_streaming_latency
            ),
            request_timeout_s=float(raw.get("request_timeout_s", cls.request_timeout_s)),
            cache_dir=raw.get("cache_dir", cls.cache_dir),
            precache=tuple(raw.get("precache", cls.precache)),
        )


@dataclass(frozen=True)
class CameraConfig:
    """The `camera:` block: the RTSP audio source and the native format it
    carries straight through to the transcriber, with no transcode step
    (PROV-07). Read by the camera `AudioSource` (plan 02-03), which declares
    `transports/base.py`'s `SourceFormat(encoding, sample_rate)` from these
    same two fields -- there is one place the camera's format is written,
    not two that could disagree.

    `encoding` is validated against the two encodings this codebase actually
    carries end to end (`transports/base.py`'s `SourceFormat` docstring:
    `"pcm"` and `"alaw"`), rather than accepted as an arbitrary string. Any
    other value would mean a transcode step no code in this phase
    implements, which would produce silence at turn time with no test to
    catch it first.
    """

    rtsp_url: str = ""
    encoding: str = "alaw"
    sample_rate: int = 8000
    channels: int = 1
    preroll_ms: int = 1500

    @classmethod
    def from_config(cls, raw: dict | None) -> "CameraConfig":
        raw = raw or {}
        encoding = raw.get("encoding", cls.encoding)
        supported = ("alaw", "pcm")
        if encoding not in supported:
            raise ConfigError(
                f"camera.encoding {encoding!r} is not one this pipeline carries "
                f"through to the transcriber untranscoded -- supported values are "
                f"{supported!r}. Any other value means a transcode step this phase "
                "does not implement, which would produce silence at turn time."
            )
        return cls(
            rtsp_url=raw.get("rtsp_url", cls.rtsp_url),
            encoding=encoding,
            sample_rate=raw.get("sample_rate", cls.sample_rate),
            channels=raw.get("channels", cls.channels),
            preroll_ms=raw.get("preroll_ms", cls.preroll_ms),
        )


@dataclass(frozen=True)
class SpeakerConfig:
    """The `speaker:` block: the go2rtc backchannel and the FIFO the
    long-lived ffmpeg supervisor (plan 02-03) reads from.

    `respawn_backoff_s` must be positive: a zero or negative backoff turns
    the supervisor into a busy loop restarting a dead subprocess with no
    delay between attempts, which is a worse failure than refusing to start
    (RESEARCH.md Pitfalls 4 and 5).
    """

    go2rtc_url: str = "http://frigate:1984"
    stream: str = "cam"
    ensure_url: str = ""
    fifo_path: str = "/run/spire/speaker.alaw"
    respawn_backoff_s: float = 2.0

    @classmethod
    def from_config(cls, raw: dict | None) -> "SpeakerConfig":
        raw = raw or {}
        respawn_backoff_s = float(raw.get("respawn_backoff_s", cls.respawn_backoff_s))
        if respawn_backoff_s <= 0:
            raise ConfigError(
                f"speaker.respawn_backoff_s must be positive, got {respawn_backoff_s!r} "
                "-- a zero or negative backoff turns the ffmpeg supervisor into a busy "
                "loop against a dead subprocess"
            )
        return cls(
            go2rtc_url=raw.get("go2rtc_url", cls.go2rtc_url),
            stream=raw.get("stream", cls.stream),
            ensure_url=raw.get("ensure_url", cls.ensure_url),
            fifo_path=raw.get("fifo_path", cls.fifo_path),
            respawn_backoff_s=respawn_backoff_s,
        )


@dataclass(frozen=True)
class SessionConfig:
    """The `debug:` block: one directory per turn, its retention window, and
    the sweep interval that enforces it. Read by the session recorder and
    the retention sweep (plans 02-07 and 02-08).

    Reads the **existing** `debug:` block rather than introducing a second
    `session:` section -- a parallel block would give `retain_days` two
    homes that could disagree, the same shadowing failure
    `_check_macros_do_not_collide` exists to prevent below.
    `expiry_interval_s` is the one field this block does not have yet: how
    often the retention sweep runs, defaulted well under a day so a
    long-running process sweeps more than once between restarts.
    """

    dir: str = "/data/sessions"
    record_audio: bool = True
    # Seven days: a conservative floor on audio of a real home. The operator
    # raises this deliberately; it is never lowered for convenience (D-14,
    # T-02-05).
    retain_days: int = 7
    stdout_summary: bool = True
    expiry_interval_s: int = 3600

    @classmethod
    def from_config(cls, raw: dict | None) -> "SessionConfig":
        raw = raw or {}
        retain_days = raw.get("retain_days", cls.retain_days)
        if not isinstance(retain_days, int) or isinstance(retain_days, bool) or retain_days <= 0:
            raise ConfigError(
                f"debug.retain_days must be a positive integer, got {retain_days!r} -- "
                "a zero or negative retention would mean deleting a turn's recording "
                "while the turn is still being written"
            )
        expiry_interval_s = raw.get("expiry_interval_s", cls.expiry_interval_s)
        if expiry_interval_s <= 0:
            raise ConfigError(
                f"debug.expiry_interval_s must be positive, got {expiry_interval_s!r}"
            )
        return cls(
            dir=raw.get("dir", cls.dir),
            record_audio=raw.get("record_audio", cls.record_audio),
            retain_days=retain_days,
            stdout_summary=raw.get("stdout_summary", cls.stdout_summary),
            expiry_interval_s=expiry_interval_s,
        )


def _validate_and_normalize_override(cls: type, raw_override: dict, label: str) -> dict:
    """Validate `raw_override`'s keys against `cls`'s own fields, coercing a
    list into a tuple for any field whose global default is a tuple, and
    rebuilding a nested sub-config field (one built via
    `field(default_factory=...)`, such as `WakeConfig.openwakeword`) through
    its own `from_config` -- matching what `from_config` would have done had
    the same value arrived globally rather than per-source.

    Shared by `WakeConfig`, `GateConfig` and `BargeInConfig`'s per-source
    overrides (D-04, D-12): one shape, reused three times, rather than a
    second configuration pattern invented for barge-in. Raises `ConfigError`
    naming the first field `cls` does not have -- an override targeting a
    typo'd field name silently doing nothing is the failure mode this
    validation exists for.

    A field built via `default_factory` has no class-level attribute at all
    (`getattr(cls, key, None)` returns `None` for it, the same as for any
    other unset name) -- `fields(cls)` is consulted instead, so a raw dict
    naming a real nested field (`wake.sources.camera.openwakeword`, say)
    cannot slip through untouched. Left unhandled, that raw dict would pass
    this validation (the field name is real) and only fail much later, as
    an `AttributeError` inside `_resolve_threshold`, when a caller expects a
    real sub-config and gets a plain `dict` instead -- exactly the "raised,
    not returned" doctrine this module states for itself (module docstring)
    being silently missed for one specific field shape (found in code
    review).
    """
    field_by_name = {f.name: f for f in fields(cls) if f.name != "sources"}
    normalized: dict = {}
    for key, value in raw_override.items():
        field_def = field_by_name.get(key)
        if field_def is None:
            raise ConfigError(
                f"{label} sets unknown field {key!r} -- valid fields are "
                f"{sorted(field_by_name)!r}"
            )
        default_value = getattr(cls, key, None)
        if default_value is None and field_def.default_factory is not MISSING:
            nested_default = field_def.default_factory()
            nested_from_config = getattr(type(nested_default), "from_config", None)
            if nested_from_config is not None:
                if not isinstance(value, dict):
                    raise ConfigError(
                        f"{label}.{key} must be a mapping, not "
                        f"{type(value).__name__} -- it configures a nested "
                        f"{type(nested_default).__name__} block"
                    )
                normalized[key] = nested_from_config(value)
                continue
        if isinstance(default_value, tuple) and not isinstance(value, tuple):
            if isinstance(value, str):
                raise ConfigError(f"{label}.{key} must be a list, not a string")
            value = tuple(value)
        normalized[key] = value
    return normalized


_WAKE_ENGINES = ("openwakeword", "vosk")


@dataclass(frozen=True)
class OpenWakeWordConfig:
    """The `wake.openwakeword` sub-block: model path, detection threshold,
    and how many consecutive frames must clear it before a hit counts.

    Ships as code regardless of which engine is selected (a PROJECT.md Key
    Decision) -- `model_path` names a file that does not exist yet
    (RESEARCH.md Pitfall 3); this class parses the shape without asserting
    the file is present.
    """

    model_path: str = "/models/hey_spire.onnx"
    threshold: float = 0.55
    trigger_frames: int = 2

    @classmethod
    def from_config(cls, raw: dict | None) -> "OpenWakeWordConfig":
        raw = raw or {}
        threshold = float(raw.get("threshold", cls.threshold))
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(
                f"wake.openwakeword.threshold must be within [0.0, 1.0], got {threshold!r}"
            )
        return cls(
            model_path=raw.get("model_path", cls.model_path),
            threshold=threshold,
            trigger_frames=raw.get("trigger_frames", cls.trigger_frames),
        )


@dataclass(frozen=True)
class VoskWakeConfig:
    """The `wake.vosk` sub-block: model path and the grammar restricting the
    decoder to the wake phrase plus `[unk]`, which is what turns a full ASR
    engine into a cheap, low-false-positive wake detector.
    """

    model_path: str = "/models/vosk-model-small-en-us-0.15"
    grammar: tuple[str, ...] = ("hey spire", "[unk]")

    @classmethod
    def from_config(cls, raw: dict | None) -> "VoskWakeConfig":
        raw = raw or {}
        grammar_raw = raw.get("grammar", cls.grammar)
        if isinstance(grammar_raw, str):
            raise ConfigError("wake.vosk.grammar must be a list, not a string")
        return cls(
            model_path=raw.get("model_path", cls.model_path),
            grammar=tuple(grammar_raw),
        )


@dataclass(frozen=True)
class WakeConfig:
    """The `wake:` block: the engine selector, the phrase, the refractory
    window, and both engines' sub-blocks -- both ship as code regardless of
    which is selected (a PROJECT.md Key Decision). Read by the wake-word
    detector (plan 02-04).

    `engine` is a closed set of two, mirroring `ServerConfig.from_config`'s
    own transport-selector rejection: an unrecognized engine raises at load
    rather than silently falling back to one. The default is `vosk`, not
    `openwakeword`: no "hey spire" model exists for openWakeWord yet, and
    producing one is an offline training pipeline outside this phase's scope
    (RESEARCH.md Pitfall 3, D-08) -- the shipped default must be the engine
    that can actually run today.

    `threshold` (on `OpenWakeWordConfig`) stays a float end to end with no
    rounding anywhere on the path from configuration to comparison: plan
    02-04 asserts the comparison semantics against exactly the value this
    class returns.

    Carries the same per-source override shape `GateConfig` and
    `BargeInConfig` use (D-04, D-12): a global policy plus an optional
    mapping from source name to a partial override, resolved through
    `resolve()`.
    """

    engine: str = "vosk"
    phrase: str = "hey spire"
    refractory_s: float = 2.0
    openwakeword: OpenWakeWordConfig = field(default_factory=OpenWakeWordConfig)
    vosk: VoskWakeConfig = field(default_factory=VoskWakeConfig)
    sources: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: dict | None) -> "WakeConfig":
        raw = raw or {}
        engine = raw.get("engine", cls.engine)
        if engine not in _WAKE_ENGINES:
            raise ConfigError(f"wake.engine {engine!r} is not one of {_WAKE_ENGINES!r}")
        refractory_s = float(raw.get("refractory_s", cls.refractory_s))
        if refractory_s < 0:
            raise ConfigError(f"wake.refractory_s must be non-negative, got {refractory_s!r}")
        sources_raw = raw.get("sources", {}) or {}
        sources = {
            name: _validate_and_normalize_override(cls, override or {}, f"wake.sources.{name}")
            for name, override in sources_raw.items()
        }
        return cls(
            engine=engine,
            phrase=raw.get("phrase", cls.phrase),
            refractory_s=refractory_s,
            openwakeword=OpenWakeWordConfig.from_config(raw.get("openwakeword")),
            vosk=VoskWakeConfig.from_config(raw.get("vosk")),
            sources=sources,
        )

    def resolve(self, source: str) -> "WakeConfig":
        """The effective wake policy for `source`: the global policy
        unchanged if `source` has no override, or the merged policy
        otherwise (D-04, D-12)."""
        override = self.sources.get(source)
        if not override:
            return self
        return replace(self, **override)


_REMOVED_GATE_IDENTITY_KEYS = ("require_face", "face_names", "face_window_s")


@dataclass(frozen=True)
class GateConfig:
    """The `gate:` block: which media players suppress the wake word, with a
    per-source override (D-04). Read by the gate that decides whether a wake
    hit counts (plan 02-04/02-05).

    The three identity keys (`require_face`, `face_names`, `face_window_s`)
    are deleted, not defaulted off (D-01): PROJECT.md's Out of Scope says
    the operator chose to let anyone in earshot command the house, and the
    denylist in `mcp/spire_mcp/safety.py` is what protects it, not identity.
    Following `BrainConfig.from_config`'s own precedent for a deliberately
    removed key, a configuration file still setting any of the three stops
    startup by name -- an operator whose file predates this phase must be
    told the gate is gone, not left believing a removed gate still protects
    them.
    """

    mute_when_playing: tuple[str, ...] = ()
    sources: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: dict | None) -> "GateConfig":
        raw = raw or {}
        for key in _REMOVED_GATE_IDENTITY_KEYS:
            if key in raw:
                raise ConfigError(
                    f"gate.{key} no longer exists: identity gating was removed by "
                    "project decision (PROJECT.md Out of Scope, D-01) -- the "
                    "operator chose to let anyone in earshot command the house, and "
                    "the denylist in code is what protects it, not identity. Remove "
                    "this key from your configuration."
                )
        mute_raw = raw.get("mute_when_playing", cls.mute_when_playing)
        if isinstance(mute_raw, str):
            raise ConfigError("gate.mute_when_playing must be a list, not a string")
        sources_raw = raw.get("sources", {}) or {}
        sources = {
            name: _validate_and_normalize_override(cls, override or {}, f"gate.sources.{name}")
            for name, override in sources_raw.items()
        }
        return cls(mute_when_playing=tuple(mute_raw), sources=sources)

    def resolve(self, source: str) -> "GateConfig":
        """The effective gate policy for `source`: the global policy
        unchanged if `source` has no override, or the merged policy
        otherwise (D-04)."""
        override = self.sources.get(source)
        if not override:
            return self
        return replace(self, **override)


@dataclass(frozen=True)
class BargeInConfig:
    """The `barge_in:` block: known-output suppression, not acoustic echo
    cancellation (D-09) -- the interrupt triggers on sustained energy above
    a floor for a minimum duration, never a single frame and never the wake
    word (D-10). Global with a per-source override (D-12), same shape as
    `GateConfig`. Read by `_speak`'s interrupt point in
    `turn/controller.py` (plan 02-06).

    `energy_floor` is derived from `SttConfig.vad_threshold`, already tuned
    for this camera's across-a-room noise floor -- the only noise-floor
    measurement this camera has today (CD-3). It is provisional until real
    camera sessions exist to tune it directly. The floor and the measure
    compared against it are in the same unit everywhere on this path, with
    no implicit conversion: a configured floor means one thing only.
    """

    enabled: bool = True
    energy_floor: float = 0.08
    min_duration_ms: int = 300
    post_playback_guard_ms: int = 150
    sources: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: dict | None) -> "BargeInConfig":
        raw = raw or {}
        sources_raw = raw.get("sources", {}) or {}
        sources = {
            name: _validate_and_normalize_override(cls, override or {}, f"barge_in.sources.{name}")
            for name, override in sources_raw.items()
        }
        return cls(
            enabled=raw.get("enabled", cls.enabled),
            energy_floor=float(raw.get("energy_floor", cls.energy_floor)),
            min_duration_ms=raw.get("min_duration_ms", cls.min_duration_ms),
            post_playback_guard_ms=raw.get(
                "post_playback_guard_ms", cls.post_playback_guard_ms
            ),
            sources=sources,
        )

    def resolve(self, source: str) -> "BargeInConfig":
        """The effective barge-in policy for `source`: the global policy
        unchanged if `source` has no override, or the merged policy
        otherwise (D-12)."""
        override = self.sources.get(source)
        if not override:
            return self
        return replace(self, **override)


@dataclass(frozen=True)
class McpServerConfig:
    """One `mcp.servers.<name>` block: the stdio child's args and env.

    Parsed but not yet consumed. Phase 1 spawns exactly one built-in tool
    server and hardcodes how, in `McpToolHost.start`; this block is the shape
    Phase 6 reads once a plugin is any MCP server an admin adds.

    `args` deliberately carries no interpreter. The child imports the same
    `mcp` SDK this process does, so it must run under the same one --
    `McpToolHost.start` spawns it with `sys.executable`. A configurable
    interpreter string is a foot-gun here, not a feature: a literal "python3"
    is whatever PATH resolves first, and outside an activated virtualenv that
    is the system interpreter with no SDK installed.
    """

    args: tuple[str, ...] = field(default_factory=tuple)
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: dict | None) -> "McpServerConfig":
        raw = raw or {}
        if "command" in raw:
            raise ConfigError(
                "mcp server blocks take 'args', not 'command': the interpreter is "
                "not configurable -- the child runs under the same interpreter as "
                "this process so it shares the mcp SDK"
            )
        args = raw.get("args", ())
        if isinstance(args, str):
            raise ConfigError("mcp server args must be a list, not a string")
        return cls(args=tuple(args), env=dict(raw.get("env", {})))


@dataclass(frozen=True)
class MacroActionConfig:
    """One action a macro runs -- the same `tool`/`arguments` shape a
    model-issued tool call carries.

    `arguments` is not validated here: the tool schema and `allow_call`
    (`mcp/spire_mcp/safety.py`) both check it at fire time, and duplicating
    that check in this module would create the second path `safety.py`'s own
    doctrine forbids.
    """

    tool: str = ""
    arguments: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: dict | None) -> "MacroActionConfig":
        raw = raw or {}
        tool = raw.get("tool", "")
        if not tool:
            raise ConfigError("a macro action is missing its 'tool' name")
        return cls(tool=tool, arguments=dict(raw.get("arguments", {})))


@dataclass(frozen=True)
class MacroConfig:
    """One `macros:` entry: a phrase (plus aliases) that skips the language
    model and runs a fixed list of actions (D-10, D-11).

    A zero-action macro would make its precached `reply` an unconditional
    lie -- there is nothing that could have succeeded -- so `from_config`
    refuses one. `normalized_keys` is the deduplicated set of `normalize()`
    applied to `phrase` and every alias; `Config.from_config` walks it across
    the whole macro list to find a cross-macro collision, because that check
    spans more than one macro and cannot live on a single `MacroConfig`.
    """

    phrase: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    reply: str = ""
    actions: tuple[MacroActionConfig, ...] = field(default_factory=tuple)

    @classmethod
    def from_config(cls, raw: dict | None) -> "MacroConfig":
        raw = raw or {}
        phrase = raw.get("phrase", "")
        if not phrase:
            raise ConfigError("a macro is missing its 'phrase'")
        reply = raw.get("reply", "")
        if not reply:
            raise ConfigError(f"macro {phrase!r} is missing its 'reply'")
        aliases_raw = raw.get("aliases", ())
        if isinstance(aliases_raw, str):
            raise ConfigError(f"macro {phrase!r} aliases must be a list, not a string")
        actions_raw = raw.get("actions", ())
        if not actions_raw:
            raise ConfigError(
                f"macro {phrase!r} has no actions -- a zero-action macro's "
                "precached reply would be an unconditional lie"
            )
        return cls(
            phrase=phrase,
            aliases=tuple(aliases_raw),
            reply=reply,
            actions=tuple(MacroActionConfig.from_config(a) for a in actions_raw),
        )

    @property
    def normalized_keys(self) -> frozenset[str]:
        """`normalize()` applied to `phrase` and every alias, deduplicated --
        a macro whose own alias normalizes to its own phrase yields one key,
        which is what makes that self-collision legal while a cross-macro
        collision is not.
        """
        return frozenset(normalize(k) for k in (self.phrase, *self.aliases))


@dataclass(frozen=True)
class Config:
    """The top-level configuration: one section per subsystem, plus the
    safety policy built from the `safety:` block.

    `Config.from_config` is the only place `Policy` is constructed from
    configuration (D-12, D-13) -- `Policy` is imported from
    `spire_mcp.safety`, never copied or re-implemented here.

    Phase 2 adds six sections here: `camera` and `speaker` are read by the
    camera `AudioSource` and its FIFO-backed speaker supervisor (plan
    02-03); `wake` is read by the wake-word detector (plan 02-04); `gate` is
    read by the gate that decides whether a wake hit counts (plan
    02-04/02-05); `barge_in` is read by `_speak`'s interrupt point in
    `turn/controller.py` (plan 02-06); `session` is read by the session
    recorder and the retention sweep (plan 02-07/02-08).
    """

    server: ServerConfig
    stt: SttConfig
    brain: BrainConfig
    tts: TtsConfig
    camera: CameraConfig
    speaker: SpeakerConfig
    wake: WakeConfig
    gate: GateConfig
    barge_in: BargeInConfig
    session: SessionConfig
    mcp_servers: dict[str, McpServerConfig]
    policy: Policy
    # A tuple, not a dict: `macros:` is a list in the config file and there is
    # no natural name key the way `mcp.servers` has one.
    macros: tuple[MacroConfig, ...] = ()
    # The `safety:` block exactly as written, kept alongside the parsed
    # `policy` because the process that ENFORCES the policy is the MCP child,
    # not this one. It receives an explicit env, not this Config object, so
    # the block is forwarded to it verbatim rather than re-serialized from
    # `Policy` -- one parser, in `safety.py`, on both sides of the boundary.
    raw_safety: dict | None = None

    @classmethod
    def from_config(cls, raw: dict | None) -> "Config":
        raw = raw or {}
        mcp_servers_raw = raw.get("mcp", {}).get("servers", {}) or {}
        macros_raw = raw.get("macros", ()) or ()
        macros = tuple(MacroConfig.from_config(m) for m in macros_raw)
        _check_macros_do_not_collide(macros)
        return cls(
            server=ServerConfig.from_config(raw.get("server")),
            stt=SttConfig.from_config(raw.get("stt")),
            brain=BrainConfig.from_config(raw.get("brain")),
            tts=TtsConfig.from_config(raw.get("tts")),
            camera=CameraConfig.from_config(raw.get("camera")),
            speaker=SpeakerConfig.from_config(raw.get("speaker")),
            wake=WakeConfig.from_config(raw.get("wake")),
            gate=GateConfig.from_config(raw.get("gate")),
            barge_in=BargeInConfig.from_config(raw.get("barge_in")),
            session=SessionConfig.from_config(raw.get("debug")),
            mcp_servers={
                name: McpServerConfig.from_config(server_raw)
                for name, server_raw in mcp_servers_raw.items()
            },
            policy=Policy.from_config(raw.get("safety")),
            macros=macros,
            raw_safety=raw.get("safety"),
        )


def _check_macros_do_not_collide(macros: tuple[MacroConfig, ...]) -> None:
    """Raise `ConfigError` on the first pair of macros whose normalized keys
    collide, naming both by their written (un-normalized) phrases.

    Lives at module level, not on `MacroConfig`, because the check spans the
    whole macro list -- a single macro has no way to know about another one.
    Naming both macros is what makes the error actionable: an error naming
    only the second one sends the operator to the wrong line.
    """
    seen: dict[str, MacroConfig] = {}
    for macro in macros:
        for key in macro.normalized_keys:
            earlier = seen.get(key)
            if earlier is not None and earlier is not macro:
                raise ConfigError(
                    f"macros {earlier.phrase!r} and {macro.phrase!r} both "
                    f"normalize to {key!r} -- one would silently shadow the "
                    "other at runtime"
                )
            seen[key] = macro


def load_config(path: str | os.PathLike) -> Config:
    """Read `path`, expand its environment placeholders, and build a `Config`."""
    with open(path, encoding="utf-8") as fh:
        raw_text = fh.read()
    expanded_text = expand_env(raw_text)
    raw = yaml.safe_load(expanded_text)
    return Config.from_config(raw)
