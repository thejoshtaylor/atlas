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
from dataclasses import dataclass, field

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
    """

    server: ServerConfig
    stt: SttConfig
    brain: BrainConfig
    tts: TtsConfig
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
