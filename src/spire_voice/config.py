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
class BrainConfig:
    """The `brain:` block: the language model endpoint and tool-round cap."""

    base_url: str = "https://api.x.ai/v1"
    api_key: str = ""
    model: str = "grok-4.6"
    cache_system_prompt: bool = True
    temperature: float = 0.0
    max_tokens: int = 400
    max_tool_rounds: int = 3

    @classmethod
    def from_config(cls, raw: dict | None) -> "BrainConfig":
        raw = raw or {}
        return cls(
            base_url=raw.get("base_url", cls.base_url),
            api_key=raw.get("api_key", cls.api_key),
            model=raw.get("model", cls.model),
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
            raw_safety=raw.get("safety"),
        )


def load_config(path: str | os.PathLike) -> Config:
    """Read `path`, expand its environment placeholders, and build a `Config`."""
    with open(path, encoding="utf-8") as fh:
        raw_text = fh.read()
    expanded_text = expand_env(raw_text)
    raw = yaml.safe_load(expanded_text)
    return Config.from_config(raw)
