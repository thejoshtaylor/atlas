"""Config loader: env expansion, per-section validation, the safety handoff.

Requirement coverage: SRC-01 (env expansion, transport selector), SAFE-01
(the safety block reaches `Policy.from_config` unchanged).
"""

import pytest

from spire_mcp.safety import Denied, allow_call
from spire_voice.config import Config, ConfigError, ServerConfig, expand_env


def _minimal_raw_config() -> dict:
    """A full raw config dict, shaped like config.example.yaml, with
    placeholder values already in place of every `${NAME}` -- this
    exercises `Config.from_config` directly, without going through
    `expand_env`, which env-expansion tests cover on their own.
    """
    return {
        "server": {"bind_host": "127.0.0.1", "port": 8080, "transport": "websocket"},
        "stt": {
            "url": "wss://api.x.ai/v1/stt",
            "api_key": "test-key",
            "endpointing_ms": 200,
            "smart_turn": 0.7,
            "smart_turn_timeout_ms": 1200,
            "vad_threshold": 0.08,
            "interim_results": True,
            "language": "en",
            "max_utterance_s": 15,
        },
        "brain": {
            "base_url": "https://api.x.ai/v1",
            "api_key": "test-key",
            "model": "grok-4.6",
            "cache_system_prompt": True,
            "temperature": 0.0,
            "max_tokens": 400,
            "max_tool_rounds": 3,
        },
        "tts": {
            "url": "https://api.x.ai/v1/tts",
            "api_key": "test-key",
            "voice_id": "eve",
            "language": "en",
            "codec": "alaw",
            "sample_rate": 8000,
            "browser_codec": "pcm",
            "browser_sample_rate": 24000,
            "optimize_streaming_latency": 2,
        },
        "mcp": {
            "servers": {
                "ha": {
                    "args": ["-m", "spire_mcp.ha"],
                    "env": {"HA_URL": "http://ha.invalid", "HA_TOKEN": "test-token"},
                },
            },
        },
        "safety": {},
    }


def test_env_expansion_replaces_placeholders(monkeypatch):
    # A value that itself contains "${" is not expanded a second time --
    # expansion runs once, over the raw text, before the YAML parse.
    monkeypatch.setenv("SPIRE_TEST_VAR", "value-with-${NOT_EXPANDED}")
    raw = "key: ${SPIRE_TEST_VAR}\n"
    assert expand_env(raw) == "key: value-with-${NOT_EXPANDED}\n"


def test_missing_env_var_is_a_startup_error(monkeypatch):
    monkeypatch.delenv("SPIRE_TEST_MISSING_VAR", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        expand_env("key: ${SPIRE_TEST_MISSING_VAR}\n")
    assert "SPIRE_TEST_MISSING_VAR" in str(exc_info.value)


def test_unknown_transport_value_is_a_startup_error():
    with pytest.raises(ConfigError):
        ServerConfig.from_config({"transport": "carrier-pigeon"})

    assert ServerConfig.from_config({"transport": "websocket"}).transport == "websocket"
    assert ServerConfig.from_config({"transport": "webrtc"}).transport == "webrtc"

    defaults = ServerConfig.from_config({})
    assert defaults.bind_host == "127.0.0.1"
    assert defaults.port == 8080
    assert defaults.transport == "websocket"


def test_safety_block_is_handed_to_policy_from_config():
    raw = _minimal_raw_config()
    raw["safety"] = {
        "mode": "allowlist_only",
        "allow_entities": ["light.example_lamp"],
    }
    config = Config.from_config(raw)

    # The allowed entity passes allow_call; a different, unreviewed entity
    # is denied -- proving the block reached Policy.from_config intact,
    # not a copy or a reinterpretation.
    assert allow_call(config.policy, "light", "turn_on", "light.example_lamp")[2] == [
        "light.example_lamp"
    ]
    with pytest.raises(Denied):
        allow_call(config.policy, "light", "turn_on", "light.example_other")


def test_mcp_server_block_rejects_a_configurable_interpreter():
    """`command:` is refused outright, naming why, rather than silently ignored.

    The old spelling took a full argv whose first element was the interpreter,
    and `config.example.yaml` shipped `["python3", "-m", "spire_mcp.ha"]`. That
    value crashes startup everywhere `python3` is not the venv interpreter,
    because the child needs the same `mcp` SDK this process has. The key was
    also parsed and then consumed by nothing, so an operator could edit it,
    observe no effect, and reasonably conclude the interpreter was theirs to
    choose.

    Failing loudly at load, naming the replacement, is the honest behavior: a
    config that silently ignores what you wrote is worse than one that refuses
    it.
    """
    import pytest

    from spire_voice.config import ConfigError, McpServerConfig

    with pytest.raises(ConfigError) as exc:
        McpServerConfig.from_config({"command": ["python3", "-m", "spire_mcp.ha"]})
    assert "args" in str(exc.value)

    ok = McpServerConfig.from_config({"args": ["-m", "spire_mcp.ha"], "env": {"HA_URL": "u"}})
    assert ok.args == ("-m", "spire_mcp.ha")
    assert ok.env == {"HA_URL": "u"}
