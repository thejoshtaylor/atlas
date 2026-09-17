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
            "models": [{"model": "grok-4.20-0309-non-reasoning"}, {"model": "grok-4.6"}],
            "filler_after_ms": 600,
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


# --- Task 1: brain.models replaces brain.model, position decides the top tier ---


def test_brain_models_list_produces_tiers_in_written_order():
    from spire_voice.config import BrainConfig

    brain = BrainConfig.from_config(
        {"models": [{"model": "grok-4.20-0309-non-reasoning"}, {"model": "grok-4.6"}]}
    )
    assert [tier.model for tier in brain.models] == ["grok-4.20-0309-non-reasoning", "grok-4.6"]


def test_brain_model_key_is_a_startup_error_naming_models():
    from spire_voice.config import BrainConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        BrainConfig.from_config({"model": "grok-4.6"})
    assert "models" in str(exc.value)


def test_brain_empty_or_absent_models_list_is_a_startup_error():
    from spire_voice.config import BrainConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        BrainConfig.from_config({"models": []})
    assert "models" in str(exc.value)

    with pytest.raises(ConfigError):
        BrainConfig.from_config({})


def test_brain_models_must_be_a_list_not_a_string():
    from spire_voice.config import BrainConfig, ConfigError

    with pytest.raises(ConfigError):
        BrainConfig.from_config({"models": "grok-4.6"})


def test_brain_single_tier_is_both_the_only_candidate_and_the_top_tier():
    from spire_voice.config import BrainConfig

    brain = BrainConfig.from_config({"models": [{"model": "grok-4.6"}]})
    assert brain.top_tier.model == "grok-4.6"
    assert brain.triage_tiers == ()


def test_brain_tier_rejects_a_configurable_calls_tools_key():
    from spire_voice.config import BrainConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        BrainConfig.from_config({"models": [{"model": "grok-4.6", "calls_tools": True}]})
    assert "position" in str(exc.value)


def test_brain_top_tier_is_the_last_entry_and_triage_tiers_is_the_rest():
    from spire_voice.config import BrainConfig

    brain = BrainConfig.from_config(
        {"models": [{"model": "a"}, {"model": "b"}, {"model": "c"}]}
    )
    assert brain.top_tier.model == "c"
    assert [tier.model for tier in brain.triage_tiers] == ["a", "b"]


def test_xai_brain_resolves_against_top_tier_by_default_and_explicit_model_override():
    from spire_voice.config import BrainConfig
    from spire_voice.providers.brain_xai import XaiBrain

    brain_config = BrainConfig.from_config(
        {"api_key": "test-key", "models": [{"model": "a"}, {"model": "b"}]}
    )

    default_brain = XaiBrain(brain_config)
    assert default_brain._model == "b"

    explicit_brain = XaiBrain(brain_config, model="a")
    assert explicit_brain._model == "a"


# --- Task 2: macros: block and the normalization that decides sameness ---


def test_normalize_folds_case_punctuation_and_whitespace():
    from spire_voice.turn.macros import normalize

    assert normalize("  Good, Night!!  ") == "good night"
    assert normalize("GOOD NIGHT") == "good night"


def test_normalize_compares_by_code_point_through_nfkc():
    from spire_voice.turn.macros import normalize

    # A precomposed accented character and its NFKC-equivalent decomposed
    # form (base letter + combining accent) differ in code point count and
    # UTF-8 byte length, but must normalize to the same string.
    precomposed = "café"
    decomposed = "café"
    assert normalize(precomposed) == normalize(decomposed)


def test_macro_collision_on_phrase_raises_naming_both():
    from spire_voice.config import Config, ConfigError

    raw = _minimal_raw_config()
    raw["macros"] = [
        {"phrase": "Good Night", "reply": "ok", "actions": [{"tool": "ha_call_service"}]},
        {"phrase": "good   night!", "reply": "ok", "actions": [{"tool": "ha_call_service"}]},
    ]
    with pytest.raises(ConfigError) as exc:
        Config.from_config(raw)
    assert "Good Night" in str(exc.value)
    assert "good   night!" in str(exc.value)


def test_macro_alias_colliding_with_another_macros_phrase_raises_naming_both():
    from spire_voice.config import Config, ConfigError

    raw = _minimal_raw_config()
    raw["macros"] = [
        {"phrase": "good night", "reply": "ok", "actions": [{"tool": "ha_call_service"}]},
        {
            "phrase": "movie time",
            "aliases": ["Good Night"],
            "reply": "ok",
            "actions": [{"tool": "ha_call_service"}],
        },
    ]
    with pytest.raises(ConfigError) as exc:
        Config.from_config(raw)
    assert "good night" in str(exc.value)
    assert "movie time" in str(exc.value)


def test_macro_alias_matching_its_own_phrase_loads_cleanly_as_one_key():
    from spire_voice.config import MacroConfig

    macro = MacroConfig.from_config(
        {
            "phrase": "good night",
            "aliases": ["GOOD NIGHT!"],
            "reply": "ok",
            "actions": [{"tool": "ha_call_service"}],
        }
    )
    assert macro.normalized_keys == frozenset({"good night"})


def test_macro_blank_phrase_is_a_startup_error():
    from spire_voice.config import ConfigError, MacroConfig

    with pytest.raises(ConfigError):
        MacroConfig.from_config({"reply": "ok", "actions": [{"tool": "ha_call_service"}]})


def test_macro_blank_reply_is_a_startup_error_naming_the_macro():
    from spire_voice.config import ConfigError, MacroConfig

    with pytest.raises(ConfigError) as exc:
        MacroConfig.from_config({"phrase": "good night", "actions": [{"tool": "ha_call_service"}]})
    assert "good night" in str(exc.value)


def test_macro_zero_actions_is_a_startup_error_naming_the_macro():
    from spire_voice.config import ConfigError, MacroConfig

    with pytest.raises(ConfigError) as exc:
        MacroConfig.from_config({"phrase": "good night", "reply": "ok", "actions": []})
    assert "good night" in str(exc.value)


def test_macro_with_no_aliases_key_loads_cleanly_matching_on_phrase_alone():
    from spire_voice.config import MacroConfig

    macro = MacroConfig.from_config(
        {"phrase": "good night", "reply": "ok", "actions": [{"tool": "ha_call_service"}]}
    )
    assert macro.aliases == ()
    assert macro.normalized_keys == frozenset({"good night"})


def test_macro_action_blank_tool_name_is_a_startup_error():
    from spire_voice.config import ConfigError, MacroActionConfig

    with pytest.raises(ConfigError):
        MacroActionConfig.from_config({"arguments": {}})

    with pytest.raises(ConfigError):
        MacroActionConfig.from_config({"tool": "", "arguments": {}})


def test_config_and_turn_macros_import_in_either_order():
    """Proves no runtime import cycle between the two modules -- config.py
    imports normalize() from turn/macros.py, and turn/macros.py must never
    import spire_voice.config back."""
    import importlib

    import spire_voice.config  # noqa: F401
    import spire_voice.turn.macros  # noqa: F401

    importlib.reload(spire_voice.turn.macros)
    importlib.reload(spire_voice.config)


# --- Task 3: the two tts: precache keys, wired end to end ---


def test_tts_config_parses_cache_dir_and_precache():
    from spire_voice.config import TtsConfig

    tts = TtsConfig.from_config({"cache_dir": "/tmp/cache", "precache": ["ok", "done"]})
    assert tts.cache_dir == "/tmp/cache"
    assert tts.precache == ("ok", "done")


def test_tts_config_defaults_cache_dir_and_precache_when_absent():
    from spire_voice.config import TtsConfig

    tts = TtsConfig.from_config({})
    assert tts.cache_dir == "/data/tts-cache"
    assert tts.precache == ()


def test_example_config_loads_end_to_end(monkeypatch):
    from spire_voice.config import load_config

    for name in (
        "XAI_API_KEY",
        "TAPO_USER",
        "TAPO_PASSWORD",
        "SPEAKER_ENSURE_URL",
        "HA_URL",
        "HA_TOKEN",
    ):
        monkeypatch.setenv(name, "test-value")

    config = load_config("config/config.example.yaml")

    assert len(config.brain.models) == 3
    assert config.brain.top_tier.model == "grok-4.6"
    assert len(config.macros) == 2
    assert config.macros[0].normalized_keys == {"good night", "goodnight", "night night"}
    assert config.tts.cache_dir == "/data/tts-cache"
    assert config.tts.precache == (
        "ok",
        "done",
        "sorry, i didn't catch that",
        "i can't do that one",
    )

    # Phase 2's six new sections: the example file and the parser drifting
    # apart is the failure this block prevents.
    assert config.camera.rtsp_url
    assert config.speaker.fifo_path
    assert config.wake.engine == "vosk"
    assert config.gate.sources  # the browser override is written explicitly
    assert config.barge_in.enabled is True
    # CR-02 fix: the shipped example turns barge-in off for the camera
    # specifically -- its microphone and speaker are the same device with
    # no AEC, and BargeInMonitor's own gate (energy floor + guard window,
    # not known-output correlation) cannot tell the assistant's own voice
    # apart from a real interruption on that path. The global default
    # (asserted above) stays on for every other source.
    assert config.barge_in.resolve("camera").enabled is False
    # Plan 02-12's gap closure: the correlation ships inert -- off by
    # default globally, so the camera's own resolved policy inherits it
    # rather than needing a second override written for it.
    assert config.barge_in.correlation_enabled is False
    assert config.barge_in.resolve("camera").correlation_enabled is False
    assert config.barge_in.correlation_tolerance == 0.03
    assert config.barge_in.tracking_adaptation_rate == 0.1
    assert config.session.retain_days == 7

    # Plan 02-11's gap closure: the calibration block loads, and its route
    # ships off by default (T-02-47) -- the example file and the parser
    # drifting apart on this specific key is the failure this line prevents.
    assert config.calibration.dir == "/data/calibration"
    assert config.calibration.route_enabled is False


# --- Task 3: one rejection test per Phase 2 configuration path ---


def test_camera_config_rejects_an_unsupported_encoding():
    from spire_voice.config import CameraConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        CameraConfig.from_config({"encoding": "opus"})
    assert "opus" in str(exc.value)


def test_speaker_config_rejects_a_non_positive_respawn_backoff():
    from spire_voice.config import ConfigError, SpeakerConfig

    with pytest.raises(ConfigError):
        SpeakerConfig.from_config({"respawn_backoff_s": 0})
    with pytest.raises(ConfigError):
        SpeakerConfig.from_config({"respawn_backoff_s": -1.0})


def test_speaker_config_rejects_a_non_positive_reopen_timeout():
    """CR-04: a zero or negative `reopen_timeout_s` would fail every FIFO
    reopen immediately, including one a respawning ffmpeg child was about
    to win -- reject it at config load, same posture as respawn_backoff_s
    above."""
    from spire_voice.config import ConfigError, SpeakerConfig

    with pytest.raises(ConfigError):
        SpeakerConfig.from_config({"reopen_timeout_s": 0})
    with pytest.raises(ConfigError):
        SpeakerConfig.from_config({"reopen_timeout_s": -1.0})


def test_session_config_rejects_a_non_positive_retention():
    from spire_voice.config import ConfigError, SessionConfig

    with pytest.raises(ConfigError):
        SessionConfig.from_config({"retain_days": 0})
    with pytest.raises(ConfigError):
        SessionConfig.from_config({"retain_days": -7})


def test_wake_config_rejects_an_unknown_engine():
    from spire_voice.config import ConfigError, WakeConfig

    with pytest.raises(ConfigError) as exc:
        WakeConfig.from_config({"engine": "shazam"})
    assert "openwakeword" in str(exc.value)
    assert "vosk" in str(exc.value)


def test_gate_config_still_carrying_an_identity_key_is_a_startup_error():
    from spire_voice.config import ConfigError, GateConfig

    with pytest.raises(ConfigError) as exc:
        GateConfig.from_config({"require_face": True})
    assert "require_face" in str(exc.value)

    with pytest.raises(ConfigError) as exc:
        GateConfig.from_config({"face_names": ["placeholder"]})
    assert "face_names" in str(exc.value)

    with pytest.raises(ConfigError) as exc:
        GateConfig.from_config({"face_window_s": 10})
    assert "face_window_s" in str(exc.value)


def test_source_override_naming_an_unknown_field_is_a_startup_error():
    from spire_voice.config import ConfigError, GateConfig

    with pytest.raises(ConfigError) as exc:
        GateConfig.from_config({"sources": {"browser": {"require_video": True}}})
    assert "require_video" in str(exc.value)


def test_gate_resolve_returns_global_policy_unchanged_with_no_override():
    from spire_voice.config import GateConfig

    gate = GateConfig.from_config({"mute_when_playing": ["media_player.example_tv"]})
    assert gate.resolve("camera") == gate


def test_gate_resolve_returns_merged_policy_for_a_source_with_an_override():
    from spire_voice.config import GateConfig

    gate = GateConfig.from_config(
        {
            "mute_when_playing": ["media_player.example_tv"],
            "sources": {"browser": {"mute_when_playing": []}},
        }
    )
    resolved = gate.resolve("browser")
    assert resolved.mute_when_playing == ()
    assert gate.mute_when_playing == ("media_player.example_tv",)


def test_wake_threshold_survives_the_load_as_a_float_with_no_rounding():
    from spire_voice.config import WakeConfig

    wake = WakeConfig.from_config({"openwakeword": {"threshold": 0.123456}})
    assert wake.openwakeword.threshold == 0.123456
    assert isinstance(wake.openwakeword.threshold, float)


def test_wake_source_override_of_a_nested_engine_block_builds_a_real_subconfig():
    """WR-01 fix: `openwakeword`/`vosk` are built via
    `field(default_factory=...)`, so `getattr(WakeConfig, "openwakeword",
    None)` is `None` -- before this fix, `_validate_and_normalize_override`
    only knew how to coerce a field whose *global* default was a `tuple`,
    so a per-source override of a nested engine block passed through as a
    raw `dict` untouched. `WakeConfig.resolve("camera").openwakeword` would
    then be a `dict`, not an `OpenWakeWordConfig`, and the very next
    `_resolve_threshold(...)` call in `sources/runner.py` would raise
    `AttributeError: 'dict' object has no attribute 'threshold'` at
    startup -- a crash whose message does not point at the actual mistake,
    contradicting this module's own "raised, not returned" doctrine."""
    from spire_voice.config import OpenWakeWordConfig, WakeConfig

    wake = WakeConfig.from_config(
        {
            "engine": "openwakeword",
            "sources": {"camera": {"openwakeword": {"threshold": 0.9}}},
        }
    )
    resolved = wake.resolve("camera").openwakeword
    assert isinstance(resolved, OpenWakeWordConfig), (
        f"a per-source openwakeword override must build a real OpenWakeWordConfig, "
        f"not a {type(resolved).__name__}"
    )
    assert resolved.threshold == 0.9
    # The rest of the sub-config falls back to OpenWakeWordConfig's own
    # defaults, the same as a top-level `openwakeword:` block would.
    assert resolved.model_path == OpenWakeWordConfig().model_path
    # The *global* wake config is untouched by the per-source override.
    assert wake.openwakeword.threshold == OpenWakeWordConfig().threshold


def test_wake_source_override_of_a_nested_engine_block_rejects_a_non_mapping():
    from spire_voice.config import ConfigError, WakeConfig

    with pytest.raises(ConfigError) as exc:
        WakeConfig.from_config({"sources": {"camera": {"openwakeword": 0.9}}})
    assert "openwakeword" in str(exc.value)


# --- Plan 02-11: CalibrationConfig's three named ConfigError rejections ---


def test_calibration_config_rejects_a_non_positive_probe_duration():
    from spire_voice.config import CalibrationConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        CalibrationConfig.from_config({"probe_duration_s": 0})
    assert "probe_duration_s" in str(exc.value)

    with pytest.raises(ConfigError):
        CalibrationConfig.from_config({"probe_duration_s": -1.0})


def test_calibration_config_rejects_a_negative_settle_period():
    from spire_voice.config import CalibrationConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        CalibrationConfig.from_config({"settle_s": -0.1})
    assert "settle_s" in str(exc.value)

    # Zero is a valid settle period -- "negative" is the rejection, not
    # "non-positive": a calibration with no settle wait at all is a real,
    # if aggressive, operator choice, and never having to wait is not a
    # configuration error the way a negative duration is.
    CalibrationConfig.from_config({"settle_s": 0})


def test_calibration_config_rejects_a_non_positive_max_age_days():
    from spire_voice.config import CalibrationConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        CalibrationConfig.from_config({"max_age_days": 0})
    assert "max_age_days" in str(exc.value)

    with pytest.raises(ConfigError):
        CalibrationConfig.from_config({"max_age_days": -5})


def test_calibration_config_rejects_a_negative_tail():
    from spire_voice.config import CalibrationConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        CalibrationConfig.from_config({"tail_s": -0.01})
    assert "tail_s" in str(exc.value)


def test_calibration_config_route_enabled_defaults_off():
    from spire_voice.config import CalibrationConfig

    assert CalibrationConfig.from_config(None).route_enabled is False
    assert CalibrationConfig.from_config({}).route_enabled is False


# --- Plan 02-12: BargeInConfig's two correlation ConfigError rejections ---


def test_barge_in_config_rejects_a_negative_correlation_tolerance():
    from spire_voice.config import BargeInConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        BargeInConfig.from_config({"correlation_tolerance": -0.01})
    assert "correlation_tolerance" in str(exc.value)

    # Zero is a valid tolerance -- "negative" is the rejection.
    BargeInConfig.from_config({"correlation_tolerance": 0})


def test_barge_in_config_rejects_an_adaptation_rate_outside_its_bounds():
    from spire_voice.config import BargeInConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        BargeInConfig.from_config({"tracking_adaptation_rate": 0})
    assert "tracking_adaptation_rate" in str(exc.value)

    with pytest.raises(ConfigError):
        BargeInConfig.from_config({"tracking_adaptation_rate": 1.5})

    with pytest.raises(ConfigError):
        BargeInConfig.from_config({"tracking_adaptation_rate": -0.1})

    # Exactly 1 is the inclusive upper bound -- a valid, if aggressive,
    # adaptation rate.
    BargeInConfig.from_config({"tracking_adaptation_rate": 1.0})


def test_barge_in_config_correlation_defaults_off():
    from spire_voice.config import BargeInConfig

    resolved = BargeInConfig.from_config(None)
    assert resolved.correlation_enabled is False


def test_barge_in_config_correlation_enabled_survives_a_per_source_override_round_trip():
    """The camera's own override sets `enabled: false` (the shipped
    default) alongside whatever `correlation_enabled` the global block
    carries -- proving the two keys coexist under `resolve()` the same way
    every other barge-in field already does."""
    from spire_voice.config import BargeInConfig

    config = BargeInConfig.from_config(
        {"correlation_enabled": True, "sources": {"camera": {"enabled": False}}}
    )
    resolved = config.resolve("camera")
    assert resolved.enabled is False
    assert resolved.correlation_enabled is True
