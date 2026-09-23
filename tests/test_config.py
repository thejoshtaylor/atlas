"""Config loader: env expansion, per-section validation, the retired safety key.

Requirement coverage: SRC-01 (env expansion, transport selector). The safety
block's own parsing lives in `mcp/spire_mcp/safety.py` and is exercised by
`tests/test_safety_integration.py`; this file only proves `Config` rejects a
`safety:` key rather than reading one (D-11, Phase 3).
"""

import pytest

from spire_voice.config import Config, ConfigError, ServerConfig, SttConfig, expand_env


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
        "database": {"url": "postgresql+asyncpg://spire:test-value@db.invalid:5432/spire"},
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


def test_server_timezone_defaults_to_unset():
    assert ServerConfig.from_config({}).timezone is None
    assert ServerConfig.from_config({"transport": "websocket"}).timezone is None


def test_server_timezone_accepts_a_real_zoneinfo_name():
    config = ServerConfig.from_config({"timezone": "America/Los_Angeles"})
    assert config.timezone == "America/Los_Angeles"


def test_server_timezone_rejects_an_unrecognized_name():
    with pytest.raises(ConfigError) as exc_info:
        ServerConfig.from_config({"timezone": "Mars/Olympus_Mons"})
    assert "server.timezone" in str(exc_info.value)


def test_stt_local_settings_have_shipped_defaults():
    """D-09/D-11: the local speech-to-text entry's three keys ship with a
    default even when the operator never sets them -- the xAI-only config
    dict `_minimal_raw_config()` uses elsewhere in this file carries none
    of these three, and loading must not fail on their absence."""
    config = SttConfig.from_config({"url": "wss://api.x.ai/v1/stt"})

    assert config.local_model_dir == "/models/faster-whisper"
    assert config.local_model_size == "small"
    assert config.local_compute_type == "int8"


def test_stt_local_compute_type_rejects_an_unsupported_value():
    with pytest.raises(ConfigError) as exc_info:
        SttConfig.from_config({"local_compute_type": "float64"})
    assert "stt.local_compute_type" in str(exc_info.value)

    for compute_type in ("int8", "int8_float32", "float32"):
        assert SttConfig.from_config({"local_compute_type": compute_type}).local_compute_type == compute_type


def test_safety_key_still_present_is_a_startup_error_naming_it():
    """D-11: the policy `safety:` used to carry now lives in the database,
    seeded by the first migration -- a config file still carrying the key
    raises `ConfigError` naming it, the same way `brain.model` and
    `gate.require_face` already do for the keys retired before it."""
    raw = _minimal_raw_config()
    raw["safety"] = {
        "mode": "allowlist_only",
        "allow_entities": ["light.example_lamp"],
    }
    with pytest.raises(ConfigError) as exc:
        Config.from_config(raw)
    assert "safety" in str(exc.value)

    # An explicitly empty block is still the key being present -- rejected
    # the same way, not treated as "nothing to reject."
    raw_empty = _minimal_raw_config()
    raw_empty["safety"] = {}
    with pytest.raises(ConfigError) as exc:
        Config.from_config(raw_empty)
    assert "safety" in str(exc.value)

    # No safety: key at all loads cleanly -- this is the shape every other
    # test in this file already exercises via _minimal_raw_config().
    Config.from_config(_minimal_raw_config())


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


# --- 260922-woc: brain.turn_timeout_s bounds the whole tier race ---


def test_brain_turn_timeout_s_defaults_to_25_seconds():
    from spire_voice.config import BrainConfig

    brain = BrainConfig.from_config({"models": [{"model": "grok-4.6"}]})
    assert brain.turn_timeout_s == 25.0


def test_brain_turn_timeout_s_is_configurable():
    from spire_voice.config import BrainConfig

    brain = BrainConfig.from_config({"models": [{"model": "grok-4.6"}], "turn_timeout_s": 10})
    assert brain.turn_timeout_s == 10.0


def test_brain_turn_timeout_s_must_be_positive():
    from spire_voice.config import BrainConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        BrainConfig.from_config({"models": [{"model": "grok-4.6"}], "turn_timeout_s": 0})
    assert "turn_timeout_s" in str(exc.value)

    with pytest.raises(ConfigError):
        BrainConfig.from_config({"models": [{"model": "grok-4.6"}], "turn_timeout_s": -1})


# --- 260922-lim: brain.local_intents gates the local on/off matcher ---


def test_brain_local_intents_defaults_to_true():
    from spire_voice.config import BrainConfig

    brain = BrainConfig.from_config({"models": [{"model": "grok-4.6"}]})
    assert brain.local_intents is True


def test_brain_local_intents_is_configurable():
    from spire_voice.config import BrainConfig

    brain = BrainConfig.from_config(
        {"models": [{"model": "grok-4.6"}], "local_intents": False}
    )
    assert brain.local_intents is False


def test_brain_local_intents_must_be_a_bool():
    from spire_voice.config import BrainConfig, ConfigError

    with pytest.raises(ConfigError) as exc:
        BrainConfig.from_config({"models": [{"model": "grok-4.6"}], "local_intents": "yes"})
    assert "local_intents" in str(exc.value)


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
    """Plan 04-05: macros no longer live on `Config` (D-09), so this check
    is exercised directly against `_check_macros_do_not_collide` -- the
    same function `alembic/versions/0005_macro_tables.py`'s seed step and
    plan 04-06's write routes both call, now that `Config.from_config` no
    longer parses `macros:` at all."""
    from spire_voice.config import ConfigError, MacroConfig, _check_macros_do_not_collide

    macros = (
        MacroConfig.from_config(
            {"phrase": "Good Night", "reply": "ok", "actions": [{"tool": "ha_call_service"}]}
        ),
        MacroConfig.from_config(
            {"phrase": "good   night!", "reply": "ok", "actions": [{"tool": "ha_call_service"}]}
        ),
    )
    with pytest.raises(ConfigError) as exc:
        _check_macros_do_not_collide(macros)
    assert "Good Night" in str(exc.value)
    assert "good   night!" in str(exc.value)


def test_macro_alias_colliding_with_another_macros_phrase_raises_naming_both():
    from spire_voice.config import ConfigError, MacroConfig, _check_macros_do_not_collide

    macros = (
        MacroConfig.from_config(
            {"phrase": "good night", "reply": "ok", "actions": [{"tool": "ha_call_service"}]}
        ),
        MacroConfig.from_config(
            {
                "phrase": "movie time",
                "aliases": ["Good Night"],
                "reply": "ok",
                "actions": [{"tool": "ha_call_service"}],
            }
        ),
    )
    with pytest.raises(ConfigError) as exc:
        _check_macros_do_not_collide(macros)
    assert "good night" in str(exc.value)
    assert "movie time" in str(exc.value)


def test_macros_key_still_present_is_a_startup_error_naming_it():
    """D-09: macros used to carry now live in the database, seeded by the
    migration -- a config file still carrying the key raises `ConfigError`
    naming it, the same way `safety:` already does (D-11)."""
    from spire_voice.config import Config, ConfigError

    raw = _minimal_raw_config()
    raw["macros"] = [
        {"phrase": "good night", "reply": "ok", "actions": [{"tool": "ha_call_service"}]},
    ]
    with pytest.raises(ConfigError) as exc:
        Config.from_config(raw)
    assert "macros" in str(exc.value)


def test_mcp_key_still_present_is_a_startup_error_naming_it():
    """D-01 (plan 06-01): every MCP server used to carry now lives in the
    `plugins` table, seeded by `alembic/versions/0008_plugin_tables.py` --
    a config file still carrying the `mcp:` key raises `ConfigError`
    naming it, the same way `safety:`/`macros:` already do (D-11, D-09)."""
    from spire_voice.config import Config, ConfigError

    raw = _minimal_raw_config()
    raw["mcp"] = {
        "servers": {
            "ha": {
                "args": ["-m", "spire_mcp.ha"],
                "env": {"HA_URL": "http://ha.invalid", "HA_TOKEN": "test-token"},
            },
        },
    }
    with pytest.raises(ConfigError) as exc:
        Config.from_config(raw)
    assert "mcp" in str(exc.value)

    # An explicitly empty block is still the key being present -- rejected
    # the same way, matching `test_safety_key_still_present_is_a_startup_
    # error_naming_it`'s own coverage of that edge case for `safety:`.
    raw_empty = _minimal_raw_config()
    raw_empty["mcp"] = {}
    with pytest.raises(ConfigError) as exc:
        Config.from_config(raw_empty)
    assert "mcp" in str(exc.value)

    # No mcp: key at all loads cleanly -- this is the shape every other
    # test in this file already exercises via _minimal_raw_config().
    Config.from_config(_minimal_raw_config())


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
        # Phase 7 (D-15): server.bind_host -- a plain string passthrough,
        # so "test-value" is as honest a placeholder here as it is above.
        "BIND_HOST",
    ):
        monkeypatch.setenv(name, "test-value")
    # security.cookie_secure is unquoted in config.example.yaml, so
    # expansion must produce a real YAML boolean literal here, not an
    # arbitrary string -- "false" keeps the assertion below (cookie_secure
    # is False) true, matching this file's own shipped default.
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    # 260922-cmo (D-1): camera.rtsp_url is a plain string passthrough like
    # the vars in the loop above, so "test-value" would be as honest a
    # placeholder as any -- a real rtsp:// string is used instead so this
    # assertion reads the same as a genuine deployment would produce.
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    # speaker.backend is validated against a fixed allowlist
    # ({go2rtc, tapo_talk}) -- it cannot share the generic "test-value"
    # placeholder the loop above uses without failing that validation.
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    # Plan 04-03: mcp.servers.weather's two placeholders -- a coordinate,
    # not an arbitrary string, since spire_mcp.weather parses these as
    # floats (never exercised by this test, which only loads Config, but
    # a numeric-shaped value is the honest placeholder for what these
    # actually carry).
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    # database.url must be a valid postgresql+asyncpg:// string --
    # DatabaseConfig.from_config validates the scheme eagerly, unlike the
    # plain passthrough values above.
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://spire:test-value@db.invalid:5432/spire"
    )

    config = load_config("config/config.example.yaml")

    assert len(config.brain.models) == 3
    assert config.brain.top_tier.model == "grok-4.6"
    # Plan 04-05 (D-09): macros: is retired from this file -- Config no
    # longer carries a macros field at all, the same way it carries no
    # safety field (D-11). This is the load-bearing half of the assertion
    # `grep -v '^#' config/config.example.yaml | grep -c '^macros:'`
    # reports 0 already covers structurally; this proves the loader itself
    # agrees by not raising on the shipped file.
    assert not hasattr(config, "macros")
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

    # Plan 03-01's gap closure: the bootstrap seam (D-01) loads, and its
    # derived migration URL swaps the driver without touching host,
    # credentials, or database name.
    assert config.database.url == "postgresql+asyncpg://spire:test-value@db.invalid:5432/spire"
    assert config.database.migration_url == "postgresql+psycopg://spire:test-value@db.invalid:5432/spire"

    # Plan 06-01: `mcp:` (and `Config.mcp_servers`) is retired -- every MCP
    # server, weather included, is a plugin row now, seeded by
    # `alembic/versions/0008_plugin_tables.py` and read back by
    # `PluginRepository`, not by `Config`. This file's own
    # `test_mcp_key_still_present_is_a_startup_error_naming_it` covers the
    # rejection; there is no `Config` field left to assert on here.

    assert config.database.run_migrations_at_startup is True
    assert config.security.secret_key_env == "SPIRE_SECRET_KEY"
    assert config.security.access_token_ttl_s == 900
    assert config.security.refresh_token_ttl_s == 1209600
    assert config.security.cookie_secure is False


def test_example_config_camera_url_and_speaker_backend_expand_from_env(monkeypatch):
    """260922-cmo (D-1, T-CMO-02): env expansion runs over raw text before
    the YAML parse and before SpeakerConfig.from_config's allowlist check
    -- proving that a ${SPEAKER_BACKEND}-supplied value of `tapo_talk`
    still passes that allowlist, not only the shipped `go2rtc` default,
    and that camera.rtsp_url comes through as exactly what the
    environment supplied, with no embedded host or credential left in
    the committed file for it to compete with."""
    from spire_voice.config import load_config

    for name in (
        "XAI_API_KEY",
        "TAPO_USER",
        "TAPO_PASSWORD",
        "SPEAKER_ENSURE_URL",
        "HA_URL",
        "HA_TOKEN",
        "BIND_HOST",
    ):
        monkeypatch.setenv(name, "test-value")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://spire:test-value@db.invalid:5432/spire"
    )
    camera_url = "rtsp://real-user:real-pass@a-real-camera.invalid:554/stream1"
    monkeypatch.setenv("CAMERA_RTSP_URL", camera_url)
    monkeypatch.setenv("SPEAKER_BACKEND", "tapo_talk")

    config = load_config("config/config.example.yaml")

    assert config.camera.rtsp_url == camera_url
    assert config.speaker.backend == "tapo_talk"


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


def test_speaker_config_backend_defaults_to_go2rtc():
    from spire_voice.config import SpeakerConfig

    assert SpeakerConfig.from_config({}).backend == "go2rtc"
    assert SpeakerConfig().backend == "go2rtc"


def test_speaker_config_accepts_tapo_talk_backend():
    from spire_voice.config import SpeakerConfig

    assert SpeakerConfig.from_config({"backend": "tapo_talk"}).backend == "tapo_talk"


def test_speaker_config_rejects_an_unknown_backend():
    from spire_voice.config import ConfigError, SpeakerConfig

    with pytest.raises(ConfigError) as exc:
        SpeakerConfig.from_config({"backend": "sonos"})
    assert "speaker.backend" in str(exc.value)


def test_session_config_rejects_a_non_positive_retention():
    from spire_voice.config import ConfigError, SessionConfig

    with pytest.raises(ConfigError):
        SessionConfig.from_config({"retain_days": 0})
    with pytest.raises(ConfigError):
        SessionConfig.from_config({"retain_days": -7})


def test_workflow_config_defaults():
    from spire_voice.config import WorkflowConfig

    workflow = WorkflowConfig.from_config(None)
    assert workflow.poll_interval_s == 5.0
    assert workflow.max_steps_per_poll == 20
    assert workflow.late_threshold_s == 60.0
    assert workflow.max_attempts == 3
    assert workflow.retry_backoff_s == 30.0


def test_workflow_config_rejects_a_non_positive_poll_interval():
    from spire_voice.config import ConfigError, WorkflowConfig

    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"poll_interval_s": 0})
    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"poll_interval_s": -5})


def test_workflow_config_rejects_a_non_positive_max_steps_per_poll():
    from spire_voice.config import ConfigError, WorkflowConfig

    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"max_steps_per_poll": 0})
    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"max_steps_per_poll": -1})
    # Bounds a *count* of steps per tick -- a fractional value is not one.
    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"max_steps_per_poll": 1.5})


def test_workflow_config_rejects_a_negative_late_threshold():
    from spire_voice.config import ConfigError, WorkflowConfig

    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"late_threshold_s": -1})
    # Zero is allowed: "the instant it was due" is a real, if aggressive,
    # threshold an operator may choose.
    assert WorkflowConfig.from_config({"late_threshold_s": 0}).late_threshold_s == 0


def test_workflow_config_rejects_a_max_attempts_below_one():
    from spire_voice.config import ConfigError, WorkflowConfig

    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"max_attempts": 0})
    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"max_attempts": -1})


def test_workflow_config_rejects_a_non_positive_retry_backoff():
    from spire_voice.config import ConfigError, WorkflowConfig

    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"retry_backoff_s": 0})
    with pytest.raises(ConfigError):
        WorkflowConfig.from_config({"retry_backoff_s": -30})


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


# --- 260922-fmi (D-15/WR-03): route_enabled is a boolean or it is refused,
# mirroring SecurityConfig's cookie_secure rejections above ---


def test_calibration_config_refuses_a_route_enabled_that_is_not_a_boolean():
    """`expand_env` substitutes raw text before the YAML parse, so
    `CALIBRATION_ROUTE_ENABLED`'s value decides this field's type, exactly
    as `COOKIE_SECURE` decides `cookie_secure`'s (WR-03 above). An EMPTY
    value parses as `None`, not `false`, and this route makes a real home
    play a sound and record the room (T-FMI-01) -- a configuration nobody
    means is refused by name here instead."""
    from spire_voice.config import CalibrationConfig, ConfigError

    for value in (None, 1, 0, "true", "false", "yes", "on", ""):
        with pytest.raises(ConfigError) as exc:
            CalibrationConfig.from_config({"route_enabled": value})
        assert "route_enabled" in str(exc.value)
        assert "CALIBRATION_ROUTE_ENABLED" in str(exc.value)


def test_calibration_config_accepts_both_real_booleans_for_route_enabled():
    from spire_voice.config import CalibrationConfig

    assert CalibrationConfig.from_config({"route_enabled": True}).route_enabled is True
    assert CalibrationConfig.from_config({"route_enabled": False}).route_enabled is False


def test_an_empty_calibration_route_enabled_variable_is_refused_by_the_real_loader(
    tmp_path, monkeypatch
):
    """The end-to-end form of the case above, through `load_config` and the
    real `${VAR}` expansion -- the path the chart and Compose both take --
    rather than through `CalibrationConfig.from_config` alone."""
    from spire_voice.config import ConfigError, load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "database:\n"
        "  url: postgresql+asyncpg://spire:test-value@db.invalid:5432/spire\n"
        "brain:\n"
        "  base_url: https://brain.invalid/v1\n"
        "  api_key: test-value\n"
        "  models:\n"
        "    - model: fake-model\n"
        "calibration:\n"
        "  route_enabled: ${CALIBRATION_ROUTE_ENABLED}\n"
    )
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "")

    with pytest.raises(ConfigError) as exc:
        load_config(config_path)
    assert "route_enabled" in str(exc.value)


def test_shipped_config_resolves_calibration_route_enabled_from_the_real_env_var(monkeypatch):
    """The unquoted `${CALIBRATION_ROUTE_ENABLED}` in the shipped example
    file must resolve to a real YAML boolean, exactly as `${COOKIE_SECURE}`
    already does -- proven end-to-end through `load_config` on the shipped
    file rather than through a hand-built fixture."""
    from spire_voice.config import load_config

    for name in (
        "XAI_API_KEY",
        "TAPO_USER",
        "TAPO_PASSWORD",
        "SPEAKER_ENSURE_URL",
        "HA_URL",
        "HA_TOKEN",
        "BIND_HOST",
    ):
        monkeypatch.setenv(name, "test-value")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://spire:test-value@db.invalid:5432/spire"
    )

    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "true")
    config = load_config("config/config.example.yaml")
    assert config.calibration.route_enabled is True

    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    config = load_config("config/config.example.yaml")
    assert config.calibration.route_enabled is False


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


# --- Plan 03-01: DatabaseConfig, SecurityConfig, read_secret_key (D-01, D-02, D-07) ---


def test_database_config_derives_migration_url_from_the_runtime_url():
    from spire_voice.config import DatabaseConfig

    database = DatabaseConfig.from_config(
        {"url": "postgresql+asyncpg://spire:test-value@db.invalid:5432/spire"}
    )
    assert database.url == "postgresql+asyncpg://spire:test-value@db.invalid:5432/spire"
    assert database.migration_url == "postgresql+psycopg://spire:test-value@db.invalid:5432/spire"
    assert database.run_migrations_at_startup is True


def test_database_config_rejects_a_synchronous_scheme():
    from spire_voice.config import ConfigError, DatabaseConfig

    with pytest.raises(ConfigError) as exc:
        DatabaseConfig.from_config(
            {"url": "postgresql+psycopg://spire:test-value@db.invalid:5432/spire"}
        )
    assert "database.url" in str(exc.value)


def test_database_config_rejects_a_missing_url():
    from spire_voice.config import ConfigError, DatabaseConfig

    with pytest.raises(ConfigError) as exc:
        DatabaseConfig.from_config({})
    assert "database.url" in str(exc.value)


def test_database_config_explicit_migration_url_overrides_the_derived_one():
    from spire_voice.config import DatabaseConfig

    database = DatabaseConfig.from_config(
        {
            "url": "postgresql+asyncpg://spire:test-value@db.invalid:5432/spire",
            "migration_url": "postgresql+psycopg://migrator:test-value@migrations.invalid:5432/spire",
        }
    )
    assert database.migration_url == (
        "postgresql+psycopg://migrator:test-value@migrations.invalid:5432/spire"
    )


def test_read_secret_key_raises_a_named_config_error_when_absent(monkeypatch):
    from spire_voice.config import ConfigError, SecurityConfig, read_secret_key

    monkeypatch.delenv("SPIRE_SECRET_KEY", raising=False)
    security = SecurityConfig.from_config(None)
    with pytest.raises(ConfigError) as exc:
        read_secret_key(security)
    assert "SPIRE_SECRET_KEY" in str(exc.value)


def test_read_secret_key_returns_the_environment_value_when_present(monkeypatch):
    from spire_voice.config import SecurityConfig, read_secret_key

    monkeypatch.setenv("SPIRE_SECRET_KEY", "test-value")
    security = SecurityConfig.from_config(None)
    assert read_secret_key(security) == "test-value"


def test_security_config_rejects_an_access_lifetime_that_does_not_outlive_a_shorter_refresh():
    from spire_voice.config import ConfigError, SecurityConfig

    with pytest.raises(ConfigError) as exc:
        SecurityConfig.from_config({"access_token_ttl_s": 1000, "refresh_token_ttl_s": 900})
    assert "access_token_ttl_s" in str(exc.value)
    assert "refresh_token_ttl_s" in str(exc.value)

    # Equal is also rejected -- an access token that never actually outlives
    # its own refresh token is the same nobody-means-this configuration.
    with pytest.raises(ConfigError):
        SecurityConfig.from_config({"access_token_ttl_s": 900, "refresh_token_ttl_s": 900})


def test_security_config_defaults():
    from spire_voice.config import SecurityConfig

    security = SecurityConfig.from_config(None)
    assert security.secret_key_env == "SPIRE_SECRET_KEY"
    assert security.access_token_ttl_s == 900
    assert security.refresh_token_ttl_s == 1209600
    assert security.cookie_name == "spire_session"
    assert security.cookie_secure is False


# --- WR-03 (code review): cookie_secure is a boolean or it is refused ---


def test_security_config_refuses_a_cookie_secure_that_is_not_a_boolean():
    """WR-03. `expand_env` substitutes raw text before the YAML parse, so
    `COOKIE_SECURE`'s value decides this field's type. An EMPTY value --
    reachable from `COOKIE_SECURE=` in a `.env`, and from `--set-string
    config.cookieSecure=""` in the chart -- parsed as `None`: falsy,
    accepted with no error into a field typed `bool`, and then handed to
    `auth/tokens.py` as `secure=None` on the session cookie. An insecure
    cookie behind a TLS Ingress, with a startup log line as the only
    signal that anything was wrong.

    Every one of these is what the real loader produced for a value an
    operator could plausibly type; `true`/`false` are the only two this
    field has ever meant.
    """
    from spire_voice.config import ConfigError, SecurityConfig

    for value in (None, 1, 0, "true", "false", "yes", "on", ""):
        with pytest.raises(ConfigError) as exc:
            SecurityConfig.from_config({"cookie_secure": value})
        assert "cookie_secure" in str(exc.value)
        assert "COOKIE_SECURE" in str(exc.value)


def test_security_config_accepts_both_real_booleans():
    from spire_voice.config import SecurityConfig

    assert SecurityConfig.from_config({"cookie_secure": True}).cookie_secure is True
    assert SecurityConfig.from_config({"cookie_secure": False}).cookie_secure is False


def test_an_empty_cookie_secure_variable_is_refused_by_the_real_loader(tmp_path, monkeypatch):
    """The end-to-end form of the case above, through `load_config` and
    the real `${VAR}` expansion -- the path the chart and Compose both
    take -- rather than through `SecurityConfig.from_config` alone."""
    from spire_voice.config import ConfigError, load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "database:\n"
        "  url: postgresql+asyncpg://spire:test-value@db.invalid:5432/spire\n"
        "brain:\n"
        "  base_url: https://brain.invalid/v1\n"
        "  api_key: test-value\n"
        "  models:\n"
        "    - model: fake-model\n"
        "security:\n"
        "  cookie_secure: ${COOKIE_SECURE}\n"
    )
    monkeypatch.setenv("COOKIE_SECURE", "")

    with pytest.raises(ConfigError) as exc:
        load_config(config_path)
    assert "cookie_secure" in str(exc.value)
