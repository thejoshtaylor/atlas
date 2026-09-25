"""RED for EdgeConfig/load_config (10-08-PLAN.md Task 2)."""

from __future__ import annotations

import os
import stat

import pytest

from atlas_edge.config import EdgeConfig, EdgeConfigError, load_config


def _write_config(path, text: str, mode: int = 0o600) -> None:
    path.write_text(text)
    os.chmod(path, mode)


def test_loads_required_and_optional_keys(tmp_path):
    path = tmp_path / "config.toml"
    _write_config(
        path,
        """
        server_url = "wss://atlas.example.test/ws/edge"
        token = "tok1234"
        capture_device = "MyMic"
        vad_threshold = 0.7
        vad_min_silence_ms = 300
        """,
    )
    config = load_config(path)
    assert config.server_url == "wss://atlas.example.test/ws/edge"
    assert config.token == "tok1234"
    assert config.capture_device == "MyMic"
    assert config.playback_device == "reSpeaker"  # default, untouched
    assert config.vad_threshold == 0.7
    assert config.vad_min_silence_ms == 300
    assert config.allow_plaintext is False


def test_refuses_group_readable_file(tmp_path):
    path = tmp_path / "config.toml"
    _write_config(
        path,
        'server_url = "wss://atlas.example.test/ws/edge"\ntoken = "tok1234"\n',
        mode=0o640,
    )
    with pytest.raises(EdgeConfigError, match="readable by group or others"):
        load_config(path)


def test_refuses_other_readable_file(tmp_path):
    path = tmp_path / "config.toml"
    _write_config(
        path,
        'server_url = "wss://atlas.example.test/ws/edge"\ntoken = "tok1234"\n',
        mode=0o604,
    )
    with pytest.raises(EdgeConfigError, match="readable by group or others"):
        load_config(path)


def test_owner_only_permissions_are_accepted(tmp_path):
    path = tmp_path / "config.toml"
    _write_config(
        path,
        'server_url = "wss://atlas.example.test/ws/edge"\ntoken = "tok1234"\n',
        mode=0o600,
    )
    load_config(path)  # does not raise


def test_missing_required_key_raises(tmp_path):
    path = tmp_path / "config.toml"
    _write_config(path, 'server_url = "wss://atlas.example.test/ws/edge"\n')
    with pytest.raises(EdgeConfigError, match="token"):
        load_config(path)


def test_url_validate_server_url_refuses_raises_edge_config_error(tmp_path):
    path = tmp_path / "config.toml"
    _write_config(
        path,
        'server_url = "ws://atlas.example.test/ws/edge"\ntoken = "tok1234"\n',
    )
    with pytest.raises(EdgeConfigError, match="wss"):
        load_config(path)


def test_repr_never_contains_the_token(tmp_path):
    path = tmp_path / "config.toml"
    _write_config(
        path,
        'server_url = "wss://atlas.example.test/ws/edge"\ntoken = "tok1234"\n',
    )
    config = load_config(path)
    assert "tok1234" not in repr(config)


def test_frozen_dataclass_stat_mode_bits():
    # Sanity: 0o077 masks group+other read/write/execute.
    assert stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH == 0o077
