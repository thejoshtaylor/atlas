"""The curated plugin list (D-13, PLUG-03): a static file in the
repository, read with no network access of any kind, validated at read
time rather than at the moment an admin tries to install from a broken
entry.
"""

from __future__ import annotations

import json

import pytest

from atlas.plugins.catalog import (
    DEFAULT_CATALOG_PATH,
    CatalogConfigKey,
    CatalogError,
    find_entry,
    load_catalog,
)


def test_the_shipped_catalog_parses_and_seeds_the_two_builtin_plugins():
    """Task 1's own instruction: seed the file with the two capabilities
    this project already ships as children, plus nothing else."""
    entries = load_catalog(DEFAULT_CATALOG_PATH)
    assert [entry.name for entry in entries] == ["Home Assistant", "Weather"]

    ha = find_entry(entries, "Home Assistant")
    assert ha is not None
    assert ha.transport == "stdio"
    assert ha.args == ("-m", "atlas_mcp.ha")
    assert ha.url is None
    assert ha.config_keys == (
        CatalogConfigKey(key="HA_URL", label="Home Assistant URL", secret=False),
        CatalogConfigKey(key="HA_TOKEN", label="Home Assistant long-lived access token", secret=True),
    )

    weather = find_entry(entries, "Weather")
    assert weather is not None
    assert weather.transport == "stdio"
    assert weather.args == ("-m", "atlas_mcp.weather")
    assert weather.url is None
    assert [ck.key for ck in weather.config_keys] == [
        "WEATHER_LATITUDE",
        "WEATHER_LONGITUDE",
        "WEATHER_UNITS",
    ]
    assert all(not ck.secret for ck in weather.config_keys)
    assert weather.config_keys[2] == CatalogConfigKey(
        key="WEATHER_UNITS", label="Units (celsius or fahrenheit)", secret=False
    )


def test_no_entry_declares_a_secret_value_only_which_keys_are_secret():
    """Task 1's own behavior: a catalog declares which keys are secret,
    never what any of them contain."""
    entries = load_catalog(DEFAULT_CATALOG_PATH)
    raw = json.loads(open(DEFAULT_CATALOG_PATH, encoding="utf-8").read())
    assert "value" not in json.dumps(raw)  # no config key in the shipped file carries a value at all
    for entry in entries:
        for config_key in entry.config_keys:
            assert isinstance(config_key.secret, bool)


def test_find_entry_returns_none_for_an_unknown_name():
    entries = load_catalog(DEFAULT_CATALOG_PATH)
    assert find_entry(entries, "does-not-exist") is None


def test_an_unreadable_file_raises_naming_the_path():
    with pytest.raises(CatalogError, match="config/does-not-exist.json"):
        load_catalog("config/does-not-exist.json")


def test_a_file_with_zero_entries_parses_to_an_empty_tuple(tmp_path):
    """A real state, not an error (Task 1's own behavior)."""
    catalog_path = tmp_path / "empty-catalog.json"
    catalog_path.write_text(json.dumps({"entries": []}), encoding="utf-8")
    assert load_catalog(catalog_path) == ()


def test_malformed_json_raises_naming_the_path(tmp_path):
    catalog_path = tmp_path / "broken.json"
    catalog_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CatalogError, match=str(catalog_path)):
        load_catalog(catalog_path)


def test_an_entry_declaring_both_args_and_url_is_refused(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "Bad Entry",
                        "description": "declares both",
                        "transport": "stdio",
                        "args": ["-m", "some_module"],
                        "url": "https://example.invalid",
                        "config_keys": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="both"):
        load_catalog(catalog_path)


def test_a_command_entry_with_no_args_is_refused(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "Bad Entry",
                        "description": "declares neither",
                        "transport": "stdio",
                        "args": [],
                        "url": None,
                        "config_keys": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="no 'args'"):
        load_catalog(catalog_path)


def test_a_url_entry_with_no_url_is_refused(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "Bad Entry",
                        "description": "declares neither",
                        "transport": "remote",
                        "args": [],
                        "url": None,
                        "config_keys": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="no 'url'"):
        load_catalog(catalog_path)


def test_a_malformed_config_key_raises_naming_the_entry(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "Bad Entry",
                        "description": "a config key missing its secret flag",
                        "transport": "stdio",
                        "args": ["-m", "some_module"],
                        "url": None,
                        "config_keys": [{"key": "SOME_KEY", "label": "Some Key"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="Bad Entry"):
        load_catalog(catalog_path)


def test_the_default_catalog_path_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    """IN-01 (code review): `DEFAULT_CATALOG_PATH` was the relative string
    `"config/plugin-catalog.json"`, resolved against the process's current
    working directory -- unlike `MCP_ROOT` and `FRONTEND_DIR`, both
    derived from `__file__`. Started from any directory but the repository
    root, the catalog route raised `CatalogError`.
    """
    import os

    from atlas.plugins.catalog import DEFAULT_CATALOG_PATH, load_catalog

    assert os.path.isabs(DEFAULT_CATALOG_PATH)

    monkeypatch.chdir(tmp_path)
    assert load_catalog(DEFAULT_CATALOG_PATH), "the shipped catalog must load from anywhere"
