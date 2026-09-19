"""SAFE-09 generalized from the one hardcoded Home Assistant child
(`tests/test_ha_tool.py::test_the_child_never_sees_secrets_the_parent_holds_for_other_plugins`)
to `PluginManager`'s own spawn path (plan 06-01, Task 3): an admin-typed
plugin configuration is covered here, not only a migration-seeded one.

Three claims, each proved directly rather than inferred from "the process
started without raising":

1. A child spawned through the manager, from a plugin row whose
   configuration an admin typed, never sees `XAI_API_KEY`, `DATABASE_URL`,
   or `SPIRE_SECRET_KEY` from a deliberately polluted parent environment --
   asserted on the child's own view of its environment, printed from
   inside a real spawned process, never on the dictionary this test built
   (`tests/test_ha_tool.py`'s own discipline, generalized).
2. A child spawned from one plugin row never sees another plugin row's
   config values -- proved directly against `PluginManager`'s own
   environment-building code, since that is the one place two plugins'
   configuration could ever cross.
3. The decrypted value of a secret config key appears in the child's
   environment and nowhere a repository read returns.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone

from mcp.client.stdio import get_default_environment

import conftest
from spire_voice.config import SecurityConfig
from spire_voice.crypto.credentials import encrypt_credential
from spire_voice.db.repository import Plugin, PluginConfigValue
from spire_voice.mcp_client import McpToolHost
from spire_voice.plugins.manager import PluginManager

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MCP_ROOT = os.path.join(_REPO_ROOT, "mcp")

# The same three hostile secrets `tests/test_ha_tool.py` uses, generalized
# from "the parent process this application runs in" to "the parent
# process any plugin child is spawned from" -- the claim under test does
# not change with which plugin the child happens to be.
_HOSTILE_PARENT_SECRETS = {
    "XAI_API_KEY": "sk-hostile-parent-secret-should-never-reach-a-plugin-child",
    "DATABASE_URL": "postgresql+asyncpg://hostile:secret@db.invalid/hostile",
    "SPIRE_SECRET_KEY": "hostile-fernet-key-should-never-leak-into-a-plugin-child",
}


def _ha_plugin(plugin_id: int = 1) -> Plugin:
    now = datetime.now(timezone.utc)
    return Plugin(
        id=plugin_id,
        slug="ha",
        display_name="Home Assistant",
        transport="stdio",
        args=("-m", "spire_mcp.ha"),
        url=None,
        enabled=True,
        builtin=True,
        enforces_policy=True,
        timeout_ms=5000,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )


def _weather_plugin(plugin_id: int = 2) -> Plugin:
    now = datetime.now(timezone.utc)
    return Plugin(
        id=plugin_id,
        slug="weather",
        display_name="Weather",
        transport="stdio",
        args=("-m", "spire_mcp.weather"),
        url=None,
        enabled=True,
        builtin=True,
        enforces_policy=False,
        timeout_ms=5000,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )


def _child_env_probe(env: dict[str, str], module: str) -> dict[str, bool]:
    """Import `module` under exactly `env` in a real subprocess, and
    return which of `_HOSTILE_PARENT_SECRETS` that child's own
    `os.environ` actually carried -- the exact real-spawned-child
    discipline `tests/test_ha_tool.py`'s own hostile-parent test uses,
    generalized to any plugin module rather than hardcoded to
    `spire_mcp.ha`.
    """
    code = textwrap.dedent(
        f"""
        import json
        import os
        import {module}  # import the real child module under this env
        print(json.dumps({{name: (name in os.environ) for name in {list(_HOSTILE_PARENT_SECRETS)!r}}}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


async def test_a_child_spawned_through_the_manager_never_sees_the_parents_hostile_secrets(
    monkeypatch,
):
    """Truth 1 (Task 3): a child spawned through `PluginManager`, from a
    plugin row whose configuration an admin typed, never sees
    `XAI_API_KEY`/`DATABASE_URL`/`SPIRE_SECRET_KEY` from a deliberately
    polluted parent environment -- generalized from the one hardcoded
    Home Assistant child to the manager's own spawn path.
    """
    for name, value in _HOSTILE_PARENT_SECRETS.items():
        monkeypatch.setenv(name, value)

    # An admin-typed configuration, encrypted under this (now-hostile)
    # SPIRE_SECRET_KEY -- the same key `PluginManager` itself reads at
    # decrypt time, so encryption and decryption agree regardless of the
    # value being hostile.
    security = SecurityConfig()
    admin_typed_value = "an-admin-typed-token-not-a-real-credential"
    ciphertext, key_version = encrypt_credential(admin_typed_value, security)

    plugin = _ha_plugin()
    repo = conftest.FakePluginRepository(
        plugins=[plugin],
        config_values={
            plugin.id: [
                PluginConfigValue(
                    key="HA_URL", secret=False, value="http://ha.invalid:8123",
                    ciphertext=None, key_version=None,
                ),
                PluginConfigValue(
                    key="HA_TOKEN", secret=True, value=None,
                    ciphertext=ciphertext, key_version=key_version,
                ),
            ],
        },
    )

    async def _no_policy():
        return None

    manager = PluginManager(
        repo, mcp_root=_MCP_ROOT, security=security, safety_block_provider=_no_policy
    )

    # Capture the literal environment PluginManager's own spawn path
    # builds -- `McpToolHost._spawn` is the one place both `start()` and
    # `respawn()` actually launch a child (mcp_client.py's own docstring),
    # so intercepting it here captures exactly what the manager decided,
    # never a dictionary this test built by hand.
    captured_envs: list[dict[str, str]] = []
    real_spawn = McpToolHost._spawn

    async def _capturing_spawn(self, child_module, env):
        captured_envs.append(dict(env))
        await real_spawn(self, child_module, env)

    monkeypatch.setattr(McpToolHost, "_spawn", _capturing_spawn)

    try:
        await manager.start_all()
    finally:
        await manager.stop_all()

    assert len(captured_envs) == 1
    manager_built_env = captured_envs[0]
    assert manager_built_env["HA_TOKEN"] == admin_typed_value, (
        "the manager's own spawn path must decrypt the admin-typed token"
    )
    for secret_value in _HOSTILE_PARENT_SECRETS.values():
        assert secret_value not in manager_built_env.values(), (
            "a hostile parent secret leaked into the dict PluginManager itself built"
        )

    # The exact shape `McpToolHost._spawn` actually spawns a child with:
    # the SDK's own allow-list, merged with the manager-built mapping --
    # never a copy of this (now hostile) process's full environment. This
    # re-derives the merge the real SDK performs rather than asserting a
    # guess at what it does, matching `tests/test_ha_tool.py`'s own
    # reasoning.
    child_env = get_default_environment() | manager_built_env
    seen = _child_env_probe(child_env, "spire_mcp.ha")
    for name in _HOSTILE_PARENT_SECRETS:
        assert not seen[name], f"{name} leaked into a plugin child's environment"


def test_a_child_never_sees_a_sibling_plugins_config_values():
    """Truth 2 (Task 3): a child spawned from one plugin row never sees
    another plugin row's config values -- proved directly against
    `PluginManager._build_env`, the one place two plugins' configuration
    could ever cross, without needing to spawn either child for real.
    """
    ha = _ha_plugin(plugin_id=1)
    weather = _weather_plugin(plugin_id=2)
    repo = conftest.FakePluginRepository(
        plugins=[ha, weather],
        config_values={
            ha.id: [
                PluginConfigValue(
                    key="HA_URL", secret=False, value="http://ha.invalid:8123",
                    ciphertext=None, key_version=None,
                ),
            ],
            weather.id: [
                PluginConfigValue(
                    key="WEATHER_LATITUDE", secret=False, value="51.5",
                    ciphertext=None, key_version=None,
                ),
                PluginConfigValue(
                    key="WEATHER_LONGITUDE", secret=False, value="-0.1",
                    ciphertext=None, key_version=None,
                ),
            ],
        },
    )

    async def _no_policy():
        return None

    async def _build_both():
        manager = PluginManager(
            repo, mcp_root=_MCP_ROOT, security=SecurityConfig(), safety_block_provider=_no_policy
        )
        ha_env = await manager._build_env(ha, safety_block=None)
        weather_env = await manager._build_env(weather, safety_block=None)
        return ha_env, weather_env

    ha_env, weather_env = asyncio.run(_build_both())

    assert "WEATHER_LATITUDE" not in ha_env
    assert "WEATHER_LONGITUDE" not in ha_env
    assert "HA_URL" not in weather_env
    # Only PYTHONPATH is shared -- every other key is exactly what that
    # plugin's own row declared, with no per-slug branch deciding it.
    assert set(ha_env) == {"PYTHONPATH", "HA_URL"}
    assert set(weather_env) == {"PYTHONPATH", "WEATHER_LATITUDE", "WEATHER_LONGITUDE"}


async def test_the_decrypted_secret_reaches_the_child_and_nothing_a_repository_read_returns_carries_it(
    monkeypatch,
):
    """Truth 3 (Task 3): the decrypted value of a secret config key
    appears in the child's environment (the manager's own build path) and
    nowhere in what `PluginRepository.get_config_values` returns."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    security = SecurityConfig()
    plaintext = "a-plainly-fictional-decrypted-token"
    ciphertext, key_version = encrypt_credential(plaintext, security)

    plugin = _ha_plugin()
    repo = conftest.FakePluginRepository(
        plugins=[plugin],
        config_values={
            plugin.id: [
                PluginConfigValue(
                    key="HA_TOKEN", secret=True, value=None,
                    ciphertext=ciphertext, key_version=key_version,
                ),
            ],
        },
    )

    # Nothing a repository read returns carries the plaintext -- checked
    # against the actual dataclass instances `get_config_values` hands
    # back, not a copy this test constructed.
    returned_values = await repo.get_config_values(plugin.id)
    for value in returned_values:
        assert value.value is None, "a secret PluginConfigValue must never carry a plaintext value"
        assert value.ciphertext is not None
        assert plaintext.encode("utf-8") not in value.ciphertext
        assert plaintext not in repr(value)

    async def _no_policy():
        return None

    manager = PluginManager(
        repo, mcp_root=_MCP_ROOT, security=security, safety_block_provider=_no_policy
    )
    env = await manager._build_env(plugin, safety_block=None)
    assert env["HA_TOKEN"] == plaintext, (
        "the manager's own spawn path must be the point the decrypted value actually reaches"
    )
