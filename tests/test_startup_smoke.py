"""Boots the real application -- the carried blocker from Phase 1.

Three of Phase 1's Critical defects (`inputSchema` vs `input_schema`, the
turn-ending hang, the MCP child's interpreter bug) lived on the startup
path: `lifespan` opens the providers, spawns the MCP child, fetches the
entity catalog, and precaches every filler phrase, and no test ever ran any
of it. Every other test in this repository imports a handler function or a
class directly. This is the one that starts the app.

`TestClient(app)` used as a context manager is what runs Starlette's
`lifespan` startup and shutdown (FastAPI's own testing docs) -- a plain
`TestClient(app)` with no `with` block never triggers it at all, which
would make this test pass for the wrong reason.

Four external systems sit on this path and are made inert here, at the
`spire_voice.app` module level, the same structurally-identical-fake
convention `tests/conftest.py` already uses -- never `unittest.mock`:

- the brain tiers, each of which calls `resolve_model()` over the network
  (`brain_race.build_tiers` is replaced with a fake tier whose `.brain`
  resolves instantly);
- the MCP child, spawned as a real subprocess against Home Assistant
  (`McpToolHost` is replaced outright: `FakeHomeAssistant`'s `httpx.
  MockTransport` is in-process and cannot answer a request from a separate
  child process, so `test_policy_reaches_the_child.py`'s real-subprocess
  fixture does not compose here -- this is the one place this plan falls
  back to a fake rather than the real thing, and it is a fake tool host,
  not a fake child);
- the entity catalog, fetched through that same child;
- the filler/macro-reply precache, which synthesizes every phrase through
  the TTS provider (`precache_all` is replaced with a fake that returns
  immediately, touching neither the network nor the disk cache).

Plan 03-05 (Phase 3) adds a database, migrations, an application-level
setup gate, and an authentication dependency to this same startup path --
this file now also covers: `run_migrations`/`build_engine` substituted so
no reachable Postgres is needed (`_fake_run_migrations`/`_fake_build_engine`
below, and the dedicated migration-failure test); `_build_repositories`
substituted with `FakePolicyRepository`/`FakeAccountRepository`, the latter
pre-seeded with one fake admin so this file's own (unrelated) wiring
assertions are not all turned into 503s by the new setup gate;
`SPIRE_SECRET_KEY` set to a structurally-valid test value by this file's own
autouse fixture, since `validate_secret_key_strength` now runs before
anything else in `lifespan`; and one dedicated test
(`test_the_real_boot_answers_create_admin_while_every_other_route_reports_setup_incomplete`)
proving the setup gate itself is real against a genuinely empty account
repository, through a real boot rather than a unit call.

What this file cannot and does not claim to cover: whether the real RTSP
URL, go2rtc, or a wake model work against actual hardware (unchanged from
Phase 1/2); and, new as of this plan, whether a real browser can actually
sign in through the built webapp, and whether the built application
renders at all -- both remain human checks (this plan's own `<verify>`
block names the clean-install sign-in as one), not something any fake here
can stand in for.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient
from mcp.types import Tool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

import conftest
import spire_voice.app as app_module
from spire_voice.db.repository import Setting, User
from spire_voice.transports.base import SourceFormat
from spire_voice.turn.brain_race import TierBrain

# A plainly fictional, but structurally valid (32 raw bytes, real byte
# variety), urlsafe-base64 key -- shaped exactly like
# `Fernet.generate_key()`'s own output so it passes
# `auth/tokens.py::validate_secret_key_strength`'s structural checks
# (plan 03-05). Every test in this file boots the real `lifespan`, which
# now calls that function before anything else -- never a real secret,
# never reused outside this test module.
import base64 as _base64

_TEST_SECRET_KEY = _base64.urlsafe_b64encode(bytes(range(32))).decode("ascii")


@pytest.fixture(autouse=True)
def _set_test_secret_key(monkeypatch):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)


# Every attribute `lifespan` assigns to `app.state`, read off `app.py` by
# name rather than spot-checked -- "a resource never assigned to
# `app.state`" is exactly one of the three defect shapes STATE.md names for
# Phase 1, and a spot check could miss a fourth one the same way.
_EXPECTED_STATE_ATTRS = [
    "config",
    "stt",
    "tier_brains",
    "brain",
    "tts",
    "tool_host",
    # Plan 04-03 (D-13): the lookup every `run_turn` call site actually
    # passes -- always present (one host, or two), even when no
    # `mcp.servers.weather` block is configured at all. `weather_tool_host`
    # is deliberately NOT in this list: it is a legitimate `None` both
    # when weather is unconfigured and when its child failed to start
    # (T-04-16), so "present" cannot mean "non-None" for that one
    # attribute the way it does for every other resource here.
    "tool_host_lookup",
    "tools_schema",
    "catalog_prompt",
    "macros",
    "filler_cache",
    "background_turns",
    # Plan 02-03: the room-listens spine. None of these need a reachable
    # camera, go2rtc, or wake model to land on `app.state` -- a camera that
    # never connects, a speaker FIFO nobody reads yet, and a fake wake
    # detector (below) are all expected, not-yet-reachable states, never
    # startup errors (T-02-13).
    "speaker_writer",
    "wake_detector",
    "camera_source",
    "source_runners",
    "source_runner_tasks",
    "ffmpeg_supervisor",
    # Plan 02-08: the retention sweep. Starts and stops on the same
    # lifespan-owned pattern as `ffmpeg_supervisor` above; needs no
    # reachable session directory on disk to land on `app.state`.
    "retention_scheduler",
    # Plan 03-02: the database engine and the repositories built on top of
    # it. `run_migrations`/`build_engine` are substituted with fakes below
    # (matching `_build_wake_detector`'s own monkeypatch precedent), so
    # none of these three need a reachable Postgres to land on `app.state`.
    "db_engine",
    "policy_repo",
    "safety_block",
    # Plan 03-05: the account repository `require_setup_complete` and
    # every account/invite route read off `app.state`.
    "account_repo",
    # Plan 03-07: the credential repository the resolution order reads
    # before any provider is constructed, and `routes/credentials.py`'s
    # own list/write routes read off `app.state` the same way.
    "credential_repo",
]


def _write_fake_config(tmp_path: Path, *, extra: dict | None = None) -> Path:
    """A full config, shaped like `config.example.yaml`, with plainly
    fictional values -- no `${NAME}` placeholder and no `.env` read, since
    this test never expands the environment, matching `tests/test_config.py`
    `_minimal_raw_config()`'s convention: never a real host or entity id.

    `extra` merges additional top-level sections (`camera`, `wake`, `gate`,
    `barge_in`, ...) into the base dict below -- every key `Config.from_config`
    reads is optional (`raw.get(name)`), so a caller only names the section
    it actually needs a non-default value from.
    """
    raw = {
        "server": {"bind_host": "127.0.0.1", "port": 8080, "transport": "websocket"},
        "stt": {"url": "wss://stt.invalid/v1/stt", "api_key": "test-key"},
        "brain": {
            "base_url": "https://brain.invalid/v1",
            "api_key": "test-key",
            "models": [{"model": "fake-model"}],
        },
        "tts": {
            "url": "https://tts.invalid/v1/tts",
            "api_key": "test-key",
            "voice_id": "eve",
            "cache_dir": str(tmp_path / "tts-cache"),
        },
        "mcp": {
            "servers": {
                "ha": {
                    "args": ["-m", "spire_mcp.ha"],
                    "env": {"HA_URL": "http://ha.invalid", "HA_TOKEN": "test-token"},
                },
            },
        },
        # Plan 03-01: database.url has no default (D-01/D-02) -- a real
        # boot never reaches this far without one, and this smoke test's
        # `lifespan` run is no exception. Never actually connected here:
        # `run_migrations`/`build_engine` are both monkeypatched with fakes
        # below, so a syntactically valid, unreachable connection string is
        # all `Config.from_config` needs to build without raising.
        "database": {"url": "postgresql+asyncpg://spire:test-value@db.invalid:5432/spire"},
        # Plan 03-02: the safety: key is retired (D-11) -- Config.from_config
        # now raises if it is present, so this fixture stops emitting one.
        # SecurityConfig's own fields all default, so an empty block (or no
        # block at all) is enough to load cleanly.
        "security": {},
    }
    raw.update(extra or {})
    path = tmp_path / "smoke-config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


class _FakeResolvingBrain:
    """Stands in for `XaiBrain`: `lifespan` awaits `resolve_model()` on every
    tier before anything else happens, and a real one dials out."""

    async def resolve_model(self) -> str:
        return "fake-model"


def _fake_build_tiers(brain_config: object) -> tuple[TierBrain, ...]:
    """Replaces `brain_race.build_tiers`: one tier, no `instructor` client,
    no real `XaiBrain` -- `lifespan` never calls anything else on a tier
    besides `resolve_model()` before assigning it to `app.state`."""
    return (TierBrain(index=0, model="fake-model", brain=_FakeResolvingBrain(), envelope_client=None, calls_tools=True),)


class _FakeToolHost:
    """Replaces `McpToolHost` outright.

    `FakeHomeAssistant` (`tests/conftest.py`) answers over an in-process
    `httpx.MockTransport`, which cannot intercept a request from a real
    spawned subprocess -- the shape the MCP child actually is. Reusing
    `test_policy_reaches_the_child.py`'s real-child fixture would need a
    real listening HTTP server standing in for Home Assistant, which is out
    of this plan's scope. This fake proves `lifespan`'s wiring (every
    resource lands on `app.state`) without spawning a process at all.
    """

    def __init__(self) -> None:
        self.tools = [
            Tool(name="ha_list_entities", description="List entities.", inputSchema={"type": "object", "properties": {}}),
        ]

    async def start(self, **kwargs: object) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict) -> object:
        return SimpleNamespace(structuredContent=[], content=[])

    async def aclose(self) -> None:
        return None


class _FakeWeatherToolHost:
    """A second, distinctly-tooled twin of `_FakeToolHost` (D-13, plan
    04-03): its `.tools` list carries a name no HA-shaped fake in this
    file advertises, so `McpToolHostLookup`'s own ambiguous-name check
    (T-04-12) has nothing to trip on when both the Home Assistant and
    weather children in a test are built from fakes defined in this same
    module.
    """

    def __init__(self) -> None:
        self.tools = [
            Tool(
                name="weather_current",
                description="Current outdoor weather.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    async def start(self, **kwargs: object) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict) -> object:
        return SimpleNamespace(structuredContent={}, content=[])

    async def aclose(self) -> None:
        return None


class _FakeRaisingWeatherToolHost:
    """A weather-shaped fake whose `start()` raises -- proving T-04-16:
    a weather child that fails to start must not stop the whole
    application, and the Home Assistant host must still be reachable
    afterwards.
    """

    def __init__(self) -> None:
        self.tools: list[Tool] = []

    async def start(self, **kwargs: object) -> None:
        raise RuntimeError("simulated weather child startup failure")

    async def call_tool(self, name: str, arguments: dict) -> object:
        raise AssertionError("call_tool must never be reached on a host that failed to start")

    async def aclose(self) -> None:
        return None


def _make_two_host_fake_tool_host_factory():
    """A callable, monkeypatched over `McpToolHost` itself, that hands out
    an HA-shaped fake on its first call and a weather-shaped fake on
    every call after -- matching `lifespan`'s own two `McpToolHost()`
    call order (Home Assistant, then weather, D-13). A bare class cannot
    do this: both calls construct with no arguments, so only call order
    can tell them apart.
    """
    calls = {"count": 0}

    def _factory(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeToolHost()
        return _FakeWeatherToolHost()

    return _factory


def _make_failing_weather_tool_host_factory():
    """Same call-order trick as `_make_two_host_fake_tool_host_factory`,
    but the second (weather) host is the one whose `start()` raises."""
    calls = {"count": 0}

    def _factory(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeToolHost()
        return _FakeRaisingWeatherToolHost()

    return _factory


def _weather_mcp_servers_extra() -> dict:
    """The `mcp.servers` block `_write_fake_config`'s own `extra` merges
    in whole (it replaces the top-level `mcp` key outright, not a deep
    merge) -- repeats the base fixture's `ha` block alongside a `weather`
    one, matching `config.example.yaml`'s own two-child shape.
    """
    return {
        "mcp": {
            "servers": {
                "ha": {
                    "args": ["-m", "spire_mcp.ha"],
                    "env": {"HA_URL": "http://ha.invalid", "HA_TOKEN": "test-token"},
                },
                "weather": {
                    "args": ["-m", "spire_mcp.weather"],
                    "env": {"WEATHER_LATITUDE": "0.0", "WEATHER_LONGITUDE": "0.0"},
                },
            },
        },
    }


async def _fake_precache_all(tts: object, cache_dir: Path, texts: list[str], voice_id: str, sink: object) -> dict:
    """Replaces `precache_all`: the real one synthesizes every phrase
    through the TTS provider over the network, once per phrase."""
    return {text: b"" for text in texts}


def _fake_run_migrations(migration_url: str) -> None:
    """Replaces `spire_voice.db.engine.run_migrations`: the real one runs
    Alembic against a reachable Postgres, which this smoke test never has.
    A no-op here proves `lifespan`'s wiring (awaited, in sequence, before
    anything else) without needing a real database -- the one test that
    proves a *failing* migration stops the boot substitutes a raising
    fake instead, over this same monkeypatch point."""
    return None


def _fake_build_engine(database_config: object) -> object:
    """Replaces `spire_voice.db.engine.build_engine`: the real one builds a
    real `AsyncEngine`. This smoke test only needs *something* non-`None`
    to land on `app.state.db_engine` and to have a no-op `dispose()` for
    the teardown block to call."""

    class _FakeEngine:
        async def dispose(self) -> None:
            return None

    return _FakeEngine()


def _fake_build_repositories(config: object, engine: object) -> dict:
    """Replaces `spire_voice.app._build_repositories`: the real one builds
    `PostgresPolicyRepository`/`PostgresAccountRepository` against a real
    sessionmaker. Substituting `conftest.FakePolicyRepository`/
    `conftest.FakeAccountRepository` here is what lets this smoke test
    cover `lifespan`'s repository wiring without a reachable Postgres --
    the same Postgres-free precedent D-04 sets for the rest of the suite.

    Plan 03-05: every route this application registers now sits behind
    `require_setup_complete` (an application-level dependency reading
    `app.state.account_repo`), so a test that boots the real app and calls
    any route at all -- not just one this file's own tests exercise --
    needs this key present, not only `policy_repo`.

    The `FakeAccountRepository` here is pre-seeded with one fake admin
    account, directly (bypassing the async `create_user`, which this sync
    function cannot `await`) -- every test in this file is asserting on
    *resource wiring*, not on setup completeness (`tests/test_auth_setup.py`
    owns that, against its own, deliberately empty fake), so the "already
    set up" state is this file's correct default rather than every one of
    its existing assertions turning into an unrelated 503.
    """
    account_repo = conftest.FakeAccountRepository()
    account_repo.users[1] = User(
        id=1,
        email="smoke-admin@example.invalid",
        display_name="Smoke Admin",
        password_hash="not-a-real-hash-never-checked",
        role="admin",
        created_at=datetime.now(timezone.utc),
        disabled_at=None,
    )
    account_repo._next_user_id = 2
    return {
        "policy_repo": conftest.FakePolicyRepository(),
        "account_repo": account_repo,
        "credential_repo": conftest.FakeCredentialRepository(),
        # Plan 03-09: `lifespan` now also resolves the wizard's own
        # audio-source setting (`resolve_audio_source`) before any provider
        # is constructed, which needs `repositories["settings_repo"]`
        # present -- omitting it here would `KeyError` on every boot this
        # fake drives, the same reasoning `credential_repo` was added for
        # in plan 03-07.
        "setup_repo": conftest.FakeSetupRepository(),
        "settings_repo": conftest.FakeSettingsRepository(),
        # Plan 04-05: `lifespan` now also reads macros from a repository
        # (`macro_repo`) rather than from `config.macros` -- omitting it
        # here would `KeyError` on every boot this fake drives, the same
        # reasoning `credential_repo`/`settings_repo` were added for above.
        "macro_repo": conftest.FakeMacroRepository(),
    }


class _FakeWakeDetector:
    """Stands in for `VoskWakeDetector`: the real one needs a model
    directory that is a deployment artifact and is not in this repository
    (RESEARCH.md Pitfall 3). This fake proves `lifespan`'s wiring lands the
    detector on `app.state` without ever loading a real model."""

    def process(self, chunk: bytes) -> None:
        return None

    def close(self) -> None:
        return None


def _fake_build_wake_detector(wake_config: object) -> _FakeWakeDetector:
    return _FakeWakeDetector()


class _FakeFfmpegSupervisor:
    """Stands in for `FfmpegSupervisor`: the real one spawns an actual
    `ffmpeg` child pointed at `speaker.fifo_path`, which this test never
    creates. `start()`/`stop()` are no-ops -- proving `lifespan`'s wiring
    without requiring the `ffmpeg` binary or a real FIFO to be present.

    `handle_reconnect()` counts its own calls rather than being a no-op:
    `test_camera_reconnect_is_wired_to_the_speaker_backchannel` below calls
    whatever `app.py` passed as `CameraAudioSource`'s `on_reconnect`
    argument and asserts this counter moved, proving the wiring is a real
    connection between the two supervisors and not merely a non-`None`
    value (Rule 2 fix, plan 02-08)."""

    def __init__(self) -> None:
        self.reconnect_count = 0

    def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def handle_reconnect(self) -> None:
        self.reconnect_count += 1


def _fake_build_ffmpeg_supervisor(config: object, http_client: object) -> _FakeFfmpegSupervisor:
    return _FakeFfmpegSupervisor()


class _FakeCameraSource:
    """Stands in for `CameraAudioSource`: real construction below is only
    there to prove `app.py` actually passes `on_reconnect` through, never
    to exercise RTSP/PyAV against a fake URL. Records the `on_reconnect`
    keyword argument it was constructed with; `start()`/`close()` are
    no-ops, matching every other lifespan resource this file fakes."""

    def __init__(self, config: object, speaker: object, *, on_reconnect=None, **_kwargs: object) -> None:
        self.on_reconnect = on_reconnect
        self._config = config

    def start(self) -> None:
        return None

    async def frames(self):
        # An immediately-finished, empty stream -- `SourceRunner.run()`'s
        # `async for` loop just ends, matching `_FakeWakeDetector`'s own
        # "no real camera/model reachable" posture for this smoke test.
        return
        yield  # pragma: no cover -- makes this an async generator

    def decode_for_detector(self, chunk: bytes) -> bytes:
        return chunk

    def source_format(self) -> SourceFormat:
        # `app.py` calls this to size the `PrerollBuffer` it builds for the
        # runner (CR-01 fix) -- read off the fake config's own `camera`
        # section the same way the real `CameraAudioSource.source_format()`
        # reads `CameraConfig`, rather than a value this fake invents.
        return SourceFormat(encoding=self._config.encoding, sample_rate=self._config.sample_rate)

    async def close(self) -> None:
        return None


def test_lifespan_starts_and_assigns_every_owned_resource(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)

    with TestClient(app_module.app) as client:
        response = client.get("/transport")
        assert response.status_code == 200

        missing = [name for name in _EXPECTED_STATE_ATTRS if getattr(app_module.app.state, name, None) is None]
        assert not missing, (
            "lifespan never assigned app.state." + ", app.state.".join(missing) + " -- "
            "this is the exact defect shape three of Phase 1's Critical findings shared"
        )


def test_a_failing_migration_stops_the_boot_rather_than_yielding_a_running_application(
    tmp_path, monkeypatch
):
    """Phase 1 shipped three Critical defects on startup paths no test
    exercised. This is that same discipline applied to the newest startup
    path this phase adds: a migration failure must propagate uncaught and
    stop the process, never leave a route table answering against an
    unmigrated schema (T-03-09). Substitutes a `run_migrations` that raises,
    and asserts entering the `TestClient` context manager -- which is what
    actually runs `lifespan` -- raises too, rather than yielding a working
    application.
    """

    def _raising_run_migrations(migration_url: str) -> None:
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _raising_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)

    with pytest.raises(RuntimeError, match="simulated migration failure"):
        with TestClient(app_module.app):
            pass


def test_lifespan_starts_a_weather_child_alongside_home_assistant_and_builds_a_lookup(
    tmp_path, monkeypatch
):
    """D-13, T-04-13: a second `McpToolHost` for weather, a small
    name-to-host lookup over both, `app.state.tool_host` still the exact
    Home Assistant host object, and the merged tool schema carrying both
    children's tool names.
    """
    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(_write_fake_config(tmp_path, extra=_weather_mcp_servers_extra())),
    )
    monkeypatch.setattr(app_module, "McpToolHost", _make_two_host_fake_tool_host_factory())
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)

    with TestClient(app_module.app) as client:
        response = client.get("/transport")
        assert response.status_code == 200

        ha_host = app_module.app.state.tool_host
        weather_host = app_module.app.state.weather_tool_host
        lookup = app_module.app.state.tool_host_lookup

        assert weather_host is not None
        assert lookup is not None
        # T-04-13: the policy editor's respawn (routes/policy.py) and
        # _make_state_fetch both read app.state.tool_host by name -- it
        # must still be the exact object the Home Assistant child was
        # started on, not the lookup and not the weather host.
        assert app_module.app.state.tool_host is ha_host
        assert app_module.app.state.tool_host is not lookup
        assert app_module.app.state.tool_host is not weather_host

        tool_names = {entry["function"]["name"] for entry in app_module.app.state.tools_schema}
        assert "ha_list_entities" in tool_names
        assert "weather_current" in tool_names


def test_a_weather_child_that_fails_to_start_does_not_stop_the_boot(tmp_path, monkeypatch):
    """T-04-16: a weather child that will not start must not take the
    whole application down with it -- the opposite posture from the Home
    Assistant child, whose `start()` failure does propagate. The
    application still boots, with the Home Assistant tools reachable and
    no weather host in the lookup.
    """
    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(_write_fake_config(tmp_path, extra=_weather_mcp_servers_extra())),
    )
    monkeypatch.setattr(app_module, "McpToolHost", _make_failing_weather_tool_host_factory())
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)

    with TestClient(app_module.app) as client:
        response = client.get("/transport")
        assert response.status_code == 200

        assert app_module.app.state.weather_tool_host is None
        assert app_module.app.state.tool_host_lookup is not None

        tool_names = {entry["function"]["name"] for entry in app_module.app.state.tools_schema}
        assert "ha_list_entities" in tool_names
        assert "weather_current" not in tool_names


def test_camera_reconnect_is_wired_to_the_speaker_backchannel(tmp_path, monkeypatch):
    """Rule 2 fix, plan 02-08: plan 02-07 added `CameraAudioSource`'s
    `on_reconnect` hook and `FfmpegSupervisor.handle_reconnect()`, but
    `app.py` was outside that plan's file scope and never connected the
    two. Left unfixed, a camera that drops and reconnects gets its
    microphone back while the speaker stays silent, and nothing raises --
    the failure is invisible until an operator notices the house cannot
    answer. This constructs the real `lifespan` and calls whatever
    `on_reconnect` it actually passed, proving the wiring is a real
    connection between the two supervisors rather than merely present."""
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with TestClient(app_module.app):
        camera_source = app_module.app.state.camera_source
        assert camera_source.on_reconnect is not None, (
            "app.py must construct CameraAudioSource with on_reconnect="
            "ffmpeg_supervisor.handle_reconnect, or a camera reconnect never "
            "re-establishes the go2rtc speaker backchannel"
        )

        asyncio.run(camera_source.on_reconnect())

        assert app_module.app.state.ffmpeg_supervisor.reconnect_count == 1, (
            "on_reconnect must be the real ffmpeg_supervisor.handle_reconnect -- "
            "calling it did not reach the supervisor that owns the backchannel"
        )


def test_camera_runner_is_wired_with_the_configured_gate_and_barge_in_policy(tmp_path, monkeypatch):
    """CR-01 fix (code review): `app.py`'s only production `SourceRunner`
    used to be built with none of `wake_config`/`gate_config`/
    `barge_in_config`/`preroll` -- every one of `SourceRunner.__init__`'s
    own "absent configuration" fallbacks then applied silently: refractory
    0.0, an empty mute list, `BargeInConfig(enabled=False)` regardless of
    what the config file said, and no `PrerollBuffer` at all. Two plans'
    own SUMMARY files recorded this as a gap for a later plan to close, and
    no later plan did -- `test_lifespan_starts_and_assigns_every_owned_resource`
    above only asserts each resource is *assigned* to `app.state`, never
    that it is *configured*, so nothing in the suite caught it either.

    This sets distinctive, non-default values for every one of those four
    and inspects the constructed runner's own resolved state directly --
    the shape this test's own docstring, and CR-01's own Fix text, asks
    for -- rather than a behavioral test that would need to boot a real STT/
    brain/TTS pipeline just to observe a wake hit's side effects.
    """
    extra = {
        "camera": {"preroll_ms": 640, "encoding": "alaw", "sample_rate": 8000},
        "wake": {"refractory_s": 9.5},
        "gate": {"mute_when_playing": ["media_player.example_smoke_test_tv"]},
        "barge_in": {"min_duration_ms": 555},
    }
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path, extra=extra)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with TestClient(app_module.app):
        runner = app_module.app.state.source_runners[0]

        # The gate (WakeConfig.refractory_s + GateConfig.mute_when_playing,
        # resolved against "camera") -- reading `WakeGate`'s own private
        # state is what "inspects the constructed runner's resolved gate ...
        # state" (CR-01's Fix text) actually means: there is no public
        # accessor, and the previous bug's whole shape was this exact state
        # silently resolving to its inert default.
        assert runner._gate._refractory_s == 9.5, (
            "wake_config was not threaded through to SourceRunner -- the "
            "configured refractory window never takes effect"
        )
        assert runner._gate._mute_when_playing == ("media_player.example_smoke_test_tv",), (
            "gate_config was not threaded through to SourceRunner -- a "
            "configured mute_when_playing list is silently ignored"
        )

        # Barge-in: the resolved BargeInConfig, not the inert
        # BargeInConfig(enabled=False) SourceRunner.__init__ falls back to
        # when barge_in_config=None.
        assert runner._barge_in_config.min_duration_ms == 555, (
            "barge_in_config was not threaded through to SourceRunner -- "
            "the configured value never reaches BargeInMonitor"
        )

        # Pre-roll: a PrerollBuffer sized from the camera's own declared
        # format and config.camera.preroll_ms, not the "no preroll at all"
        # default.
        assert runner._preroll is not None, (
            "no PrerollBuffer was constructed -- VOICE-06 never replays "
            "pre-wake audio into the transcriber"
        )
        # alaw is 1 byte/sample at 8000 Hz: 8 bytes/ms * 640ms.
        assert runner._preroll._max_bytes == 8 * 640


# --- Plan 02-12 Task 3: correlation wiring and the startup refusal ----------


def _write_calibration(calib_dir: Path, *, taken_at) -> None:
    """A plainly fictional, valid `EchoCalibration`, written under
    `calib_dir` with the same filename shape `calibration/runner.py`'s own
    `_timestamped_filename` produces -- `find_latest_calibration` globs by
    that shape, not by any name."""
    from spire_voice.calibration.record import EchoCalibration

    calibration = EchoCalibration(
        schema_version=1,
        probe_format_version=1,
        probe_seed=1,
        source="camera",
        delay_s=0.05,
        confidence=0.95,
        echo_level=0.1,
        gain=1.2,
        agc_verdict="absent",
        segment_levels=(0.1, 0.1, 0.1),
        encoding="alaw",
        sample_rate=8000,
        channels=1,
        placement_note="smoke test fixture, no real room",
        taken_at=taken_at,
    )
    stamp = taken_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    calibration.save(calib_dir / f"echo_path-{stamp}Z.json")


def test_correlation_enabled_with_no_calibration_refuses_to_start(tmp_path, monkeypatch):
    """`ConfigError` is read off `app_module` itself, not re-imported from
    `spire_voice.config` fresh -- `tests/test_config.py`'s own
    `test_config_and_turn_macros_import_in_either_order` reloads
    `spire_voice.config` (proving no import cycle), which mints a *new*
    `ConfigError` class distinct from the one `app.py` captured at its own
    import time. A fresh import here would build a `pytest.raises` that
    can never match what `app.py` actually raises once that reload has run
    earlier in the same test session -- `app_module.ConfigError` is
    guaranteed to be the exact class `app.py`'s code raises, regardless.
    """
    ConfigError = app_module.ConfigError

    extra = {
        "barge_in": {"correlation_enabled": True},
        "calibration": {"dir": str(tmp_path / "calibration")},
    }
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path, extra=extra)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with pytest.raises(ConfigError, match="correlation_enabled.*scripts/dev-calibrate-echo.sh"):
        with TestClient(app_module.app):
            pass


def test_correlation_enabled_with_a_stale_calibration_refuses_to_start(tmp_path, monkeypatch):
    """See the sibling test above for why `ConfigError` comes off
    `app_module` rather than a fresh import."""
    from datetime import datetime, timedelta, timezone as _timezone

    ConfigError = app_module.ConfigError

    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    stale_taken_at = datetime.now(_timezone.utc) - timedelta(days=31)  # past the default max_age_days=30
    _write_calibration(calib_dir, taken_at=stale_taken_at)

    extra = {
        "barge_in": {"correlation_enabled": True},
        "calibration": {"dir": str(calib_dir)},
    }
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path, extra=extra)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with pytest.raises(ConfigError, match="days old.*max_age_days"):
        with TestClient(app_module.app):
            pass


def test_correlation_enabled_with_a_valid_calibration_wires_the_runner(tmp_path, monkeypatch):
    from datetime import datetime, timezone as _timezone

    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    fresh_taken_at = datetime.now(_timezone.utc)
    _write_calibration(calib_dir, taken_at=fresh_taken_at)

    extra = {
        "barge_in": {"correlation_enabled": True},
        "calibration": {"dir": str(calib_dir)},
    }
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path, extra=extra)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with TestClient(app_module.app):
        runner = app_module.app.state.source_runners[0]
        assert runner._calibration is not None, (
            "correlation_enabled with a valid calibration on file must attach it to the runner"
        )
        assert runner._calibration.gain == 1.2
        assert runner._barge_in_config.correlation_enabled is True


def test_correlation_disabled_boots_unchanged_and_attaches_no_calibration(tmp_path, monkeypatch):
    """The shipped default: no calibration directory even exists, and
    nothing about startup looks at it."""
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with TestClient(app_module.app):
        runner = app_module.app.state.source_runners[0]
        assert runner._calibration is None
        assert runner._barge_in_config.correlation_enabled is False


def test_an_unreadable_calibration_with_correlation_off_boots_normally(tmp_path, monkeypatch):
    """A corrupt record must never stop the house from listening (T-02-58)
    -- with correlation off, startup never even reads the directory."""
    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    (calib_dir / "echo_path-20260101T000000000000Z.json").write_text("not valid json{{{", encoding="utf-8")

    extra = {"calibration": {"dir": str(calib_dir)}}
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path, extra=extra)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with TestClient(app_module.app) as client:
        response = client.get("/transport")
        assert response.status_code == 200
        runner = app_module.app.state.source_runners[0]
        assert runner._calibration is None


# --- Plan 03-05 Task 4: the setup gate, through a real boot -----------------


def test_the_real_boot_answers_create_admin_while_every_other_route_reports_setup_incomplete(
    tmp_path, monkeypatch
):
    """Phase 1's own lesson, applied to the newest startup path this phase
    adds: `tests/test_auth_setup.py` already proves the setup gate
    exhaustively, over every registered route, but against a hand-rolled
    minimal boot. This is that same guarantee's cheapest possible real-boot
    form, in the file whose whole purpose is "the real `lifespan`, not a
    unit call" -- a genuinely empty `FakeAccountRepository` (unlike this
    file's own default, pre-seeded for its unrelated wiring assertions),
    `POST /api/auth/create-admin` answering, and an ordinary route
    (`/transport`) refusing with the named 503 until it does.
    """
    import conftest

    def _empty_repositories(config: object, engine: object) -> dict:
        return {
            "policy_repo": conftest.FakePolicyRepository(),
            "account_repo": conftest.FakeAccountRepository(),
            "credential_repo": conftest.FakeCredentialRepository(),
            "setup_repo": conftest.FakeSetupRepository(),
            "settings_repo": conftest.FakeSettingsRepository(),
            "macro_repo": conftest.FakeMacroRepository(),
        }

    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _empty_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with TestClient(app_module.app) as client:
        before = client.get("/transport")
        assert before.status_code == 503
        assert "setup incomplete" in before.json()["detail"].lower()

        created = client.post(
            "/api/auth/create-admin",
            json={
                "email": "smoke-boot-admin@example.invalid",
                "display_name": "Smoke Boot Admin",
                "password": "a-plainly-fictional-test-password",
            },
        )
        assert created.status_code == 201, created.text

        after = client.get("/transport")
        assert after.status_code == 200
        assert "transport" in after.json()


# --- Plan 03-07 Task 3: credential resolution, through a real boot ---------


def _write_fake_config_with_credential(tmp_path: Path, *, api_key: str, ha_token: str) -> Path:
    """Like `_write_fake_config`, but with every provider credential set
    to exactly `api_key`/`ha_token` -- `_write_fake_config`'s own base
    dict already carries `"test-key"`/`"test-token"`, but this file's two
    new tests below need full control over the value (empty, to prove a
    clean boot with nothing stored and nothing in the environment; a
    single known literal, to prove startup logs its source and never its
    value)."""
    extra = {
        "stt": {"url": "wss://stt.invalid/v1/stt", "api_key": api_key},
        "brain": {
            "base_url": "https://brain.invalid/v1",
            "api_key": api_key,
            "models": [{"model": "fake-model"}],
        },
        "tts": {
            "url": "https://tts.invalid/v1/tts",
            "api_key": api_key,
            "voice_id": "eve",
            "cache_dir": str(tmp_path / "tts-cache"),
        },
        "mcp": {
            "servers": {
                "ha": {
                    "args": ["-m", "spire_mcp.ha"],
                    "env": {"HA_URL": "http://ha.invalid", "HA_TOKEN": ha_token},
                },
            },
        },
    }
    return _write_fake_config(tmp_path, extra=extra)


def test_the_application_starts_with_every_credential_slot_unset(tmp_path, monkeypatch):
    """No stored credential and no environment value for any slot -- the
    clean-install state DEP-03 describes -- must be a boot, not an error,
    and `GET /api/credentials` must answer every slot unset rather than
    failing."""
    from spire_voice.auth.tokens import issue_access_token
    from spire_voice.config import SecurityConfig

    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(_write_fake_config_with_credential(tmp_path, api_key="", ha_token="")),
    )
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)

    with TestClient(app_module.app) as client:
        # `_fake_build_repositories` pre-seeds one fake admin (id=1) so
        # this file's own unrelated wiring assertions are not all 503s
        # (that function's own docstring) -- sign in as that admin to
        # reach the admin-only credentials listing route.
        security = SecurityConfig()
        token = issue_access_token(user_id=1, role="admin", security=security)
        client.cookies.set(security.cookie_name, token)

        response = client.get("/api/credentials")
        assert response.status_code == 200, response.text
        entries = response.json()
        assert len(entries) == 4
        for entry in entries:
            assert entry["is_set"] is False, f"{entry['slot']} was reported set on a clean boot"
            assert entry["source"] == "unset"


def test_startup_logs_which_source_won_per_slot_never_by_value(tmp_path, monkeypatch, caplog):
    """T-03-43: startup logs the winning source per slot, by name, and no
    log line anywhere carries a credential value."""
    secret_value = "a-plainly-fictional-test-provider-value-never-a-real-secret"

    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(_write_fake_config_with_credential(tmp_path, api_key=secret_value, ha_token=secret_value)),
    )
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)

    with caplog.at_level(logging.INFO, logger="spire_voice.app"):
        with TestClient(app_module.app):
            pass

    resolution_records = [r for r in caplog.records if "credential slot" in r.getMessage()]
    assert len(resolution_records) == 4, (
        f"expected one resolution log line per slot, got {[r.getMessage() for r in resolution_records]!r}"
    )
    for record in resolution_records:
        message = record.getMessage()
        assert "resolved from environment" in message
        assert secret_value not in message

    for record in caplog.records:
        assert secret_value not in record.getMessage(), (
            f"a credential value leaked into a log line: {record.getMessage()!r}"
        )


def _fake_build_repositories_with_audio_source(stored_source: str | None):
    """Like `_fake_build_repositories`, but the returned `settings_repo`
    is pre-seeded with `audio_source` when `stored_source` is not `None` --
    a plain dict assignment on the fake's own `.settings` mapping, not a
    call through its async `set_setting` (this factory runs synchronously,
    inside `lifespan`'s own synchronous `_build_repositories(...)` call, on
    a thread already driving an event loop -- `asyncio.run()` would raise
    there)."""

    def _factory(config: object, engine: object) -> dict:
        repositories = _fake_build_repositories(config, engine)
        if stored_source is not None:
            settings_repo: conftest.FakeSettingsRepository = repositories["settings_repo"]
            settings_repo.settings["audio_source"] = Setting(
                id=1,
                key="audio_source",
                value=stored_source,
                updated_at=datetime.now(timezone.utc),
                updated_by_user_id=None,
            )
        return repositories

    return _factory


def test_the_application_reads_the_stored_audio_source_setting_when_present(
    tmp_path, monkeypatch, caplog
):
    """Plan 03-09 Task 2: the stored `audio_source` setting is read in
    `lifespan`, in preference to this file's own shipped default, proven
    through a real boot -- not a unit call against `resolve_audio_source`
    alone."""
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(
        app_module, "_build_repositories", _fake_build_repositories_with_audio_source("camera")
    )
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)

    with caplog.at_level(logging.INFO, logger="spire_voice.app"):
        with TestClient(app_module.app):
            pass

    resolution_records = [r for r in caplog.records if "audio source resolved" in r.getMessage()]
    assert len(resolution_records) == 1, resolution_records
    message = resolution_records[0].getMessage()
    assert "'camera'" in message
    assert "from database" in message


def test_the_application_falls_back_to_the_configuration_file_when_no_setting_is_stored(
    tmp_path, monkeypatch, caplog
):
    """The reverse of the test above: nothing stored in `settings` at
    all -- the shipped default (`camera`) wins, and the resolution is
    reported as coming from the configuration, not the database."""
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", _fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", _fake_build_engine)
    monkeypatch.setattr(
        app_module, "_build_repositories", _fake_build_repositories_with_audio_source(None)
    )
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)

    with caplog.at_level(logging.INFO, logger="spire_voice.app"):
        with TestClient(app_module.app):
            pass

    resolution_records = [r for r in caplog.records if "audio source resolved" in r.getMessage()]
    assert len(resolution_records) == 1, resolution_records
    message = resolution_records[0].getMessage()
    assert "'camera'" in message
    assert "from config" in message


# --- CR-01 fix: the seed migration must actually run against a real,
# populated safety: block, through the real lifespan, before the check
# that would otherwise reject it -- not `run_migrations` called directly
# (tests/test_db_migrations.py already covers that in isolation), but the
# actual boot path an operator's upgrade takes. ---------------------------

_TEST_DB_URL = os.environ.get("SPIRE_TEST_DATABASE_URL")

skip_without_postgres = pytest.mark.skipif(
    _TEST_DB_URL is None,
    reason=(
        "SPIRE_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite, to exercise this test instead of skipping it"
    ),
)


async def _reset_policy_schema(async_url: str) -> None:
    """Drop every table either policy migration touches, plus Alembic's own
    bookkeeping table (`alembic_version`) -- the same list `tests/
    test_db_migrations.py`'s own `_reset_schema` drops. This test tells a
    first boot from a later one by real Alembic revision history
    (`db.engine.get_current_revision`), so it needs a genuinely empty
    database, not whatever a previous run of this same throwaway Postgres
    left behind.

    Plan 04-05 extends this list with the three tables migration `0005`
    adds (`macros`, `macro_actions`, `macro_aliases`) -- the same
    generalization this file's own boot tests below apply to the two-stage
    boot itself: one shared helper handling a second legacy key's tables,
    not a second, copy-pasted reset function.
    """
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
        for table in (
            "macro_actions",
            "macro_aliases",
            "macros",
            "settings",
            "setup_steps",
            "setup_state",
            "provider_credentials",
            "refresh_tokens",
            "invites",
            "users",
            "policy_rules",
            "safety_policy",
            "audit_log",
            "alembic_version",
        ):
            await conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    await engine.dispose()


@pytest.mark.integration
@skip_without_postgres
def test_first_boot_seeds_a_legacy_safety_block_through_the_real_lifespan_and_the_next_boot_rejects_it(
    tmp_path, monkeypatch
):
    """CR-01 fix, and the test its own Fix section named as the gap this
    project's "every phase boots the real lifespan" convention was supposed
    to catch and did not.

    First boot: a real, freshly-emptied Postgres, and a config file
    carrying a populated `safety:` block (D-11's "the operator's real,
    already-deployed safety: block" scenario, named in 03-CONTEXT.md). The
    application must start -- not raise `ConfigError` before the seed
    migration ever runs, which is exactly what CR-01 caught -- and the
    seeded policy must match the block that named it, not merely a
    non-empty one.

    Second boot: the identical config file, still carrying the same
    `safety:` block, against the database the first boot just seeded. This
    boot must refuse to start, naming the key by the same message this
    project has always used -- proving the rejection D-11 depends on (two
    sources of truth for the safety boundary is the thing SAFE-05 exists to
    end) still fires once the seed has actually happened, and was not
    simply disabled by this fix.

    `run_migrations`, `build_engine`, and `_build_repositories` are left
    real for both boots -- the one thing every other test in this file
    fakes out (D-04's Postgres-free default) and the one thing this
    specific test cannot fake without proving nothing.
    """
    asyncio.run(_reset_policy_schema(_TEST_DB_URL))

    safety_block = {
        "mode": "allow_all_except_denylist",
        "deny_entities": [
            "switch.example_cr01_seed_test_socket",
            "switch.example_cr01_seed_test_heater",
        ],
        "deny_patterns": ["switch.example_cr01_seed_test_camera_*"],
        "allow_entities": ["light.example_cr01_seed_test_lamp"],
    }
    config_path = _write_fake_config(
        tmp_path,
        extra={"database": {"url": _TEST_DB_URL}, "safety": safety_block},
    )
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(config_path))
    # The migration itself (`alembic/versions/0001_policy_tables.py`) reads
    # SPIRE_CONFIG independently of app.py's CONFIG_PATH (its own module
    # docstring) -- both must point at the same file for the seed to read
    # the block this test just wrote.
    monkeypatch.setenv("SPIRE_CONFIG", str(config_path))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    # First boot: the safety: block is present and the database is
    # genuinely empty -- this must succeed, and must seed the block
    # verbatim.
    with TestClient(app_module.app):
        pass

    # Read the seeded rows back through a fresh connection, not
    # `app.state.policy_repo` -- that repository's pool was opened inside
    # `TestClient`'s own portal event loop and is torn down (and, even
    # while open, is bound to a loop this test's plain `asyncio.run` is not
    # running on) by the time the `with` block above exits, the same
    # "query through a new engine" precedent `tests/test_db_migrations.py`'s
    # own `_fetch_rows` already sets for reading a migration's seeded state
    # back out.
    async def _fetch_seeded_policy() -> tuple[str, list[tuple[str, str]]]:
        engine = create_async_engine(_TEST_DB_URL)
        try:
            async with engine.connect() as conn:
                mode = (
                    await conn.execute(text("SELECT mode FROM safety_policy WHERE id = 1"))
                ).scalar_one()
                rules = (
                    await conn.execute(text("SELECT kind, value FROM policy_rules"))
                ).fetchall()
                return mode, [(r.kind, r.value) for r in rules]
        finally:
            await engine.dispose()

    seeded_mode, seeded_rules = asyncio.run(_fetch_seeded_policy())

    assert seeded_mode == safety_block["mode"]
    assert {v for k, v in seeded_rules if k == "deny_entity"} == set(safety_block["deny_entities"])
    assert {v for k, v in seeded_rules if k == "deny_pattern"} == set(safety_block["deny_patterns"])
    assert {v for k, v in seeded_rules if k == "allow_entity"} == set(safety_block["allow_entities"])

    # Second boot: identical config file, same safety: block still
    # present, against the database the first boot just seeded -- this
    # must refuse to start, naming the same key this project has always
    # named. `ConfigError` comes off `app_module` itself, matching the
    # sibling tests above (`test_correlation_enabled_with_no_calibration_
    # refuses_to_start`'s docstring explains why: a fresh import mints a
    # distinct class a real `app.py` raise could never match).
    ConfigError = app_module.ConfigError
    with pytest.raises(ConfigError, match="safety:.*no longer exists"):
        with TestClient(app_module.app):
            pass


# --- Plan 04-05 (D-09, D-16, T-04-23): the same CR-01 proof, generalized to
# a second legacy key. Written first, against a mechanism that does not
# exist yet -- these four collect and fail until app.py's two-stage boot,
# config.py's retirement of the macros: key, and the migration that seeds
# it are all in place (Task 1's own instruction). ---------------------------


def _example_macros_block() -> list[dict]:
    """One invented, two-action, two-alias macro -- fictional throughout
    (`switch.example_*`/`light.example_*`, this repository's own naming
    convention for a value that must never be a real entity id), shaped
    exactly like `MacroConfig.from_config` and `config/config.example.yaml`
    both already expect: a phrase, aliases, a reply, and an ordered action
    list. Content-asserted below, not merely counted -- a seed that dropped
    every alias would still pass a row-count check, and aliases are how a
    macro matches more than one spoken phrase.
    """
    return [
        {
            "phrase": "example goodnight macro",
            "aliases": ["example goodnight", "example night night macro"],
            "reply": "okay, goodnight",
            "actions": [
                {
                    "tool": "ha_call_service",
                    "arguments": {
                        "domain": "switch",
                        "service": "turn_off",
                        "entity_id": "switch.example_macro_seed_fan",
                    },
                },
                {
                    "tool": "ha_call_service",
                    "arguments": {
                        "domain": "light",
                        "service": "turn_off",
                        "entity_id": "light.example_macro_seed_lamp",
                    },
                },
            ],
        },
    ]


async def _fetch_seeded_macros(async_url: str) -> list[dict]:
    """Read `macros`/`macro_actions`/`macro_aliases` back through a fresh
    connection -- the same "never through `app.state`'s own repository"
    precedent the safety-block test above sets, for the same reason: that
    repository's pool is bound to a loop this test's plain `asyncio.run`
    is not running on, and is torn down by the time the `TestClient`
    `with` block exits.

    Returns one dict per macro, actions already ordered by `position` and
    aliases as a plain set -- the shape the tests below compare directly
    against `_example_macros_block()`.
    """
    engine = create_async_engine(async_url)
    try:
        async with engine.connect() as conn:
            macro_rows = (
                await conn.execute(text("SELECT id, phrase, reply FROM macros"))
            ).fetchall()
            macros: list[dict] = []
            for row in macro_rows:
                action_rows = (
                    await conn.execute(
                        text(
                            "SELECT tool, arguments FROM macro_actions "
                            "WHERE macro_id = :macro_id ORDER BY position"
                        ),
                        {"macro_id": row.id},
                    )
                ).fetchall()
                alias_rows = (
                    await conn.execute(
                        text("SELECT alias FROM macro_aliases WHERE macro_id = :macro_id"),
                        {"macro_id": row.id},
                    )
                ).fetchall()
                macros.append(
                    {
                        "phrase": row.phrase,
                        "reply": row.reply,
                        "aliases": {a.alias for a in alias_rows},
                        "actions": [
                            {"tool": a.tool, "arguments": a.arguments} for a in action_rows
                        ],
                    }
                )
            return macros
    finally:
        await engine.dispose()


@pytest.mark.integration
@skip_without_postgres
def test_first_boot_seeds_a_legacy_macros_block_through_the_real_lifespan_and_starts(
    tmp_path, monkeypatch
):
    """The macros: sibling of the safety: test above. First boot, a real,
    freshly-emptied Postgres, a config file carrying a populated macros:
    block: the application must start, and the seeded rows must match the
    block's phrase, every alias, the reply, and the actions in written
    order -- not merely a non-zero row count (D-09).
    """
    asyncio.run(_reset_policy_schema(_TEST_DB_URL))

    macros_block = _example_macros_block()
    config_path = _write_fake_config(
        tmp_path, extra={"database": {"url": _TEST_DB_URL}, "macros": macros_block}
    )
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(config_path))
    # The migration itself reads SPIRE_CONFIG independently of app.py's
    # CONFIG_PATH (mirroring the safety-block test above) -- both must
    # point at the same file for the seed to read the block this test just
    # wrote.
    monkeypatch.setenv("SPIRE_CONFIG", str(config_path))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with TestClient(app_module.app):
        pass

    seeded_macros = asyncio.run(_fetch_seeded_macros(_TEST_DB_URL))
    assert len(seeded_macros) == 1
    seeded = seeded_macros[0]
    expected = macros_block[0]
    assert seeded["phrase"] == expected["phrase"]
    assert seeded["reply"] == expected["reply"]
    assert seeded["aliases"] == set(expected["aliases"])
    assert seeded["actions"] == expected["actions"]


@pytest.mark.integration
@skip_without_postgres
def test_a_second_boot_with_the_legacy_macros_block_still_present_refuses_naming_the_key(
    tmp_path, monkeypatch
):
    """The reject half of the same pair: identical config file, same
    macros: block still present, against the database the first boot just
    seeded -- this must refuse to start, naming the same key by the same
    wording `MACROS_KEY_REJECTED_ERROR` gives it, proving the rejection
    still fires once the seed has actually happened and was not simply
    disabled by the generalization (CR-01's own lesson, applied to the
    second key).
    """
    asyncio.run(_reset_policy_schema(_TEST_DB_URL))

    config_path = _write_fake_config(
        tmp_path,
        extra={"database": {"url": _TEST_DB_URL}, "macros": _example_macros_block()},
    )
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(config_path))
    monkeypatch.setenv("SPIRE_CONFIG", str(config_path))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    # First boot: seeds silently.
    with TestClient(app_module.app):
        pass

    # Second boot: the same macros: block, against the now-seeded
    # database -- must refuse, naming the key.
    ConfigError = app_module.ConfigError
    with pytest.raises(ConfigError, match="macros:.*no longer exists"):
        with TestClient(app_module.app):
            pass


@pytest.mark.integration
@skip_without_postgres
def test_a_first_boot_with_both_legacy_safety_and_macros_keys_present_seeds_both_and_starts(
    tmp_path, monkeypatch
):
    """T-04-23's own named worry: a copy-pasted second branch is most
    likely to get the two-key case wrong. A single boot, both safety: and
    macros: present in the same file, against a genuinely empty database
    -- the application must start, and both must be seeded, proving one
    shared sequence handles two legacy keys rather than two independent
    ones that happen to both work alone.
    """
    asyncio.run(_reset_policy_schema(_TEST_DB_URL))

    safety_block = {
        "mode": "allow_all_except_denylist",
        "deny_entities": ["switch.example_both_keys_seed_test_socket"],
    }
    macros_block = _example_macros_block()
    config_path = _write_fake_config(
        tmp_path,
        extra={
            "database": {"url": _TEST_DB_URL},
            "safety": safety_block,
            "macros": macros_block,
        },
    )
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(config_path))
    monkeypatch.setenv("SPIRE_CONFIG", str(config_path))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    with TestClient(app_module.app):
        pass

    async def _fetch_seeded_policy_mode() -> str:
        engine = create_async_engine(_TEST_DB_URL)
        try:
            async with engine.connect() as conn:
                return (
                    await conn.execute(text("SELECT mode FROM safety_policy WHERE id = 1"))
                ).scalar_one()
        finally:
            await engine.dispose()

    seeded_mode = asyncio.run(_fetch_seeded_policy_mode())
    assert seeded_mode == safety_block["mode"]

    seeded_macros = asyncio.run(_fetch_seeded_macros(_TEST_DB_URL))
    assert len(seeded_macros) == 1
    assert seeded_macros[0]["phrase"] == macros_block[0]["phrase"]


@pytest.mark.integration
@skip_without_postgres
def test_a_boot_with_migrations_disabled_and_a_macros_block_present_refuses(
    tmp_path, monkeypatch
):
    """A boot that never runs the migration must not warn-and-continue on
    a lingering macros: key -- nothing seeded it, so tolerating the key
    here would strand the operator on the same "the log line describes a
    seed that never happened" failure CR-01 already found once. Setting
    `database.run_migrations_at_startup: false` reaches the classification
    step with `is_first_seed_boot` false unconditionally (the same
    generalized rule the two boots above exercise from the other side),
    so this must refuse rather than seed-and-warn.
    """
    asyncio.run(_reset_policy_schema(_TEST_DB_URL))

    config_path = _write_fake_config(
        tmp_path,
        extra={
            "database": {"url": _TEST_DB_URL, "run_migrations_at_startup": False},
            "macros": _example_macros_block(),
        },
    )
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(config_path))
    monkeypatch.setenv("SPIRE_CONFIG", str(config_path))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", _fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", _FakeCameraSource)

    ConfigError = app_module.ConfigError
    with pytest.raises(ConfigError, match="macros:.*no longer exists"):
        with TestClient(app_module.app):
            pass
