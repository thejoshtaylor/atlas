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

What this test cannot and does not claim to cover: whether the real RTSP
URL, go2rtc, or a wake model work against actual hardware. That is a live
human check, tracked in this phase's `<live_verification_debt>`, not
something a fake can stand in for.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml
from fastapi.testclient import TestClient
from mcp.types import Tool

import spire_voice.app as app_module
from spire_voice.turn.brain_race import TierBrain

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
]


def _write_fake_config(tmp_path: Path) -> Path:
    """A full config, shaped like `config.example.yaml`, with plainly
    fictional values -- no `${NAME}` placeholder and no `.env` read, since
    this test never expands the environment, matching `tests/test_config.py`
    `_minimal_raw_config()`'s convention: never a real host or entity id.
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
        "safety": {},
    }
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


async def _fake_precache_all(tts: object, cache_dir: Path, texts: list[str], voice_id: str, sink: object) -> dict:
    """Replaces `precache_all`: the real one synthesizes every phrase
    through the TTS provider over the network, once per phrase."""
    return {text: b"" for text in texts}


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
    without requiring the `ffmpeg` binary or a real FIFO to be present."""

    def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _fake_build_ffmpeg_supervisor(config: object, http_client: object) -> _FakeFfmpegSupervisor:
    return _FakeFfmpegSupervisor()


def test_lifespan_starts_and_assigns_every_owned_resource(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(_write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "McpToolHost", _FakeToolHost)
    monkeypatch.setattr(app_module, "precache_all", _fake_precache_all)
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
