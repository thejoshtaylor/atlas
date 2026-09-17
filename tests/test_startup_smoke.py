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

import asyncio
from pathlib import Path
from types import SimpleNamespace

import yaml
from fastapi.testclient import TestClient
from mcp.types import Tool

import spire_voice.app as app_module
from spire_voice.transports.base import SourceFormat
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
    # Plan 02-08: the retention sweep. Starts and stops on the same
    # lifespan-owned pattern as `ffmpeg_supervisor` above; needs no
    # reachable session directory on disk to land on `app.state`.
    "retention_scheduler",
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
        "safety": {},
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
