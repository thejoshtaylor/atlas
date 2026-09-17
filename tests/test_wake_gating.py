"""Real assertions for wake-hit gating (VOICE-03): the threshold boundary,
the refractory boundary, media-player suppression, and the record every
block leaves (D-03), plus per-source policy resolution with no branch on a
source's name (D-04).

`WakeGate` is tested directly, as the pure policy it is; `SourceRunner` is
tested for the parts only it owns -- resolving the policy per source and
recording a block rather than dropping it.
"""

from __future__ import annotations

import logging
import math

from spire_voice.config import GateConfig, WakeConfig
from spire_voice.sources.runner import SourceRunner
from spire_voice.wake.gate import WakeGate

from tests.conftest import FakeAudioSource, FakeWakeHit


# --- WakeGate: the pure boundary tests --------------------------------


def test_score_exactly_at_threshold_is_a_hit():
    gate = WakeGate(threshold=0.55, refractory_s=2.0, mute_when_playing=())
    decision = gate.evaluate(score=0.55, now=100.0, last_hit_at=None)
    assert decision.allowed is True
    assert decision.reason is None


def test_score_one_representable_step_below_threshold_is_not_a_hit():
    threshold = 0.55
    just_below = math.nextafter(threshold, -math.inf)
    gate = WakeGate(threshold=threshold, refractory_s=2.0, mute_when_playing=())
    decision = gate.evaluate(score=just_below, now=100.0, last_hit_at=None)
    assert decision.allowed is False
    assert decision.reason == "below_threshold"


def test_hit_exactly_at_end_of_refractory_window_counts():
    gate = WakeGate(threshold=0.5, refractory_s=2.0, mute_when_playing=())
    decision = gate.evaluate(score=1.0, now=102.0, last_hit_at=100.0)
    assert decision.allowed is True
    assert decision.reason is None


def test_hit_inside_refractory_window_is_suppressed():
    gate = WakeGate(threshold=0.5, refractory_s=2.0, mute_when_playing=())
    decision = gate.evaluate(score=1.0, now=101.999, last_hit_at=100.0)
    assert decision.allowed is False
    assert decision.reason == "refractory"


def test_hit_blocked_while_a_configured_player_is_playing():
    gate = WakeGate(
        threshold=0.5,
        refractory_s=0.0,
        mute_when_playing=("media_player.example_tv",),
        is_media_playing=lambda players: True,
    )
    decision = gate.evaluate(score=1.0, now=100.0, last_hit_at=None)
    assert decision.allowed is False
    assert decision.reason == "media_playing"


def test_empty_mute_list_never_calls_the_media_playing_check():
    """A house with `mute_when_playing` empty pays nothing (D-02) -- the
    injected callable is never even reached."""
    calls: list[tuple[str, ...]] = []

    def _is_media_playing(players: tuple[str, ...]) -> bool:
        calls.append(players)
        return True

    gate = WakeGate(threshold=0.5, refractory_s=0.0, mute_when_playing=(), is_media_playing=_is_media_playing)
    decision = gate.evaluate(score=1.0, now=100.0, last_hit_at=None)

    assert decision.allowed is True
    assert calls == []


# --- SourceRunner: the recorded block and per-source resolution --------


class _AlwaysHitWakeDetector:
    """Fires a hit of a fixed score on every call -- unlike
    `conftest.FakeWakeDetector`, which fires once by design."""

    def __init__(self, score: float = 1.0) -> None:
        self._score = score
        self.calls = 0

    def process(self, chunk: bytes) -> FakeWakeHit:
        self.calls += 1
        return FakeWakeHit(score=self._score)

    def close(self) -> None:
        pass


async def test_blocked_hit_below_threshold_produces_a_record_not_a_silent_drop(caplog):
    source = FakeAudioSource(frames=[b"\x00\x01"])
    detector = _AlwaysHitWakeDetector(score=0.1)
    wake_config = WakeConfig(engine="openwakeword", refractory_s=0.0)  # default threshold 0.55
    gate_config = GateConfig()

    turns_started: list[object] = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
    )

    with caplog.at_level(logging.INFO, logger="spire_voice.sources.runner"):
        await runner.run()

    assert turns_started == []
    blocked = [r for r in caplog.records if getattr(r, "block_reason", None) is not None]
    assert len(blocked) == 1
    assert blocked[0].block_reason == "below_threshold"
    assert blocked[0].source_name == "camera"
    assert blocked[0].score == 0.1


async def test_hit_inside_refractory_window_is_suppressed_and_recorded(caplog):
    source = FakeAudioSource(frames=[b"\x00", b"\x01", b"\x02"])
    detector = _AlwaysHitWakeDetector(score=1.0)
    wake_config = WakeConfig(engine="vosk", refractory_s=2.0)
    gate_config = GateConfig()

    turns_started: list[object] = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    # First hit at t=100 (allowed), second at t=100.5 (0.5s later --
    # inside the 2s window, suppressed), third at t=103.0 (3.0s after the
    # first ALLOWED hit -- at/past the boundary, counts again).
    times = iter([100.0, 100.5, 103.0])

    def clock() -> float:
        return next(times)

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
        clock=clock,
    )

    with caplog.at_level(logging.INFO, logger="spire_voice.sources.runner"):
        await runner.run()

    assert len(turns_started) == 2  # the first and the third hit
    blocked = [r for r in caplog.records if getattr(r, "block_reason", None) is not None]
    assert len(blocked) == 1
    assert blocked[0].block_reason == "refractory"
    assert blocked[0].source_name == "camera"


async def test_hit_blocked_while_media_player_playing_is_recorded(caplog):
    source = FakeAudioSource(frames=[b"\x00"])
    detector = _AlwaysHitWakeDetector(score=1.0)
    wake_config = WakeConfig(engine="vosk", refractory_s=0.0)
    gate_config = GateConfig(mute_when_playing=("media_player.example_tv",))

    turns_started: list[object] = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    runner = SourceRunner(
        "camera",
        source,
        detector,
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
        is_media_playing=lambda players: True,
    )

    with caplog.at_level(logging.INFO, logger="spire_voice.sources.runner"):
        await runner.run()

    assert turns_started == []
    blocked = [r for r in caplog.records if getattr(r, "block_reason", None) is not None]
    assert len(blocked) == 1
    assert blocked[0].block_reason == "media_playing"


async def test_source_without_override_runs_global_policy_source_with_override_runs_merged():
    """Two source names, no code branch between them: a source absent from
    `gate.sources` runs the global (empty) mute list, and a source present
    there runs the merged one -- the browser's difference is data in
    configuration, never a hardcoded special case (D-04)."""
    gate_config = GateConfig(
        mute_when_playing=(),
        sources={"browser": {"mute_when_playing": ("media_player.example_tv",)}},
    )
    wake_config = WakeConfig(engine="vosk", refractory_s=0.0)

    def _build_runner(name: str, calls: list[tuple[str, ...]], turns: list[object]) -> SourceRunner:
        async def _run_turn(src: object) -> None:
            turns.append(src)

        return SourceRunner(
            name,
            FakeAudioSource(frames=[b"\x00"]),
            _AlwaysHitWakeDetector(score=1.0),
            lambda chunk: chunk,
            _run_turn,
            wake_config=wake_config,
            gate_config=gate_config,
            is_media_playing=lambda players: calls.append(players) or True,
        )

    camera_calls: list[tuple[str, ...]] = []
    camera_turns: list[object] = []
    await _build_runner("camera", camera_calls, camera_turns).run()

    browser_calls: list[tuple[str, ...]] = []
    browser_turns: list[object] = []
    await _build_runner("browser", browser_calls, browser_turns).run()

    # "camera" has no override: the global, empty mute list never reaches
    # the injected callable at all, so the hit is allowed.
    assert camera_calls == []
    assert len(camera_turns) == 1

    # "browser" has an override: the merged mute list reaches the
    # callable, which reports the player playing, so the hit is blocked.
    assert browser_calls == [("media_player.example_tv",)]
    assert browser_turns == []
