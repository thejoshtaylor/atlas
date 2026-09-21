"""The server half of DBG-05: a threshold an operator moves takes effect
on the very next wake hit, with no restart (D-15), and the read side that
exposes the recorded wake attempts the tuning screen partitions (D-13,
D-14a, D-16).

Task 1 covers the gate- and runner-level plumbing only: `WakeGate.
set_threshold`/`.threshold` and `SourceRunner.wake_threshold`/
`.set_wake_threshold`. Tasks 2 and 3 (the route and the boot-time read)
add their own test classes below as they land.
"""

from __future__ import annotations

import logging
import math

from spire_voice.config import GateConfig, WakeConfig
from spire_voice.sources.runner import SourceRunner
from spire_voice.wake.gate import WakeGate

from tests.conftest import FakeAudioSource, FakeWakeHit


# --- WakeGate: set_threshold takes effect on the very next evaluation ------


def test_a_threshold_set_after_construction_governs_the_next_evaluation():
    gate = WakeGate(threshold=0.20, refractory_s=0.0, mute_when_playing=())

    # Under the original threshold, this score is a hit.
    assert gate.evaluate(score=0.25, now=1.0, last_hit_at=None).allowed is True

    gate.set_threshold(0.90)

    # The same score, under the new threshold, is not.
    decision = gate.evaluate(score=0.25, now=2.0, last_hit_at=None)
    assert decision.allowed is False
    assert decision.reason == "below_threshold"


def test_a_score_exactly_equal_to_the_new_threshold_is_a_hit():
    gate = WakeGate(threshold=0.10, refractory_s=0.0, mute_when_playing=())
    gate.set_threshold(0.75)

    decision = gate.evaluate(score=0.75, now=1.0, last_hit_at=None)
    assert decision.allowed is True
    assert decision.reason is None


def test_a_score_one_representable_step_below_the_new_threshold_is_not_a_hit():
    gate = WakeGate(threshold=0.10, refractory_s=0.0, mute_when_playing=())
    gate.set_threshold(0.75)
    just_below = math.nextafter(0.75, -math.inf)

    decision = gate.evaluate(score=just_below, now=1.0, last_hit_at=None)
    assert decision.allowed is False
    assert decision.reason == "below_threshold"


def test_threshold_property_reads_the_live_value_not_the_constructed_one():
    gate = WakeGate(threshold=0.30, refractory_s=0.0, mute_when_playing=())
    assert gate.threshold == 0.30
    gate.set_threshold(0.60)
    assert gate.threshold == 0.60


# --- SourceRunner: named accessors delegating to the gate -------------------


def _build_runner(*, engine: str = "openwakeword") -> SourceRunner:
    source = FakeAudioSource(frames=[])
    wake_config = WakeConfig(engine=engine, refractory_s=0.0)
    gate_config = GateConfig()

    async def _run_turn(src: object) -> None:
        pass

    return SourceRunner(
        "camera",
        source,
        _AlwaysHitWakeDetector(score=1.0),
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
    )


class _AlwaysHitWakeDetector:
    def __init__(self, score: float) -> None:
        self._score = score

    def process(self, chunk: bytes):
        return FakeWakeHit(score=self._score)


def test_source_runner_wake_threshold_reports_the_gates_current_value():
    runner = _build_runner()
    # `WakeConfig(engine="openwakeword")`'s default `OpenWakeWordConfig.
    # threshold` is 0.55 -- proving this property reads through the gate,
    # not a value `SourceRunner` cached separately.
    assert runner.wake_threshold == 0.55


def test_source_runner_set_wake_threshold_changes_the_next_hit_evaluated():
    runner = _build_runner()
    runner.set_wake_threshold(0.90)
    assert runner.wake_threshold == 0.90
    # And the gate itself, reached the only way this module permits, has
    # actually moved.
    assert runner._gate.evaluate(score=0.80, now=1.0, last_hit_at=None).allowed is False
    assert runner._gate.evaluate(score=0.90, now=1.0, last_hit_at=None).allowed is True


async def test_a_runner_nobody_calls_set_wake_threshold_on_behaves_as_before(caplog):
    """Neither addition changes the behaviour of a runner nobody calls
    them on: the existing threshold-boundary test in
    `tests/test_wake_gating.py` already proves this at the `run()` level;
    this is the same assertion made directly against `wake_threshold`."""
    source = FakeAudioSource(frames=[b"\x00\x01"])
    wake_config = WakeConfig(engine="openwakeword", refractory_s=0.0)
    gate_config = GateConfig()

    turns_started: list[object] = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    runner = SourceRunner(
        "camera",
        source,
        _AlwaysHitWakeDetector(score=0.55),
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
    )

    assert runner.wake_threshold == 0.55
    with caplog.at_level(logging.INFO, logger="spire_voice.sources.runner"):
        await runner.run()
    assert len(turns_started) == 1
