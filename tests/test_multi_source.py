"""Real assertions for concurrent multi-source operation (SRC-03): two
sources fire independently, one source's refractory state never touches
another's, a slow turn on one never delays another's start, and the
empty/singleton list edge cases `assemble_source_runners()` owns. Turned
green by plan 02-04.
"""

from __future__ import annotations

import asyncio
from typing import Callable

import pytest

from spire_voice.config import GateConfig, WakeConfig
from spire_voice.sources.runner import SourceRunnerSpec, assemble_source_runners

from tests.conftest import FakeAudioSource, FakeWakeHit


class _AlwaysHitWakeDetector:
    """Fires a hit of a fixed score on every call, unlike
    `conftest.FakeWakeDetector`, which fires once by design."""

    def __init__(self, score: float = 1.0) -> None:
        self._score = score

    def process(self, chunk: bytes) -> FakeWakeHit:
        return FakeWakeHit(score=self._score)

    def close(self) -> None:
        pass


def _spec(
    frames: list[bytes],
    run_turn_fn: Callable,
    *,
    refractory_s: float = 2.0,
    clock: Callable[[], float] | None = None,
) -> SourceRunnerSpec:
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return SourceRunnerSpec(
        source=FakeAudioSource(frames=frames),
        wake_detector=_AlwaysHitWakeDetector(),
        decode_for_detector=lambda chunk: chunk,
        run_turn_fn=run_turn_fn,
        wake_config=WakeConfig(engine="vosk", refractory_s=refractory_s),
        gate_config=GateConfig(),
        **kwargs,
    )


async def test_two_sources_firing_in_the_same_tick_each_start_their_own_turn():
    turns_started: list[str] = []

    async def _camera_turn(source: object) -> None:
        turns_started.append("camera")

    async def _browser_turn(source: object) -> None:
        turns_started.append("browser")

    runners = assemble_source_runners(
        {
            "camera": _spec([b"\x00"], _camera_turn),
            "browser": _spec([b"\x00"], _browser_turn),
        }
    )

    await asyncio.gather(*(runner.run() for runner in runners))

    assert sorted(turns_started) == ["browser", "camera"]


async def test_one_sources_refractory_suppression_does_not_affect_the_other():
    camera_turns: list[str] = []
    browser_turns: list[str] = []

    async def _camera_turn(source: object) -> None:
        camera_turns.append("camera")

    async def _browser_turn(source: object) -> None:
        browser_turns.append("browser")

    # Two hits on the camera 0.5s apart -- inside its own 2s refractory
    # window, so the second is suppressed. The browser's own single hit
    # must still start a turn, unaffected by the camera's state entirely.
    times = iter([100.0, 100.5])

    def camera_clock() -> float:
        return next(times)

    runners = assemble_source_runners(
        {
            "camera": _spec([b"\x00", b"\x01"], _camera_turn, refractory_s=2.0, clock=camera_clock),
            "browser": _spec([b"\x00"], _browser_turn, refractory_s=2.0),
        }
    )

    await asyncio.gather(*(runner.run() for runner in runners))

    assert camera_turns == ["camera"]  # only the first of the two camera hits
    assert browser_turns == ["browser"]  # unaffected by the camera's refractory state


async def test_a_slow_turn_on_one_source_does_not_delay_the_others_start():
    order: list[str] = []
    slow_may_finish = asyncio.Event()

    async def _slow_turn(source: object) -> None:
        order.append("slow_started")
        await slow_may_finish.wait()
        order.append("slow_finished")

    async def _fast_turn(source: object) -> None:
        order.append("fast_started")
        # Only once the fast turn has actually started does the slow one
        # get permission to finish -- proving the fast source's start was
        # never blocked behind the slow one, with no real sleep anywhere.
        slow_may_finish.set()

    runners = assemble_source_runners(
        {
            "camera": _spec([b"\x00"], _slow_turn),
            "browser": _spec([b"\x00"], _fast_turn),
        }
    )

    await asyncio.gather(*(runner.run() for runner in runners))

    assert order[0] == "slow_started"
    assert "fast_started" in order
    assert order[-1] == "slow_finished"


def test_zero_configured_sources_raises_at_startup_naming_the_empty_key():
    with pytest.raises(ValueError, match="sources"):
        assemble_source_runners({})


async def test_one_source_runs_with_no_multi_source_coordination_path_taken():
    turns_started: list[str] = []

    async def _camera_turn(source: object) -> None:
        turns_started.append("camera")

    runners = assemble_source_runners({"camera": _spec([b"\x00"], _camera_turn)})

    assert len(runners) == 1
    await runners[0].run()

    assert turns_started == ["camera"]
