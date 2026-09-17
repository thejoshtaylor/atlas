"""`SourceRunner`: the only thing that turns a wake hit into a `run_turn` call.

`run_turn` itself (`turn/controller.py`) stays exactly as ignorant of wake
words as it was before this plan -- this module is its caller, the same
way the WebSocket and WebRTC routes in `app.py` already are for the two
browser sources.

**VOICE-03 is enforced here, and nowhere else:** `run()` consumes
`source.frames()` in a single continuous loop, forwarding each raw chunk
through the source's own detector-decode step and into the wake detector.
Before a hit, nothing this runner holds is passed to `run_turn_fn` -- the
transcription provider's `stream()` is not called, and its socket is not
opened, until `wake_detector.process()` returns a `WakeHit` the gate also
allows. Only then does this loop await one full turn; while that await is
pending, this loop makes no other call to `source.frames()`, so the turn's
own internal read of `source.frames()` (via `run_turn` -> `stt.stream()`)
is the sole consumer of the underlying queue for the duration of the turn.

**The pre-roll (VOICE-06)** is optional and additive: a `SourceRunner`
built with no `preroll` behaves exactly as it did before this plan. When a
`PrerollBuffer` is supplied, every raw chunk is pushed into it before and
independently of the detector, and a hit drains it into a
`PrerollReplayingSource` wrapper the turn runs against instead of the bare
source -- `run_turn` receives one continuous frame iterator and needs no
knowledge that part of it is replay.

**The gate (VOICE-03/D-03/D-04)** is resolved once, at construction, from
the source's own name against the global `WakeConfig`/`GateConfig` --
never branched on in code. Refractory state (`_last_hit_at`) lives on the
instance, so it is per source by construction, not by discipline.

**Multi-source (SRC-03):** nothing about wake state, refractory timing, or
gate policy lives at module scope or on a shared object -- every
`SourceRunner` instance owns all of it. Two runners built from the same
configuration can fire in the same event-loop tick and each start its own
turn: they are two independent coroutines with nothing between them to
serialize on. `assemble_source_runners()` is where the list is built, and
it treats one source and many sources identically -- a single-camera
deployment is the degenerate case of the list, not a second code path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from spire_voice.audio.ring import PrerollBuffer
from spire_voice.config import GateConfig, WakeConfig
from spire_voice.wake.base import WakeDetector
from spire_voice.wake.gate import WakeGate

logger = logging.getLogger("spire_voice.sources.runner")


def _resolve_threshold(wake_config: WakeConfig) -> float:
    """The score a `WakeHit` must reach to count, for whichever engine
    `wake_config.engine` names.

    Only `OpenWakeWordConfig` carries a graded `threshold` today --
    `VoskWakeDetector` reports a fixed `score=1.0` for every match, because
    a grammar-constrained decode is a categorical hit, not a graded one
    (`wake/vosk_engine.py`'s own comment). `1.0` here is that same
    categorical threshold: any Vosk hit already reports exactly `1.0`, so
    it always clears it.
    """
    if wake_config.engine == "openwakeword":
        return wake_config.openwakeword.threshold
    return 1.0


class PrerollReplayingSource:
    """Wraps one `AudioSource`, yielding a fixed list of pre-roll chunks
    before `wrapped.frames()` continues.

    A full `AudioSource` implementation: `send_audio()`, `send_event()`,
    and `source_format()` all delegate straight to `wrapped`, so `run_turn`
    receives one continuous frame iterator and needs no knowledge that
    part of it is replay.
    """

    def __init__(self, wrapped: Any, preroll_chunks: list[bytes]) -> None:
        self._wrapped = wrapped
        self._preroll_chunks = preroll_chunks

    async def frames(self) -> AsyncIterator[bytes]:
        for chunk in self._preroll_chunks:
            yield chunk
        async for chunk in self._wrapped.frames():
            yield chunk

    async def send_audio(self, chunk: bytes) -> None:
        await self._wrapped.send_audio(chunk)

    async def send_event(self, event: dict[str, Any]) -> None:
        await self._wrapped.send_event(event)

    def source_format(self) -> Any:
        return self._wrapped.source_format()


class SourceRunner:
    """Ties one named source to one wake detector and one `run_turn` caller."""

    def __init__(
        self,
        name: str,
        source: Any,
        wake_detector: WakeDetector,
        decode_for_detector: Callable[[bytes], bytes],
        run_turn_fn: Callable[[Any], Awaitable[None]],
        *,
        wake_config: WakeConfig | None = None,
        gate_config: GateConfig | None = None,
        is_media_playing: Callable[[tuple[str, ...]], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        preroll: PrerollBuffer | None = None,
    ) -> None:
        self._name = name
        self._source = source
        self._wake_detector = wake_detector
        self._decode_for_detector = decode_for_detector
        self._run_turn_fn = run_turn_fn
        self._clock = clock
        self._preroll = preroll

        # Resolved once, here, against this runner's own name -- never
        # re-resolved per hit, and never branched on by name anywhere in
        # this class (D-04). Absent configuration (the pre-02-04 tracer
        # shape) resolves to a gate that never blocks anything: threshold
        # 0.0 accepts any score, refractory 0.0 never suppresses, and an
        # empty mute list never checks media state.
        resolved_wake = wake_config.resolve(name) if wake_config is not None else WakeConfig(refractory_s=0.0)
        resolved_gate = gate_config.resolve(name) if gate_config is not None else GateConfig()
        threshold = _resolve_threshold(resolved_wake) if wake_config is not None else 0.0
        self._gate = WakeGate(
            threshold=threshold,
            refractory_s=resolved_wake.refractory_s,
            mute_when_playing=resolved_gate.mute_when_playing,
            is_media_playing=is_media_playing,
        )
        self._last_hit_at: float | None = None

    async def run(self) -> None:
        """Consume `source.frames()` until it ends, running one turn per
        wake hit the gate allows along the way."""
        async for chunk in self._source.frames():
            if self._preroll is not None:
                self._preroll.push(chunk)

            detector_chunk = self._decode_for_detector(chunk)
            if not detector_chunk:
                continue
            hit = self._wake_detector.process(detector_chunk)
            if hit is None:
                continue

            now = self._clock()
            decision = self._gate.evaluate(hit.score, now, self._last_hit_at)
            if not decision.allowed:
                self._record_blocked_hit(hit.score, decision.reason, now)
                continue

            self._last_hit_at = now
            logger.info("wake hit on source %r (score=%.3f)", self._name, hit.score)

            turn_source = self._source
            if self._preroll is not None:
                preroll_chunks = self._preroll.drain()
                if preroll_chunks:
                    turn_source = PrerollReplayingSource(self._source, preroll_chunks)

            await self._run_turn_fn(turn_source)

    def _record_blocked_hit(self, score: float, reason: str | None, at: float) -> None:
        """A wake hit the gate blocks is recorded, never dropped silently
        (D-03). No transcript text and no audio ever reach this record --
        only the source, the score, the reason, and the timestamp, in the
        same structured-log shape `timing.py` emits its own line.

        Plan 02-05's session store consumes these once it exists; until
        then this log line is already more than the silence CONTEXT.md
        rejects.
        """
        logger.info(
            "wake hit blocked",
            extra={"source_name": self._name, "score": score, "block_reason": reason, "hit_at": at},
        )


@dataclass(frozen=True)
class SourceRunnerSpec:
    """Everything `assemble_source_runners()` needs to build one
    `SourceRunner`, keyed by source name in the mapping it accepts."""

    source: Any
    wake_detector: WakeDetector
    decode_for_detector: Callable[[bytes], bytes]
    run_turn_fn: Callable[[Any], Awaitable[None]]
    wake_config: WakeConfig | None = None
    gate_config: GateConfig | None = None
    is_media_playing: Callable[[tuple[str, ...]], bool] | None = None
    clock: Callable[[], float] = field(default=time.monotonic)
    preroll: PrerollBuffer | None = None


def assemble_source_runners(specs: Mapping[str, SourceRunnerSpec]) -> list[SourceRunner]:
    """Build one `SourceRunner` per entry in `specs`, in no particular
    order and with no branch on how many entries there are.

    Zero configured sources is a startup error naming the empty `sources`
    key, not a silently idle application -- an application that starts,
    logs nothing unusual, and never responds to anything is the worst of
    the available failures. One source runs through this exact same loop
    as two would: the one-source case is the degenerate case of the list,
    never a second code path.
    """
    if not specs:
        raise ValueError(
            "no audio sources configured under 'sources' -- at least one source must be "
            "configured, or the application would start and never respond to anything"
        )
    return [
        SourceRunner(
            name,
            spec.source,
            spec.wake_detector,
            spec.decode_for_detector,
            spec.run_turn_fn,
            wake_config=spec.wake_config,
            gate_config=spec.gate_config,
            is_media_playing=spec.is_media_playing,
            clock=spec.clock,
            preroll=spec.preroll,
        )
        for name, spec in specs.items()
    ]
