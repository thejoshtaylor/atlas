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

**Barge-in (VOICE-07, D-09/D-10/D-11/D-12)** adds a second consumer of the
same per-chunk tap the wake detector and the pre-roll buffer already read --
`BargeInMonitor` holds the boundary semantics (the energy floor, the
sustained-duration accumulation, the post-playback guard window) and is
covered directly, with no asyncio at all, by `tests/test_barge_in.py`.

Wiring that pure detector to *live* frames while a turn is in flight runs
into `CameraAudioSource`'s own shape (`transports/camera.py`): `frames()` is
backed by one `asyncio.Queue`, and only one reader may ever drain it at a
time -- two concurrent calls to `.frames()` would split the packets between
them, corrupting both. Before this plan, that was never a problem: `run()`'s
own loop stood still (awaiting `run_turn_fn` directly) for the whole turn,
so the turn's own `stt.stream(source.frames())` was always the *sole*
reader for as long as it ran. Barge-in needs a second reader, but only
*after* the first one retires -- and `stt_xai.py`'s own `stream()` already
cancels its frame-reading `sender()` task the instant a final transcript
arrives (its own `finally` block), well before `_speak` ever writes a byte
to the speaker. That retirement is the exact moment this module's own
listener may safely start reading, and `BargeInMonitor.transcript_done` is
the signal `run_turn` (`turn/controller.py`) sets, right after draining the
final transcript, to say so -- no window exists where zero or two readers
are ever active at once.

`run()` itself keeps its former shape unchanged in one respect only
(effectively, one turn still runs to completion before `run()`'s own loop
resumes): `_run_one_turn` replaces the old bare `await
self._run_turn_fn(turn_source)` with `await asyncio.gather` of that same
call *and* the barge-in listener, so the listener's lifetime is scoped to
the turn's -- cancelled the instant the turn task finishes, so it never
starves the next wake evaluation once `run()`'s loop resumes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from spire_voice.audio.energy import rms_amplitude
from spire_voice.audio.ring import PrerollBuffer
from spire_voice.config import BargeInConfig, GateConfig, WakeConfig
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


class BargeInMonitor:
    """One turn's barge-in state: the boundary semantics (VOICE-07, D-10)
    and the two facts the turn and the runner exchange to act on them.

    Constructed fresh per turn (never reused across turns) from the
    per-source `BargeInConfig` resolved at `SourceRunner.__init__` time
    (D-12). Two call sites drive it, on two sides of the turn/runner
    boundary, and neither reaches into the other's task (module docstring):

    - `turn/controller.py`'s `_speak` calls `mark_playback_started()` once
      per utterance (filler and answer both count as "playing"), and reads
      `interrupt_requested` between chunks -- the runner sets it, the turn
      reads it, exactly as `sources/runner.py`'s own module docstring says.
    - `run_turn` calls `mark_transcript_done()` the instant the final
      transcript is drained, before anything else touches `source`, so this
      module's own listener knows precisely when it may safely become the
      sole reader of `source.frames()` (see the module docstring's
      single-consumer-queue explanation).

    `process_energy` is the pure boundary logic, and only that: it takes
    `now` as a parameter rather than reading a clock itself, so
    `tests/test_barge_in.py` drives every boundary case -- floor, sustained
    duration, and the post-playback guard window -- without any asyncio at
    all. Energy exactly at the floor never counts; energy above it must
    hold for the *entire* configured minimum duration; and no energy inside
    the guard window ever counts, however loud, because the guard window
    exists specifically to absorb whatever the room's own acoustics do to
    the assistant's own speech in the first moments after it starts
    (FIFO buffering, go2rtc's own latency, and acoustic flight time --
    none of which this process can derive from first principles, per
    RESEARCH.md's own barge-in section; the constants are named, provisional
    configuration, not measured ones, until real camera sessions exist to
    tune them from).

    Deliberately the *opposite* boundary convention from `WakeGate`'s own
    threshold (`wake/gate.py`: a score exactly at the threshold **is** a
    hit): a missed wake word costs an inconvenience, while an interrupt that
    fires on the room's own noise floor costs the assistant's ability to
    finish a sentence. The two thresholds are not the same decision and
    must not be harmonized to look alike.

    **What this class does not do, stated plainly (found in code review):**
    it never reads or compares against the bytes `_speak` actually wrote to
    the speaker FIFO. Every input here is the energy floor, the sustained
    duration, and the guard window above -- nothing about *known output*.
    On a device whose microphone and speaker are the same unit with no
    acoustic echo cancellation, that means the assistant's own voice
    returning through the open mic can satisfy floor-and-duration past the
    guard window and read as a real interruption. `BargeInConfig`'s own
    docstring (`config.py`) and `config.example.yaml`'s
    `barge_in.sources.camera.enabled: false` are this project's answer
    until a real camera corpus proves the guard window sufficient.
    """

    def __init__(self, *, floor: float, min_duration_s: float, guard_window_s: float, enabled: bool) -> None:
        self.floor = floor
        self.min_duration_s = min_duration_s
        self.guard_window_s = guard_window_s
        self.enabled = enabled
        self.playback_started_at: float | None = None
        self.interrupt_requested = False
        self.transcript_done = asyncio.Event()
        self._above_floor_since: float | None = None

    def mark_transcript_done(self) -> None:
        """`run_turn` calls this once, right after draining the final
        transcript -- the signal this module's listener waits on before it
        may safely start reading `source.frames()` itself (module
        docstring)."""
        self.transcript_done.set()

    def mark_playback_started(self, now: float) -> None:
        """`_speak` calls this on the first chunk of every utterance it
        writes -- filler and answer alike restart the guard window, because
        each is a fresh moment of "the assistant's own speech just started
        arriving in the room," not a continuation of a previous one."""
        self.playback_started_at = now
        self._above_floor_since = None

    def process_energy(self, energy: float, now: float) -> None:
        """Feed one energy reading at time `now`; sets `interrupt_requested`
        the instant the sustained-duration boundary is crossed, and never
        clears it again (D-11 -- once interrupted, stays interrupted for
        this turn).

        No-ops before `mark_playback_started` has ever been called: nothing
        is playing yet, so there is nothing to interrupt, however loud the
        room already is.
        """
        if not self.enabled or self.interrupt_requested or self.playback_started_at is None:
            return
        if (now - self.playback_started_at) < self.guard_window_s:
            # Inside the guard window: never counts, however loud (module
            # docstring) -- and never lets a run that started before the
            # window opened carry an accumulated duration past it.
            self._above_floor_since = None
            return
        if energy <= self.floor:
            # At or below the floor resets any run in progress -- a single
            # transient (a door closing) must not interrupt a reply (D-10).
            self._above_floor_since = None
            return
        if self._above_floor_since is None:
            self._above_floor_since = now
        if (now - self._above_floor_since) >= self.min_duration_s:
            self.interrupt_requested = True


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
        barge_in_config: BargeInConfig | None = None,
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

        # Same absent-configuration discipline as the gate above: a
        # `SourceRunner` built with no `barge_in_config` at all (every test
        # and tracer construction that predates this plan) resolves to
        # "disabled" rather than D-12's "on by default" -- that default is
        # `BargeInConfig()`'s own (`enabled=True`), reached once an operator
        # actually configures `barge_in:` (or accepts its own defaults) and
        # `app.py` passes the resolved object through. Absent entirely, this
        # runner must behave exactly as it did before this plan.
        self._barge_in_config = (
            barge_in_config.resolve(name) if barge_in_config is not None else BargeInConfig(enabled=False)
        )

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

            monitor = BargeInMonitor(
                floor=self._barge_in_config.energy_floor,
                min_duration_s=self._barge_in_config.min_duration_ms / 1000.0,
                guard_window_s=self._barge_in_config.post_playback_guard_ms / 1000.0,
                enabled=self._barge_in_config.enabled,
            )
            # Duck-typed, the same way `turn/controller.py`'s own
            # `_emit_event` already reaches for `send_event` -- `run_turn`
            # and `_speak` read this back with `getattr(source, "barge_in",
            # None)` rather than a positional `run_turn_fn` never had.
            turn_source.barge_in = monitor

            await self._run_one_turn(turn_source, monitor)

    async def _run_one_turn(self, turn_source: Any, monitor: "BargeInMonitor") -> None:
        """Run one turn, plus its own barge-in listener -- see the module
        docstring for why the listener cannot simply run inside `run()`'s
        own loop, and why it is safe to start once `monitor.transcript_done`
        fires.

        Still, in effect, one turn at a time per source: this awaits the
        turn task directly (same as the bare `await self._run_turn_fn(...)`
        this replaces), so `run()`'s own loop does not resume -- and cannot
        evaluate the next wake hit -- until the turn this monitor belongs to
        has fully finished. The listener's own lifetime is scoped inside
        that same window and is always cancelled before this returns, so it
        never starves the next hit's frame consumption.
        """
        turn_task = asyncio.ensure_future(self._run_turn_fn(turn_source))
        listener_task = asyncio.ensure_future(self._watch_barge_in(monitor))
        try:
            await turn_task
        finally:
            listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener_task

    async def _watch_barge_in(self, monitor: "BargeInMonitor") -> None:
        """Become the sole reader of `self._source.frames()` once it is
        safe to (module docstring), feeding every chunk's energy into
        `monitor` until cancelled.

        A no-op for a disabled policy (D-12's per-source override): this
        never even waits on `transcript_done`, so a source with barge-in
        turned off never opens a second reader of its own queue at all.
        """
        if not monitor.enabled:
            return
        await monitor.transcript_done.wait()
        async for chunk in self._source.frames():
            detector_chunk = self._decode_for_detector(chunk)
            if not detector_chunk:
                continue
            monitor.process_energy(rms_amplitude(detector_chunk), self._clock())

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
    barge_in_config: BargeInConfig | None = None
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
            barge_in_config=spec.barge_in_config,
            is_media_playing=spec.is_media_playing,
            clock=spec.clock,
            preroll=spec.preroll,
        )
        for name, spec in specs.items()
    ]
