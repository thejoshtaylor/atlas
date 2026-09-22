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
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from spire_voice.audio.energy import rms_amplitude
from spire_voice.audio.ring import PrerollBuffer
from spire_voice.calibration.record import EchoCalibration
from spire_voice.config import BargeInConfig, GateConfig, WakeConfig
from spire_voice.db.repository import WakeEventRepository
from spire_voice.providers.tts_xai import SinkFormat
from spire_voice.speaker.output_trace import EmittedAudioTrace
from spire_voice.wake.base import WakeDetector
from spire_voice.wake.gate import WakeGate

logger = logging.getLogger("spire_voice.sources.runner")

# How many wake-event writes one source may have in flight before it stops
# scheduling more. Not a tuned number: a household's wake hits arrive
# seconds apart at worst and each write is a single insert, so a healthy
# store never approaches this. It exists for the unhealthy one -- a
# repository that hangs rather than raises, where every hit from an ambient
# noise source would otherwise add a task that never completes and is never
# discarded.
MAX_PENDING_WAKE_EVENT_WRITES = 64


def resolve_wake_threshold(wake_config: WakeConfig) -> float:
    """The score a `WakeHit` must reach to count, for whichever engine
    `wake_config.engine` names.

    Public as of the Phase 8 review (IN-06): `routes/wake.py` needs the
    configured threshold for the case where no source is running to report
    a live one, and a second implementation of this rule there could
    disagree with this one about the categorical engine.
    `_resolve_threshold` stays as a module alias for the existing callers,
    the same shape `session/retention.py` used when its own directory
    parser was promoted.

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


# Alias kept for every existing caller in this module.
_resolve_threshold = resolve_wake_threshold


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

    **What this class did not do, until plan 02-12 (found in code
    review):** it never read or compared against the bytes `_speak`
    actually wrote to the speaker FIFO. Every input was the energy floor,
    the sustained duration, and the guard window above -- nothing about
    *known output*. On a device whose microphone and speaker are the same
    unit with no acoustic echo cancellation, that meant the assistant's own
    voice returning through the open mic could satisfy floor-and-duration
    past the guard window and read as a real interruption.

    **What is true now (D-18, D-19):** when constructed with both a `trace`
    (`speaker/output_trace.py`'s `EmittedAudioTrace`) and a `calibration`
    (`calibration/record.py`'s `EchoCalibration`), `process_energy` gains
    one further condition, checked after the floor and before the
    sustained-duration accumulation: is the observed energy explained by
    what was actually playing, `calibration.delay_s` earlier? A camera
    whose `agc_verdict` is `"absent"` gets a fixed comparison against the
    calibrated gain; `"present"` or `"indeterminate"` gets a slowly-tracking
    estimate instead, because a static comparison drifts wrong the moment
    the camera's own gain control moves the echo path underneath it, and an
    indeterminate verdict is not evidence the path is stable (D-19).
    Neither mode's estimate is ever updated from a reading that was *not*
    explained, and `process_energy`'s own top-of-function guard means
    neither mode's estimate can ever update again once `interrupt_requested`
    is set for this turn -- an interrupting voice raising the
    observed-to-emitted ratio must never teach the estimator that the
    interruption was the assistant's own voice getting louder.

    **What is still not true, and will not become true here:** this cannot
    separate a voice speaking at the exact moment the assistant is from the
    assistant's own echo -- it answers "is this energy explained by what I
    am playing," not "is this a person," and a television talking over a
    reply interrupts it exactly as a person would (accepted risk, T-02-54).
    Closing that gap is full acoustic echo cancellation, deferred per
    CONTEXT.md's own Deferred section, worth revisiting only against
    evidence this tier is insufficient on real hardware. Constructed with
    no `trace` or no `calibration` (every call site and every test that
    predates plan 02-12, and every source whose `BargeInConfig.
    correlation_enabled` is left off), this class behaves exactly as it
    did before this plan -- `config/config.example.yaml`'s
    `barge_in.sources.camera.enabled: false` and its `correlation_enabled`
    defaulting off are this project's answer until a real calibration and a
    real camera corpus both exist.
    """

    def __init__(
        self,
        *,
        floor: float,
        min_duration_s: float,
        guard_window_s: float,
        enabled: bool,
        trace: EmittedAudioTrace | None = None,
        calibration: EchoCalibration | None = None,
        correlation_tolerance: float = 0.0,
        tracking_adaptation_rate: float = 0.0,
    ) -> None:
        self.floor = floor
        self.min_duration_s = min_duration_s
        self.guard_window_s = guard_window_s
        self.enabled = enabled
        self.trace = trace
        self.calibration = calibration
        self.correlation_tolerance = correlation_tolerance
        self.tracking_adaptation_rate = tracking_adaptation_rate
        self.playback_started_at: float | None = None
        self.interrupt_requested = False
        self.transcript_done = asyncio.Event()
        self._above_floor_since: float | None = None
        # Correlation is active only when both a trace to compare against
        # and a calibration to align/scale by exist -- absent either, this
        # monitor behaves exactly as it did before Tier 2 (D-18's own
        # "constructed with no trace or no calibration" clause above).
        self._correlation_active = trace is not None and calibration is not None
        # D-19: "absent" gets the fixed comparison; "present" and
        # "indeterminate" both get tracking -- a verdict that could not be
        # established is treated exactly as conservatively as one that
        # found gain control, never decayed into the fixed case.
        self._tracking_mode = calibration is not None and calibration.agc_verdict != "absent"
        self._estimated_gain = calibration.gain if calibration is not None else 0.0

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
        arriving in the room," not a continuation of a previous one.

        Also resets `trace` (plan 02-12), for the same reason: a fresh
        utterance's playback offsets start at zero again, never continuing
        the previous utterance's cursor. A no-op when `trace` is `None`.
        """
        self.playback_started_at = now
        self._above_floor_since = None
        if self.trace is not None:
            self.trace.reset()

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
        if self._correlation_active and self._explained_by_known_output(energy, now):
            # D-18: known output explains this reading -- treat it exactly
            # like at-or-below-the-floor energy, resetting any run in
            # progress rather than letting the assistant's own echo
            # accumulate toward an interrupt.
            self._above_floor_since = None
            return
        if self._above_floor_since is None:
            self._above_floor_since = now
        if (now - self._above_floor_since) >= self.min_duration_s:
            self.interrupt_requested = True

    def _explained_by_known_output(self, energy: float, now: float) -> bool:
        """D-18's alignment: the sound arriving at the microphone at `now`
        was emitted at playback offset `(now - playback_started_at) -
        delay_s`, never the un-shifted `now - playback_started_at` -- write
        time and playback time are different quantities (`speaker/
        output_trace.py`'s own module docstring), and only playback time is
        comparable against a microphone reading taken at a real moment.

        `level_at` returning `None` means nothing was playing at that
        offset, so nothing explains this energy -- `False`, unconditionally,
        with no estimate update. Only an *explained* reading ever updates
        the tracking estimate (D-19's own docstring paragraph above): an
        unexplained reading is exactly the shape a real interruption takes,
        and updating from one would teach the estimator to explain away the
        very thing this class exists to detect.
        """
        assert self.trace is not None and self.calibration is not None
        offset = (now - self.playback_started_at) - self.calibration.delay_s
        level = self.trace.level_at(offset)
        if level is None:
            return False
        if self._tracking_mode:
            predicted = level * self._estimated_gain
            explained = energy <= predicted + self.correlation_tolerance
            if explained and level > 0:
                ratio = energy / level
                self._estimated_gain = (
                    (1 - self.tracking_adaptation_rate) * self._estimated_gain
                    + self.tracking_adaptation_rate * ratio
                )
            return explained
        predicted = level * self.calibration.gain
        return energy <= predicted + self.correlation_tolerance


class PrerollReplayingSource:
    """Wraps one `AudioSource`, yielding a fixed list of pre-roll chunks
    before `wrapped.frames()` continues.

    A full `AudioSource` implementation: `send_audio()`, `send_event()`,
    and `source_format()` all delegate straight to `wrapped`, so `run_turn`
    receives one continuous frame iterator and needs no knowledge that
    part of it is replay. `sink_format()` (260922-cts) delegates the same
    way, conditionally -- see that method's own docstring.
    """

    def __init__(self, wrapped: Any, preroll_chunks: list[bytes]) -> None:
        self._wrapped = wrapped
        self._preroll_chunks = preroll_chunks

    @property
    def preroll_bytes(self) -> int:
        """How many of the bytes `frames()` is about to yield are replay,
        rather than audio captured after the wake hit -- a property, not a
        bare attribute, because this wrapper is the only object in the
        system that knows this count (plan 08-11). `run_turn` reads this
        off the still-unwrapped source, before `_RecordingAudioSource`
        wraps it, and hands it to `SessionRecorder.set_preroll_bytes`.
        """
        return sum(len(chunk) for chunk in self._preroll_chunks)

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

    def sink_format(self) -> "SinkFormat | None":
        """The wrapped source's own playback sink, forwarded conditionally
        the same way `session/observers.py`'s `ObserverPublishingSource`
        already does (260922-cts): `None` when `self._wrapped` -- always
        the raw camera source in this runner, but a test double in
        `tests/test_preroll_buffer.py`/`tests/test_preroll_alignment.py`
        may carry none -- has no `sink_format` of its own, rather than
        this method raising `AttributeError` the moment `_speak` calls it.
        Without this forward, a turn that replays pre-roll (the common
        case: `config.camera.preroll_ms` defaults on) would lose the
        camera's sink the instant a hit wraps it here, and the
        camera-static bug this plan fixes would still reach the speaker.
        """
        wrapped_sink_format = getattr(self._wrapped, "sink_format", None)
        return wrapped_sink_format() if wrapped_sink_format is not None else None


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
        calibration: EchoCalibration | None = None,
        wake_event_repo: WakeEventRepository | None = None,
    ) -> None:
        self._name = name
        self._source = source
        self._wake_detector = wake_detector
        self._decode_for_detector = decode_for_detector
        self._run_turn_fn = run_turn_fn
        self._clock = clock
        self._preroll = preroll
        # Plan 08-03 (D-13, D-14): `None` (every test and tracer
        # construction that predates this plan) means "record nothing" --
        # the same absent-configuration discipline the gate and the
        # barge-in policy above already carry in this constructor. Every
        # scheduled write's own task is held here so it cannot be
        # garbage-collected before it runs (a task with no strong
        # reference can be); the done-callback below discards it once it
        # finishes.
        self._wake_event_repo = wake_event_repo
        self._pending_wake_event_tasks: set[asyncio.Task] = set()
        # One line per episode of saturation, not one per wake hit -- the
        # same "the operator gets the fact once instead of a log they stop
        # reading" discipline the degraded-slot refusal already uses.
        self._pending_writes_warned = False
        # Plan 02-12 Task 3: `app.py`'s own startup refusal is what keeps
        # this from ever being a *stale or missing* calibration for a
        # source whose policy has correlation turned on -- by the time it
        # reaches here, it is either `None` (correlation not requested, or
        # not yet measured) or a calibration already proven valid and
        # current enough to trust.
        self._calibration = calibration

        # Resolved once, here, against this runner's own name -- never
        # re-resolved per hit, and never branched on by name anywhere in
        # this class (D-04). Absent configuration (the pre-02-04 tracer
        # shape) resolves to a gate that never blocks anything: threshold
        # 0.0 accepts any score, refractory 0.0 never suppresses, and an
        # empty mute list never checks media state.
        resolved_wake = wake_config.resolve(name) if wake_config is not None else WakeConfig(refractory_s=0.0)
        resolved_gate = gate_config.resolve(name) if gate_config is not None else GateConfig()
        threshold = _resolve_threshold(resolved_wake) if wake_config is not None else 0.0
        # Plan 08-03: resolved once, here, from the same `wake_config` the
        # threshold above is resolved from -- never re-read from global
        # configuration inside a write call, matching the gate's own
        # resolve-once-at-construction discipline (D-04).
        self._wake_engine_name = resolved_wake.engine
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

    @property
    def wake_threshold(self) -> float:
        """The score this runner's gate currently requires a wake hit to
        reach -- read through the gate's own named `threshold` accessor,
        never a reach into `self._gate._threshold` from here (D-15): one
        named accessor per object is what keeps the next reader from
        reaching across two modules' private attributes."""
        return self._gate.threshold

    def set_wake_threshold(self, threshold: float) -> None:
        """Move this runner's gate to a new threshold. Effective on the
        very next wake hit evaluated on this source -- no restart (D-15).
        A runner nobody calls this on behaves exactly as it did before."""
        self._gate.set_threshold(threshold)

    async def run(self) -> None:
        """Consume `source.frames()` until it ends, running one turn per
        wake hit the gate allows along the way.

        Each chunk's processing is contained (CR-03, code review): before
        this fix, an exception from `self._wake_detector.process(...)` or
        `self._decode_for_detector(...)` -- both real possibilities, since
        PyAV can raise on a malformed packet after a lossy reconnect and
        Vosk's native bindings can raise -- propagated straight out of this
        `async for` loop and ended the task permanently, with nothing
        supervising or restarting it: the assistant would silently stop
        hearing the wake word for the rest of the process's life. Every
        other long-lived loop this phase built (the camera reconnect
        supervisor, the ffmpeg egress supervisor, the retention scheduler)
        already has this discipline; this loop did not. `source.frames()`
        itself is deliberately outside the `try` -- a raise from the
        iterator itself is the underlying source's own reconnect
        supervisor's problem (`transports/camera.py`'s module docstring),
        not this loop's to retry.
        """
        async for chunk in self._source.frames():
            try:
                await self._process_chunk(chunk)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "source %r: error processing one audio chunk -- continuing to listen "
                    "for the wake word rather than ending the whole source's task",
                    self._name,
                )

    async def _process_chunk(self, chunk: bytes) -> None:
        """One chunk of `run()`'s own loop body, split out so `run()` can
        wrap it in the containment `try`/`except` above without also
        catching an exception from `self._source.frames()` itself."""
        if self._preroll is not None:
            self._preroll.push(chunk)

        detector_chunk = self._decode_for_detector(chunk)
        if not detector_chunk:
            return
        hit = self._wake_detector.process(detector_chunk)
        if hit is None:
            return

        now = self._clock()
        decision = self._gate.evaluate(hit.score, now, self._last_hit_at)
        if not decision.allowed:
            self._record_blocked_hit(hit.score, decision.reason, now)
            return

        self._last_hit_at = now
        logger.info("wake hit on source %r (score=%.3f)", self._name, hit.score)
        self._schedule_wake_event_write(score=hit.score, allowed=True, block_reason=None)

        turn_source = self._source
        if self._preroll is not None:
            preroll_chunks = self._preroll.drain()
            if preroll_chunks:
                turn_source = PrerollReplayingSource(self._source, preroll_chunks)

        # D-18: correlation is only ever wired up when this source's own
        # resolved policy asks for it *and* a calibration is actually on
        # hand -- `app.py`'s startup refusal is what guarantees the second
        # half of that whenever the first half is true (Task 3), so this
        # constructor-time check is a second, cheap confirmation, never the
        # only one. A fresh `EmittedAudioTrace` every turn, sized from this
        # source's own declared format, mirrors `PrerollBuffer`'s own
        # per-source-format construction just above.
        correlation_active = self._barge_in_config.correlation_enabled and self._calibration is not None
        trace: EmittedAudioTrace | None = None
        if correlation_active:
            source_format = self._source.source_format()
            trace = EmittedAudioTrace(encoding=source_format.encoding, sample_rate=source_format.sample_rate)
        monitor = BargeInMonitor(
            floor=self._barge_in_config.energy_floor,
            min_duration_s=self._barge_in_config.min_duration_ms / 1000.0,
            guard_window_s=self._barge_in_config.post_playback_guard_ms / 1000.0,
            enabled=self._barge_in_config.enabled,
            trace=trace,
            calibration=self._calibration if correlation_active else None,
            correlation_tolerance=self._barge_in_config.correlation_tolerance,
            tracking_adaptation_rate=self._barge_in_config.tracking_adaptation_rate,
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

        Plan 08-03: this used to say plan 02-05's session store would
        consume these once it existed. That never happened, and could
        not have: `SessionRecorder` is only ever constructed after the
        gate *allows* a hit, so a blocked hit has no session directory to
        live in. Phase 8 built a separate, persistent wake-event record
        instead (`db.repository.WakeEventRepository`), deliberately
        distinct from the per-turn session store -- see
        `_schedule_wake_event_write`, called below beside this log line,
        which stays exactly as it was and independently useful.
        """
        logger.info(
            "wake hit blocked",
            extra={"source_name": self._name, "score": score, "block_reason": reason, "hit_at": at},
        )
        self._schedule_wake_event_write(score=score, allowed=False, block_reason=reason)

    def _schedule_wake_event_write(
        self, *, score: float, allowed: bool, block_reason: str | None
    ) -> None:
        """Build one `WakeEvent` and schedule its write -- never awaited.

        The allowed branch of `_process_chunk` runs immediately before a
        turn starts, and this project's latency budget is load-bearing (a
        reply under one second, per the project constraints) -- awaiting
        a database write on that path would cost the turn real time for
        no benefit a house could feel. `asyncio.create_task` starts the
        write independently; the task is held in
        `self._pending_wake_event_tasks` until its own done-callback
        discards it, because a task with no strong reference can be
        garbage-collected before it ever runs. A no-op when this runner
        was built with no repository at all (D-13, D-14's absent-
        configuration discipline).

        The set is bounded. A repository that *hangs* rather than raises
        -- Postgres reachable but wedged, or a pool exhausted -- leaves
        every scheduled write in flight forever, and a talking television
        is a wake source that never stops. Past `MAX_PENDING_WAKE_EVENT_
        WRITES` this skips the write and says so once, which is the
        bounded version of what D-14 asks for: a wake hit is never
        dropped *silently*.
        """
        if self._wake_event_repo is None:
            return
        if len(self._pending_wake_event_tasks) >= MAX_PENDING_WAKE_EVENT_WRITES:
            if not self._pending_writes_warned:
                self._pending_writes_warned = True
                logger.warning(
                    "source %r: %d wake-event writes are already in flight and none are "
                    "completing; skipping further writes until they drain. The store is "
                    "reachable but not answering.",
                    self._name,
                    len(self._pending_wake_event_tasks),
                )
            return
        self._pending_writes_warned = False
        task = asyncio.create_task(
            self._write_wake_event(score=score, allowed=allowed, block_reason=block_reason)
        )
        self._pending_wake_event_tasks.add(task)
        task.add_done_callback(self._pending_wake_event_tasks.discard)

    async def drain_pending_wake_events(self, timeout: float = 2.0) -> None:
        """Let every in-flight wake-event write finish before the engine
        under it is disposed.

        Nothing drained this set. `lifespan`'s teardown cancels and gathers
        the source-runner *tasks* only, so a write scheduled by the last
        wake before a restart was left dangling and the database engine was
        disposed underneath it -- the record D-14 says must not disappear
        silently, disappearing with a swallowed log line at best and a
        "Task was destroyed but it is pending" warning at worst.

        Bounded, not unbounded: a store that is down must not hold up a
        shutdown either. Anything still running past `timeout` is
        cancelled, which is the same outcome as before for that write and
        a clean one for every write that would have made it.
        """
        pending = set(self._pending_wake_event_tasks)
        if not pending:
            return
        _done, still_running = await asyncio.wait(pending, timeout=timeout)
        for task in still_running:
            task.cancel()
        if still_running:
            # Awaited, not merely cancelled: `cancel()` only *requests*,
            # and this method's whole purpose is that nothing is still
            # touching the engine when it returns.
            await asyncio.gather(*still_running, return_exceptions=True)
            logger.warning(
                "source %r: %d wake-event write(s) did not finish within %.1fs of shutdown "
                "and were cancelled",
                self._name,
                len(still_running),
                timeout,
            )

    async def _write_wake_event(
        self, *, score: float, allowed: bool, block_reason: str | None
    ) -> None:
        """The awaited body `_schedule_wake_event_write` schedules, never
        awaits itself. Wrapped so a repository that raises is logged and
        swallowed here -- never escaping as an unretrieved-exception
        warning, and never stopping this source's own listening loop: a
        store that is down must not stop a house listening, the same
        containment discipline `run()`'s own try/except around chunk
        processing already applies (module docstring)."""
        assert self._wake_event_repo is not None
        try:
            await self._wake_event_repo.record_wake_event(
                source=self._name,
                engine=self._wake_engine_name,
                score=score,
                allowed=allowed,
                block_reason=block_reason,
                recorded_at=datetime.now(timezone.utc),
            )
        except Exception:
            logger.exception(
                "source %r: failed to record a wake event -- continuing to listen; a "
                "store that is down must never stop a house listening",
                self._name,
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
    calibration: EchoCalibration | None = None
    wake_event_repo: WakeEventRepository | None = None


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
            calibration=spec.calibration,
            wake_event_repo=spec.wake_event_repo,
        )
        for name, spec in specs.items()
    ]
