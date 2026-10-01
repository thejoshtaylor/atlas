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

**Parallel turns (Phase 12, D-01):** a runner built with a `ParallelTurns` spec
takes the other branch. `run()` stays the only caller of `source.frames()`. It
pushes every chunk into the `TurnGroup` fan-out before detection, and an allowed
hit starts a `TurnRun` task instead of awaiting a turn, so wake detection keeps
running during every turn. A runner with no spec (the camera, the browser
listener) runs the serial path above unchanged.

**Detector work runs off the loop (D2, quick task 260924-4is):** every
call this module makes into `wake_detector.process()` or
`decode_for_detector()` -- for the wake hit and for the barge-in
listener alike -- runs on `self._detector_executor`, one worker thread
per runner. Before this fix, both ran directly on the loop thread, and
`asyncio.Queue.get()` never yields when the queue already holds a chunk,
so a backlog drained in one uninterrupted loop step with no checkpoint
anything else in the process could run at. One worker, not the shared
default pool `asyncio.to_thread` would use, because Vosk recognizers and
PyAV codec contexts are stateful and not thread-safe, and a cancelled
await does not stop a call already running in a thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from atlas.audio.energy import rms_amplitude
from atlas.audio.ring import PrerollBuffer
from atlas.calibration.record import EchoCalibration
from atlas.config import BargeInConfig, GateConfig, WakeConfig
from atlas.db.repository import WakeEventRepository
from atlas.providers.tts_xai import SinkFormat
from atlas.sources.frame_fanout import TurnFrameSource
from atlas.sources.reply_timing import ReplyCursor, in_any_window, says_wake_word, wake_word_windows
from atlas.sources.turn_group import ParallelTurns, TurnGroup, TurnHooks
from atlas.speaker.output_trace import EmittedAudioTrace
from atlas.timers.ring_stop import RingStopWindow
from atlas.transports.edge import LED_IDLE, LED_LISTENING
from atlas.turn.answer_window import WAKE_EVIDENCE_WINDOW_S, interrupt_resume_request
from atlas.turn.follow_up import MAX_CHAINED_FOLLOW_UPS, AnswerScope, FollowUpChannel, FollowUpRequest
from atlas.wake.base import WakeDetector, WakeHit
from atlas.wake.gate import WakeGate

logger = logging.getLogger("atlas.sources.runner")

# How many wake-event writes one source may have in flight before it stops
# scheduling more. Not a tuned number: a household's wake hits arrive
# seconds apart at worst and each write is a single insert, so a healthy
# store never approaches this. It exists for the unhealthy one -- a
# repository that hangs rather than raises, where every hit from an ambient
# noise source would otherwise add a task that never completes and is never
# discarded.
MAX_PENDING_WAKE_EVENT_WRITES = 64

# The block reason for a playback wake hit that lines up with the reply's own
# wake word (Phase 13 D-02). `wake_events.block_reason` is free text, so a new
# reason needs no migration.
REPLY_WAKE_WORD_BLOCK = "reply_wake_word"

# WR-02: the block reason for a wake hit in an answer window that lands too
# long after the window's first audio to belong to the transcribed utterance.
ANSWER_WINDOW_LATE_HIT_BLOCK = "answer_window_late_hit"

# Phase 13 (RESEARCH Pitfall 9): at most this many turns may be started by an
# interrupt between two idle wake hits, counted across interrupt turns and the
# answer windows between them (WR-06). A television that says the wake word
# again and again cannot chain turns forever. The wake gate refractory still
# applies to each hit. After the cap the runner goes back to listening for the
# wake word.
MAX_CHAINED_INTERRUPT_TURNS = 3


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
        wake_interrupts: bool = False,
        holds_for_playback: bool = False,
        on_interrupt: "Callable[[], None] | None" = None,
        wake_phrase: str = "",
        reply_margin_s: float = 1.0,
        echo_delay_s: float = 0.0,
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
        # Phase 13: the wake word heard during playback interrupts (D-01), the
        # turn stays open until the reply has played (RESEARCH Finding 1), and
        # `playback` is the queue-end cursor `_speak` feeds (D-02).
        self.wake_interrupts = wake_interrupts
        self.holds_for_playback = holds_for_playback
        # D-02: where the reply's own wake word falls in time, so a hit that
        # lines up with it is not taken for the operator.
        self.wake_phrase = wake_phrase
        self.reply_margin_s = reply_margin_s
        self.echo_delay_s = echo_delay_s
        self.playback = ReplyCursor()
        self.interrupted = asyncio.Event()
        self.interrupt_kind: str | None = None
        # Phase 13 (D-15, D-16): the frames the listener read from the interrupt
        # on. The listener appends each one before any await, so a cancelled
        # await never drops a chunk it already holds (RESEARCH Pitfall 2). The
        # next turn replays them. `after_interrupt` is True on the monitor of
        # the turn that an interrupt started.
        self.handover: list[bytes] = []
        self.after_interrupt = False
        # D-16: what a VAD interrupt's no-wake-word turn may reach. `run_turn`
        # notes the turn's command and the tools it dispatched, and a later
        # note replaces an earlier one.
        self.resume_transcript: str | None = None
        self.resume_scope: AnswerScope | None = None
        # D-A: on the turn a wake interrupt started, the window request it runs
        # under if its transcript fails verification.
        self.unverified_request: FollowUpRequest | None = None
        # D-B: True once an utterance of this turn's reply says the wake word.
        # The listener then needs the full phrase.
        self.reply_says_wake_word = False
        self._on_interrupt = on_interrupt
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

    @property
    def active(self) -> bool:
        """True when any interrupt path is on: the VAD or energy path
        (`enabled`) or the wake word path (`wake_interrupts`)."""
        return self.enabled or self.wake_interrupts

    def _latch_interrupt(self, kind: str) -> None:
        """Set the interrupt once, in this order: the flag, the `on_interrupt`
        hook, then the `interrupted` event. The hook runs before the event, so
        the stop task it schedules is queued ahead of the held answer's
        wake-up. A hook error is logged and never reaches the caller."""
        if self.interrupt_requested:
            return
        self.interrupt_requested = True
        self.interrupt_kind = kind
        if self._on_interrupt is not None:
            try:
                self._on_interrupt()
            except Exception:
                logger.warning("barge-in interrupt hook failed", exc_info=True)
        self.interrupted.set()

    def note_resume_context(self, transcript: str, scope: AnswerScope) -> None:
        """`run_turn` calls this with the turn's command text and the scope
        of the tools it dispatched. A second call replaces the first, so the
        last note is the one a VAD interrupt resumes from."""
        self.resume_transcript = transcript
        self.resume_scope = scope

    def note_reply_text(self, text: str) -> None:
        """`_speak` calls this with each utterance's text before it writes the
        first chunk, so the flag is set before the word can play (D-B)."""
        if says_wake_word(text, self.wake_phrase):
            self.reply_says_wake_word = True

    def reply_wake_word_at(self, now: float) -> bool:
        """True when `now` falls inside the estimated play time of the wake
        word in a reply already written to the speaker (D-02). A hit during
        the write burst comes before its utterance is recorded, so it is not
        suppressed. False with no wake phrase."""
        return in_any_window(
            now,
            wake_word_windows(
                self.playback.utterances,
                wake_phrase=self.wake_phrase,
                margin_s=self.reply_margin_s,
                echo_delay_s=self.echo_delay_s,
            ),
        )

    def process_wake_hit(self, now: float) -> bool:
        """Latch an interrupt for a wake hit the gate allowed. Returns False
        when the wake path is off or the monitor is already interrupted.

        No guard window applies, and playback need not have started. A hit
        while the brain still thinks silences the reply at its first chunk,
        through the check in `_speak_direct`'s write loop."""
        if not self.wake_interrupts or self.interrupt_requested:
            return False
        self._latch_interrupt("wake")
        return True

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
            self._latch_interrupt("energy")

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

    def process_speech_start(self, now: float) -> None:
        """Plan 10-10 (D-16): the interrupt path for a source whose own
        chip cancels its echo, proven by the D-18 spike (`aec: proven`).
        A `vad.start` arriving during playback answers "a person started
        speaking" directly -- there is no floor, no sustained-duration
        accumulation, and no known-output correlation to run first,
        because the source's own on-chip AEC is what already separates a
        real voice from the assistant's own echo before this server ever
        sees a byte. `process_energy` answers "the room got loud"; this
        answers a different, stronger question, and only a source with
        that proof wired (`SourceRunner._watch_barge_in`'s own
        `speech_signals` branch) ever calls this method at all.

        Same three no-op boundaries as `process_energy`, checked the same
        way: disabled, no playback yet, and the one-way latch once
        already interrupted (D-11) -- an interrupt this method already
        set is never revisited. Unlike `process_energy`, there is a
        fourth boundary this method alone needs: the guard window still
        applies (D-16 does not exempt this path from it) -- a `vad.start`
        arriving before the guard window elapses is exactly the "the
        assistant's own first syllable, arriving back through the room"
        shape the window exists to absorb, whichever detection path
        raised it.
        """
        if not self.enabled or self.interrupt_requested or self.playback_started_at is None:
            return
        if (now - self.playback_started_at) < self.guard_window_s:
            return
        self._latch_interrupt("vad")


class PrerollReplayingSource:
    """Wraps one `AudioSource`, yielding a fixed list of pre-roll chunks
    before `wrapped.frames()` continues.

    A full `AudioSource` implementation: `send_audio()`, `send_event()`,
    and `source_format()` all delegate straight to `wrapped`, so `run_turn`
    receives one continuous frame iterator and needs no knowledge that
    part of it is replay. `sink_format()` (260922-cts) delegates the same
    way, conditionally -- see that method's own docstring.

    260922-woc: the pre-roll replay is one-shot across the *lifetime of
    this wrapper instance*, not per `frames()` call. `run_turn` drains
    `frames()` a second time when a final transcript turns out to be only
    the wake phrase (`turn/wake_echo.py::is_wake_only`) -- the command the
    operator actually spoke is still arriving live. Without this, that
    second drain would replay the wake word itself before ever reaching
    the command, because nothing about `frames()`'s pre-fix body remembered
    a previous call. `_preroll_yielded` is set the instant the first call
    begins iterating the pre-roll list (before the first `yield`, not
    after the last one), so even a first drain that ends early -- closed by
    `is_wake_only`'s own second-drain path, or by any other caller that
    does not exhaust the generator -- still marks the pre-roll spent for
    every later call. `preroll_bytes` is unaffected: it is computed from
    `self._preroll_chunks` directly, which this flag never touches, so a
    session recorder built after a second drain still reports the same
    total pre-roll size a first-drain-only turn always has.
    """

    def __init__(self, wrapped: Any, preroll_chunks: list[bytes]) -> None:
        self._wrapped = wrapped
        self._preroll_chunks = preroll_chunks
        self._preroll_yielded = False

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
        if not self._preroll_yielded:
            self._preroll_yielded = True
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

    @property
    def speech_signals(self) -> Any:
        """Forwarded the same conditional way `sink_format` above already
        is (10-05-PLAN.md, D-09 through D-13): `None` when `self._wrapped`
        has none of its own -- every source but the edge source -- so
        `turn/controller.py::run_turn`'s `getattr(source, "speech_signals",
        None)` reads through this wrapper to the edge source's real
        `SpeechSignals` rather than always seeing an absent attribute.
        """
        return getattr(self._wrapped, "speech_signals", None)


def follow_up_window_opens_at(
    requested: FollowUpRequest, *, echo_tail_s: float, calibration: EchoCalibration | None, now: float
) -> float:
    """When a follow-up window opens: the readback's estimated playback end (or
    `now`, when there is none) plus the echo tail. The tail is the larger of the
    configured one and a real calibration's measured delay plus a fixed 0.2 s
    margin (D-09), so a measured acoustic fact is never overridden by a smaller
    constant."""
    tail_s = echo_tail_s
    if calibration is not None:
        tail_s = max(tail_s, calibration.delay_s + 0.2)
    return (requested.playback_ends_at if requested.playback_ends_at is not None else now) + tail_s


class FollowUpSource:
    """Wraps one `AudioSource`, dropping every frame chunk read before
    `opens_at` -- the one mechanism that keeps a follow-up turn's own STT
    from ever hearing the readback's own audio return through the open
    microphone (T-09-34, D-09, plan 09-06).

    `opens_at` and `clock` share the same domain as `SourceRunner`'s own
    `self._clock` (real time by default, injectable for tests) -- never
    `time.monotonic()` called directly, so a scripted clock drives this
    wrapper exactly as it drives every other timing decision in this
    module. Every other method delegates straight through to `wrapped`,
    the same "no tap beyond the one stated purpose" discipline
    `PrerollReplayingSource` above already follows for its own single tap.
    """

    def __init__(
        self,
        wrapped: Any,
        opens_at: float,
        clock: Callable[[], float],
        on_chunk: "Callable[[bytes], Awaitable[None]] | None" = None,
    ) -> None:
        self._wrapped = wrapped
        self._opens_at = opens_at
        self._clock = clock
        self._on_chunk = on_chunk

    async def frames(self) -> AsyncIterator[bytes]:
        """`on_chunk`, when set, is awaited for each chunk at or after
        `opens_at`, before it is yielded. It never sees an earlier chunk."""
        async for chunk in self._wrapped.frames():
            if self._clock() < self._opens_at:
                continue
            if self._on_chunk is not None:
                await self._on_chunk(chunk)
            yield chunk

    async def send_audio(self, chunk: bytes) -> None:
        await self._wrapped.send_audio(chunk)

    async def send_event(self, event: dict[str, Any]) -> None:
        wrapped_send_event = getattr(self._wrapped, "send_event", None)
        if wrapped_send_event is not None:
            await wrapped_send_event(event)

    def source_format(self) -> Any:
        return self._wrapped.source_format()

    def sink_format(self) -> "SinkFormat | None":
        """Forwarded conditionally, the same reason and the same shape
        `PrerollReplayingSource.sink_format` above already uses -- without
        this, a follow-up turn on the camera would lose its own sink the
        instant a chained window wraps it here."""
        wrapped_sink_format = getattr(self._wrapped, "sink_format", None)
        return wrapped_sink_format() if wrapped_sink_format is not None else None

    @property
    def speech_signals(self) -> Any:
        """Forwarded conditionally, the same reason and shape
        `PrerollReplayingSource.speech_signals` above already uses
        (10-05-PLAN.md)."""
        return getattr(self._wrapped, "speech_signals", None)


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
        follow_up_window_s: Callable[[], float] | None = None,
        follow_up_echo_tail_s: float = 0.8,
        parallel: "ParallelTurns | None" = None,
        ring_window: "RingStopWindow | None" = None,
    ) -> None:
        self._name = name
        # Quick task 260930-e3r: `None` (every caller that predates it) means
        # no bare-stop listening while a timer or alarm rings.
        self._ring_window = ring_window
        # The parallel path runs the window as one task on a fan-out subscription.
        self._ring_window_task: "asyncio.Task[None] | None" = None
        self._source = source
        self._wake_detector = wake_detector
        self._decode_for_detector = decode_for_detector
        self._run_turn_fn = run_turn_fn
        self._clock = clock
        self._preroll = preroll
        # Plan 09-06 (D-06, D-09): `None` (every test and every caller that
        # predates this plan) attaches no `FollowUpChannel` at all and
        # behaves exactly as before -- a `SourceRunner` built with no
        # follow-up window opts nothing in. `follow_up_window_s` is a
        # callable, not a plain float, so a live setting change reaches
        # the very next turn without rebuilding this runner (the same
        # "resolved live, not just at construction" shape `wake_threshold`
        # already gives an operator for the wake gate).
        self._follow_up_window_s = follow_up_window_s
        self._follow_up_echo_tail_s = follow_up_echo_tail_s
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
        # Phase 13: each stop send runs as its own task, held here until its
        # done-callback discards it (the pattern the wake-event writes use).
        self._pending_stop_tasks: set[asyncio.Task] = set()
        # WR-06: the interrupt-started turns since the last idle wake hit. It
        # spans the whole call chain (interrupt turns and answer windows), so
        # the cap bounds alternation between the two.
        self._interrupt_turns = 0
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

        # Quick task 260924-4is (D2): every native call this runner makes
        # against `wake_detector`/`decode_for_detector` -- the wake hit
        # itself and the barge-in listener's own decode -- runs on this
        # one worker, never the loop thread and never the default pool.
        # One worker, not `asyncio.to_thread`'s shared default pool: Vosk
        # recognizers and PyAV codec contexts are stateful and not
        # thread-safe, and a cancelled await does not stop a call already
        # running in a thread, so a second call for this same source must
        # queue behind the first rather than run beside it. `run()`'s own
        # `finally` shuts this down with `wait=True`, so a model is never
        # freed while a call against it is still in flight.
        self._detector_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"atlas-wake-{name}")

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
        self._wake_phrase = resolved_wake.phrase
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

        # Phase 12: `None` keeps the serial path. A spec builds the group of live turns.
        self._turn_group: TurnGroup | None = None
        if parallel is not None:
            hooks = TurnHooks(
                run_turn=run_turn_fn,
                new_monitor=self._new_barge_in_monitor,
                watch_barge_in=self._watch_barge_in,
                record_blocked_hit=self._record_blocked_hit,
                clock=clock,
                follow_up_window_s=follow_up_window_s,
                follow_up_window_opens_at=self._follow_up_opens_at,
                follow_up_source=lambda wrapped, opens_at: FollowUpSource(wrapped, opens_at, clock),
            )
            self._turn_group = TurnGroup(name, source, parallel, hooks)

    @property
    def ring_window(self) -> "RingStopWindow | None":
        return self._ring_window

    @property
    def turn_group(self) -> "TurnGroup | None":
        return self._turn_group

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

        Quick task 260924-4is (D2): the wake detector's own `process()`
        call and its PyAV decode run on `self._detector_executor`, this
        runner's own single worker thread -- never on this loop thread.
        Before this fix, a backlog on `self._source.frames()`'s queue
        (nothing reads it, from the moment a turn's own transcript
        finalizes until this loop resumes) drained in one uninterrupted
        loop step, because `asyncio.Queue.get()` on a non-empty queue
        never yields: every chunk's decode and detector call ran back to
        back with no checkpoint the rest of the process could run at,
        and the longer that backlog grew, the longer the loop went
        unresponsive. This `finally` shuts the executor down with
        `wait=True` -- so `run()` returns only once its own in-flight
        detector call has actually finished, which is what keeps a
        caller's `wake_detector.close()` right after from freeing a
        model a worker thread is still using.
        """
        cancelled = False
        try:
            async for chunk in self._source.frames():
                if self._turn_group is not None:
                    self._turn_group.push_frame(chunk, getattr(self._source, "last_yielded_frame_index", None))
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
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if self._ring_window_task is not None:
                self._ring_window_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._ring_window_task
            if self._turn_group is not None:
                # Before the executor goes: a live turn's barge-in listener still uses it.
                self._turn_group.close_frames()
                await self._turn_group.drain(cancel=cancelled)
            self._detector_executor.shutdown(wait=True, cancel_futures=True)

    def _ensure_ring_window_task(self) -> None:
        """Parallel path: start the ring-stop window as one task, if none runs."""
        task = self._ring_window_task
        if task is not None and not task.done():
            return
        assert self._turn_group is not None and self._ring_window is not None
        subscription = self._turn_group.fanout.subscribe()
        window = self._ring_window

        async def _listen() -> None:
            try:
                await window.run(TurnFrameSource(subscription, self._source))
            finally:
                subscription.close()

        self._ring_window_task = asyncio.create_task(_listen())

    def _detect(self, chunk: bytes) -> WakeHit | None:
        """Runs on `self._detector_executor`'s single worker thread, never
        the loop thread (D2): decode, then the wake detector's own native
        call, both off-loop for the same reason."""
        detector_chunk = self._decode_for_detector(chunk)
        if not detector_chunk:
            return None
        return self._wake_detector.process(detector_chunk)

    def _detect_during_playback(self, chunk: bytes) -> WakeHit | None:
        """`_detect` for the barge-in listener, on the same worker thread.
        A detector with `process_during_playback` (Vosk, through
        `SegmentBoundedWakeDetector`) accepts the keyword alone while a
        reply plays, because the echo suppressor clips the "hey". A detector
        without it uses `process`."""
        detector_chunk = self._decode_for_detector(chunk)
        if not detector_chunk:
            return None
        process = getattr(self._wake_detector, "process_during_playback", None)
        if process is None:
            return self._wake_detector.process(detector_chunk)
        return process(detector_chunk)

    def _barge_in_energy(self, chunk: bytes) -> float | None:
        """The barge-in listener's own off-loop pair (D2): the same decode
        `_detect` makes, then an energy reading instead of a wake-detector
        call."""
        detector_chunk = self._decode_for_detector(chunk)
        if not detector_chunk:
            return None
        return rms_amplitude(detector_chunk)

    async def _process_chunk(self, chunk: bytes) -> None:
        """One chunk of `run()`'s own loop body, split out so `run()` can
        wrap it in the containment `try`/`except` above without also
        catching an exception from `self._source.frames()` itself."""
        if self._preroll is not None and self._turn_group is None:
            self._preroll.push(chunk)

        # Plan 11-07 (D-02, Research Assumption A4): duck-typed, the same
        # way `mark_wake_hit` below is -- only `EdgeAudioSource` has
        # `wake_suppressed`, so a camera or browser source is untouched.
        # A prompted enrollment phrase must never start a real turn; the
        # pre-roll buffer still gets the chunk just above, unconditionally,
        # so a turn started right after enrollment ends still has one.
        if getattr(self._source, "wake_suppressed", False):
            return

        # While a ring plays, the room has no wake word: a bare "stop" ends
        # it. The window reads the source the way a turn does, and it
        # starts no turn.
        if self._ring_window is not None and self._ring_window.active():
            if self._turn_group is None:
                await self._ring_window.run(self._source)
            else:
                self._ensure_ring_window_task()
            return

        hit = await asyncio.get_running_loop().run_in_executor(self._detector_executor, self._detect, chunk)
        if hit is None:
            return

        now = self._clock()
        decision = self._gate.evaluate(hit.score, now, self._last_hit_at)
        if not decision.allowed:
            self._record_blocked_hit(hit.score, decision.reason, now)
            return

        if self._turn_group is not None and not self._turn_group.check_admission(hit.score, now):
            return

        self._last_hit_at = now
        logger.info("wake hit on source %r (score=%.3f)", self._name, hit.score)
        self._schedule_wake_event_write(score=hit.score, allowed=True, block_reason=None)
        # Plan 11-04 (D-09): duck-typed, the same way `send_event` just
        # above is -- only `EdgeAudioSource` has `mark_wake_hit`, so a
        # camera or browser source is untouched. Tells the attached
        # `SpeakerTracker` (if any) which frame index the wake hit landed
        # on, so `open_turn()` can find the segment that holds it.
        mark_wake_hit = getattr(self._source, "mark_wake_hit", None)
        if mark_wake_hit is not None:
            mark_wake_hit()
        # Tells a source with a screen (the browser listener) that a turn is
        # starting. The camera's own `send_event` is a logged no-op.
        send_event = getattr(self._source, "send_event", None)
        if send_event is not None:
            await send_event({"type": "wake.heard"})

        if self._turn_group is not None:
            # Phase 12 (D-01): the turn runs as its own task; this loop keeps reading frames.
            assert self._turn_group.fanout.latest_index is not None
            self._turn_group.start_wake_turn(self._turn_group.fanout.latest_index)
            return

        # A source with an LED ring (the edge Pi) shows the turn state. The
        # ring stays dark until this point and goes dark again in `finally`.
        await self._set_led_state(LED_LISTENING)
        try:
            turn_source = self._source
            if self._preroll is not None:
                preroll_chunks = self._preroll.drain()
                if preroll_chunks:
                    turn_source = PrerollReplayingSource(self._source, preroll_chunks)

            monitor = self._new_barge_in_monitor()
            # Duck-typed, the same way `turn/controller.py`'s own
            # `_emit_event` already reaches for `send_event` -- `run_turn`
            # and `_speak` read this back with `getattr(source, "barge_in",
            # None)` rather than a positional `run_turn_fn` never had.
            turn_source.barge_in = monitor

            # Plan 09-06 (D-06): `None` (every caller that predates this plan)
            # attaches no channel at all -- `getattr(source, "follow_up",
            # None)` in `turn/controller.py` then finds nothing, and every
            # proposal on this source keeps speaking
            # `CONFIRMATION_UNAVAILABLE_REPLY`, byte-identical to before this
            # plan. Given a `follow_up_window_s`, a fresh, empty channel (no
            # `incoming` -- this is the wake turn itself, never answering a
            # prior request) is attached next to `barge_in`, and
            # `_run_follow_ups` below is what actually opens a window once
            # this turn's own `run_turn_fn` leaves something on it.
            follow_up_channel: "FollowUpChannel | None" = None
            if self._follow_up_window_s is not None:
                follow_up_channel = FollowUpChannel()
                turn_source.follow_up = follow_up_channel

            self._interrupt_turns = 0
            await self._run_one_turn(turn_source, monitor)
            # Phase 13: an interrupt hands its speech to the next turn.
            monitor, follow_up_channel = await self._continue_after_interrupts(monitor, follow_up_channel)

            if follow_up_channel is not None:
                await self._run_follow_ups(follow_up_channel)
        finally:
            await self._set_led_state(LED_IDLE)

    async def _set_led_state(self, state: str) -> None:
        """Tell the source to show `state` on its LED ring. A source with no
        `set_led_state` (camera, browser, test doubles) is skipped. A
        failure is logged and never reaches the turn."""
        set_led_state = getattr(self._source, "set_led_state", None)
        if set_led_state is None:
            return
        try:
            await set_led_state(state)
        except Exception:
            logger.warning("source %r: could not set the led state %r", self._name, state, exc_info=True)

    def _new_barge_in_monitor(self) -> "BargeInMonitor":
        """Build a fresh `BargeInMonitor`, with correlation wired up
        exactly the way `_process_chunk` always has (D-18) -- shared by
        the wake turn and by every follow-up turn in a chain
        (`_run_follow_ups` below), so neither duplicates the other's own
        correlation-wiring logic.

        D-18: correlation is only ever wired up when this source's own
        resolved policy asks for it *and* a calibration is actually on
        hand -- `app.py`'s startup refusal is what guarantees the second
        half of that whenever the first half is true (Task 3), so this
        constructor-time check is a second, cheap confirmation, never the
        only one. A fresh `EmittedAudioTrace` every turn, sized from the
        speaker's own sink format (260923-pyj, D5): the trace records the
        chunks `_speak` writes to the speaker, so it must be sized from
        the format those bytes are actually in. `source_format()` is only
        a fallback, for a source (a test double) that declares no sink at
        all -- it mirrors `PrerollBuffer`'s own per-source-format
        construction the same way `_process_chunk` always has.
        """
        correlation_active = self._barge_in_config.correlation_enabled and self._calibration is not None
        trace: EmittedAudioTrace | None = None
        if correlation_active:
            sink_format = getattr(self._source, "sink_format", None)
            sink = sink_format() if sink_format is not None else None
            if sink is not None:
                trace = EmittedAudioTrace(encoding=sink.codec, sample_rate=sink.sample_rate)
            else:
                source_format = self._source.source_format()
                trace = EmittedAudioTrace(encoding=source_format.encoding, sample_rate=source_format.sample_rate)
        # Wake interrupts and the playback hold need a source that can stop its
        # own playback (the camera has no `stop_playback`, D-04) on the serial
        # path. A Phase 12 parallel runner keeps its own wake detection in
        # `run()`, so it never gets them either.
        can_stop = getattr(self._source, "stop_playback", None) is not None and self._turn_group is None
        wake_interrupts = self._barge_in_config.wake_word and can_stop
        return BargeInMonitor(
            floor=self._barge_in_config.energy_floor,
            min_duration_s=self._barge_in_config.min_duration_ms / 1000.0,
            guard_window_s=self._barge_in_config.post_playback_guard_ms / 1000.0,
            enabled=self._barge_in_config.enabled,
            trace=trace,
            calibration=self._calibration if correlation_active else None,
            correlation_tolerance=self._barge_in_config.correlation_tolerance,
            tracking_adaptation_rate=self._barge_in_config.tracking_adaptation_rate,
            wake_interrupts=wake_interrupts,
            holds_for_playback=can_stop and (wake_interrupts or self._barge_in_config.enabled),
            on_interrupt=self._schedule_stop_playback if can_stop else None,
            wake_phrase=self._wake_phrase,
            reply_margin_s=self._barge_in_config.atlas_margin_ms / 1000.0,
            echo_delay_s=self._calibration.delay_s if self._calibration is not None else 0.0,
        )

    def _schedule_stop_playback(self) -> None:
        """Send the Pi its `stop` message from a task. The hook fires in the
        frame loop for a wake latch and in a speech-signal callback for a VAD
        latch, and the held turn ends as soon as the latch fires, which
        cancels the listener. A task sends the stop in every case."""
        task = asyncio.create_task(self._source.stop_playback(self._barge_in_config.fade_ms))
        self._pending_stop_tasks.add(task)
        task.add_done_callback(self._pending_stop_tasks.discard)

    async def _run_follow_ups(self, channel: "FollowUpChannel") -> None:
        """After the wake turn (or the previous follow-up turn) left a
        request on `channel.requested`, open one more no-wake-word window
        per request in the chain -- stopping the instant a turn leaves no
        request at all, or a request's own `chain_depth` exceeds
        `MAX_CHAINED_FOLLOW_UPS` (T-09-37) -- after which this simply
        returns, and `run()`'s own loop resumes listening for the wake
        word (D-06).

        `dispatch_handoff` (`turn/handoff.py`) already refuses to request
        a follow-up past `MAX_CHAINED_FOLLOW_UPS` on its own, so the
        `chain_depth` check here is a second, redundant guard against the
        same DoS shape (T-09-37) -- never the only one, the same
        two-independent-controls posture T-09-27 already established for
        a different guarantee in this same phase.

        The echo tail added on top of the readback's own estimated
        playback end is the larger of the configured
        `follow_up_echo_tail_s` and a real echo-path calibration's own
        measured delay plus a fixed 0.2 s margin (D-09) -- a calibration,
        when one exists, is a measured acoustic fact and must never be
        overridden by a smaller configured constant.
        """
        while channel.requested is not None and channel.requested.chain_depth <= MAX_CHAINED_FOLLOW_UPS:
            requested = channel.requested
            opens_at = self._follow_up_opens_at(requested)

            new_channel = FollowUpChannel(
                incoming=requested, window_opens_at=opens_at, window_s=self._follow_up_window_s()
            )
            # Plan 13-06: only an answer window listens for the wake word, and
            # only on the serial path. A confirmation or clarification window
            # never runs the detector.
            on_chunk = (
                self._answer_window_wake_tap(new_channel)
                if requested.kind == "answer" and self._turn_group is None
                else None
            )
            follow_up_source = FollowUpSource(self._source, opens_at, self._clock, on_chunk=on_chunk)
            monitor = self._new_barge_in_monitor()
            follow_up_source.barge_in = monitor
            follow_up_source.follow_up = new_channel

            # A follow-up window listens with no wake word. The Pi keeps the
            # replying color until its reply audio drains, so the listening
            # color shows after the reply ends.
            await self._set_led_state(LED_LISTENING)
            await self._run_one_turn(follow_up_source, monitor)

            # An interrupted turn requests no window (plan 13-01), so its own
            # channel is dropped. The turn that its interrupt starts may open one.
            monitor, new_channel = await self._continue_after_interrupts(monitor, new_channel)
            if new_channel is None:
                return
            channel = new_channel

    def _answer_window_wake_tap(self, channel: "FollowUpChannel") -> "Callable[[bytes], Awaitable[None]]":
        """The per-chunk tap for an answer window: the wake detector keeps
        running on the window's frames, so "Atlas, ..." said inside the window
        is still a wake command (plan 13-06). An allowed hit sets
        `channel.wake_heard`. The controller widens the scope only when the
        transcript also opens with the wake phrase. The frames start at
        `opens_at`, so the reply's own echo tail never reaches the detector.
        Only a hit within `WAKE_EVIDENCE_WINDOW_S` of the window's first audio
        counts (WR-02), so a hit cannot pair with an utterance that began
        later. Logs scores only."""
        started = False
        first_audio_at: "float | None" = None

        async def tap(chunk: bytes) -> None:
            nonlocal started, first_audio_at
            loop = asyncio.get_running_loop()
            if first_audio_at is None:
                first_audio_at = self._clock()
            if not started:
                started = True
                reset = getattr(self._wake_detector, "reset", None)
                if reset is not None:
                    await loop.run_in_executor(self._detector_executor, reset)
            # WR-02: only a hit at the start of the window's audio is evidence
            # for the utterance that the turn transcribes.
            if channel.wake_heard or self._clock() - first_audio_at > WAKE_EVIDENCE_WINDOW_S:
                return
            try:
                hit = await loop.run_in_executor(self._detector_executor, self._detect, chunk)
            except Exception:
                # WR-04: a detector error must not end the window's turn.
                logger.exception("source %r: wake detector failed in an answer window -- continuing", self._name)
                return
            if hit is None:
                return
            now = self._clock()
            if now - first_audio_at > WAKE_EVIDENCE_WINDOW_S:
                self._record_blocked_hit(hit.score, ANSWER_WINDOW_LATE_HIT_BLOCK, now)
                return
            decision = self._gate.evaluate(hit.score, now, self._last_hit_at)
            if not decision.allowed:
                self._record_blocked_hit(hit.score, decision.reason, now)
                return
            channel.wake_heard = True
            self._last_hit_at = now
            logger.info("wake hit inside an answer window on source %r (score=%.3f)", self._name, hit.score)
            self._schedule_wake_event_write(score=hit.score, allowed=True, block_reason=None)
            mark_wake_hit = getattr(self._source, "mark_wake_hit", None)
            if mark_wake_hit is not None:
                mark_wake_hit()

        return tap

    async def _continue_after_interrupts(
        self, monitor: "BargeInMonitor", channel: "FollowUpChannel | None"
    ) -> "tuple[BargeInMonitor, FollowUpChannel | None]":
        """While the turn that just ended was interrupted by a wake hit or a
        VAD start, run the speech that interrupted it as the next turn (D-15,
        D-16). Returns the last monitor and the channel its turn left, or
        `None` when `MAX_CHAINED_INTERRUPT_TURNS` ran and the last turn was
        interrupted again. The camera's energy interrupt starts no turn (D-04).

        The cap counts across the whole wake-started call chain (WR-06): every
        interrupt turn and every answer window it opens share one counter, so
        a television that says the wake word over each reply ends the chain
        here and the runner goes back to idle."""
        while monitor.interrupt_kind in ("wake", "vad") and self._interrupt_turns < MAX_CHAINED_INTERRUPT_TURNS:
            kind = monitor.interrupt_kind
            # WR-05: the Pi has its `stop` before the next turn's cue goes out.
            if self._pending_stop_tasks:
                await asyncio.gather(*list(self._pending_stop_tasks), return_exceptions=True)
            # One wrapper per turn: it replays the hand-over once (260922-woc).
            turn_source = PrerollReplayingSource(self._source, list(monitor.handover))
            next_monitor = self._new_barge_in_monitor()
            next_monitor.after_interrupt = True
            if kind == "wake":
                next_monitor.unverified_request = interrupt_resume_request(monitor)
            turn_source.barge_in = next_monitor
            channel = self._interrupt_channel(monitor, kind)
            if channel is not None:
                turn_source.follow_up = channel
            send_event = getattr(self._source, "send_event", None)
            if kind == "wake" and send_event is not None:
                await send_event({"type": "wake.heard"})
            await self._set_led_state(LED_LISTENING)
            # Kind and count only, never audio or text (T-13-26).
            logger.info(
                "source %r: a %s interrupt starts a turn (%d hand-over chunks)",
                self._name,
                kind,
                len(monitor.handover),
            )
            await self._run_one_turn(turn_source, next_monitor)
            monitor = next_monitor
            self._interrupt_turns += 1
        if monitor.interrupt_kind in ("wake", "vad"):
            logger.warning(
                "source %r: %d interrupt turns in a row, back to listening for the wake word",
                self._name,
                MAX_CHAINED_INTERRUPT_TURNS,
            )
            return monitor, None
        return monitor, channel

    def _interrupt_channel(self, monitor: "BargeInMonitor", kind: str) -> "FollowUpChannel | None":
        """The channel for the turn an interrupt of `kind` starts.

        A wake interrupt starts an ordinary wake turn: a fresh channel with
        nothing incoming, when follow-up windows are on, and that turn may
        open a window itself. If its transcript fails verification, it runs
        with the VAD request below instead (`unverified_request`, D-A).

        A VAD interrupt has no wake word, so it runs as a no-wake-word answer
        turn with the interrupted turn's scope (RESEARCH Finding 4, D-16). The
        scope is the tools that turn dispatched, and it fails closed to none.
        The window opens now, because the hand-over already starts at the VAD
        start. It is dead code while `enabled` is false (`EDGE_BARGE_IN_PROVEN`).
        """
        if kind != "vad":
            return FollowUpChannel() if self._follow_up_window_s is not None else None
        return FollowUpChannel(
            incoming=interrupt_resume_request(monitor),
            window_opens_at=self._clock(),
            window_s=self._follow_up_window_s() if self._follow_up_window_s is not None else None,
        )

    def _follow_up_opens_at(self, requested: FollowUpRequest) -> float:
        # The clock is read only when there is no playback end, as it always was.
        now = self._clock() if requested.playback_ends_at is None else 0.0
        return follow_up_window_opens_at(
            requested, echo_tail_s=self._follow_up_echo_tail_s, calibration=self._calibration, now=now
        )

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
                try:
                    await listener_task
                except Exception:
                    # WR-04: a failed listener must not hide the turn's own
                    # outcome or skip its follow-ups.
                    logger.warning(
                        "source %r: the barge-in listener ended with an error", self._name, exc_info=True
                    )
            # The listener pushed playback audio into the pre-roll to find a
            # wake hit. A wake hit already took it for the hand-over. What is
            # left must not lead the next idle turn's pre-roll.
            if self._preroll is not None and monitor.wake_interrupts:
                self._preroll.clear()

    async def _watch_barge_in(self, monitor: "BargeInMonitor", frames: "AsyncIterator[bytes] | None" = None) -> None:
        """Become the sole reader of `self._source.frames()` once it is
        safe to (module docstring), feeding every chunk into `monitor`
        until cancelled -- by energy for most sources, or by `vad.start`
        for a source that proves it needs no energy floor at all.

        A no-op for a disabled policy (D-12's per-source override): this
        never even waits on `transcript_done`, so a source with barge-in
        turned off never opens a second reader of its own queue at all.

        Plan 10-10 (D-16): `getattr(self._source, "speech_signals", None)`
        is the duck-typed signal that this source carries its own
        `vad.start`/`vad.end` history (`transports/edge.py`'s
        `SpeechSignals`) -- every other source (the camera, the browser)
        has no such attribute and falls through to the energy path
        unchanged below. For a source that has it, this method subscribes
        with `replay_segment=False`: only a `vad.start` arriving *while
        this listener is subscribed* (during this very playback) is ever
        a candidate interrupt, never a stale event from before the turn
        started. `monitor.process_speech_start` -- not `process_energy`
        -- is what a `vad.start` feeds, because D-16 makes this a
        different, stronger signal than microphone energy for a source
        whose own chip already cancels its echo. The frame loop itself
        still runs, still as the sole reader of `self._source.frames()`
        (module docstring) -- every chunk is drained and discarded, never
        fed to the energy path, which this source's own chip makes moot.
        The subscription is torn down in `finally`, so a cancelled
        listener never leaves a dangling subscriber on `speech_signals`.

        Quick task 260924-4is (D2): each chunk's decode and energy
        reading run on `self._detector_executor`, the same single worker
        thread `_process_chunk` uses -- never the loop thread, and never
        beside a wake-detector call for this source (one worker, so the
        two native calls for one source can never overlap). Unaffected by
        the `speech_signals` branch above: that branch never decodes or
        measures energy at all.

        Phase 12: `frames` replaces `self._source.frames()` when given. A
        parallel turn passes its own subscription, so this listener is never a
        second reader of the source.
        """
        if not monitor.active:
            return
        await monitor.transcript_done.wait()
        speech_signals = getattr(self._source, "speech_signals", None)
        # Phase 13 (D-01): a listener given its own frames is a Phase 12
        # parallel turn, and it never runs the wake detector.
        detect_wake = monitor.wake_interrupts and frames is None
        if detect_wake:
            # RESEARCH Pitfall 1: the turn reader consumed the frames since the
            # hit that started this turn, so a stale partial must not fire.
            reset = getattr(self._wake_detector, "reset", None)
            if reset is not None:
                await asyncio.get_running_loop().run_in_executor(self._detector_executor, reset)
        if speech_signals is not None:
            def _on_speech_signal(event: dict) -> None:
                # "vad.start" mirrors the edge protocol's own stable event
                # type string (`atlas_edge.protocol.MSG_VAD_START` /
                # `atlas.transports.edge.MSG_VAD_START`) -- duck-typed
                # here rather than imported, the same convention this
                # module's own docstring already uses for `send_event`/
                # `barge_in`/`follow_up` and `turn/controller.py`'s own
                # edge-event forwarding (`_EDGE_EVENT_PREFIX`).
                if event.get("type") == "vad.start":
                    monitor.process_speech_start(self._clock())

            unsubscribe = (
                speech_signals.subscribe(_on_speech_signal, replay_segment=False) if monitor.enabled else None
            )
            try:
                async for chunk in (frames if frames is not None else self._source.frames()):
                    # Phase 13 (RESEARCH Pitfall 2): every chunk this listener
                    # holds goes to the hand-over or the pre-roll before any
                    # await, so a cancelled await never drops one. A Phase 12
                    # listener (`frames` given) has no hand-over.
                    if monitor.interrupt_requested:
                        if frames is None:
                            monitor.handover.append(chunk)
                        continue
                    if not detect_wake:
                        continue
                    if self._preroll is not None:
                        self._preroll.push(chunk)
                    # D-B: a reply that says the wake word needs the full phrase.
                    detect = self._detect if monitor.reply_says_wake_word else self._detect_during_playback
                    try:
                        hit = await asyncio.get_running_loop().run_in_executor(
                            self._detector_executor, detect, chunk
                        )
                    except asyncio.CancelledError:
                        # A VAD start latched during the await: the chunk is
                        # part of what the next turn replays.
                        if monitor.interrupt_requested:
                            monitor.handover.append(chunk)
                        raise
                    except Exception:
                        # WR-04: the same containment `run()` gives the idle
                        # detector (CR-03). A bad chunk costs one decode, not
                        # the listener and the turn's follow-ups.
                        logger.exception(
                            "source %r: wake detector failed during playback -- continuing", self._name
                        )
                        if monitor.interrupt_requested:
                            monitor.handover.append(chunk)
                        continue
                    if monitor.interrupt_requested:
                        monitor.handover.append(chunk)
                    elif hit is not None:
                        self._playback_wake_hit(monitor, hit)
            finally:
                if unsubscribe is not None:
                    unsubscribe()
            return
        loop = asyncio.get_running_loop()
        async for chunk in (frames if frames is not None else self._source.frames()):
            energy = await loop.run_in_executor(self._detector_executor, self._barge_in_energy, chunk)
            if energy is None:
                continue
            monitor.process_energy(energy, self._clock())

    def _playback_wake_hit(self, monitor: "BargeInMonitor", hit: WakeHit) -> None:
        """A wake hit heard while a reply plays. The same threshold and
        refractory apply as when idle (D-03). Logs scores and reasons only,
        never text."""
        now = self._clock()
        decision = self._gate.evaluate(hit.score, now, self._last_hit_at)
        if not decision.allowed:
            self._record_blocked_hit(hit.score, decision.reason, now)
            logger.info(
                "wake hit during playback blocked on source %r (score=%.3f, reason=%s)",
                self._name,
                hit.score,
                decision.reason,
            )
            return
        if monitor.reply_wake_word_at(now):
            # Scores and the reason only, never the reply text (D-02).
            self._record_blocked_hit(hit.score, REPLY_WAKE_WORD_BLOCK, now)
            logger.info(
                "wake hit during playback suppressed: it lines up with the reply's own wake word "
                "(source %r, score=%.3f)",
                self._name,
                hit.score,
            )
            return
        if monitor.process_wake_hit(now):
            self._last_hit_at = now
            logger.info(
                "wake hit during playback on source %r (score=%.3f): interrupting the reply",
                self._name,
                hit.score,
            )
            self._schedule_wake_event_write(score=hit.score, allowed=True, block_reason=None)
            # D-15: the pre-roll holds the wake phrase and this chunk. It goes
            # to the next turn, and the frames read after it are appended by
            # the frame loop. All of this runs before the next await.
            monitor.handover = self._preroll.drain() if self._preroll is not None else []
            # The next turn's speaker span anchors to this frame, as an idle
            # hit does in `_process_chunk`.
            mark_wake_hit = getattr(self._source, "mark_wake_hit", None)
            if mark_wake_hit is not None:
                mark_wake_hit()

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
    follow_up_window_s: Callable[[], float] | None = None
    follow_up_echo_tail_s: float = 0.8
    parallel: "ParallelTurns | None" = None


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
            follow_up_window_s=spec.follow_up_window_s,
            follow_up_echo_tail_s=spec.follow_up_echo_tail_s,
            parallel=spec.parallel,
        )
        for name, spec in specs.items()
    ]
