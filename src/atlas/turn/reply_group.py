"""The reply coordinator for one edge source (Phase 12, plan 12-05).

Turns that run at the same time on one edge source form a group. This module
decides how their replies reach the speaker:

- Statements that are ready close together play as one merged reply. The
  texts join in speech order (`order_frame`) and go to ONE write call, which
  is one TTS call. No model rewrites the text, so a partial-failure report
  stays word for word (D-06).
- In a group of two or more turns, each line starts with its speaker's name
  (D-07). A turn that runs alone speaks its own text unchanged.
- The merge waits at most `merge_wait_s` after the first ready statement
  (D-08). A statement that arrives after the group's first flush plays on
  its own.
- A reply that asks a question plays after the statements, in its own write
  call. A second question waits until the first asking turn finishes its
  follow-up chain (D-09).
- A group plays at most one filler, and none once any group speech has
  started (D-11).

The coordinator is pure asyncio with an injected write function. It never
imports `turn/controller.py`, because the controller imports this module.
The write function owns the reply lock: it takes `handle.reply_lock` around
its own TTS call and write, so two replies never overlap on the speaker.

Every wait is bounded by `merge_wait_s`, and a finished handle counts as
ready (T-12-17). The label is an untrusted speaker guess and is used for
address only (T-12-15).
"""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from atlas.turn.follow_up import estimate_playback_end

REPLY_GROUP_EVENT = "reply.group"

# `write(text, needs_live_tts)` makes one TTS call and one write under the
# reply lock, and returns a `SpeechResult`-shaped object. `needs_live_tts`
# is True whenever `text` differs from the turn's own reply text (a prefix
# or a join), because a cached TTS holds only exact phrases.
WriteFn = Callable[[str, bool], Awaitable[Any]]

Role = Literal["alone", "lead", "follow", "late", "question"]

_END_MARKS = (".", "!", "?", "…")


def compose_group_line(label: str | None, text: str, *, prefixed: bool) -> str:
    """One line of a group reply: `"{label}, {text}"` when `prefixed` and a
    label exists, else `text`. The line ends with a period when it has no
    end mark, so joined lines read as sentences. Blank text gives a blank
    line, never a bare name.

    The join happens in code, after the model (T-12-15). The label is a
    member display name, never transcribed text.
    """
    body = text.strip()
    if not body:
        return ""
    line = f"{label}, {body}" if prefixed and label else body
    if not line.endswith(_END_MARKS):
        line += "."
    return line


@dataclass(frozen=True)
class GroupSpeech:
    """What one `ReplyHandle.speak` call got.

    `role` is `alone` (no group, own text), `lead` (wrote the merged reply),
    `follow` (its text was in the lead's merged reply), `late` (came after
    the group's first flush and played on its own), or `question` (a reply
    that asks, played after the statements).
    """

    result: Any
    role: Role
    merged_count: int
    prefixed: bool
    merge_wait_ms: float | None
    tts_ms: float | None


@dataclass(frozen=True)
class ReplyRoute:
    """What `_speak` needs to route one reply through its group."""

    handle: "ReplyHandle"
    live_tts: Any
    record_event: Callable[[dict], None] | None = None


# `run_turn` sets this for one turn's task, so `_speak` finds its group
# without a new parameter on every call site.
current_reply_route: ContextVar[ReplyRoute | None] = ContextVar("atlas_reply_route", default=None)


@dataclass(eq=False)
class _Waiter:
    ready: Callable[[], bool]
    event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(eq=False)
class _Statement:
    handle: "ReplyHandle"
    text: str
    write: WriteFn
    # Resolves to the follower's `GroupSpeech`, once the lead has written.
    future: "asyncio.Future[GroupSpeech | None]"


@dataclass(eq=False)
class _Batch:
    """The statements of the group's first flush."""

    first_at: float
    leader: _Statement
    statements: list[_Statement] = field(default_factory=list)
    forced: bool = False
    started: bool = False
    done: "asyncio.Future[None] | None" = None


class ReplyHandle:
    """One turn's seat in its group. `GroupSpeaker.register` makes it."""

    def __init__(
        self,
        speaker: "GroupSpeaker",
        turn_key: str,
        order_frame: int,
        group_id: str,
        *,
        joined: bool,
    ) -> None:
        self._speaker = speaker
        self.turn_key = turn_key
        self.order_frame = order_frame
        self.group_id = group_id
        # True when another turn was live at registration.
        self.joined = joined
        self.label: str | None = None
        # idle: has not spoken. pending: a statement waits for the flush.
        # asking: a question waits. spoken: has played in this group.
        self._state: Literal["idle", "pending", "asking", "spoken"] = "idle"

    @property
    def reply_lock(self) -> asyncio.Lock:
        """The speaker's lock. A write takes it, so two replies never overlap."""
        return self._speaker.reply_lock

    def set_label(self, label: str | None) -> None:
        self.label = label

    def claim_filler(self) -> bool:
        return self._speaker._claim_filler()

    def note_playback(self, ends_at: float | None) -> None:
        self._speaker._note_playback(ends_at)

    async def speak(self, reply_text: str, *, expects_answer: bool, write: WriteFn) -> GroupSpeech:
        if expects_answer:
            return await self._speaker._speak_question(self, reply_text, write)
        return await self._speaker._speak_statement(self, reply_text, write)

    def finish(self) -> None:
        self._speaker._finish(self)


class GroupSpeaker:
    """The live handles, pending statements, and flush state of one edge
    source's current group. All state changes run on the event loop with no
    `await` between check and set, so no state lock is needed.
    """

    def __init__(self, *, merge_wait_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._merge_wait_s = merge_wait_s
        self._clock = clock
        self.reply_lock = asyncio.Lock()
        self._playback_listener: Callable[[float | None], None] | None = None
        self._waiters: list[_Waiter] = []
        # Held by the turn whose question is the group's current one, until
        # that handle's `finish()`. It is free whenever a group resets.
        self._question_lock = asyncio.Lock()
        self._reset()

    def _reset(self) -> None:
        self._live: list[ReplyHandle] = []
        self._registered = 0
        self._batch: _Batch | None = None
        self._flush_started = False
        self._write_started = False
        self._filler_claimed = False
        self._question_holder: ReplyHandle | None = None

    def set_playback_listener(self, listener: Callable[[float | None], None] | None) -> None:
        self._playback_listener = listener

    def register(self, turn_key: str, order_frame: int, *, group_id: str) -> ReplyHandle:
        handle = ReplyHandle(self, turn_key, order_frame, group_id, joined=bool(self._live))
        self._live.append(handle)
        self._registered += 1
        return handle

    # -- waiting ---------------------------------------------------------

    def _others_ready(self, handle: ReplyHandle) -> bool:
        """Every other live handle has a pending statement, has a question
        waiting, has spoken already, or has finished."""
        return all(other is handle or other._state != "idle" for other in self._live)

    def _notify(self) -> None:
        for waiter in list(self._waiters):
            if waiter.ready():
                waiter.event.set()

    async def _wait_until(self, ready: Callable[[], bool]) -> None:
        """Wait until `ready()` holds, for at most `merge_wait_s`."""
        if ready():
            return
        waiter = _Waiter(ready)
        self._waiters.append(waiter)
        try:
            await asyncio.wait_for(waiter.event.wait(), self._merge_wait_s)
        except TimeoutError:
            pass
        finally:
            self._waiters.remove(waiter)

    # -- writing ---------------------------------------------------------

    async def _write(self, write: WriteFn, text: str, needs_live_tts: bool) -> tuple[Any, float | None]:
        self._write_started = True
        started_at = self._clock()
        result = await write(text, needs_live_tts)
        first_write_at = getattr(result, "first_write_at", None)
        tts_ms = None if first_write_at is None else max(0.0, (first_write_at - started_at) * 1000.0)
        return result, tts_ms

    async def _write_alone(self, handle: ReplyHandle, text: str, write: WriteFn) -> GroupSpeech:
        """The reply of a group that has had one member: own text, cached TTS."""
        self._flush_started = True
        result, tts_ms = await self._write(write, text, False)
        handle._state = "spoken"
        self._notify()
        return GroupSpeech(result, "alone", 1, False, None, tts_ms)

    # -- statements ------------------------------------------------------

    async def _write_own_line(self, handle: ReplyHandle, text: str, write: WriteFn, role: Role) -> GroupSpeech:
        """One reply that plays on its own inside a group: a late statement,
        a question, or a statement whose lead failed. The line is prefixed
        because the group has had two or more members."""
        self._flush_started = True
        prefixed = self._registered >= 2
        line = compose_group_line(handle.label, text, prefixed=True) if prefixed else text
        result, tts_ms = await self._write(write, line, line != text)
        handle._state = "spoken"
        self._notify()
        return GroupSpeech(result, role, 1, prefixed, None, tts_ms)

    async def _speak_statement(self, handle: ReplyHandle, text: str, write: WriteFn) -> GroupSpeech:
        if self._registered == 1:
            return await self._write_alone(handle, text, write)
        if self._flush_started:
            return await self._write_own_line(handle, text, write, "late")
        loop = asyncio.get_running_loop()
        statement = _Statement(handle, text, write, loop.create_future())
        batch = self._batch
        if batch is None:
            batch = self._batch = _Batch(first_at=self._clock(), leader=statement)
            batch.done = loop.create_future()
        batch.statements.append(statement)
        handle._state = "pending"
        self._notify()
        try:
            if batch.leader is statement:
                return await self._lead_flush(batch, statement)
            speech = await statement.future
        except BaseException:
            if batch.leader is statement:
                self._fail_batch(batch, statement)
            else:
                self._withdraw(batch, statement)
            raise
        if speech is None:
            # The lead failed. This turn's own text still has to be heard.
            return await self._write_own_line(handle, text, write, "late")
        return speech

    def _fail_batch(self, batch: _Batch, leader: _Statement) -> None:
        """The lead raised or was cancelled: release every follower so it
        writes its own line, and let a later statement start a fresh batch
        when nothing was flushed yet."""
        if self._batch is batch and not batch.started:
            self._batch = None
        for item in batch.statements:
            if item is not leader and not item.future.done():
                item.future.set_result(None)
        if batch.done is not None and not batch.done.done():
            batch.done.set_result(None)
        self._notify()

    def _withdraw(self, batch: _Batch, statement: _Statement) -> None:
        """A follower was cancelled before the flush: its text leaves the merge."""
        if not batch.started and statement in batch.statements:
            batch.statements.remove(statement)
            statement.handle._state = "idle"
            self._notify()

    # -- questions -------------------------------------------------------

    async def _speak_question(self, handle: ReplyHandle, text: str, write: WriteFn) -> GroupSpeech:
        """A reply that asks plays after the statements, in its own write, and
        the handle keeps the question slot until its `finish()` (D-09)."""
        if self._registered == 1:
            await self._take_question_slot(handle)
            return await self._write_alone(handle, text, write)
        # An asking handle counts as ready for everyone else's merge. The slot
        # is taken in arrival order, so the first question plays first.
        handle._state = "asking"
        self._notify()
        await self._take_question_slot(handle)
        await self._wait_until(lambda: self._others_ready(handle))
        batch = self._batch
        if batch is not None and batch.done is not None and not batch.done.done():
            # Statements come first. Flush them now, then wait for the write.
            batch.forced = True
            self._notify()
            await asyncio.shield(batch.done)
        return await self._write_own_line(handle, text, write, "question")

    async def _take_question_slot(self, handle: ReplyHandle) -> None:
        # A handle that asks again in its follow-up chain already holds it.
        if self._question_holder is handle:
            return
        await self._question_lock.acquire()
        self._question_holder = handle

    async def _lead_flush(self, batch: _Batch, leader: _Statement) -> GroupSpeech:
        await self._wait_until(lambda: batch.forced or self._others_ready(leader.handle))
        batch.started = True
        self._flush_started = True
        statements = sorted(batch.statements, key=lambda item: item.handle.order_frame)
        prefixed = self._registered >= 2
        lines = [compose_group_line(item.handle.label, item.text, prefixed=prefixed) for item in statements]
        merged = " ".join(line for line in lines if line)
        merge_wait_ms = (self._clock() - batch.first_at) * 1000.0
        result, tts_ms = await self._write(leader.write, merged, merged != leader.text)
        count = len(statements)
        for item in statements:
            item.handle._state = "spoken"
            if item is not leader and not item.future.done():
                item.future.set_result(GroupSpeech(result, "follow", count, prefixed, merge_wait_ms, tts_ms))
        if batch.done is not None and not batch.done.done():
            batch.done.set_result(None)
        self._notify()
        return GroupSpeech(result, "lead", count, prefixed, merge_wait_ms, tts_ms)

    # -- filler, playback, end of group ----------------------------------

    def _claim_filler(self) -> bool:
        """True once per group, and never after a write in the group started."""
        if self._write_started or self._filler_claimed:
            return False
        self._filler_claimed = True
        return True

    def _note_playback(self, ends_at: float | None) -> None:
        if self._playback_listener is not None:
            self._playback_listener(ends_at)

    def _finish(self, handle: ReplyHandle) -> None:
        if handle not in self._live:
            return
        if self._question_holder is handle:
            self._question_holder = None
            self._question_lock.release()
        self._live.remove(handle)
        self._notify()
        if not self._live:
            self._reset()


async def speak_in_group(
    route: ReplyRoute,
    reply_text: str,
    *,
    expects_answer: bool,
    sink: Any,
    write: WriteFn,
) -> GroupSpeech:
    """Speak one reply through its group, record one `reply.group` event,
    and report the write's playback end. The event carries no label and no
    reply text."""
    handle = route.handle
    speech = await handle.speak(reply_text, expects_answer=expects_answer, write=write)
    if route.record_event is not None:
        route.record_event(
            {
                "type": REPLY_GROUP_EVENT,
                "group_id": handle.group_id,
                "turn_key": handle.turn_key,
                "role": speech.role,
                "merged_count": speech.merged_count,
                "prefixed": speech.prefixed,
                "merge_wait_ms": speech.merge_wait_ms,
                "tts_ms": speech.tts_ms,
            }
        )
    # A follower's audio is the lead's write, so the lead reports it once.
    if speech.role != "follow":
        handle.note_playback(estimate_playback_end(speech.result, sink))
    return speech
