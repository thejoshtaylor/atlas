"""`GroupSpeaker` and its helpers (Phase 12, plan 12-05): the reply
coordinator that merges the replies of turns that run at the same time on
one edge source.

The coordinator is pure asyncio with an injected write function, so every
test here uses a fake write that records its calls. Waits are short real
waits, well under one second.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from types import SimpleNamespace

from atlas.turn.reply_group import (
    REPLY_GROUP_EVENT,
    GroupSpeaker,
    ReplyRoute,
    compose_group_line,
    current_reply_route,
    speak_in_group,
)

_TIMEOUT_S = 2.0


@dataclass
class FakeResult:
    """A `SpeechResult`-shaped value: what one write returns."""

    bytes_sent: int
    first_write_at: float | None
    last_write_at: float | None


class FakeWrite:
    """A write function that records `(text, needs_live_tts)` per call."""

    def __init__(self, *, delay_s: float = 0.0, lock: asyncio.Lock | None = None):
        self.calls: list[tuple[str, bool]] = []
        self.starts: list[float] = []
        self.ends: list[float] = []
        self._delay_s = delay_s
        self._lock = lock

    async def __call__(self, text: str, needs_live_tts: bool) -> FakeResult:
        if self._lock is None:
            return await self._write(text, needs_live_tts)
        async with self._lock:
            return await self._write(text, needs_live_tts)

    async def _write(self, text: str, needs_live_tts: bool) -> FakeResult:
        self.calls.append((text, needs_live_tts))
        self.starts.append(time.monotonic())
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        now = time.monotonic()
        self.ends.append(now)
        return FakeResult(bytes_sent=len(text), first_write_at=now, last_write_at=now)


def _speak(handle, text: str, write, *, expects_answer: bool = False) -> asyncio.Task:
    return asyncio.create_task(handle.speak(text, expects_answer=expects_answer, write=write))


async def _gather(*tasks):
    return await asyncio.wait_for(asyncio.gather(*tasks), _TIMEOUT_S)


def _two_handles(speaker: GroupSpeaker):
    """Handle A (frame 100, "Josh") and handle B (frame 300, "Sam"), one group."""
    a = speaker.register("turn-a", 100, group_id="group-1")
    a.set_label("Josh")
    b = speaker.register("turn-b", 300, group_id="group-1")
    b.set_label("Sam")
    return a, b


def test_compose_group_line_prefixes_the_label_and_ends_with_a_period():
    assert compose_group_line("Josh", "the lamp is on", prefixed=True) == "Josh, the lamp is on."


def test_compose_group_line_keeps_an_existing_end_mark():
    assert compose_group_line("Sam", "is it late?", prefixed=True) == "Sam, is it late?"
    assert compose_group_line(None, "done!", prefixed=False) == "done!"


def test_compose_group_line_without_a_prefix_or_a_label_is_the_text():
    assert compose_group_line("Josh", "the lamp is on", prefixed=False) == "the lamp is on."
    assert compose_group_line(None, "the lamp is on", prefixed=True) == "the lamp is on."


def test_compose_group_line_of_blank_text_is_empty():
    assert compose_group_line("Josh", "  ", prefixed=True) == ""


def test_the_reply_route_context_variable_defaults_to_none():
    assert current_reply_route.get() is None
    assert current_reply_route.name == "atlas_reply_route"


async def test_two_statements_play_as_one_merged_reply_in_speech_order():
    speaker = GroupSpeaker(merge_wait_s=1.0)
    a, b = _two_handles(speaker)
    assert b.joined is True
    assert a.joined is False
    write = FakeWrite()

    # B speaks first, but A has the earlier order_frame.
    task_b = _speak(b, "it is 18 degrees", write)
    await asyncio.sleep(0)
    task_a = _speak(a, "the lamp is on", write)
    speech_a, speech_b = await _gather(task_a, task_b)

    assert write.calls == [("Josh, the lamp is on. Sam, it is 18 degrees.", True)]
    assert speech_a.result is speech_b.result
    assert speech_b.role == "lead"
    assert speech_a.role == "follow"
    assert speech_a.merged_count == speech_b.merged_count == 2
    assert speech_a.prefixed is True
    assert speech_b.prefixed is True


async def test_the_merge_flushes_at_once_when_every_live_turn_has_spoken():
    speaker = GroupSpeaker(merge_wait_s=5.0)
    a, b = _two_handles(speaker)
    write = FakeWrite()

    started = time.monotonic()
    task_a = _speak(a, "the lamp is on", write)
    await asyncio.sleep(0)
    task_b = _speak(b, "it is 18 degrees", write)
    await _gather(task_a, task_b)

    assert time.monotonic() - started < 1.0


async def test_a_handle_alone_speaks_its_own_text_unchanged_and_at_once():
    speaker = GroupSpeaker(merge_wait_s=5.0)
    handle = speaker.register("turn-a", 100, group_id="group-1")
    handle.set_label("Josh")
    assert handle.joined is False
    write = FakeWrite()

    started = time.monotonic()
    speech = await asyncio.wait_for(
        handle.speak("done", expects_answer=False, write=write), _TIMEOUT_S
    )

    assert write.calls == [("done", False)]
    assert speech.role == "alone"
    assert speech.merged_count == 1
    assert speech.prefixed is False
    assert speech.merge_wait_ms is None
    assert time.monotonic() - started < 1.0


async def test_speak_in_group_records_one_event_without_a_label_or_text():
    speaker = GroupSpeaker(merge_wait_s=1.0)
    a, b = _two_handles(speaker)
    write = FakeWrite()
    events_a: list[dict] = []
    events_b: list[dict] = []
    ends: list[float | None] = []
    speaker.set_playback_listener(ends.append)
    route_a = ReplyRoute(handle=a, live_tts=object(), record_event=events_a.append)
    route_b = ReplyRoute(handle=b, live_tts=object(), record_event=events_b.append)
    sink = SimpleNamespace(codec="alaw", sample_rate=8000)

    task_b = asyncio.create_task(
        speak_in_group(route_b, "it is 18 degrees", expects_answer=False, sink=sink, write=write)
    )
    await asyncio.sleep(0)
    task_a = asyncio.create_task(
        speak_in_group(route_a, "the lamp is on", expects_answer=False, sink=sink, write=write)
    )
    await _gather(task_a, task_b)

    assert len(events_a) == len(events_b) == 1
    expected_keys = {
        "type",
        "group_id",
        "turn_key",
        "role",
        "merged_count",
        "prefixed",
        "merge_wait_ms",
        "tts_ms",
    }
    for event, role, turn_key in ((events_a[0], "follow", "turn-a"), (events_b[0], "lead", "turn-b")):
        assert set(event) == expected_keys
        assert event["type"] == REPLY_GROUP_EVENT == "reply.group"
        assert event["group_id"] == "group-1"
        assert event["turn_key"] == turn_key
        assert event["role"] == role
        assert event["merged_count"] == 2
        assert event["prefixed"] is True
        assert "Josh" not in repr(event)
        assert "lamp" not in repr(event)
    # Only the write's owner reports a playback end; a follower does not.
    assert len(ends) == 1
    assert ends[0] is not None


async def test_speak_in_group_without_a_recorder_or_a_sink_still_reports_playback():
    speaker = GroupSpeaker(merge_wait_s=1.0)
    handle = speaker.register("turn-a", 100, group_id="group-1")
    ends: list[float | None] = []
    speaker.set_playback_listener(ends.append)
    route = ReplyRoute(handle=handle, live_tts=object())
    write = FakeWrite()

    speech = await speak_in_group(route, "done", expects_answer=False, sink=None, write=write)

    assert speech.role == "alone"
    assert ends == [speech.result.last_write_at]


# -- Task 2: the bounded wait, late replies, questions last, one filler, a
# -- leader that fails ------------------------------------------------------


async def test_the_wait_is_bounded_and_a_later_statement_plays_on_its_own():
    speaker = GroupSpeaker(merge_wait_s=0.05)
    a, b = _two_handles(speaker)
    write = FakeWrite()

    started = time.monotonic()
    speech_a = await asyncio.wait_for(
        a.speak("the lamp is on", expects_answer=False, write=write), _TIMEOUT_S
    )
    waited = time.monotonic() - started

    assert 0.03 <= waited < 1.0
    assert write.calls == [("Josh, the lamp is on.", True)]
    assert speech_a.role == "lead"
    assert speech_a.merged_count == 1
    assert speech_a.prefixed is True
    assert speech_a.merge_wait_ms is not None and speech_a.merge_wait_ms >= 30

    speech_b = await asyncio.wait_for(
        b.speak("it is 18 degrees", expects_answer=False, write=write), _TIMEOUT_S
    )

    assert write.calls[1] == ("Sam, it is 18 degrees.", True)
    assert speech_b.role == "late"
    assert speech_b.merged_count == 1
    assert speech_b.prefixed is True


async def test_a_finished_turn_lets_the_pending_statement_flush_at_once():
    speaker = GroupSpeaker(merge_wait_s=5.0)
    a, b = _two_handles(speaker)
    write = FakeWrite()

    started = time.monotonic()
    task_a = _speak(a, "the lamp is on", write)
    await asyncio.sleep(0.01)
    assert write.calls == []
    b.finish()
    speech_a = await asyncio.wait_for(task_a, _TIMEOUT_S)

    assert time.monotonic() - started < 1.0
    assert speech_a.role == "lead"
    assert speech_a.merged_count == 1
    assert write.calls == [("Josh, the lamp is on.", True)]


async def test_a_question_plays_after_the_pending_statement_in_its_own_write():
    speaker = GroupSpeaker(merge_wait_s=1.0)
    a, b = _two_handles(speaker)
    write = FakeWrite()

    task_b = _speak(b, "it is 18 degrees", write)
    await asyncio.sleep(0)
    task_a = _speak(a, "shall I turn it off?", write, expects_answer=True)
    speech_a, speech_b = await _gather(task_a, task_b)

    assert write.calls == [
        ("Sam, it is 18 degrees.", True),
        ("Josh, shall I turn it off?", True),
    ]
    assert speech_b.role == "lead"
    assert speech_a.role == "question"
    assert speech_a.merged_count == 1
    assert speech_a.prefixed is True


async def test_a_question_that_arrives_first_still_plays_after_the_statement():
    speaker = GroupSpeaker(merge_wait_s=1.0)
    a, b = _two_handles(speaker)
    write = FakeWrite()

    task_a = _speak(a, "shall I turn it off?", write, expects_answer=True)
    await asyncio.sleep(0.01)
    assert write.calls == []
    task_b = _speak(b, "it is 18 degrees", write)
    speech_a, speech_b = await _gather(task_a, task_b)

    assert [text for text, _ in write.calls] == [
        "Sam, it is 18 degrees.",
        "Josh, shall I turn it off?",
    ]
    assert speech_a.role == "question"
    assert speech_b.role == "lead"


async def test_a_question_alone_in_its_group_plays_at_once_with_its_own_text():
    speaker = GroupSpeaker(merge_wait_s=5.0)
    handle = speaker.register("turn-a", 100, group_id="group-1")
    write = FakeWrite()

    speech = await asyncio.wait_for(
        handle.speak("shall I turn it off?", expects_answer=True, write=write), _TIMEOUT_S
    )

    assert write.calls == [("shall I turn it off?", False)]
    assert speech.role == "alone"


async def test_the_second_question_waits_until_the_first_asker_finishes():
    speaker = GroupSpeaker(merge_wait_s=0.05)
    a, b = _two_handles(speaker)
    write = FakeWrite()

    task_a = _speak(a, "shall I turn it off?", write, expects_answer=True)
    task_b = _speak(b, "shall I turn it on?", write, expects_answer=True)
    await asyncio.sleep(0.15)

    assert write.calls == [("Josh, shall I turn it off?", True)]
    assert task_a.done()
    assert not task_b.done()

    a.finish()
    speech_b = await asyncio.wait_for(task_b, _TIMEOUT_S)

    assert write.calls[1] == ("Sam, shall I turn it on?", True)
    assert speech_b.role == "question"


async def test_a_handle_that_asks_again_in_its_chain_keeps_its_question_slot():
    speaker = GroupSpeaker(merge_wait_s=0.05)
    a, b = _two_handles(speaker)
    b.finish()
    write = FakeWrite()

    first = await asyncio.wait_for(
        a.speak("which lamp?", expects_answer=True, write=write), _TIMEOUT_S
    )
    second = await asyncio.wait_for(
        a.speak("the big one?", expects_answer=True, write=write), _TIMEOUT_S
    )

    assert first.role == second.role == "question"
    assert [text for text, _ in write.calls] == ["Josh, which lamp?", "Josh, the big one?"]


async def test_one_filler_per_group_and_none_once_a_write_has_started():
    speaker = GroupSpeaker(merge_wait_s=0.05)
    a, b = _two_handles(speaker)

    assert a.claim_filler() is True
    assert a.claim_filler() is False
    assert b.claim_filler() is False

    a.finish()
    b.finish()
    c = speaker.register("turn-c", 500, group_id="group-2")
    assert c.claim_filler() is True

    c.finish()
    d = speaker.register("turn-d", 600, group_id="group-3")
    seen: list[bool] = []

    async def write(text: str, needs_live_tts: bool) -> FakeResult:
        seen.append(d.claim_filler())
        now = time.monotonic()
        return FakeResult(len(text), now, now)

    await d.speak("done", expects_answer=False, write=write)

    assert seen == [False]
    assert d.claim_filler() is False


async def test_no_filler_after_the_group_first_flush_started():
    speaker = GroupSpeaker(merge_wait_s=0.05)
    a, b = _two_handles(speaker)
    write = FakeWrite()

    await a.speak("the lamp is on", expects_answer=False, write=write)

    assert b.claim_filler() is False


async def test_two_writes_never_overlap_through_the_reply_lock():
    speaker = GroupSpeaker(merge_wait_s=0.02)
    a, b = _two_handles(speaker)
    c = speaker.register("turn-c", 500, group_id="group-1")
    c.set_label("Kim")
    write = FakeWrite(delay_s=0.1, lock=speaker.reply_lock)

    task_a = _speak(a, "the lamp is on", write)
    task_b = _speak(b, "it is 18 degrees", write)
    await asyncio.sleep(0.06)
    # The merged write (A and B) is running. C is late and must wait for it.
    assert len(write.calls) == 1
    task_c = _speak(c, "the door is shut", write)
    await _gather(task_a, task_b, task_c)

    assert len(write.calls) == 2
    assert write.calls[0][0] == "Josh, the lamp is on. Sam, it is 18 degrees."
    assert write.calls[1][0] == "Kim, the door is shut."
    assert write.starts[1] >= write.ends[0]
    assert speaker.reply_lock.locked() is False


async def test_a_failed_leader_lets_each_follower_write_its_own_line():
    speaker = GroupSpeaker(merge_wait_s=1.0)
    a, b = _two_handles(speaker)
    write_a = FakeWrite()

    async def failing_write(text: str, needs_live_tts: bool):
        raise RuntimeError("speaker is down")

    task_b = _speak(b, "it is 18 degrees", failing_write)
    await asyncio.sleep(0)
    task_a = _speak(a, "the lamp is on", write_a)
    results = await asyncio.wait_for(
        asyncio.gather(task_a, task_b, return_exceptions=True), _TIMEOUT_S
    )

    speech_a, error_b = results
    assert isinstance(error_b, RuntimeError)
    assert write_a.calls == [("Josh, the lamp is on.", True)]
    assert speech_a.role == "late"
    assert speech_a.merged_count == 1


async def test_a_cancelled_leader_releases_its_followers():
    speaker = GroupSpeaker(merge_wait_s=5.0)
    a, b = _two_handles(speaker)
    c = speaker.register("turn-c", 500, group_id="group-1")
    write = FakeWrite()

    task_b = _speak(b, "it is 18 degrees", write)
    await asyncio.sleep(0)
    task_a = _speak(a, "the lamp is on", write)
    await asyncio.sleep(0.01)
    assert c.claim_filler() is True  # C is idle, so the merge is still waiting
    task_b.cancel()
    speech_a = await asyncio.wait_for(task_a, _TIMEOUT_S)

    assert speech_a.role == "late"
    assert write.calls == [("Josh, the lamp is on.", True)]


async def test_a_cancelled_follower_leaves_the_merge():
    speaker = GroupSpeaker(merge_wait_s=1.0)
    a, b = _two_handles(speaker)
    c = speaker.register("turn-c", 500, group_id="group-1")
    write = FakeWrite()

    task_b = _speak(b, "it is 18 degrees", write)
    await asyncio.sleep(0)
    task_a = _speak(a, "the lamp is on", write)
    await asyncio.sleep(0.01)
    task_a.cancel()
    await asyncio.sleep(0)
    c.finish()
    speech_b = await asyncio.wait_for(task_b, _TIMEOUT_S)

    assert speech_b.merged_count == 1
    assert write.calls == [("Sam, it is 18 degrees.", True)]


async def test_a_new_registration_after_the_last_finish_starts_a_fresh_group():
    speaker = GroupSpeaker(merge_wait_s=0.05)
    a, b = _two_handles(speaker)
    a.finish()
    b.finish()
    b.finish()  # a second finish is harmless

    c = speaker.register("turn-c", 500, group_id="group-2")
    c.set_label("Kim")
    write = FakeWrite()
    speech = await asyncio.wait_for(
        c.speak("done", expects_answer=False, write=write), _TIMEOUT_S
    )

    assert c.joined is False
    assert speech.role == "alone"
    assert speech.prefixed is False
    assert write.calls == [("done", False)]
