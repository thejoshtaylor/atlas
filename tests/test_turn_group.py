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
