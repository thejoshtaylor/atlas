"""Plan 13-04 Task 3: a silent answer window leaves no session folder and no
live-feed card (T-13-21).

An answer window opens after every edge answer, and most hear nothing. Two
parts:

- `QuietStartPublisher` on its own, with a list-backed registry.
- The real `_make_run_turn_for_source` closure with a stubbed `run_turn`, the
  same technique `tests/test_observer_feed.py` uses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import atlas.app as app_module
from atlas.session.observers import ObserverRegistry
from atlas.session.quiet_feed import QuietStartPublisher
from atlas.turn.answer_window import ANSWER_WINDOW_SILENT
from atlas.turn.follow_up import FollowUpChannel, FollowUpRequest

from tests.conftest import FakeAudioSource
from tests.test_observer_feed import _fake_run_turn_app_state, _fake_run_turn_config

_STARTED = {"type": "turn.started", "source": "edge", "turn_id": "t1", "session_id": "s1"}


class _ListRegistry:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)


# --- the publisher -----------------------------------------------------------


def test_nothing_is_forwarded_while_only_content_free_events_arrive():
    registry = _ListRegistry()
    publisher = QuietStartPublisher(registry, _STARTED)

    publisher.publish({"type": "transcript.final", "text": ""})
    publisher.publish({"type": "transcript.partial", "text": "   "})
    publisher.publish({"type": "turn.timing", "turn_outcome": "answer_window_silent"})
    publisher.publish({"type": "reply.interrupted"})

    assert registry.events == []
    assert not publisher.started


def test_the_first_transcript_text_publishes_the_held_start_first():
    registry = _ListRegistry()
    publisher = QuietStartPublisher(registry, _STARTED)
    partial = {"type": "transcript.partial", "text": "and tomorrow"}
    timing = {"type": "turn.timing"}

    publisher.publish({"type": "turn.timing"})
    publisher.publish(partial)
    publisher.publish(timing)

    assert registry.events == [_STARTED, partial, timing]
    assert publisher.started


def test_a_reply_text_also_starts_the_card():
    registry = _ListRegistry()
    publisher = QuietStartPublisher(registry, _STARTED)
    reply = {"type": "reply.text", "text": "Rain."}

    publisher.publish(reply)

    assert registry.events == [_STARTED, reply]


# --- the real closure --------------------------------------------------------


def _answer_window_source() -> FakeAudioSource:
    source = FakeAudioSource()
    source.follow_up = FollowUpChannel(
        incoming=FollowUpRequest(
            kind="answer", chain_depth=1, original_transcript="what is the weather", question="It is sunny."
        )
    )
    return source


async def _run_closure(tmp_path, monkeypatch, source: FakeAudioSource, stub) -> list[dict[str, Any]]:
    monkeypatch.setattr(app_module, "run_turn", stub)
    registry = ObserverRegistry()
    queue = registry.subscribe()
    app_stub = SimpleNamespace(state=_fake_run_turn_app_state(registry))
    run_turn_for_source = app_module._make_run_turn_for_source(
        app_stub, _fake_run_turn_config(tmp_path), app_module.EDGE_SOURCE_NAME, room_speaker=False
    )

    await run_turn_for_source(source)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


async def test_a_silent_answer_window_publishes_no_card_and_leaves_no_folder(tmp_path, monkeypatch):
    async def stub(source, *args, **kwargs):
        timings = args[7]
        await source.send_event({"type": "transcript.final", "text": ""})
        timings.turn_outcome = ANSWER_WINDOW_SILENT
        await source.send_event({"type": "turn.timing", "turn_outcome": ANSWER_WINDOW_SILENT})
        # The recorder opened at window start, so its folder exists until the closure removes it.
        assert list(tmp_path.iterdir())

    events = await _run_closure(tmp_path, monkeypatch, _answer_window_source(), stub)

    assert events == []
    assert list(tmp_path.iterdir()) == []


async def test_an_answer_window_that_hears_speech_keeps_its_card_and_folder(tmp_path, monkeypatch):
    async def stub(source, *args, **kwargs):
        await source.send_event({"type": "transcript.final", "text": "and tomorrow"})
        await source.send_event({"type": "turn.timing", "turn_outcome": "completed"})

    events = await _run_closure(tmp_path, monkeypatch, _answer_window_source(), stub)

    assert [event["type"] for event in events] == ["turn.started", "transcript.final", "turn.timing"]
    assert all(event["source"] == "edge" for event in events)
    assert len(list(tmp_path.iterdir())) == 1


async def test_a_window_that_ended_on_a_stop_phrase_keeps_its_folder(tmp_path, monkeypatch):
    async def stub(source, *args, **kwargs):
        await source.send_event({"type": "transcript.final", "text": "never mind"})
        args[7].turn_outcome = "stopped"

    events = await _run_closure(tmp_path, monkeypatch, _answer_window_source(), stub)

    assert events[0]["type"] == "turn.started"
    assert len(list(tmp_path.iterdir())) == 1


async def test_a_wake_turn_publishes_its_start_at_once_and_keeps_its_folder(tmp_path, monkeypatch):
    async def stub(source, *args, **kwargs):
        # Nothing is emitted: the start was already published.
        args[7].turn_outcome = ANSWER_WINDOW_SILENT

    events = await _run_closure(tmp_path, monkeypatch, FakeAudioSource(), stub)

    assert [event["type"] for event in events] == ["turn.started"]
    assert len(list(tmp_path.iterdir())) == 1
