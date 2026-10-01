"""`DesktopEventBridge`: the admission gate, the translation of
`wake.confirmed`, and the subscribe-once loop (Phase 15, D-05, D-12, D-13).

`handle` runs against a hub double that records `broadcast` calls. One test
runs `start()` against a real `ObserverRegistry`.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from atlas.desktop.bridge import DesktopEventBridge
from atlas.desktop.protocol import SERVER_MESSAGE_ADAPTER
from atlas.session.observers import ObserverRegistry


class _RecordingHub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.keys: list[str | None] = []

    def broadcast(self, frame: str, *, frame_type: str, coalesce_key=None) -> int:
        self.calls.append((frame, frame_type))
        self.keys.append(coalesce_key)
        return 1


def _bridge(hub=None, registry=None, **overrides) -> tuple[DesktopEventBridge, _RecordingHub]:
    hub = hub or _RecordingHub()
    overrides.setdefault("follow_up_window_ms", lambda: 8800)
    bridge = DesktopEventBridge(registry or ObserverRegistry(), hub, **overrides)
    return bridge, hub


def test_a_wake_confirmed_event_is_broadcast_as_the_wire_frame() -> None:
    bridge, hub = _bridge()

    bridge.handle({"type": "wake.confirmed", "source": "camera", "turn_id": "t-1"})

    [(frame, frame_type)] = hub.calls
    assert frame_type == "wake.confirmed"
    assert json.loads(frame) == {"type": "wake.confirmed", "turn_id": "t-1"}


@pytest.mark.parametrize(
    "event_type", ["transcript.partial", "transcript.final", "reply.text", "turn.timing"]
)
def test_an_event_of_a_turn_that_was_never_admitted_is_dropped(event_type) -> None:
    bridge, hub = _bridge()

    bridge.handle({"type": event_type, "text": "x", "source": "camera", "turn_id": "t-tv"})

    assert hub.calls == []


def test_an_event_with_no_turn_id_is_dropped() -> None:
    bridge, hub = _bridge()

    bridge.handle({"type": "wake.confirmed", "source": "camera"})
    bridge.handle({"type": "transcript.partial", "text": "x"})

    assert hub.calls == []


def test_wake_heard_broadcasts_nothing() -> None:
    bridge, hub = _bridge()

    bridge.handle({"type": "wake.heard", "source": "camera", "turn_id": "t-1"})

    assert hub.calls == []


def test_the_admitted_set_never_grows_past_its_cap_and_evicts_the_oldest() -> None:
    bridge, _hub = _bridge(max_admitted=3)

    for turn in ("t-1", "t-2", "t-3", "t-4"):
        bridge.handle({"type": "wake.confirmed", "turn_id": turn})

    assert list(bridge._admitted) == ["t-2", "t-3", "t-4"]


async def test_start_subscribes_once_delivers_a_published_wake_and_stop_unsubscribes() -> None:
    registry = ObserverRegistry()
    bridge, hub = _bridge(registry=registry)

    bridge.start()
    bridge.start()  # a second start must not subscribe a second queue
    assert len(registry._queues) == 1

    registry.publish({"type": "wake.confirmed", "source": "camera", "turn_id": "t-9"})
    for _ in range(100):
        if hub.calls:
            break
        await asyncio.sleep(0.005)
    assert [frame_type for _frame, frame_type in hub.calls] == ["wake.confirmed"]

    await bridge.stop()
    assert len(registry._queues) == 0


# --- Plan 15-05: the whole turn, the outcome map and follow-up admission -----


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _admitted(source: str = "camera", turn_id: str = "t-1", **overrides):
    clock = _Clock()
    bridge, hub = _bridge(clock=clock, **overrides)
    bridge.handle({"type": "wake.confirmed", "source": source, "turn_id": turn_id})
    hub.calls.clear()
    hub.keys.clear()
    return bridge, hub, clock


def _sent(hub: _RecordingHub) -> list[dict]:
    """Every frame sent so far, each checked against the server wire model."""
    out = []
    for frame, _frame_type in hub.calls:
        SERVER_MESSAGE_ADAPTER.validate_json(frame)
        out.append(json.loads(frame))
    return out


def _ended(turn_id="t-1", outcome="completed", **fields) -> dict:
    event = {
        "type": "turn.ended",
        "outcome": outcome,
        "follow_up": False,
        "asks_question": False,
        "playback_ms_left": 0,
        "source": "camera",
        "turn_id": turn_id,
    }
    event.update(fields)
    return event


def test_a_partial_is_sanitized_and_broadcast_with_the_turn_id_as_coalesce_key() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle(
        {"type": "transcript.partial", "text": "  hey   atlas\u202e what ", "turn_id": "t-1"}
    )

    assert _sent(hub) == [{"type": "transcript.partial", "turn_id": "t-1", "text": "hey atlas what"}]
    assert hub.calls[0][1] == "transcript.partial"
    assert hub.keys == ["t-1"]


def test_a_partial_keeps_the_newest_thousand_characters() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle({"type": "transcript.partial", "text": "a" * 1500 + "END", "turn_id": "t-1"})

    [frame] = _sent(hub)
    assert len(frame["text"]) == 1000
    assert frame["text"].endswith("END")


def test_a_partial_that_is_empty_after_sanitizing_sends_nothing() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle({"type": "transcript.partial", "text": " \u202e\x07 ", "turn_id": "t-1"})

    assert hub.calls == []


def test_the_final_transcript_is_followed_by_the_thinking_state() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle({"type": "transcript.final", "text": "what is the time", "turn_id": "t-1"})

    assert _sent(hub) == [
        {"type": "transcript.final", "turn_id": "t-1", "text": "what is the time"},
        {"type": "state", "turn_id": "t-1", "state": "thinking"},
    ]
    assert [frame_type for _f, frame_type in hub.calls] == ["transcript.final", "state"]


def test_reply_started_sends_speaking_then_one_code_built_text_card() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle({"type": "reply.started", "text": "It is sunny.", "turn_id": "t-1"})

    state, card = _sent(hub)
    assert state == {"type": "state", "turn_id": "t-1", "state": "speaking"}
    assert card["type"] == "card"
    assert card["kind"] == "text"
    assert card["card_id"] == "reply"
    assert card["data"] == {"text": "It is sunny."}
    assert [frame_type for _f, frame_type in hub.calls] == ["state", "card"]


def test_reply_started_with_blank_text_sends_speaking_and_no_card() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle({"type": "reply.started", "text": "   ", "turn_id": "t-1"})

    assert _sent(hub) == [{"type": "state", "turn_id": "t-1", "state": "speaking"}]


def test_a_barge_in_ends_as_stopped_with_no_follow_up_window() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle(_ended(outcome="barged_in", playback_ms_left=1200))

    assert _sent(hub) == [
        {
            "type": "turn.ended",
            "turn_id": "t-1",
            "outcome": "stopped",
            "follow_up_window_ms": 0,
            "playback_ms_left": 1200,
        }
    ]
    assert hub.calls[0][1] == "turn.ended"


def test_a_turn_that_asks_a_question_carries_the_follow_up_window() -> None:
    bridge, hub, _ = _admitted(follow_up_window_ms=lambda: 6800)

    bridge.handle(_ended(follow_up=True, asks_question=True, playback_ms_left=500))

    [frame] = _sent(hub)
    assert frame["follow_up_window_ms"] == 6800
    assert frame["playback_ms_left"] == 500


def test_an_unknown_internal_outcome_goes_out_as_completed() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle(_ended(outcome="brand_new_outcome"))

    assert _sent(hub)[0]["outcome"] == "completed"


def test_playback_ms_left_is_clamped_to_the_wire_range() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle(_ended(playback_ms_left=-5))
    assert _sent(hub)[0]["playback_ms_left"] == 0

    bridge2, hub2, _ = _admitted()
    bridge2.handle(_ended(playback_ms_left=10**9))
    assert _sent(hub2)[0]["playback_ms_left"] == 600_000


def test_after_turn_ended_a_late_event_of_that_turn_sends_nothing() -> None:
    bridge, hub, _ = _admitted()
    bridge.handle(_ended())
    hub.calls.clear()

    bridge.handle({"type": "transcript.partial", "text": "late", "turn_id": "t-1"})
    bridge.handle({"type": "reply.started", "text": "late", "turn_id": "t-1"})
    bridge.handle(_ended())

    assert hub.calls == []


def test_a_follow_up_turn_from_the_same_source_inside_the_window_is_admitted() -> None:
    bridge, hub, clock = _admitted(follow_up_window_ms=lambda: 6800)
    bridge.handle(_ended(follow_up=True, asks_question=True, playback_ms_left=1000))
    hub.calls.clear()

    clock.now += 5.0
    bridge.handle({"type": "turn.started", "follow_up": True, "source": "camera", "turn_id": "t-2"})
    bridge.handle({"type": "transcript.partial", "text": "yes", "turn_id": "t-2"})

    assert _sent(hub) == [
        {"type": "state", "turn_id": "t-2", "state": "listening"},
        {"type": "transcript.partial", "turn_id": "t-2", "text": "yes"},
    ]


def test_a_follow_up_turn_from_another_source_is_not_admitted() -> None:
    bridge, hub, clock = _admitted()
    bridge.handle(_ended(follow_up=True, asks_question=True))
    hub.calls.clear()

    bridge.handle({"type": "turn.started", "follow_up": True, "source": "browser", "turn_id": "t-2"})
    bridge.handle({"type": "transcript.partial", "text": "yes", "turn_id": "t-2"})

    assert hub.calls == []


def test_a_follow_up_turn_after_the_window_is_not_admitted() -> None:
    bridge, hub, clock = _admitted(follow_up_window_ms=lambda: 6800)
    bridge.handle(_ended(follow_up=True, asks_question=True, playback_ms_left=1000))
    hub.calls.clear()

    clock.now += 1.0 + 6.8 + 2.0 + 0.5  # past playback + window + slack
    bridge.handle({"type": "turn.started", "follow_up": True, "source": "camera", "turn_id": "t-2"})
    bridge.handle({"type": "transcript.partial", "text": "yes", "turn_id": "t-2"})

    assert hub.calls == []


def test_a_turn_started_without_the_follow_up_flag_is_not_admitted() -> None:
    bridge, hub, _ = _admitted()
    bridge.handle(_ended(follow_up=True, asks_question=True))
    hub.calls.clear()

    bridge.handle({"type": "turn.started", "follow_up": False, "source": "camera", "turn_id": "t-2"})
    bridge.handle({"type": "turn.started", "source": "camera", "turn_id": "t-3"})
    bridge.handle({"type": "transcript.partial", "text": "x", "turn_id": "t-2"})
    bridge.handle({"type": "transcript.partial", "text": "x", "turn_id": "t-3"})

    assert hub.calls == []


def test_a_turn_that_ends_without_a_follow_up_opens_no_window() -> None:
    bridge, hub, _ = _admitted()
    bridge.handle(_ended(follow_up=False))
    hub.calls.clear()

    bridge.handle({"type": "turn.started", "follow_up": True, "source": "camera", "turn_id": "t-2"})
    bridge.handle({"type": "transcript.partial", "text": "x", "turn_id": "t-2"})

    assert hub.calls == []


def test_bad_field_types_are_dropped_without_an_exception() -> None:
    bridge, hub, _ = _admitted()

    bridge.handle({"type": "transcript.partial", "text": 5, "turn_id": "t-1"})
    bridge.handle({"type": "transcript.final", "text": None, "turn_id": "t-1"})
    bridge.handle({"type": "reply.started", "text": ["x"], "turn_id": "t-1"})
    bridge.handle(_ended(playback_ms_left=True))
    bridge.handle(_ended(playback_ms_left="7"))
    bridge.handle(_ended(outcome=3))
    bridge.handle({"type": "transcript.partial", "text": "x", "turn_id": 7})

    assert hub.calls == []


def test_the_bridge_ignores_event_types_it_does_not_translate() -> None:
    bridge, hub, _ = _admitted()

    for event_type in ("reply.text", "reply.interrupted", "turn.timing", "wake.heard"):
        bridge.handle({"type": event_type, "text": "x", "turn_id": "t-1"})

    assert hub.calls == []


def test_the_bridge_logs_the_event_type_and_turn_id_never_the_text(caplog) -> None:
    bridge, _hub, _ = _admitted()

    with caplog.at_level("DEBUG"):
        bridge.handle({"type": "transcript.partial", "text": "SECRET-WORDS", "turn_id": "t-1"})
        bridge.handle({"type": "reply.started", "text": ["SECRET-WORDS"], "turn_id": "t-1"})
        bridge.handle(_ended(outcome="SECRET-WORDS", playback_ms_left="SECRET-WORDS"))

    assert "SECRET-WORDS" not in caplog.text
