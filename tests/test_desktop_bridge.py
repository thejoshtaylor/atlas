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
from atlas.session.observers import ObserverRegistry


class _RecordingHub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def broadcast(self, frame: str, *, frame_type: str, coalesce_key=None) -> int:
        self.calls.append((frame, frame_type))
        return 1


def _bridge(hub=None, registry=None, **overrides) -> tuple[DesktopEventBridge, _RecordingHub]:
    hub = hub or _RecordingHub()
    bridge = DesktopEventBridge(
        registry or ObserverRegistry(), hub, follow_up_window_ms=lambda: 8800, **overrides
    )
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
