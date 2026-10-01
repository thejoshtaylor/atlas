"""`PanelOutbox`: partial coalescing and the priority drop (Phase 15, D-13).

A slow Mac loses detail (partials, states) before it loses an end
(`turn.ended`, `timer.stopped`). `put` stays synchronous.
"""

from __future__ import annotations

import inspect

from atlas.desktop.outbox import PanelOutbox


def _drain(outbox: PanelOutbox) -> list[str]:
    frames = []
    while len(outbox):
        frames.append(outbox._frames.popleft().frame)
    return frames


def _types(outbox: PanelOutbox) -> list[str]:
    return [queued.frame_type for queued in outbox._frames]


def test_put_is_not_a_coroutine() -> None:
    assert not inspect.iscoroutinefunction(PanelOutbox.put)


def test_two_partials_of_one_turn_in_a_row_leave_the_newer_one() -> None:
    outbox = PanelOutbox()

    outbox.put("p1", "transcript.partial", coalesce_key="A")
    outbox.put("p2", "transcript.partial", coalesce_key="A")

    assert _drain(outbox) == ["p2"]
    assert outbox.dropped == 0


def test_a_partial_of_another_turn_is_appended() -> None:
    outbox = PanelOutbox()

    outbox.put("a", "transcript.partial", coalesce_key="A")
    outbox.put("b", "transcript.partial", coalesce_key="B")

    assert _drain(outbox) == ["a", "b"]


def test_a_partial_after_the_final_of_the_same_turn_is_appended_in_order() -> None:
    outbox = PanelOutbox()

    outbox.put("p1", "transcript.partial", coalesce_key="A")
    outbox.put("f", "transcript.final", coalesce_key="A")
    outbox.put("p2", "transcript.partial", coalesce_key="A")

    assert _drain(outbox) == ["p1", "f", "p2"]


def test_overflow_drops_the_oldest_partials_before_anything_else() -> None:
    outbox = PanelOutbox(max_frames=8)
    outbox.put("end", "turn.ended")  # the oldest frame, and the one that must survive
    for turn in "ABC":
        outbox.put(f"p{turn}", "transcript.partial", coalesce_key=turn)
    for n in range(3):
        outbox.put(f"s{n}", "state")
    assert len(outbox) == 7

    outbox.put("s3", "state")
    outbox.put("s4", "state")  # nine frames offered, so one over the bound

    assert outbox.dropped == 1
    assert _types(outbox).count("transcript.partial") == 2
    outbox.put("s5", "state")
    assert outbox.dropped == 2
    assert _types(outbox).count("transcript.partial") == 1
    assert "turn.ended" in _types(outbox)
    assert len(outbox) == 8


def test_overflow_drops_state_then_final_then_card_in_that_order() -> None:
    outbox = PanelOutbox(max_frames=4)
    outbox.put("c", "card")
    outbox.put("f", "transcript.final")
    outbox.put("s", "state")
    outbox.put("end", "turn.ended")

    outbox.put("x", "wake.confirmed")
    assert _types(outbox) == ["card", "transcript.final", "turn.ended", "wake.confirmed"]
    outbox.put("y", "wake.confirmed")
    assert _types(outbox) == ["card", "turn.ended", "wake.confirmed", "wake.confirmed"]
    outbox.put("z", "wake.confirmed")
    assert _types(outbox) == ["turn.ended", "wake.confirmed", "wake.confirmed", "wake.confirmed"]
    assert outbox.dropped == 3


def test_only_protected_frames_left_drops_the_oldest_and_keeps_the_bound() -> None:
    outbox = PanelOutbox(max_frames=4)
    for name, kind in (
        ("w", "wake.confirmed"),
        ("e", "turn.ended"),
        ("r", "timer.ringing"),
        ("s", "timer.stopped"),
    ):
        outbox.put(name, kind)

    outbox.put("e2", "turn.ended")

    assert len(outbox) == 4
    assert outbox.dropped == 1
    assert _drain(outbox) == ["e", "r", "s", "e2"]


def test_an_unlisted_frame_type_is_dropped_after_the_detail_types_and_before_ends() -> None:
    outbox = PanelOutbox(max_frames=3)
    outbox.put("end", "turn.ended")
    outbox.put("other", "something.else")
    outbox.put("c", "card")

    outbox.put("end2", "turn.ended")

    assert _types(outbox) == ["turn.ended", "something.else", "turn.ended"]


async def test_get_returns_frames_first_in_first_out() -> None:
    outbox = PanelOutbox()
    outbox.put("one", "state")
    outbox.put("two", "card")

    assert await outbox.get() == "one"
    assert await outbox.get() == "two"
    assert len(outbox) == 0
