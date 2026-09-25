"""RED for SendDelayWindow (10-09-PLAN.md Task 1)."""

from __future__ import annotations

import json

import pytest

from atlas_edge.latency import SendDelayWindow


def test_drain_message_returns_none_for_an_empty_window() -> None:
    window = SendDelayWindow()
    assert window.drain_message() is None


def test_drain_message_reports_p50_p95_max_and_frame_count() -> None:
    window = SendDelayWindow()
    # captured_at/sent_at deltas, in seconds: 0.010, 0.020, ..., 0.100 (10 frames)
    for i in range(1, 11):
        window.record(captured_at=0.0, sent_at=i * 0.010)

    message = window.drain_message()
    assert message is not None
    payload = json.loads(message)

    assert payload["type"] == "latency"
    assert payload["frames"] == 10
    assert payload["capture_to_send_ms_max"] == pytest.approx(100.0)
    # p50 of 10..100 (step 10) is 55.0, p95 is 95.5 -- numpy's default
    # linear interpolation over this exact fixture.
    assert payload["capture_to_send_ms_p50"] == pytest.approx(55.0)
    assert payload["capture_to_send_ms_p95"] == pytest.approx(95.5)


def test_drain_message_resets_the_window() -> None:
    window = SendDelayWindow()
    window.record(captured_at=0.0, sent_at=0.010)
    first = window.drain_message()
    assert first is not None

    # Nothing recorded since the first drain -- the second drain reports
    # an empty window, not a stale echo of the first.
    assert window.drain_message() is None


def test_a_single_frame_window_reports_that_frame_at_every_percentile() -> None:
    window = SendDelayWindow()
    window.record(captured_at=1.0, sent_at=1.025)  # 25ms delay

    payload = json.loads(window.drain_message())
    assert payload["frames"] == 1
    assert payload["capture_to_send_ms_p50"] == pytest.approx(25.0)
    assert payload["capture_to_send_ms_p95"] == pytest.approx(25.0)
    assert payload["capture_to_send_ms_max"] == pytest.approx(25.0)
