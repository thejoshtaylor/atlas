"""RED for Capture/find_device/DeviceNotFound (10-08-PLAN.md Task 3),
including the orchestrator-required full-duplex playback stream
(10-SPIKE.md hardware finding 1). No test opens a real audio device --
`sounddevice.query_devices` is monkeypatched to a fake device list, and
every stream is the injected fake factory below.
"""

from __future__ import annotations

import asyncio

import pytest
import sounddevice as sd

from atlas_edge.capture import Capture, DeviceNotFound, find_device

_FAKE_DEVICES = [
    {"name": "SAMSUNG", "max_input_channels": 0, "max_output_channels": 6},
    {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0},
    {
        "name": "reSpeaker XVF3800 4-Mic Array: USB Audio (hw:3,0)",
        "max_input_channels": 2,
        "max_output_channels": 2,
    },
]


@pytest.fixture(autouse=True)
def _fake_query_devices(monkeypatch):
    monkeypatch.setattr(sd, "query_devices", lambda: _FAKE_DEVICES)


class FakeStream:
    def __init__(self, kind, events, kwargs):
        self.kind = kind
        self._events = events
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start(self):
        self.started = True
        self._events.append(("start", self.kind))

    def stop(self):
        self.started = False
        self._events.append(("stop", self.kind))

    def close(self):
        self.closed = True
        self._events.append(("close", self.kind))


def _fake_factory(events, streams):
    def factory(kind, **kwargs):
        events.append(("open", kind, dict(kwargs)))
        stream = FakeStream(kind, events, kwargs)
        streams[kind] = stream
        return stream

    return factory


# --- find_device / DeviceNotFound -----------------------------------------


def test_find_device_returns_first_matching_input_device():
    index = find_device("reSpeaker", kind="input")
    assert index == 2


def test_find_device_matches_case_insensitively():
    assert find_device("respeaker", kind="input") == 2


def test_find_device_raises_device_not_found_listing_names():
    with pytest.raises(DeviceNotFound) as exc_info:
        find_device("nonexistent-device", kind="input")
    message = str(exc_info.value)
    for name in ("SAMSUNG", "MacBook Pro Microphone", "reSpeaker"):
        assert name in message


def test_find_device_never_falls_back_to_the_default_device(monkeypatch):
    # No `sd.default` attribute is ever read by find_device -- reading it
    # here would raise, proving the function never touches it.
    class _ExplodingDefault:
        def __getattr__(self, name):
            raise AssertionError("find_device must never read sd.default")

    monkeypatch.setattr(sd, "default", _ExplodingDefault())
    assert find_device("reSpeaker", kind="input") == 2


# --- Capture: input side ----------------------------------------------------


@pytest.mark.asyncio
async def test_capture_opens_raw_input_stream_with_expected_params():
    events: list = []
    streams: dict = {}
    capture = Capture("reSpeaker", stream_factory=_fake_factory(events, streams))
    capture.start()

    input_kwargs = streams["input"].kwargs
    assert input_kwargs["samplerate"] == 16000
    assert input_kwargs["channels"] == 2
    assert input_kwargs["dtype"] == "int16"
    assert input_kwargs["blocksize"] == 256
    assert input_kwargs["device"] == 2  # the fake reSpeaker index


@pytest.mark.asyncio
async def test_input_callback_enqueues_bytes_and_frames_yields_them(monkeypatch):
    import atlas_edge.capture as capture_module

    monkeypatch.setattr(capture_module.time, "monotonic", lambda: 100.0)
    events: list = []
    streams: dict = {}
    capture = Capture("reSpeaker", stream_factory=_fake_factory(events, streams))
    capture.start()

    indata = bytes(range(20)) * 26  # 520 bytes, arbitrary content
    capture._input_callback(indata, 256, None, None)

    frame, captured_at = await asyncio.wait_for(capture.frames().__anext__(), timeout=1.0)
    assert frame == bytes(indata)
    # captured_at = callback-time monotonic (100.0) minus one block (16ms).
    assert captured_at == pytest.approx(100.0 - 256 / 16000)


@pytest.mark.asyncio
async def test_full_queue_drops_oldest_frame_and_counts_the_drop():
    events: list = []
    streams: dict = {}
    capture = Capture("reSpeaker", stream_factory=_fake_factory(events, streams), max_queued=3)
    capture.start()

    for i in range(4):
        capture._input_callback(bytes([i]) * 4, 256, None, None)
        await asyncio.sleep(0)  # let call_soon_threadsafe's callback run

    assert capture.dropped == 1
    remaining = [f for f, _ in capture._queue]
    assert remaining == [bytes([1]) * 4, bytes([2]) * 4, bytes([3]) * 4]


# --- Capture: full-duplex playback (orchestrator directive) ---------------


@pytest.mark.asyncio
async def test_output_stream_opens_on_same_device_and_starts_before_input():
    events: list = []
    streams: dict = {}
    capture = Capture("reSpeaker", stream_factory=_fake_factory(events, streams))
    capture.start()

    open_events = [e for e in events if e[0] == "open"]
    assert open_events[0][1] == "output"
    assert open_events[1][1] == "input"
    assert open_events[0][2]["device"] == open_events[1][2]["device"] == 2

    start_events = [e for e in events if e[0] == "start"]
    assert start_events == [("start", "output"), ("start", "input")]


@pytest.mark.asyncio
async def test_stop_stops_input_before_output():
    events: list = []
    streams: dict = {}
    capture = Capture("reSpeaker", stream_factory=_fake_factory(events, streams))
    capture.start()
    events.clear()
    capture.stop()

    stop_events = [e for e in events if e[0] == "stop"]
    assert stop_events == [("stop", "input"), ("stop", "output")]


@pytest.mark.asyncio
async def test_output_callback_writes_zeros_when_playback_queue_is_empty():
    events: list = []
    streams: dict = {}
    capture = Capture("reSpeaker", stream_factory=_fake_factory(events, streams))
    capture.start()

    frames = 256
    needed = frames * 2 * 2  # channels * bytes/sample
    outdata = bytearray(b"\xff" * needed)
    capture._output_callback(outdata, frames, None, None)
    assert bytes(outdata) == bytes(needed)


@pytest.mark.asyncio
async def test_enqueued_playback_bytes_come_out_of_the_output_callback_in_order():
    events: list = []
    streams: dict = {}
    capture = Capture("reSpeaker", stream_factory=_fake_factory(events, streams))
    capture.start()

    frames = 256
    needed = frames * 2 * 2
    first = bytes([1]) * (needed // 2)
    second = bytes([2]) * (needed // 2)
    capture.enqueue_playback(first)
    capture.enqueue_playback(second)

    outdata = bytearray(needed)
    capture._output_callback(outdata, frames, None, None)
    assert bytes(outdata) == first + second

    # A second callback with an empty queue now gets silence.
    outdata2 = bytearray(b"\xff" * needed)
    capture._output_callback(outdata2, frames, None, None)
    assert bytes(outdata2) == bytes(needed)


@pytest.mark.asyncio
async def test_playback_queue_leftover_is_carried_to_the_next_callback():
    events: list = []
    streams: dict = {}
    capture = Capture("reSpeaker", stream_factory=_fake_factory(events, streams))
    capture.start()

    frames = 256
    needed = frames * 2 * 2
    oversized = bytes(range(256)) * ((needed * 2) // 256 + 1)
    oversized = oversized[: needed + 10]  # 10 bytes more than one callback needs
    capture.enqueue_playback(oversized)

    outdata = bytearray(needed)
    capture._output_callback(outdata, frames, None, None)
    assert bytes(outdata) == oversized[:needed]

    outdata2 = bytearray(b"\xff" * needed)
    capture._output_callback(outdata2, frames, None, None)
    assert bytes(outdata2)[:10] == oversized[needed:]
    assert bytes(outdata2)[10:] == bytes(needed - 10)
