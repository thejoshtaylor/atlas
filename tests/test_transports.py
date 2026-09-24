"""Real assertions for the browser-transport validation map.

`test_websocket_yields_pcm16` and `test_webrtc_yields_pcm16` assert against
the exact same `EXPECTED_PCM16_FRAMES` constant -- proof the two transports
are equivalent at the seam (D-02), not merely both present.
"""

import av
from aiortc import MediaStreamError
from fastapi import WebSocketDisconnect

# Two frames of 16 kHz mono PCM16 -- two int16 samples each. The WebSocket
# path receives these bytes directly, with no decode step. The WebRTC path
# receives them wrapped in an av.AudioFrame already at the seam's target
# rate/layout/format and must produce byte-identical output after its
# resample step -- proving the resample is a true no-op when the track
# already matches, not just "close enough".
EXPECTED_PCM16_FRAMES = [b"\x00\x00\x01\x00", b"\x02\x00\x03\x00"]


class _FakeWebSocket:
    """A minimal double for `starlette.websockets.WebSocket`.

    Only the three methods `WebSocketAudioSource` actually calls: receiving
    binary frames until the client disconnects, and sending audio/event
    frames back out.
    """

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent_bytes: list[bytes] = []
        self.sent_text: list[str] = []

    async def receive_bytes(self) -> bytes:
        if not self._frames:
            raise WebSocketDisconnect()
        return self._frames.pop(0)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)


async def test_websocket_yields_pcm16():
    from atlas.transports.websocket import WebSocketAudioSource

    ws = _FakeWebSocket(EXPECTED_PCM16_FRAMES)
    source = WebSocketAudioSource(ws)

    frames = [chunk async for chunk in source.frames()]

    assert frames == EXPECTED_PCM16_FRAMES


def test_websocket_declares_its_own_format():
    from atlas.transports.base import SourceFormat
    from atlas.transports.websocket import WebSocketAudioSource

    source = WebSocketAudioSource(_FakeWebSocket([]))

    assert source.source_format() == SourceFormat("pcm", 16000)


def _pcm16_audio_frame(data: bytes, *, sample_rate: int = 16000, layout: str = "mono") -> av.AudioFrame:
    """Build an `av.AudioFrame` carrying `data` as raw interleaved s16 samples."""
    channels = 2 if layout == "stereo" else 1
    sample_size = 2 * channels
    frame = av.AudioFrame(format="s16", layout=layout, samples=len(data) // sample_size)
    frame.sample_rate = sample_rate
    frame.planes[0].update(data)
    return frame


class _FakeAudioTrack:
    """A minimal double for an aiortc inbound audio track.

    `recv()` replays a scripted list of `av.AudioFrame`s, then raises
    `MediaStreamError` -- the same signal a real aiortc track raises once
    it ends, so `WebrtcTransport`'s consumer loop exits exactly the way it
    would against the real thing.
    """

    kind = "audio"

    def __init__(self, frames):
        self._frames = list(frames)

    async def recv(self):
        if not self._frames:
            raise MediaStreamError()
        return self._frames.pop(0)


async def test_webrtc_yields_pcm16():
    from atlas.transports.webrtc import WebrtcTransport

    transport = WebrtcTransport()
    assert hasattr(transport, "frames")
    assert hasattr(transport, "send_audio")
    assert hasattr(transport, "send_event")

    track = _FakeAudioTrack([_pcm16_audio_frame(chunk) for chunk in EXPECTED_PCM16_FRAMES])
    transport._pc.emit("track", track)

    frames = [chunk async for chunk in transport.frames()]

    assert frames == EXPECTED_PCM16_FRAMES


def test_webrtc_declares_its_own_format():
    from atlas.transports.base import SourceFormat
    from atlas.transports.webrtc import WebrtcTransport

    transport = WebrtcTransport()

    assert transport.source_format() == SourceFormat("pcm", 16000)


async def test_webrtc_resamples_non_16khz_mono_frames():
    """A track that negotiated a different rate still leaves the transport
    at 16 kHz mono -- the seam's contract regardless of what Opus settled
    on, per the plan's behavior spec.

    The exact output sample count after a real resample is filter-latency
    dependent (swresample buffers internally rather than resampling
    sample-for-sample), so this asserts the total is close to the expected
    2x-upsample count rather than pinning an exact figure that would make
    this test change every time the resampler's internal filter does.
    """
    from atlas.transports.webrtc import WebrtcTransport

    transport = WebrtcTransport()
    frame_samples = 80
    frame_count = 20
    source_frames = [
        _pcm16_audio_frame(bytes(frame_samples * 2), sample_rate=8000) for _ in range(frame_count)
    ]
    track = _FakeAudioTrack(source_frames)
    transport._pc.emit("track", track)

    frames = [chunk async for chunk in transport.frames()]

    total_samples = sum(len(chunk) for chunk in frames) // 2
    expected_samples = frame_samples * frame_count * 2  # 8kHz -> 16kHz doubles the count
    assert abs(total_samples - expected_samples) <= 64  # swresample's filter warm-up latency
