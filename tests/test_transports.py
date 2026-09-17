"""Real assertions for the browser-transport validation map.

`test_websocket_yields_pcm16` is turned green by plan 01-02.
`test_webrtc_yields_pcm16` stays a stub for plan 01-04.
"""

from fastapi import WebSocketDisconnect


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
    from spire_voice.transports.websocket import WebSocketAudioSource

    ws = _FakeWebSocket([b"\x00\x01", b"\x02\x03"])
    source = WebSocketAudioSource(ws)

    frames = [chunk async for chunk in source.frames()]

    assert frames == [b"\x00\x01", b"\x02\x03"]


def test_webrtc_yields_pcm16():
    raise AssertionError("not implemented: SRC-01")
