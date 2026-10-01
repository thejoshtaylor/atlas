"""Phase 15 tracer: a wake the server confirms reaches every connected Mac as
`wake.confirmed`, and a wake it does not confirm reaches none (PANEL-01, D-05,
D-12, D-13).

The first test runs a real `run_turn` through a real `ObserverPublishingSource`,
`ObserverRegistry`, `DesktopEventBridge` and `DesktopHub`, with two fake Mac
sockets. The second boots the real application and checks that the lifespan
wires the bridge to `/ws/desktop`. Token literals stay under 8 characters.
"""

from __future__ import annotations

import asyncio
import json

from test_desktop_tracer import _boot_with_desktop_repo, _create_admin, _fixture_message

from atlas.desktop.bridge import DesktopEventBridge
from atlas.desktop.hub import DesktopHub
from atlas.providers.base import FinalTranscript, PartialTranscript
from atlas.session.observers import ObserverPublishingSource, ObserverRegistry
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from tests.conftest import BrainReply
from tests.desktop_fakes import FakeDesktopSocket, fake_desktop_device

HELLO = _fixture_message("hello.json")
WAKE_FIXTURE = _fixture_message("wake_confirmed.json")


class _EventSource:
    """A source that takes events and drops them, like a camera source."""

    def __init__(self) -> None:
        self.sent_audio: list[bytes] = []

    async def frames(self):
        yield b"\x2a" * 8

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    async def send_event(self, event: dict) -> None:
        return None

    def source_format(self):
        from atlas.transports.base import SourceFormat

        return SourceFormat("pcm", 16000)


class _Stt:
    def __init__(self, events) -> None:
        self._events = list(events)

    async def stream(self, frames, source_format=None):
        async for _ in frames:
            pass
        for event in self._events:
            yield event


class _Brain:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools=None, response_format=None):
        self.calls += 1
        return BrainReply(text="it is noon")


class _Tts:
    async def synthesize(self, text_deltas, sink=None):
        async for _ in text_deltas:
            pass
        yield b"\x01\x02"


async def _until(predicate, *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true within the timeout")


async def _connect_mac(hub: DesktopHub, device_id: int):
    socket = FakeDesktopSocket()
    socket.push_text(json.dumps(HELLO))
    task = asyncio.create_task(hub.serve(socket, fake_desktop_device(id=device_id)))
    await _until(lambda: bool(socket.sent) and socket.sent_types()[0] == "hello.ack")
    return socket, task


async def _run_wake_turn(registry: ObserverRegistry, events, *, verify_wake: bool) -> TurnTimings:
    timings = TurnTimings()
    source = ObserverPublishingSource(_EventSource(), "camera", registry, turn_id=timings.turn_id)
    await run_turn(
        source,
        _Stt(events),
        _Brain(),
        _Tts(),
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        wake_phrase="hey atlas",
        verify_wake=verify_wake,
    )
    return timings


def _wake_frames(socket: FakeDesktopSocket) -> list[dict]:
    return [frame for frame in socket.sent_json() if frame["type"] == "wake.confirmed"]


async def test_a_confirmed_wake_reaches_every_connected_mac_and_an_unconfirmed_one_reaches_none() -> None:
    registry = ObserverRegistry()
    hub = DesktopHub(hello_timeout_s=0.5, ping_timeout_s=0.1, idle_timeout_s=2.0)
    bridge = DesktopEventBridge(registry, hub, follow_up_window_ms=lambda: 8800)
    bridge.start()
    first, first_task = await _connect_mac(hub, 1)
    second, second_task = await _connect_mac(hub, 2)
    try:
        timings = await _run_wake_turn(
            registry,
            [
                PartialTranscript(text="hey atlas"),
                PartialTranscript(text="hey atlas what time"),
                FinalTranscript(text="hey atlas what time is it"),
            ],
            verify_wake=False,
        )
        await _until(lambda: len(_wake_frames(first)) >= 1 and len(_wake_frames(second)) >= 1)
        await asyncio.sleep(0.05)

        for socket in (first, second):
            [frame] = _wake_frames(socket)
            assert frame["turn_id"] == timings.turn_id
            assert set(frame) == set(WAKE_FIXTURE)
        sent_before = (len(first.sent), len(second.sent))

        # A television sentence: the detector fired, the transcript did not
        # open with the wake phrase, so nothing may reach a Mac.
        await _run_wake_turn(
            registry,
            [FinalTranscript(text="You always say AM in the morning.")],
            verify_wake=True,
        )
        await asyncio.sleep(0.05)
        assert (len(first.sent), len(second.sent)) == sent_before
    finally:
        await bridge.stop()
        first.push_disconnect()
        second.push_disconnect()
        await asyncio.gather(first_task, second_task)


def test_the_app_bridge_forwards_a_published_wake_to_a_paired_mac(tmp_path, monkeypatch) -> None:
    client, _repo = _boot_with_desktop_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        created = client.post("/api/desktop-devices", json={"name": "Test Mac"})
        assert created.status_code == 201, created.text
        token = created.json()["token"]

        with client.websocket_connect(
            "/ws/desktop", headers={"Authorization": f"Bearer {token}"}
        ) as socket:
            socket.send_text(json.dumps(HELLO))
            assert json.loads(socket.receive_text())["type"] == "hello.ack"

            registry = client.app.state.observer_registry

            async def publish() -> None:
                # The registry is not thread safe: publish on the app's loop.
                registry.publish({"type": "transcript.partial", "text": "hey", "turn_id": "t-other"})
                registry.publish({"type": "wake.confirmed", "source": "camera", "turn_id": "t-wake"})

            client.portal.call(publish)
            frame = json.loads(socket.receive_text())

        assert frame == {"type": "wake.confirmed", "turn_id": "t-wake"}
