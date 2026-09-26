"""The phase tracer (10-02-PLAN.md Task 1): a real Pi-side client, a real
device token, and one spoken command against the real application booted
in a real `uvicorn.Server` -- the wake word fires on the server from the
ASR channel, speech-to-text hears only that channel, and the reply comes
back on the same socket.

Every layer on the wire path is real: `atlas_edge.client.run_session` (the
Pi side), the token check (`require_edge_device`), `EdgeAudioSource`, the
channel selector (`atlas.audio.channels`), `SourceRunner`, `run_turn`, and
`send_audio`. Only the wake engine, the brain, and the two provider slots
are fakes -- the same class of substitution `tests/test_startup_smoke.py`
already makes for a startup-path test, extended here to a path that also
runs one real turn.
"""

from __future__ import annotations

import asyncio
import socket
import struct
from datetime import datetime, timezone

import pytest
import uvicorn
import websockets.exceptions

from atlas.audio.channels import select_channel
from atlas.auth.edge_tokens import hash_edge_token
from atlas.db.repository import Setting
from atlas.transports.base import SourceFormat
from atlas.turn.brain_race import TierBrain
from atlas_edge.client import run_session
from atlas_edge.protocol import Hello, vad_end, vad_start

from tests import test_startup_smoke as smoke
from tests.conftest import BrainReply, FakeTts, FakeWakeDetector, FinalTranscript
from tests.edge_fakes import FakeEdgeDeviceRepository, fake_edge_device, interleave


def _free_port() -> int:
    """A loopback TCP port free at the instant this returns -- the same
    bind-to-0-then-close trick `tests/test_plugin_remote_transport.py`
    already uses."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _repositories_with_edge_device(config: object, engine: object) -> dict:
    """`smoke._fake_build_repositories`, plus the two things this tracer
    needs that no other smoke test does: the `audio_source` setting
    resolved to `"edge"`, and a device repository holding one active
    device for token `"t-1"`."""
    repositories = smoke._fake_build_repositories(config, engine)
    settings_repo = repositories["settings_repo"]
    settings_repo.settings["audio_source"] = Setting(
        id=1,
        key="audio_source",
        value="edge",
        updated_at=datetime.now(timezone.utc),
        updated_by_user_id=None,
    )
    edge_device_repo = FakeEdgeDeviceRepository()
    edge_device_repo.add(hash_edge_token("t-1"), fake_edge_device(device_id=1))
    repositories["edge_device_repo"] = edge_device_repo
    return repositories


def _fake_build_edge_wake_detector(wake_config: object) -> FakeWakeDetector:
    """Fires on the 4th chunk it is handed (index 3) -- almost the whole
    scripted stream is left for the speech-to-text double to receive,
    matching `tests/test_room_tracer.py`'s own `fire_at_call` precedent."""
    return FakeWakeDetector(fire_at_call=3)


class _EdgeTracerBrain:
    """A brain double with both halves a real turn needs: `resolve_model()`
    (called once at boot, `test_startup_smoke._FakeResolvingBrain`'s own
    job) and `chat()` (called during the one real turn this tracer runs,
    which that fake has no method for at all). Replies with plain text and
    no tool calls, the same shape `tests/test_room_tracer.py`'s `FakeBrain`
    scripts for its own tracer.
    """

    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text
        self.chat_calls = 0

    async def resolve_model(self) -> str:
        return "fake-model"

    async def chat(self, messages, tools=None):
        self.chat_calls += 1
        return BrainReply(text=self._reply_text)


def _fake_build_tiers_with_working_brain(brain_config: object) -> "tuple[TierBrain, ...]":
    return (
        TierBrain(
            index=0,
            model="fake-model",
            brain=_EdgeTracerBrain("the light is on"),
            envelope_client=None,
            calls_tools=True,
        ),
    )


def _only_channel_values(chunk: bytes) -> "set[int]":
    """Every distinct int16 sample value in one de-interleaved,
    single-channel chunk -- what the detector/speech-to-text-received
    assertions check against `{-2000, -3000}` (channel 1 only, never the
    1000 channel 0 ever carried)."""
    return set(struct.unpack(f"<{len(chunk) // 2}h", chunk))


class _ChannelAwareStt:
    """Replaces `app.state.stt`: records the `SourceFormat` and every
    frame `_drain_to_final_transcript` hands it, and stops draining the
    instant it sees the marker frame (every sample equal to the fixed
    marker value) -- proving D-09's "speech-to-text reads only the ASR
    channel" without needing a real xAI/faster-whisper socket.
    """

    def __init__(self, marker_chunk: bytes) -> None:
        self.received_frames: list[bytes] = []
        self.received_format: "SourceFormat | None" = None
        self._marker_chunk = marker_chunk

    async def stream(self, frames, source_format, *, finalize=None):
        # 10-05-PLAN.md: the edge source's real `SpeechSignals` now reaches
        # `run_turn` through `speech_signals` forwarding, so
        # `_drain_to_final_transcript` calls `stream(..., finalize=...)`
        # here exactly as it would against a real provider -- accepted and
        # ignored, since this double proves D-09's channel isolation, not
        # the early-finalize path (10-05-SUMMARY.md's own coverage of that
        # is `tests/test_early_finalize.py`).
        self.received_format = source_format
        async for chunk in frames:
            self.received_frames.append(chunk)
            if chunk == self._marker_chunk:
                break
        yield FinalTranscript(text="turn on the light")


async def test_one_spoken_command_from_a_pi_shaped_client_reaches_a_reply_on_the_same_socket(
    tmp_path, monkeypatch
):
    import atlas.app as app_module
    from atlas.plugins import manager as plugin_manager_module

    monkeypatch.setenv("ATLAS_SECRET_KEY", smoke._TEST_SECRET_KEY)
    session_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(
            smoke._write_fake_config(
                tmp_path,
                extra={
                    "edge": {"asr_channel": 1, "pre_roll_ms": 200, "tail_ms": 300},
                    "debug": {"dir": str(session_dir)},
                    # `EdgeAudioSource.sink_format()` is real (unlike the
                    # browser listener's, which is absent), so the wake
                    # cue would otherwise play through `send_audio` before
                    # the reply -- turned off here so `received_reply`
                    # holds exactly the FakeTts bytes this test asserts
                    # against, nothing more.
                    "wake": {"cue": False},
                },
            )
        ),
    )
    monkeypatch.setattr(plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _repositories_with_edge_device)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers_with_working_brain)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_edge_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app_module.app, host="127.0.0.1", port=port, log_level="error"))
    server_task = asyncio.create_task(server.serve())
    try:
        for _ in range(500):
            if server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError("test server did not start listening in time")

        # Post-boot doubles: `_fake_build_tiers_with_working_brain` and
        # `_build_repositories`/`_build_wake_detector` above are read at
        # boot; `stt`/`tts` are replaced here, after boot, the same
        # "swap the slot after lifespan has already wired everything
        # else" shape `tests/test_startup_smoke.py`'s own precache
        # substitution uses.
        marker_frame = interleave(1000, -3000, 256)
        marker_channel_chunk = select_channel(marker_frame, 2, 1)
        stt_double = _ChannelAwareStt(marker_channel_chunk)
        app_module.app.state.stt = stt_double

        reply_chunks = [b"edge-reply-chunk-one", b"edge-reply-chunk-two"]
        expected_reply = b"".join(reply_chunks)
        fake_tts = FakeTts(chunks=reply_chunks)
        app_module.app.state.tts = fake_tts

        assert [r._name for r in app_module.app.state.source_runners] == ["edge"]

        received_hello: list[Hello] = []
        received_reply = bytearray()
        done = asyncio.Event()

        async def _make_outbound(hello: Hello):
            received_hello.append(hello)
            yield vad_start(1)
            for _ in range(20):
                yield interleave(1000, -2000, 256)
            yield marker_frame
            yield vad_end(2)

        def _on_reply_audio(chunk: bytes) -> None:
            received_reply.extend(chunk)
            if bytes(received_reply) == expected_reply:
                done.set()

        await asyncio.wait_for(
            run_session(
                f"ws://127.0.0.1:{port}/ws/edge",
                "t-1",
                make_outbound=_make_outbound,
                on_reply_audio=_on_reply_audio,
                allow_plaintext=True,
                stop=done,
            ),
            timeout=15.0,
        )

        # 1. The hello carries the measured configuration, never a
        # default the Pi would have to guess.
        assert len(received_hello) == 1
        hello = received_hello[0]
        assert hello.asr_channel == 1
        assert hello.pre_roll_ms == 200
        assert hello.tail_ms == 300

        # 2. The detector and the speech-to-text double saw only channel
        # 1's samples (-2000/-3000), never channel 0's 1000.
        wake_detector = app_module.app.state.wake_detector.inner  # unwrap SegmentBoundedWakeDetector
        assert wake_detector.chunks, "the wake detector never received a chunk"
        assert _only_channel_values(b"".join(wake_detector.chunks)) <= {-2000, -3000}
        assert stt_double.received_frames, "speech-to-text never received a frame"
        assert _only_channel_values(b"".join(stt_double.received_frames)) <= {-2000, -3000}

        # 3. Speech-to-text received a one-channel format.
        assert stt_double.received_format == SourceFormat("pcm", 16000)

        # 4. The client's reply-audio callback received the FakeTts bytes.
        assert bytes(received_reply) == expected_reply

        # 6. A second connection with an unknown token fails at the
        # handshake -- `require_edge_device` refuses before `accept()` is
        # ever called, so the ASGI server denies the opening handshake at
        # the HTTP level (403) rather than accepting and then closing --
        # the route body (`edge_ws`, which would call `serve()`) never ran.
        async def _no_outbound(_hello: Hello):
            return
            yield  # pragma: no cover -- makes this an async generator

        with pytest.raises(websockets.exceptions.InvalidStatus) as exc_info:
            await run_session(
                f"ws://127.0.0.1:{port}/ws/edge",
                "bad",
                make_outbound=_no_outbound,
                on_reply_audio=lambda chunk: None,
                allow_plaintext=True,
            )
        assert exc_info.value.response.status_code == 403
    finally:
        server.should_exit = True
        await server_task
