"""The Phase 11 tracer (11-04-PLAN.md Task 1): a real application, a real
Pi-side client, an enrolled voice answered, an unenrolled voice's turn
stopped in silence.

Modeled line for line on `tests/test_edge_tracer.py` -- the same real
`uvicorn.Server`, the same real `atlas_edge.client.run_session`, the same
class of monkeypatched builders (wake detector, brain tiers, repositories).
This tracer adds exactly one more real layer on top: `speaker_id.wiring.
build_speaker_context` and the live `SpeakerTracker`/`evaluate_turn_speaker`
gate, with only the embedding model itself faked (`tests/speaker_fakes.py
::FakeEmbedder`) -- sherpa-onnx needs a real model file this repository
does not carry.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import time
from pathlib import Path

import pytest
import uvicorn

from atlas.audio.channels import select_channel
from atlas.auth.edge_tokens import hash_edge_token
from atlas.config import SPEAKER_MODEL_FILES
from atlas.db.repository import Setting
from atlas.db.speaker_repository import ReferenceEmbedding
from atlas.transports.base import SourceFormat
from atlas.turn.brain_race import TierBrain
from atlas_edge.client import run_session
from atlas_edge.protocol import Hello, vad_end, vad_start

from tests import test_edge_tracer, test_startup_smoke as smoke
from tests.conftest import BrainReply, FakeTts, FakeWakeHit, FinalTranscript
from tests.edge_fakes import FakeEdgeDeviceRepository, fake_edge_device, interleave
from tests.speaker_fakes import FakeEmbedder
from tests.speaker_repo_fakes import FakeSpeakerRepository, fake_speaker

_CAMPPLUS_MODEL_ID = SPEAKER_MODEL_FILES["campplus"][: -len(".onnx")]

# The marker frame's ASR-channel value -- never embedded (this test's own
# speech-frame counts leave the trailing partial window under
# `min_window_ms`, so the marker's own 16ms never reaches the accumulator's
# flush threshold at `vad.end`), only a stop signal the STT double
# recognizes so it can finalize without needing `EdgeAudioSource.frames()`
# to end (it never does, mid-connection).
_MARKER_VALUE = -9000


def _reference_vector_for(value: int) -> "list[float]":
    """The exact vector a fresh `FakeEmbedder` produces for a constant
    PCM16 buffer at `value` -- deterministic across every `FakeEmbedder`
    instance (seeded by bucket, `tests/speaker_fakes.py`), so this can be
    computed once, here, and compared against what the *live* pipeline's
    own, separately-constructed `FakeEmbedder` produces for real speech at
    the same value."""
    buffer = struct.pack("<250h", *([value] * 250))
    return list(FakeEmbedder().embed(buffer))


class _RepeatingEdgeWakeDetector:
    """Fires once per segment, on the first frame after each `vad.start`.

    `tests/conftest.py::FakeWakeDetector` fires exactly once across a
    whole fixture stream, which this tracer's two separate turns (two
    separate segments on the one long-lived `SourceRunner`) cannot share.
    `reset()` -- absent from `FakeWakeDetector` -- is what
    `SegmentBoundedWakeDetector` calls on the first frame of every new
    segment, so this class implements it, unlike its conftest sibling.
    """

    def __init__(self) -> None:
        self._fired_this_segment = False
        self.chunks: "list[bytes]" = []
        self.closed = False

    def process(self, chunk: bytes) -> "FakeWakeHit | None":
        self.chunks.append(chunk)
        if self._fired_this_segment:
            return None
        self._fired_this_segment = True
        return FakeWakeHit(score=1.0)

    def reset(self) -> None:
        self._fired_this_segment = False

    def close(self) -> None:
        self.closed = True


def _fake_build_edge_wake_detector(wake_config: object) -> _RepeatingEdgeWakeDetector:
    return _RepeatingEdgeWakeDetector()


def _fake_build_tiers_with_working_brain(brain_config: object) -> "tuple[TierBrain, ...]":
    return (
        TierBrain(
            index=0,
            model="fake-model",
            brain=test_edge_tracer._EdgeTracerBrain("the light is on"),
            envelope_client=None,
            calls_tools=True,
        ),
    )


def _repositories_with_edge_device_and_speaker(config: object, engine: object) -> dict:
    """`test_edge_tracer._repositories_with_edge_device`, plus a
    `FakeSpeakerRepository` seeded with one enrolled member ("Member A",
    id 1) holding five phrase embeddings under the campplus model id --
    each equal to `FakeEmbedder`'s own vector for a constant -2000 buffer
    (D-06's "one embedding per phrase" simplified to one repeated value,
    since matching only reads the mean of the enrolled vectors)."""
    repositories = test_edge_tracer._repositories_with_edge_device(config, engine)
    speaker_repo = FakeSpeakerRepository()
    speaker_repo._speakers[1] = fake_speaker(speaker_id=1, display_name="Member A")
    speaker_repo._next_id = 2
    reference_vector = _reference_vector_for(-2000)
    for phrase_index in range(5):
        speaker_repo._embeddings[(1, phrase_index, _CAMPPLUS_MODEL_ID)] = ReferenceEmbedding(
            speaker_id=1,
            display_name="Member A",
            phrase_index=phrase_index,
            model_id=_CAMPPLUS_MODEL_ID,
            vector=reference_vector,
        )
    repositories["speaker_repo"] = speaker_repo
    return repositories


def _only_channel_values(chunks: "list[bytes]") -> "set[int]":
    """Every distinct int16 sample value across `chunks` (already
    channel-selected, mono) -- what this test's own channel-isolation
    assertions check against the expected ASR-channel value set, never
    channel 0's fixed 1000."""
    values: "set[int]" = set()
    for chunk in chunks:
        values.update(struct.unpack(f"<{len(chunk) // 2}h", chunk))
    return values


async def _wait_for_session_dirs(session_dir: Path, expected_count: int, *, timeout: float = 5.0) -> "list[Path]":
    """Poll `session_dir` until at least `expected_count` subdirectories
    exist and the `expected_count`-th one (sorted by name -- a UTC
    timestamp prefix, `session/recorder.py::_directory_name`) has both
    `events.jsonl` and a non-empty `timing.json` on disk.

    `SessionRecorder.close()` (the only place either file is actually
    written) runs from `run_turn`'s own `finally` block, strictly after
    this test's own `on_reply_audio`/`stop` callback can fire (turn 1) or
    strictly after the fixed 3s wait ends (turn 2) -- but both are
    separate asyncio tasks, so a short poll, not a single synchronous
    check, is what makes this deterministic rather than racy.
    """
    deadline = time.monotonic() + timeout
    while True:
        dirs = sorted(p for p in session_dir.iterdir() if p.is_dir())
        if len(dirs) >= expected_count:
            candidate = dirs[expected_count - 1]
            timing_path = candidate / "timing.json"
            events_path = candidate / "events.jsonl"
            if events_path.exists() and timing_path.exists() and timing_path.stat().st_size > 0:
                return dirs
        if time.monotonic() > deadline:
            raise AssertionError(
                f"expected {expected_count} complete session directories under {session_dir} "
                f"within {timeout}s, found {len(dirs)}"
            )
        await asyncio.sleep(0.05)


def _read_events(directory: Path) -> "list[dict]":
    lines = (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_timing(directory: Path) -> dict:
    return json.loads((directory / "timing.json").read_text(encoding="utf-8"))


def _find_speaker_result(events: "list[dict]") -> dict:
    for event in events:
        if event.get("type") == "speaker.result":
            return event
    raise AssertionError(f"no speaker.result event found among {[e.get('type') for e in events]}")


async def test_an_enrolled_voice_is_answered_and_an_unenrolled_voice_is_silenced(
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
                    "wake": {"cue": False, "refractory_s": 0},
                    "speaker_id": {
                        "mode": "enforce",
                        "model": "campplus",
                        "threshold": 0.5,
                        "window_ms": 500,
                        "min_window_ms": 250,
                        "speech_rms_floor": 0.01,
                        "change_similarity_floor": 0.3,
                    },
                },
            )
        ),
    )
    monkeypatch.setattr(plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _repositories_with_edge_device_and_speaker)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", _fake_build_tiers_with_working_brain)
    monkeypatch.setattr(app_module, "_build_wake_detector", _fake_build_edge_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)

    # This tracer proves the wake-turn speaker gate. Its second utterance is a
    # new wake turn, sent in a burst right after the first reply, so it must
    # not land in the no-wake-word answer window (13-04) that the first
    # answer would otherwise open.
    real_make_run_turn = app_module._make_run_turn_for_source
    monkeypatch.setattr(
        app_module,
        "_make_run_turn_for_source",
        lambda *args, **kwargs: real_make_run_turn(*args, **{**kwargs, "answer_windows": False}),
    )

    created_embedders: "list[FakeEmbedder]" = []

    def _fake_build_speaker_embedder(speaker_config: object) -> FakeEmbedder:
        embedder = FakeEmbedder()
        created_embedders.append(embedder)
        return embedder

    monkeypatch.setattr(app_module, "_build_speaker_embedder", _fake_build_speaker_embedder)

    port = test_edge_tracer._free_port()
    server = uvicorn.Server(uvicorn.Config(app_module.app, host="127.0.0.1", port=port, log_level="error"))
    server_task = asyncio.create_task(server.serve())
    try:
        for _ in range(500):
            if server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError("test server did not start listening in time")

        marker_frame = interleave(1000, _MARKER_VALUE, 256)
        marker_channel_chunk = select_channel(marker_frame, 2, 1)
        stt_double = test_edge_tracer._ChannelAwareStt(marker_channel_chunk)
        app_module.app.state.stt = stt_double

        reply_chunks = [b"speaker-reply-chunk-one", b"speaker-reply-chunk-two"]
        expected_reply = b"".join(reply_chunks)
        fake_tts = FakeTts(chunks=reply_chunks)
        app_module.app.state.tts = fake_tts

        assert [r._name for r in app_module.app.state.source_runners] == ["edge"]
        assert app_module.app.state.speaker_id_context is not None
        assert app_module.app.state.speaker_id_context.tracker is not None

        # -- Turn 1: an enrolled voice (-2000) gets its reply -----------

        received_hello: "list[Hello]" = []
        received_reply = bytearray()
        done = asyncio.Event()

        async def _make_outbound_enrolled(hello: Hello):
            received_hello.append(hello)
            yield vad_start(1)
            for _ in range(63):  # 63 * 16ms = 1008ms -- at least 1s (D-09).
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
                make_outbound=_make_outbound_enrolled,
                on_reply_audio=_on_reply_audio,
                allow_plaintext=True,
                stop=done,
            ),
            timeout=15.0,
        )

        assert len(received_hello) == 1
        assert bytes(received_reply) == expected_reply

        session_dirs = await _wait_for_session_dirs(session_dir, 1)
        turn1_events = _read_events(session_dirs[0])
        turn1_result = _find_speaker_result(turn1_events)
        assert turn1_result["status"] == "identified"
        assert turn1_result["speaker_id"] == 1
        assert turn1_result["speaker_name"] == "Member A"
        assert turn1_result["score"] is not None and turn1_result["score"] > 0.5
        assert turn1_result["blocked"] is False

        turn1_timing = _read_timing(session_dirs[0])
        assert turn1_timing["turn_outcome"] == "completed"

        # -- Turn 2: a new session, an unenrolled voice (-4000) --------

        received_reply2 = bytearray()
        stop2 = asyncio.Event()

        async def _make_outbound_unenrolled(hello: Hello):
            yield vad_start(1)
            for _ in range(63):
                yield interleave(1000, -4000, 256)
            yield marker_frame
            yield vad_end(2)
            # D-09: the client waits 3s past its own `vad.end` and never
            # sees a reply -- `stop` ends the session cleanly afterward,
            # rather than this test racing a server-side timeout.
            await asyncio.sleep(3.0)
            stop2.set()

        await asyncio.wait_for(
            run_session(
                f"ws://127.0.0.1:{port}/ws/edge",
                "t-1",
                make_outbound=_make_outbound_unenrolled,
                on_reply_audio=lambda chunk: received_reply2.extend(chunk),
                allow_plaintext=True,
                stop=stop2,
            ),
            timeout=15.0,
        )

        assert bytes(received_reply2) == b"", "an unenrolled voice must receive no reply audio at all"

        session_dirs = await _wait_for_session_dirs(session_dir, 2)
        turn2_events = _read_events(session_dirs[1])
        turn2_result = _find_speaker_result(turn2_events)
        assert turn2_result["status"] == "unknown"
        assert turn2_result["blocked"] is True
        assert turn2_result["reason"] == "unknown_speaker"

        turn2_timing = _read_timing(session_dirs[1])
        assert turn2_timing["turn_outcome"] == "unknown_speaker"
        # 260929-j08: a blocked turn must keep its recording for retroactive enrollment.
        assert (session_dirs[1] / "audio.pcm").is_file()

        # -- Channel isolation: the wake detector and STT never saw -----
        # -- channel 0's fixed 1000 value. -------------------------------
        wake_detector = app_module.app.state.wake_detector.inner  # unwrap SegmentBoundedWakeDetector
        assert wake_detector.chunks, "the wake detector never received a chunk"
        assert _only_channel_values(wake_detector.chunks) <= {-2000, -4000, _MARKER_VALUE}
        assert stt_double.received_frames, "speech-to-text never received a frame"
        assert _only_channel_values(stt_double.received_frames) <= {-2000, -4000, _MARKER_VALUE}
        assert stt_double.received_format == SourceFormat("pcm", 16000)

        # -- Embedding inference ran off the event loop thread (D-09). --
        assert created_embedders, "no speaker embedder was ever built"
        embedder = created_embedders[0]
        assert embedder.thread_names, "the embedder was never called"
        assert all(name.startswith("atlas-speaker-id") for name in embedder.thread_names)
    finally:
        server.should_exit = True
        await server_task
