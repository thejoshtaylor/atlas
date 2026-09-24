"""The end-to-end tracer: one wake hit on real camera bytes runs one turn
and the reply reaches the camera speaker (VOICE-03, VOICE-04, VOICE-05,
SRC-02, PROV-07).

Drives a genuine A-law byte buffer through the real `CameraAudioSource`
(only its container-open call is swapped for a local fixture file, the
same way a real deployment swaps it for a live RTSP socket -- the class
under test never changes) and a real `SourceRunner`, `FifoWriter`, and
named pipe. The only faked component is the wake detector itself: the
shipped Vosk engine needs a model directory that is a deployment artifact
and is not in this repository, so the live human check in
`02-03-PLAN.md` Task 1 is what exercises the real one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from typing import Any

import av

from atlas.config import CameraConfig
from atlas.sources.runner import SourceRunner
from atlas.speaker.fifo_writer import FifoWriter
from atlas.timing import TurnTimings
from atlas.transports.camera import CameraAudioSource
from atlas.turn.controller import run_turn

from tests.conftest import BrainReply, FakeBrain, FakeTts, FakeWakeDetector, FinalTranscript, RecordingFakeStt


# A deterministic, non-repeating byte sequence -- not all-zero -- so a
# byte-identity assertion actually proves something rather than passing
# vacuously against a constant buffer. 4000 bytes of 8 kHz mono A-law is
# 500ms, comfortably more than one demux packet (RESEARCH.md Pattern 2).
_RAW_ALAW_BYTES = bytes((i * 7 + 3) % 256 for i in range(4000))


def _write_alaw_fixture(path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(_RAW_ALAW_BYTES)


def _open_fixture_container(_config: CameraConfig) -> Any:
    """Stands in for `_open_rtsp_container`: opens the local fixture file
    with PyAV's raw A-law demuxer instead of a live RTSP socket, so the
    packets and the decode inside `CameraAudioSource` are real rather than
    faked (Task 1's own instruction)."""
    return av.open(
        _FIXTURE_PATH,
        format="alaw",
        options={"sample_rate": "8000", "ar": "8000", "ac": "1"},
    )


_FIXTURE_PATH = ""  # set per-test from tmp_path, read by the closure above


class _UncalledToolHost:
    """No tool call is ever expected on this path -- `FakeBrain` below
    replies with plain text and no tool calls. A call here means the tracer
    accidentally exercised the tool-round path this test does not cover."""

    async def call_tool(self, name: str, arguments: dict) -> Any:
        raise AssertionError(f"tool_host.call_tool({name!r}) should never be reached by this tracer")


def _drain_fifo(path: str, out: list[bytes], expected_length: int, done: threading.Event) -> None:
    """Runs in a background thread: reads the FIFO's read side until
    `expected_length` bytes have arrived, then signals `done`.

    Opened with `buffering=0` deliberately: a buffered reader's `read(n)`
    blocks until either `n` bytes accumulate or the writer closes (EOF) --
    it does not return on a short read the way a raw, unbuffered read does.
    The reply here is far smaller than any default buffer size, so a
    buffered reader would never see it until `fifo_writer.close()` runs at
    the end of this test, defeating the whole point of polling for it.
    """
    total = 0
    with open(path, "rb", buffering=0) as fh:
        while total < expected_length:
            chunk = fh.read(4096)
            if not chunk:
                break
            out.append(chunk)
            total += len(chunk)
    done.set()


async def test_one_wake_hit_runs_one_turn_and_the_reply_reaches_the_speaker(tmp_path):
    global _FIXTURE_PATH
    fixture_path = str(tmp_path / "camera_fixture.alaw")
    _write_alaw_fixture(fixture_path)
    _FIXTURE_PATH = fixture_path

    fifo_path = str(tmp_path / "speaker.alaw")
    # Created here, before the reader thread starts: `FifoWriter` also
    # creates the node itself if it is missing (a real deployment's mount
    # may only provide the directory), but racing that self-creation
    # against a reader thread's own `open()` call is exactly the kind of
    # startup-ordering hazard Pitfall 4 warns about -- the reader's open
    # would raise `FileNotFoundError` and never retry, and the writer's own
    # open would then block forever waiting for a reader that already gave
    # up. Creating the node upfront removes that race for this test.
    os.mkfifo(fifo_path)

    camera_config = CameraConfig(rtsp_url="rtsp://redacted@camera.invalid/stream1", encoding="alaw", sample_rate=8000)
    fifo_writer = FifoWriter(fifo_path)
    camera_source = CameraAudioSource(camera_config, fifo_writer, open_container=_open_fixture_container)

    reply_chunks = [b"reply-audio-chunk-one", b"reply-audio-chunk-two"]
    expected_reply = b"".join(reply_chunks)
    received_by_speaker: list[bytes] = []
    speaker_done = threading.Event()
    reader_thread = threading.Thread(
        target=_drain_fifo, args=(fifo_path, received_by_speaker, len(expected_reply), speaker_done), daemon=True
    )
    reader_thread.start()

    # Blocks (in an executor thread) until `reader_thread`'s own open()
    # attaches -- Pitfall 4, proven rather than merely avoided: this await
    # would hang the test forever if `FifoWriter.open()` were ever changed
    # to run inline on the event loop.
    await fifo_writer.open()

    recording_stt = RecordingFakeStt(events=[FinalTranscript(text="turn on the light")])
    fake_brain = FakeBrain(replies=[BrainReply(text="the light is on")])
    fake_tts = FakeTts(chunks=reply_chunks)

    async def _run_turn_for_source(source: Any) -> None:
        await run_turn(
            source,
            recording_stt,
            fake_brain,
            fake_tts,
            _UncalledToolHost(),
            tools_schema=[],
            system_prompt="test system prompt",
            max_tool_rounds=3,
            timings=TurnTimings(),
            max_utterance_s=5.0,
        )

    # Fires on the very first chunk the runner hands it -- almost the whole
    # fixture is left over for `recording_stt` to receive, which is what
    # makes the byte-identity assertion below meaningful rather than
    # trivially checking an empty list.
    wake_detector = FakeWakeDetector(fire_at_call=0)
    raw_chunks_seen_before_hit: list[bytes] = []

    def _asserting_decode_for_detector(chunk: bytes) -> bytes:
        # VOICE-03, checked at the exact moment of the hit: any bytes
        # reaching `recording_stt` here (i.e. before `run_turn` was ever
        # called) would mean an implementation streamed audio before the
        # wake word fired.
        assert recording_stt.received == [], (
            "the transcriber received audio before the wake hit -- VOICE-03 violated"
        )
        detector_chunk = camera_source.decode_for_detector(chunk)
        if detector_chunk:
            raw_chunks_seen_before_hit.append(chunk)
        return detector_chunk

    runner = SourceRunner("camera", camera_source, wake_detector, _asserting_decode_for_detector, _run_turn_for_source)

    camera_source.start()
    runner_task = asyncio.create_task(runner.run())

    for _ in range(200):
        if speaker_done.is_set():
            break
        await asyncio.sleep(0.02)

    runner_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await runner_task
    await camera_source.close()
    await fifo_writer.close()
    reader_thread.join(timeout=2)

    assert speaker_done.is_set(), "reply audio never reached the speaker FIFO reader"
    assert b"".join(received_by_speaker) == expected_reply

    consumed_before_hit = b"".join(raw_chunks_seen_before_hit)
    assert consumed_before_hit, "the wake detector's own decode path saw no packets at all"
    assert consumed_before_hit == _RAW_ALAW_BYTES[: len(consumed_before_hit)]

    # PROV-07 / VOICE-04: what actually reached the transcriber is exactly
    # the camera's own undecoded bytes -- the remainder of the fixture
    # after the one packet the wake detector consumed, bit for bit.
    assert recording_stt.received
    assert b"".join(recording_stt.received) == _RAW_ALAW_BYTES[len(consumed_before_hit) :]

    assert wake_detector.calls >= 1
