#!/usr/bin/env python3
"""Measures VOICE-02 end to end: replays a recorded utterance through the
real WebSocket transport and the real WebRTC offer/answer route, at a real
microphone's cadence, and prints per-stage medians for both headline
numbers.

Phase 01 recorded four consecutive open unrun-verify entries for exactly
this measurement, every one of them for want of credentials. `.env` is
present now, and this script is what closes them: criterion 7 is claimed
from this script's output, not from the instrumentation merely existing.

This script never parses `.env` itself -- run it through
`scripts/dev-measure.sh`, which sources `.env` the way `dev-run.sh` does,
boots the application, and waits for it to report itself up before
invoking this script. This script reads no environment variable directly
and prints no credential, value, length, or prefix.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_WAV = _REPO_ROOT / "tests" / "fixtures" / "one_command_16k_mono.wav"
_DEFAULT_URL = "http://127.0.0.1:8080"

# The AudioWorklet render quantum `web/public/dev-mic/pcm-worklet.js` posts one
# Int16Array per: 128 frames, 256 bytes at 16-bit mono. Chunking and pacing
# to anything else measures a cadence no real microphone stream produces
# (Pitfall 5, 01.1-RESEARCH.md) -- referenced by name below, never
# rewritten as a second literal.
_CHUNK_FRAMES = 128

_EXPECTED_CHANNELS = 1
_EXPECTED_SAMPWIDTH = 2
_EXPECTED_FRAMERATE = 16000

_MIN_TURNS = 5
_TRANSPORTS = ("websocket", "webrtc")
_TIMING_EVENT_TIMEOUT_S = 30.0
_VOICE02_BUDGET_MS = 1500.0
_HEADLINE_KEYS = ("end_of_speech_to_first_audio_ms", "end_of_speech_to_answer_audio_ms")


class WavFormatError(ValueError):
    """A WAV fixture is not 16 kHz mono PCM16 -- raised before any socket
    opens, naming what was found and what was expected."""


def validate_wav_format(nchannels: int, sampwidth: int, framerate: int) -> None:
    """Raise `WavFormatError` naming what was found and what was expected.

    Called from `read_wav_chunks` before either transport driver opens a
    socket -- a wrong fixture must stop the run with a named reason, not
    surface later as a nonsensical measured duration.
    """
    problems = []
    if nchannels != _EXPECTED_CHANNELS:
        problems.append(f"{nchannels} channel(s), expected {_EXPECTED_CHANNELS} (mono)")
    if sampwidth != _EXPECTED_SAMPWIDTH:
        problems.append(f"{sampwidth}-byte samples, expected {_EXPECTED_SAMPWIDTH} (16-bit)")
    if framerate != _EXPECTED_FRAMERATE:
        problems.append(f"{framerate} Hz, expected {_EXPECTED_FRAMERATE} Hz")
    if problems:
        raise WavFormatError("WAV fixture is not usable: " + "; ".join(problems))


def read_wav_chunks(path: Path, chunk_frames: int = _CHUNK_FRAMES) -> list[bytes]:
    """Read `path` as `chunk_frames`-frame chunks, validated first.

    The final chunk is shorter than `chunk_frames` whenever the recording's
    frame count is not an exact multiple of it -- never padded with
    silence, never dropped. A missing file raises `FileNotFoundError`
    straight from `wave.open`, which is this script's own missing-fixture
    error path when Task 1's recording was not made.
    """
    with wave.open(str(path), "rb") as wav_file:
        validate_wav_format(wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate())
        chunks: list[bytes] = []
        while True:
            frames = wav_file.readframes(chunk_frames)
            if not frames:
                break
            chunks.append(frames)
        return chunks


def pacing_schedule(
    chunks: Sequence[bytes],
    *,
    framerate: int = _EXPECTED_FRAMERATE,
    sampwidth: int = _EXPECTED_SAMPWIDTH,
    channels: int = _EXPECTED_CHANNELS,
) -> list[float]:
    """One `asyncio.sleep` duration per chunk, summing to the recording's
    real playback duration.

    For a recording of N frames, `sum(pacing_schedule(chunks)) == N /
    framerate` -- a paced send takes exactly as long as the recording
    itself, so the STT's silence-gap endpointing sees the cadence a real
    microphone stream would produce rather than a burst that arrives in a
    few milliseconds of wall clock (Pitfall 5).
    """
    frame_size = sampwidth * channels
    return [(len(chunk) / frame_size) / framerate for chunk in chunks]


@dataclass(frozen=True)
class StageAggregate:
    """One stage's summary over every turn that reached it.

    `median`/`minimum`/`maximum` are `None` -- never a fabricated zero --
    when no turn in the run reached this stage at all. `reached` counts how
    many of `total` turns did, so a stage every turn reached looks
    different from one only some did, rather than silently averaging the
    turns that never got there out of the picture.
    """

    median: float | None
    minimum: float | None
    maximum: float | None
    reached: int
    total: int


def aggregate_turn_events(events: Sequence[dict[str, Any]]) -> dict[str, StageAggregate]:
    """Median/min/max per stage, and for both headline numbers, over every
    `turn.timing` event a run collected.

    Consumes exactly the shape `timing.py`'s `to_event()` sends
    (`stage_durations_ms` plus the two headline keys) -- this harness is a
    client receiving the same events the browser panel renders, and never
    re-derives a duration from a raw timestamp itself.
    """
    total = len(events)
    if total == 0:
        return {}
    stage_names = list(events[0].get("stage_durations_ms", {}).keys())
    result: dict[str, StageAggregate] = {}
    for key in (*stage_names, *_HEADLINE_KEYS):
        values: list[float] = []
        for event in events:
            value = event.get(key) if key in _HEADLINE_KEYS else event.get("stage_durations_ms", {}).get(key)
            if value is not None:
                values.append(value)
        if values:
            result[key] = StageAggregate(
                median=statistics.median(values),
                minimum=min(values),
                maximum=max(values),
                reached=len(values),
                total=total,
            )
        else:
            result[key] = StageAggregate(median=None, minimum=None, maximum=None, reached=0, total=total)
    return result


def _turns_type(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--turns must be an integer, got {raw!r}") from exc
    if value < _MIN_TURNS:
        raise argparse.ArgumentTypeError(
            f"--turns must be at least {_MIN_TURNS} -- criterion 7 requires at least five real "
            f"turns per transport, got {value}"
        )
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded utterance through the real transport, at a real microphone's "
            "cadence, and print per-stage medians -- the instrument VOICE-02's criterion 7 is "
            "claimed from."
        )
    )
    parser.add_argument(
        "--transport",
        choices=_TRANSPORTS,
        required=True,
        help="which transport this server implements to drive",
    )
    parser.add_argument(
        "--turns",
        type=_turns_type,
        default=_MIN_TURNS,
        help=f"turns to run, at least {_MIN_TURNS} (default: {_MIN_TURNS})",
    )
    parser.add_argument(
        "--wav",
        type=Path,
        default=_DEFAULT_WAV,
        help=f"the recorded utterance to replay (default: {_DEFAULT_WAV})",
    )
    parser.add_argument(
        "--url",
        default=_DEFAULT_URL,
        help=f"the application's base URL (default: {_DEFAULT_URL})",
    )
    return parser


def _ws_url(base_url: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :] + "/ws/turn"
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :] + "/ws/turn"
    raise ValueError(f"--url must start with http:// or https://, got {base_url!r}")


async def _run_websocket_turn(ws_url: str, chunks: list[bytes], schedule: list[float]) -> dict[str, Any] | None:
    """Drive one turn over `/ws/turn`: paced binary frames out, the
    `turn.timing` JSON text frame in -- the raw-PCM16-passthrough contract
    `transports/websocket.py`'s own docstring states.
    """
    async with websockets.connect(ws_url) as ws:
        timing_event: dict[str, Any] | None = None

        async def _receive() -> None:
            nonlocal timing_event
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    continue  # reply audio -- never inspected; numbers only, per timing.py's doctrine
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "turn.timing":
                    timing_event = event
                    return

        receiver = asyncio.create_task(_receive())
        for chunk, delay in zip(chunks, schedule):
            await ws.send(chunk)
            await asyncio.sleep(delay)
        try:
            await asyncio.wait_for(receiver, timeout=_TIMING_EVENT_TIMEOUT_S)
        except asyncio.TimeoutError:
            receiver.cancel()
        return timing_event


async def _wait_ice_gathering_complete(pc: RTCPeerConnection) -> None:
    """Mirror `web/public/dev-mic/webrtc.js`'s `waitForIceGatheringComplete`: wait for
    every ICE candidate to be gathered so the one POST this harness makes
    already carries all of them, rather than trickling candidates in after
    the fact -- the same non-trickle exchange the real page performs.
    """
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    await done.wait()


async def _run_webrtc_turn(offer_url: str, wav_path: Path) -> dict[str, Any] | None:
    """Drive one turn over `/webrtc/offer`, mirroring `web/public/dev-mic/webrtc.js`:
    build the same offer, POST it, read the same data channel the page
    reads. `aiortc.contrib.media.MediaPlayer` supplies the recording as a
    real-time-paced track -- no hand-rolled `MediaStreamTrack` reading raw
    frames off a clock.
    """
    pc = RTCPeerConnection()
    player = MediaPlayer(str(wav_path))
    pc.addTrack(player.audio)
    data_channel = pc.createDataChannel("events")

    timing_event: dict[str, Any] | None = None
    got_timing = asyncio.Event()

    @data_channel.on("message")
    def _on_message(message: Any) -> None:
        nonlocal timing_event
        if isinstance(message, (bytes, bytearray)):
            return
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            return
        if event.get("type") == "turn.timing":
            timing_event = event
            got_timing.set()

    try:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await _wait_ice_gathering_complete(pc)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                offer_url,
                json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
                timeout=_TIMING_EVENT_TIMEOUT_S,
            )
            response.raise_for_status()
            answer = response.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

        try:
            await asyncio.wait_for(got_timing.wait(), timeout=_TIMING_EVENT_TIMEOUT_S)
        except asyncio.TimeoutError:
            pass
        return timing_event
    finally:
        await pc.close()


async def _run_turns(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run `args.turns` real turns against `args.transport`, printing an
    honest per-turn line as each one finishes rather than only a final
    summary -- a short run, an errored turn, or a stalled connection is
    visible while it happens, not just inferable from a missing number at
    the end.
    """
    chunks = read_wav_chunks(args.wav)  # validated before either transport opens a socket
    schedule = pacing_schedule(chunks)

    events: list[dict[str, Any]] = []
    errors = 0
    for turn in range(1, args.turns + 1):
        try:
            if args.transport == "websocket":
                event = await _run_websocket_turn(_ws_url(args.url), chunks, schedule)
            else:
                event = await _run_webrtc_turn(f"{args.url}/webrtc/offer", args.wav)
        except Exception as exc:  # one turn's failure must not stop the run
            errors += 1
            print(f"turn {turn}/{args.turns}: errored ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        if event is None:
            errors += 1
            print(
                f"turn {turn}/{args.turns}: no turn.timing event within {_TIMING_EVENT_TIMEOUT_S:.0f}s",
                file=sys.stderr,
            )
            continue
        events.append(event)
        print(f"turn {turn}/{args.turns}: ok")

    if errors:
        print(f"{errors}/{args.turns} turns did not complete", file=sys.stderr)
    return events


def _print_report(transport: str, events: list[dict[str, Any]], total_turns: int) -> None:
    print(f"\n--- {transport}: {len(events)}/{total_turns} turns completed ---")
    if not events:
        print("no turns completed -- nothing to report")
        return
    aggregates = aggregate_turn_events(events)
    for key, agg in aggregates.items():
        if key in _HEADLINE_KEYS:
            continue
        if agg.median is None:
            print(f"  {key}: never reached (0/{agg.total})")
        else:
            print(
                f"  {key}: median={agg.median:.0f}ms min={agg.minimum:.0f}ms "
                f"max={agg.maximum:.0f}ms ({agg.reached}/{agg.total} reached)"
            )
    print(f"  --- headline (VOICE-02 budget: {_VOICE02_BUDGET_MS:.0f}ms) ---")
    for key in _HEADLINE_KEYS:
        agg = aggregates[key]
        if agg.median is None:
            print(f"  {key}: never reached (0/{agg.total})")
            continue
        verdict = "within budget" if agg.median <= _VOICE02_BUDGET_MS else "OVER BUDGET"
        print(
            f"  {key}: median={agg.median:.0f}ms min={agg.minimum:.0f}ms "
            f"max={agg.maximum:.0f}ms ({agg.reached}/{agg.total} reached) -- {verdict}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        events = asyncio.run(_run_turns(args))
    except FileNotFoundError as exc:
        print(
            f"error: recorded fixture not found: {exc.filename or args.wav} -- "
            "record it first (plan 01.1-07's checkpoint names the exact steps)",
            file=sys.stderr,
        )
        return 1
    except WavFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_report(args.transport, events, args.turns)
    return 0 if events else 1


if __name__ == "__main__":
    sys.exit(main())
