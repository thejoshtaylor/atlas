#!/usr/bin/env python3
"""Proves which finalize wire message the live xAI speech-to-text socket
actually honours (10-05-PLAN.md Task 3, D-12; 10-RESEARCH.md Pitfall 1).

xAI's own documentation names the message `{"type": "Finalize"}`. D-12
wrote lowercase `{"type": "finalize"}`. Neither had been checked against
the live socket until this script ran. It streams one short, invented
command's own synthesized audio into three separate speech-to-text
sessions -- one per candidate spelling, plus a control that sends no
finalize message at all -- and times how long each takes to produce a
`speech_final` partial. A candidate "works" when its final arrives at
least 400 ms before the control's, or the control itself produced no
final within the 5 second limit.

This script loads configuration through `atlas.config.load_config`, so
`XAI_API_KEY` reaches it only through the same environment expansion the
application itself uses (`${XAI_API_KEY}` in `config/config.example.yaml`).
It never prints the key, its length, or a prefix -- run it through
`scripts/dev-verify-xai-finalize.sh`, which sources `.env` the way
`scripts/dev-run.sh` does and never echoes a value either.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import websockets

from atlas.config import ConfigError, SttConfig, load_config
from atlas.providers.stt_xai import XaiStt
from atlas.providers.tts_xai import CHUNK_BYTES, SinkFormat, XaiTts
from atlas.transports.base import SourceFormat

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.example.yaml"

# One short, invented command -- never a real house's own phrase (this
# repository's own convention, `atlas_mcp.safety._demo`).
_INVENTED_COMMAND = "turn on the desk lamp"

# 20 ms of 16 kHz mono PCM16 -- the real-time pace `<action>` calls for,
# matching `tts_xai.py::CHUNK_BYTES` exactly (this module's own 20 ms frame
# size, not a second, independently chosen constant).
_CHUNK_INTERVAL_S = 0.02

_SPEECH_FINAL_TIMEOUT_S = 5.0

# The gap a candidate must beat the control by to count as "works" --
# 10-05-PLAN.md's own threshold.
_WORKS_MARGIN_MS = 400.0

# Name, wire message. `None` is the control: no finalize message sent at
# all, so xAI's own endpointing is the only thing that can ever produce a
# final in that session.
_VARIANTS: "list[tuple[str, dict[str, Any] | None]]" = [
    ("Finalize", {"type": "Finalize"}),
    ("finalize", {"type": "finalize"}),
    ("control", None),
]


def _chunk_audio(pcm16: bytes, chunk_bytes: int) -> "list[bytes]":
    return [pcm16[i : i + chunk_bytes] for i in range(0, len(pcm16), chunk_bytes)]


async def _synthesize_command_audio(tts_config: Any) -> bytes:
    """16 kHz PCM16 of `_INVENTED_COMMAND`, through the real xAI text-to-
    speech endpoint -- real synthesized speech, not silence or a tone, the
    same "no fake microphone" discipline `scripts/measure_local_providers.py`
    already established for a different measurement."""
    tts = XaiTts(tts_config)
    try:
        return await tts.synthesize_once(_INVENTED_COMMAND, sink=SinkFormat("pcm", 16000))
    finally:
        await tts.aclose()


async def _run_variant(
    stt_config: SttConfig,
    source_format: SourceFormat,
    audio_chunks: "list[bytes]",
    finalize_message: "dict[str, Any] | None",
) -> "tuple[float | None, str]":
    """Open one speech-to-text session, stream `audio_chunks` at real-time
    pace, send `finalize_message` (or nothing, for the control) once
    streaming ends, and time the first `speech_final` partial from that
    moment. Deliberately sends no trailing `audio.done` -- the point is
    whether `finalize_message` alone ends the utterance, not whether the
    provider's own end-of-stream signal does.
    """
    headers = {"Authorization": f"Bearer {stt_config.api_key}"}
    url = XaiStt(stt_config).build_url(source_format)

    async with websockets.connect(url, additional_headers=headers) as ws:
        ready = json.loads(await ws.recv())
        if ready.get("type") != "transcript.created":
            raise RuntimeError(f"unexpected first event from xAI STT: {ready!r}")

        for chunk in audio_chunks:
            await ws.send(chunk)
            await asyncio.sleep(_CHUNK_INTERVAL_S)

        started_at = time.monotonic()
        if finalize_message is not None:
            await ws.send(json.dumps(finalize_message))

        speech_final_ms: float | None = None
        transcript_text = ""
        try:
            async with asyncio.timeout(_SPEECH_FINAL_TIMEOUT_S):
                async for raw in ws:
                    event: dict[str, Any] = json.loads(raw)
                    if event.get("type") == "transcript.partial" and event.get("speech_final"):
                        speech_final_ms = (time.monotonic() - started_at) * 1000
                        transcript_text = event.get("text", "")
                        break
        except TimeoutError:
            pass

        return speech_final_ms, transcript_text


async def run_verification(config: Any) -> "dict[str, Any]":
    audio = await _synthesize_command_audio(config.tts)
    audio_chunks = _chunk_audio(audio, CHUNK_BYTES)
    source_format = SourceFormat("pcm", 16000)

    results: "dict[str, Any]" = {}
    for name, message in _VARIANTS:
        speech_final_ms, text = await _run_variant(config.stt, source_format, audio_chunks, message)
        results[name] = {"speech_final_ms": speech_final_ms, "text": text}

    control_ms = results["control"]["speech_final_ms"]
    for name, payload in results.items():
        if name == "control":
            continue
        candidate_ms = payload["speech_final_ms"]
        if control_ms is None:
            # The control never finalized at all within the limit -- any
            # candidate that finalized at all beats "never" outright.
            payload["works"] = candidate_ms is not None
        else:
            payload["works"] = candidate_ms is not None and (control_ms - candidate_ms) >= _WORKS_MARGIN_MS

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prove which finalize wire message the live xAI speech-to-text socket "
            "honours -- 'Finalize' (xAI's own docs), 'finalize' (D-12), or neither."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=f"the application config to read stt.*/tts.* from (default: {_DEFAULT_CONFIG_PATH})",
    )
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    results = asyncio.run(run_verification(config))
    print(json.dumps(results))
    return 0 if any(payload.get("works") for payload in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
