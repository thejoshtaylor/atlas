"""`python -m atlas_edge --config PATH`: the Pi's own service entry point.

Builds `Capture`, `Playback` (writing reply audio through `Capture`'s own
`enqueue_playback` seam -- see `playback.py`'s own module docstring for why
this project never opens a second output stream on the XVF3800), a
`SileroGate` factory, a `DoaPoller`, and a `SendDelayWindow`, then hands
them to `run_service` (10-08). Installs a SIGTERM/SIGINT handler that sets
the stop event `run_service` already knows how to end on. Logs the array's
firmware version once at startup, and never logs the token (T-10-26).

Every hardware import (`sounddevice`, `sherpa_onnx`, `usb`) stays lazy,
inside whatever module already owns it -- this module imports none of
them directly, so `python -m atlas_edge --help` runs on a dev host with
no XVF3800 attached.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from typing import Any, AsyncIterator, Callable

from atlas_edge import xvf3800
from atlas_edge.capture import Capture
from atlas_edge.client import run_forever
from atlas_edge.config import EdgeConfig, EdgeConfigError, load_config
from atlas_edge.doa import DoaPoller
from atlas_edge.latency import SendDelayWindow
from atlas_edge.playback import Playback
from atlas_edge.service import run_service
from atlas_edge.vad import SileroGate

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "/etc/atlas-edge/config.toml"

# 10-SPIKE.md Q2: doa_parameter -- the read path this Pi's own vendor
# control interface proved on the real hardware. Angle accuracy is
# deferred (10-SPIKE.md); D-17 only records this value in Phase 10, and no
# Phase 10 decision reads it.
DOA_PARAMETER_NAME = "DOA_VALUE"

# How long a live frame's own arrival is trusted as "we are still inside a
# segment" before DoaPoller.in_segment() reports False again.
# service.py's own events_hook has no direct handle on the per-session
# Segmenter (10-08) -- events_hook() is called fresh per session with no
# arguments -- but Segmenter only ever causes a live frame to be sent
# while a segment is open (idle audio yields nothing at all, per
# segmenter.py's own docstring), so "a live frame arrived recently" is an
# accurate, if indirect, proxy for segment state. A few multiples of the
# frame period (256 samples / 16kHz = 16ms) and the 5Hz DoA poll period
# (200ms) bounds how long DoA polling overshoots after a segment's real
# tail ends.
SEGMENT_GRACE_S = 0.5

LATENCY_TICK_S = 1.0

Factories = "dict[str, Callable[..., Any]]"


class _SegmentTracker:
    """Infers whether a speech segment is currently open from the
    presence of a recently-sent live frame -- see this module's own
    docstring for why."""

    def __init__(
        self, *, grace_s: float = SEGMENT_GRACE_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._grace_s = grace_s
        self._clock = clock
        self._last_live_frame_at: "float | None" = None

    def mark_live_frame(self, sent_at: float) -> None:
        self._last_live_frame_at = sent_at

    def in_segment(self) -> bool:
        if self._last_live_frame_at is None:
            return False
        return (self._clock() - self._last_live_frame_at) < self._grace_s


def _default_capture(config: EdgeConfig) -> Any:
    return Capture(config.capture_device)


def _default_playback(config: EdgeConfig, capture: Any) -> Any:
    return Playback(capture.enqueue_playback)


def _default_gate_factory(config: EdgeConfig) -> Callable[[], Any]:
    def _build_gate() -> Any:
        return SileroGate(
            config.vad_model_path,
            threshold=config.vad_threshold,
            min_silence_ms=config.vad_min_silence_ms,
        )

    return _build_gate


def _default_doa_poller(config: EdgeConfig, in_segment: Callable[[], bool]) -> Any:
    device = xvf3800.find_device()
    return DoaPoller(device, DOA_PARAMETER_NAME, in_segment=in_segment)


def _default_latency_window(config: EdgeConfig) -> Any:
    return SendDelayWindow()


DEFAULT_FACTORIES: Factories = {
    "capture": _default_capture,
    "playback": _default_playback,
    "gate_factory": _default_gate_factory,
    "doa_poller": _default_doa_poller,
    "latency_window": _default_latency_window,
    "runner": run_forever,
}


async def _latency_ticker(
    window: Any,
    *,
    interval_s: float = LATENCY_TICK_S,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> AsyncIterator[str]:
    while True:
        await sleep(interval_s)
        message = window.drain_message()
        if message is not None:
            yield message


async def _merge_events(
    doa_messages: AsyncIterator[str], latency_messages: AsyncIterator[str]
) -> AsyncIterator[str]:
    """Interleaves DoA and latency messages into one outbound events
    stream -- the same shape `service.py::_merge` already uses to fold
    `events_hook` into the capture stream, written here rather than
    imported from that module's own private helper."""
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    sentinel = object()

    async def _drain(iterator: AsyncIterator[str]) -> None:
        async for item in iterator:
            await queue.put(item)
        await queue.put(sentinel)

    task_a = asyncio.ensure_future(_drain(doa_messages))
    task_b = asyncio.ensure_future(_drain(latency_messages))
    remaining = 2
    try:
        while remaining > 0:
            item = await queue.get()
            if item is sentinel:
                remaining -= 1
                continue
            yield item
    finally:
        for task in (task_a, task_b):
            if not task.done():
                task.cancel()
        await asyncio.gather(task_a, task_b, return_exceptions=True)


def build_service(config: EdgeConfig, factories: "Factories | None" = None) -> Callable[..., Any]:
    """Builds `Capture`, `Playback`, a `SileroGate` factory, a `DoaPoller`
    and a `SendDelayWindow` through `factories` (the real ones by
    default), wires them into `run_service`, and returns an async
    `_run(stop=None)` callable `main` awaits."""
    built: Factories = {**DEFAULT_FACTORIES, **(factories or {})}

    capture = built["capture"](config)
    playback = built["playback"](config, capture)
    gate_factory = built["gate_factory"](config)
    tracker = _SegmentTracker()
    window = built["latency_window"](config)
    doa_poller = built["doa_poller"](config, tracker.in_segment)

    def on_live_frame_sent(captured_at: float, sent_at: float) -> None:
        tracker.mark_live_frame(sent_at)
        window.record(captured_at, sent_at)

    async def events_hook() -> AsyncIterator[str]:
        async for item in _merge_events(doa_poller.messages(), _latency_ticker(window)):
            yield item

    async def _run(stop: "asyncio.Event | None" = None) -> None:
        await run_service(
            config,
            capture=capture,
            gate_factory=gate_factory,
            on_reply_audio=playback.write,
            on_live_frame_sent=on_live_frame_sent,
            events_hook=events_hook,
            runner=built["runner"],
            stop=stop,
        )

    return _run


def _log_firmware_version_once() -> None:
    """Logs the array's firmware version once at startup -- and never the
    token (T-10-26). A failure here (the array not answering yet) is only
    a missing diagnostic, not a reason to refuse to start."""
    try:
        device = xvf3800.find_device()
        version = xvf3800.decode_version(
            xvf3800.read_parameter(device, xvf3800.PARAMETERS["VERSION"])
        )
        logger.info("XVF3800 firmware version %s", ".".join(str(part) for part in version))
    except Exception as exc:  # noqa: BLE001 -- a diagnostic log must never stop startup
        logger.warning("could not read the XVF3800 firmware version: %s", exc)


def _validate_startup_prerequisites(config: EdgeConfig) -> None:
    """Refuses to start with a message naming the missing file, rather
    than starting and failing lazily on the first real speech frame --
    the Silero model is fetched by `fetch-vad-model.sh` (Task 3), never
    downloaded here."""
    if not os.path.exists(config.vad_model_path):
        raise FileNotFoundError(
            f"Silero VAD model not found at {config.vad_model_path!r} -- "
            "run edge/scripts/fetch-vad-model.sh first"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas_edge",
        description=(
            "The Pi's own edge-microphone service: captures, gates, streams, plays "
            "replies through the XVF3800, and reports DoA and send delay."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"path to the Pi's config.toml (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser


def main(argv: "list[str] | None" = None, *, factories: "Factories | None" = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)  # journald reads stderr

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        _validate_startup_prerequisites(config)
        _log_firmware_version_once()
        service_runner = build_service(config, factories)
    except (EdgeConfigError, OSError) as exc:
        logger.error("failed to start: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 -- any other startup failure also returns 1
        logger.error("failed to start: %s", exc)
        return 1

    stop_event = asyncio.Event()

    def _handle_stop_signal(*_args: Any) -> None:
        stop_event.set()

    async def _run_until_stopped() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _handle_stop_signal)
            except NotImplementedError:
                pass  # no signal support on this platform (never the Pi)
        await service_runner(stop=stop_event)

    try:
        asyncio.run(_run_until_stopped())
    except Exception as exc:  # noqa: BLE001 -- systemd restarts on a non-zero exit
        logger.error("edge service ended with an error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
