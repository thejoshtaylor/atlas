"""RED for build_service/main (10-09-PLAN.md Task 2). Every test token
literal below stays under 8 characters (tests/test_repo_hygiene.py's
credential scan). No test opens a real audio device or USB backend --
every factory is a fake, injected through `main(..., factories=...)`.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time

import pytest

import atlas_edge.__main__ as main_module
from atlas_edge.__main__ import build_service, main


def _write_config(tmp_path, *, token: str = "tok1234", vad_model_path=None) -> str:
    if vad_model_path is None:
        vad_model_path = tmp_path / "silero_vad.onnx"
        vad_model_path.write_text("fake model bytes")
    path = tmp_path / "config.toml"
    path.write_text(
        f'server_url = "wss://svr.test/ws/edge"\n'
        f'token = "{token}"\n'
        f'vad_model_path = "{vad_model_path}"\n'
    )
    os.chmod(path, 0o600)
    return str(path)


class FakeCapture:
    sample_rate = 16000
    channels = 2
    frame_samples = 256

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.enqueued: "list[bytes]" = []

    def enqueue_playback(self, data: bytes) -> None:
        self.enqueued.append(data)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    async def frames(self):
        return
        yield  # pragma: no cover -- never iterated by the fake runner


class FakePlayback:
    def __init__(self, enqueue) -> None:
        self.enqueue = enqueue
        self.written: "list[bytes]" = []

    async def write(self, data: bytes) -> None:
        self.written.append(data)


class FakeDoaPoller:
    def __init__(self, in_segment) -> None:
        self.in_segment = in_segment

    async def messages(self):
        return
        yield  # pragma: no cover


class FakeWindow:
    def __init__(self) -> None:
        self.recorded: "list[tuple[float, float]]" = []

    def record(self, captured_at: float, sent_at: float) -> None:
        self.recorded.append((captured_at, sent_at))

    def drain_message(self):
        return None


async def _fake_runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop):
    await stop.wait()


def _fake_factories(**overrides):
    factories = {
        "capture": lambda config: FakeCapture(),
        "playback": lambda config, capture: FakePlayback(capture.enqueue_playback),
        "gate_factory": lambda config: (lambda: object()),
        "doa_poller": lambda config, in_segment: FakeDoaPoller(in_segment),
        "latency_window": lambda config: FakeWindow(),
        "runner": _fake_runner,
    }
    factories.update(overrides)
    return factories


# --- main(): startup failure ------------------------------------------------


def test_missing_vad_model_logs_the_reason_and_returns_1(tmp_path, caplog) -> None:
    missing = tmp_path / "does-not-exist.onnx"
    config_path = _write_config(tmp_path, vad_model_path=missing)

    with caplog.at_level(logging.ERROR):
        exit_code = main(["--config", config_path], factories=_fake_factories())

    assert exit_code == 1
    assert any("failed to start" in record.message for record in caplog.records)


def test_a_missing_config_file_logs_the_reason_and_returns_1(tmp_path, caplog) -> None:
    missing_config = str(tmp_path / "no-such-config.toml")

    with caplog.at_level(logging.ERROR):
        exit_code = main(["--config", missing_config], factories=_fake_factories())

    assert exit_code == 1
    assert any("failed to start" in record.message for record in caplog.records)


# --- main(): runs until SIGTERM ---------------------------------------------


def test_main_runs_until_sigterm_then_returns_0(tmp_path) -> None:
    config_path = _write_config(tmp_path)

    def _send_signal_soon() -> None:
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    timer_thread = threading.Thread(target=_send_signal_soon)
    timer_thread.start()

    exit_code = main(["--config", config_path], factories=_fake_factories())
    timer_thread.join(timeout=5)

    assert not timer_thread.is_alive()
    assert exit_code == 0


def test_never_logs_the_token(tmp_path, caplog) -> None:
    config_path = _write_config(tmp_path, token="tok1234")

    def _send_signal_soon() -> None:
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    timer_thread = threading.Thread(target=_send_signal_soon)
    timer_thread.start()

    with caplog.at_level(logging.DEBUG):
        main(["--config", config_path], factories=_fake_factories())
    timer_thread.join(timeout=5)

    for record in caplog.records:
        assert "tok1234" not in record.getMessage()


# --- build_service(): wiring -------------------------------------------------


@pytest.mark.asyncio
async def test_build_service_wires_reply_audio_live_frame_and_runner(monkeypatch) -> None:
    captured: "dict" = {}

    async def fake_run_service(config, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_module, "run_service", fake_run_service)

    capture = FakeCapture()
    factories = _fake_factories(
        capture=lambda config: capture,
        playback=lambda config, cap: FakePlayback(cap.enqueue_playback),
    )

    service_runner = build_service(object(), factories)
    await service_runner(stop=None)

    assert captured["capture"] is capture
    assert captured["runner"] is _fake_runner
    assert callable(captured["events_hook"])
    assert callable(captured["on_reply_audio"])
    assert callable(captured["on_live_frame_sent"])


@pytest.mark.asyncio
async def test_on_live_frame_sent_records_delay_and_marks_the_segment_tracker_active(
    monkeypatch,
) -> None:
    captured: "dict" = {}

    async def fake_run_service(config, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_module, "run_service", fake_run_service)

    window = FakeWindow()
    doa_poller_box: "list[FakeDoaPoller]" = []

    def _doa_poller_factory(config, in_segment):
        poller = FakeDoaPoller(in_segment)
        doa_poller_box.append(poller)
        return poller

    factories = _fake_factories(
        latency_window=lambda config: window,
        doa_poller=_doa_poller_factory,
    )

    service_runner = build_service(object(), factories)
    await service_runner(stop=None)

    doa_poller = doa_poller_box[0]
    assert doa_poller.in_segment() is False  # no live frame sent yet

    # The tracker's own clock is real time.monotonic() (D-08's own
    # capture/send clock domain) -- captured_at/sent_at must be real
    # monotonic values, not arbitrary small floats, for in_segment()'s
    # elapsed-time check to mean anything.
    now = time.monotonic()
    captured["on_live_frame_sent"](now - 0.02, now)

    assert window.recorded == [(now - 0.02, now)]
    assert doa_poller.in_segment() is True  # a live frame just arrived


# --- events merging -----------------------------------------------------


@pytest.mark.asyncio
async def test_merge_events_interleaves_both_sources() -> None:
    async def gen_a():
        yield "a1"
        yield "a2"

    async def gen_b():
        yield "b1"

    items = [item async for item in main_module._merge_events(gen_a(), gen_b())]

    assert sorted(items) == ["a1", "a2", "b1"]


@pytest.mark.asyncio
async def test_latency_ticker_yields_only_non_none_drains() -> None:
    class Window:
        def __init__(self) -> None:
            self.calls = 0

        def drain_message(self):
            self.calls += 1
            return "msg" if self.calls == 2 else None

    async def fake_sleep(_interval: float) -> None:
        return None

    window = Window()
    items = []
    async for item in main_module._latency_ticker(window, sleep=fake_sleep):
        items.append(item)
        break

    assert items == ["msg"]
    assert window.calls == 2
