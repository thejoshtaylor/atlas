"""RED for build_service/main (10-09-PLAN.md Task 2). Every test token
literal below stays under 8 characters (tests/test_repo_hygiene.py's
credential scan). No test opens a real audio device or USB backend --
every factory is a fake, injected through `main(..., factories=...)`.
"""

from __future__ import annotations

import asyncio
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
        self.stops: "list[int]" = []

    def stop_playback(self, fade_ms: int) -> int:
        self.stops.append(fade_ms)
        return 42

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
        self.cleared: "list[int]" = []

    def clear(self, keep_bytes: int = 0) -> None:
        self.cleared.append(keep_bytes)

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


class FakeVolume:
    async def apply(self, request) -> int:
        return 50


class FakeLed:
    def __init__(self) -> None:
        self.states: "list[str]" = []
        self.closed = False

    def set_state(self, state: str) -> None:
        self.states.append(state)

    async def close(self) -> None:
        self.closed = True


async def _fake_runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop, on_led=None, on_volume=None, on_stop=None):
    await stop.wait()


def _fake_factories(**overrides):
    factories = {
        "capture": lambda config: FakeCapture(),
        "playback": lambda config, capture: FakePlayback(capture.enqueue_playback),
        "gate_factory": lambda config: (lambda: object()),
        "doa_poller": lambda config, in_segment: FakeDoaPoller(in_segment),
        "latency_window": lambda config: FakeWindow(),
        "led": lambda config, playback: FakeLed(),
        "music": lambda config, duck_active: None,
        "volume": lambda config: FakeVolume(),
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


@pytest.mark.asyncio
async def test_the_service_turns_the_ring_off_first_feeds_led_states_and_closes_the_led() -> None:
    led = FakeLed()

    async def runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop, on_led=None, on_volume=None, on_stop=None):
        on_led("listening")

    service_runner = build_service(
        object(),
        _fake_factories(led=lambda config, playback: led, runner=runner),
    )
    await service_runner(stop=None)

    assert led.states == ["idle", "listening"]
    assert led.closed is True


@pytest.mark.asyncio
async def test_on_stop_fades_the_capture_queue_clears_the_playback_and_leaves_the_music() -> None:
    capture = FakeCapture()
    playback = FakePlayback(capture.enqueue_playback)
    music = FakeMusic(lambda: False)
    captured: dict = {}

    async def runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop, on_led=None, on_volume=None, on_stop=None):
        captured["on_stop"] = on_stop
        on_stop(120)

    service_runner = build_service(
        object(),
        _fake_factories(
            capture=lambda config: capture,
            playback=lambda config, cap: playback,
            music=lambda config, duck_active: music,
            runner=runner,
        ),
    )
    await service_runner(stop=None)

    assert captured["on_stop"] is not None
    assert capture.stops == [120]
    assert playback.cleared == [42]
    assert music.events == ["start", "stop"]  # only the service lifecycle, no stop call


@pytest.mark.asyncio
async def test_the_led_closes_when_the_runner_raises() -> None:
    led = FakeLed()

    async def runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop, on_led=None, on_volume=None, on_stop=None):
        raise RuntimeError("runner failed")

    service_runner = build_service(
        object(),
        _fake_factories(led=lambda config, playback: led, runner=runner),
    )
    with pytest.raises(RuntimeError, match="runner failed"):
        await service_runner(stop=None)

    assert led.closed is True


# --- music mixer wiring ----------------------------------------------------------


class FakeMusic:
    def __init__(self, duck_active) -> None:
        self.duck_active = duck_active
        self.events: "list[str]" = []

    def mix(self, block: bytes, reply_active: bool) -> bytes:
        return block

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")


@pytest.mark.asyncio
async def test_music_is_wired_started_and_stopped_and_ducks_on_turn_state() -> None:
    box: "dict" = {}
    capture = FakeCapture()
    led = FakeLed()
    seen: "dict" = {}

    def music_factory(config, duck_active):
        box["music"] = FakeMusic(duck_active)
        return box["music"]

    async def runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop, on_led=None, on_volume=None, on_stop=None):
        music = box["music"]
        seen["started"] = list(music.events)
        seen["initial"] = music.duck_active()
        for state, expected in (
            ("listening", True),
            ("thinking", True),
            ("replying", True),
            ("idle", False),
        ):
            on_led(state)
            seen[state] = music.duck_active()
            assert seen[state] is expected
        now = time.monotonic()
        on_live_frame_sent(now - 0.02, now)
        # A live frame (an open VAD segment) with no turn in progress does
        # not duck music, because the VAD fires on music.
        seen["segment"] = music.duck_active()

    service_runner = build_service(
        object(),
        _fake_factories(capture=lambda config: capture, led=lambda c, p: led,
                        music=music_factory, runner=runner),
    )
    await service_runner(stop=None)

    music = box["music"]
    assert capture.mixer == music.mix
    assert seen["started"] == ["start"]
    assert seen["initial"] is False
    assert seen["segment"] is False
    assert music.events == ["start", "stop"]
    assert led.states == ["idle", "listening", "thinking", "replying", "idle"]


@pytest.mark.asyncio
async def test_music_is_stopped_when_the_runner_raises() -> None:
    box: "dict" = {}

    def music_factory(config, duck_active):
        box["music"] = FakeMusic(duck_active)
        return box["music"]

    async def runner(config, *, make_outbound, on_reply_audio, on_live_frame_sent, stop, on_led=None, on_volume=None, on_stop=None):
        raise RuntimeError("runner failed")

    service_runner = build_service(object(), _fake_factories(music=music_factory, runner=runner))
    with pytest.raises(RuntimeError):
        await service_runner(stop=None)
    assert box["music"].events == ["start", "stop"]


@pytest.mark.asyncio
async def test_no_music_means_no_mixer_and_run_still_works() -> None:
    capture = FakeCapture()
    stop = asyncio.Event()
    stop.set()
    service_runner = build_service(object(), _fake_factories(capture=lambda config: capture))
    await service_runner(stop=stop)
    assert not hasattr(capture, "mixer")


def test_default_music_is_none_when_the_fifo_is_empty() -> None:
    class Cfg:
        music_fifo = ""
        music_duck_level = 0.2
        music_duck_ramp_ms = 150

    assert main_module._default_music(Cfg(), lambda: False) is None
