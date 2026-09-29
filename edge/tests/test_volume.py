"""Pi-side volume control (260929-mii): message parsing, target
resolution, the `amixer` runner, the session round trip, and the two config
keys. No hardware and no real `amixer`."""

from __future__ import annotations

import json
import os

import pytest

from atlas_edge.client import run_session
from atlas_edge.config import EdgeConfigError, load_config
from atlas_edge.protocol import ProtocolError, Volume, parse_server_message, volume_result
from atlas_edge.volume import VolumeControl, VolumeError, parse_level, resolve_target, run_amixer

_HELLO = json.dumps(
    {
        "type": "hello",
        "protocol": 1,
        "device_id": 1,
        "sample_rate": 16000,
        "channels": 2,
        "asr_channel": 0,
        "pre_roll_ms": 100,
        "tail_ms": 100,
        "frame_samples": 256,
    }
)


def _message(**fields) -> str:
    return json.dumps({"type": "volume", "id": 1, "min_percent": 30, "max_percent": 100, **fields})


def _relative(direction: str = "up", step: int = 10, **kwargs) -> Volume:
    return Volume(id=1, min_percent=30, max_percent=100, direction=direction, step_percent=step, **kwargs)


# --- resolve_target ----------------------------------------------------------


def test_resolve_target_clamps_and_steps() -> None:
    assert resolve_target(Volume(id=1, min_percent=30, max_percent=100, level=20), None) == 30
    assert resolve_target(_relative("up"), 95) == 100
    assert resolve_target(_relative("down"), 35) == 30
    assert resolve_target(_relative("up"), 50) == 60
    assert resolve_target(Volume(id=1, min_percent=0, max_percent=80, level=90), None) == 80


# --- parse_server_message ----------------------------------------------------


def test_parse_accepts_absolute_and_relative() -> None:
    assert parse_server_message(_message(level=55)) == Volume(id=1, min_percent=30, max_percent=100, level=55)
    assert parse_server_message(_message(direction="down", step_percent=10)) == _relative("down")


@pytest.mark.parametrize(
    "fields",
    [
        {"level": 101},
        {"level": -1},
        {"level": "50"},
        {"level": True},
        {"direction": "sideways", "step_percent": 10},
        {"level": 5, "direction": "up", "step_percent": 10},
        {},
        {"direction": "up", "step_percent": 0},
        {"level": 5, "min_percent": 90, "max_percent": 50},
    ],
)
def test_parse_rejects_bad_volume_messages(fields) -> None:
    payload = {"type": "volume", "id": 1, "min_percent": 30, "max_percent": 100, **fields}
    with pytest.raises(ProtocolError):
        parse_server_message(json.dumps(payload))


def test_parse_rejects_a_missing_id() -> None:
    with pytest.raises(ProtocolError):
        parse_server_message(json.dumps({"type": "volume", "level": 5, "min_percent": 0, "max_percent": 100}))


# --- VolumeControl -----------------------------------------------------------


class _FakeAmixer:
    def __init__(self, *outputs: str) -> None:
        self.calls: "list[list[str]]" = []
        self._outputs = list(outputs)

    async def __call__(self, args) -> str:
        self.calls.append(list(args))
        return self._outputs.pop(0)


async def test_step_up_reads_then_sets() -> None:
    amixer = _FakeAmixer("[50%]", "[60%]")
    level = await VolumeControl("Array", "PCM,0", run=amixer).apply(_relative("up"))
    assert level == 60
    assert amixer.calls == [["-c", "Array", "sget", "PCM,0"], ["-c", "Array", "sset", "PCM,0", "60%"]]


async def test_absolute_level_skips_the_read() -> None:
    amixer = _FakeAmixer("Mono: Playback 18 [45%]")
    level = await VolumeControl("Array", "PCM,0", run=amixer).apply(Volume(id=1, min_percent=30, max_percent=100, level=45))
    assert level == 45
    assert amixer.calls == [["-c", "Array", "sset", "PCM,0", "45%"]]


def test_parse_level() -> None:
    assert parse_level("Front Left: 18 [30%] [-12.00dB] [on]") == 30
    for bad in ("no level here", "[101%]"):
        with pytest.raises(VolumeError):
            parse_level(bad)


async def test_run_amixer_without_the_binary_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(VolumeError, match="amixer is not installed on this device"):
        await run_amixer(["-c", "Array", "sget", "PCM,0"])


async def test_run_amixer_reports_a_non_zero_exit(monkeypatch, tmp_path) -> None:
    script = tmp_path / "amixer"
    script.write_text("#!/bin/sh\necho 'no such card' >&2\nexit 3\n")
    os.chmod(script, 0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(VolumeError, match="code 3: no such card"):
        await run_amixer(["-c", "Array", "sget", "PCM,0"])


async def test_run_amixer_kills_a_slow_amixer(monkeypatch, tmp_path) -> None:
    script = tmp_path / "amixer"
    script.write_text("#!/bin/sh\nexec sleep 5\n")
    os.chmod(script, 0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(VolumeError, match="did not finish"):
        await run_amixer(["sget"], timeout_s=0.1)


# --- run_session round trip --------------------------------------------------


class _FakeWebsocket:
    def __init__(self, incoming=()) -> None:
        self._incoming = list(incoming)
        self.sent: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def recv(self):
        return _HELLO

    async def send(self, item):
        self.sent.append(item)

    async def close(self, code=1000):
        pass

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for item in self._incoming:
            yield item


def _connect(ws):
    def connect(url, **kwargs):
        return ws

    return connect


async def _empty_outbound(hello):
    return
    yield  # pragma: no cover


async def _run(ws, **kwargs) -> None:
    await run_session(
        "wss://svr.test/ws/edge",
        "tok1234",
        make_outbound=_empty_outbound,
        on_reply_audio=lambda data: None,
        connect=_connect(ws),
        **kwargs,
    )


async def test_session_answers_a_volume_message_and_survives_a_bad_one() -> None:
    ws = _FakeWebsocket(
        [_message(level=60), _message(id=2, level=150), json.dumps({"type": "ping", "id": 7, "server_t_ms": 5})]
    )

    async def on_volume(request: Volume) -> int:
        return 60

    await _run(ws, on_volume=on_volume)
    sent = [json.loads(item) for item in ws.sent if isinstance(item, str)]
    assert {"type": "volume.result", "id": 1, "level": 60} in sent
    assert any(item["type"] == "pong" for item in sent)
    assert not any(item.get("id") == 2 for item in sent if item["type"] == "volume.result")


async def test_session_sends_an_error_reply_when_on_volume_raises() -> None:
    ws = _FakeWebsocket([_message(level=60)])

    async def on_volume(request: Volume) -> int:
        raise VolumeError("amixer is not installed on this device")

    await _run(ws, on_volume=on_volume)
    assert json.loads(ws.sent[0]) == json.loads(
        volume_result(1, error="amixer is not installed on this device")
    )


async def test_session_without_on_volume_replies_with_an_error() -> None:
    ws = _FakeWebsocket([_message(level=60)])
    await _run(ws)
    assert json.loads(ws.sent[0])["error"] == "this device has no volume control"


# --- config ------------------------------------------------------------------


def _write(path, text: str) -> None:
    path.write_text('server_url = "wss://atlas.example.test/ws/edge"\ntoken = "tok1234"\n' + text)
    os.chmod(path, 0o600)


def test_config_defaults_and_overrides(tmp_path) -> None:
    path = tmp_path / "config.toml"
    _write(path, "")
    config = load_config(path)
    assert (config.volume_card, config.volume_control) == ("Array", "PCM,0")
    _write(path, 'volume_card = "Other"\nvolume_control = "Master"\n')
    config = load_config(path)
    assert (config.volume_card, config.volume_control) == ("Other", "Master")


@pytest.mark.parametrize("line", ['volume_card = ""', "volume_card = 5", 'volume_control = "-D hw"'])
def test_config_refuses_a_bad_mixer_name(tmp_path, line: str) -> None:
    path = tmp_path / "config.toml"
    _write(path, line + "\n")
    with pytest.raises(EdgeConfigError):
        load_config(path)
