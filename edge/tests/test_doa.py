"""RED for DoaPoller (10-09-PLAN.md Task 1). No real USB hardware -- every
device is a fake, and `sleep` is injected so a test drives poll ticks
without a real clock."""

from __future__ import annotations

import json
import struct

import pytest

from atlas_edge.doa import DoaPoller


class FakeDevice:
    """Returns one pre-baked response per parameter name; raises the
    injected exception (once) if `raise_on` matches."""

    def __init__(self, responses: "dict[str, bytes]", *, raise_times: int = 0) -> None:
        self.responses = responses
        self.raise_times = raise_times
        self.calls: "list[str]" = []
        self._call_count = 0

    def read(self, parameter_name: str) -> bytes:
        self.calls.append(parameter_name)
        self._call_count += 1
        if self._call_count <= self.raise_times:
            raise RuntimeError("simulated read failure")
        return self.responses[parameter_name]


class _StopPolling(Exception):
    """A test-only sentinel: PEP 479 converts a real StopAsyncIteration
    raised inside an async generator into RuntimeError, so ending this
    fake sleep needs its own exception type instead."""


def _fake_read_parameter(device: FakeDevice, parameter):
    # Mirrors xvf3800.read_parameter's own contract: the caller gets the
    # payload with the leading status byte already stripped.
    return device.read(parameter.name)[1:]


@pytest.fixture(autouse=True)
def _patch_read_parameter(monkeypatch):
    import atlas_edge.doa as doa_module

    monkeypatch.setattr(doa_module.xvf3800, "read_parameter", _fake_read_parameter)


def _make_sleep(ticks: int):
    """A fake `asyncio.sleep` that lets `ticks` calls through, then raises
    `_StopPolling` to end the test's `async for` cleanly."""
    remaining = [ticks]

    async def _sleep(_interval: float) -> None:
        if remaining[0] <= 0:
            raise _StopPolling
        remaining[0] -= 1

    return _sleep


@pytest.mark.asyncio
async def test_yields_nothing_while_not_in_segment() -> None:
    device = FakeDevice({"DOA_VALUE": bytes([0]) + struct.pack("<HH", 90, 1)})
    poller = DoaPoller(device, "DOA_VALUE", in_segment=lambda: False, sleep=_make_sleep(5))

    messages = []
    with pytest.raises(_StopPolling):
        async for message in poller.messages():
            messages.append(message)

    assert messages == []
    assert device.calls == []


@pytest.mark.asyncio
async def test_doa_value_yields_one_azimuth_in_0_360() -> None:
    device = FakeDevice({"DOA_VALUE": bytes([0]) + struct.pack("<HH", 90, 1)})
    poller = DoaPoller(device, "DOA_VALUE", in_segment=lambda: True, sleep=_make_sleep(2))

    messages = []
    with pytest.raises(_StopPolling):
        async for message in poller.messages():
            messages.append(message)

    assert len(messages) == 2
    payload = json.loads(messages[0])
    assert payload["type"] == "doa"
    assert payload["azimuth_deg"] == [90.0]
    assert payload["speech_energy"] == []
    for azimuth in payload["azimuth_deg"]:
        assert 0.0 <= azimuth < 360.0


@pytest.mark.asyncio
async def test_aec_azimuth_values_carries_all_four_beams() -> None:
    import math

    radians = (0.0, math.pi / 2, math.pi, 3.0 * math.pi / 2.0)
    device = FakeDevice({"AEC_AZIMUTH_VALUES": bytes([0]) + struct.pack("<4f", *radians)})
    poller = DoaPoller(device, "AEC_AZIMUTH_VALUES", in_segment=lambda: True, sleep=_make_sleep(1))

    messages = []
    with pytest.raises(_StopPolling):
        async for message in poller.messages():
            messages.append(message)

    payload = json.loads(messages[0])
    assert [round(a, 1) for a in payload["azimuth_deg"]] == [0.0, 90.0, 180.0, 270.0]
    assert payload["speech_energy"] == []


@pytest.mark.asyncio
async def test_aec_azimuth_values_with_speech_energy_carries_four_energies() -> None:
    import math

    radians = (0.0, 0.0, 0.0, 0.0)
    energies = (1.0, 2.0, 3.0, 4.0)
    device = FakeDevice(
        {
            "AEC_AZIMUTH_VALUES": bytes([0]) + struct.pack("<4f", *radians),
            "AEC_SPENERGY_VALUES": bytes([0]) + struct.pack("<4f", *energies),
        }
    )
    poller = DoaPoller(
        device,
        "AEC_AZIMUTH_VALUES",
        speech_energy_parameter_name="AEC_SPENERGY_VALUES",
        in_segment=lambda: True,
        sleep=_make_sleep(1),
    )

    messages = []
    with pytest.raises(_StopPolling):
        async for message in poller.messages():
            messages.append(message)

    payload = json.loads(messages[0])
    assert payload["speech_energy"] == pytest.approx(list(energies))
    _ = math  # imported only for pi below in this test's own radians tuple


@pytest.mark.asyncio
async def test_a_failing_read_logs_once_per_streak_and_keeps_polling(caplog) -> None:
    import logging

    device = FakeDevice({"DOA_VALUE": bytes([0]) + struct.pack("<HH", 90, 1)}, raise_times=2)
    poller = DoaPoller(device, "DOA_VALUE", in_segment=lambda: True, sleep=_make_sleep(3))

    messages = []
    with caplog.at_level(logging.WARNING):
        with pytest.raises(_StopPolling):
            async for message in poller.messages():
                messages.append(message)

    # Two failed polls, then one that succeeds -- only the real read
    # yields a message.
    assert len(messages) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # once per failure streak, not once per failure


@pytest.mark.asyncio
async def test_never_raises_into_the_caller() -> None:
    device = FakeDevice({"DOA_VALUE": bytes([0]) + struct.pack("<HH", 90, 1)}, raise_times=10_000)
    poller = DoaPoller(device, "DOA_VALUE", in_segment=lambda: True, sleep=_make_sleep(5))

    messages = []
    with pytest.raises(_StopPolling):
        async for message in poller.messages():
            messages.append(message)

    assert messages == []  # every poll failed, but the generator itself never raised
