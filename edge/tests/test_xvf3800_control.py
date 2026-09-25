"""Proves the XVF3800 vendor-control request shape and payload decoding
against a fake device object -- no real USB hardware anywhere in this
file."""

from __future__ import annotations

import struct
import sys
import types

import pytest

import xvf3800_control
from xvf3800_control import (
    PARAMETERS,
    XvfControlError,
    XvfNotFound,
    decode_azimuth_values,
    decode_doa_value,
    decode_spenergy,
    decode_version,
    read_parameter,
)


class FakeDevice:
    """Records the exact `ctrl_transfer` arguments it was called with, and
    returns a pre-baked response (status byte first, per the vendor's own
    request shape)."""

    def __init__(self, response: bytes) -> None:
        self.response = response
        self.calls: list[tuple[int, int, int, int, int, int]] = []

    def ctrl_transfer(self, bmRequestType, bRequest, wValue, wIndex, wLength, timeout):
        self.calls.append((bmRequestType, bRequest, wValue, wIndex, wLength, timeout))
        return self.response


class BusyThenReadyDevice(FakeDevice):
    """Answers status 64 (busy) `busy` times before the real response."""

    def __init__(self, response: bytes, busy: int) -> None:
        super().__init__(response)
        self.busy = busy

    def ctrl_transfer(self, *args):
        super().ctrl_transfer(*args)
        if len(self.calls) <= self.busy:
            return bytes([64]) + bytes(len(self.response) - 1)
        return self.response


class TestReadParameter:
    def test_retries_while_the_device_answers_busy(self) -> None:
        response = bytes([0]) + struct.pack("<HH", 90, 1)
        device = BusyThenReadyDevice(response, busy=3)

        assert read_parameter(device, PARAMETERS["DOA_VALUE"]) == struct.pack("<HH", 90, 1)
        assert len(device.calls) == 4

    def test_gives_up_on_a_device_that_stays_busy(self) -> None:
        device = BusyThenReadyDevice(bytes([0, 0, 0, 0, 0]), busy=10_000)

        with pytest.raises(XvfControlError) as excinfo:
            read_parameter(device, PARAMETERS["DOA_VALUE"])
        assert excinfo.value.status == 64

    def test_sends_the_documented_request_shape(self) -> None:
        parameter = PARAMETERS["DOA_VALUE"]
        response = bytes([0]) + struct.pack("<HH", 90, 1)
        device = FakeDevice(response)

        data = read_parameter(device, parameter)

        assert device.calls == [
            (0xC0, 0, 0x80 | parameter.command_id, parameter.resource_id, parameter.length + 1, 1000)
        ]
        assert data == struct.pack("<HH", 90, 1)

    def test_raises_xvfcontrolerror_on_nonzero_status(self) -> None:
        parameter = PARAMETERS["VERSION"]
        response = bytes([1, 0, 0, 0])  # status byte 1 -- rejected
        device = FakeDevice(response)

        with pytest.raises(XvfControlError) as excinfo:
            read_parameter(device, parameter)

        assert parameter.name in str(excinfo.value)
        assert excinfo.value.status == 1

    def test_uses_a_1000ms_timeout_not_the_sdk_examples_100s(self) -> None:
        parameter = PARAMETERS["VERSION"]
        device = FakeDevice(bytes([0, 2, 0, 6]))

        read_parameter(device, parameter)

        (_, _, _, _, _, timeout) = device.calls[0]
        assert timeout == 1000


class TestDecoders:
    def test_decode_doa_value(self) -> None:
        assert decode_doa_value(b"\x5a\x00\x01\x00") == (90, True)

    def test_decode_doa_value_no_speech(self) -> None:
        assert decode_doa_value(b"\x5a\x00\x00\x00") == (90, False)

    def test_decode_version(self) -> None:
        assert decode_version(bytes([2, 0, 6])) == (2, 0, 6)

    def test_decode_azimuth_values(self) -> None:
        import math

        radians = (0.0, math.pi / 2, math.pi, 3.0 * math.pi / 2.0)
        data = struct.pack("<4f", *radians)
        degrees = decode_azimuth_values(data)
        assert [round(d, 1) for d in degrees] == [0.0, 90.0, 180.0, 270.0]

    def test_decode_spenergy(self) -> None:
        values = (1.0, 2.0, 3.0, 4.0)
        data = struct.pack("<4f", *values)
        assert decode_spenergy(data) == pytest.approx(values)


class TestFindDevice:
    def test_raises_xvfnotfound_when_no_device_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_core = types.SimpleNamespace(find=lambda idVendor, idProduct: None)
        fake_usb = types.ModuleType("usb")
        fake_usb.core = fake_core  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "usb", fake_usb)
        monkeypatch.setitem(sys.modules, "usb.core", fake_core)

        with pytest.raises(XvfNotFound):
            xvf3800_control.find_device()
