"""USB vendor-control reads for the Seeed reSpeaker XVF3800.

Vendor control interface, per the Seeed Python SDK documentation
(wiki.seeedstudio.com/respeaker_xvf3800_python_sdk/) and the vendor's own
`host_control/` command table
(github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py,
fetched 2026-09-25 -- the dict literal at module top names each command's
`(resource_id, command_id, length, access, format, description)` tuple):

    dev.ctrl_transfer(
        CTRL_IN | CTRL_TYPE_VENDOR | CTRL_RECIPIENT_DEVICE,
        0,                       # bRequest
        0x80 | command_id,       # wValue
        resource_id,             # wIndex
        length + 1,              # wLength -- the status byte comes first
        timeout,
    )

The first returned byte is a status byte. A non-zero status byte means the
device rejected the read, and this module raises rather than returning the
bytes, so a caller never mistakes a rejected read for real data (T-10-SP2).

This module imports `usb.core` only inside `find_device`, so importing this
module -- and running `--help` on the spike CLI that imports it -- never
requires a USB backend or a connected device (D-18's spike must run its
`--help` on a dev host with no XVF3800 attached).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

VENDOR_ID = 0x2886
PRODUCT_ID = 0x001A

# USB control-transfer constants (bmRequestType bits), spelled out rather
# than imported from usb.util so this module's constants are readable
# without opening pyusb's own source.
_CTRL_IN = 0x80
_CTRL_TYPE_VENDOR = 0x40
_CTRL_RECIPIENT_DEVICE = 0x00
_BM_REQUEST_TYPE = _CTRL_IN | _CTRL_TYPE_VENDOR | _CTRL_RECIPIENT_DEVICE
_B_REQUEST = 0

_TIMEOUT_MS = 1000  # not the SDK example's 100_000ms -- T-10-SP2


@dataclass(frozen=True)
class Parameter:
    """One vendor-control parameter: the resource/command id pair the
    XVF3800's control interface expects, its payload length in bytes
    (excluding the status byte), and a human label for error messages."""

    name: str
    resource_id: int
    command_id: int
    length: int
    kind: str


PARAMETERS: dict[str, Parameter] = {
    "VERSION": Parameter("VERSION", resource_id=48, command_id=0, length=3, kind="uint8"),
    "DOA_VALUE": Parameter("DOA_VALUE", resource_id=20, command_id=18, length=4, kind="uint16x2"),
    "AEC_AZIMUTH_VALUES": Parameter(
        "AEC_AZIMUTH_VALUES", resource_id=33, command_id=75, length=16, kind="float32x4"
    ),
    # Ids confirmed live against
    # github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/python_control/xvf_host.py
    # this session (2026-09-25): resource 33, command 80, 4 float32 values.
    # Not on the Seeed wiki page -- only in the vendor's own command table.
    "AEC_SPENERGY_VALUES": Parameter(
        "AEC_SPENERGY_VALUES", resource_id=33, command_id=80, length=16, kind="float32x4"
    ),
}


class XvfNotFound(RuntimeError):
    """No XVF3800 (VID 0x2886, PID 0x001A) is attached."""


class XvfControlError(RuntimeError):
    """The device returned a non-zero status byte for a parameter read."""

    def __init__(self, parameter: Parameter, status: int) -> None:
        self.parameter = parameter
        self.status = status
        super().__init__(
            f"{parameter.name}: device returned non-zero status byte {status}"
        )


def find_device():
    """Locate the XVF3800 over USB. Raises `XvfNotFound` if none is
    attached. Imports `usb.core` here, not at module scope, so this module
    loads on a host with no pyusb backend installed."""
    import usb.core

    device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if device is None:
        raise XvfNotFound(
            f"no XVF3800 found (VID {VENDOR_ID:#06x}, PID {PRODUCT_ID:#06x})"
        )
    return device


def read_parameter(device, parameter: Parameter) -> bytes:
    """Read `parameter` from `device` and return its payload, with the
    leading status byte stripped. Raises `XvfControlError` if the status
    byte is non-zero."""
    w_value = 0x80 | parameter.command_id
    w_index = parameter.resource_id
    length = parameter.length + 1  # status byte first

    raw = device.ctrl_transfer(
        _BM_REQUEST_TYPE, _B_REQUEST, w_value, w_index, length, _TIMEOUT_MS
    )
    payload = bytes(raw)
    status, data = payload[0], payload[1:]
    if status != 0:
        raise XvfControlError(parameter, status)
    return data


def decode_version(data: bytes) -> tuple[int, int, int]:
    """VERSION: 3 uint8 values -- major, minor, patch."""
    return (data[0], data[1], data[2])


def decode_doa_value(data: bytes) -> tuple[int, bool]:
    """DOA_VALUE: two little-endian uint16 words -- azimuth in degrees
    (0-359), then a VAD flag (1 = speech)."""
    azimuth, vad_flag = struct.unpack_from("<HH", data)
    return (azimuth, vad_flag == 1)


def decode_azimuth_values(data: bytes) -> tuple[float, float, float, float]:
    """AEC_AZIMUTH_VALUES: four little-endian float32 radians -- focused
    beam 1, focused beam 2, free-running beam, auto-selected beam. Returned
    as degrees in [0, 360)."""
    radians = struct.unpack_from("<4f", data)
    return tuple((r * 180.0 / 3.141592653589793) % 360.0 for r in radians)  # type: ignore[return-value]


def decode_spenergy(data: bytes) -> tuple[float, float, float, float]:
    """AEC_SPENERGY_VALUES: four little-endian float32 speech-energy
    values -- focused beam 1, focused beam 2, free-running beam,
    auto-selected beam."""
    return struct.unpack_from("<4f", data)  # type: ignore[return-value]
