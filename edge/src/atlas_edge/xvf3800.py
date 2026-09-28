"""USB vendor-control reads for the Seeed reSpeaker XVF3800.

This is the runtime copy of `edge/spike/xvf3800_control.py` (10-01/10-03),
moved into the service package unchanged in behavior per plan
10-09-PLAN.md Task 1 -- the spike module itself stays put as spike tooling,
so this file is a deliberate duplicate, not an import of it.

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

A write has the same shape with `CTRL_OUT` (bmRequestType 0x40), a wValue of
`command_id` with no `0x80` read bit, and the payload bytes in place of
`wLength`. A write has no status byte.

The first returned byte of a read is a status byte. A non-zero status byte means the
device rejected the read, and this module raises rather than returning the
bytes, so a caller never mistakes a rejected read for real data (T-10-SP2).

This module imports `usb.core` only inside `find_device`, so importing this
module never requires a USB backend or a connected device -- the whole
`edge/` suite runs on a dev host with neither attached.
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
_CTRL_OUT = 0x00
_BM_REQUEST_TYPE_WRITE = _CTRL_OUT | _CTRL_TYPE_VENDOR | _CTRL_RECIPIENT_DEVICE
_B_REQUEST = 0

_TIMEOUT_MS = 1000  # not the SDK example's 100_000ms -- T-10-SP2
# The device answers 64 (SERVICER_COMMAND_RETRY) while its control servicer is
# busy; the vendor's xvf_host retries the same read. Bounded so a wedged
# device still raises instead of spinning (orchestrator directive 2,
# 10-SPIKE.md hardware finding 3, commit 915ba8a).
_STATUS_RETRY = 64
_MAX_RETRIES = 100


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
    # (2026-09-25): resource 33, command 80, 4 float32 values. Not on the
    # Seeed wiki page -- only in the vendor's own command table.
    "AEC_SPENERGY_VALUES": Parameter(
        "AEC_SPENERGY_VALUES", resource_id=33, command_id=80, length=16, kind="float32x4"
    ),
    # LED ring control. Ids come from the vendor command table (xvf_host.py,
    # fetched 2026-09-28). LED_EFFECT values: 0 off, 1 breath, 2 rainbow,
    # 3 single color, 4 doa, 5 ring. LED_COLOR is one 0xRRGGBB word for
    # effect 3. LED_RING_COLOR is twelve 0xRRGGBB words, one per LED, for
    # effect 5.
    "LED_EFFECT": Parameter("LED_EFFECT", resource_id=20, command_id=12, length=1, kind="uint8"),
    "LED_COLOR": Parameter("LED_COLOR", resource_id=20, command_id=16, length=4, kind="uint32"),
    "LED_RING_COLOR": Parameter("LED_RING_COLOR", resource_id=20, command_id=19, length=48, kind="uint32x12"),
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
    leading status byte stripped. Retries while the device answers 64
    (busy), then raises `XvfControlError` if the status byte is non-zero."""
    w_value = 0x80 | parameter.command_id
    w_index = parameter.resource_id
    length = parameter.length + 1  # status byte first

    for _ in range(_MAX_RETRIES):
        raw = device.ctrl_transfer(
            _BM_REQUEST_TYPE, _B_REQUEST, w_value, w_index, length, _TIMEOUT_MS
        )
        payload = bytes(raw)
        status, data = payload[0], payload[1:]
        if status != _STATUS_RETRY:
            break
    if status != 0:
        raise XvfControlError(parameter, status)
    return data


def write_parameter(device, parameter: Parameter, payload: bytes) -> None:
    """Write `payload` to `parameter` on `device`. A write has no status
    byte, and its wValue has no `0x80` read bit. Raises `ValueError`,
    before any transfer, when the payload length is not the length the
    parameter takes."""
    if len(payload) != parameter.length:
        raise ValueError(
            f"{parameter.name}: payload is {len(payload)} bytes, the parameter takes {parameter.length}"
        )
    device.ctrl_transfer(
        _BM_REQUEST_TYPE_WRITE,
        _B_REQUEST,
        parameter.command_id,
        parameter.resource_id,
        payload,
        _TIMEOUT_MS,
    )


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
