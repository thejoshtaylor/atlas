"""DoA readings that ride the speech segment (D-17): `DoaPoller` polls the
XVF3800's vendor control interface at `poll_hz` while `in_segment()` says
a segment is open, and stays silent otherwise -- a quiet room sends no
`doa` messages at all, matching D-05's "a quiet room sends nothing" rule
for the audio path itself.

`poll_hz` defaults to 5 Hz: one reading every 200 ms is enough to follow a
talker who moves, and 5 USB control transfers a second cost nothing on a
Pi 4.

A failed read is logged once per failure *streak* (matching
`client.py::run_forever`'s own auth-refusal convention) and never raises
into the audio path -- `capture.frames()` keeps flowing even if this
Pi's XVF3800 control interface wedges.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from atlas_edge import protocol, xvf3800

logger = logging.getLogger(__name__)

DEFAULT_POLL_HZ = 5.0


class DoaPoller:
    def __init__(
        self,
        device,
        parameter_name: str,
        *,
        speech_energy_parameter_name: "str | None" = None,
        poll_hz: float = DEFAULT_POLL_HZ,
        in_segment: Callable[[], bool],
        sleep: Callable[[float], "asyncio.Future | None"] = asyncio.sleep,
    ) -> None:
        self._device = device
        self._parameter = xvf3800.PARAMETERS[parameter_name]
        self._spenergy_parameter = (
            xvf3800.PARAMETERS[speech_energy_parameter_name]
            if speech_energy_parameter_name is not None
            else None
        )
        self._poll_hz = poll_hz
        self._in_segment = in_segment
        self._sleep = sleep
        self._fail_streak = 0

    def _decode(self, data: bytes, spenergy_data: "bytes | None") -> "tuple[list[float], list[float] | None]":
        if self._parameter.name == "DOA_VALUE":
            azimuth, _speech_detected = xvf3800.decode_doa_value(data)
            return [float(azimuth)], None
        if self._parameter.name == "AEC_AZIMUTH_VALUES":
            azimuths = [float(a) for a in xvf3800.decode_azimuth_values(data)]
            energies = (
                [float(e) for e in xvf3800.decode_spenergy(spenergy_data)]
                if spenergy_data is not None
                else None
            )
            return azimuths, energies
        raise ValueError(f"unsupported DoA parameter: {self._parameter.name!r}")

    async def messages(self):
        """Yields `protocol.doa(...)` strings at `poll_hz`, only while
        `in_segment()` is true. Runs forever -- the caller (`__main__.py`'s
        `events_hook`) is what bounds this generator's lifetime."""
        interval = 1.0 / self._poll_hz
        while True:
            await self._sleep(interval)
            if not self._in_segment():
                continue
            try:
                data = xvf3800.read_parameter(self._device, self._parameter)
                spenergy_data = (
                    xvf3800.read_parameter(self._device, self._spenergy_parameter)
                    if self._spenergy_parameter is not None
                    else None
                )
            except Exception as exc:  # noqa: BLE001 -- never raise into the audio path
                self._fail_streak += 1
                if self._fail_streak == 1:
                    logger.warning("DoA read failed, will keep polling: %s", exc)
                continue
            self._fail_streak = 0
            azimuths, energies = self._decode(data, spenergy_data)
            yield protocol.doa(azimuths, energies)
