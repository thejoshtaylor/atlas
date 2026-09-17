"""Echo-path calibration: what was measured, when, and in which room.

`record.py` is the whole of this package so far -- `EchoCalibration`, its
`save`/`load`/`is_stale`, and the schema version and default directory it
persists under. Plan 02-11 adds the runner that calls `measure_echo_path`
(`audio/echo_path.py`) and writes one of these; plan 02-12 adds the reader
that loads one at startup.
"""

from __future__ import annotations

from spire_voice.calibration.record import (
    DEFAULT_CALIBRATION_DIR,
    SCHEMA_VERSION,
    CalibrationError,
    EchoCalibration,
)

__all__ = [
    "DEFAULT_CALIBRATION_DIR",
    "SCHEMA_VERSION",
    "CalibrationError",
    "EchoCalibration",
]
