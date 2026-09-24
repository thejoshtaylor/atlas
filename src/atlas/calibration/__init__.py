"""Echo-path calibration: what was measured, when, and in which room.

`record.py` (plan 02-10) is `EchoCalibration`, its `save`/`load`/
`is_stale`, and the schema version and default directory it persists
under. `runner.py` (plan 02-11) is `run_echo_calibration`, the one
implementation that calls `measure_echo_path` (`audio/echo_path.py`) and
writes one of these -- a command-line script and an HTTP route are both
callers of it, never a second measurement. Plan 02-12 adds the reader that
loads the latest one at startup.

`runner.py` is deliberately **not** re-exported here: it imports
`atlas.config` for `CalibrationConfig`/`CameraConfig`, and `config.py`
itself imports `DEFAULT_CALIBRATION_DIR` from `record.py` -- importing
`runner.py` at this package's own import time would make importing
`atlas.config` circular through this very module. A caller reaches
the runner directly: `from atlas.calibration.runner import
run_echo_calibration`.
"""

from __future__ import annotations

from atlas.calibration.record import (
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
