"""`EchoCalibration`: what plan 02-10's `measure_echo_path` found, when, and
in which room.

A frozen dataclass in the shape `config.py`'s sections already use, but
this is a *record* of a measurement taken once, not a config section
re-read from a YAML file on every startup -- `save`/`load` read and write
one JSON file rather than `from_config` reading one YAML section, and
`is_stale` has no `config.py` analogue at all: no config section carries an
age that can expire.

Software cannot detect that somebody moved the camera, and this module
does not pretend to. What it persists instead is provenance a human can
read: when the measurement was taken, which source it came from, the
camera's own declared (non-secret) format, and the operator's free-text
note about the room and the hardware. Phase 3's wizard shows those and
offers a re-run; `is_stale` gives it a default prompt to hang that offer
on -- it is a staleness clock, never a movement detector.

The credential boundary is structural, not a matter of discipline. The
RTSP URL carries this camera's username and password inline
(`CameraConfig.rtsp_url`), this record is written to disk, and plan
02-11 serves it over HTTP to a browser -- so no field here can hold a URL.
`placement_note` is the one free-text field, and it is the one field
validated on construction: a value containing a URL scheme separator or a
userinfo marker is rejected by `CalibrationError`, not accepted and hoped
against. No fingerprint is derived from the RTSP URL anywhere in this
module either -- a hash of a short structured string is guessable offline
and buys nothing here.

`load` never raises for the three ordinary bad states this module
recognizes -- file absent, schema version mismatched, JSON malformed (or
otherwise unreadable as this shape) -- because plan 02-12 calls it during
application startup, and a calibration that cannot be read must leave
barge-in exactly as it already is rather than stopping the house from
listening. Each of those three states returns `None` with a logged reason
instead.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("spire_voice.calibration.record")

# Bumped whenever a field is added, removed, or reinterpreted. `load`
# refuses a file recorded under a different version rather than handing a
# stale shape to the barge-in gate that reads it (plan 02-12).
SCHEMA_VERSION = 1

# Where a real deployment stores calibration records by default, mirroring
# `SessionConfig.dir`'s ("/data/sessions") and `tts.cache_dir`'s
# ("/data/tts-cache") own "/data/..." convention -- already covered by the
# repository's existing `data/` ignore rule (`.gitignore`,
# `tests/test_repo_hygiene.py`), with no new ignore entry needed.
DEFAULT_CALIBRATION_DIR = "/data/calibration"

_URL_SCHEME_SEPARATOR = "://"
_USERINFO_MARKER = "@"


class CalibrationError(Exception):
    """Raised instead of returned, so a caller cannot silently persist a
    record this module refuses to hold."""


def _validate_placement_note(note: str) -> None:
    """Reject a placement note shaped like it might carry a URL -- the RTSP
    URL is the one credential-bearing string in this system that could
    plausibly be pasted into a free-text field, and this is the structural
    check that keeps it out rather than trusting an operator not to."""
    if _URL_SCHEME_SEPARATOR in note:
        raise CalibrationError(
            f"placement note may not contain a URL scheme separator ({_URL_SCHEME_SEPARATOR!r}) "
            "-- this field is never allowed to carry the camera's RTSP URL"
        )
    if _USERINFO_MARKER in note:
        raise CalibrationError(
            f"placement note may not contain a userinfo marker ({_USERINFO_MARKER!r}) "
            "-- this field is never allowed to carry a credential"
        )


@dataclass(frozen=True)
class EchoCalibration:
    """One measured echo path, with enough provenance for a human to judge
    whether it still applies. Every field here is either a measured
    number, an identifier, or operator-supplied text -- never a URL and
    never a credential (see the module docstring's credential-boundary
    paragraph).
    """

    schema_version: int
    probe_format_version: int
    probe_seed: int
    source: str
    delay_s: float
    confidence: float
    echo_level: float
    gain: float
    agc_verdict: str
    segment_levels: tuple[float, ...]
    encoding: str
    sample_rate: int
    channels: int
    placement_note: str
    taken_at: datetime

    def __post_init__(self) -> None:
        _validate_placement_note(self.placement_note)

    def is_stale(self, now: datetime, max_age_days: int) -> bool:
        """`True` once `now` is more than `max_age_days` past `taken_at` --
        read from this record's own recorded timestamp, never the file's
        mtime, which a backup, a copy, or a container restore can rewrite
        without anything having actually changed about the room
        (`session/retention.py` applies this same discipline to a
        session's age). Exactly at the boundary is not yet stale; strictly
        past it is.
        """
        return (now - self.taken_at) > timedelta(days=max_age_days)

    def _to_json_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["segment_levels"] = list(self.segment_levels)
        payload["taken_at"] = self.taken_at.astimezone(timezone.utc).isoformat()
        return payload

    def save(self, path: str | Path) -> None:
        """Write this record as JSON, creating the parent directory if it
        does not exist yet -- the same eager-directory-creation habit
        `session/recorder.py`'s `SessionRecorder` already uses."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._to_json_dict(), sort_keys=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EchoCalibration | None":
        """Read a record back, or `None` for any of three ordinary bad
        states -- never raised, because a calibration that cannot be read
        must leave the caller exactly where an absent calibration would.
        """
        target = Path(path)
        if not target.exists():
            return None

        try:
            raw = target.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "calibration record unreadable, ignoring", extra={"path": str(target), "reason": str(exc)}
            )
            return None

        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            logger.warning(
                "calibration record schema version mismatch, ignoring",
                extra={"path": str(target), "found_version": data.get("schema_version") if isinstance(data, dict) else None},
            )
            return None

        try:
            payload = dict(data)
            payload["segment_levels"] = tuple(payload["segment_levels"])
            payload["taken_at"] = datetime.fromisoformat(payload["taken_at"])
            return cls(**payload)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "calibration record malformed, ignoring", extra={"path": str(target), "reason": str(exc)}
            )
            return None
