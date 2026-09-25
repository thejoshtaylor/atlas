"""Pi-side config: a TOML file naming the server URL, the bearer device
token, and Silero/device overrides -- with a permission check the same
rule OpenSSH applies to a private key (D-03): this file holds a bearer
credential for the house's microphone.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field

from atlas_edge.client import validate_server_url

DEFAULT_CAPTURE_DEVICE = "reSpeaker"
DEFAULT_PLAYBACK_DEVICE = "reSpeaker"
DEFAULT_VAD_MODEL_PATH = "/var/lib/atlas-edge/silero_vad.onnx"
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_MIN_SILENCE_MS = 250


class EdgeConfigError(Exception):
    """Raised by `load_config` for a config file readable by group or
    others, a missing required key, or a `server_url` that
    `validate_server_url` refuses."""


@dataclass(frozen=True)
class EdgeConfig:
    server_url: str
    # `repr=False`: a config's repr must never leak the bearer token into a
    # log line or an exception message (T-10-26).
    token: str = field(repr=False)
    capture_device: str = DEFAULT_CAPTURE_DEVICE
    playback_device: str = DEFAULT_PLAYBACK_DEVICE
    vad_model_path: str = DEFAULT_VAD_MODEL_PATH
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    vad_min_silence_ms: int = DEFAULT_VAD_MIN_SILENCE_MS
    allow_plaintext: bool = False


def load_config(path: "str | os.PathLike[str]") -> EdgeConfig:
    """Read and validate the config file at `path`. Raises
    `EdgeConfigError` if the file is readable by group or others (the
    OpenSSH private-key rule, D-03), if `server_url` or `token` is
    missing, or if `server_url` fails `validate_server_url`."""
    mode = os.stat(path).st_mode
    if mode & 0o077:
        raise EdgeConfigError(
            f"{path} is readable by group or others (mode {oct(mode & 0o777)}) -- "
            "chmod 600 it. It holds this Pi's bearer token for /ws/edge (D-03)."
        )

    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    try:
        server_url = raw["server_url"]
        token = raw["token"]
    except KeyError as exc:
        raise EdgeConfigError(f"{path} is missing required key {exc.args[0]!r}") from exc

    allow_plaintext = bool(raw.get("allow_plaintext", False))
    try:
        validate_server_url(server_url, allow_plaintext)
    except ValueError as exc:
        raise EdgeConfigError(str(exc)) from exc

    return EdgeConfig(
        server_url=server_url,
        token=token,
        capture_device=raw.get("capture_device", DEFAULT_CAPTURE_DEVICE),
        playback_device=raw.get("playback_device", DEFAULT_PLAYBACK_DEVICE),
        vad_model_path=raw.get("vad_model_path", DEFAULT_VAD_MODEL_PATH),
        vad_threshold=float(raw.get("vad_threshold", DEFAULT_VAD_THRESHOLD)),
        vad_min_silence_ms=int(raw.get("vad_min_silence_ms", DEFAULT_VAD_MIN_SILENCE_MS)),
        allow_plaintext=allow_plaintext,
    )
