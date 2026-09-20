#!/usr/bin/env python3
"""Provisions the Vosk wake-word model into the directory `wake.vosk.model_path`
names (default `/models/vosk-model-small-en-us-0.15`), so the shipped default
`wake.engine: vosk` can start without hitting `VoskWakeDetector`'s own
`WakeError` for a missing model directory -- confirmed real, this session
(07-05-SUMMARY.md's Known Gap): a bare `docker compose up`, with no model
provisioned, exits the app container with exactly that error, and DEP-03
says a clean clone must reach a running assistant.

This is a separate, operator-run step from `scripts/fetch_models.py`: that
script provisions the PROV-06 *local provider set* (speech-to-text,
text-to-speech, D-09); the wake-word model is a Phase 2 concept the wake
engine needs regardless of which providers are selected. D-11's "nothing
downloads at boot" posture applies here identically -- this script exists
as the same kind of documented, operator-run step, not a second copy of
one, and the application never imports or calls anything in this module.

The upstream distribution (alphacephei.com, the project Vosk itself
publishes its models from) is a ZIP archive whose own top-level entry is
already a directory named `vosk-model-small-en-us-0.15` -- extracted with
the standard library's `zipfile`, so no system `unzip` binary is required
inside the application image. Every archive member is checked against a
"must resolve under the expected top-level directory" rule before
anything is extracted -- a `..`-shaped or absolute entry in a downloaded
archive is refused by name rather than silently written outside the
models volume, the same posture `scripts/fetch_models.py::
resolve_under_root` already holds for its own downloads.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import httpx

from spire_voice.config import ConfigError, load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.example.yaml"

_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
_REQUEST_TIMEOUT_S = 300.0
_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB


class FetchError(Exception):
    """Raised before or during the fetch -- a destination occupied by
    something other than a directory, an archive whose member names would
    escape the expected top-level directory, or an archive with no
    top-level entry at all. Never raised when the model is already
    present."""


Downloader = Callable[[str, Path], None]


def default_download(url: str, dest_zip: Path) -> None:
    """The real network implementation -- a fake stands in for every test,
    the same seam `fetch_models.py`'s own `Downloader` uses."""
    with httpx.Client(follow_redirects=True, timeout=_REQUEST_TIMEOUT_S) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with dest_zip.open("wb") as fh:
                for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    fh.write(chunk)


def _safe_member_names(archive: "zipfile.ZipFile", expected_top: str) -> "list[str]":
    """Every member name in `archive`, refused by name if any would escape
    a plain `{expected_top}/...` shape -- an absolute path, a `..`
    segment, or a second top-level entry the model directory does not
    expect."""
    names = archive.namelist()
    if not names:
        raise FetchError("the downloaded archive is empty")
    for name in names:
        if name.startswith("/") or ".." in Path(name).parts:
            raise FetchError(f"refusing to extract unsafe archive member: {name!r}")
        top = name.split("/", 1)[0]
        if top != expected_top:
            raise FetchError(
                f"archive member {name!r} is not under the expected top-level "
                f"directory {expected_top!r}"
            )
    return names


def fetch_vosk_model(
    model_dir: Path, *, url: str = _MODEL_URL, download: "Downloader" = default_download
) -> str:
    """Provision `model_dir` -- the exact directory `VoskWakeDetector` opens
    (`os.path.isdir(config.model_path)`) -- from `url`. Returns
    "already-present" if `model_dir` already holds files, or "fetched"
    after a real extraction. Idempotent: a re-run after an interrupted run
    costs nothing, matching `fetch_models.py::fetch_one`'s own posture."""
    if model_dir.is_dir() and any(model_dir.iterdir()):
        return "already-present"
    if model_dir.exists() and not model_dir.is_dir():
        raise FetchError(f"{model_dir} exists and is not a directory")

    expected_top = model_dir.name
    with tempfile.TemporaryDirectory(prefix="spire-vosk-fetch-") as tmp_str:
        tmp = Path(tmp_str)
        zip_path = tmp / "vosk-model.zip"
        download(url, zip_path)

        with zipfile.ZipFile(zip_path) as archive:
            _safe_member_names(archive, expected_top)
            archive.extractall(tmp)

        extracted_dir = tmp / expected_top
        if not extracted_dir.is_dir():
            raise FetchError(
                f"archive extracted no directory named {expected_top!r} -- model_path "
                "must name a directory whose name matches the archive's own top level"
            )
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_dir), str(model_dir))

    return "fetched"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Provision the Vosk wake-word model into the directory wake.vosk.model_path "
            "names (D-11). Run this once, by hand, before starting the application with "
            "the shipped default wake.engine: vosk -- nothing in the application ever "
            "calls this."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=f"the application config to read wake.vosk.model_path from (default: {_DEFAULT_CONFIG_PATH})",
    )
    return parser


def main(argv: "list[str] | None" = None, *, download: "Downloader" = default_download) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    model_dir = Path(config.wake.resolve("camera").vosk.model_path)
    try:
        status = fetch_vosk_model(model_dir, download=download)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    verb = "already present" if status == "already-present" else "fetched"
    print(f"{verb}: {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
