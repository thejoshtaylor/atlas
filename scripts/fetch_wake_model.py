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

The archive is also checked against a sha256 pinned in this file before
it is opened at all (WR-05, code review) -- its contents are handed to
native Kaldi code, and this script previously performed no verification
of any kind.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import httpx

from atlas.config import ConfigError, load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.example.yaml"

_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
_REQUEST_TIMEOUT_S = 300.0
_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB

# WR-05 (code review). This archive is extracted into the models volume
# and then handed to native Kaldi code (libvosk), so a compromised mirror
# or a hijacked upstream account was arbitrary native-code input with
# nothing in its way -- this script performed no verification at all
# before, not even a Content-Length check. It is a fixed, published
# release artifact, so its digest does not change and pinning it costs
# nothing operationally. Computed from the bytes alphacephei.com actually
# served, this session; re-derive with `shasum -a 256` on a download of
# your own rather than trusting this line.
_MODEL_SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"


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


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_vosk_model(
    model_dir: Path,
    *,
    url: str = _MODEL_URL,
    download: "Downloader" = default_download,
    expected_sha256: "str | None" = _MODEL_SHA256,
) -> str:
    """Provision `model_dir` -- the exact directory `VoskWakeDetector` opens
    (`os.path.isdir(config.model_path)`) -- from `url`. Returns
    "already-present" if `model_dir` already holds files, or "fetched"
    after a real extraction. Idempotent: a re-run after an interrupted run
    costs nothing, matching `fetch_models.py::fetch_one`'s own posture.

    `expected_sha256` defaults to the digest this repository pins for
    `_MODEL_URL` and is checked before the archive is opened. A test
    supplying its own fake archive passes `expected_sha256=None`; no
    real fetch ever does."""
    if model_dir.is_dir() and any(model_dir.iterdir()):
        return "already-present"
    if model_dir.exists() and not model_dir.is_dir():
        raise FetchError(f"{model_dir} exists and is not a directory")

    expected_top = model_dir.name
    with tempfile.TemporaryDirectory(prefix="atlas-vosk-fetch-") as tmp_str:
        tmp = Path(tmp_str)
        zip_path = tmp / "vosk-model.zip"
        download(url, zip_path)

        # WR-05 (code review): before anything is opened, let alone
        # extracted and handed to libvosk. `expected_sha256=None` is for
        # a test supplying its own fake archive, and is never what a real
        # fetch passes -- the default is the pinned digest.
        if expected_sha256 is not None:
            actual_sha256 = _sha256_of_file(zip_path)
            if actual_sha256 != expected_sha256:
                raise FetchError(
                    f"the downloaded archive's sha256 {actual_sha256} does not match the "
                    f"digest this repository pins for {url} ({expected_sha256}) -- "
                    "refusing to extract it"
                )

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
        # WR-04 (code review): `shutil.move` into an EXISTING directory
        # moves the source *inside* it, so an operator who had already
        # created the path `config.example.yaml` names -- a natural thing
        # to do, since the file names it -- got the model one level too
        # deep (`.../vosk-model-small-en-us-0.15/vosk-model-small-en-us-
        # 0.15/am/final.mdl`), and this function still returned "fetched".
        # `VoskWakeDetector` then failed at startup with an opaque Kaldi
        # error, and the documented remedy could not repair it: the
        # directory was now non-empty, so a re-run printed "already
        # present" and changed nothing.
        #
        # The guard at the top of this function means an existing
        # `model_dir` is empty by the time execution reaches here, so
        # `rmdir` succeeds. If that ever stops being true, `rmdir` raises
        # rather than quietly nesting -- which is the whole point.
        if model_dir.is_dir():
            model_dir.rmdir()
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


def main(
    argv: "list[str] | None" = None,
    *,
    download: "Downloader" = default_download,
    expected_sha256: "str | None" = _MODEL_SHA256,
) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    model_dir = Path(config.wake.resolve("camera").vosk.model_path)
    try:
        status = fetch_vosk_model(
            model_dir, download=download, expected_sha256=expected_sha256
        )
    # WR-06 (code review): the same widening `fetch_models.py::main`
    # carries, plus `zipfile.BadZipFile` -- a truncated or substituted
    # archive is this script's own version of the same failure, and it
    # raised straight through the old handler.
    except (FetchError, httpx.HTTPError, OSError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    verb = "already present" if status == "already-present" else "fetched"
    print(f"{verb}: {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
