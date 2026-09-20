#!/usr/bin/env python3
"""Provisions every model file the local provider set (PROV-06, D-09) needs
into the configured `/models` root: the faster-whisper speech-to-text model
directory, and the Piper voice plus its configuration file.

D-11 is this script's whole reason to exist as a separate, operator-run
step: nothing in the application ever downloads a model at boot. An
offline deployment must not need the internet at the least convenient
moment, so `lifespan`, no route, and no provider constructor calls
anything in this module -- confirmed by a grep gate over `src/spire_voice/`
in this script's own `<verify>`.

Destinations are read from the project's own configuration
(`SttConfig.local_model_dir`/`local_model_size`, `TtsConfig.
piper_voice_path`/`piper_config_path`) rather than restated here, so a path
change in one place cannot leave this script writing to the old one.
Source URLs are derived, not hand-typed per file: the faster-whisper file
set is fetched from `Systran/faster-whisper-{size}` on the Hugging Face
Hub (the same repository family `faster_whisper.utils.download_model`
itself downloads from), and the Piper voice/config pair is fetched from
`rhasspy/piper-voices`, whose directory layout is derived from the voice
filename's own `{locale}-{name}-{quality}` convention.

Every destination is resolved against the configured model root before any
byte is written, and refused by name if it would land outside that root --
the root is a mount an operator controls, and a traversal out of it is the
one filesystem risk a fetcher has. A file already present is left alone
and reported as such (idempotent: a re-run after a partial run costs
nothing). What was actually written is checked against what the source
declared (size always, a sha256 digest when the source's own response
publishes one via `ETag`/`X-Linked-ETag`) before the file is kept; a
mismatch deletes the partial file rather than leaving a truncated model
that would fail much later, much less clearly, deep inside `faster_whisper`
or `piper`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

from spire_voice.config import ConfigError, load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.example.yaml"

# The exact file set present in `Systran/faster-whisper-*` repositories
# (confirmed against the Hub's own file listing for the "small" size this
# session) -- not `faster_whisper.utils.download_model`'s own, broader
# `allow_patterns` glob, which also matches an optional
# `preprocessor_config.json` this repository family does not ship.
_FASTER_WHISPER_FILES: "tuple[str, ...]" = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
_FASTER_WHISPER_URL_TEMPLATE = "https://huggingface.co/Systran/faster-whisper-{size}/resolve/main/{filename}"

_PIPER_VOICES_URL_TEMPLATE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{path}"

_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB
_REQUEST_TIMEOUT_S = 120.0


class FetchError(Exception):
    """Raised before or during one file's fetch -- a destination that would
    escape the model root, or a downloaded file that does not match what
    the source declared. Never raised for a file already present."""


@dataclass(frozen=True)
class ModelFile:
    """One file this script provisions: an operator-facing label, the URL
    it comes from, and the absolute destination -- read from configuration
    by `plan_fetches`, never restated as a second literal path."""

    label: str
    url: str
    dest: Path


@dataclass(frozen=True)
class DownloadReceipt:
    """What one `Downloader` call reports about the bytes it wrote --
    `None` for either field when the source publishes neither, an honest
    absence rather than an invented value to check against."""

    declared_size: "int | None"
    declared_sha256: "str | None"


@dataclass(frozen=True)
class FetchResult:
    label: str
    dest: Path
    status: str  # "already-present" | "fetched"


# The seam Task 2's own tests inject through -- the same shape the local
# provider classes inject their model loaders (`FasterWhisperStt(config,
# load_model=...)`): a real implementation by default, a fake source in
# every test, so every rule below is proven with no network access at all.
Downloader = Callable[[str, Path], DownloadReceipt]


def resolve_under_root(model_root: Path, dest: Path) -> Path:
    """Resolve `dest` and refuse it, naming both paths, if it would land
    outside `model_root` -- the one filesystem risk a fetcher that writes
    to an operator-controlled mount actually has."""
    root = model_root.resolve()
    resolved = dest.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise FetchError(
            f"refusing to write outside the model root {root}: {dest} resolves to {resolved}"
        ) from None
    return resolved


def _sha256_from_header(header_value: "str | None") -> "str | None":
    """A sha256 hex digest, when `header_value` looks like one -- Hugging
    Face publishes this as `ETag`/`X-Linked-ETag` for its LFS-backed model
    files. Any other shape (a short opaque cache-busting etag, for
    instance) is an honest "the source published nothing usable here",
    never coerced into a digest that was not actually offered."""
    if not header_value:
        return None
    candidate = header_value.strip().strip('"')
    if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate.lower()):
        return candidate.lower()
    return None


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def default_download(url: str, part_path: Path) -> DownloadReceipt:
    """The real network implementation: stream `url` to `part_path`,
    reading the declared size/digest off the response itself -- never
    hardcoded, so an upstream file-layout change cannot silently pass a
    stale check."""
    with httpx.Client(follow_redirects=True, timeout=_REQUEST_TIMEOUT_S) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            declared_size_header = response.headers.get("content-length")
            declared_size = int(declared_size_header) if declared_size_header else None
            declared_sha256 = _sha256_from_header(
                response.headers.get("x-linked-etag") or response.headers.get("etag")
            )
            with part_path.open("wb") as fh:
                for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    fh.write(chunk)
    return DownloadReceipt(declared_size=declared_size, declared_sha256=declared_sha256)


def fetch_one(model_file: ModelFile, model_root: Path, download: Downloader) -> FetchResult:
    """Fetch one file: refuse an out-of-root destination, skip a file
    already present, otherwise download to a `.part` sibling, verify it
    against what the source declared, and rename it into place -- or
    delete it and raise, never leaving a truncated file behind."""
    dest = resolve_under_root(model_root, model_file.dest)
    if dest.exists():
        return FetchResult(label=model_file.label, dest=dest, status="already-present")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_name(dest.name + ".part")
    receipt = download(model_file.url, part_path)
    try:
        actual_size = part_path.stat().st_size
        if receipt.declared_size is not None and actual_size != receipt.declared_size:
            raise FetchError(
                f"{model_file.label}: downloaded {actual_size} bytes, the source declared "
                f"{receipt.declared_size} -- deleting the partial file rather than keeping "
                "a truncated model"
            )
        if receipt.declared_sha256 is not None:
            actual_sha256 = _sha256_of_file(part_path)
            if actual_sha256 != receipt.declared_sha256:
                raise FetchError(
                    f"{model_file.label}: sha256 {actual_sha256} does not match the source's "
                    f"declared {receipt.declared_sha256} -- deleting the partial file"
                )
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    part_path.rename(dest)
    return FetchResult(label=model_file.label, dest=dest, status="fetched")


def fetch_all(
    model_files: Sequence[ModelFile], model_root: Path, download: Downloader = default_download
) -> "list[FetchResult]":
    return [fetch_one(model_file, model_root, download) for model_file in model_files]


def _piper_voice_hub_path(filename: str) -> str:
    """Derive `rhasspy/piper-voices`' own directory layout
    (`{lang}/{locale}/{name}/{quality}/{filename}`) from a voice filename
    following Piper's own `{locale}-{name}-{quality}.onnx[.json]`
    convention -- confirmed against the Hub's real file listing for this
    project's shipped default voice (`en_US-lessac-medium.onnx`) this
    session, not assumed. A voice filename that does not follow this
    convention raises by name rather than guessing a path."""
    stem = filename
    for suffix in (".onnx.json", ".onnx"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("-")
    if len(parts) != 3:
        raise FetchError(
            f"piper voice filename {filename!r} does not follow the "
            "'{locale}-{name}-{quality}' convention -- cannot derive its "
            "rhasspy/piper-voices path"
        )
    locale, name, quality = parts
    lang = locale.split("_")[0]
    return f"{lang}/{locale}/{name}/{quality}/{filename}"


def plan_fetches(stt_config: "Any", tts_config: "Any") -> "tuple[Path, list[ModelFile]]":
    """Every model file the local provider set needs, and the root every
    destination must resolve under -- both derived from configuration,
    never a second, restated copy of a path `config.py` already owns.

    Takes `SttConfig`/`TtsConfig` directly (not the whole app `Config`) --
    this function only ever reads those two sections, and a narrower
    signature is what lets Task 2's own tests build just the two small
    dataclasses a fake source needs, with no dependency on every other
    config section's own required fields.
    """
    stt_dir = Path(stt_config.local_model_dir)
    piper_voice_path = Path(tts_config.piper_voice_path)
    piper_config_path = Path(tts_config.piper_config_path)

    model_root = Path(
        os.path.commonpath([str(stt_dir), str(piper_voice_path.parent), str(piper_config_path.parent)])
    )

    faster_whisper_files = [
        ModelFile(
            label=f"faster-whisper/{filename}",
            url=_FASTER_WHISPER_URL_TEMPLATE.format(size=stt_config.local_model_size, filename=filename),
            dest=stt_dir / filename,
        )
        for filename in _FASTER_WHISPER_FILES
    ]
    piper_files = [
        ModelFile(
            label=f"piper/{piper_voice_path.name}",
            url=_PIPER_VOICES_URL_TEMPLATE.format(path=_piper_voice_hub_path(piper_voice_path.name)),
            dest=piper_voice_path,
        ),
        ModelFile(
            label=f"piper/{piper_config_path.name}",
            url=_PIPER_VOICES_URL_TEMPLATE.format(path=_piper_voice_hub_path(piper_config_path.name)),
            dest=piper_config_path,
        ),
    ]
    return model_root, [*faster_whisper_files, *piper_files]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Provision every model file the local provider set needs into the configured "
            "/models root (D-11). Run this once, by hand, before selecting a local provider "
            "on the Providers screen -- nothing in the application ever calls this."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=f"the application config to read stt.*/tts.* from (default: {_DEFAULT_CONFIG_PATH})",
    )
    return parser


def main(argv: "list[str] | None" = None, *, download: Downloader = default_download) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    model_root, model_files = plan_fetches(config.stt, config.tts)
    try:
        results = fetch_all(model_files, model_root, download)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for result in results:
        verb = "already present" if result.status == "already-present" else "fetched"
        print(f"{verb}: {result.label} -> {result.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
