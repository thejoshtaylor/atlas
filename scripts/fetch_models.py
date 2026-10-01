#!/usr/bin/env python3
"""Provisions every model file the local provider set (PROV-06, D-09) needs,
plus both Phase 11 speaker-embedding candidates (D-05), into the configured
`/models` root: the faster-whisper speech-to-text model directory, the
Piper voice plus its configuration file, and the CAM++ and TitaNet-small
speaker embedding models.

D-11 is this script's whole reason to exist as a separate, operator-run
step: nothing in the application ever downloads a model at boot. An
offline deployment must not need the internet at the least convenient
moment, so `lifespan`, no route, and no provider constructor calls
anything in this module -- confirmed by a grep gate over `src/atlas/`
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
nothing).

**What every downloaded file is checked against, exactly** (WR-05, code
review -- this paragraph previously described an integrity control the
code deliberately did not perform):

  * A sha256 digest pinned in this file, per file, in `_PINNED_SHA256`
    below. This is the real control: every destination here is handed to
    a native extension (ctranslate2, onnxruntime), so a compromised
    mirror or a hijacked upstream account would otherwise be arbitrary
    native-code input with nothing in its way. Each of these is a fixed,
    published release artifact whose digest does not change, so pinning
    costs nothing operationally. A file with no pinned digest is refused
    before it is downloaded, not fetched unverified.
  * The size the source declared in its own `Content-Length`, when it
    sends one -- kept as the cheap first gate, since it catches a
    truncated transfer without hashing hundreds of megabytes.
  * A digest the `Downloader` itself reports, if it reports one.
    `default_download` never does: see its own docstring for why the
    Hub's `ETag` is not a whole-file content digest and treating it as
    one would delete correct downloads.

A mismatch on any of these deletes the partial file rather than leaving a
truncated or substituted model that would fail much later, much less
clearly, deep inside `faster_whisper` or `piper`.

**Parakeet** (`--only parakeet`, quick task 261001-mp8) is opt-in and is not
part of `all`. It downloads the pinned sherpa-onnx archive (about 480 MB)
and copies only four named files out of it into `stt.parakeet_model_dir`.
No archive member path is ever used as a destination. Then it downloads the
pinned NeMo file (about 2.4 GB), reads the sentencepiece tokenizer from
inside it, and writes `bpe.vocab` (`piece<TAB>score` for every id) beside
the model. Both downloads are deleted afterwards. `sentencepiece` is needed
only for that last step and is imported inside the function that uses it.
Without `bpe.vocab` the provider still runs, with greedy decoding and no
keyterm biasing.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import httpx

from atlas.config import (
    PARAKEET_BPE_VOCAB,
    PARAKEET_MODEL_FILES,
    SPEAKER_MODEL_FILES,
    ConfigError,
    load_config,
)

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

# Both Phase 11 D-05 spike candidates, from the same GitHub Releases tag
# (`speaker-recongition-models` -- that exact spelling, confirmed against
# the live release this session, 11-01-PLAN.md's `<interfaces>`). Fetched
# unconditionally by filename from `atlas.config.SPEAKER_MODEL_FILES`, never
# gated on which one `speaker_id.model` currently selects -- the spike
# needs both on disk to compare them.
_SPEAKER_MODEL_URL_TEMPLATE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{filename}"
)

# Parakeet TDT 0.6b v2, int8 (261001-mp8, D-10). The archive is the
# sherpa-onnx export; the NeMo file is read once for the sentencepiece
# tokenizer inside it. Model licence: CC-BY-4.0 (NVIDIA), attribution required.
_PARAKEET_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
)
_PARAKEET_ARCHIVE_TOP_DIR = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
_PARAKEET_NEMO_URL = (
    "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/parakeet-tdt-0.6b-v2.nemo"
)
_PARAKEET_NEMO_FILENAME = "parakeet-tdt-0.6b-v2.nemo"
_PARAKEET_TOKENIZER_SUFFIX = "_tokenizer.model"

_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB
_REQUEST_TIMEOUT_S = 120.0

# IN-03 (code review): the independent reference `resolve_under_root`
# checks against. This is the mount point the Helm chart, the Compose
# file and config.example.yaml all already name -- stated once, here,
# rather than derived from the destinations being checked.
_DEFAULT_MODEL_ROOT = Path("/models")

# WR-05 (code review). The sha256 of every file this script is allowed to
# write, keyed by the URL it comes from -- keyed by URL rather than by
# destination, because the URL is what an attacker controls and the
# destination is what this repository controls.
#
# Provenance, so a later reader can re-derive rather than trust: the
# large files are Hugging Face LFS objects, and the Hub's own API
# publishes their sha256 as `lfs.oid`
# (`https://huggingface.co/api/models/{repo}/tree/main`) -- that is the
# digest, not the `ETag`/`xetHash` header, which is a chunk-store hash
# (see `default_download` below). The small non-LFS files carry only a
# git blob sha1 in that listing, so their digests here were computed
# from the bytes the Hub actually served, this session.
#
# faster-whisper's file set is per model size, so a size with no pinned
# digests is refused by name rather than fetched unverified. `small` and
# `base` are the two sizes 07-CONTEXT.md's D-09 names.
_PINNED_SHA256: "dict[str, str]" = {
    # Systran/faster-whisper-small
    "https://huggingface.co/Systran/faster-whisper-small/resolve/main/config.json":
        "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828",
    "https://huggingface.co/Systran/faster-whisper-small/resolve/main/model.bin":
        "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671",
    "https://huggingface.co/Systran/faster-whisper-small/resolve/main/tokenizer.json":
        "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
    "https://huggingface.co/Systran/faster-whisper-small/resolve/main/vocabulary.txt":
        "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
    # Systran/faster-whisper-base
    "https://huggingface.co/Systran/faster-whisper-base/resolve/main/config.json":
        "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a",
    "https://huggingface.co/Systran/faster-whisper-base/resolve/main/model.bin":
        "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
    "https://huggingface.co/Systran/faster-whisper-base/resolve/main/tokenizer.json":
        "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
    "https://huggingface.co/Systran/faster-whisper-base/resolve/main/vocabulary.txt":
        "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
    # rhasspy/piper-voices, the shipped default voice
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx":
        "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json":
        "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
    # k2-fsa/sherpa-onnx, release tag speaker-recongition-models (Phase 11,
    # D-05). GitHub's Releases API publishes no `digest` field for these
    # assets, so both digests here were computed from the bytes the release
    # actually served, 2026-09-28 (11-01-PLAN.md Task 2) -- the same
    # provenance this file's own header already states for its non-LFS
    # entries above.
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx":
        "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx":
        "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
    # Parakeet TDT 0.6b v2 (261001-mp8, D-10), looked up 2026-10-01. The
    # archive digest is the GitHub Releases API `digest` field (482468385
    # bytes). The NeMo digest is the Hugging Face tree API `lfs.oid`
    # (2472222720 bytes).
    _PARAKEET_ARCHIVE_URL:
        "157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad",
    _PARAKEET_NEMO_URL:
        "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c",
}


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
    to an operator-controlled mount actually has.

    IN-03 (code review): this is only as strong as where `model_root`
    comes from. Derived with `os.path.commonpath` over the very
    destinations it is then used to check, every planned file is under it
    by construction -- so the guard could fire for a symlink inside the
    mount but never for a traversal in the plan, much weaker than the
    module docstring advertised. `plan_fetches` takes the root as a
    parameter now and `main` passes `--model-root` (default `/models`,
    the mount point both deployment targets use and the shipped
    configuration names), so the check has a reference that does not
    depend on what it is checking."""
    root = model_root.resolve()
    resolved = dest.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise FetchError(
            f"refusing to write outside the model root {root}: {dest} resolves to {resolved}"
        ) from None
    return resolved


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
    reading the declared size off the response itself -- never hardcoded,
    so an upstream file-layout change cannot silently pass a stale check.

    Deliberately never derives a digest from `ETag`/`X-Linked-ETag`.
    Confirmed live against Hugging Face Hub's real `Systran/faster-whisper-small` model
    file this session: its `ETag` response header (`429ffc84...`) does not
    equal the sha256 of the bytes actually served (`3e305921...`) -- the
    Hub's Xet-backed CDN publishes a chunk-store hash under that header
    name for large files, not a whole-file content digest. Treating it as
    one would delete a correct download as if it were corrupt, which is
    worse than not checking a digest at all. The size check below is the
    verification this script actually performs for a real fetch;
    `DownloadReceipt.declared_sha256` stays a real, exercised code path
    for a `Downloader` that gets a digest from a source that publishes one
    honestly (or a test double)."""
    with httpx.Client(follow_redirects=True, timeout=_REQUEST_TIMEOUT_S) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            declared_size_header = response.headers.get("content-length")
            declared_size = int(declared_size_header) if declared_size_header else None
            with part_path.open("wb") as fh:
                for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    fh.write(chunk)
    return DownloadReceipt(declared_size=declared_size, declared_sha256=None)


def pinned_sha256(url: str, pinned: "Mapping[str, str] | None" = None) -> str:
    """The sha256 this repository pins for `url`, or a refusal naming the
    URL (WR-05).

    Refusing an unpinned URL before any byte is written is the whole
    point: "download it anyway and skip the check" is how a documented
    integrity control becomes a comment. The one realistic way to reach
    this is an `stt.local_model_size` naming a size this repository has
    not pinned, so the message says exactly that.

    `pinned` is the injection seam a test uses to pin its own fake
    source, the same shape as the `Downloader` seam beside it -- real
    calls pass nothing and get `_PINNED_SHA256`."""
    pinned = _PINNED_SHA256 if pinned is None else pinned
    digest = pinned.get(url)
    if digest is None:
        raise FetchError(
            f"no pinned sha256 for {url} -- this script only fetches files whose "
            "digest is pinned in scripts/fetch_models.py. If you changed "
            "stt.local_model_size or tts.piper_voice_path, either set it back to a "
            f"pinned value ({', '.join(sorted(pinned_faster_whisper_sizes(pinned)))} for "
            "the model size) or add the new file's digest to _PINNED_SHA256 there, "
            "verified against the source yourself"
        )
    return digest


def pinned_faster_whisper_sizes(pinned: "Mapping[str, str] | None" = None) -> "set[str]":
    """The faster-whisper model sizes `_PINNED_SHA256` covers -- derived
    from the table rather than restated beside it, so adding a size to
    the table is the only edit needed."""
    pinned = _PINNED_SHA256 if pinned is None else pinned
    prefix = "https://huggingface.co/Systran/faster-whisper-"
    return {
        url[len(prefix) :].split("/", 1)[0]
        for url in pinned
        if url.startswith(prefix)
    }


def fetch_one(
    model_file: ModelFile,
    model_root: Path,
    download: Downloader,
    pinned: "Mapping[str, str] | None" = None,
) -> FetchResult:
    """Fetch one file: refuse an out-of-root destination or an unpinned
    URL, skip a file already present, otherwise download to a `.part`
    sibling, verify it against the pinned digest (and against what the
    source declared), and rename it into place -- or delete it and raise,
    never leaving a truncated or substituted file behind."""
    dest = resolve_under_root(model_root, model_file.dest)
    if dest.exists():
        return FetchResult(label=model_file.label, dest=dest, status="already-present")

    # Before the download, not after: an unpinned URL is not fetched at
    # all, so there is never a moment where unverifiable bytes exist on
    # this filesystem.
    expected_sha256 = pinned_sha256(model_file.url, pinned)

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
        actual_sha256 = _sha256_of_file(part_path)
        if actual_sha256 != expected_sha256:
            raise FetchError(
                f"{model_file.label}: sha256 {actual_sha256} does not match the digest "
                f"this repository pins for it ({expected_sha256}) -- deleting the "
                "downloaded file rather than handing unverified bytes to a native "
                "extension"
            )
        if receipt.declared_sha256 is not None and actual_sha256 != receipt.declared_sha256:
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
    model_files: Sequence[ModelFile],
    model_root: Path,
    download: Downloader = default_download,
    pinned: "Mapping[str, str] | None" = None,
) -> "list[FetchResult]":
    return [fetch_one(model_file, model_root, download, pinned) for model_file in model_files]


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


def plan_fetches(
    stt_config: "Any",
    tts_config: "Any",
    model_root: "Path | None" = None,
    *,
    speaker_id_config: "Any | None" = None,
) -> "tuple[Path, list[ModelFile]]":
    """Every model file the local provider set needs, and the root every
    destination must resolve under. The destinations come from
    configuration, never a second, restated copy of a path `config.py`
    already owns.

    Takes `SttConfig`/`TtsConfig` directly (not the whole app `Config`) --
    this function only ever reads those two sections, and a narrower
    signature is what lets Task 2's own tests build just the two small
    dataclasses a fake source needs, with no dependency on every other
    config section's own required fields.

    `speaker_id_config` (Phase 11, D-05) is optional and keyword-only,
    added alongside the original two rather than folded into a growing
    positional list: omitted (the default), this function's behavior is
    byte-for-byte what it was before Phase 11 -- no speaker files planned,
    no speaker directory in the derived-root fallback below. Given, one
    `ModelFile` is planned per `atlas.config.SPEAKER_MODEL_FILES` entry,
    at `speaker_id_config.model_dir`, regardless of which model
    `speaker_id.model` currently selects -- the D-05 spike needs both on
    disk to compare them.

    IN-03 (code review): `model_root` is a parameter. Given, it is an
    independent reference, and a configured destination that escapes it
    is refused -- which is what `resolve_under_root`'s own docstring
    always claimed. Omitted, it falls back to the old `commonpath`
    derivation over the destinations themselves, which by construction
    cannot refuse any of them. That fallback is for a caller with no
    `/models` mount to check against (every test in
    `tests/test_fetch_models.py`), and it is honestly the weaker of the
    two.
    """
    stt_dir = Path(stt_config.local_model_dir)
    piper_voice_path = Path(tts_config.piper_voice_path)
    piper_config_path = Path(tts_config.piper_config_path)
    speaker_id_dir = Path(speaker_id_config.model_dir) if speaker_id_config is not None else None

    if model_root is None:
        commonpath_inputs = [str(stt_dir), str(piper_voice_path.parent), str(piper_config_path.parent)]
        if speaker_id_dir is not None:
            commonpath_inputs.append(str(speaker_id_dir))
        model_root = Path(os.path.commonpath(commonpath_inputs))

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
    speaker_id_files = (
        [
            ModelFile(
                label=f"speaker-id/{filename}",
                url=_SPEAKER_MODEL_URL_TEMPLATE.format(filename=filename),
                dest=speaker_id_dir / filename,
            )
            for filename in SPEAKER_MODEL_FILES.values()
        ]
        if speaker_id_dir is not None
        else []
    )
    return model_root, [*faster_whisper_files, *piper_files, *speaker_id_files]


def _load_sentencepiece(model_bytes: bytes) -> Any:
    """Load a sentencepiece processor from model bytes. Imported here, not
    at module scope: this is a fetch-time dependency only, and nothing under
    `src/atlas` imports it."""
    try:
        import sentencepiece
    except ImportError as exc:
        raise FetchError(
            "sentencepiece is needed only to build bpe.vocab from the Parakeet tokenizer. "
            "Run `pip install sentencepiece` (or install the project's `parakeet-fetch` "
            "extra) in the environment that runs this script, then re-run "
            "`--only parakeet`. The model files are already in place, and Parakeet runs "
            "with greedy decoding and no keyterm biasing until bpe.vocab exists."
        ) from exc
    return sentencepiece.SentencePieceProcessor(model_proto=model_bytes)


def _extract_parakeet_files(archive_path: Path, model_dir: Path) -> "list[Path]":
    """Copy the four allowlisted files out of the archive into `model_dir`.

    A member is accepted only when it is a regular file named exactly
    `<top dir>/<allowlisted basename>`. `extract` and `extractall` are never
    called, so no member path is ever used as a destination. Files are
    written to `.part` siblings and renamed only when all four are present,
    so a bad archive leaves no partial model behind."""
    wanted = set(PARAKEET_MODEL_FILES)
    parts: "dict[str, Path]" = {}
    try:
        with tarfile.open(archive_path, "r:bz2") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                pure = PurePosixPath(member.name)
                if len(pure.parts) != 2 or pure.parts[0] != _PARAKEET_ARCHIVE_TOP_DIR:
                    continue
                basename = pure.parts[1]
                if basename not in wanted:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                part_path = model_dir / (basename + ".part")
                with source, part_path.open("wb") as out:
                    shutil.copyfileobj(source, out, _DOWNLOAD_CHUNK_BYTES)
                parts[basename] = part_path
        missing = [name for name in PARAKEET_MODEL_FILES if name not in parts]
        if missing:
            raise FetchError(
                f"the Parakeet archive does not hold {', '.join(missing)} under "
                f"{_PARAKEET_ARCHIVE_TOP_DIR}/ -- nothing was installed"
            )
    except BaseException:
        for part_path in parts.values():
            part_path.unlink(missing_ok=True)
        raise
    placed: "list[Path]" = []
    for name in PARAKEET_MODEL_FILES:
        dest = model_dir / name
        parts[name].replace(dest)
        placed.append(dest)
    return placed


def _build_bpe_vocab(nemo_path: Path, vocab_path: Path, load_sentencepiece: Callable[[bytes], Any]) -> None:
    """Write `piece<TAB>score` for every id of the tokenizer inside the
    NeMo file (a tar) to `vocab_path`."""
    with tarfile.open(nemo_path, "r:*") as nemo:
        candidates = [
            member
            for member in nemo
            if member.isfile() and member.name.endswith(_PARAKEET_TOKENIZER_SUFFIX)
        ]
        if len(candidates) != 1:
            raise FetchError(
                f"expected exactly one *{_PARAKEET_TOKENIZER_SUFFIX} in {nemo_path.name}, "
                f"found {len(candidates)}"
            )
        source = nemo.extractfile(candidates[0])
        if source is None:
            raise FetchError(f"cannot read the tokenizer inside {nemo_path.name}")
        with source:
            model_bytes = source.read()
    processor = load_sentencepiece(model_bytes)
    part_path = vocab_path.with_name(vocab_path.name + ".part")
    try:
        with part_path.open("w", encoding="utf-8") as out:
            for index in range(processor.get_piece_size()):
                out.write(f"{processor.id_to_piece(index)}\t{processor.get_score(index)}\n")
        part_path.replace(vocab_path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise


def fetch_parakeet(
    stt_config: "Any",
    model_root: Path,
    download: Downloader = default_download,
    pinned: "Mapping[str, str] | None" = None,
    *,
    load_sentencepiece: Callable[[bytes], Any] = _load_sentencepiece,
) -> "list[FetchResult]":
    """Provision the Parakeet model directory: the four model files from the
    pinned sherpa-onnx archive, then `bpe.vocab` from the tokenizer inside
    the pinned NeMo file. A step whose output is already present is skipped,
    so a second run downloads nothing."""
    model_dir = resolve_under_root(model_root, Path(stt_config.parakeet_model_dir))
    results: "list[FetchResult]" = []

    model_paths = [model_dir / name for name in PARAKEET_MODEL_FILES]
    if all(path.exists() for path in model_paths):
        results.extend(
            FetchResult(label=f"parakeet/{path.name}", dest=path, status="already-present")
            for path in model_paths
        )
    else:
        archive_path = model_dir / _PARAKEET_ARCHIVE_URL.rsplit("/", 1)[1]
        fetched = fetch_one(
            ModelFile(label="parakeet/archive", url=_PARAKEET_ARCHIVE_URL, dest=archive_path),
            model_root,
            download,
            pinned,
        )
        try:
            placed = _extract_parakeet_files(fetched.dest, model_dir)
        finally:
            archive_path.unlink(missing_ok=True)
        results.extend(
            FetchResult(label=f"parakeet/{path.name}", dest=path, status="fetched") for path in placed
        )

    vocab_path = model_dir / PARAKEET_BPE_VOCAB
    if vocab_path.exists():
        results.append(FetchResult(label=f"parakeet/{PARAKEET_BPE_VOCAB}", dest=vocab_path, status="already-present"))
    else:
        nemo_path = model_dir / _PARAKEET_NEMO_FILENAME
        try:
            fetch_one(
                ModelFile(label="parakeet/nemo", url=_PARAKEET_NEMO_URL, dest=nemo_path),
                model_root,
                download,
                pinned,
            )
            _build_bpe_vocab(nemo_path, vocab_path, load_sentencepiece)
        finally:
            nemo_path.unlink(missing_ok=True)
        results.append(FetchResult(label=f"parakeet/{PARAKEET_BPE_VOCAB}", dest=vocab_path, status="fetched"))
    return results


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
    parser.add_argument(
        "--model-root",
        default=str(_DEFAULT_MODEL_ROOT),
        help=(
            "the directory every destination must resolve under (default: "
            f"{_DEFAULT_MODEL_ROOT}, the mount point both deployment targets use). "
            "Pass the directory your own configuration writes into if it is "
            "elsewhere; pass an empty string to derive it from the destinations, "
            "which cannot refuse any of them"
        ),
    )
    parser.add_argument(
        "--only",
        choices=("all", "local-providers", "speaker-id", "parakeet"),
        default="all",
        help=(
            "which file set to plan (default: all). 'local-providers' plans only the "
            "faster-whisper and Piper files the local STT/TTS provider set needs; "
            "'speaker-id' plans only the two Phase 11 speaker-embedding models (D-05); "
            "'parakeet' is opt-in and not part of 'all': it downloads about 480 MB plus a "
            "2.4 GB NeMo file that is read once for its tokenizer and then deleted"
        ),
    )
    return parser


def main(
    argv: "list[str] | None" = None,
    *,
    download: Downloader = default_download,
    pinned: "Mapping[str, str] | None" = None,
) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        requested_root = Path(args.model_root) if args.model_root else None
        if args.only == "parakeet":
            model_root = requested_root or Path(config.stt.parakeet_model_dir).parent
            results = fetch_parakeet(config.stt, model_root, download, pinned)
        else:
            model_root, model_files = plan_fetches(
                config.stt, config.tts, requested_root, speaker_id_config=config.speaker_id
            )
            if args.only == "local-providers":
                model_files = [f for f in model_files if not f.label.startswith("speaker-id/")]
            elif args.only == "speaker-id":
                model_files = [f for f in model_files if f.label.startswith("speaker-id/")]
            results = fetch_all(model_files, model_root, download, pinned)
    # WR-06 (code review): `FetchError` alone left every failure the real
    # download path can actually produce as a raw traceback --
    # `httpx.HTTPStatusError` from `raise_for_status()` (a 404 on a
    # model size the Hub does not publish), `httpx.ConnectError` (no
    # network, which is the normal state of the offline deployment D-11
    # is written for), and `OSError` from a full disk mid-write. Every
    # other failure in this script reports by name; these did not.
    #
    # `plan_fetches` is inside the try as well: it raises `FetchError`
    # for a voice filename that does not follow Piper's convention, and
    # that was the one refusal the old shape could not report either.
    except (FetchError, httpx.HTTPError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for result in results:
        verb = "already present" if result.status == "already-present" else "fetched"
        print(f"{verb}: {result.label} -> {result.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
