"""`scripts/fetch_models.py`, against a fake source in a temporary
directory -- no network access (PROV-06, D-11).

Loaded by file path, the same way `test_score_wake_engines.py` loads
`score_wake_engines.py`: `scripts/` is not on `pythonpath` (only `src`/
`mcp` are, per `pyproject.toml`).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fetch_models.py"
_spec = importlib.util.spec_from_file_location("fetch_models", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
fetch_models = importlib.util.module_from_spec(_spec)
sys.modules["fetch_models"] = fetch_models
_spec.loader.exec_module(fetch_models)

from spire_voice.config import SttConfig, TtsConfig


def _model_file(dest: Path, url: str = "https://example.invalid/f") -> "fetch_models.ModelFile":
    return fetch_models.ModelFile(label="a-file", url=url, dest=dest)


def _pinned(*contents: bytes, url: str = "https://example.invalid/f") -> "dict[str, str]":
    """WR-05 (code review): `fetch_one` refuses a URL with no pinned
    sha256, so a test with a fake source has to pin its own fake bytes.
    This is the same injection shape as the `Downloader` seam beside it
    -- a real call passes nothing and gets the module's own table.

    Several contents may be pinned under one URL only where a test is
    proving the MISMATCH path; the last one wins, so the expected digest
    is the first argument and the served bytes are whatever the fake
    downloader actually writes."""
    return {url: hashlib.sha256(contents[0]).hexdigest()}


# --- resolve_under_root ------------------------------------------------


def test_resolve_under_root_accepts_a_destination_under_the_root(tmp_path):
    root = tmp_path / "models"
    dest = root / "faster-whisper" / "model.bin"

    resolved = fetch_models.resolve_under_root(root, dest)

    assert resolved == dest.resolve()


def test_resolve_under_root_refuses_a_destination_outside_the_root(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "evil.bin"

    with pytest.raises(fetch_models.FetchError) as excinfo:
        fetch_models.resolve_under_root(root, outside)

    message = str(excinfo.value)
    assert str(root.resolve()) in message
    assert str(outside) in message


def test_resolve_under_root_refuses_a_traversal_that_climbs_back_out(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    traversal = root / ".." / "evil.bin"

    with pytest.raises(fetch_models.FetchError):
        fetch_models.resolve_under_root(root, traversal)


# --- fetch_one -----------------------------------------------------------


def test_fetch_one_skips_a_file_already_present(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    dest = root / "already-here.bin"
    dest.write_bytes(b"existing content")

    calls: list[str] = []

    def _download(url, part_path):
        calls.append(url)
        raise AssertionError("must not be called for a file already present")

    result = fetch_models.fetch_one(_model_file(dest), root, _download)

    assert result.status == "already-present"
    assert result.dest == dest
    assert calls == []
    assert dest.read_bytes() == b"existing content"


def test_fetch_one_downloads_a_missing_file_and_verifies_size(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    dest = root / "sub" / "new-file.bin"
    content = b"the real file contents"

    def _download(url, part_path):
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(content)
        return fetch_models.DownloadReceipt(declared_size=len(content), declared_sha256=None)

    result = fetch_models.fetch_one(_model_file(dest), root, _download, _pinned(content))

    assert result.status == "fetched"
    assert dest.read_bytes() == content
    # The `.part` sibling is renamed away, never left behind on success.
    assert not dest.with_name(dest.name + ".part").exists()


def test_fetch_one_verifies_a_declared_digest_when_the_source_publishes_one(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    dest = root / "checked.bin"
    content = b"digest-checked content"
    digest = hashlib.sha256(content).hexdigest()

    def _download(url, part_path):
        part_path.write_bytes(content)
        return fetch_models.DownloadReceipt(declared_size=len(content), declared_sha256=digest)

    result = fetch_models.fetch_one(_model_file(dest), root, _download, _pinned(content))

    assert result.status == "fetched"
    assert dest.read_bytes() == content


def test_fetch_one_deletes_the_partial_file_when_the_size_does_not_match(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    dest = root / "truncated.bin"

    def _download(url, part_path):
        part_path.write_bytes(b"short")  # 5 bytes, source declared 999
        return fetch_models.DownloadReceipt(declared_size=999, declared_sha256=None)

    with pytest.raises(fetch_models.FetchError) as excinfo:
        fetch_models.fetch_one(_model_file(dest), root, _download, _pinned(b"short"))

    assert "999" in str(excinfo.value)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_fetch_one_deletes_the_partial_file_when_the_digest_does_not_match(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    dest = root / "corrupt.bin"

    def _download(url, part_path):
        part_path.write_bytes(b"some bytes")
        return fetch_models.DownloadReceipt(
            declared_size=len(b"some bytes"), declared_sha256="0" * 64
        )

    with pytest.raises(fetch_models.FetchError) as excinfo:
        fetch_models.fetch_one(_model_file(dest), root, _download, _pinned(b"some bytes"))

    assert "0" * 64 in str(excinfo.value)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_fetch_one_refuses_a_destination_outside_the_root_before_downloading(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "evil.bin"

    calls: list[str] = []

    def _download(url, part_path):
        calls.append(url)
        return fetch_models.DownloadReceipt(declared_size=None, declared_sha256=None)

    with pytest.raises(fetch_models.FetchError):
        fetch_models.fetch_one(_model_file(outside), root, _download)

    assert calls == []


# --- fetch_all -------------------------------------------------------------


def test_fetch_all_reports_every_file_present_or_fetched(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    already = root / "already.bin"
    already.write_bytes(b"x")
    missing = root / "missing.bin"

    def _download(url, part_path):
        part_path.write_bytes(b"new content")
        return fetch_models.DownloadReceipt(declared_size=len(b"new content"), declared_sha256=None)

    results = fetch_models.fetch_all(
        [_model_file(already), _model_file(missing)], root, _download, _pinned(b"new content")
    )

    by_dest = {result.dest: result.status for result in results}
    assert by_dest[already] == "already-present"
    assert by_dest[missing] == "fetched"


# --- plan_fetches ------------------------------------------------------


def test_plan_fetches_reads_destinations_from_config_not_a_restated_path(tmp_path):
    stt_config = SttConfig(local_model_dir=str(tmp_path / "models" / "faster-whisper"), local_model_size="small")
    tts_config = TtsConfig(
        piper_voice_path=str(tmp_path / "models" / "piper" / "en_US-lessac-medium.onnx"),
        piper_config_path=str(tmp_path / "models" / "piper" / "en_US-lessac-medium.onnx.json"),
    )

    model_root, model_files = fetch_models.plan_fetches(stt_config, tts_config)

    assert model_root == tmp_path / "models"
    dests = {model_file.dest for model_file in model_files}
    assert tmp_path / "models" / "faster-whisper" / "model.bin" in dests
    assert tmp_path / "models" / "faster-whisper" / "config.json" in dests
    assert tmp_path / "models" / "piper" / "en_US-lessac-medium.onnx" in dests
    assert tmp_path / "models" / "piper" / "en_US-lessac-medium.onnx.json" in dests


def test_plan_fetches_derives_the_faster_whisper_url_from_the_configured_size(tmp_path):
    stt_config = SttConfig(local_model_dir=str(tmp_path / "models" / "faster-whisper"), local_model_size="base")
    tts_config = TtsConfig(
        piper_voice_path=str(tmp_path / "models" / "piper" / "en_US-lessac-medium.onnx"),
        piper_config_path=str(tmp_path / "models" / "piper" / "en_US-lessac-medium.onnx.json"),
    )

    _model_root, model_files = fetch_models.plan_fetches(stt_config, tts_config)

    model_bin = next(f for f in model_files if f.dest.name == "model.bin")
    assert "faster-whisper-base" in model_bin.url


def test_plan_fetches_derives_the_piper_hub_path_from_the_voice_filename(tmp_path):
    stt_config = SttConfig(local_model_dir=str(tmp_path / "models" / "faster-whisper"))
    tts_config = TtsConfig(
        piper_voice_path=str(tmp_path / "models" / "piper" / "en_US-lessac-medium.onnx"),
        piper_config_path=str(tmp_path / "models" / "piper" / "en_US-lessac-medium.onnx.json"),
    )

    _model_root, model_files = fetch_models.plan_fetches(stt_config, tts_config)

    voice_file = next(f for f in model_files if f.dest.name == "en_US-lessac-medium.onnx")
    assert voice_file.url.endswith("en/en_US/lessac/medium/en_US-lessac-medium.onnx")


def test_piper_voice_hub_path_raises_naming_a_filename_that_does_not_follow_the_convention():
    with pytest.raises(fetch_models.FetchError):
        fetch_models._piper_voice_hub_path("not-a-recognized-voice-name.onnx")


# --- main ------------------------------------------------------------------


def test_main_prints_fetched_and_already_present_lines(tmp_path, monkeypatch, capsys):
    models_root = tmp_path / "models"
    (models_root / "piper").mkdir(parents=True)
    # Piper's config file already exists; everything else is missing.
    (models_root / "piper" / "en_US-lessac-medium.onnx.json").write_bytes(b"{}")

    config_text = f"""
stt:
  local_model_dir: "{models_root / 'faster-whisper'}"
  local_model_size: "small"
tts:
  piper_voice_path: "{models_root / 'piper' / 'en_US-lessac-medium.onnx'}"
  piper_config_path: "{models_root / 'piper' / 'en_US-lessac-medium.onnx.json'}"
brain:
  models:
    - model: "fake-model"
database:
  url: "postgresql+asyncpg://u:p@localhost/db"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    def _download(url, part_path):
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(b"fake model bytes")
        return fetch_models.DownloadReceipt(declared_size=len(b"fake model bytes"), declared_sha256=None)

    # Every URL `plan_fetches` derives, pinned to the fake bytes
    # `_download` writes -- `main` refuses an unpinned URL, which is the
    # whole point of WR-05's fix.
    _root, planned = fetch_models.plan_fetches(
        SttConfig(
            local_model_dir=str(models_root / "faster-whisper"), local_model_size="small"
        ),
        TtsConfig(
            piper_voice_path=str(models_root / "piper" / "en_US-lessac-medium.onnx"),
            piper_config_path=str(models_root / "piper" / "en_US-lessac-medium.onnx.json"),
        ),
    )
    fake_pins = {f.url: hashlib.sha256(b"fake model bytes").hexdigest() for f in planned}

    exit_code = fetch_models.main(
        ["--config", str(config_path)], download=_download, pinned=fake_pins
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "already present: piper/en_US-lessac-medium.onnx.json" in out
    assert "fetched: faster-whisper/model.bin" in out


def test_main_reports_a_config_error_without_a_traceback(tmp_path, capsys):
    missing_config = tmp_path / "does-not-exist.yaml"

    exit_code = fetch_models.main(["--config", str(missing_config)])

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


# --- WR-05 (code review): a pinned digest, actually checked -------------


def test_a_url_with_no_pinned_digest_is_refused_before_anything_is_downloaded(tmp_path):
    """WR-05. The module docstring claimed "a sha256 digest when the
    source's own response publishes one", but `default_download` returns
    `declared_sha256=None` unconditionally and for a good reason, so no
    digest was ever checked on a real fetch. Every one of these files is
    handed to a native extension.

    "Download it anyway and skip the check" is how a documented integrity
    control becomes a comment, so an unpinned URL is refused, and refused
    before any byte is written."""
    root = tmp_path / "models"
    root.mkdir()
    calls: list[str] = []

    def _download(url, part_path):
        calls.append(url)
        return fetch_models.DownloadReceipt(declared_size=None, declared_sha256=None)

    with pytest.raises(fetch_models.FetchError) as excinfo:
        fetch_models.fetch_one(
            _model_file(root / "unpinned.bin", url="https://evil.invalid/model.bin"),
            root,
            _download,
        )

    assert "https://evil.invalid/model.bin" in str(excinfo.value)
    assert calls == [], "an unpinned URL must not be fetched at all"


def test_bytes_that_do_not_match_the_pinned_digest_are_deleted_not_kept(tmp_path):
    """The substitution case the pin exists for: the source answers, the
    Content-Length matches, and the bytes are not the ones this
    repository pinned."""
    root = tmp_path / "models"
    root.mkdir()
    dest = root / "substituted.bin"
    served = b"bytes a compromised mirror served"

    def _download(url, part_path):
        part_path.write_bytes(served)
        return fetch_models.DownloadReceipt(declared_size=len(served), declared_sha256=None)

    with pytest.raises(fetch_models.FetchError) as excinfo:
        fetch_models.fetch_one(
            _model_file(dest), root, _download, _pinned(b"the bytes we actually expect")
        )

    assert "sha256" in str(excinfo.value)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_every_url_the_shipped_configuration_plans_carries_a_pinned_digest():
    """The pin table and the plan cannot drift: whatever
    `config.example.yaml`'s own `stt.*`/`tts.*` values make this script
    fetch must be a URL the table covers, or the shipped configuration
    itself would be unfetchable."""
    import yaml

    # Parsed directly rather than through `load_config`, which would
    # demand every ${VAR} in the file be set in this process's
    # environment. Only the four local-model paths matter here, and none
    # of them is a placeholder.
    raw = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config" / "config.example.yaml").read_text()
    )
    _root, planned = fetch_models.plan_fetches(
        SttConfig.from_config(raw["stt"]), TtsConfig.from_config(raw["tts"])
    )

    assert planned
    for model_file in planned:
        assert fetch_models.pinned_sha256(model_file.url), model_file.url


def test_the_pinned_sizes_are_the_two_the_project_actually_names():
    """D-09 names small or base. The refusal message lists these, so a
    silent third entry would make that message wrong."""
    assert fetch_models.pinned_faster_whisper_sizes() == {"small", "base"}
