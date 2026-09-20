"""`scripts/fetch_wake_model.py`, against a fake source and a fake archive in
a temporary directory -- no network access (D-11), and no real Vosk model
downloaded during the test run.

Loaded by file path, the same way `test_fetch_models.py` loads
`fetch_models.py`: `scripts/` is not on `pythonpath` (only `src`/`mcp` are,
per `pyproject.toml`).
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fetch_wake_model.py"
_spec = importlib.util.spec_from_file_location("fetch_wake_model", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
fetch_wake_model = importlib.util.module_from_spec(_spec)
sys.modules["fetch_wake_model"] = fetch_wake_model
_spec.loader.exec_module(fetch_wake_model)


def _write_zip(zip_path: Path, members: "dict[str, bytes]") -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _fake_download_from(zip_path: Path):
    def _download(url: str, dest_zip: Path) -> None:
        dest_zip.write_bytes(zip_path.read_bytes())

    return _download


# --- fetch_vosk_model: already present ------------------------------------


def test_fetch_vosk_model_skips_a_model_directory_already_present(tmp_path):
    model_dir = tmp_path / "vosk-model-small-en-us-0.15"
    model_dir.mkdir()
    (model_dir / "README").write_bytes(b"already here")

    def _download(url, dest_zip):
        raise AssertionError("must not download when the model directory already has content")

    status = fetch_wake_model.fetch_vosk_model(model_dir, download=_download)

    assert status == "already-present"


def test_fetch_vosk_model_refuses_a_destination_that_is_not_a_directory(tmp_path):
    model_dir = tmp_path / "vosk-model-small-en-us-0.15"
    model_dir.write_bytes(b"not a directory")

    with pytest.raises(fetch_wake_model.FetchError):
        fetch_wake_model.fetch_vosk_model(model_dir, download=lambda url, dest: None)


# --- fetch_vosk_model: a real (fake) archive ------------------------------


def test_fetch_vosk_model_extracts_the_expected_top_level_directory(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "vosk-model.zip"
    _write_zip(
        fake_zip,
        {
            "vosk-model-small-en-us-0.15/README": b"a real model file would be here",
            "vosk-model-small-en-us-0.15/conf/model.conf": b"config",
        },
    )

    status = fetch_wake_model.fetch_vosk_model(
        model_dir, download=_fake_download_from(fake_zip)
    )

    assert status == "fetched"
    assert (model_dir / "README").read_bytes() == b"a real model file would be here"
    assert (model_dir / "conf" / "model.conf").read_bytes() == b"config"


def test_fetch_vosk_model_is_idempotent_after_a_successful_fetch(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "vosk-model.zip"
    _write_zip(fake_zip, {"vosk-model-small-en-us-0.15/README": b"content"})

    first = fetch_wake_model.fetch_vosk_model(model_dir, download=_fake_download_from(fake_zip))

    def _download_must_not_run(url, dest):
        raise AssertionError("a second run must not re-download an already-fetched model")

    second = fetch_wake_model.fetch_vosk_model(model_dir, download=_download_must_not_run)

    assert first == "fetched"
    assert second == "already-present"


# --- fetch_vosk_model: refused shapes --------------------------------------


def test_fetch_vosk_model_refuses_an_empty_archive(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "empty.zip"
    _write_zip(fake_zip, {})

    with pytest.raises(fetch_wake_model.FetchError):
        fetch_wake_model.fetch_vosk_model(model_dir, download=_fake_download_from(fake_zip))


def test_fetch_vosk_model_refuses_a_member_outside_the_expected_top_level_directory(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "wrong-top.zip"
    _write_zip(fake_zip, {"some-other-model/README": b"wrong top-level directory"})

    with pytest.raises(fetch_wake_model.FetchError):
        fetch_wake_model.fetch_vosk_model(model_dir, download=_fake_download_from(fake_zip))


def test_fetch_vosk_model_refuses_a_traversal_member(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "evil.zip"
    _write_zip(
        fake_zip,
        {"vosk-model-small-en-us-0.15/../../evil.txt": b"escape attempt"},
    )

    with pytest.raises(fetch_wake_model.FetchError):
        fetch_wake_model.fetch_vosk_model(model_dir, download=_fake_download_from(fake_zip))


def test_fetch_vosk_model_refuses_an_absolute_path_member(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "absolute.zip"
    _write_zip(fake_zip, {"/etc/passwd": b"escape attempt"})

    with pytest.raises(fetch_wake_model.FetchError):
        fetch_wake_model.fetch_vosk_model(model_dir, download=_fake_download_from(fake_zip))


# --- main() ----------------------------------------------------------------


def _minimal_config_text(model_dir: Path) -> str:
    """The smallest configuration `load_config` accepts (mirroring
    `test_fetch_models.py`'s own minimal fixture: `brain.models` and
    `database.url` are both required fields with no usable default), plus
    the one `wake.vosk.model_path` override this script actually reads."""
    return f"""
brain:
  models:
    - model: "fake-model"
database:
  url: "postgresql+asyncpg://u:p@localhost/db"
wake:
  vosk:
    model_path: "{model_dir}"
"""


def test_main_reads_the_model_path_from_the_named_config(tmp_path, capsys):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_minimal_config_text(model_dir), encoding="utf-8")
    fake_zip = tmp_path / "source" / "vosk-model.zip"
    _write_zip(fake_zip, {"vosk-model-small-en-us-0.15/README": b"content"})

    exit_code = fetch_wake_model.main(
        ["--config", str(config_path)], download=_fake_download_from(fake_zip)
    )

    assert exit_code == 0
    assert (model_dir / "README").read_bytes() == b"content"
    assert "fetched" in capsys.readouterr().out


def test_main_reports_already_present_without_downloading(tmp_path, capsys):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    model_dir.mkdir(parents=True)
    (model_dir / "README").write_bytes(b"already here")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_minimal_config_text(model_dir), encoding="utf-8")

    def _download(url, dest):
        raise AssertionError("must not download when already present")

    exit_code = fetch_wake_model.main(["--config", str(config_path)], download=_download)

    assert exit_code == 0
    assert "already present" in capsys.readouterr().out


def test_main_returns_nonzero_on_a_missing_config_file(tmp_path, capsys):
    missing_config = tmp_path / "does-not-exist.yaml"

    exit_code = fetch_wake_model.main(["--config", str(missing_config)])

    assert exit_code == 1
    assert "error" in capsys.readouterr().err
