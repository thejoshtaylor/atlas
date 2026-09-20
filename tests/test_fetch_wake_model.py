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

    status = fetch_wake_model.fetch_vosk_model(model_dir, download=_download, expected_sha256=None)

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
        model_dir, download=_fake_download_from(fake_zip), expected_sha256=None
    )

    assert status == "fetched"
    assert (model_dir / "README").read_bytes() == b"a real model file would be here"
    assert (model_dir / "conf" / "model.conf").read_bytes() == b"config"


def test_fetch_vosk_model_is_idempotent_after_a_successful_fetch(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "vosk-model.zip"
    _write_zip(fake_zip, {"vosk-model-small-en-us-0.15/README": b"content"})

    first = fetch_wake_model.fetch_vosk_model(
        model_dir, download=_fake_download_from(fake_zip), expected_sha256=None
    )

    def _download_must_not_run(url, dest):
        raise AssertionError("a second run must not re-download an already-fetched model")

    second = fetch_wake_model.fetch_vosk_model(
        model_dir, download=_download_must_not_run, expected_sha256=None
    )

    assert first == "fetched"
    assert second == "already-present"


# --- fetch_vosk_model: refused shapes --------------------------------------


def test_fetch_vosk_model_refuses_an_empty_archive(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "empty.zip"
    _write_zip(fake_zip, {})

    with pytest.raises(fetch_wake_model.FetchError):
        fetch_wake_model.fetch_vosk_model(
            model_dir, download=_fake_download_from(fake_zip), expected_sha256=None
        )


def test_fetch_vosk_model_refuses_a_member_outside_the_expected_top_level_directory(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "wrong-top.zip"
    _write_zip(fake_zip, {"some-other-model/README": b"wrong top-level directory"})

    with pytest.raises(fetch_wake_model.FetchError):
        fetch_wake_model.fetch_vosk_model(
            model_dir, download=_fake_download_from(fake_zip), expected_sha256=None
        )


def test_fetch_vosk_model_refuses_a_traversal_member(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "evil.zip"
    _write_zip(
        fake_zip,
        {"vosk-model-small-en-us-0.15/../../evil.txt": b"escape attempt"},
    )

    with pytest.raises(fetch_wake_model.FetchError):
        fetch_wake_model.fetch_vosk_model(
            model_dir, download=_fake_download_from(fake_zip), expected_sha256=None
        )


def test_fetch_vosk_model_refuses_an_absolute_path_member(tmp_path):
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "absolute.zip"
    _write_zip(fake_zip, {"/etc/passwd": b"escape attempt"})

    with pytest.raises(fetch_wake_model.FetchError):
        fetch_wake_model.fetch_vosk_model(
            model_dir, download=_fake_download_from(fake_zip), expected_sha256=None
        )


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
        ["--config", str(config_path)],
        download=_fake_download_from(fake_zip),
        expected_sha256=None,
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


# --- WR-04 (code review): a pre-created empty destination -----------------


def test_a_pre_created_empty_destination_gets_the_model_at_the_top_not_nested(tmp_path):
    """WR-04. `shutil.move` into an existing directory moves the source
    INSIDE it, so an operator who had already created the path
    `config.example.yaml` names -- a natural thing to do, since the file
    names it -- ended up with `.../vosk-model-small-en-us-0.15/vosk-model-
    small-en-us-0.15/am/final.mdl`, while this function reported
    "fetched". `VoskWakeDetector` then failed at startup with an opaque
    Kaldi error.

    Worse, the documented remedy could not repair it: the directory was
    now non-empty, so a re-run reported "already present" and changed
    nothing. The operator was left with a deployment that would not boot
    and a script claiming success both times -- and
    `docs/runbooks/deploy-compose.md` names this script as the fix for
    "the one way a fresh clone fails to start".
    """
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    model_dir.mkdir(parents=True)
    assert list(model_dir.iterdir()) == []

    source_zip = tmp_path / "source" / "vosk.zip"
    _write_zip(
        source_zip,
        {
            "vosk-model-small-en-us-0.15/am/final.mdl": b"model-bytes",
            "vosk-model-small-en-us-0.15/conf/model.conf": b"conf-bytes",
        },
    )

    status = fetch_wake_model.fetch_vosk_model(
        model_dir, download=_fake_download_from(source_zip), expected_sha256=None
    )

    assert status == "fetched"
    # The exact path `VoskWakeDetector` opens, and the files immediately
    # under it -- never a directory of the same name one level down.
    assert (model_dir / "am" / "final.mdl").read_bytes() == b"model-bytes"
    assert (model_dir / "conf" / "model.conf").read_bytes() == b"conf-bytes"
    assert not (model_dir / model_dir.name).exists(), "the model was nested one level too deep"


def test_a_pre_created_empty_destination_is_reported_as_fetched_and_stays_repairable(tmp_path):
    """The second half: after the fix, a re-run correctly reports
    "already-present" because the directory really does hold the model --
    not because a nested copy happened to make it non-empty."""
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    model_dir.mkdir(parents=True)
    source_zip = tmp_path / "source" / "vosk.zip"
    _write_zip(source_zip, {"vosk-model-small-en-us-0.15/am/final.mdl": b"model-bytes"})

    fetch_wake_model.fetch_vosk_model(
        model_dir, download=_fake_download_from(source_zip), expected_sha256=None
    )
    second = fetch_wake_model.fetch_vosk_model(
        model_dir,
        download=lambda url, dest: (_ for _ in ()).throw(
            AssertionError("must not download a second time")
        ),
        expected_sha256=None,
    )

    assert second == "already-present"
    assert (model_dir / "am" / "final.mdl").exists()


# --- WR-05 (code review): the archive is checked before it is opened -----


def test_an_archive_whose_digest_does_not_match_is_refused_before_extraction(tmp_path):
    """WR-05. This script performed no verification at all -- not even a
    Content-Length check -- before extracting a ZIP into the models
    volume and handing it to native Kaldi code. A compromised mirror or a
    hijacked upstream account was arbitrary native-code input with
    nothing in its way.

    Refused before `zipfile.ZipFile` opens it, so nothing is extracted
    and nothing is left on the models volume."""
    model_dir = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    fake_zip = tmp_path / "source" / "substituted.zip"
    _write_zip(fake_zip, {"vosk-model-small-en-us-0.15/am/final.mdl": b"not the real model"})

    with pytest.raises(fetch_wake_model.FetchError) as excinfo:
        fetch_wake_model.fetch_vosk_model(
            model_dir,
            download=_fake_download_from(fake_zip),
            expected_sha256="0" * 64,
        )

    assert "sha256" in str(excinfo.value)
    assert not model_dir.exists(), "nothing may be written when the digest does not match"


def test_the_real_default_pins_a_digest_rather_than_trusting_the_download():
    """The default is the pin, not `None` -- a fetch that forgets to pass
    `expected_sha256` verifies, rather than skipping the check."""
    import inspect

    signature = inspect.signature(fetch_wake_model.fetch_vosk_model)
    assert signature.parameters["expected_sha256"].default == fetch_wake_model._MODEL_SHA256
    assert len(fetch_wake_model._MODEL_SHA256) == 64
    signature = inspect.signature(fetch_wake_model.main)
    assert signature.parameters["expected_sha256"].default == fetch_wake_model._MODEL_SHA256
