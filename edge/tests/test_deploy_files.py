"""RED for the deployment artifacts (10-09-PLAN.md Task 3): the systemd
unit, the udev rule, and the model-fetch script. Every assertion reads
the file as text -- none of these run on this dev host (no systemd, no
real USB device, no network fetch)."""

from __future__ import annotations

import re
from pathlib import Path

_EDGE_ROOT = Path(__file__).resolve().parents[1]
_SERVICE_PATH = _EDGE_ROOT / "systemd" / "atlas-edge.service"
_UDEV_PATH = _EDGE_ROOT / "udev" / "99-atlas-edge-xvf3800.rules"
_FETCH_SCRIPT_PATH = _EDGE_ROOT / "scripts" / "fetch-vad-model.sh"


def _service_text() -> str:
    return _SERVICE_PATH.read_text()


class TestSystemdUnit:
    def test_restart_always_with_no_start_limit(self) -> None:
        text = _service_text()
        assert "Restart=always" in text
        assert "StartLimitIntervalSec=0" in text

    def test_runs_as_the_dedicated_unprivileged_user(self) -> None:
        text = _service_text()
        assert "User=atlas-edge" in text
        assert "Group=atlas-edge" in text

    def test_sandboxing_directives_present(self) -> None:
        text = _service_text()
        assert "NoNewPrivileges=yes" in text
        assert "ProtectSystem=strict" in text
        assert "ProtectHome=yes" in text

    def test_exec_start_runs_the_module(self) -> None:
        text = _service_text()
        match = re.search(r"^ExecStart=(.+)$", text, re.MULTILINE)
        assert match is not None
        assert "-m atlas_edge" in match.group(1)

    def test_supplementary_groups_include_audio_and_plugdev(self) -> None:
        text = _service_text()
        assert "SupplementaryGroups=audio plugdev" in text

    def test_wanted_by_multi_user_target(self) -> None:
        assert "WantedBy=multi-user.target" in _service_text()


class TestUdevRule:
    def test_names_exactly_the_xvf3800_vendor_and_product_id(self) -> None:
        text = _UDEV_PATH.read_text()
        assert 'idVendor}=="2886"' in text
        assert 'idProduct}=="001a"' in text
        assert 'GROUP="plugdev"' in text

    def test_is_exactly_one_rule_line(self) -> None:
        lines = [
            line
            for line in _UDEV_PATH.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(lines) == 1


class TestFetchVadModelScript:
    def test_carries_a_pinned_64_hex_digit_sha256(self) -> None:
        text = _FETCH_SCRIPT_PATH.read_text()
        match = re.search(r"EXPECTED_SHA256=\"([0-9a-f]{64})\"", text)
        assert match is not None, "no pinned 64-hex-digit sha256 found"

    def test_checks_the_download_against_the_pinned_digest(self) -> None:
        text = _FETCH_SCRIPT_PATH.read_text()
        assert "sha256sum" in text
        assert "EXPECTED_SHA256" in text
        # A mismatch must delete the downloaded file and exit non-zero,
        # not just warn.
        assert re.search(r"rm -f", text)
        assert re.search(r"exit 1", text)

    def test_the_service_itself_never_invokes_the_fetch_script(self) -> None:
        # D-11: nothing under src/atlas_edge shells out to this script or
        # otherwise fetches a model at boot -- naming it in an error
        # message that tells the operator what to run by hand is fine
        # (__main__.py's own startup check does exactly that); actually
        # invoking it (subprocess/os.system/exec) is not.
        src_root = _EDGE_ROOT / "src" / "atlas_edge"
        invocation_re = re.compile(r"(subprocess|os\.system|os\.exec\w*)\([^)]*fetch-vad-model")
        for path in src_root.rglob("*.py"):
            text = path.read_text()
            assert not invocation_re.search(text), f"{path} invokes the fetch script"
