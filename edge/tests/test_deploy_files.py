"""RED for the deployment artifacts (10-09-PLAN.md Task 3): the systemd
unit, the udev rule, and the model-fetch script. Every assertion reads
the file as text -- none of these run on this dev host (no systemd, no
real USB device, no network fetch). TestUpdaterWatchdogInstall is the
exception. It runs install_current of the updater with stub commands."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from atlas_edge.config import DEFAULT_MUSIC_FIFO

_EDGE_ROOT = Path(__file__).resolve().parents[1]
_SERVICE_PATH = _EDGE_ROOT / "systemd" / "atlas-edge.service"
_LIBRESPOT_PATH = _EDGE_ROOT / "systemd" / "atlas-librespot.service"
_WATCHDOG_PATH = _EDGE_ROOT / "systemd" / "atlas-librespot-watchdog.service"
_UDEV_PATH = _EDGE_ROOT / "udev" / "99-atlas-edge-xvf3800.rules"
_FETCH_SCRIPT_PATH = _EDGE_ROOT / "scripts" / "fetch-vad-model.sh"
# tests/test_repo_hygiene.py reads the plain file name as a Home Assistant
# entity id, so the code builds the name with with_suffix.
_UPDATER_PATH = (_EDGE_ROOT / "scripts" / "update").with_suffix(".sh")


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

    def test_owns_the_runtime_directory_for_the_music_fifo(self) -> None:
        text = _service_text()
        assert "RuntimeDirectory=atlas-edge" in text
        assert "RuntimeDirectoryMode=0750" in text
        assert "RuntimeDirectoryPreserve=restart" in text


class TestLibrespotUnit:
    def _text(self) -> str:
        return _LIBRESPOT_PATH.read_text()

    def _exec_start(self) -> str:
        match = re.search(r"^ExecStart=(.+)$", self._text(), re.MULTILINE)
        assert match is not None
        return match.group(1)

    def test_exec_start_pipes_s16_pcm_into_the_config_fifo(self) -> None:
        exec_start = self._exec_start()
        assert exec_start.startswith("/usr/bin/librespot ")
        assert "--backend pipe" in exec_start
        assert f"--device {DEFAULT_MUSIC_FIFO}" in exec_start
        assert "--format S16" in exec_start
        assert "--disable-audio-cache" in exec_start

    def test_runs_as_its_own_user_never_the_token_holder(self) -> None:
        text = self._text()
        assert re.search(r"^User=atlas-librespot$", text, re.MULTILINE)
        assert "SupplementaryGroups=atlas-edge" in text
        assert not re.search(r"^User=atlas-edge$", text, re.MULTILINE)
        # A dynamic UID breaks Avahi discovery over D-Bus on stock Pi OS.
        assert not re.search(r"^DynamicUser=", text, re.MULTILINE)

    def test_restart_sandbox_and_name_settings(self) -> None:
        text = self._text()
        for line in (
            "Restart=always",
            "StartLimitIntervalSec=0",
            "NoNewPrivileges=yes",
            "Environment=LIBRESPOT_NAME=Atlas",
            "EnvironmentFile=-/etc/atlas-edge/librespot.env",
            "WantedBy=multi-user.target",
        ):
            assert line in text


class TestLibrespotWatchdogUnit:
    def _text(self) -> str:
        return _WATCHDOG_PATH.read_text()

    def _script(self) -> str:
        match = re.search(r"^ExecStart=/bin/sh -c '(.+)'$", self._text(), re.MULTILINE)
        assert match is not None
        return match.group(1)

    def test_follows_only_new_librespot_journal_lines(self) -> None:
        assert "journalctl -u atlas-librespot -f -n 0 -o cat" in self._script()

    def test_a_fixed_string_match_restarts_librespot_inside_the_pipe(self) -> None:
        parts = self._script().rsplit("|", 1)
        assert len(parts) == 2
        last_stage = parts[1]
        match_at = last_stage.index('grep -q -F "Connection to server closed"')
        restart_at = last_stage.index("systemctl restart atlas-librespot")
        assert match_at < restart_at

    def test_script_parses_as_posix_sh(self) -> None:
        # -n only parses the script. It never runs it.
        result = subprocess.run(["sh", "-n", "-c", self._script()], check=False)
        assert result.returncode == 0

    def test_runs_as_root_without_a_dynamic_user(self) -> None:
        text = self._text()
        assert re.search(r"^User=", text, re.MULTILINE) is None
        assert re.search(r"^DynamicUser=", text, re.MULTILINE) is None

    def test_keeps_watching_with_a_debounce(self) -> None:
        text = self._text()
        for line in (
            "Restart=always",
            "StartLimitIntervalSec=0",
            "NoNewPrivileges=yes",
            "WantedBy=multi-user.target",
        ):
            assert line in text
        after = re.search(r"^After=(.+)$", text, re.MULTILINE)
        assert after is not None
        assert "atlas-librespot.service" in after.group(1)
        restart_sec = re.search(r"^RestartSec=(\d+)$", text, re.MULTILINE)
        assert restart_sec is not None
        assert int(restart_sec.group(1)) >= 10


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


_WATCHDOG_NAME = "atlas-librespot-watchdog.service"

_STUB_SYSTEMCTL = """#!/bin/sh
echo "$*" >> "$STUB_LOG"
if [ "$1" = "show" ]; then
  echo 0
fi
if [ "$1" = "enable" ]; then
  exit "$STUB_ENABLE_RC"
fi
exit 0
"""

_STUB_NOOP = "#!/bin/sh\nexit 0\n"


class TestUpdaterWatchdogInstall:
    def _run(
        self,
        tmp_path: Path,
        *,
        librespot_installed: bool,
        watchdog_installed: bool = False,
        watchdog_in_checkout: bool = True,
        enable_rc: int = 0,
    ) -> tuple[subprocess.CompletedProcess[str], list[str], Path]:
        checkout = tmp_path / "checkout"
        systemd_dir = checkout / "edge" / "systemd"
        shutil.copytree(_EDGE_ROOT / "systemd", systemd_dir)
        if not watchdog_in_checkout:
            (systemd_dir / _WATCHDOG_NAME).unlink()

        units = tmp_path / "units"
        units.mkdir()
        if librespot_installed:
            (units / "atlas-librespot.service").write_text("placeholder\n")
        if watchdog_installed:
            (units / _WATCHDOG_NAME).write_text("placeholder\n")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, body in (
            ("systemctl", _STUB_SYSTEMCTL),
            ("uv", _STUB_NOOP),
            ("sleep", _STUB_NOOP),
        ):
            stub = bin_dir / name
            stub.write_text(body)
            stub.chmod(0o755)

        lines = _UPDATER_PATH.read_text().splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        assert lines[-1] == 'main "$@"'
        functions = tmp_path / "updater_functions.bash"
        functions.write_text("\n".join(lines[:-1]) + "\n")

        stub_log = tmp_path / "stub.log"
        stub_log.write_text("")
        env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "ATLAS_EDGE_DIR": str(checkout),
            "STUB_LOG": str(stub_log),
            "STUB_ENABLE_RC": str(enable_rc),
        }
        script = 'source "$1"; UNITS_DIR="$2"; if install_current; then exit 0; fi; exit 1'
        result = subprocess.run(
            ["bash", "-c", script, "_", str(functions), str(units)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return result, stub_log.read_text().splitlines(), units

    def test_installs_and_enables_the_watchdog_next_to_librespot(self, tmp_path: Path) -> None:
        result, log, units = self._run(tmp_path, librespot_installed=True)
        assert result.returncode == 0, result.stderr
        assert (units / _WATCHDOG_NAME).read_bytes() == _WATCHDOG_PATH.read_bytes()
        reload_at = log.index("daemon-reload")
        enable_at = log.index("enable --now atlas-librespot-watchdog")
        assert reload_at < enable_at
        assert "restart atlas-edge" in log

    def test_skips_the_watchdog_without_librespot(self, tmp_path: Path) -> None:
        result, log, units = self._run(tmp_path, librespot_installed=False)
        assert result.returncode == 0, result.stderr
        assert not (units / _WATCHDOG_NAME).exists()
        assert not any(line.startswith("enable") for line in log)

    def test_refreshes_an_installed_watchdog_but_never_enables_it_again(
        self, tmp_path: Path
    ) -> None:
        result, log, units = self._run(
            tmp_path, librespot_installed=True, watchdog_installed=True
        )
        assert result.returncode == 0, result.stderr
        assert (units / _WATCHDOG_NAME).read_bytes() == _WATCHDOG_PATH.read_bytes()
        assert not any(line.startswith("enable") for line in log)

    def test_a_failed_enable_only_logs(self, tmp_path: Path) -> None:
        result, log, _units = self._run(tmp_path, librespot_installed=True, enable_rc=1)
        assert result.returncode == 0, result.stderr
        assert "atlas-librespot-watchdog" in result.stderr
        assert "restart atlas-edge" in log

    def test_a_checkout_without_the_watchdog_skips_the_block(self, tmp_path: Path) -> None:
        result, log, units = self._run(
            tmp_path, librespot_installed=True, watchdog_in_checkout=False
        )
        assert result.returncode == 0, result.stderr
        assert not (units / _WATCHDOG_NAME).exists()
        assert not any(line.startswith("enable") for line in log)

    def test_the_updater_script_parses_as_bash(self) -> None:
        result = subprocess.run(["bash", "-n", str(_UPDATER_PATH)], check=False)
        assert result.returncode == 0
