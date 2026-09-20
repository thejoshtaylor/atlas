"""D-10a: the frontend type check this project reported "tsc clean" on for
three phases (6, 7, and 8) proved nothing. `bunx tsc --noEmit` run bare from
`web/` exits 0 even with a blatant, unambiguous type error present, because
root `web/tsconfig.json` declares `"files": []` and only `"references"` to
`tsconfig.app.json`/`tsconfig.node.json` -- the bare `--noEmit` form does not
follow project references, so it type-checks the (empty) file set the root
config declares and reports nothing wrong because there is nothing to check.

A verification command that cannot fail is not a verification command. This
module proves the real form -- `tsc -b`, now `web/package.json`'s own
`typecheck` script -- actually fails on a deliberate type error, so this
gap cannot silently regress again.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEB_DIR = _REPO_ROOT / "web"
_PROBE_PATH = _WEB_DIR / "src" / "__typecheck_probe__.ts"

skip_without_bun = pytest.mark.skipif(
    shutil.which("bun") is None,
    reason="bun is not on PATH -- the frontend toolchain is not available in this environment",
)


@skip_without_bun
def test_the_real_typecheck_fails_on_a_deliberate_type_error():
    """Write a probe file with an unambiguous type error under `web/src/`,
    run the working form of the check (`tsc -b`, matching `web/package.json`'s
    `typecheck` script and `build` script), and assert it actually fails and
    names the offending file. The probe is removed in a `finally` block, so a
    failing assertion here never leaves the working tree dirty."""
    _PROBE_PATH.write_text(
        "const spireVoiceTypecheckProbe: number = \"this is not a number\"\n"
        "export default spireVoiceTypecheckProbe\n",
        encoding="utf-8",
    )
    try:
        proc = subprocess.run(
            ["bunx", "tsc", "-b"],
            cwd=_WEB_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined_output = proc.stdout + proc.stderr

        assert proc.returncode != 0, (
            "`bunx tsc -b` exited 0 with a deliberate type error present -- the "
            "real type check has regressed to being vacuous again (D-10a). "
            f"output: {combined_output!r}"
        )
        assert _PROBE_PATH.name in combined_output, (
            "`bunx tsc -b` failed, but its output never named the probe file -- "
            f"cannot confirm it failed for the right reason. output: {combined_output!r}"
        )
    finally:
        _PROBE_PATH.unlink(missing_ok=True)


def test_web_package_json_declares_a_typecheck_script():
    """A cheap, always-run companion to the test above: `web/package.json`
    must name the working form of the check so a future phase can call it
    by name instead of re-deriving `tsc -b` from scratch."""
    import json

    package_json = json.loads((_WEB_DIR / "package.json").read_text(encoding="utf-8"))
    assert package_json.get("scripts", {}).get("typecheck") == "tsc -b", (
        "web/package.json's `typecheck` script must be exactly `tsc -b` -- the "
        "bare `tsc --noEmit` form does not follow project references (D-10a)"
    )
