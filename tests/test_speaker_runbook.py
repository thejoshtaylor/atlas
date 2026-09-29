"""Tests for `docs/runbooks/edge-microphone.md`'s new speaker identification
section (11-09-PLAN.md, Task 3).

Mirrors `tests/test_score_wake_engines.py`'s own runbook-proving pattern
(lines 705-730): every script path the prose names must actually exist, and
the section must state the facts a stranger following it needs.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNBOOK_PATH = _REPO_ROOT / "docs" / "runbooks" / "edge-microphone.md"


def _section_text() -> str:
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    match = re.search(r"## 15\. Speaker identification.*?(?=\n## Troubleshooting)", text, re.DOTALL)
    assert match is not None, "expected a '## 15. Speaker identification' section before Troubleshooting"
    return match.group(0)


def test_runbook_has_exactly_one_speaker_identification_section():
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    assert text.count("## 15. Speaker identification") == 1


def test_every_named_script_path_exists():
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    referenced = set(re.findall(r"scripts/[\w.\-]+\.py", text))
    assert referenced, "expected the runbook to name at least one script"
    for rel_path in referenced:
        path = _REPO_ROOT / rel_path
        assert path.exists(), f"{rel_path} named in the runbook does not exist"


def test_section_names_all_three_modes():
    section = _section_text()
    for mode in ("`off`", "`record`", "`enforce`"):
        assert mode in section, f"expected the section to name mode {mode}"


def test_section_names_the_three_commands():
    section = _section_text()
    assert "scripts/fetch_models.py --only speaker-id" in section
    assert "scripts/tune_speaker_threshold.py" in section
    assert "scripts/measure_speaker_id.py" in section


def test_section_says_a_speaker_label_never_authorizes_anything():
    section = _section_text()
    assert "never grants permission" in section.lower()


def test_section_says_the_models_are_downloaded_not_stored_in_the_repository():
    section = _section_text()
    assert "download" in section.lower()
    assert "does not store them" in section.lower() or "not stored" in section.lower()


def test_troubleshooting_table_gains_two_speaker_id_rows():
    text = _RUNBOOK_PATH.read_text(encoding="utf-8")
    troubleshooting = text[text.index("## Troubleshooting") :]
    assert "enforce` mode stops an enrolled member's turns" in troubleshooting
    assert "enforce` mode lets everyone through" in troubleshooting
