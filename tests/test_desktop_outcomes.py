"""The internal-to-wire outcome map and its coverage walk (Phase 15, D-09).

The Mac shows fixed text for four wire outcomes. The server maps the open set
of internal `turn_outcome` values to those four with an explicit table. The
walk below reads every `turn_outcome = "..."` literal in `src/atlas` and
fails when one has no table entry, so a new outcome gets a decision.
"""

from __future__ import annotations

import re
from pathlib import Path

import atlas
from atlas.desktop.outcomes import OUTCOME_TO_WIRE, wire_outcome
from atlas.desktop.protocol import WIRE_OUTCOMES

_LITERAL = re.compile(r'turn_outcome\s*=\s*"([a-z_]+)"')


def _literals() -> set[str]:
    found: set[str] = set()
    for path in Path(atlas.__file__).parent.rglob("*.py"):
        found.update(_LITERAL.findall(path.read_text(encoding="utf-8")))
    return found


def test_every_outcome_literal_in_src_atlas_has_a_table_entry() -> None:
    literals = _literals()

    assert len(literals) >= 25, "the walk found too few literals; the regex or the path is wrong"
    assert sorted(literals - set(OUTCOME_TO_WIRE)) == []


def test_the_outcomes_the_turn_adds_without_a_literal_assignment_have_entries() -> None:
    for outcome in ("failed", "cancelled", "unknown", "answer_window_silent", "macro_failed"):
        assert outcome in OUTCOME_TO_WIRE


def test_every_mapped_value_is_a_wire_outcome() -> None:
    assert set(OUTCOME_TO_WIRE.values()) <= set(WIRE_OUTCOMES)


def test_each_group_holds_the_values_the_ui_copy_expects() -> None:
    assert wire_outcome("empty_transcript") == "no_speech"
    assert wire_outcome("barged_in") == "stopped"
    assert wire_outcome("cancelled") == "stopped"
    assert wire_outcome("brain_timeout") == "failed"
    assert wire_outcome("failed") == "failed"
    assert wire_outcome("completed") == "completed"
    assert wire_outcome("email_read") == "completed"


def test_an_unknown_outcome_maps_to_completed() -> None:
    assert wire_outcome("a_value_nobody_wrote_yet") == "completed"
    assert wire_outcome("") == "completed"
