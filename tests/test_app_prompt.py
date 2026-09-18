"""`_catalog_prompt`/`_state_message` — the D-14 prompt split.

`_catalog_prompt` must be byte-identical across turns no matter how many
entity states change between calls: that is the entire reason
`brain.cache_system_prompt` has anything to cache. `_state_message` is the
opposite -- rebuilt every turn -- and is tested separately so the two can
never be reunited into one message by a future edit without a test noticing.

Every entity id below comes from `tests/conftest.py`'s `_FAKE_STATES`, per
`tests/test_repo_hygiene.py`'s invented-id rule.
"""

from __future__ import annotations

from datetime import datetime as _real_datetime
from zoneinfo import ZoneInfo

import conftest

import spire_voice.app as app_module
from spire_voice.app import _catalog_prompt, _state_message


def _entities_from_fake_states() -> list[dict[str, str]]:
    """One entity dict per `_FAKE_STATES` entry, in the same
    `entity_id`/`friendly_name`/`state` shape `ha.py`'s `handle_list_entities`
    returns.
    """
    return [
        {
            "entity_id": entity_id,
            "friendly_name": entity_id.split(".", 1)[1].replace("_", " "),
            "state": data["state"],
        }
        for entity_id, data in conftest._FAKE_STATES.items()
    ]


def test_catalog_prompt_is_byte_identical_when_every_state_changes():
    """The load-bearing test: mutate every entity's state between two
    builds and assert the two catalog strings are still equal. A single
    volatile value surviving into this string would invalidate
    `brain.cache_system_prompt`'s cached prefix on every turn.
    """
    entities = _entities_from_fake_states()
    first = _catalog_prompt(entities)

    for entity in entities:
        entity["state"] = f"MUTATED-{entity['state']}"

    second = _catalog_prompt(entities)

    assert first == second


def test_catalog_prompt_contains_ids_and_names_but_no_state_values():
    entities = _entities_from_fake_states()
    prompt = _catalog_prompt(entities)
    entity_lines = [line for line in prompt.splitlines() if line.startswith("- ")]

    for entity in entities:
        assert entity["entity_id"] in prompt
        assert entity["friendly_name"] in prompt

    # Scoped to the entity-listing lines themselves, not the whole prompt --
    # a short state value like "on" is a substring of ordinary instruction
    # prose ("control"), so the meaningful claim is that no *line naming an
    # entity* also carries its state value.
    distinct_state_values = {entity["state"] for entity in entities}
    for line in entity_lines:
        for state_value in distinct_state_values:
            assert state_value not in line, f"state value {state_value!r} leaked into catalog line: {line!r}"


def test_catalog_prompt_on_empty_entity_list_has_no_entity_lines_and_does_not_raise():
    prompt = _catalog_prompt([])

    assert "Known entities:" in prompt
    assert "\n- " not in prompt


def test_catalog_prompt_keeps_the_never_invent_instruction():
    prompt = _catalog_prompt(_entities_from_fake_states())

    assert "never invent an entity id" in prompt.lower()
    # The instruction and the list it refers to live in the same message --
    # moving it into the per-turn state message would separate an
    # instruction from the list it names.
    assert "Known entities:" in prompt


def test_state_message_contains_entity_ids_and_current_states():
    entities = _entities_from_fake_states()
    states = {entity["entity_id"]: entity["state"] for entity in entities}

    message = _state_message(states)

    for entity_id, state in states.items():
        assert f"- {entity_id}: {state}" in message


def test_state_message_on_empty_mapping_is_not_an_empty_string():
    message = _state_message({})

    assert message != ""
    assert "current state" in message.lower()


def test_catalog_prompt_carries_no_date_or_time_content():
    """D-01 (phase 4) adds the current date/weekday/time/timezone to
    `_state_message` only -- a volatile value leaking into
    `_catalog_prompt` would invalidate `brain.cache_system_prompt`'s
    cached prefix on every turn (see that function's own docstring)."""
    prompt = _catalog_prompt(_entities_from_fake_states())

    assert "Current time" not in prompt
    assert "Current date" not in prompt


class _FixedNowDatetime:
    """A stand-in for the `datetime` class `app.py` imports -- only `.now`
    is overridden here, returning a scripted queue of fixed instants (one
    per call) rather than the real clock. Proves `_state_message` reads a
    fresh instant on every call instead of caching its first one; every
    other `datetime` behavior (`strftime`, `.astimezone`, ...) still runs
    on the real `datetime.datetime` instances this queue hands back.
    """

    def __init__(self, instants: list) -> None:
        self._instants = list(instants)

    def now(self, tz=None):
        return self._instants.pop(0)


def test_state_message_carries_date_weekday_time_and_resolved_timezone(monkeypatch):
    zone = ZoneInfo("America/Los_Angeles")
    # 2026-09-18 is a Friday.
    fixed = _real_datetime(2026, 9, 18, 14, 30, tzinfo=zone)
    monkeypatch.setattr(app_module, "datetime", _FixedNowDatetime([fixed]))
    monkeypatch.setattr(app_module, "_resolved_timezone", zone)

    message = _state_message({})

    assert "Friday" in message
    assert "September 18, 2026" in message
    assert "14:30" in message
    assert "America/Los_Angeles" in message


def test_state_message_is_rebuilt_on_every_call_not_cached(monkeypatch):
    """Two calls a minute apart describe two different minutes -- this is
    what lets a spoken time question be answered correctly on any turn,
    not just the first one after the process started."""
    zone = ZoneInfo("America/Los_Angeles")
    first_instant = _real_datetime(2026, 9, 18, 9, 30, tzinfo=zone)
    second_instant = _real_datetime(2026, 9, 18, 9, 31, tzinfo=zone)
    monkeypatch.setattr(app_module, "datetime", _FixedNowDatetime([first_instant, second_instant]))
    monkeypatch.setattr(app_module, "_resolved_timezone", zone)

    first = _state_message({})
    second = _state_message({})

    assert first != second
    assert "09:30" in first
    assert "09:31" in second


def test_state_message_falls_back_to_the_process_zone_when_unconfigured(monkeypatch):
    """`_resolved_timezone` unset (`None`, the default before `lifespan`
    has run, and the value an operator who wrote no `server.timezone` key
    keeps) still produces a well-formed message -- the process's own local
    zone, never a raise."""
    monkeypatch.setattr(app_module, "_resolved_timezone", None)

    message = _state_message({})

    assert "Current date:" in message
    assert "Current time:" in message
