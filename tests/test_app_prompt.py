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
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import conftest

import atlas.app as app_module
from atlas.app import _catalog_prompt, _state_message
from atlas.db.repository import WorkflowRun, WorkflowStep


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


# --- Plan 06-04, Task 3: the ownership block (D-10) ----------------------


def test_catalog_prompt_with_no_ownership_block_is_unchanged_from_before_the_parameter_existed():
    """A deployment with no colliding plugins passes `""` (the default)
    and must see a prompt byte-identical to one built before this
    parameter existed at all."""
    entities = _entities_from_fake_states()

    with_default = _catalog_prompt(entities)
    with_explicit_empty = _catalog_prompt(entities, "")

    assert with_default == with_explicit_empty
    assert "ownership" not in with_default.lower()


def test_catalog_prompt_carries_the_ownership_block_when_given_one():
    entities = _entities_from_fake_states()
    ownership_block = "- weather__notify is provided by Weather."

    prompt = _catalog_prompt(entities, ownership_block)

    assert ownership_block in prompt


def test_catalog_prompt_is_byte_identical_across_two_calls_with_the_same_ownership_block():
    entities = _entities_from_fake_states()
    ownership_block = "- weather__notify is provided by Weather."

    first = _catalog_prompt(entities, ownership_block)
    second = _catalog_prompt(entities, ownership_block)

    assert first == second


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


# --- Task 2 (plan 05-05): the pending-run block, D-09 --------------------


def _pending_run(run_id: int, summary: str, *, due_at) -> WorkflowRun:
    """One pending run with one still-pending step due at `due_at` --
    enough for `summarise_pending_runs` to render a full line without
    pulling in a real repository."""
    now = _real_datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
    return WorkflowRun(
        id=run_id,
        origin="voice",
        status="pending",
        summary=summary,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
        steps=(
            WorkflowStep(
                id=run_id * 10,
                run_id=run_id,
                position=0,
                kind="speak",
                arguments={"text": "example"},
                due_at=due_at,
                status="pending",
                attempts=0,
                result_detail=None,
                fired_at=None,
            ),
        ),
    )


def test_state_message_pending_runs_block_appears_after_the_entity_states():
    entities = _entities_from_fake_states()
    states = {entity["entity_id"]: entity["state"] for entity in entities}
    now = _real_datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
    runs = (_pending_run(1, "turn off the porch light", due_at=now + timedelta(minutes=10)),)

    message = _state_message(states, runs)

    state_index = message.index("Current state:")
    runs_index = message.index("Scheduled runs:")
    assert runs_index > state_index
    # Nothing follows the pending-run block -- it is the message's own tail.
    assert message.rstrip().endswith("step remaining")


def test_state_message_empty_pending_runs_renders_the_explicit_no_runs_line():
    message = _state_message({}, ())

    assert "nothing is scheduled" in message.lower()


def test_state_message_two_pending_runs_render_both_in_a_stable_order():
    now = _real_datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
    runs = (
        _pending_run(1, "turn off the porch light", due_at=now + timedelta(minutes=10)),
        _pending_run(2, "start the coffee maker", due_at=now + timedelta(hours=2)),
    )

    first = _state_message({}, runs)
    second = _state_message({}, runs)

    assert first == second
    assert "turn off the porch light" in first
    assert "start the coffee maker" in first
    assert first.index("turn off the porch light") < first.index("start the coffee maker")


def test_state_message_an_overdue_pending_run_is_flagged_as_overdue(monkeypatch):
    zone = ZoneInfo("America/Los_Angeles")
    now = _real_datetime(2026, 9, 18, 12, 0, tzinfo=zone)
    monkeypatch.setattr(app_module, "datetime", _FixedNowDatetime([now]))
    monkeypatch.setattr(app_module, "_resolved_timezone", zone)
    overdue_due_at = now.astimezone(timezone.utc) - timedelta(minutes=5)
    runs = (_pending_run(1, "turn off the porch light", due_at=overdue_due_at),)

    message = _state_message({}, runs)

    assert "overdue" in message.lower()


# --- 260924-4iv (items a, b): domains filter and the unavailable lines ---


def test_state_message_with_no_domains_is_byte_identical_to_the_pre_4iv_output():
    entities = _entities_from_fake_states()
    states = {entity["entity_id"]: entity["state"] for entity in entities}

    with_default = _state_message(states)
    with_explicit_none = _state_message(states, (), domains=None)

    assert with_default == with_explicit_none
    for entity_id in states:
        assert entity_id in with_default


def test_state_message_domains_filter_keeps_only_the_configured_domains():
    from atlas.app import _state_message

    states = {
        "light.example_lamp": "on",
        "sensor.example_temperature": "72",
    }

    message = _state_message(states, domains=frozenset({"light"}))

    assert "light.example_lamp" in message
    assert "sensor.example_temperature" not in message
    assert "Current state (only these domains are listed: light" in message


def test_state_message_states_none_renders_the_unavailable_line_and_no_entity_lines():
    from atlas.app import _STATE_UNAVAILABLE_LINE, _state_message

    entities = _entities_from_fake_states()
    populated = _state_message({entity["entity_id"]: entity["state"] for entity in entities})
    message = _state_message(None)

    assert _STATE_UNAVAILABLE_LINE in message
    for entity in entities:
        assert entity["entity_id"] not in message
    # The date/time lines are unaffected by an unavailable state read.
    assert "Current date:" in message
    assert "Current time:" in message
    assert message != populated


def test_state_message_pending_runs_none_renders_the_unavailable_line_not_no_runs():
    from atlas.app import _PENDING_RUNS_UNAVAILABLE_LINE, _state_message
    from atlas.workflow.summary import _NO_RUNS_LINE

    message = _state_message({}, None)

    assert _PENDING_RUNS_UNAVAILABLE_LINE in message
    assert _NO_RUNS_LINE not in message


def test_tool_result_json_unwraps_the_sdks_result_envelope_for_a_list():
    """The MCP SDK returns a list-valued tool result as
    `structuredContent={"result": [...]}`. Live, reading that dict as "no
    entities" left the brain an empty house (it could not find a switch
    that Home Assistant listed)."""
    from types import SimpleNamespace

    from atlas.app import _tool_result_json

    entities = [{"entity_id": "switch.example_fan_socket", "friendly_name": "Fan Socket", "state": "off"}]
    wrapped = SimpleNamespace(structured_content={"result": entities}, content=[])
    assert _tool_result_json(wrapped) == entities

    plain_object = SimpleNamespace(structured_content={"entity_id": "switch.example_x", "state": "on"}, content=[])
    assert _tool_result_json(plain_object) == {"entity_id": "switch.example_x", "state": "on"}


def test_catalog_prompt_tells_the_brain_to_match_misheard_device_names():
    prompt = _catalog_prompt([{"entity_id": "switch.example_fan_socket", "friendly_name": "Fan Socket"}])
    assert "misheard" in prompt
    assert "- switch.example_fan_socket (Fan Socket)" in prompt
