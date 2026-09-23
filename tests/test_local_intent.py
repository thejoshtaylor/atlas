"""260922-lim-02: the local on/off matcher, pure and table-tested.

Every entity id below is invented (`example_*`), per
`tests/test_repo_hygiene.py`'s own convention -- no real house appears here.
"""

from __future__ import annotations

from spire_voice.turn.local_intent import LocalIntent, match_on_off

# The main fixture: one clear target (a switch with a generic physical-object
# suffix a spoken command routinely omits) plus two unrelated devices whose
# names are close enough to matter for the confidence margin, but not close
# enough to beat the real target.
_ENTITIES = [
    {"entity_id": "switch.example_cooler_socket", "friendly_name": "Example Cooler Socket", "state": "off"},
    {"entity_id": "light.example_lamp", "friendly_name": "Example Lamp", "state": "off"},
    {"entity_id": "fan.example_fan", "friendly_name": "Example Fan", "state": "off"},
    {"entity_id": "switch.example_porch_socket", "friendly_name": "Example Porch Socket", "state": "on"},
]


def test_a_plain_off_command_matches_the_named_switch():
    intent = match_on_off("turn off the example cooler", _ENTITIES)
    assert intent == LocalIntent(
        domain="switch",
        service="turn_off",
        entity_id="switch.example_cooler_socket",
        friendly_name="Example Cooler Socket",
    )


def test_a_garbled_verb_still_matches_by_name_and_a_trailing_off_token():
    """"we're off the example cool" -- the garbled-verb path: "were" (what
    "we're" normalizes to once its apostrophe is stripped) plus a trailing
    "off" and a confident name match is enough, with no "turn"/"switch"
    verb at all.
    """
    intent = match_on_off("we're off the example cool", _ENTITIES)
    assert intent == LocalIntent(
        domain="switch",
        service="turn_off",
        entity_id="switch.example_cooler_socket",
        friendly_name="Example Cooler Socket",
    )


def test_a_plain_on_command_matches_a_light():
    intent = match_on_off("turn on example lamp", _ENTITIES)
    assert intent == LocalIntent(
        domain="light", service="turn_on", entity_id="light.example_lamp", friendly_name="Example Lamp"
    )


def test_a_question_never_matches():
    assert match_on_off("what time is it", _ENTITIES) is None


def test_a_bare_name_and_off_with_no_verb_never_matches():
    """Keyterm biasing turned unclear television speech into a bare
    "<device> off" on the live camera; with no verb it goes to the brain."""
    assert match_on_off("example cooler off", _ENTITIES) is None


def test_an_unsupported_verb_never_matches():
    """"dim" is not turn/switch/shut, and there is no on/off token at all --
    the brain handles dimming, this matcher does not."""
    assert match_on_off("dim the example lamp", _ENTITIES) is None


def test_an_equally_close_ambiguous_pair_matches_neither():
    ambiguous_entities = [
        {"entity_id": "switch.example_twin_a", "friendly_name": "Example Twin Plug", "state": "off"},
        {"entity_id": "switch.example_twin_b", "friendly_name": "Example Twin Plug", "state": "off"},
    ]
    assert match_on_off("turn off the example twin plug", ambiguous_entities) is None


def test_only_the_four_supported_domains_are_ever_candidates():
    """A `sensor` entity that happens to share the target's exact name must
    never win -- a sensor has no `turn_on`/`turn_off` service to call."""
    entities = [
        {"entity_id": "sensor.example_cooler", "friendly_name": "Example Cooler", "state": "42.0"},
        {"entity_id": "switch.example_cooler_socket", "friendly_name": "Example Cooler Socket", "state": "off"},
    ]
    intent = match_on_off("turn off the example cooler", entities)
    assert intent is not None
    assert intent.entity_id == "switch.example_cooler_socket"


def test_an_unavailable_entity_is_never_a_candidate():
    """The best-named match is skipped because it is `unavailable`; nothing
    else is close enough to win, so the command falls back to the brain."""
    entities = [
        {"entity_id": "light.example_lamp", "friendly_name": "Example Lamp", "state": "unavailable"},
        {
            "entity_id": "input_boolean.example_guest_mode",
            "friendly_name": "Example Guest Mode",
            "state": "off",
        },
    ]
    assert match_on_off("turn on example lamp", entities) is None


def test_shut_alone_with_no_explicit_on_or_off_token_still_signals_off():
    intent = match_on_off("shut the example lamp", _ENTITIES)
    assert intent == LocalIntent(
        domain="light", service="turn_off", entity_id="light.example_lamp", friendly_name="Example Lamp"
    )


def test_a_bare_of_typo_after_turn_is_accepted_as_off():
    intent = match_on_off("turn of the example lamp", _ENTITIES)
    assert intent == LocalIntent(
        domain="light", service="turn_off", entity_id="light.example_lamp", friendly_name="Example Lamp"
    )


def test_a_floating_of_with_no_verb_before_it_never_signals_off():
    """The "of" typo is only accepted right after turn/switch/shut -- a
    genuine "of" elsewhere in a sentence must not be misread as "off"."""
    assert match_on_off("the temperature of the example lamp", _ENTITIES) is None
