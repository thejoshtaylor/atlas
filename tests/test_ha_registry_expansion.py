"""An area, device, or label target expands to entity ids before
`allow_call` runs, or the call is refused -- never guessed at (SAFE-03,
SAFE-04).

`mcp/spire_mcp/ha.py`'s `handle_call_service` already forwards
`area_id`/`device_id`/`label_id` to `allow_call` as `unresolved_targets`,
which are always refused today -- Phase 1 never built expansion at all.
03-RESEARCH.md's Pitfall 2 names the specific way a naive expansion fails
even once built: reading only `entity_registry` and filtering on
`area_id` misses every entity whose area is inherited from its *device*
rather than set on the entity itself, which is the common case in a real
Home Assistant install. "Turn off everything in the office" would then
silently control fewer devices than the operator expects, with no error --
a functional bug that looks like a safety bug from the operator's chair.
The second test is the refusal side of the same decision: a target that
cannot be resolved at all must be refused with a named reason, never
treated as an empty (and therefore harmless-looking) expansion. The third
test is SAFE-04 restated for an expanded target specifically: one denied
entity inside an otherwise-allowed area must refuse the *whole* call, not
silently operate every other entity and skip the denied one.
"""

from __future__ import annotations


def test_an_area_expands_to_entity_ids_including_device_inherited_area():
    """An area target must expand to every entity whose own `area_id`
    matches, plus every entity of a device whose `area_id` matches and
    whose entity has no override -- 03-RESEARCH.md Pitfall 2 -- plan 03-04
    fills this in."""
    raise AssertionError(
        "plan 03-04 fills this in (SAFE-03: area expansion includes device-inherited area)"
    )


def test_an_unresolvable_target_is_refused_rather_than_guessed():
    """An area, device, or label id the registry does not recognize must be
    refused with a named reason, never expand to an empty or partial entity
    list treated as success -- plan 03-04 fills this in."""
    raise AssertionError(
        "plan 03-04 fills this in (SAFE-03: an unresolved target is refused, never guessed)"
    )


def test_one_denied_entity_in_an_expanded_area_refuses_the_whole_call():
    """An area whose expansion includes even one denied entity (e.g.
    `switch.example_server_socket` alongside allowed entities like
    `light.example_lamp`) must refuse the entire call, not operate the
    allowed entities and skip the denied one -- plan 03-04 fills this in."""
    raise AssertionError(
        "plan 03-04 fills this in (SAFE-04: one denied entity in an expanded call refuses it whole)"
    )
