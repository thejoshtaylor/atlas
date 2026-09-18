"""Which entities voice is allowed to change.

This is the security boundary of the project. It lives alone, it stays pure, and it
is the one module with a real self-check.

Three rules that are easy to get backwards:

1. The denylist blocks control. It never blocks reads. How much power a machine
   draws is a fair question. Switching that machine off is not. Reads go through
   `allow_read`, which permits every well-formed entity id. Service calls go
   through `allow_call`, which does not.

2. The check belongs in code. It does not belong in a model prompt. A prompt is a
   suggestion to a language model. This pipeline listens to a room that contains a
   television, and a television can say any sentence in the language, including a
   well-formed instruction to cut power to the machine that runs this process.

3. This module holds no real entity id. The policy arrives from configuration,
   which stays out of the repository. A default policy ships with generic domain
   and service rules only. See `config/config.example.yaml`.

An area target, a device target, or a label target must be resolved to entity ids
before it reaches `allow_call`. This module refuses an unresolved target rather
than guessing. A denied switch inside an area is still denied when the caller says
"turn off everything in the office", and the only way to know that is to expand the
area first.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

# Domains where a voice command makes no sense and only widens what a misheard
# sentence can reach. These are generic to Home Assistant, so they carry no
# information about any particular house.
DEFAULT_DENY_DOMAINS: frozenset[str] = frozenset(
    {
        "automation",             # turning off the automations that run the house
        "backup",
        "hassio",
        "homeassistant",          # restart, stop, reload_config_entry
        "persistent_notification",
        "recorder",
        "shell_command",          # reachable only through a declared macro
        "system_log",
        "update",                 # firmware and add-on updates
    }
)

# Services that destroy or reconfigure rather than operate. Denied in every domain.
DEFAULT_DENY_SERVICES: frozenset[str] = frozenset(
    {
        "delete",
        "reload",
        "remove",
        "restart",
        "set_value",
        "stop",
    }
)

_ENTITY_RE = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")
_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

Mode = Literal["allow_all_except_denylist", "allowlist_only"]


class Denied(Exception):
    """Raised instead of returned, so a caller cannot ignore the result.

    `reason` reaches the user through text to speech, so it reads as speech.
    `entity_id` is for the log, which records the full attempt.
    """

    def __init__(self, reason: str, entity_id: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.entity_id = entity_id


@dataclass(frozen=True)
class Policy:
    """What voice may control. Built from configuration, never from source.

    `mode` picks one of two stances:

    - `allow_all_except_denylist` treats every entity as controllable unless a
      rule denies it. This suits a household that wants to speak to everything.
    - `allowlist_only` denies every entity unless a rule allows it. This suits a
      shared or rented space, and it fails closed when a new integration adds
      entities nobody has reviewed.

    A deny rule always wins over an allow rule.
    """

    mode: Mode = "allow_all_except_denylist"
    deny_entities: frozenset[str] = frozenset()
    deny_patterns: tuple[str, ...] = ()
    deny_domains: frozenset[str] = DEFAULT_DENY_DOMAINS
    deny_services: frozenset[str] = DEFAULT_DENY_SERVICES
    allow_entities: frozenset[str] = frozenset()
    allow_patterns: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, raw: dict | None) -> "Policy":
        """Build a policy from the `safety` block of the configuration file.

        An absent block gives the default policy, which denies the generic
        domains and services above and nothing else.
        """
        raw = raw or {}
        mode = raw.get("mode", "allow_all_except_denylist")
        if mode not in ("allow_all_except_denylist", "allowlist_only"):
            raise ValueError(f"unknown safety mode: {mode!r}")
        return cls(
            mode=mode,
            deny_entities=frozenset(_norm(e) for e in raw.get("deny_entities", ())),
            deny_patterns=tuple(raw.get("deny_patterns", ())),
            deny_domains=frozenset(raw.get("deny_domains", DEFAULT_DENY_DOMAINS)),
            deny_services=frozenset(raw.get("deny_services", DEFAULT_DENY_SERVICES)),
            allow_entities=frozenset(_norm(e) for e in raw.get("allow_entities", ())),
            allow_patterns=tuple(raw.get("allow_patterns", ())),
        )

    @classmethod
    def from_db_rows(
        cls,
        mode: str,
        deny_entities: Iterable[str],
        deny_patterns: Iterable[str],
        allow_entities: Iterable[str],
        allow_patterns: Iterable[str],
    ) -> "Policy":
        """Build a policy from database rows -- the DB-backed sibling of
        `from_config`, additive and changing nothing about the existing
        constructor, `denies_entity`, or `allow_call`.

        `deny_domains`/`deny_services` stay the code-owned defaults
        (`DEFAULT_DENY_DOMAINS`/`DEFAULT_DENY_SERVICES`): only entity-level
        rules come from the database, matching what the webapp's policy
        editor actually lets an operator edit (SAFE-07's UI scope). The
        domain and service denylists are generic to Home Assistant and
        carry no information about any particular house.
        """
        if mode not in ("allow_all_except_denylist", "allowlist_only"):
            raise ValueError(f"unknown safety mode: {mode!r}")
        return cls(
            mode=mode,
            deny_entities=frozenset(_norm(e) for e in deny_entities),
            deny_patterns=tuple(deny_patterns),
            allow_entities=frozenset(_norm(e) for e in allow_entities),
            allow_patterns=tuple(allow_patterns),
        )

    def denies_entity(self, entity_id: str) -> bool:
        if entity_id in self.deny_entities:
            return True
        if any(fnmatch.fnmatchcase(entity_id, p) for p in self.deny_patterns):
            return True
        if self.mode == "allowlist_only":
            allowed = entity_id in self.allow_entities or any(
                fnmatch.fnmatchcase(entity_id, p) for p in self.allow_patterns
            )
            return not allowed
        return False


def _norm(value: str) -> str:
    return (value or "").strip().lower()


def allow_read(entity_id: str) -> str:
    """Check an entity id for a state read. A read is never denied.

    Only the shape is checked, which keeps a malformed id out of a URL path. The
    power draw of a denied switch stays readable on purpose.
    """
    entity_id = _norm(entity_id)
    if not _ENTITY_RE.match(entity_id):
        raise Denied(f"that does not look like an entity id: {entity_id!r}")
    return entity_id


def allow_call(
    policy: Policy,
    domain: str,
    service: str,
    entity_ids: Sequence[str] | str | None,
    *,
    unresolved_targets: Iterable[str] = (),
) -> tuple[str, str, list[str]]:
    """Check a service call. Return the normalized call, or raise Denied.

    Every service call in this project passes through this function. There is no
    second path.

    `entity_ids` accepts one id or a list of ids. Every id must pass. One denied
    entity denies the whole call, because a partly applied command is worse than a
    refused one: the speaker hears a confirmation and believes all of it worked.

    `unresolved_targets` carries any `area_id`, `device_id`, or `label_id` the
    caller could not expand into entity ids. A non-empty value is refused. The
    caller must expand these against the Home Assistant registry first, then pass
    the resulting entity ids. Without that step an area target reaches a denied
    entity and nothing here would see it.
    """
    domain = _norm(domain)
    service = _norm(service)

    if not _NAME_RE.match(domain):
        raise Denied(f"that is not a valid domain: {domain!r}")
    if not _NAME_RE.match(service):
        raise Denied(f"that is not a valid service: {service!r}")

    unresolved = [t for t in unresolved_targets if t]
    if unresolved:
        # Refuse rather than guess. This is the area and device bypass.
        raise Denied("i need the exact things you mean, not a whole area")

    if domain in policy.deny_domains:
        raise Denied(f"i am not allowed to call {domain} services by voice")
    if service in policy.deny_services:
        raise Denied(f"i am not allowed to run {service} by voice")

    if entity_ids is None:
        # A service call with no target reaches every entity in the domain. No
        # spoken command means that, and `light.turn_off` with no target is a
        # cheap way to darken a whole house by accident.
        raise Denied("i need to know which thing you mean")

    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]
    if not entity_ids:
        raise Denied("i need to know which thing you mean")

    checked: list[str] = []
    for raw in entity_ids:
        entity_id = allow_read(raw)
        if entity_id.split(".", 1)[0] != domain:
            raise Denied(f"{entity_id} is not a {domain}")
        if policy.denies_entity(entity_id):
            # Vague to the speaker, complete in the log.
            raise Denied("that one is off limits", entity_id=entity_id)
        checked.append(entity_id)

    return domain, service, checked


def _demo() -> None:
    """Runnable check of the boundary. Run `python3 -m spire_mcp.safety`.

    Every entity id below is invented. No real house appears in this file.
    """

    def denies(**kwargs) -> Denied:
        try:
            allow_call(**kwargs)
        except Denied as exc:
            return exc
        raise AssertionError(f"call was allowed: {kwargs}")

    policy = Policy.from_config(
        {
            "mode": "allow_all_except_denylist",
            "deny_entities": ["switch.example_server_socket"],
            "deny_patterns": ["switch.example_camera_*"],
        }
    )

    # A denied entity stays denied whatever service reaches for it.
    for service in ("turn_on", "turn_off", "toggle"):
        exc = denies(
            policy=policy,
            domain="switch",
            service=service,
            entity_ids="switch.example_server_socket",
        )
        assert exc.entity_id == "switch.example_server_socket"

    # Reading a denied switch is allowed, including its power draw.
    assert allow_read("sensor.example_server_power") == "sensor.example_server_power"
    assert allow_read("  SENSOR.Example_Server_Power ") == "sensor.example_server_power"

    # A pattern denies a family of entities.
    assert policy.denies_entity("switch.example_camera_recordings")
    assert not policy.denies_entity("switch.example_fan")

    # An ordinary call still works, and normalizes.
    assert allow_call(policy, "switch", "turn_on", " Switch.Example_Fan ") == (
        "switch",
        "turn_on",
        ["switch.example_fan"],
    )

    # Several entities in one call all get checked.
    assert allow_call(
        policy, "light", "turn_on", ["light.a", "light.b"]
    ) == ("light", "turn_on", ["light.a", "light.b"])

    # One denied entity in a list denies the whole call. A partly applied command
    # is worse than a refused one, because the speaker hears a confirmation.
    denies(
        policy=policy,
        domain="switch",
        service="turn_off",
        entity_ids=["switch.example_fan", "switch.example_server_socket"],
    )

    # An unexpanded area or device target is refused, not guessed at.
    denies(
        policy=policy,
        domain="switch",
        service="turn_off",
        entity_ids=["switch.example_fan"],
        unresolved_targets=["area_id:office"],
    )

    # A targetless call is refused rather than fanning out over a domain.
    denies(policy=policy, domain="light", service="turn_off", entity_ids=None)
    denies(policy=policy, domain="light", service="turn_off", entity_ids=[])

    # The domain and the entity must agree.
    denies(
        policy=policy, domain="switch", service="turn_on", entity_ids="light.example"
    )

    # Generic domain and service denials apply with the default policy.
    for domain in ("automation", "homeassistant", "update"):
        denies(
            policy=policy,
            domain=domain,
            service="turn_off",
            entity_ids=f"{domain}.example",
        )
    denies(policy=policy, domain="switch", service="reload", entity_ids="switch.example_fan")

    # Allowlist mode fails closed. A new entity nobody reviewed is denied.
    strict = Policy.from_config(
        {"mode": "allowlist_only", "allow_entities": ["light.example_lamp"]}
    )
    assert allow_call(strict, "light", "turn_on", "light.example_lamp")[2] == [
        "light.example_lamp"
    ]
    denies(policy=strict, domain="light", service="turn_on", entity_ids="light.brand_new")

    # A deny rule beats an allow rule.
    both = Policy.from_config(
        {
            "mode": "allowlist_only",
            "allow_patterns": ["switch.*"],
            "deny_entities": ["switch.example_server_socket"],
        }
    )
    denies(
        policy=both,
        domain="switch",
        service="turn_off",
        entity_ids="switch.example_server_socket",
    )

    # An unknown mode is a configuration error, caught at load.
    try:
        Policy.from_config({"mode": "permissive"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode was accepted")

    print("safety: ok")


if __name__ == "__main__":
    _demo()
