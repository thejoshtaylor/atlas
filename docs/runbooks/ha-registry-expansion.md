# Confirming Home Assistant's registry transport (SAFE-03)

This runbook exists because of one finding this project could not check against a
real house: Home Assistant's REST API has no endpoint for the area, device, or
label registry. Only the WebSocket API answers those. Official documentation and
independent community reports both say so, but neither is the same as watching a
real Home Assistant answer the four registry commands `mcp/spire_mcp/registry.py`
depends on. Run the probe below once, against your own instance, before trusting
that "turn off everything in the office" reaches the right switches.

Take about ten minutes. You need `.env` filled in with a real `HA_URL` and
`HA_TOKEN` -- the same two variables the running assistant already requires.

## Running the probe

```bash
scripts/dev-probe-ha-registry.sh
```

This connects, authenticates, fetches all four registries, and prints their sizes.
Once that answers cleanly, run it again naming one area you actually have, to see
what it expands to:

```bash
scripts/dev-probe-ha-registry.sh --area <an area id you actually have>
```

Find an area id in Home Assistant under Settings -> Areas -- the id is the
lowercase, underscored form of the area's name (an area named "Office" is usually
`office`).

## What a healthy answer looks like

```
areas: 6
devices: 41
labels: 2
entities: 118
entities with their own area set: 22
entities inheriting their area from a device: 63
area 'office' expands to 7 entities:
  light.example_office_lamp
  switch.example_office_fan
  ...
```

Read three things off this:

1. **The four counts are non-zero** (unless your house genuinely has zero areas,
   devices, or labels, which is unlikely for anyone running this project). A zero
   here alongside no error means the commands answered but with nothing in them,
   not that they were rejected -- keep reading rather than treating this as a
   failure.
2. **"entities inheriting their area from a device" is non-zero for almost every
   real Home Assistant install.** This is the case 03-RESEARCH.md's Pitfall 2
   names: most entities never get their own `area_id` set directly, they inherit
   it from the device they belong to. A registry client that reads only the
   entity registry would report this number as zero and silently under-expand
   every area target. If your own instance genuinely reports zero here, that is
   plausible for a very newly set up house, but worth a second look before you
   trust it.
3. **The named area expanded to what you expected**, with nothing missing. Cross-
   check the printed entity ids against what you actually have in that area in
   Home Assistant's own UI.

## What an authentication failure looks like

```
authentication failed: invalid access token
the registry commands were never reached -- check HA_TOKEN
```

This means the WebSocket connection opened, but Home Assistant rejected the
token during the handshake. `HA_TOKEN` is wrong, expired, or was revoked --
generate a fresh long-lived access token (Home Assistant -> Profile -> Security
-> Long-lived access tokens) and update `.env`. This is not the transport
question this runbook exists to settle; it is an ordinary credential problem.

## If the registry commands are rejected

```
registry commands were rejected or unreachable: home assistant rejected
config/area_registry/list: unknown_command
```

**Stop, and say so.** This is the central assumption this phase's SAFE-03 work
rests on turning out to be wrong: a Home Assistant instance that answers
`auth_ok` but refuses one or more of the four `config/*_registry/list` commands.
If this happens:

1. Note your Home Assistant version (Settings -> About) and the exact error this
   probe printed.
2. Do not proceed with area, device, or label targets in voice commands against
   this instance -- entity-id-only calls are unaffected, since those never reach
   this expansion path at all.
3. Report this back before plan 03-07 (the policy editor) or any later plan
   builds further on top of an expansion path that cannot actually run. This
   phase's own `03-RESEARCH.md` (Pitfall 3, Open Question 2) is the place that
   finding belongs, since it changes a MEDIUM-confidence claim to a falsified
   one.

## What the child holds, and does not

`mcp/spire_mcp/registry.py` reaches Home Assistant over this same WebSocket
connection using the one `HA_TOKEN` the child already holds for its existing
REST calls -- no second credential, no environment variable beyond `HA_URL` and
`HA_TOKEN`, and no database connection of any kind (SAFE-09). This probe script
reads the same two variables directly from the environment, the same way the
running child does, because it is exercising the exact connection the child
makes -- not a second, parallel way of reaching Home Assistant.
