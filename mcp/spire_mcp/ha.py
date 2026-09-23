"""The stdio MCP tool server for Home Assistant control.

One rule that is easy to get backwards, stated here before any code, the
same way `safety.py`'s own docstring states its rules first: every tool
handler in this file, reads included, calls through `safety.allow_read` or
`safety.allow_call` before touching `httpx`. A future write-shaped tool
added by copying the read handler must not be able to inherit a missing
check just because reads are usually harmless -- the call site exists
uniformly here, so nothing is structurally different about how a new tool
gets added.

`Denied` is never caught in this file. It propagates out of a handler
unchanged, so its `reason` -- written to be spoken aloud -- reaches the
caller verbatim. The MCP framework converts an uncaught exception raised
inside a tool function into an error-shaped `CallToolResult` whose content
is `str(exc)`, which for a `Denied` is exactly `reason` (see
`spire_mcp.safety.Denied.__init__`). That is the mechanism, not an accident:
it is what lets the turn controller speak a refusal without a language
model ever rewording it.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import httpx

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from spire_mcp.registry import HaRegistryClient, RegistryError, UnknownRegistryTargetError, expand_target, ws_url_from_http
from spire_mcp.safety import Denied, Policy, allow_call, allow_read

# `handle_call_service`'s three target kinds, in the order they are
# checked -- a fixed tuple rather than three near-identical `if` blocks, so
# adding a fourth kind later is one line here, not three call sites.
_TARGET_KINDS: tuple[str, ...] = ("area", "device", "label")


def _ha_message(response: httpx.Response) -> str:
    """Pull Home Assistant's own explanation out of a reply.

    Home Assistant answers a rejected call with a JSON body carrying a
    `message` key. This returns that message, stripped, when the body
    parses as a dict with a usable one. A body that is not JSON, is JSON
    but not a dict, or has no usable `message`, falls back to the
    response's own text, stripped and cut to 200 characters -- a bound
    that keeps a large or malformed body from swelling the tool result the
    brain reads.
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        message = body.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return response.text.strip()[:200]


async def _resolve_target(
    registry: "HaRegistryClient | None",
    kind: str,
    target_id: str,
) -> list[str]:
    """Resolve one area/device/label target to entity ids, or raise
    `Denied` naming exactly why it could not be done. Never guesses
    (D-12): every non-success outcome raises here, before `allow_call`
    ever sees this target.

    Three distinguishable reasons, because the operator hears one of them
    and each means something different: the registry does not know this
    id at all; the registry knows it and it holds nothing; or the
    registry itself could not be reached, in which case this never falls
    back to expanding against whatever snapshot this process last
    happened to have cached.
    """
    if registry is None:
        raise Denied(f"i can't reach the home assistant registry to look up that {kind}")
    try:
        snapshot = await registry.get_snapshot()
    except RegistryError as exc:
        raise Denied(f"i can't reach the home assistant registry to look up that {kind}") from exc
    try:
        entity_ids = expand_target(snapshot, kind, target_id)
    except UnknownRegistryTargetError:
        raise Denied(f"i don't know a {kind} called {target_id!r}") from None
    if not entity_ids:
        raise Denied(f"that {kind} has nothing in it")
    return sorted(entity_ids)


async def handle_call_service(
    policy: Policy,
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    domain: str,
    service: str,
    entity_id: str | None = None,
    *,
    area_id: str | None = None,
    device_id: str | None = None,
    label_id: str | None = None,
    registry: "HaRegistryClient | None" = None,
    transition: float | None = None,
) -> dict[str, Any]:
    """Run one Home Assistant service call, gated by `allow_call`.

    An `area_id`, `device_id`, or `label_id` is expanded to entity ids
    against `registry`'s snapshot before `allow_call` ever runs (SAFE-03).
    A target that cannot be resolved raises `Denied` immediately, naming
    which of the three ways it failed -- unknown to the registry, known
    but empty, or the registry unreachable -- rather than falling through
    to `allow_call`'s own generic unresolved-target refusal. This is what
    makes the three reasons distinguishable: `allow_call` itself carries
    one fixed message for "unresolved", and this function's job is to
    never let a resolvable-or-not target reach that line un-classified.

    Every entity id that did resolve -- direct or expanded -- is checked
    by the one, unchanged `allow_call`. A denied call, or a call whose
    target could not be resolved, never reaches this function's
    `client.post` line at all -- both raise before that line runs.

    The whole checked entity id list is posted in one request, not the
    first id alone: an expanded area can carry several entities, and
    SAFE-04's guarantee -- one denied entity refuses the whole call, as
    one call -- is meaningless if the call that actually runs only touches
    the first one. Posting the list as one call, rather than looping over
    it, is what keeps a refused call from partly succeeding (CMD-07).

    `transition` is Home Assistant's own light-fade parameter (D-06): a
    light fade posts one service call carrying it, never a ramp of
    repeated calls built here. It is refused, before any request is
    built, on any domain but `light` -- what Home Assistant itself does
    with an unrecognised service-data key on a non-light domain was not
    confirmed by this phase's research (05-RESEARCH.md Open Question 1),
    so this project refuses at its own layer regardless of the hub's
    leniency. A negative or non-finite value is refused the same way: a
    fade cannot run backwards or forever.

    A response-only service, such as `todo.get_items`, is always rejected
    the first time: Home Assistant refuses to run it at all unless the
    caller already asked for the response, and it says so with a 400. On
    exactly that reply, this function posts once more with
    `?return_response` on the same URL, and returns both the changed
    states and the service's own response. The flag is never sent on the
    first post, because Home Assistant rejects it on services that give no
    response.
    """
    if transition is not None:
        if domain != "light":
            raise Denied(f"transition is only supported for lights, not {domain}")
        if not math.isfinite(transition) or transition < 0:
            raise Denied(f"transition must be a non-negative number of seconds, got {transition!r}")

    entity_ids: list[str] = [entity_id] if entity_id else []
    for kind, target_id in zip(_TARGET_KINDS, (area_id, device_id, label_id)):
        if target_id:
            entity_ids.extend(await _resolve_target(registry, kind, target_id))

    domain, service, checked_entity_ids = allow_call(policy, domain, service, entity_ids or None)
    service_data: dict[str, Any] = {"entity_id": checked_entity_ids}
    if transition is not None:
        service_data["transition"] = transition
    url = f"{base_url}/api/services/{domain}/{service}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = await client.post(url, headers=headers, json=service_data)
    if response.status_code == 400 and "requires responses" in _ha_message(response).lower():
        # Home Assistant rejected the first request before the service ran.
        # The service cannot run twice, so this retry is safe. The flag is
        # not sent on the first post, or on every post: Home Assistant
        # rejects it on services that give no response at all.
        response = await client.post(f"{url}?return_response", headers=headers, json=service_data)
    if response.status_code // 100 != 2:
        # A non-2xx response is surfaced as an error result, never an empty
        # success -- the prior incident on this host hid itself exactly this
        # way, because nothing recorded the failure (CMD-01). The message
        # names Home Assistant's own reason, so the brain can tell what
        # went wrong instead of only that something did.
        message = _ha_message(response)
        if message:
            return {"error": f"home assistant returned {response.status_code}: {message}"}
        return {"error": f"home assistant returned {response.status_code}"}
    body = response.json()
    if isinstance(body, dict) and "service_response" in body:
        return {"changed": body.get("changed_states", []), "response": body["service_response"]}
    return {"changed": body}


async def handle_get_state(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    entity_id: str,
) -> dict[str, Any]:
    """Read one entity's current state and attributes, gated by `allow_read`.

    `allow_read` runs first even though a read is never denied -- it is the
    call site a future write-shaped tool inherits by being copied from this
    handler, not a special case for the read path (RESEARCH.md Pitfall 5).
    A denied entity's state, including its power draw, is answered exactly
    like any other entity's.
    """
    entity_id = allow_read(entity_id)
    response = await client.get(
        f"{base_url}/api/states/{entity_id}", headers={"Authorization": f"Bearer {token}"}
    )
    if response.status_code == 404:
        # A question about an entity Home Assistant does not have must not
        # answer as though it did.
        return {"error": f"no such entity: {entity_id}"}
    response.raise_for_status()
    state = response.json()
    return {
        "entity_id": state["entity_id"],
        "state": state["state"],
        "attributes": state.get("attributes", {}),
    }


async def handle_list_entities(
    policy: Policy,
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
) -> list[dict[str, Any]]:
    """Fetch every entity's state, gated by `allow_read` per entity.

    A read is never denied, but the call site is uniform with the write
    path above -- nothing here is structurally different for a future
    write-shaped tool that might otherwise inherit a missing check by
    copying this handler.
    """
    response = await client.get(f"{base_url}/api/states", headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    entities: list[dict[str, Any]] = []
    for state in response.json():
        entity_id = allow_read(state["entity_id"])
        entities.append(
            {
                "entity_id": entity_id,
                "friendly_name": state.get("attributes", {}).get("friendly_name", entity_id),
                "state": state["state"],
            }
        )
    return entities


mcp_server = MCPServer("spire-ha")

def _load_policy() -> Policy:
    """Build this process's policy from the `safety:` block its parent sent.

    This is the process that calls Home Assistant, so this is the process
    whose policy decides. It cannot read the config file: it receives an
    explicit env rather than an inherited one, which is what keeps
    `XAI_API_KEY` out of here. `mcp_client.py` therefore sends the raw block
    as JSON in `SPIRE_SAFETY`.

    Absent means the operator wrote no `safety:` block, and `safety.py`'s
    compiled defaults apply -- the generic destructive domains and services,
    no entity rules.

    Malformed is different, and fails closed by refusing to start. A policy
    the operator wrote and this process could not parse must never degrade
    into "no entity rules": that is the silent failure where a denylist looks
    configured and enforces nothing. Dying at startup is loud, and the
    operator finds out before a sentence does.

    A database-backed, per-house policy is still Phase 3 (CONTEXT.md D-13);
    reading it from configuration was always Phase 1's job.
    """
    raw = os.environ.get("SPIRE_SAFETY")
    if raw is None:
        return Policy.from_config(None)
    try:
        block = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"SPIRE_SAFETY is not valid JSON: {exc}") from exc
    if block is not None and not isinstance(block, dict):
        raise SystemExit(f"SPIRE_SAFETY must be a mapping, got {type(block).__name__}")
    return Policy.from_config(block)


_policy: Policy = _load_policy()
_http_client: httpx.AsyncClient | None = None
_base_url: str = ""
_token: str = ""
_registry_client: HaRegistryClient | None = None


@mcp_server.tool()
async def ha_call_service(
    domain: str,
    service: str,
    entity_id: str | None = None,
    area_id: str | None = None,
    device_id: str | None = None,
    label_id: str | None = None,
    transition: float | None = None,
) -> dict[str, Any]:
    """Run one Home Assistant service call against an allowed entity.

    `area_id`, `device_id`, and `label_id` are expanded to entity ids
    against Home Assistant's own area/device/label registry before the
    safety check runs (SAFE-03) -- an area target reaches every entity
    that area actually holds, including one that inherits its area from
    its device rather than carrying its own. A target the registry does
    not know, a target that resolves to nothing, or a registry this
    process could not reach right now is refused by name, never guessed
    at.

    `transition` is a fade duration in seconds, supported only by lights
    (`light.turn_on`/`light.turn_off`/`light.toggle`). Given on any other
    domain, or a negative value, this call is refused before Home
    Assistant is ever reached.

    `Denied` is caught here and re-raised as `ToolError(exc.reason)` --
    `safety.py` stays a plain `Exception`, importing nothing from the `mcp`
    SDK (its own docstring: "it lives alone, it stays pure"), but the SDK's
    own tool runner treats a bare `Exception` as a crash and withholds its
    text from the caller, replacing it with a generic "Error executing
    tool" message (`mcp.server.mcpserver.tools.base`'s own docstring:
    "the exception's own text stays on the server"). `ToolError` is the
    SDK's "a failure you anticipated" channel -- its message is exactly
    what reaches `CallToolResult.content`, which is what makes a `Denied`
    reason speakable rather than silently swallowed at the process
    boundary this file's own module docstring describes.

    The two halves compose, and the composition is the point: expansion
    above raises three *distinguishable* `Denied` reasons, and every one
    of them reaches the operator's ears only because of the wrapping
    below. Without it the SDK would replace "i don't know a area called
    'x'" with "Error executing tool" and CMD-08 -- a refused command
    gives a spoken reason naming which case applied -- would be false
    for exactly the cases this phase added.
    """
    assert _http_client is not None, "ha_call_service invoked before startup"
    try:
        return await handle_call_service(
            _policy,
            _http_client,
            _base_url,
            _token,
            domain,
            service,
            entity_id,
            area_id=area_id,
            device_id=device_id,
            label_id=label_id,
            registry=_registry_client,
            transition=transition,
        )
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def ha_get_state(entity_id: str) -> dict[str, Any]:
    """Read one Home Assistant entity's current state and attributes."""
    assert _http_client is not None, "ha_get_state invoked before startup"
    try:
        return await handle_get_state(_http_client, _base_url, _token, entity_id)
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def ha_list_entities() -> list[dict[str, Any]]:
    """List every Home Assistant entity's id, friendly name, and state."""
    assert _http_client is not None, "ha_list_entities invoked before startup"
    try:
        return await handle_list_entities(_policy, _http_client, _base_url, _token)
    except Denied as exc:
        raise ToolError(exc.reason) from exc


def _startup() -> None:
    global _http_client, _base_url, _token, _registry_client
    _base_url = os.environ["HA_URL"]
    _token = os.environ["HA_TOKEN"]
    # Explicit, not relying on httpx's own default (5.0s on every axis --
    # already bounded, but this project states its own outbound bound
    # rather than depending on a library default an upgrade could change
    # silently). The poller (plan 05-01) holds a claimed step row's lock
    # for the duration of this call (05-RESEARCH.md Assumption A3,
    # Pitfall 3) -- an unbounded client here is a wedged step there.
    _http_client = httpx.AsyncClient(timeout=10.0)
    # The registry client authenticates with this same `_token` -- no
    # second credential, no environment variable of its own (SAFE-09). It
    # opens no connection here; `HaRegistryClient.get_snapshot()` fetches
    # on first use and caches, per `registry.py`'s own module doctrine.
    _registry_client = HaRegistryClient(ws_url_from_http(_base_url), _token)


async def _run() -> None:
    _startup()
    try:
        await mcp_server.run_stdio_async()
    finally:
        if _http_client is not None:
            await _http_client.aclose()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_run())
