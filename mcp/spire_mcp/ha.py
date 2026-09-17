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
import os
from typing import Any

import httpx

from mcp.server.mcpserver import MCPServer
from spire_mcp.safety import Policy, allow_call, allow_read


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
) -> dict[str, Any]:
    """Run one Home Assistant service call, gated by `allow_call`.

    `allow_call` runs before any `httpx` request is constructed. A denied
    call, or a call carrying an `area_id`/`device_id`/`label_id` that was
    never expanded to entity ids, never reaches this function's
    `client.post` line at all -- both raise before that line runs.

    Per D-15 this phase builds no registry expansion: `area_id`,
    `device_id`, and `label_id` are accepted so the caller's target is
    never silently dropped, then handed to `allow_call`'s
    `unresolved_targets` so it can refuse rather than guess. SAFE-03 in
    Phase 3 is what fills the expansion in.
    """
    unresolved_targets = [t for t in (area_id, device_id, label_id) if t]
    domain, service, entity_ids = allow_call(
        policy, domain, service, entity_id, unresolved_targets=unresolved_targets
    )
    response = await client.post(
        f"{base_url}/api/services/{domain}/{service}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"entity_id": entity_ids[0]},
    )
    if response.status_code // 100 != 2:
        # A non-2xx response is surfaced as an error result, never an empty
        # success -- the prior incident on this host hid itself exactly this
        # way, because nothing recorded the failure (CMD-01).
        return {"error": f"home assistant returned {response.status_code}"}
    return {"changed": response.json()}


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


@mcp_server.tool()
async def ha_call_service(
    domain: str,
    service: str,
    entity_id: str | None = None,
    area_id: str | None = None,
    device_id: str | None = None,
    label_id: str | None = None,
) -> dict[str, Any]:
    """Run one Home Assistant service call against an allowed entity.

    `area_id`, `device_id`, and `label_id` are accepted so a target selector
    is never silently dropped, but any of them being non-empty is refused
    rather than expanded -- this phase builds no registry expansion (D-15).
    """
    assert _http_client is not None, "ha_call_service invoked before startup"
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
    )


@mcp_server.tool()
async def ha_get_state(entity_id: str) -> dict[str, Any]:
    """Read one Home Assistant entity's current state and attributes."""
    assert _http_client is not None, "ha_get_state invoked before startup"
    return await handle_get_state(_http_client, _base_url, _token, entity_id)


@mcp_server.tool()
async def ha_list_entities() -> list[dict[str, Any]]:
    """List every Home Assistant entity's id, friendly name, and state."""
    assert _http_client is not None, "ha_list_entities invoked before startup"
    return await handle_list_entities(_policy, _http_client, _base_url, _token)


def _startup() -> None:
    global _http_client, _base_url, _token
    _base_url = os.environ["HA_URL"]
    _token = os.environ["HA_TOKEN"]
    _http_client = httpx.AsyncClient()


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
