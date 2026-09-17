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
    entity_id: str,
) -> dict[str, Any]:
    """Run one Home Assistant service call, gated by `allow_call`.

    `allow_call` runs before any `httpx` request is constructed. A denied
    call never reaches this function's `client.post` line at all -- it
    raises before that line runs.
    """
    domain, service, entity_ids = allow_call(policy, domain, service, entity_id)
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

# Built once at process start, never per call, matching `safety.py`'s own
# `from_config` habit. The child process receives only `HA_URL`, `HA_TOKEN`,
# and `PYTHONPATH` from its parent (see `mcp_client.py`) -- no `safety:`
# config reaches this process, so the default policy applies here. A
# database-backed, per-house policy threaded through the child process is
# Phase 3 (CONTEXT.md); the generic deny domains/services in `safety.py`
# already gate the destructive/administrative surface in the meantime.
_policy: Policy = Policy.from_config(None)
_http_client: httpx.AsyncClient | None = None
_base_url: str = ""
_token: str = ""


@mcp_server.tool()
async def ha_call_service(domain: str, service: str, entity_id: str) -> dict[str, Any]:
    """Run one Home Assistant service call against an allowed entity."""
    assert _http_client is not None, "ha_call_service invoked before startup"
    return await handle_call_service(_policy, _http_client, _base_url, _token, domain, service, entity_id)


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
