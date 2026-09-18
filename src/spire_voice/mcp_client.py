"""The stdio MCP client wrapper: owns `spire_mcp.ha`'s subprocess for the
process lifetime, and converts its tool schemas for the language model.

Opened once, in the FastAPI lifespan, and held until shutdown -- never
per-turn. The subprocess's `env` is built explicitly, with exactly three
keys, rather than inherited from this process's environment: `HA_URL` and
`HA_TOKEN` are what the child needs, and `PYTHONPATH` is what lets
`-m spire_mcp.ha` resolve. Building it literally, not by copying and
filtering `os.environ`, is what keeps `XAI_API_KEY` out of the child process
-- the parent's provider credential has no reason to exist inside the
process that only ever talks to Home Assistant.

The child runs under `sys.executable`, never a bare `python3`. The child
imports the same `mcp` SDK this process does, so it must be the same
interpreter. A bare `python3` is whatever comes first on PATH, which outside
an activated virtualenv is the system interpreter with no SDK installed --
and because this repository has its own top-level `mcp/` directory, the
failure surfaces as a namespace-package shadow (`No module named
'mcp.server'`) rather than an honest "SDK not installed". That import runs
inside the child, at startup, so no unit test that imports the tool handlers
directly can see it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool


class McpToolHost:
    """Wraps one MCP stdio child process for its whole lifetime -- and,
    since Phase 3, can replace that child with a fresh one carrying a new
    policy (`respawn`).

    One `asyncio.Lock` (`_lock`) guards both `respawn()` and `call_tool()`,
    held for the duration of each. That serializes every tool call behind
    one lock, which is acceptable here for two reasons: there is exactly one
    child this host ever owns, and `turn/controller.py`'s own tool rounds
    are already sequential -- nothing in this codebase calls `call_tool`
    concurrently with itself today. What the lock buys is narrower and more
    important: it keeps a tool call from landing on a session that
    `respawn()` is in the middle of tearing down. No second liveness flag
    is layered on top of it -- `self._stack`/`self.session` stay the single
    source of truth for whether a child is running, the same discipline
    `FfmpegSupervisor` already follows with its own subprocess.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: list[Tool] = []
        self._lock = asyncio.Lock()
        # The three spawn arguments `respawn()` needs to repeat -- stored
        # only after a successful `start()`, so a `respawn()` called before
        # any `start()` fails the same way `call_tool()` does rather than
        # spawning a child with `None`s baked into its environment.
        self._ha_url: str | None = None
        self._ha_token: str | None = None
        self._mcp_root: str | os.PathLike[str] | None = None

    async def start(
        self,
        ha_url: str,
        ha_token: str,
        mcp_root: str | os.PathLike[str],
        safety_block: dict | None = None,
    ) -> None:
        """Spawn the tool server. `safety_block` is the raw `safety:` config.

        The child is the process that actually calls Home Assistant, so it is
        the process whose `Policy` decides. It cannot read the config file --
        it receives an explicit env, not an inherited one, which is what keeps
        `XAI_API_KEY` out of it -- so the block travels as JSON on that same
        explicit env under `SPIRE_SAFETY`.

        Passing `None` leaves the child on `safety.py`'s compiled defaults:
        the generic destructive domains and services, and no entity rules. An
        empty house policy is a real choice an operator can make; a policy the
        operator wrote and the enforcing process never received is not, which
        is why the child refuses to start on a malformed block rather than
        quietly falling back to defaults.
        """
        await self._spawn(ha_url, ha_token, mcp_root, safety_block)
        self._ha_url = ha_url
        self._ha_token = ha_token
        self._mcp_root = mcp_root

    async def respawn(self, safety_block: dict | None) -> None:
        """Replace the running child with a fresh one carrying `safety_block`.

        A policy change reaches the enforcing process this way -- by
        replacing it -- and never by mutating a running child's policy in
        place; there is no in-place update path here to get wrong. Repeats
        the same `ha_url`/`ha_token`/`mcp_root` the original `start()` call
        used, so only the policy differs between the old child and the new
        one.
        """
        async with self._lock:
            if self._ha_url is None or self._ha_token is None or self._mcp_root is None:
                raise RuntimeError("McpToolHost.respawn called before start()")
            await self._stack.aclose()
            self._stack = AsyncExitStack()
            await self._spawn(self._ha_url, self._ha_token, self._mcp_root, safety_block)

    async def _spawn(
        self,
        ha_url: str,
        ha_token: str,
        mcp_root: str | os.PathLike[str],
        safety_block: dict | None,
    ) -> None:
        """The literal-env, real-subprocess spawn both `start()` and
        `respawn()` perform -- factored out so there is exactly one place
        that builds the child's environment, not two that could drift apart."""
        env = {"HA_URL": ha_url, "HA_TOKEN": ha_token, "PYTHONPATH": str(mcp_root)}
        if safety_block is not None:
            env["SPIRE_SAFETY"] = json.dumps(safety_block)
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "spire_mcp.ha"],
            env=env,
        )
        read, write = await self._stack.enter_async_context(stdio_client(server_params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        result = await self.session.list_tools()
        self.tools = result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call one tool and return the SDK's own `CallToolResult` unchanged.

        This is what makes a refusal a distinguishable result rather than a
        generic error string: `ClientSession.call_tool` never raises for a
        tool-level failure. A `Denied` raised inside `spire_mcp.ha` becomes,
        on the wire, `CallToolResult(is_error=True, content=[TextContent(text=
        str(exc))])` -- and `str(exc)` on a `Denied` is exactly `reason` (see
        `spire_mcp.safety.Denied.__init__`). The turn controller reads
        `is_error` and `content[0].text` straight off this return value, so
        the reason crosses this boundary unchanged, not paraphrased.

        Guarded by the same `_lock` `respawn()` holds, so a call cannot land
        on a session mid-teardown.
        """
        async with self._lock:
            if self.session is None:
                raise RuntimeError("McpToolHost.call_tool called before start()")
            return await self.session.call_tool(name, arguments)

    async def aclose(self) -> None:
        await self._stack.aclose()


_MISSING = object()


def mcp_tools_to_openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Rename and nest each MCP tool's schema into the `tools=[...]` shape
    a chat-completions call expects. A rename and a nest, not a rewrite --
    the schema already is a JSON Schema object.

    The installed `mcp>=2.2,<3` SDK's `Tool` model exposes this field as the
    Python attribute `input_schema`; `inputSchema` is only its wire-format
    alias (`model_dump(by_alias=True)`), not an accessible attribute -- a
    bare `tool.inputSchema` raises `AttributeError` against a real `Tool`.
    `getattr` with a fallback, matching `app.py::_tool_result_json` and
    `turn/controller.py::_is_error`'s own camelCase/snake_case handling,
    keeps this working against either shape.

    The fallback checks presence with a sentinel, not truthiness with `or`
    -- a schema that legitimately serializes to `{}` must still win over
    the second attribute name, and a `Tool` exposing neither name must
    still fail loudly here rather than silently sending `None` as a tool's
    `parameters` inside a live chat-completions call.
    """
    results = []
    for tool in tools:
        schema = getattr(tool, "input_schema", _MISSING)
        if schema is _MISSING:
            schema = getattr(tool, "inputSchema", _MISSING)
        if schema is _MISSING:
            raise AttributeError(f"MCP Tool {tool.name!r} exposes neither input_schema nor inputSchema")
        results.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return results
