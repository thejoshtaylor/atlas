"""The stdio MCP client wrapper: owns `spire_mcp.ha`'s subprocess for the
process lifetime, and converts its tool schemas for the language model.

Opened once, in the FastAPI lifespan, and held until shutdown -- never
per-turn. The subprocess's `env` is built explicitly, with exactly three
keys, rather than inherited from this process's environment: `HA_URL` and
`HA_TOKEN` are what the child needs, and `PYTHONPATH` is what lets
`python3 -m spire_mcp.ha` resolve. Building it literally, not by copying and
filtering `os.environ`, is what keeps `XAI_API_KEY` out of the child process
-- the parent's provider credential has no reason to exist inside the
process that only ever talks to Home Assistant.
"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool


class McpToolHost:
    """Wraps one MCP stdio child process for its whole lifetime."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: list[Tool] = []

    async def start(self, ha_url: str, ha_token: str, mcp_root: str | os.PathLike[str]) -> None:
        server_params = StdioServerParameters(
            command="python3",
            args=["-m", "spire_mcp.ha"],
            env={"HA_URL": ha_url, "HA_TOKEN": ha_token, "PYTHONPATH": str(mcp_root)},
        )
        read, write = await self._stack.enter_async_context(stdio_client(server_params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        result = await self.session.list_tools()
        self.tools = result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.session is None:
            raise RuntimeError("McpToolHost.call_tool called before start()")
        return await self.session.call_tool(name, arguments)

    async def aclose(self) -> None:
        await self._stack.aclose()


def mcp_tools_to_openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Rename and nest each MCP tool's schema into the `tools=[...]` shape
    a chat-completions call expects. A rename and a nest, not a rewrite --
    `inputSchema` already is a JSON Schema object.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools
    ]
