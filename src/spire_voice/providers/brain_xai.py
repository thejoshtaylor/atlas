"""The xAI streaming chat completion client with tool schemas.

Constructed from `BrainConfig` using the OpenAI-compatible client xAI's own
docs document -- `XAI_API_KEY` must already be in the environment before
this class is constructed, because the client reads it at construction time
(`BrainConfig.api_key` is already the expanded value by the time it reaches
here, per `config.py`'s env-expansion contract).
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from spire_voice.config import BrainConfig
from spire_voice.providers.base import BrainReply, ToolCall

logger = logging.getLogger("spire_voice.providers.brain_xai")


class XaiBrain:
    """Streaming chat completions against xAI's OpenAI-compatible endpoint."""

    def __init__(self, config: BrainConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self._resolved_model = config.model

    async def resolve_model(self) -> str:
        """Confirm `BrainConfig.model` against `GET /v1/models`, and log it.

        `config.example.yaml` documents this as resolved rather than pinned
        blindly. An exact match wins; otherwise the configured value is kept
        as-is and logged as unresolved -- Phase 1 has no way to know in
        advance which ids the operator's own xAI account can see, so an
        unresolved id is a deployment-time signal, not a startup error.
        """
        try:
            models = await self._client.models.list()
        except Exception:  # pragma: no cover - network/credential dependent
            logger.warning("could not resolve brain model %r against /v1/models", self._config.model)
            return self._resolved_model
        ids = {model.id for model in models.data}
        if self._config.model in ids:
            self._resolved_model = self._config.model
        else:
            logger.warning(
                "configured brain model %r not found in /v1/models; using it unresolved",
                self._config.model,
            )
        return self._resolved_model

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> BrainReply:
        stream = await self._client.chat.completions.create(
            model=self._resolved_model,
            messages=messages,
            tools=tools or None,
            stream=True,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return await accumulate_stream(stream)


async def accumulate_stream(chunks: AsyncIterator[Any]) -> BrainReply:
    """Assemble one `BrainReply` from a streamed chat completion.

    xAI's own docs state a function call arrives whole, in a single chunk,
    not fragmented across chunks the way some other OpenAI-compatible
    endpoints behave. The by-index accumulator below handles both shapes
    identically and costs nothing extra when a call never fragments -- cheap
    insurance against an undocumented multi-tool-call edge, per Open
    Question 2.
    """
    tool_calls: dict[int, dict[str, str]] = {}
    text_parts: list[str] = []
    async for chunk in chunks:
        delta = chunk.choices[0].delta
        if delta.content:
            text_parts.append(delta.content)
        for tc in delta.tool_calls or []:
            entry = tool_calls.setdefault(tc.index, {"name": "", "arguments": ""})
            if tc.function and tc.function.name:
                entry["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                entry["arguments"] += tc.function.arguments
    calls = [
        ToolCall(name=entry["name"], arguments=json.loads(entry["arguments"]) if entry["arguments"] else {})
        for entry in tool_calls.values()
    ]
    return BrainReply(tool_calls=calls, text="".join(text_parts))
