"""The xAI streaming chat completion client with tool schemas.

Constructed from `BrainConfig` using the OpenAI-compatible client xAI's own
docs document -- `XAI_API_KEY` must already be in the environment before
this class is constructed, because the client reads it at construction time
(`BrainConfig.api_key` is already the expanded value by the time it reaches
here, per `config.py`'s env-expansion contract).

260924-4iv (item c): xAI caches a repeated prompt prefix automatically, per
server -- there is no cache-id parameter to set on a request. What varies
is *which* server answers a given request; `x-grok-conv-id` (an ordinary
HTTP header, sent through the OpenAI SDK's `default_headers`) routes every
request carrying the same value to the same server, so a process that
keeps sending the same id keeps hitting that one server's own cache of the
(large, unchanging) entity catalog. `cache_routing_headers` below builds
that header from a per-process id, gated by `BrainConfig.cache_system_prompt`.

260924-4iv (item d): `build_http_client`/`warm` exist to make a wake-time
connection warm actually useful -- see `providers/base.py`'s own
`PROVIDER_KEEPALIVE_EXPIRY_S` docstring for why the default keepalive
would make a warm call pure waste.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from atlas.config import BrainConfig
from atlas.providers.base import BrainReply, ToolCall, provider_http_limits

logger = logging.getLogger("atlas.providers.brain_xai")

# 260924-4iv (item c, T-4iv-01): one id per process, generated once at
# import time -- random on purpose, never derived from the API key, the
# base URL, an entity name, or any other house data. Every `XaiBrain` in
# this process (every tier, and the shared envelope client `brain_race.py`
# builds) sends the same value, which is the whole point: xAI's own
# `x-grok-conv-id` routes same-id requests to the same server.
_CONVERSATION_ID = f"atlas-{uuid.uuid4().hex}"


def cache_routing_headers(config: BrainConfig) -> dict[str, str]:
    """`{"x-grok-conv-id": _CONVERSATION_ID}` when
    `config.cache_system_prompt` is `True`, else `{}` -- the one
    `default_headers` dict every brain-facing `AsyncOpenAI` client in this
    process is built with (260924-4iv, item c).
    """
    if not config.cache_system_prompt:
        return {}
    return {"x-grok-conv-id": _CONVERSATION_ID}


def build_http_client() -> DefaultAsyncHttpxClient:
    """One pooled, long-keepalive httpx client for every brain-facing
    `AsyncOpenAI` in this process to share (260924-4iv, item d) --
    `openai`'s own `DefaultAsyncHttpxClient` (a thin `httpx.AsyncClient`
    subclass its SDK expects) built against `provider_http_limits()`
    rather than the SDK's own 5 s-keepalive default.
    """
    return DefaultAsyncHttpxClient(limits=provider_http_limits())


class XaiBrain:
    """Streaming chat completions against xAI's OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: BrainConfig,
        model: str | None = None,
        *,
        http_client: Any | None = None,
    ) -> None:
        self._config = config
        # 260924-4iv (item d): `http_client=None` (every caller that
        # predates this plan) builds its own client here, unchanged in
        # shape from before this plan except for the added routing
        # headers below. `brain_race.py::build_tiers` passes one shared
        # client to every tier instead, so a warm call on any one tier
        # also warms every other tier's own pool.
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            http_client=http_client if http_client is not None else build_http_client(),
            default_headers=cache_routing_headers(config),
        )
        # No explicit model -> the top tier (BrainConfig.top_tier), so
        # app.py's existing `XaiBrain(config.brain)` call keeps working
        # unchanged; an explicit model is the seam plan 01.1-04 uses to build
        # one XaiBrain per tier.
        self._model = model if model is not None else config.top_tier.model
        self._resolved_model = self._model

    async def warm(self) -> None:
        """Open (or reuse) this instance's pooled connection at wake, so a
        turn that starts moments later skips the TLS handshake (260924-4iv,
        item d).

        A cheap `GET /v1/models` call, on the exact client `chat()` uses --
        never a separate "connect only" mechanism. Every `Exception` is
        swallowed, logged at debug: a warm call must never reach a turn or
        raise past the background task `app.py::_warm_providers` runs it
        in.
        """
        try:
            await self._client.models.list()
        except Exception:
            logger.debug("brain warm call failed for model %r", self._model, exc_info=True)

    async def resolve_model(self) -> str:
        """Confirm this instance's model id against `GET /v1/models`, and log it.

        `config.example.yaml` documents this as resolved rather than pinned
        blindly. An exact match wins; otherwise the configured value is kept
        as-is and logged as unresolved -- Phase 1 has no way to know in
        advance which ids the operator's own xAI account can see, so an
        unresolved id is a deployment-time signal, not a startup error.
        """
        try:
            models = await self._client.models.list()
        except Exception:  # pragma: no cover - network/credential dependent
            logger.warning("could not resolve brain model %r against /v1/models", self._model)
            return self._resolved_model
        ids = {model.id for model in models.data}
        if self._model in ids:
            self._resolved_model = self._model
        else:
            logger.warning(
                "configured brain model %r not found in /v1/models; using it unresolved",
                self._model,
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
