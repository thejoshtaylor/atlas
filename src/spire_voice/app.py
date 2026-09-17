"""The FastAPI application: lifespan-owned resources and the turn endpoint.

Per D-19, there is no database and no durable state -- every resource here
lives and dies with the process. The `httpx` client, the MCP stdio session,
the resolved brain model id, and the entity catalog are all opened once in
`lifespan` and closed once at shutdown, never per-turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from spire_voice.config import Config, load_config
from spire_voice.mcp_client import McpToolHost, mcp_tools_to_openai_tools
from spire_voice.providers.brain_xai import XaiBrain
from spire_voice.providers.stt_xai import XaiStt
from spire_voice.providers.tts_xai import XaiTts
from spire_voice.timing import TurnTimings
from spire_voice.transports.webrtc import WebrtcTransport, create_offer_answer
from spire_voice.transports.websocket import WebSocketAudioSource
from spire_voice.turn.controller import run_turn

logger = logging.getLogger("spire_voice.app")

CONFIG_PATH = os.environ.get("SPIRE_CONFIG", "config/config.example.yaml")
STATIC_DIR = Path(__file__).parent / "static"
# The repository's own `mcp/` directory -- three levels up from this file
# (src/spire_voice/app.py -> src/spire_voice -> src -> repo root -> mcp).
MCP_ROOT = Path(__file__).resolve().parents[2] / "mcp"


def _system_prompt(entities: list[dict[str, Any]]) -> str:
    """A stable prefix listing every known entity, so the model never needs
    a discovery round trip -- and so `brain.cache_system_prompt` has a
    byte-identical prefix to cache across turns.
    """
    lines = [
        "You control a home over voice through the tools you are given. "
        "Never invent an entity id that is not listed below.",
        "Known entities:",
    ]
    for entity in entities:
        lines.append(f"- {entity['entity_id']} ({entity['friendly_name']}): {entity['state']}")
    return "\n".join(lines)


def _tool_result_json(result: Any) -> Any:
    """Best-effort extraction of a tool result's JSON payload.

    `structuredContent` is preferred when the MCP framework provides it;
    otherwise the first text content block is parsed as JSON, matching how
    `ha.py`'s handlers return plain Python data that the framework
    serializes on the way out.
    """
    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    content = getattr(result, "content", None) or []
    if content:
        text = getattr(content[0], "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return []
    return []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_config(CONFIG_PATH)
    app.state.config = config

    app.state.stt = XaiStt(config.stt)
    brain = XaiBrain(config.brain)
    app.state.brain = brain
    app.state.tts = XaiTts(config.tts)

    resolved_model = await brain.resolve_model()
    logger.info("resolved brain model: %s", resolved_model)

    tool_host = McpToolHost()
    ha_config = config.mcp_servers.get("ha")
    ha_env = ha_config.env if ha_config else {}
    await tool_host.start(
        ha_url=ha_env.get("HA_URL", ""),
        ha_token=ha_env.get("HA_TOKEN", ""),
        mcp_root=MCP_ROOT,
    )
    app.state.tool_host = tool_host
    app.state.tools_schema = mcp_tools_to_openai_tools(tool_host.tools)

    entities_result = await tool_host.call_tool("ha_list_entities", {})
    entities = _tool_result_json(entities_result)
    app.state.system_prompt = _system_prompt(entities if isinstance(entities, list) else [])

    # Background WebRTC turns (see `webrtc_offer` below) run detached from
    # the HTTP request that started them -- this set is what keeps asyncio
    # from garbage-collecting a still-running task the moment the request
    # handler returns, a well-known `asyncio.create_task` pitfall.
    app.state.background_turns = set()

    yield

    await tool_host.aclose()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/transport")
async def get_transport() -> dict[str, str]:
    """The one thing the page needs to pick a transport without guessing.

    Selected straight from `ServerConfig.transport` -- an unrecognized
    value already stopped startup back in `config.py`, so by the time this
    route can answer, the value is guaranteed to be one of the two the page
    knows how to branch on.
    """
    config: Config = app.state.config
    return {"transport": config.server.transport}


class WebrtcOfferPayload(BaseModel):
    sdp: str
    type: str


class WebrtcAnswerPayload(BaseModel):
    sdp: str
    type: str


@app.post("/webrtc/offer")
async def webrtc_offer(offer: WebrtcOfferPayload) -> WebrtcAnswerPayload:
    """The one stateless request/response the WebRTC path signals over.

    This route does not check `ServerConfig.transport` before answering --
    it always exists and always answers whatever offer arrives. Which
    transport gets *used* is a client-side decision the page makes from
    `GET /transport`; this route has no second event channel of its own to
    grow, matching the WebSocket transport's own single-socket shape.
    """
    transport = WebrtcTransport()
    answer = await create_offer_answer(transport, {"sdp": offer.sdp, "type": offer.type})

    config: Config = app.state.config
    timings = TurnTimings()
    task = asyncio.create_task(
        _run_webrtc_turn(
            transport,
            app.state.stt,
            app.state.brain,
            app.state.tts,
            app.state.tool_host,
            app.state.tools_schema,
            app.state.system_prompt,
            config.brain.max_tool_rounds,
            timings,
        )
    )
    app.state.background_turns.add(task)
    task.add_done_callback(app.state.background_turns.discard)

    return WebrtcAnswerPayload(**answer)


async def _run_webrtc_turn(transport: WebrtcTransport, *args: Any) -> None:
    """Run one turn against `transport`, then close its peer connection.

    T-1-13 accepts the DoS risk of repeated offers because "a peer
    connection is closed when its turn ends" -- the `finally` here is what
    makes that true regardless of whether the turn finished cleanly or
    raised.
    """
    try:
        await run_turn(transport, *args)
    finally:
        await transport.close()


@app.websocket("/ws/turn")
async def turn_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    source = WebSocketAudioSource(websocket)
    config: Config = websocket.app.state.config
    timings = TurnTimings()
    await run_turn(
        source,
        websocket.app.state.stt,
        websocket.app.state.brain,
        websocket.app.state.tts,
        websocket.app.state.tool_host,
        websocket.app.state.tools_schema,
        websocket.app.state.system_prompt,
        config.brain.max_tool_rounds,
        timings,
    )


if __name__ == "__main__":
    import uvicorn

    _cfg = load_config(CONFIG_PATH)
    uvicorn.run(app, host=_cfg.server.bind_host, port=_cfg.server.port)
