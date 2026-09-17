"""The FastAPI application: lifespan-owned resources and the turn endpoint.

Per D-19, there is no database and no durable state -- every resource here
lives and dies with the process. The `httpx` client, the MCP stdio session,
the resolved brain model id, and the entity catalog are all opened once in
`lifespan` and closed once at shutdown, never per-turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from spire_voice.config import Config, WakeConfig, load_config
from spire_voice.mcp_client import McpToolHost, mcp_tools_to_openai_tools
from spire_voice.providers.stt_xai import XaiStt
from spire_voice.providers.tier_reply import FILLER_TEXT
from spire_voice.providers.tts_cache import precache_all
from spire_voice.providers.tts_xai import XaiTts
from spire_voice.session.recorder import SessionRecorder
from spire_voice.session.retention import RetentionScheduler
from spire_voice.sources.runner import SourceRunner
from spire_voice.speaker.ffmpeg_supervisor import FfmpegSupervisor
from spire_voice.speaker.fifo_writer import FifoWriter, SpeakerError
from spire_voice.timing import TurnTimings
from spire_voice.transports.camera import CameraAudioSource
from spire_voice.transports.webrtc import WebrtcTransport, create_offer_answer
from spire_voice.transports.websocket import WebSocketAudioSource
from spire_voice.turn import brain_race
from spire_voice.turn.controller import run_turn
from spire_voice.wake.base import WakeDetector, WakeError
from spire_voice.wake.vosk_engine import VoskWakeDetector

logger = logging.getLogger("spire_voice.app")

CONFIG_PATH = os.environ.get("SPIRE_CONFIG", "config/config.example.yaml")
STATIC_DIR = Path(__file__).parent / "static"
# The repository's own `mcp/` directory -- three levels up from this file
# (src/spire_voice/app.py -> src/spire_voice -> src -> repo root -> mcp).
MCP_ROOT = Path(__file__).resolve().parents[2] / "mcp"


def _catalog_prompt(entities: list[dict[str, Any]]) -> str:
    """Byte-identical across every turn -- the cacheable prefix.

    Carries only each known entity's id and friendly name, never a state
    value: a single volatile value here would invalidate
    `brain.cache_system_prompt`'s cached prefix on every turn any entity's
    state changed, defeating the whole split D-14 exists to make. Live
    state lives in `_state_message` instead, rebuilt every turn.
    """
    lines = [
        "You control a home over voice through the tools you are given. "
        "Never invent an entity id that is not listed below.",
        "Known entities:",
    ]
    for entity in entities:
        lines.append(f"- {entity['entity_id']} ({entity['friendly_name']})")
    return "\n".join(lines)


def _state_message(states: dict[str, str]) -> str:
    """Rebuilt every turn -- deliberately not part of the cached prefix.

    An empty mapping still renders the header with no entity lines: that is
    a message the model can read as "nothing is known," which is a
    different claim than no message at all reaching it.
    """
    lines = ["Current state:"]
    for entity_id, state in states.items():
        lines.append(f"- {entity_id}: {state}")
    return "\n".join(lines)


def _make_state_fetch(tool_host: McpToolHost) -> Callable[[], Any]:
    """Build the per-turn `state_fetch` factory `run_turn` awaits
    concurrently with the operator still speaking (D-15).

    This is a read, gated by `allow_read` inside the MCP child -- a denied
    entity's state is still returned and injected into the prompt. SAFE-02
    is deliberate here, not an oversight: a question about a denied entity
    is exactly the case that must keep working. Reuses `_tool_result_json`
    rather than re-implementing the MCP payload walk a second time.
    """

    async def _fetch() -> list[dict[str, Any]]:
        result = await tool_host.call_tool("ha_list_entities", {})
        entities = _tool_result_json(result)
        return entities if isinstance(entities, list) else []

    return _fetch


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


def _build_wake_detector(wake_config: WakeConfig) -> WakeDetector:
    """Build the wake engine `wake_config.engine` names.

    A separate, easily monkeypatched function -- `tests/test_startup_smoke.py`
    replaces this with a fake rather than exercising the real engine, since
    the real one needs a model directory that is a deployment artifact and
    is not in this repository (Task 1's own `<read_first>` note). A real
    deployment missing that model directory hits `VoskWakeDetector`'s own
    `WakeError` here and stops startup by name, which is the intended
    behavior, not a bug this function papers over.
    """
    if wake_config.engine == "vosk":
        return VoskWakeDetector(wake_config.vosk, wake_config.phrase)
    raise WakeError(
        f"wake.engine {wake_config.engine!r} has no implementation yet -- "
        "openwakeword ships as configuration (PROJECT.md Key Decision) but "
        "has no trained 'hey spire' model and no detector class in this "
        "phase (RESEARCH.md Pitfall 3)"
    )


def _build_ffmpeg_supervisor(config: Config, http_client: httpx.AsyncClient) -> FfmpegSupervisor:
    """Build the egress supervisor -- a separate, monkeypatchable function
    for the same reason `_build_wake_detector` is one: `tests/
    test_startup_smoke.py` replaces this with a fake whose `start()`/
    `stop()` never spawn a real `ffmpeg` process, so the smoke test needs
    neither the `ffmpeg` binary nor a real `speaker.fifo_path` to reach a
    running application (T-02-13's same posture, extended to this
    resource)."""
    return FfmpegSupervisor(config.speaker, http_client=http_client)


async def _open_speaker_writer(writer: FifoWriter) -> None:
    """Open `writer` in the background, never inline in `lifespan`.

    Opening a FIFO for writing blocks until a reader attaches
    (`fifo_writer.py`'s own module docstring) -- doing this inline would
    hang the whole application before the egress supervisor's `ffmpeg`
    child (plan 02-03 Task 2) ever gets a chance to attach as that reader.
    A failure here (the mount not existing yet, for instance) is logged and
    leaves the writer unopened rather than crashing startup; the FIFO
    reopen path (`FifoWriter`) has no fixed number of attempts, so this is
    the only place the failure needs handling.
    """
    try:
        await writer.open()
    except SpeakerError:
        logger.warning("speaker FIFO not yet available at startup", exc_info=True)


def _make_run_turn_for_source(app: FastAPI, config: Config) -> Callable[[Any], Any]:
    """Build the one-argument `run_turn` caller `SourceRunner` needs.

    A fresh `TurnTimings()` per call -- never shared across turns, the same
    per-turn lifetime the WebSocket/WebRTC routes below already give it --
    is why this returns a closure rather than a bound partial over a single
    `TurnTimings` instance.
    """

    async def _run(source: Any) -> None:
        timings = TurnTimings()
        session_recorder = SessionRecorder(config.session, timings)
        await run_turn(
            source,
            app.state.stt,
            app.state.brain,
            app.state.tts,
            app.state.tool_host,
            app.state.tools_schema,
            app.state.catalog_prompt,
            config.brain.max_tool_rounds,
            timings,
            config.stt.max_utterance_s,
            tiers=app.state.tier_brains,
            filler_after_ms=config.brain.filler_after_ms,
            filler_cache=app.state.filler_cache,
            macros=config.macros,
            state_fetch=_make_state_fetch(app.state.tool_host),
            session_recorder=session_recorder,
        )

    return _run


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_config(CONFIG_PATH)
    app.state.config = config

    app.state.stt = XaiStt(config.stt)
    tier_brains = brain_race.build_tiers(config.brain)
    app.state.tier_brains = tier_brains
    # Nothing else in this file reads `app.state.brain` today, but it stays
    # pointed at the top tier's `XaiBrain` so any future reader keeps seeing
    # the one that actually reaches Home Assistant.
    app.state.brain = tier_brains[-1].brain
    app.state.tts = XaiTts(config.tts)

    for tier in tier_brains:
        resolved_model = await tier.brain.resolve_model()
        logger.info("resolved brain tier %d model: %s", tier.index, resolved_model)

    tool_host = McpToolHost()
    ha_config = config.mcp_servers.get("ha")
    ha_env = ha_config.env if ha_config else {}
    await tool_host.start(
        ha_url=ha_env.get("HA_URL", ""),
        ha_token=ha_env.get("HA_TOKEN", ""),
        mcp_root=MCP_ROOT,
        safety_block=config.raw_safety,
    )
    app.state.tool_host = tool_host
    app.state.tools_schema = mcp_tools_to_openai_tools(tool_host.tools)

    entities_result = await tool_host.call_tool("ha_list_entities", {})
    entities = _tool_result_json(entities_result)
    app.state.catalog_prompt = _catalog_prompt(entities if isinstance(entities, list) else [])

    app.state.macros = config.macros

    # Every filler phrase, every operator-configured extra (short
    # confirmations, per `config.example.yaml`'s own `tts.precache` comment),
    # and every configured macro's `reply` is rendered once, here, against
    # the browser sink -- Phase 1's only consumer. A macro reply missing
    # from this list would raise at turn time instead of here (Pitfall 4,
    # 01.1-RESEARCH.md), the exact silent-REST-call regression this precache
    # step exists to prevent. A failure here propagates uncaught, matching
    # this function's existing posture toward `tool_host.start()`: a broken
    # startup should stop the process, not start it half-configured with a
    # filler (or macro-reply) path that will fall over on the first turn.
    filler_phrases = [*FILLER_TEXT.values(), *config.tts.precache, *(m.reply for m in config.macros)]
    app.state.filler_cache = await precache_all(
        app.state.tts,
        Path(config.tts.cache_dir),
        filler_phrases,
        config.tts.voice_id,
        app.state.tts.browser_sink(),
    )
    logger.info("precached %d phrases", len(app.state.filler_cache))

    # Background WebRTC turns (see `webrtc_offer` below) run detached from
    # the HTTP request that started them -- this set is what keeps asyncio
    # from garbage-collecting a still-running task the moment the request
    # handler returns, a well-known `asyncio.create_task` pitfall.
    app.state.background_turns = set()

    # The room-listens spine (plan 02-03): the speaker FIFO, the wake
    # detector, the camera source, and one `SourceRunner` task per source.
    # Built as a list from the start, even holding one entry here -- plan
    # 02-04 adds the second source, and a list that was always a list needs
    # no restructuring.
    speaker_writer = FifoWriter(config.speaker.fifo_path)
    app.state.speaker_writer = speaker_writer
    app.state.background_turns.add(asyncio.create_task(_open_speaker_writer(speaker_writer)))

    # The supervisor's own client, not the browser's httpx usage elsewhere
    # in this file (there isn't one shared today) -- opened and closed with
    # the supervisor's own lifetime, since nothing else in this process
    # needs to issue the go2rtc backchannel PUT.
    speaker_http_client = httpx.AsyncClient()
    ffmpeg_supervisor = _build_ffmpeg_supervisor(config, speaker_http_client)
    ffmpeg_supervisor.start()
    app.state.ffmpeg_supervisor = ffmpeg_supervisor

    wake_detector = _build_wake_detector(config.wake.resolve("camera"))
    app.state.wake_detector = wake_detector

    # `on_reconnect=ffmpeg_supervisor.handle_reconnect` closes a gap plan
    # 02-07 recorded and deliberately left open (outside its own file
    # scope): without this, a camera that drops and reconnects gets its
    # microphone back but never re-issues the go2rtc backchannel PUT, so
    # the speaker stays silent after a recovery nothing else would notice
    # (T-02-33's same "audio outlives what nobody noticed" shape, applied
    # to the speaker rather than the recordings). `backoff_s` is left on
    # its own default -- this plan does not change it.
    camera_source = CameraAudioSource(config.camera, speaker_writer, on_reconnect=ffmpeg_supervisor.handle_reconnect)
    camera_source.start()
    app.state.camera_source = camera_source

    camera_runner = SourceRunner(
        "camera",
        camera_source,
        wake_detector,
        camera_source.decode_for_detector,
        _make_run_turn_for_source(app, config),
    )
    app.state.source_runners = [camera_runner]
    app.state.source_runner_tasks = [asyncio.create_task(camera_runner.run())]

    # The retention sweep (DBG-06, plan 02-08): runs once at startup and
    # again every `debug.expiry_interval_s`, for the life of the process --
    # the same `start()`/`stop()` shape `ffmpeg_supervisor` above already
    # uses, and no scheduling dependency, since this is one loop that
    # sleeps and calls one function.
    retention_scheduler = RetentionScheduler(
        config.session.dir,
        config.session.retain_days,
        config.session.expiry_interval_s,
    )
    retention_scheduler.start()
    app.state.retention_scheduler = retention_scheduler

    yield

    for task in app.state.source_runner_tasks:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await camera_source.close()
    wake_detector.close()
    await ffmpeg_supervisor.stop()
    await retention_scheduler.stop()
    await speaker_http_client.aclose()
    await speaker_writer.close()
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
            app.state.catalog_prompt,
            config.brain.max_tool_rounds,
            timings,
            config.stt.max_utterance_s,
            tiers=app.state.tier_brains,
            filler_after_ms=config.brain.filler_after_ms,
            filler_cache=app.state.filler_cache,
            macros=config.macros,
            state_fetch=_make_state_fetch(app.state.tool_host),
            session_recorder=SessionRecorder(config.session, timings),
        )
    )
    app.state.background_turns.add(task)
    task.add_done_callback(app.state.background_turns.discard)

    return WebrtcAnswerPayload(**answer)


async def _run_webrtc_turn(transport: WebrtcTransport, *args: Any, **kwargs: Any) -> None:
    """Run one turn against `transport`, then close its peer connection.

    `**kwargs` forwards `run_turn`'s keyword-only `tiers`/`filler_after_ms`/
    `filler_cache`/`macros` -- `*args` alone cannot carry them. Missing this
    forward would leave the WebRTC transport silently running Phase 01's
    single-model path while the WebSocket transport races tiers (or skips
    macros entirely, per plan 01.1-05), which is exactly the
    two-different-systems failure this plan exists to avoid.

    T-1-13 accepts the DoS risk of repeated offers because "a peer
    connection is closed when its turn ends" -- the `finally` here is what
    makes that true regardless of whether the turn finished cleanly or
    raised.
    """
    try:
        await run_turn(transport, *args, **kwargs)
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
        websocket.app.state.catalog_prompt,
        config.brain.max_tool_rounds,
        timings,
        config.stt.max_utterance_s,
        tiers=websocket.app.state.tier_brains,
        filler_after_ms=config.brain.filler_after_ms,
        filler_cache=websocket.app.state.filler_cache,
        macros=config.macros,
        state_fetch=_make_state_fetch(websocket.app.state.tool_host),
        session_recorder=SessionRecorder(config.session, timings),
    )


if __name__ == "__main__":
    import uvicorn

    _cfg = load_config(CONFIG_PATH)
    uvicorn.run(app, host=_cfg.server.bind_host, port=_cfg.server.port)
