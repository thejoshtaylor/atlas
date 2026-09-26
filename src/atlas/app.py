"""The FastAPI application: lifespan-owned resources and the turn endpoint.

Per D-19, there is no database and no durable state -- every resource here
lives and dies with the process. The `httpx` client, the MCP stdio session,
the resolved brain model id, and the entity catalog are all opened once in
`lifespan` and closed once at shutdown, never per-turn.
"""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable
from zoneinfo import ZoneInfo

import httpx
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from atlas.audio.cue import silence
from atlas.audio.ring import PrerollBuffer
from atlas.auth.dependencies import Role, require_role, require_setup_complete
from atlas.auth.edge_tokens import require_edge_device
from atlas.auth.tokens import validate_secret_key_strength
from atlas.calibration.record import EchoCalibration
from atlas.calibration.runner import find_latest_calibration, run_echo_calibration
from atlas.config import (
    EDGE_BARGE_IN_PROVEN,
    MACROS_KEY_REJECTED_ERROR,
    MCP_KEY_REJECTED_ERROR,
    SAFETY_KEY_REJECTED_ERROR,
    BargeInConfig,
    Config,
    ConfigError,
    DatabaseConfig,
    SecurityConfig,
    WakeConfig,
    load_config,
    load_raw_config,
)
from atlas.crypto.credentials import CredentialSlot, resolve_credential_value
from atlas.db.edge_postgres import PostgresEdgeDeviceRepository
from atlas.db.edge_repository import EdgeDevice
from atlas.db.engine import build_engine, get_current_revision, run_migrations
from atlas.db.google_postgres import PostgresGoogleAccountRepository
from atlas.db.pending_action_repository import PostgresPendingActionRepository
from atlas.db.postgres import (
    PostgresAccountRepository,
    PostgresCredentialRepository,
    PostgresMacroRepository,
    PostgresPluginRepository,
    PostgresPolicyRepository,
    PostgresProviderSelectionRepository,
    PostgresSettingsRepository,
    PostgresSetupRepository,
    PostgresWakeEventRepository,
    PostgresWorkflowRepository,
)
from atlas.db.repository import (
    CredentialRepository,
    MacroRepository,
    PluginRepository,
    ProviderSelectionRepository,
    SettingsRepository,
    WakeEventRepository,
    WorkflowRepository,
)
from atlas_mcp.google_tools import CODE_ONLY_TOOL_NAMES, GOOGLE_PLUGIN_MODULE

from atlas.google.env import GoogleEnvBuilder
from atlas.google.plugin import refresh_google_plugin
from atlas.google.scheduler import GoogleTokenRefreshScheduler
from atlas.google.token_service import GoogleTokenService
from atlas.google.turn_context import build_handoff_context
from atlas.loop_stall import LoopStallReporter
from atlas.mcp_client import McpToolHostLookup, UnknownToolError, mcp_tools_to_openai_tools
from atlas.plugins.manager import PluginManager
from atlas.policy_snapshot import safety_block_from_policy
from atlas.providers import registry as provider_registry
from atlas.providers.boot import (
    ProviderSlotStatus,
    degraded_turn_refusal,
    resolve_slot,
)
from atlas.providers.tier_reply import FILLER_TEXT
from atlas.providers.tts_cache import CachedTts, precache_all
from atlas.providers.tts_xai import SinkFormat
from atlas.routes import register_routers
from atlas.routes.follow_up_settings import FOLLOW_UP_WINDOW_SETTING_KEY
from atlas.routes.wake import WAKE_THRESHOLD_SETTING_KEY
from atlas.routes.wizard import resolve_audio_source, resolve_timezone
from atlas.session.observers import ObserverPublishingSource, ObserverRegistry
from atlas.session.recorder import SessionRecorder
from atlas.session.retention import RetentionScheduler
from atlas.sources.runner import SourceRunner
from atlas.speaker.ffmpeg_supervisor import FfmpegSupervisor, build_tcp_argv
from atlas.speaker.fifo_writer import FifoWriter, SpeakerError
from atlas.speaker.tapo_talk import TapoTalkSupervisor, camera_host_from_rtsp_url
from atlas.timing import TurnTimings
from atlas.transports.camera import CameraAudioSource
from atlas.transports.edge import CLOSE_NOT_CONFIGURED, EdgeAudioSource, SegmentBoundedWakeDetector
from atlas.transports.webrtc import WebrtcTransport, create_offer_answer
from atlas.transports.websocket import WebSocketAudioSource
from atlas.turn import brain_race
from atlas.turn.controller import _speak, run_turn
from atlas.wake.base import WakeDetector, WakeError
from atlas.wake.vosk_engine import VoskWakeDetector
from atlas.workflow.scheduler import WorkflowScheduler
from atlas.workflow.steps import execute_step
from atlas.workflow.summary import summarise_pending_runs
from atlas.workflow.tool import WorkflowToolHost

logger = logging.getLogger("atlas.app")

# D-08: the names every source-turn path publishes to the observer feed
# under. Not derived from `app.state.source_runners` (`SourceRunner` keeps
# its own name private) -- this application has exactly three turn-starting
# paths today, and all three names are fixed at the call site, not
# discovered:
#
#   - the camera's wake-word listener (`_make_run_turn_for_source`),
#   - the browser microphone behind `/ws/turn`,
#   - the browser's WebRTC transport behind `POST /webrtc/offer`, which
#     `GET /transport` can select instead of `/ws/turn`,
#   - the always-on browser listener behind `/ws/listen`, which reuses the
#     camera's own `_make_run_turn_for_source` path, so it adds no fourth
#     `run_turn` call site.
#
# The third was missing until the Phase 8 review found it: its turns were
# recorded and appeared in the Sessions list while `/live` sat on the idle
# "Listening for ..." state through the whole turn. D-08 says every turn is
# labelled with the source it came from and all sources appear, so a turn
# path that publishes nothing is a defect in this constant block as much as
# in the route. `tests/test_observer_feed.py` asserts the count of
# `run_turn` call sites against the count of wrapped sources, so a fourth
# path cannot go missing the same way.
CAMERA_SOURCE_NAME = "camera"
BROWSER_MIC_SOURCE_NAME = "browser_mic"
WEBRTC_SOURCE_NAME = "browser_webrtc"
LISTEN_SOURCE_NAME = "browser_listen"
# Phase 10 (D-01, D-06, D-07): the Pi + XVF3800 source, behind `/ws/edge`.
# One of `EDGE_SOURCE_NAME`/`CAMERA_SOURCE_NAME` is ever the resolved
# `audio_source` at a time (the `resolved_audio_source` branch below) --
# never both, until a later plan promotes this to a per-device list.
EDGE_SOURCE_NAME = "edge"

# What `observer.opened` advertises: every name above, in one place, so the
# opening message cannot drift from the set of paths that actually publish.
OBSERVED_SOURCE_NAMES = (
    CAMERA_SOURCE_NAME,
    BROWSER_MIC_SOURCE_NAME,
    WEBRTC_SOURCE_NAME,
    LISTEN_SOURCE_NAME,
    EDGE_SOURCE_NAME,
)

CONFIG_PATH = os.environ.get("ATLAS_CONFIG", "config/config.example.yaml")
# The repository's own `mcp/` directory -- three levels up from this file
# (src/atlas/app.py -> src/atlas -> src -> repo root -> mcp).
MCP_ROOT = Path(__file__).resolve().parents[2] / "mcp"
# `web/vite.config.ts`'s own `build.outDir` -- that file's own comment
# names this exact path as the consumer a rename there would break. Three
# levels up from this file, the same computation `MCP_ROOT` above uses,
# since `web/` lives at the repo root beside `src/`, not under it.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"

# IN-01 (STATE.md, deferred here from Phase 3's code review): the one name
# every platform's own resolver maps to loopback without a network round
# trip, checked alongside a literal IP address's own `is_loopback` --
# never a comparison against "0.0.0.0" or any other single hard-coded
# spelling of "reachable from the network" (see `_bind_is_loopback` below).
_LOOPBACK_HOSTNAMES = frozenset({"localhost"})


def _bind_is_loopback(bind_host: str) -> bool:
    """Whether `server.bind_host` reaches this process from loopback only
    -- judged by what the address means, not by comparing it against one
    hard-coded literal (IN-01). A literal IPv4/IPv6 address is asked
    directly (`ipaddress.ip_address(...).is_loopback`); `localhost` is the
    one name resolved the same way on every platform without a network
    round trip. Anything else -- an any-address bind (`0.0.0.0`, `::`), a
    named host, or a value this function cannot parse as an address at all
    -- is judged reachable: warning when the bind turns out to already be
    loopback is harmless, silently waving through a spelling this function
    did not recognize is the mistake IN-01 exists to catch.
    """
    if bind_host.strip().lower() in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(bind_host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class _LegacyConfigKey:
    """One retired top-level configuration key `lifespan`'s two-stage boot
    still tolerates on exactly one boot -- the boot whose migration just
    seeded it -- before rejecting it by name on every boot after (CR-01
    fix, generalized plan 04-05, D-16).

    `name` is the raw top-level key (`"safety"`, `"macros"`); `rejected_error`
    is the exact `ConfigError` message this project has always raised for
    it, lifted to module level in `atlas.config` so `Config.from_config`
    and this file's own database-aware check raise identical words;
    `seeded_tables_hint` names where an operator finds what was seeded, for
    the one-boot warning below. Adding a second retired key means adding a
    second entry to `_LEGACY_CONFIG_KEYS`, never a second copy of the
    sequence that reads this tuple.
    """

    name: str
    rejected_error: str
    seeded_tables_hint: str


_LEGACY_CONFIG_KEYS: tuple[_LegacyConfigKey, ...] = (
    _LegacyConfigKey(
        name="safety",
        rejected_error=SAFETY_KEY_REJECTED_ERROR,
        seeded_tables_hint="safety_policy/policy_rules tables, or the webapp's policy editor",
    ),
    _LegacyConfigKey(
        name="macros",
        rejected_error=MACROS_KEY_REJECTED_ERROR,
        seeded_tables_hint=(
            "macros/macro_actions/macro_aliases tables, or the webapp's macro editor"
        ),
    ),
    # Plan 06-01 (D-01): one more tuple entry, not a second copy of the
    # sequence this tuple drives -- the whole point of this generalization,
    # stated in its own docstring above.
    _LegacyConfigKey(
        name="mcp",
        rejected_error=MCP_KEY_REJECTED_ERROR,
        seeded_tables_hint="plugins/plugin_config_values tables, or the webapp's plugins screen",
    ),
)

# D-01 (phase 4), extended by the 260924-h2f quick task (issue #1): the
# zone `_state_message` reads the current time and date against.
# `lifespan` sets this from `routes.wizard.resolve_timezone`'s own result
# -- the saved-through-the-webapp setting, then `config.server.timezone`,
# then Home Assistant's own `GET /api/config`, then the process's own
# zone, in that order. `None` -- the default, and every module-load
# before `lifespan` has run a single time, and the "process" source's own
# outcome -- means the process's own local zone, resolved fresh on every
# call by `_current_moment()` below rather than baked in once at import
# time. `app.state.timezone_resolution` carries the full result
# (including *why* it resolved the way it did); Phase 9 (calendar and
# mail) reads that, never this module global, and nothing past `lifespan`
# resolves the zone a second time.
_resolved_timezone: "ZoneInfo | None" = None


def _current_moment() -> datetime:
    """The instant `_state_message` describes, in `_resolved_timezone`.

    `datetime.now()` with no argument returns a naive local-time value;
    `.astimezone(tz)` treats a naive `self` as already being in the
    system's own zone and converts it to an aware one in `tz` -- and when
    `tz` is `None`, "convert to `tz`" means "attach the system's own zone",
    per `datetime.astimezone`'s own documented behavior. That is exactly
    why this one call handles both cases without a branch: `_resolved_timezone`
    unset resolves to the process zone, and a configured `ZoneInfo`
    resolves to that zone, through the same expression.
    """
    return datetime.now().astimezone(_resolved_timezone)


def _resolved_timezone_name() -> str:
    """The zone name `_state_message` speaks and `lifespan` logs at
    startup -- the configured IANA key, or the system's own abbreviated
    name (e.g. "UTC", "PST") when no `server.timezone` was set.
    """
    if _resolved_timezone is not None:
        return str(_resolved_timezone)
    return _current_moment().tzname() or "the local zone"


def _catalog_prompt(entities: list[dict[str, Any]], tool_ownership_prompt: str = "") -> str:
    """Byte-identical across every turn -- the cacheable prefix.

    Carries only each known entity's id and friendly name, never a state
    value: a single volatile value here would invalidate
    `brain.cache_system_prompt`'s cached prefix on every turn any entity's
    state changed, defeating the whole split D-14 exists to make. Live
    state lives in `_state_message` instead, rebuilt every turn.

    `tool_ownership_prompt` (plan 06-04, Task 3, D-10) is
    `PluginManager.tool_ownership_prompt` -- one line per tool name two
    plugins both publish, naming which plugin owns which prefixed tool.
    Defaults to `""`, which appends nothing at all: every caller that
    predates this plan, and every deployment with no colliding plugins,
    gets a prompt byte-identical to before this parameter existed.
    """
    lines = [
        "You control a home over voice through the tools you are given. "
        "Never invent an entity id that is not listed below.",
        "The request is transcribed from a low-quality narrowband microphone and "
        "often contains misheard words. When a named device is not listed, pick the "
        "listed entity whose friendly name sounds closest (\"living groom light\" "
        "means the living room light) and act on it; ask only if two entities are "
        "equally close.",
        "Known entities:",
    ]
    for entity in entities:
        lines.append(f"- {entity['entity_id']} ({entity['friendly_name']})")
    if tool_ownership_prompt:
        lines.append("")
        lines.append(
            "Some tool names below are shared by two plugins and are offered to you "
            "prefixed with their owning plugin, so you can tell them apart:"
        )
        lines.append(tool_ownership_prompt)
    return "\n".join(lines)


# 260924-4iv (items a, b): the two lines `_state_message` speaks instead of
# an empty list/sequence when a prefetched read missed its deadline
# (`brain.state_timeout_ms`) or raised (T-4iv-02). An empty `states`/
# `pending_runs` is a different, legitimate claim -- "read successfully,
# and there is nothing there" -- and must never be confused with "could not
# be read this turn." The catalog prompt (`_catalog_prompt`) still names
# every real entity id regardless of either line, so the brain is never
# taught to invent one; the MCP child's own `allow_call` policy still runs
# on every `call_service` this turn makes, so a turn with no live state
# still cannot reach an entity the operator marked off limits (T-4iv-02).
_STATE_UNAVAILABLE_LINE = (
    "Current state: not available this turn. Do not assume any entity's "
    "current state. Read an entity with a tool before you answer about its "
    "state or act on a command that depends on it."
)
_PENDING_RUNS_UNAVAILABLE_LINE = (
    "Scheduled runs: not available this turn. You cannot see which runs "
    "are scheduled, so do not tell the user that there are none."
)


def _state_message(
    states: dict[str, str] | None,
    pending_runs: tuple[Any, ...] | None = (),
    *,
    domains: frozenset[str] | None = None,
) -> str:
    """Rebuilt every turn -- deliberately not part of the cached prefix.

    Carries the current local date, day of the week, time to the minute,
    and resolved timezone (D-01, CMD-03) ahead of the entity states, read
    fresh through `_current_moment()`/`_resolved_timezone_name()` on every
    call -- two calls a minute apart describe two different minutes. This
    is what lets a spoken time or date question be answered with no tool
    call of its own: the answer is already in context before the model
    speaks, and a `get_time` tool would fail CMD-03 by definition.

    An empty `states` mapping still renders the header with no entity
    lines: that is a message the model can read as "nothing is known,"
    which is a different claim than no message at all reaching it.

    260924-4iv (item a): `states=None` -- distinct from `{}` -- means the
    live-state read did not complete this turn (a timeout or a raise in
    `turn/controller.py::run_turn`, never a choice this function makes
    itself). It renders `_STATE_UNAVAILABLE_LINE` in place of both the
    "Current state:" header and every entity line; the date/time lines
    above it are unaffected, since those never depended on the fetch.
    Degraded behavior, stated plainly: the model is never handed an empty
    list that could be misread as "every device is off" -- it is told
    plainly that it does not know, and to read before it answers about a
    state or acts on a command that depends on one. `pending_runs=None`
    is `states=None`'s own sibling, rendering `_PENDING_RUNS_UNAVAILABLE_LINE`
    in place of `summarise_pending_runs(...)`'s block for the identical
    reason -- the model must never be told "nothing is scheduled" when the
    truth is "the read did not finish."

    260924-4iv (item b): `domains`, when given, keeps only entities whose
    id carries a domain (the prefix before the first `.`) in the set --
    the header names which domains are listed and tells the model to read
    any other entity with a tool before answering about its state.
    `domains=None` (the default) keeps every entity, byte-identical to
    this function's behavior before this parameter existed. The filter
    only narrows what this one volatile message shows: `_catalog_prompt`
    (the cacheable prefix) still names every entity id and friendly name
    regardless, so nothing here hides an entity from the model's
    knowledge that it exists, only from this turn's live-state block.

    `pending_runs` (plan 05-05 Task 2, D-09) defaults to `()` -- every
    caller that predates this plan keeps producing byte-identical output.
    Given a non-empty sequence of `WorkflowRun`s (`WorkflowRepository.
    list_runs(statuses=("pending", "firing"))`'s own return shape), the
    block `workflow.summary.summarise_pending_runs` renders joins the
    entity states here, in the same volatile message -- never the
    cacheable `_catalog_prompt`, for the identical reason live entity
    state never lives there either.
    """
    now = _current_moment()
    lines = [
        f"Current date: {now:%A, %B %d, %Y}",
        f"Current time: {now:%H:%M} {_resolved_timezone_name()}",
    ]
    if states is None:
        lines.append(_STATE_UNAVAILABLE_LINE)
    elif domains is None:
        lines.append("Current state:")
        for entity_id, state in states.items():
            lines.append(f"- {entity_id}: {state}")
    else:
        sorted_domains = ", ".join(sorted(domains))
        lines.append(
            f"Current state (only these domains are listed: {sorted_domains}. "
            "Read any other entity with a tool before you answer about its state):"
        )
        for entity_id, state in states.items():
            if entity_id.split(".", 1)[0] in domains:
                lines.append(f"- {entity_id}: {state}")
    if pending_runs is None:
        lines.append(_PENDING_RUNS_UNAVAILABLE_LINE)
    else:
        lines.append(summarise_pending_runs(pending_runs, now))
    return "\n".join(lines)


def _make_state_fetch(plugin_manager: "PluginManager") -> Callable[[], Any]:
    """Build the per-turn `state_fetch` factory `run_turn` awaits
    concurrently with the operator still speaking (D-15).

    This is a read, gated by `allow_read` inside the MCP child -- a denied
    entity's state is still returned and injected into the prompt. SAFE-02
    is deliberate here, not an oversight: a question about a denied entity
    is exactly the case that must keep working. Reuses `_tool_result_json`
    rather than re-implementing the MCP payload walk a second time.

    Plan 06-01: the enforcing host is `None` when the plugin that enforces
    the house policy (`PluginManager.enforcing_host`) is disabled or
    failed to start -- an absent host returns an empty entity list rather
    than raising an attribute error, the same "unknown reads as nothing
    known" posture `_state_message` already gives an empty `states`
    mapping.

    WR-01 (code review): the host is resolved from the manager here, at
    call time, rather than being snapshotted when this factory is built.
    Every crash-driven respawn, enable and configuration save builds a
    *new* `McpToolHost` and closes the previous one, so a factory holding
    the host it was built with talks to a closed one from the first such
    change onward -- every later turn then loses live entity state with no
    operator-visible signal, and a Home Assistant that was degraded at
    boot never came back at all. The resolution is still exactly one read
    per turn, which is what keeps a single turn's own view stable.
    """

    async def _fetch() -> list[dict[str, Any]]:
        tool_host = plugin_manager.enforcing_host
        if tool_host is None:
            return []
        result = await tool_host.call_tool("ha_list_entities", {})
        entities = _tool_result_json(result)
        return entities if isinstance(entities, list) else []

    return _fetch


def _make_pending_runs_fetch(workflow_repo: WorkflowRepository) -> Callable[[], Any]:
    """Build the per-turn `pending_runs_fetch` factory `run_turn` awaits
    concurrently with the operator still speaking, the identical shape
    `_make_state_fetch` above already establishes for live entity state
    (plan 05-05 Task 2, D-09, plan 01.1-06's mechanism extended).

    Reads `("pending", "firing")` runs -- D-16's own list, the same one
    the webapp's pending-run screen reads -- and returns them as-is for
    `_state_message` to render through `summarise_pending_runs`. A raised
    fetch is this function's caller's problem to handle (`run_turn`'s own
    "logged and treated as nothing known" rule, identical to
    `state_fetch`'s), not this factory's: it performs no try/except of its
    own, the same bare-call shape `_make_state_fetch._fetch` already
    uses.
    """

    async def _fetch() -> "tuple[Any, ...]":
        runs = await workflow_repo.list_runs(statuses=("pending", "firing"))
        return tuple(runs)

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
        # The MCP SDK wraps a tool's non-object return value (a list, for
        # `ha_list_entities`) as `{"result": value}`. Without this unwrap
        # every caller that expects a list got a dict, read it as "no
        # entities", and the brain saw an empty house.
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
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
        "has no trained 'hey atlas' model and no detector class in this "
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


def _build_tapo_talk_supervisor(config: Config) -> TapoTalkSupervisor:
    """Build the `tapo_talk` egress supervisor -- the same monkeypatchable
    shape `_build_ffmpeg_supervisor` uses, for the same reason: a fake
    substituted here needs neither pytapo nor a reachable camera to reach a
    running application. The camera host is derived from
    `config.camera.rtsp_url` once, here -- `TapoTalkSupervisor` itself never
    reads `Config`, only the resolved `SpeakerConfig` and this host string
    (T-vqa-02)."""
    host = camera_host_from_rtsp_url(config.camera.rtsp_url)
    return TapoTalkSupervisor(config.speaker, host)


def _build_tcp_supervisor(config: Config) -> FfmpegSupervisor:
    """Build the `tcp` egress supervisor (260923-pds) -- the existing
    `FfmpegSupervisor` wired to `build_tcp_argv`, with no `http_client`.
    `ensure_url` is a go2rtc-only PUT that holds the camera password;
    passing no client means `_ensure_backchannel` returns at once, so that
    PUT is never sent on this path. A separate, monkeypatchable function so
    tests can substitute a fake, the same reason its siblings
    (`_build_ffmpeg_supervisor`, `_build_tapo_talk_supervisor`) are also
    separate functions.

    260923-pyj: the FIFO carries the `tts.codec`/`tts.sample_rate` pair --
    the same pair `lifespan` builds `camera_tts_sink` from, below -- so the
    egress argv must read the FIFO in that format, not a fixed one.
    `functools.partial` binds the sink into `build_tcp_argv` here, so
    `FfmpegSupervisor._supervise`'s own one-argument `build_argv(config)`
    call stays unchanged.
    """
    sink = SinkFormat(codec=config.tts.codec, sample_rate=config.tts.sample_rate)
    return FfmpegSupervisor(config.speaker, build_argv=functools.partial(build_tcp_argv, sink=sink))


def _build_repositories(config: Config, engine: AsyncEngine) -> dict[str, Any]:
    """Build every repository this process reads and writes through, keyed
    by the `app.state` attribute name each belongs under.

    A separate, monkeypatchable function -- the same shape
    `_build_wake_detector`/`_build_ffmpeg_supervisor` already use --
    specifically so `tests/test_startup_smoke.py` can substitute a dict of
    fakes (`FakePolicyRepository`/`FakeAccountRepository` from
    `conftest.py`) without a reachable Postgres. Returns a dict rather than
    a single object because a later plan in this phase (credentials) adds
    one more repository here without this factory's call site changing
    shape.
    """
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    return {
        "policy_repo": PostgresPolicyRepository(sessionmaker),
        "account_repo": PostgresAccountRepository(sessionmaker),
        "credential_repo": PostgresCredentialRepository(sessionmaker),
        "setup_repo": PostgresSetupRepository(sessionmaker),
        "settings_repo": PostgresSettingsRepository(sessionmaker),
        # Plan 04-05 (D-09): macros live here now, not on `Config`.
        "macro_repo": PostgresMacroRepository(sessionmaker),
        # Plan 06-01 (D-01, D-04): plugins live here now -- the table
        # `PluginManager` reads instead of `Config.mcp_servers`.
        "plugin_repo": PostgresPluginRepository(sessionmaker),
        # Plan 07-01 (D-01, D-03): which named provider each slot is
        # currently pointed at -- the table `resolve_slot` reads instead
        # of a hardcoded provider-class construction.
        "provider_selection_repo": PostgresProviderSelectionRepository(sessionmaker),
        # Plan 08-03 (D-13, D-14): every wake hit the detector reports,
        # allowed or blocked -- the persistent record the wake-tuning
        # screen (DBG-05) re-partitions arithmetic against instead of
        # re-running an engine over recorded audio.
        "wake_event_repo": PostgresWakeEventRepository(sessionmaker),
        # Plan 05-01: scheduled workflow runs and steps (D-01 .. D-04).
        "workflow_repo": PostgresWorkflowRepository(
            sessionmaker,
            max_attempts=config.workflow.max_attempts,
            retry_backoff_s=config.workflow.retry_backoff_s,
            claim_recovery_after_s=config.workflow.claim_recovery_after_s,
        ),
        # Phase 9 (09-01, D-01, D-03): linked Google accounts, their
        # calendars, and the one OAuth client -- read with `.get(...)`,
        # like `wake_event_repo`, since `tests/test_startup_smoke.py`'s
        # fake repository dict predates this key (Task 3's own "boots
        # unchanged with no Google repository" requirement).
        "google_account_repo": PostgresGoogleAccountRepository(sessionmaker),
        # Plan 09-04 (Task 3): where a spoken calendar write proposal is
        # stored until the operator confirms it (D-08, D-09) -- read with
        # `.get(...)`, like `google_account_repo` above, so an app (or a
        # test's own fake repository dict) with no Google repository at all
        # boots unchanged; `atlas.google.turn_context.build_handoff_context`
        # reads this same key with an identical tolerant `getattr`.
        "pending_action_repo": PostgresPendingActionRepository(sessionmaker),
        # Plan 10-04 (D-03): the edge device token store -- read with
        # `.get(...)` at the call site below, the same tolerant lookup
        # `wake_event_repo`/`google_account_repo` already use, so a test's
        # own fake repository dict that predates this key boots unchanged.
        "edge_device_repo": PostgresEdgeDeviceRepository(sessionmaker),
    }


async def _resolve_and_log_credential(
    slot: CredentialSlot, config: Config, credential_repo: CredentialRepository
) -> str:
    """`resolve_credential_value`, plus the one required log line: which
    source won, by slot name only, never by value (T-03-43). The one
    place `lifespan` asks "what is this credential, really" -- see
    `atlas.crypto.credentials`'s own module docstring for why the
    database-vs-environment decision itself lives there, not here."""
    value, source = await resolve_credential_value(slot, config, credential_repo)
    logger.info("credential slot %s resolved from %s", slot.value, source)
    return value


async def _resolve_wake_threshold(settings_repo: SettingsRepository) -> "tuple[float | None, str]":
    """`(threshold, resolved_from)` for the wake threshold, mirroring
    `resolve_audio_source`'s own database-then-configuration precedence
    (`routes/wizard.py`) rather than re-deriving a second resolution
    order. `None`/`"config"` means nothing has ever been stored through
    `PUT /api/wake-threshold` -- `SourceRunner.__init__`'s own
    configuration-resolved threshold is left exactly as it is. A stored
    value/`"database"` means an operator has moved it, and `lifespan`
    applies it after construction (D-15)."""
    setting = await settings_repo.get_setting(WAKE_THRESHOLD_SETTING_KEY)
    if setting is not None and isinstance(setting.value, (int, float)) and not isinstance(setting.value, bool):
        return float(setting.value), "database"
    return None, "config"


async def _resolve_follow_up_window_s(
    settings_repo: SettingsRepository, config_default: float
) -> "tuple[float, str]":
    """`(window_s, resolved_from)` for the follow-up window's own length
    (`FOLLOW_UP_WINDOW_SETTING_KEY`), mirroring `_resolve_wake_threshold`'s
    own database-then-configuration precedence immediately above. Unlike
    the wake threshold, a stored value outside `PUT /api/settings/
    follow-up-window`'s own 3-15 second range never refuses the boot
    (T-09-41): that route already enforces this exact range on the way
    in, so an out-of-range row can only have arrived some other way, and
    a boot must never refuse over a setting a route already validated --
    logged and the configured default used instead, the same "a boot
    never refuses over a setting the route already validated" doctrine
    `routes/follow_up_settings.py`'s own module docstring states.
    """
    setting = await settings_repo.get_setting(FOLLOW_UP_WINDOW_SETTING_KEY)
    if setting is not None and isinstance(setting.value, (int, float)) and not isinstance(setting.value, bool):
        value = float(setting.value)
        if 3.0 <= value <= 15.0:
            return value, "database"
        logger.warning(
            "stored follow-up window %r is outside [3, 15] -- falling back to the "
            "configured value %r",
            value,
            config_default,
        )
    return config_default, "config"


def _resolve_edge_barge_in_config(config: Config) -> BargeInConfig:
    """The edge source's own resolved barge-in policy (D-16), never the
    bare global `config.barge_in` handed to the camera runner unchanged.

    An operator's own `barge_in.sources.edge.enabled`
    (`config/config.example.yaml` ships one, `false`, from 10-SPIKE.md's
    `aec: not_proven`) wins exactly as `BargeInConfig.resolve` would
    apply it -- this function changes nothing about `config.barge_in`
    when an `edge` override already exists. Only a configuration that
    names no `edge` override at all falls back to `EDGE_BARGE_IN_PROVEN`,
    added here as a synthetic override rather than assumed by
    `SourceRunner` itself: the camera-era global `barge_in.enabled`
    (`true` by default) must never reach this source by accident the way
    `SourceRunner.__init__`'s own absent-configuration fallback
    (`BargeInConfig(enabled=False)`) already keeps it from doing for a
    runner built with no `barge_in_config` at all -- this is that same
    guarantee, applied to a runner that now always gets one.
    """
    if "edge" in config.barge_in.sources:
        return config.barge_in
    return replace(
        config.barge_in,
        sources={**config.barge_in.sources, "edge": {"enabled": EDGE_BARGE_IN_PROVEN}},
    )


async def _open_speaker_writer(writer: FifoWriter) -> None:
    """Open `writer` in the background, never inline in `lifespan`.

    Opening a FIFO for writing blocks until a reader attaches
    (`fifo_writer.py`'s own module docstring) -- doing this inline would
    hang the whole application before the egress supervisor's `ffmpeg`
    child (plan 02-03 Task 2) ever gets a chance to attach as that reader.
    A failure here (the mount not existing yet, for instance) is logged and
    leaves the writer unopened rather than crashing startup. This initial
    open is the only one with no timeout (module docstring); a later
    reopen after a reader loss is bounded by `speaker.reopen_timeout_s`
    and, if it still fails, raises `SpeakerError` out of `writer.write()`
    instead -- caught and logged by `sources/runner.py`'s own per-chunk
    containment (CR-03), not here.
    """
    try:
        await writer.open()
    except SpeakerError:
        logger.warning("speaker FIFO not yet available at startup", exc_info=True)


async def _current_macros(app: FastAPI) -> tuple[Any, ...]:
    """The macros the database holds right now, read fresh -- never a
    startup snapshot (D-12, T-04-28, plan 04-05's own must-have: "every
    turn reads the macros the database currently holds").

    A per-turn repository read, not `app.state`'s own cached
    `app.state.macros` (set once at boot below, kept only for the wiring
    check `tests/test_startup_smoke.py`'s `_EXPECTED_STATE_ATTRS` already
    runs and for the startup precache, which necessarily can only precache
    what exists at boot). Chosen over a cached app-state value plan 04-06's
    write route would replace, because a macro table read is small (this
    project's whole macro set, not a per-row query) and this is the
    simpler contract to keep correct: a cached value needs the write route
    to remember to refresh it on every save, forever; a live read needs
    nothing from that route at all. The cost is one extra database round
    trip per turn on the one path whose entire reason to exist is speed --
    accepted deliberately, because it lands before `match()` decides
    whether this turn is a macro turn at all, and `MacroOutcome.cacheable`
    still means the macro's own *reply* never pays a live TTS call, which
    is the latency property macros actually exist to protect. See this
    plan's own SUMMARY for the full tradeoff this docstring compresses.

    `Macro` (`atlas.db.repository`) is duck-type compatible with
    `atlas.config.MacroConfig` -- `match()`/`fire_macro()`
    (`turn/macros.py`) need no change to accept either.
    """
    macro_repo: MacroRepository = app.state.macro_repo
    return tuple(await macro_repo.list_macros())


async def _refuse_turn_if_any_slot_is_degraded(app: FastAPI, source: Any) -> bool:
    """CR-03 (code review). Answer `True` -- having already told `source`
    why -- when a degraded provider slot (D-04) makes this turn
    impossible; `False` when the turn may run.

    Every path in this file that starts a turn asks this first. Before
    this phase, `app.state.stt`/`app.state.tts` were always a constructed
    client, so no consumer needed a `None` branch and none had one; D-04
    made `None` a designed state without teaching anybody downstream.
    `run_turn` is where the crash landed, but `run_turn` is the wrong
    place to fix it: its `stt`/`tts` parameters are typed non-optional
    precisely because a turn is not a thing you can run half of. The
    refusal belongs at the boundary, where the slot statuses live.

    The reason is emitted as a `reply.text` event and nothing is spoken:
    on the deployment this actually happens to, the text-to-speech slot is
    degraded too, so there is no voice to say it with.
    """
    refusal = degraded_turn_refusal(getattr(app.state, "provider_slots", None))
    if refusal is None:
        return False
    send_event = getattr(source, "send_event", None)
    if send_event is not None:
        await send_event({"type": "reply.text", "text": refusal})
    return True


def _warm_providers(providers: Iterable[Any], keep_alive: set[asyncio.Task]) -> list[asyncio.Task]:
    """Start a background `warm()` call on every provider in `providers`
    that has one, and return the tasks started (260924-4iv, item d).

    `None` entries (a degraded slot) and a provider with no `warm`
    attribute (`getattr(..., "warm", None)` -- Piper has none) are skipped
    silently, never raising: this function's whole job is to make a wake
    faster when it can, never to decide which providers exist. Each task
    is added to `keep_alive` (the same `app.state.background_turns` set
    every other detached task in this file already uses) with
    `add_done_callback(keep_alive.discard)`, so asyncio never garbage-
    collects a still-running warm call -- and a small wrapper coroutine
    swallows any exception `warm()` itself did not, so a third-party
    provider whose `warm()` raises can never reach a turn or log an
    unretrieved-exception warning.
    """

    async def _warm_one(provider: Any) -> None:
        try:
            await provider.warm()
        except Exception:
            logger.debug("provider warm() raised for %r", provider, exc_info=True)

    tasks: list[asyncio.Task] = []
    for provider in providers:
        if provider is None:
            continue
        warm = getattr(provider, "warm", None)
        if warm is None:
            continue
        task = asyncio.create_task(_warm_one(provider))
        keep_alive.add(task)
        task.add_done_callback(keep_alive.discard)
        tasks.append(task)
    return tasks


def _make_run_turn_for_source(
    app: FastAPI, config: Config, source_name: str, *, room_speaker: bool = True
) -> Callable[[Any], Any]:
    """Build the one-argument `run_turn` caller `SourceRunner` needs.

    A fresh `TurnTimings()` per call -- never shared across turns, the same
    per-turn lifetime the WebSocket/WebRTC routes below already give it --
    is why this returns a closure rather than a bound partial over a single
    `TurnTimings` instance.

    WEB-06/D-08: `source_name` is published on this turn's own
    `turn.started` observer event and is what every event this turn emits
    is labelled with -- `ObserverPublishingSource` is constructed fresh per
    turn, here, which is what makes the turn boundary explicit rather than
    inferred by an observer from the first transcript that happens to
    arrive.

    `room_speaker=False` is for a source that plays its reply somewhere
    other than the room speaker (the browser listener): its turns must not
    wait on `app.state.speaker_lock` behind a camera reply.
    """

    # CR-03: one warning per process for the camera path, not one per wake
    # word. A degraded deployment can be woken all day; the first line
    # carries the whole reason, and the rest are debug, so the operator
    # gets the fact once instead of a log they stop reading.
    refusal_logged = False

    async def _run(source: Any) -> None:
        nonlocal refusal_logged
        if await _refuse_turn_if_any_slot_is_degraded(app, source):
            if refusal_logged:
                logger.debug("refused a wake-word turn: a provider slot is degraded")
            else:
                refusal_logged = True
                logger.warning(
                    "refusing every turn until a restart: %s",
                    degraded_turn_refusal(getattr(app.state, "provider_slots", None)),
                )
            return

        # 260924-4iv (item d): warm the chat and TTS pools in the
        # background, right at wake -- in parallel with the wake cue, the
        # STT socket opening, and the operator still speaking, so a turn
        # minutes after the last one skips the TLS handshake on both. This
        # is the wake path only (the camera source and the /ws/listen
        # browser listener, both built through this same closure); the
        # browser push-to-talk and WebRTC routes are not wake paths and are
        # left unchanged.
        _warm_providers((*(tier.brain for tier in app.state.tier_brains), app.state.tts), app.state.background_turns)

        timings = TurnTimings()
        session_recorder = SessionRecorder(config.session, timings)
        # Published before `run_turn` is ever called, from this turn's own
        # freshly-constructed `TurnTimings`, so an observer's feed opens a
        # new card before the first transcript partial can possibly arrive
        # (08-UI-SPEC.md finding 1's `turn.started {source, turn_id}`
        # boundary message).
        #
        # `session_id` (Task 3 deviation, Rule 2): `routes/sessions.py`'s
        # detail route is keyed on the session *directory* name -- a
        # timestamp plus this same turn_id, never the bare turn_id alone
        # (`session/recorder.py::_directory_name`). Without it, `/live`'s
        # own "View full session ->" link (08-UI-SPEC.md Copywriting
        # Contract) would point at an id `routes/sessions.py` can never
        # resolve. `session_recorder` is already constructed above, so its
        # real directory name costs nothing extra to read here.
        app.state.observer_registry.publish(
            {
                "type": "turn.started",
                "source": source_name,
                "turn_id": timings.turn_id,
                "session_id": session_recorder.directory.name,
            }
        )
        source = ObserverPublishingSource(source, source_name, app.state.observer_registry)
        await run_turn(
            source,
            app.state.stt,
            app.state.brain,
            app.state.tts,
            app.state.tool_host_lookup,
            app.state.tools_schema,
            app.state.catalog_prompt,
            config.brain.max_tool_rounds,
            timings,
            config.stt.max_utterance_s,
            tiers=app.state.tier_brains,
            filler_after_ms=config.brain.filler_after_ms,
            filler_cache=app.state.filler_caches,
            macros=await _current_macros(app),
            state_fetch=_make_state_fetch(app.state.plugin_manager),
            pending_runs_fetch=_make_pending_runs_fetch(app.state.workflow_repo),
            session_recorder=session_recorder,
            speech_lock=app.state.speaker_lock if room_speaker else None,
            workflow_tool_host=app.state.workflow_tool_host,
            tool_owners=app.state.plugin_manager.owners_of_bare_name,
            # 260922-woc: only the camera's wake-word turn ever pauses after
            # a spoken wake phrase -- the browser and WebRTC routes below
            # pass no `wake_phrase` at all, which is `run_turn`'s own signal
            # to skip the wake-only check entirely.
            wake_phrase=config.wake.phrase,
            wake_cue=config.wake.cue,
            brain_turn_timeout_s=config.brain.turn_timeout_s,
            # 260922-lim: only the camera's wake-word turn skips the tier
            # race for a plain on/off command -- the browser and WebRTC
            # routes below pass nothing, which is `run_turn`'s own
            # `local_intents=False` default.
            local_intents=config.brain.local_intents,
            # 260924-4iv (items a, b): every `run_turn` call site threads
            # the same two configured values -- the shared prefetched-read
            # deadline and the state-message domain filter.
            state_timeout_ms=config.brain.state_timeout_ms,
            state_domains=config.brain.state_domains,
            # Plan 09-04 (Task 3): built fresh for this turn, at this call
            # site, so it carries `app.state`'s current tool host lookup
            # and pending-action repository (D-08).
            handoff_context=build_handoff_context(app, source_name),
        )

    return _run


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Two-stage load (CR-01 fix, generalized plan 04-05, D-16). `load_config`
    # -- a single, unconditional `Config.from_config` call -- cannot be used
    # here: it rejects a lingering legacy key immediately, before the seed
    # migration that is supposed to carry that same key's data into the
    # database ever runs, which strands a real upgrade with no way forward
    # (the key cannot be removed without discarding the data it names, and
    # it cannot be kept without the process refusing to boot). The raw dict
    # is read with no validation instead, so a `DatabaseConfig` can be built
    # and migrations can run regardless of whether any legacy key is
    # present; whether to reject each one is decided further down, against
    # real Alembic revision history, once migrations have had their one
    # chance to seed it. `_LEGACY_CONFIG_KEYS` below is what makes this one
    # sequence handle a second retired key without a second copy of it: a
    # second `if` chain beside the first is precisely the shape this
    # generalization exists to prevent.
    raw_config = load_raw_config(CONFIG_PATH)
    legacy_keys_present = [key for key in _LEGACY_CONFIG_KEYS if key.name in raw_config]
    database_config = DatabaseConfig.from_config(raw_config.get("database"))
    security_config = SecurityConfig.from_config(raw_config.get("security"))

    # Refuse to start under a missing, malformed, or placeholder secret --
    # before anything else, since a running process that hashes no
    # passwords of its own but signs every session token and will encrypt
    # every provider credential under this key does not belong behind a
    # warning nobody reads (03-05 Task 1's checkpoint, CD-3). This is a
    # pure environment check with no I/O dependency, which is why it runs
    # even before the migration step below.
    validate_secret_key_strength(security_config)

    # `web/dist` is gitignored (it is a build artifact, `tests/
    # test_web_build.py`'s own first assertion), so a clean clone that has
    # not run `bun run build` has no directory here yet. Unlike the two
    # failures this project does refuse to start on (an unmigrated schema,
    # an uncalibrated correlation gate below) -- both of which would
    # otherwise enforce something wrongly -- a missing static bundle is not
    # one of them: a backend developer running only the API must still be
    # able to start the process. It must not be silent either, because "the
    # page is blank" is a terrible way to learn the frontend was never
    # built, so this warns by name rather than staying quiet.
    if not FRONTEND_DIR.is_dir():
        logger.warning(
            "frontend build directory %s does not exist -- the built admin "
            "webapp will not be served (every other route still works). "
            "Run `cd web && bun install && bun run build` to build it.",
            FRONTEND_DIR,
        )

    # CR-01 fix, generalized: captured before migrations run, and only when
    # at least one legacy key is present to classify -- `None` here means
    # this database has never had a migration applied, which means the
    # migration below (if it runs) is about to seed whatever legacy key(s)
    # are present for the first time ever. That is the one signal that
    # tells a first boot (a present key should be tolerated, logged, and
    # left for the operator to remove) apart from a later one (a previous
    # boot already seeded it, so the same key now means two sources of
    # truth and must be rejected) -- `Config.from_config` alone cannot draw
    # this distinction without a database connection it does not have.
    # Captured once, not once per key: every migration in this project
    # shares one Alembic revision history, so one pre-migration read
    # answers "was this database ever migrated before" for every legacy
    # key at once.
    revision_before_migration = (
        get_current_revision(database_config.migration_url) if legacy_keys_present else None
    )

    # Migrations run first, before any other resource is built, and awaited
    # in sequence -- never scheduled as a detached task, which is the exact
    # shape 03-RESEARCH.md Pitfall 1 names as the way this fails silently
    # (a migration that never actually ran, with no error). A failure here
    # propagates uncaught and stops the process, the same refusal-beats-
    # half-configured posture the barge-in calibration check below already
    # established for this file (D-02, DEP-04).
    if database_config.run_migrations_at_startup:
        await asyncio.to_thread(run_migrations, database_config.migration_url)

    if legacy_keys_present:
        # Only a boot that both actually ran migrations just now *and*
        # found no prior revision stamped counts as "just seeded it" --
        # anything else (an operator running migrations out of band with
        # `run_migrations_at_startup: false`, or a second boot against an
        # already-migrated database) means whatever this key names was
        # already carried into the database by an earlier boot, so the key
        # is a stale, conflicting second source of truth and must be
        # rejected with the same wording this project has always used for
        # it. One classification rule, applied to every present key in
        # turn -- not a second `if` chain for the second key.
        is_first_seed_boot = (
            database_config.run_migrations_at_startup and revision_before_migration is None
        )
        for legacy_key in legacy_keys_present:
            if is_first_seed_boot:
                logger.warning(
                    "%s: block found in %s, and its seed migration just ran "
                    "for the first time against this database -- it now "
                    "lives in the database (check the %s for your seeded "
                    "entries). Delete the %s: block from this file now: the "
                    "next boot refuses to start while it is still present.",
                    legacy_key.name,
                    CONFIG_PATH,
                    legacy_key.seeded_tables_hint,
                    legacy_key.name,
                )
            else:
                raise ConfigError(legacy_key.rejected_error)

    config = Config.from_config(
        raw_config,
        reject_legacy_safety_key=False,
        reject_legacy_macros_key=False,
        reject_legacy_mcp_key=False,
    )
    app.state.config = config

    # WEB-06/D-05: the observer fan-out, set once at boot beside the other
    # long-lived objects. Needs nothing else on `app.state` to exist --
    # `ObserverRegistry` has no dependency on config, the database, or any
    # provider -- so it is safe to construct this early, before either turn
    # path that publishes to it (the camera runner, further down, and
    # `/ws/turn` below) is wired up.
    app.state.observer_registry = ObserverRegistry()

    # IN-01 (STATE.md, deferred here from Phase 3's code review): a
    # deployment that would send its session cookie without the Secure
    # flag over a bind a browser somewhere on the network can reach says
    # so, by name, every time -- a warning, never a refusal (T-07-36): a
    # deployment behind a reverse proxy that terminates TLS is correct and
    # common, and this process cannot see the proxy. A loopback development
    # boot (this project's own shipped default) stays silent -- warning on
    # it every run would train an operator to stop reading the message.
    if not config.security.cookie_secure and not _bind_is_loopback(config.server.bind_host):
        logger.warning(
            "security.cookie_secure is false and server.bind_host (%s) is "
            "not loopback -- the session cookie will be sent without the "
            "Secure flag over a network a browser can reach it on. If a "
            "reverse proxy terminates TLS in front of this deployment, set "
            "security.cookie_secure: true; otherwise, keep server.bind_host "
            "on a loopback address (127.0.0.1).",
            config.server.bind_host,
        )

    db_engine = build_engine(config.database)
    app.state.db_engine = db_engine
    repositories = _build_repositories(config, db_engine)
    for _repo_name, _repo in repositories.items():
        setattr(app.state, _repo_name, _repo)
    policy_repo = repositories["policy_repo"]
    credential_repo = repositories["credential_repo"]
    settings_repo: SettingsRepository = repositories["settings_repo"]
    macro_repo: MacroRepository = repositories["macro_repo"]
    plugin_repo: PluginRepository = repositories["plugin_repo"]
    workflow_repo: WorkflowRepository = repositories["workflow_repo"]
    provider_selection_repo: ProviderSelectionRepository = repositories["provider_selection_repo"]
    # Plan 08-03: `.get(...)`, not `[...]` -- `SourceRunner`'s own
    # `wake_event_repo` parameter is optional (defaults to `None`), and
    # `tests/test_startup_smoke.py`'s existing fake-repository dict
    # predates this key. A `SourceRunner` built with no repository at all
    # behaves exactly as it did before this plan (same absent-
    # configuration discipline the gate and barge-in policy already carry
    # in that constructor).
    wake_event_repo: WakeEventRepository | None = repositories.get("wake_event_repo")
    # Phase 10 (D-03): same tolerant `.get(...)` -- a deployment (or a
    # test's own fake repository dict) with no edge device repository at
    # all boots exactly as it did before this plan. Plan 10-04 adds the
    # Postgres implementation; until then this is always `None` outside a
    # test that seeds one.
    app.state.edge_device_repo = repositories.get("edge_device_repo")
    # Set now, unconditionally, so `/ws/edge` always finds a defined
    # attribute -- overwritten below only when `resolved_audio_source ==
    # EDGE_SOURCE_NAME`. `None` here is what makes the route's own 4003
    # refusal ("edge is not the configured audio source") correct for
    # every other value of `resolved_audio_source`.
    app.state.edge_source = None
    # Phase 9 (09-01, Task 3): same tolerant `.get(...)` -- a deployment
    # (or a test's own fake repository dict) with no Google repository at
    # all boots exactly as it did before this plan; nothing below runs.
    google_account_repo = repositories.get("google_account_repo")

    # The wizard's own audio-source choice (`routes/wizard.py`'s
    # `AUDIO_SOURCE_SETTING_KEY`) joins the same resolution discipline the
    # credential slots above already follow: the database wins when an
    # operator has chosen one through the wizard, this file's own shipped
    # default otherwise -- one function decides, `resolve_audio_source`,
    # never re-derived here. Only `"camera"` is actually built below today
    # (the sole `SourceRunner` this application constructs), so this call
    # currently only proves the resolution order and logs which source
    # won, by name -- the same "log the source, never the value" posture
    # `_resolve_and_log_credential` already established, extended to a
    # setting that carries no secret at all.
    resolved_audio_source, audio_source_resolved_from = await resolve_audio_source(
        config, settings_repo
    )
    logger.info(
        "audio source resolved to %r from %s", resolved_audio_source, audio_source_resolved_from
    )
    # Plan 10-10: the one place a later reader (the precache step below,
    # and `run_echo_path_calibration`'s own D-15 refusal) asks "which
    # source is the room microphone" -- never re-derived, and never a
    # second call to `resolve_audio_source` itself.
    app.state.audio_source = resolved_audio_source

    # Plan 08-07 (D-15): the wake threshold an operator moved through
    # `PUT /api/wake-threshold` outlives the process that received it.
    # Same precedence as the audio source immediately above -- the
    # database wins when a value has ever been stored, this process's own
    # configured value otherwise -- and a stored value out of range
    # refuses the boot by name rather than silently clamping: it cannot
    # have come from the route, which already enforces this range, so it
    # came from something that bypassed it.
    resolved_wake_threshold, wake_threshold_resolved_from = await _resolve_wake_threshold(
        settings_repo
    )
    if resolved_wake_threshold is not None and not 0.0 <= resolved_wake_threshold <= 1.0:
        raise ConfigError(
            f"stored wake threshold {resolved_wake_threshold!r} is outside [0.0, 1.0] -- this "
            "cannot have come from PUT /api/wake-threshold, which enforces this exact range "
            "itself, so refusing to boot on a setting this process cannot honestly interpret"
        )
    logger.info("wake threshold resolved from %s", wake_threshold_resolved_from)

    # Plan 09-07 (D-09, T-09-41): the follow-up window's own length,
    # stored on `app.state` (a plain float, not a callable) so both
    # `SourceRunner` constructions below can read it live through their
    # own `follow_up_window_s=lambda: app.state.follow_up_window_s` --
    # `PUT /api/settings/follow-up-window` writes this same attribute
    # directly, with no restart needed for the very next window to see it.
    resolved_follow_up_window_s, follow_up_window_resolved_from = await _resolve_follow_up_window_s(
        settings_repo, config.follow_up.window_s
    )
    app.state.follow_up_window_s = resolved_follow_up_window_s
    logger.info("follow-up window resolved from %s", follow_up_window_resolved_from)

    # D-01 (phase 4), extended by the 260924-h2f quick task (issue #1):
    # resolve the house's own time zone -- the one process-wide reading
    # `_state_message`, `WorkflowToolHost`, and every stdio MCP child's
    # own `TZ` all read from -- in the one order `routes.wizard.
    # resolve_timezone`'s own docstring names: the zone saved through the
    # webapp, then `server.timezone`, then Home Assistant's own `GET
    # /api/config`, then the process's own zone. `app.state.
    # timezone_resolution` is the one place a later reader (Phase 9's
    # calendar and mail) goes; nothing resolves the zone a second time.
    # `ha_http_client` is the same test seam `check_hub_step` already
    # uses -- production opens and closes its own short-lived client,
    # bounded to 5s so an unreachable Home Assistant cannot stall the
    # boot (T-h2f-05).
    global _resolved_timezone
    injected_ha_client = getattr(app.state, "ha_http_client", None)
    owns_ha_client = injected_ha_client is None
    ha_client = injected_ha_client if injected_ha_client is not None else httpx.AsyncClient(timeout=5.0)
    try:
        timezone_resolution = await resolve_timezone(
            config, settings_repo, plugin_repo, client=ha_client
        )
    finally:
        if owns_ha_client:
            await ha_client.aclose()
    _resolved_timezone = timezone_resolution.zone
    app.state.timezone_resolution = timezone_resolution
    app.state.server_timezone = _resolved_timezone
    if timezone_resolution.warning:
        logger.warning("%s", timezone_resolution.warning)
    else:
        logger.info(
            "resolved timezone: %s (from %s)",
            timezone_resolution.name,
            timezone_resolution.resolved_from,
        )

    # Every provider credential is resolved exactly once, here, in the one
    # order D-07 states: the database wins when an operator has saved a
    # value through the webapp, the configuration file's own environment
    # expansion wins otherwise, and an unset slot resolves to an empty
    # string -- never a fresh env read, never a per-provider conditional
    # (see `atlas.crypto.credentials`'s own module docstring for why
    # the decision itself lives there, not here).
    resolved_stt_key = await _resolve_and_log_credential(
        CredentialSlot.STT, config, credential_repo
    )
    resolved_brain_key = await _resolve_and_log_credential(
        CredentialSlot.BRAIN, config, credential_repo
    )
    resolved_tts_key = await _resolve_and_log_credential(
        CredentialSlot.TTS, config, credential_repo
    )
    # WR-06 (code review): there is no fourth resolution here any more.
    # The Home Assistant token lives in `plugin_config_values` (D-01) and
    # is read by `PluginManager` at spawn and by the wizard's hub check;
    # `resolved_ha_token` was assigned here and never used by anything,
    # which is how the orphaned `ha_token` credential row went unnoticed.
    #
    # Plan 07-01 (D-01, D-02, D-03), plan 07-02: all three provider slots'
    # providers are read from `provider_selection_repo` once, here, at the
    # same point their credentials are already resolved above -- never
    # re-read per request. `resolve_slot` is generic; `provider_registry.
    # build_stt`/`build_tts`/`build_brain` are what actually know each
    # slot's name-to-factory mapping (T-07-01). A degraded slot (D-04)
    # leaves its `app.state` attribute `None` (or, for `tier_brains`, an
    # empty tuple -- see below) and the boot continues, one slot at a
    # time, independently of the other two. `app.state.provider_slots` is
    # what `routes/providers.py` reads live, never a value copied off
    # these local variables.
    stt_status, stt_client = await resolve_slot(
        "stt",
        provider_selection_repo,
        "xai",
        provider_registry.build_stt,
        config.stt,
        resolved_stt_key,
    )
    logger.info(
        "provider slot 'stt' resolved to %r (%s)", stt_status.selected, stt_status.state
    )
    if stt_status.state == "degraded":
        logger.warning("provider slot 'stt' is degraded: %s", stt_status.reason)
    app.state.stt = stt_client

    # The text-to-speech entry declares itself batch (D-05): `is_batch`
    # is how `resolve_slot` learns to wrap the built client in
    # `BatchTtsAdapter` before handing it back -- the one wrap site
    # (D-07), never a second one inside `registry.py` or `tts_xai.py`.
    tts_status, tts_client = await resolve_slot(
        "tts",
        provider_selection_repo,
        "xai",
        provider_registry.build_tts,
        config.tts,
        resolved_tts_key,
        is_batch=lambda name: provider_registry.TTS_REGISTRY[name].batch,
    )
    logger.info(
        "provider slot 'tts' resolved to %r (%s)", tts_status.selected, tts_status.state
    )
    if tts_status.state == "degraded":
        logger.warning("provider slot 'tts' is degraded: %s", tts_status.reason)
    app.state.tts = tts_client

    brain_status, tier_brains_result = await resolve_slot(
        "brain",
        provider_selection_repo,
        "xai",
        provider_registry.build_brain,
        config.brain,
        resolved_brain_key,
    )
    logger.info(
        "provider slot 'brain' resolved to %r (%s)", brain_status.selected, brain_status.state
    )
    if brain_status.state == "degraded":
        logger.warning("provider slot 'brain' is degraded: %s", brain_status.reason)
    # A degraded brain slot resolves to `None` (`resolve_slot`'s own
    # generic shape); normalized to `()` here so the `resolve_model()`
    # loop below, and any other `tier_brains` consumer, stays a plain
    # "iterate what is there" loop with no `None`-check of its own.
    tier_brains = tier_brains_result or ()
    app.state.tier_brains = tier_brains
    # Nothing else in this file reads `app.state.brain` today, but it stays
    # pointed at the top tier's `XaiBrain` so any future reader keeps seeing
    # the one that actually reaches Home Assistant. `None` when the brain
    # slot is degraded -- there is no top tier to point at.
    app.state.brain = tier_brains[-1].brain if tier_brains else None

    app.state.provider_slots: "dict[str, ProviderSlotStatus]" = {
        "stt": stt_status,
        "tts": tts_status,
        "brain": brain_status,
    }

    for tier in tier_brains:
        resolved_model = await tier.brain.resolve_model()
        logger.info("resolved brain tier %d model: %s", tier.index, resolved_model)

    # Plan 06-01 (D-01, D-04, PLUG-09): `PluginManager` is the only thing
    # in this process that spawns an MCP host -- Home Assistant and
    # weather are both ordinary rows in the `plugins` table now (seeded by
    # `alembic/versions/0008_plugin_tables.py`), reached through the same
    # loop as any plugin an admin installs later. `lifespan` itself
    # constructs no MCP host and names no child module.
    #
    # `_current_safety_block` is the one place this process ever derives
    # the JSON-shaped block `Policy.from_config` expects from the policy
    # repository -- never a `Policy` object constructed here and
    # serialized, matching this file's own pre-existing rule (SAFE-05,
    # D-11). `PluginManager` calls this fresh for the plugin that
    # `enforces_policy` (Home Assistant, seeded true) on every start and
    # every future respawn, so a policy write that lands between two
    # starts is never served a stale block.
    async def _current_safety_block() -> "dict | None":
        return safety_block_from_policy(await policy_repo.load_policy())

    # Plan 05-01 (D-08): the in-process `schedule_workflow` tool host --
    # in-process, not a spawned child, because it holds `workflow_repo`,
    # not a credential (SAFE-09 has no isolation boundary to cross for a
    # database connection). `zone` is `_resolved_timezone`, already
    # resolved above -- the one process-wide reading of the house's own
    # configured zone, never re-read or re-derived here. Built before the
    # plugin manager (CR-01 fix) because `_publish_tool_view` below merges
    # it with every plugin host on every rebuild, including the first one
    # `start_all()` performs.
    workflow_tool_host = WorkflowToolHost(workflow_repo, zone=_resolved_timezone)
    app.state.workflow_tool_host = workflow_tool_host

    # Entities are fetched once, after the plugins are up (below) -- the
    # cacheable catalog prompt is rebuilt from this snapshot every time
    # the plugin set changes, so the ownership block and the tool schema
    # can never describe two different plugin sets (D-10).
    app.state.entities = []

    def _publish_tool_view() -> None:
        """Republish the live view the running assistant actually reads --
        the one place `app.state.tool_host_lookup`/`tools_schema`/
        `catalog_prompt`/`tool_host` are ever assigned (CR-01, code
        review).

        Called by `PluginManager.rebuild()` through its `on_rebuild` hook,
        which is to say: at boot, on an install, an enable, a disable, a
        delete, a configuration save, a crash-driven tool withdrawal and a
        recovery. Before this existed, `lifespan` copied the manager's own
        `hosts`/`tools_schema` onto `app.state` exactly once and nothing
        ever reassigned them, while every `run_turn` call site read
        `app.state` -- so D-15 ("install, enable, disable and configuration
        edits take effect live") and PLUG-05 ("a dead plugin withdraws only
        its own tools") were both true of the manager and false of the
        assistant. A turn is still handed a genuinely immutable view
        (D-08): every attribute below is *reassigned* to a newly built
        object, never mutated, so a turn already holding the previous
        lookup and schema keeps exactly those, mid-rebuild or not.
        """
        app.state.tool_host_lookup = McpToolHostLookup(
            [*plugin_manager.hosts, workflow_tool_host]
        )
        app.state.tools_schema = plugin_manager.tools_schema + mcp_tools_to_openai_tools(
            workflow_tool_host.tools
        )
        # Plan 06-04 (D-10): the ownership block naming which plugin owns
        # each collision-prefixed tool -- rebuilt here, on the same swap as
        # the schema above, so it can never describe a plugin set that
        # schema does not.
        app.state.catalog_prompt = _catalog_prompt(
            app.state.entities, plugin_manager.tool_ownership_prompt
        )
        # `app.state.tool_host` keeps its pre-existing meaning (T-04-13):
        # the running host of the plugin that enforces the house policy,
        # `None` when that plugin is disabled, degraded, or not yet
        # started. WR-01 (code review): it is genuinely reassigned on every
        # swap now, rather than being a boot-time snapshot whose own
        # comment claimed it was. Every respawn, enable and configuration
        # save builds a *new* `McpToolHost`; the previous one is closed.
        app.state.tool_host = plugin_manager.enforcing_host

    # Phase 9 (09-01, Task 3): only when a Google repository actually
    # exists (every real deployment; `tests/test_startup_smoke.py`'s fake
    # repository dict, deliberately, does not) -- `google_http_client` is
    # the same test seam `ha_http_client` above already establishes, owned
    # by this process unless a test injected one, and closed in shutdown
    # only when this process is the one that opened it.
    google_http_client: "httpx.AsyncClient | None" = None
    owns_google_http_client = False
    custom_env_builders: "dict[str, Any] | None" = None
    if google_account_repo is not None:
        injected_google_client = getattr(app.state, "google_http_client", None)
        owns_google_http_client = injected_google_client is None
        google_http_client = (
            injected_google_client if injected_google_client is not None else httpx.AsyncClient(timeout=10.0)
        )
        app.state.google_http_client = google_http_client
        google_token_service = GoogleTokenService(google_account_repo, security_config, google_http_client)
        app.state.google_token_service = google_token_service
        custom_env_builders = {
            GOOGLE_PLUGIN_MODULE: GoogleEnvBuilder(google_account_repo, google_token_service)
        }

    plugin_manager = PluginManager(
        plugin_repo,
        mcp_root=MCP_ROOT,
        security=security_config,
        safety_block_provider=_current_safety_block,
        plugins_config=config.plugins,
        on_rebuild=_publish_tool_view,
        zone_name=timezone_resolution.child_tz,
        custom_env_builders=custom_env_builders,
        # T-09-27, plan 09-05: the Google plugin's own code-only tools
        # (`calendar_insert_event`/`calendar_delete_event`, and Gmail's own
        # pair once a later plan adds them) never reach the model's tool
        # schema, regardless of whether a Google repository is configured
        # -- `custom_env_builders` above is the one gate deciding whether
        # the Google plugin can start at all; this hides its code-only
        # tools whenever it does.
        hidden_tools_by_module={GOOGLE_PLUGIN_MODULE: CODE_ONLY_TOOL_NAMES},
    )
    app.state.plugin_manager = plugin_manager
    # Publishes an empty view first, so every attribute above exists even
    # if `start_all()` raises, and then the real one from `start_all()`'s
    # own closing `rebuild()`.
    _publish_tool_view()
    await plugin_manager.start_all()

    # Phase 9 (09-01, Task 3): started only when a Google repository
    # exists -- 1500s, below the 1800s `min_validity_s`
    # `GoogleEnvBuilder`/`GoogleTokenService` use by default, so a
    # respawned child's token always has at least five minutes left.
    google_token_scheduler: "GoogleTokenRefreshScheduler | None" = None
    if google_account_repo is not None:

        async def _refresh_google_plugin() -> None:
            await refresh_google_plugin(plugin_manager, plugin_repo)

        google_token_scheduler = GoogleTokenRefreshScheduler(_refresh_google_plugin)
        google_token_scheduler.start()
    app.state.google_token_scheduler = google_token_scheduler

    # Kept on app.state so plan 03-07's write routes can compare a would-be
    # new block against the one the running child was actually spawned
    # with, before deciding whether a respawn is needed. `None` when no
    # plugin is currently enforcing the policy at all.
    app.state.safety_block = (
        await _current_safety_block() if plugin_manager.enforcing_host is not None else None
    )

    # Plan 06-01 (D-07's own posture, extended in a following plan): the
    # Home Assistant plugin may not be running at all (disabled, or not
    # yet spawned successfully) -- this must tolerate that tool not
    # existing and build the prompt from an empty entity list rather than
    # raising, since a degraded Home Assistant is meant to be a survivable
    # boot. Reads through the merged lookup, not a single named host,
    # since that is the one object guaranteed to exist regardless of which
    # plugins are actually running.
    try:
        entities_result = await app.state.tool_host_lookup.call_tool("ha_list_entities", {})
        entities = _tool_result_json(entities_result)
    except UnknownToolError:
        entities = []
    app.state.entities = entities if isinstance(entities, list) else []
    # The entity snapshot is the one input `_publish_tool_view` cannot
    # produce for itself -- republish now that it exists, so the catalog
    # prompt carries both the entities and the current ownership block.
    _publish_tool_view()

    # Plan 04-05 (D-09): macros live in the database now, not on `config`.
    # This snapshot exists only for the wiring check
    # (`tests/test_startup_smoke.py`'s `_EXPECTED_STATE_ATTRS`) and for the
    # precache below, which can only ever precache what exists at boot --
    # every `run_turn` call site reads fresh through `_current_macros`
    # instead of this attribute (see that function's own docstring for why).
    seeded_macros = await macro_repo.list_macros()
    app.state.macros = tuple(seeded_macros)

    # Every filler phrase, every operator-configured extra (short
    # confirmations, per `config.example.yaml`'s own `tts.precache` comment),
    # and every database-held macro's `reply` is rendered once, here,
    # against every sink this process actually serves -- the browser sink
    # (Phase 1's dev-harness consumer) and the camera sink (Phase 2's real
    # speaker, 260922-cts). A macro reply missing from this list would raise
    # at turn time instead of here (Pitfall 4, 01.1-RESEARCH.md), the exact
    # silent-REST-call regression this precache step exists to prevent --
    # unchanged by the move from `config.macros` to the database (D-12): the
    # precache source changed, the guarantee it gives did not. A failure
    # here propagates uncaught, matching this function's existing posture
    # toward `tool_host.start()`: a broken startup should stop the process,
    # not start it half-configured with a filler (or macro-reply) path that
    # will fall over on the first turn.
    #
    # Plan 07-02: a degraded tts slot (D-04, missing credential) leaves
    # `app.state.tts` `None` -- there is no client to precache through,
    # and this must not crash the boot the same way a degraded stt or
    # brain slot does not. The other two slots' own startup work above is
    # already independent of this one, so skipping only this step is
    # enough; `filler_cache`/`filler_caches` stay honest empty containers
    # rather than half-populated ones.
    #
    # 260922-cts: `app.state.filler_cache` stays the flat, browser-only
    # `{text: bytes}` dict every existing reader already expects
    # (`routes/macros.py`/`routes/workflows.py`'s own save-time precache,
    # and this same function's own degraded-tts-slot test).
    # `app.state.filler_caches` is the new,
    # sink-keyed mapping `run_turn`/`_scheduled_speak` actually read from
    # (`turn/controller.py`'s own `CachedTts`), built once from the same
    # `filler_phrases` list against both sinks this process serves. A macro
    # created or edited after boot is added to every sink's cache by those
    # two routes (`tts_cache.precache_other_sinks`).
    filler_phrases = [*FILLER_TEXT.values(), *config.tts.precache, *(m.reply for m in seeded_macros)]
    camera_tts_sink = SinkFormat(codec=config.tts.codec, sample_rate=config.tts.sample_rate)
    # Plan 10-10 (D-14): the edge sink's own `SinkFormat` -- mirrors
    # `EdgeAudioSource.sink_format()` exactly (mono PCM16 at the
    # configured sample rate) without constructing the source itself this
    # early; the `resolved_audio_source` branch further down is what
    # actually builds it.
    edge_tts_sink = SinkFormat(codec="pcm", sample_rate=config.edge.sample_rate)
    if app.state.tts is not None:
        browser_sink = app.state.tts.browser_sink()
        browser_cache = await precache_all(
            app.state.tts,
            Path(config.tts.cache_dir),
            filler_phrases,
            config.tts.voice_id,
            browser_sink,
        )
        camera_cache = await precache_all(
            app.state.tts,
            Path(config.tts.cache_dir),
            filler_phrases,
            config.tts.voice_id,
            camera_tts_sink,
        )
        app.state.filler_cache = browser_cache
        app.state.filler_caches = {
            None: browser_cache,
            (camera_tts_sink.codec, camera_tts_sink.sample_rate): camera_cache,
        }
        if resolved_audio_source == EDGE_SOURCE_NAME:
            edge_sink_key = (edge_tts_sink.codec, edge_tts_sink.sample_rate)
            if browser_sink == edge_tts_sink:
                # The browser sink is already this same (codec,
                # sample_rate) pair -- reuse it rather than precaching a
                # second, byte-identical cache under a different key.
                app.state.filler_caches[edge_sink_key] = browser_cache
            else:
                app.state.filler_caches[edge_sink_key] = await precache_all(
                    app.state.tts,
                    Path(config.tts.cache_dir),
                    filler_phrases,
                    config.tts.voice_id,
                    edge_tts_sink,
                )
        logger.info("precached %d phrases for the browser and camera sinks", len(browser_cache))
    else:
        app.state.filler_cache = {}
        app.state.filler_caches = {}
        logger.warning(
            "provider slot 'tts' is degraded -- skipping the startup precache for %d phrase(s)",
            len(filler_phrases),
        )

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
    # 260923-sfi (D1): the FIFO carries camera_tts_sink's format on every
    # backend, and every reader consumes it in packets or frames -- so
    # trailing silence is harmless everywhere, and padding is not tied to
    # one backend.
    speaker_writer = FifoWriter(
        config.speaker.fifo_path,
        reopen_timeout_s=config.speaker.reopen_timeout_s,
        pad_bytes=silence(camera_tts_sink, config.speaker.tail_pad_s),
        pad_idle_s=config.speaker.tail_pad_idle_s,
    )
    app.state.speaker_writer = speaker_writer
    app.state.background_turns.add(asyncio.create_task(_open_speaker_writer(speaker_writer)))

    # The supervisor's own client, not the browser's httpx usage elsewhere
    # in this file (there isn't one shared today) -- opened and closed with
    # the supervisor's own lifetime, since nothing else in this process
    # needs to issue the go2rtc backchannel PUT. Unused by the tapo_talk
    # and tcp backends (pytapo is an in-process library call, not an HTTP
    # PUT, and the tcp backend has no HTTP client at all), but still built
    # and closed unconditionally so the lifespan's own resource-cleanup
    # shape stays the same either way.
    speaker_http_client = httpx.AsyncClient()
    # speaker.backend selects which FIFO reader owns egress (D-vqa): the
    # variable and app.state attribute names stay `ffmpeg_supervisor`
    # regardless of backend, since every other reference in this file
    # (reconnect wiring, teardown) only needs the shared start()/stop()/
    # handle_reconnect() shape both supervisors implement.
    if config.speaker.backend == "tapo_talk":
        ffmpeg_supervisor = _build_tapo_talk_supervisor(config)
    elif config.speaker.backend == "tcp":
        ffmpeg_supervisor = _build_tcp_supervisor(config)
    else:
        ffmpeg_supervisor = _build_ffmpeg_supervisor(config, speaker_http_client)
    ffmpeg_supervisor.start()
    app.state.ffmpeg_supervisor = ffmpeg_supervisor

    # Plan 05-03 (T-05-18): the room's one speaker, one lock -- created
    # once, here, before the camera source that will write through it
    # exists at all, and shared by every writer this process has: the
    # live turn path (`_make_run_turn_for_source`, above) and the
    # scheduler's own scheduled-speech closure (below, once
    # `app.state.camera_source`/`app.state.tts`/`app.state.filler_cache`
    # all exist). A scheduled step's utterance interleaved with a live
    # reply's on the same FIFO is garbled audio, not two sentences.
    app.state.speaker_lock = asyncio.Lock()

    # `on_reconnect=ffmpeg_supervisor.handle_reconnect` closes a gap plan
    # 02-07 recorded and deliberately left open (outside its own file
    # scope): without this, a camera that drops and reconnects gets its
    # microphone back but never re-issues the go2rtc backchannel PUT, so
    # the speaker stays silent after a recovery nothing else would notice
    # (T-02-33's same "audio outlives what nobody noticed" shape, applied
    # to the speaker rather than the recordings). `backoff_s` is left on
    # its own default -- this plan does not change it.
    #
    # `sink=camera_tts_sink` (260922-cts): the same `SinkFormat` the
    # precache step above just built both in-memory caches against --
    # never a second, independently-constructed value that could drift
    # from it -- so `CameraAudioSource.sink_format()` reports the format
    # this speaker actually plays, and `turn/controller.py`'s `_speak`
    # asks the real xAI/Piper provider for that same format on a cache
    # miss.
    # Phase 10 (D-06, D-07, D-14, D-15): `camera_source` is built
    # unconditionally in both branches below -- the scheduled speaker and
    # the calibration route still read `app.state.camera_source` -- but
    # `.start()` (the RTSP reconnect supervisor) runs only when the camera
    # is the resolved audio source. An edge deployment's camera microphone
    # never opens (D-15); the camera speaker code stays because SRC-02 is
    # validated and a stranger with no Pi still depends on it.
    camera_source = CameraAudioSource(
        config.camera, speaker_writer, sink=camera_tts_sink, on_reconnect=ffmpeg_supervisor.handle_reconnect
    )
    app.state.camera_source = camera_source

    # Plan 02-11 Task 3: the single in-flight guard the run route below
    # checks before calling run_echo_calibration against these same
    # camera_source/speaker_writer resources -- two probes playing at once
    # would measure each other (T-02-51).
    app.state.calibration_in_progress = False

    if resolved_audio_source == CAMERA_SOURCE_NAME:
        camera_source.start()

        wake_detector = _build_wake_detector(config.wake.resolve(CAMERA_SOURCE_NAME))

        # CR-01 fix (code review): every one of these used to be omitted,
        # which left `SourceRunner.__init__`'s own "absent configuration"
        # fallbacks in force for the one runner the application actually
        # builds -- no refractory window, no gate, no pre-roll replay, and
        # barge-in forced to `BargeInConfig(enabled=False)` regardless of
        # what `config.example.yaml` said. `wake_config`/`gate_config`/
        # `barge_in_config` are the *global* `Config` sections, not
        # pre-resolved -- `SourceRunner.__init__` itself calls
        # `.resolve("camera")` on each, the same way `wake_detector` above
        # already resolves `config.wake` for engine selection.
        # `is_media_playing` is left at its default (`None`, resolving to
        # "nothing is ever playing" inside `WakeGate`): no real
        # Home-Assistant-backed implementation exists yet, and
        # `config.example.yaml`'s own `gate.mute_when_playing` ships
        # empty, so the callable is never actually reached with the
        # shipped default (`wake/gate.py`'s own short-circuit). Wiring a
        # live one is future work, not something this fix pass invents
        # untested.
        preroll = PrerollBuffer(camera_source.source_format(), config.camera.preroll_ms)

        # Plan 02-12 Task 3: the startup refusal CR-02 left as a gap.
        # Turning `correlation_enabled` on for a source with no valid,
        # non-stale calibration on file must stop the process by name,
        # never fall back to the guard-window-plus-floor gate the code
        # review already found cannot tell the assistant's own voice from
        # the operator's -- that fallback is exactly the silent failure
        # this refusal exists to replace with a loud one.
        # `find_latest_calibration` never raises for "no calibration
        # directory yet" or "directory exists but empty" (its own
        # docstring): both read as `None` here, the same "missing" case.
        camera_barge_in = config.barge_in.resolve("camera")
        camera_calibration: EchoCalibration | None = None
        if camera_barge_in.correlation_enabled:
            camera_calibration = find_latest_calibration(config.calibration.dir)
            if camera_calibration is None:
                raise ConfigError(
                    "barge_in.sources.camera.correlation_enabled is true, but no echo-path "
                    f"calibration exists at {config.calibration.dir!r} -- run "
                    "scripts/dev-calibrate-echo.sh against the real camera before enabling "
                    "correlation, or the gate would run uncalibrated against real playback"
                )
            now = datetime.now(timezone.utc)
            if camera_calibration.is_stale(now, config.calibration.max_age_days):
                age_days = (now - camera_calibration.taken_at).total_seconds() / 86400.0
                raise ConfigError(
                    f"barge_in.sources.camera.correlation_enabled is true, but the stored "
                    f"calibration is {age_days:.1f} days old, past calibration.max_age_days="
                    f"{config.calibration.max_age_days} -- run scripts/dev-calibrate-echo.sh "
                    "again before enabling correlation on a stale measurement"
                )

        active_runner = SourceRunner(
            CAMERA_SOURCE_NAME,
            camera_source,
            wake_detector,
            camera_source.decode_for_detector,
            _make_run_turn_for_source(app, config, CAMERA_SOURCE_NAME),
            wake_config=config.wake,
            gate_config=config.gate,
            barge_in_config=config.barge_in,
            preroll=preroll,
            calibration=camera_calibration,
            wake_event_repo=wake_event_repo,
            # Plan 09-06/09-07 (D-06, D-09, T-09-41): a callable, not a
            # plain float, reading `app.state.follow_up_window_s` --
            # resolved once at boot above, and the exact attribute
            # `PUT /api/settings/follow-up-window` overwrites live -- so
            # a later change reaches the very next follow-up window with
            # no restart.
            follow_up_window_s=lambda: app.state.follow_up_window_s,
            follow_up_echo_tail_s=config.follow_up.echo_tail_ms / 1000,
        )
    elif resolved_audio_source == EDGE_SOURCE_NAME:
        wake_detector = SegmentBoundedWakeDetector(
            _build_wake_detector(config.wake.resolve(EDGE_SOURCE_NAME))
        )

        edge_source = EdgeAudioSource(config.edge)
        edge_source.on_segment_start = wake_detector.mark_segment_start
        app.state.edge_source = edge_source

        # Plan 10-10 (D-16): see `_resolve_edge_barge_in_config`'s own
        # docstring for why this is never the bare global `config.barge_in`
        # handed to the camera runner unchanged.
        edge_barge_in_config = _resolve_edge_barge_in_config(config)

        active_runner = SourceRunner(
            EDGE_SOURCE_NAME,
            edge_source,
            wake_detector,
            edge_source.decode_for_detector,
            # D-14, D-15: `room_speaker=False` -- the reply plays on the
            # XVF3800 over the same edge socket (`send_audio`), never on
            # the camera FIFO, so this turn never waits on
            # `app.state.speaker_lock`.
            _make_run_turn_for_source(app, config, EDGE_SOURCE_NAME, room_speaker=False),
            wake_config=config.wake,
            gate_config=config.gate,
            barge_in_config=edge_barge_in_config,
            preroll=PrerollBuffer(
                edge_source.source_format(),
                # 10-CONTEXT.md: the detector report lag this replay
                # covers belongs to the wake engine (D-07), the same one
                # the camera uses -- `config.camera.preroll_ms` is reused
                # deliberately, the same value `/ws/listen` already
                # reuses for its own server-side wake replay. This is not
                # the Pi's own `pre_roll_ms` (D-08), which the Pi applies
                # to its own capture buffer before this server ever sees
                # a byte.
                config.camera.preroll_ms,
            ),
            wake_event_repo=wake_event_repo,
            follow_up_window_s=lambda: app.state.follow_up_window_s,
            follow_up_echo_tail_s=config.follow_up.echo_tail_ms / 1000,
        )
    else:
        raise ConfigError(
            f"audio_source resolved to {resolved_audio_source!r}, which this application "
            f"has no source for -- expected {CAMERA_SOURCE_NAME!r} or {EDGE_SOURCE_NAME!r}"
        )

    app.state.wake_detector = wake_detector
    if resolved_wake_threshold is not None:
        # Applied by calling `set_wake_threshold` immediately after
        # construction, never by rewriting `config.wake` itself: the
        # configuration is what the file says and the setting is what the
        # operator later chose, and collapsing the two would lose which
        # is which (D-15).
        active_runner.set_wake_threshold(resolved_wake_threshold)
    app.state.source_runners = [active_runner]
    app.state.source_runner_tasks = [asyncio.create_task(active_runner.run())]

    # The retention sweep (DBG-06, plan 02-08): runs once at startup and
    # again every `debug.expiry_interval_s`, for the life of the process --
    # the same `start()`/`stop()` shape `ffmpeg_supervisor` above already
    # uses, and no scheduling dependency, since this is one loop that
    # sleeps and calls one function.
    retention_scheduler = RetentionScheduler(
        config.session.dir,
        config.session.retain_days,
        config.session.expiry_interval_s,
        # WR-06 (code review): `wake_events` is the other persistent record
        # of what this house's microphone heard, and Phase 8 shipped it
        # with no owner and no bound. Same sweep, same window -- D-12's
        # "exactly one owner" for deletion, extended to cover it.
        wake_event_repo=wake_event_repo,
    )
    retention_scheduler.start()
    app.state.retention_scheduler = retention_scheduler

    # Plan 05-03: the scheduler's own utterance sink for a `speak` step
    # and for a fire-time lateness/refusal sentence (D-14) -- plays
    # through the exact `_speak` loop a live turn's reply uses, against
    # the same camera source, sharing `app.state.speaker_lock` with the
    # live turn path (above) so the two writers can never interleave on
    # the room's one speaker (T-05-18). Reads from the startup filler/
    # macro-reply cache through the same `CachedTts` adapter a cached
    # macro reply already uses when the text was already precached
    # (04-06's own "is this text already in the cache" keying) -- falling
    # back to live synthesis, never raising on a miss the way `CachedTts`
    # alone would, for text no precache pass ever saw (a fire-time
    # refusal, a lateness sentence, or a `speak` step's own authored
    # text). A fresh `TurnTimings()` per call, matching
    # `_make_run_turn_for_source`'s own "never shared across turns" rule.
    async def _scheduled_speak(text: str) -> None:
        # CR-03 (code review): a degraded text-to-speech slot leaves
        # `app.state.tts` as `None`, and this closure had no cover of its
        # own (unlike `routes/macros.py`/`routes/workflows.py`, whose
        # `browser_sink()` calls sit inside a bare `except Exception`). A
        # scheduled step that cannot be spoken is logged by name here,
        # once per firing, rather than raising inside the scheduler.
        #
        # 260922-cts: this closure always speaks through
        # `app.state.camera_source` -- there is no other scheduled-speech
        # sink -- so the cache membership check and the sink handed to
        # `_speak` both come from that source's own `sink_format()`, never
        # `app.state.filler_cache` (the flat, browser-only dict). Reading
        # the browser dict here was the same bug the module docstring's
        # Fix section names for the filler/macro cache, applied instead to
        # a scheduled step's own utterance.
        sink = app.state.camera_source.sink_format()
        camera_cache = app.state.filler_caches.get((sink.codec, sink.sample_rate), {})
        if text not in camera_cache and app.state.tts is None:
            logger.warning(
                "not speaking a scheduled utterance: %s",
                degraded_turn_refusal(getattr(app.state, "provider_slots", None)),
            )
            return
        speaking_tts = CachedTts(app.state.filler_caches) if text in camera_cache else app.state.tts
        await _speak(
            app.state.camera_source,
            speaking_tts,
            TurnTimings(),
            text,
            kind="answer",
            speech_lock=app.state.speaker_lock,
            sink=sink,
        )

    # The workflow poller (plan 05-01, D-01 .. D-03): a fourth
    # `start()`/`stop()` scheduler, built after `app.state.tool_host_lookup`
    # exists so its own executor calls through the identical lookup the
    # live turn path calls -- never a workflow-local copy of `allow_call`
    # (D-13). A closure, not `functools.partial`: `WorkflowScheduler`
    # calls `executor(step, now)` positionally, and `execute_step`'s own
    # signature is `(step, tool_host, config, now, *, speak=None)` --
    # `tool_host`/`config` sit between `step` and `now` positionally, so a
    # `functools.partial` with `tool_host=`/`config=` pre-bound as
    # keywords collides with `now` landing in `tool_host`'s positional
    # slot the moment a step actually fires (`TypeError: execute_step()
    # got multiple values for argument 'tool_host'` -- caught by plan
    # 05-03's own SAFE-08 test, the first test in this codebase to poll a
    # step through the real `lifespan` wiring rather than a hand-built
    # executor). `tests/test_workflow_tracer.py`'s own `lambda step, now:
    # execute_step(step, ..., now)` shape is the one this closure follows.
    async def _workflow_executor(step: Any, now: datetime) -> Any:
        return await execute_step(
            step,
            app.state.tool_host_lookup,
            config.workflow,
            now,
            speak=_scheduled_speak,
            tool_owners=app.state.plugin_manager.owners_of_bare_name,
        )

    workflow_scheduler = WorkflowScheduler(workflow_repo, _workflow_executor, config.workflow)
    workflow_scheduler.start()
    app.state.workflow_scheduler = workflow_scheduler

    # Quick task 260924-4is (D1): started last, right before `yield`, so a
    # boot that fails before this point never starts the watcher thread --
    # a failed test boot leaks no threads. Stopped as the very last
    # teardown statement below, so it watches through the whole shutdown
    # sequence too.
    loop_stall_reporter = LoopStallReporter()
    loop_stall_reporter.start()

    yield

    # 260923-spd: the speaker supervisor stops FIRST, before anything else
    # in this function -- a live pod restart left the camera holding a
    # stale talk session and refusing port 8800 with 401 until a power
    # cycle, because the old order stopped it LAST, after the source
    # runners, `drain_pending_wake_events()` (DB writes), and
    # `camera_source.close()` (RTSP) had already spent most of the 30 s
    # termination grace period. `asyncio.wait_for(..., 5)` bounds this call
    # so a stuck supervisor still leaves the rest of shutdown its own share
    # of the grace period, rather than eating all of it and getting
    # SIGKILLed mid-close anyway.
    try:
        await asyncio.wait_for(ffmpeg_supervisor.stop(), timeout=5)
    except TimeoutError:
        logger.warning("ffmpeg_supervisor.stop() did not finish within 5s during shutdown; continuing shutdown")
    # CR-03 fix (code review): `SourceRunner.run()` now contains a
    # per-chunk exception rather than letting it end the task (see that
    # method's own docstring), but a task can still end with a stored
    # exception from somewhere this loop cannot anticipate (a bug in
    # `run_turn` itself, say). `return_exceptions=True` is what keeps that
    # possibility from mattering here: `task.cancel()` on an already-done
    # task is a no-op, and `await task` on one that ended with a
    # non-cancellation exception used to re-raise it, aborting every
    # cleanup call below (`camera_source.close()`, `retention_scheduler.
    # stop()`, `speaker_writer.close()`, `plugin_manager.stop_all()`) and
    # leaving every MCP child and the FIFO's open handle behind uncleanly
    # on process exit.
    for task in app.state.source_runner_tasks:
        task.cancel()
    await asyncio.gather(*app.state.source_runner_tasks, return_exceptions=True)
    # WR-05 (code review): the runner tasks are gathered above, but a wake
    # event's own write is a separate task `SourceRunner` schedules and
    # never awaits (the turn path's latency budget is why). Nothing drained
    # those, so the wake immediately before a restart was written into an
    # engine this teardown was about to dispose. Drained here, after the
    # listening loops have stopped scheduling new ones and before
    # `engine.dispose()` below.
    for runner in getattr(app.state, "source_runners", []):
        await runner.drain_pending_wake_events()
    await camera_source.close()
    # Phase 10: closed only when the edge source was actually built
    # (`resolved_audio_source == EDGE_SOURCE_NAME`) -- `app.state.
    # edge_source` defaults to `None` otherwise, and `EdgeAudioSource.
    # close()` is what a `frames()` reader still waiting on the queue
    # needs to see the process end, the same reason `camera_source.close()`
    # runs unconditionally just above.
    if app.state.edge_source is not None:
        await app.state.edge_source.close()
    wake_detector.close()
    await retention_scheduler.stop()
    await workflow_scheduler.stop()
    await speaker_http_client.aclose()
    # Quick task 260924-4iu (a): `app.state.tts` is a `BatchTtsAdapter`
    # around `XaiTts` for the xAI slot, and `BatchTtsAdapter.__getattr__`
    # forwards `aclose` to it (`aclose` is not in `_NEVER_FORWARDED`). A
    # degraded slot is `None`, and Piper has no `aclose` -- either way
    # `getattr` returns `None` and this closes nothing.
    tts_closer = getattr(app.state.tts, "aclose", None)
    if tts_closer is not None:
        await tts_closer()
    await speaker_writer.close()
    # Phase 9 (09-01, Task 3): stopped before `plugin_manager.stop_all()`
    # -- a refresh mid-shutdown would race a plugin teardown already in
    # progress -- and the http client closed only when this process is
    # the one that opened it (the same `owns_google_http_client` test seam
    # `ha_http_client` above already establishes).
    if google_token_scheduler is not None:
        await google_token_scheduler.stop()
    if google_http_client is not None and owns_google_http_client:
        await google_http_client.aclose()
    # Plan 06-01: the manager owns every plugin child's teardown now --
    # one call, not a per-host `aclose()` for however many plugins happen
    # to be running.
    await plugin_manager.stop_all()
    await db_engine.dispose()
    # Quick task 260924-4is (D1): last statement of teardown -- see the
    # `start()` call above for why.
    loop_stall_reporter.stop()


# `require_setup_complete` (WEB-01, D-08) used to be registered here as an
# application-level dependency, which meant `app.frontend()`'s own SPA
# routes below inherited it too -- FastAPI's own `_FrontendRouteGroup`
# construction carries `*include_context.dependencies` from the app that
# mounted it, confirmed directly against the installed
# `fastapi>=0.141,<0.142`'s source. A fresh deployment's very first page
# load of `/` or `/setup` then 503'd with "setup incomplete" before the
# browser ever loaded the JavaScript that could call
# `/api/auth/create-admin` and clear the gate -- the wizard could never
# appear (deferred-items.md #1).
#
# Option B (the operator's chosen fix) moves the gate off `app` itself and
# onto the backend routes instead: `register_routers` (routes/__init__.py)
# mounts every feature router through one parent that carries
# `require_setup_complete` as its own dependency, and every top-level route
# in this file carries the identical dependency explicitly (see the comment
# above this file's own route decorators for why those stay direct
# declarations rather than a second wrapping router). `app` itself now
# carries no app-level dependency at all, so the frontend mount below is
# never subject to the gate by construction, not by a path exemption a
# future route could slip past. `SETUP_GATE_EXEMPT_PATHS`
# (auth/dependencies.py) still names the handful of backend paths that
# must answer before any user exists (create-admin, setup status, the
# wizard) -- `require_setup_complete` itself still exempts those exact
# paths, so gating their routers too is a no-op, not a second gate.
app = FastAPI(lifespan=lifespan)
register_routers(app)
# The built single-page application, served at the same origin its own
# session cookie needs (D-15). `app.frontend()` (verified directly against
# the installed `fastapi==0.141.1`'s source this session, 03-RESEARCH.md
# Pattern 3) stores these as *low-priority* routes, checked only after
# every ordinary `@app.get`/`@app.post`/`@app.websocket` route above fails
# to match, regardless of where this call sits relative to them -- this
# replaces both the old `/static` mount and the old `GET /` file response
# in one call, and there is no hand-rolled catch-all route to get the
# ordering of wrong. `check_dir=False` is explicit, not `"auto"`: a clean
# clone that has not run `bun run build` yet must still start (the warning
# above already told the operator why the page will be blank), so this
# must never raise merely because the directory does not exist yet. It is
# constructed here, with `app.dependencies` still empty at this point in
# the module (`app = FastAPI(lifespan=lifespan)` above carries none), so
# `_FrontendRouteGroup` -- which captures `self.dependencies` at the moment
# `app.frontend()` is called, not lazily -- captures an empty list. It
# carries no `require_setup_complete` dependency of any kind: the entire
# point of this section.
app.frontend("/", directory=str(FRONTEND_DIR), check_dir=False)


@app.get("/health")
async def health() -> dict[str, str]:
    """Exempt from the setup gate by name (`SETUP_GATE_EXEMPT_PATHS`) --
    a gate that blocks the one route naming whether the process is even up
    is a lockout, not a safeguard. Carries no role requirement either: a
    health check answering only to an authenticated caller is not a health
    check a container orchestrator or a load balancer can use."""
    return {"status": "ok"}


# Every route below this point is a real backend route -- a control
# surface, a configuration read, or a turn-starting surface -- so every one
# of them carries `Depends(require_setup_complete)` explicitly, alongside
# whatever `require_role(...)` dependency it already needs. This is a
# per-route dependency, not a router-level one (unlike `register_routers`
# in routes/__init__.py, which mounts every *feature* router through one
# parent that carries the gate structurally): nesting one of these routes'
# `APIWebSocketRoute` (`/ws/turn`, below) inside a wrapping `APIRouter`
# that then gets `include_router`'d onto `app` hits a real limitation in
# the installed `fastapi==0.141.1` -- the resulting `_EffectiveRouteContext`
# for a WebSocket route loses its own `.path` field (verified directly
# against that version's source this session; `original_route.path` still
# carries it, but the flattening helpers this project's own tests use do
# not know to look there), which would silently break the route-enumeration
# drift guards below. Keeping this file's small, fixed set of top-level
# routes as direct `@app...` declarations, each with the dependency spelled
# out, sidesteps that quirk entirely while still making the gate structural
# where it matters most -- every *feature* router, added far more often
# than this file's own routes are. A route added below without this
# dependency is caught by
# `tests/test_auth_roles.py::test_every_backend_route_carries_the_setup_gate_except_health`.
@app.get("/transport", dependencies=[Depends(require_setup_complete)])
async def get_transport() -> dict[str, str]:
    """The one thing the page needs to pick a transport without guessing.

    Selected straight from `ServerConfig.transport` -- an unrecognized
    value already stopped startup back in `config.py`, so by the time this
    route can answer, the value is guaranteed to be one of the two the page
    knows how to branch on.

    Deliberately left without a `require_role` dependency (recorded in
    03-05-SUMMARY.md's route-by-route table): it discloses a deployment
    configuration choice (`"websocket"` or `"webrtc"`), never house data,
    and is not itself a control surface -- unlike the calibration and turn
    routes below, calling it cannot make anything in a real home happen.
    It still sits behind the setup gate above, like every other route not
    named in `SETUP_GATE_EXEMPT_PATHS`.
    """
    config: Config = app.state.config
    return {"transport": config.server.transport}


class CalibrationRunRequest(BaseModel):
    placement_note: str = ""


def _calibration_disabled_error() -> HTTPException:
    """The client error both calibration routes return when
    `calibration.route_enabled` is off, the shipped default (T-02-47).

    Registered unconditionally and checked at request time, never
    registered conditionally: a disabled route must be distinguishable
    from a route nobody wrote, which is exactly what Phase 3's wizard
    needs to be able to tell apart (this plan's own objective).
    """
    return HTTPException(
        status_code=403,
        detail=(
            "echo-path calibration is disabled -- set calibration.route_enabled: true "
            "to allow this route. It makes a real home play a sound and record the "
            "room; Phase 2 has no authentication in front of it yet (WEB-01/WEB-04 "
            "add that in Phase 3)."
        ),
    )


def _calibration_response(calibration: EchoCalibration, now: datetime) -> dict[str, Any]:
    """Everything a caller needs to show a result or an age -- never the
    camera's own configuration, never a URL. Every field here is either a
    measured number, an identifier, or operator-supplied text, mirroring
    `EchoCalibration`'s own field-level guarantee (`calibration/record.py`).
    T-02-48's own test asserts this against the JSON-serialized body, not
    just this dict, since a route is where a stray field would actually
    escape.
    """
    return {
        "delay_s": calibration.delay_s,
        "gain": calibration.gain,
        "confidence": calibration.confidence,
        "agc_verdict": calibration.agc_verdict,
        "source": calibration.source,
        "placement_note": calibration.placement_note,
        "taken_at": calibration.taken_at.isoformat(),
        "age_days": (now - calibration.taken_at).total_seconds() / 86400.0,
        "echo_cancelled": calibration.echo_cancelled,
    }


@app.get(
    "/calibration/echo-path",
    dependencies=[Depends(require_setup_complete), Depends(require_role(Role.OPERATOR))],
)
async def get_echo_path_calibration() -> dict[str, Any]:
    """The last stored echo-path calibration and its age, or the
    not-found status when nothing has been measured yet -- the state this
    project ships in, not an error.

    Behind `require_role(Role.OPERATOR)` as of plan 03-05 (T-03-32,
    recorded in 03-05-SUMMARY.md's route-by-route table): CONTEXT.md
    defines `operator` as the role that "uses the voice surfaces," which
    this measurement is one of, and it is a real acoustic probe against a
    real home -- not a read a viewer needs.
    """
    config: Config = app.state.config
    if not config.calibration.route_enabled:
        raise _calibration_disabled_error()
    calibration = find_latest_calibration(config.calibration.dir)
    if calibration is None:
        raise HTTPException(status_code=404, detail="no echo-path calibration has been taken yet")
    return _calibration_response(calibration, datetime.now(timezone.utc))


@app.post(
    "/calibration/echo-path/run",
    dependencies=[Depends(require_setup_complete), Depends(require_role(Role.OPERATOR))],
)
async def run_echo_path_calibration(payload: CalibrationRunRequest) -> dict[str, Any]:
    """Run a live echo-path calibration against this process's own
    camera source and speaker writer -- never a second RTSP connection or
    a second FIFO writer alongside the ones `lifespan` already holds open.

    Guarded by a single in-flight flag (T-02-51): a second request while a
    run is already going gets a conflict, never a second probe playing at
    the same time as the first.
    """
    config: Config = app.state.config
    if not config.calibration.route_enabled:
        raise _calibration_disabled_error()
    # Plan 10-10 (D-15): `getattr` with the camera default -- every test
    # that predates this plan installs calibration state with no
    # `audio_source` attribute at all (`_install_calibration_state`, a
    # direct route call with no lifespan boot), and that absence must
    # keep meaning "camera," exactly as it did before this plan. A real
    # boot always sets `app.state.audio_source` (`lifespan`, above).
    resolved_audio_source = getattr(app.state, "audio_source", CAMERA_SOURCE_NAME)
    if resolved_audio_source != CAMERA_SOURCE_NAME:
        raise HTTPException(
            status_code=409,
            detail=(
                "the camera microphone is off -- the edge microphone does not use "
                "echo-path calibration"
            ),
        )
    if app.state.calibration_in_progress:
        raise HTTPException(status_code=409, detail="a calibration run is already in progress")

    # 260923-pyj (D4): the probe goes into the same FIFO as the reply
    # audio, so it must be written in the format the egress reader
    # expects. The camera source already declares that format
    # (lifespan's `camera_tts_sink`) -- read duck-typed here, never from
    # `config.tts`, since this route's own tests give a config with no
    # `tts` attribute at all.
    sink_format = getattr(app.state.camera_source, "sink_format", None)
    sink = sink_format() if sink_format is not None else None

    app.state.calibration_in_progress = True
    try:
        result = await run_echo_calibration(
            app.state.camera_source,
            app.state.speaker_writer,
            config.camera,
            config.calibration,
            payload.placement_note,
            sink=sink,
            speaker_has_aec=config.speaker.cancels_own_echo,
        )
    finally:
        app.state.calibration_in_progress = False

    if result.failure_reason is not None:
        raise HTTPException(status_code=422, detail=result.failure_reason)

    assert result.calibration is not None
    return _calibration_response(result.calibration, datetime.now(timezone.utc))


class WebrtcOfferPayload(BaseModel):
    sdp: str
    type: str


class WebrtcAnswerPayload(BaseModel):
    sdp: str
    type: str


@app.post(
    "/webrtc/offer",
    dependencies=[Depends(require_setup_complete), Depends(require_role(Role.OPERATOR))],
)
async def webrtc_offer(offer: WebrtcOfferPayload) -> WebrtcAnswerPayload:
    """The one stateless request/response the WebRTC path signals over.

    Behind `require_role(Role.OPERATOR)` as of plan 03-05 (T-03-32): this
    is the unauthenticated-endpoint-that-can-start-a-turn 03-CONTEXT.md
    names explicitly as what this phase closes. See this file's module-level
    `app = FastAPI(...)` comment for why an `HTTPConnection`-typed
    dependency, not `Request`, is what makes the same guarantee reach
    `/ws/turn` below.

    This route does not check `ServerConfig.transport` before answering --
    it always exists and always answers whatever offer arrives. Which
    transport gets *used* is a client-side decision the page makes from
    `GET /transport`; this route has no second event channel of its own to
    grow, matching the WebSocket transport's own single-socket shape.
    """
    transport = WebrtcTransport()
    answer = await create_offer_answer(transport, {"sdp": offer.sdp, "type": offer.type})

    config: Config = app.state.config
    # CR-03 (code review): a degraded slot ends this turn with its own
    # reason rather than an `AttributeError` out of `run_turn`. The offer
    # is still answered -- the peer connection exists and has to be closed
    # cleanly either way -- so the refusal runs inside the turn task, not
    # in front of it.
    timings = TurnTimings()
    # A live read, not `config.macros` (D-12, T-04-28) -- see
    # `_current_macros`'s own docstring for why this call site reads the
    # repository directly rather than trusting an app-state snapshot.
    macros = await _current_macros(app)
    # IN-02 (code review): no `speech_lock=` here, unlike
    # `_make_run_turn_for_source`'s own `speech_lock=app.state.
    # speaker_lock` above. Safe today only because this turn speaks
    # through `transport` (`WebrtcTransport`'s own audio sink), never
    # through `app.state.camera_source` -- the one physical speaker
    # `app.state.speaker_lock` actually protects, and the one thing the
    # scheduler's own `_scheduled_speak` closure ever writes to. There is
    # nothing here for a scheduled step's utterance to interleave with.
    # This stops being true the moment a future change routes
    # browser-sourced audio through that same physical speaker (or
    # reuses `app.state.camera_source` as a fallback for this transport)
    # -- if that happens, thread `speech_lock=app.state.speaker_lock`
    # through here too, the same way the wake-word/camera path already
    # does.
    # WEB-06/D-08: this path publishes to the observer feed too, the same
    # way `/ws/turn` does -- `session_recorder` is built here rather than
    # inline in the call below so its real directory name is on hand for
    # the `turn.started` event's `session_id` (see `_make_run_turn_for_
    # source`'s own comment for why the bare `turn_id` cannot resolve
    # `/live`'s "View full session ->" link). The event is published inside
    # the turn task, after the degraded-slot refusal, so a refused turn
    # opens no card -- matching `/ws/turn`, which refuses before it
    # publishes.
    session_recorder = SessionRecorder(config.session, timings)
    task = asyncio.create_task(
        _run_webrtc_turn(
            app,
            transport,
            ObserverPublishingSource(transport, WEBRTC_SOURCE_NAME, app.state.observer_registry),
            {
                "type": "turn.started",
                "source": WEBRTC_SOURCE_NAME,
                "turn_id": timings.turn_id,
                "session_id": session_recorder.directory.name,
            },
            app.state.stt,
            app.state.brain,
            app.state.tts,
            app.state.tool_host_lookup,
            app.state.tools_schema,
            app.state.catalog_prompt,
            config.brain.max_tool_rounds,
            timings,
            config.stt.max_utterance_s,
            tiers=app.state.tier_brains,
            filler_after_ms=config.brain.filler_after_ms,
            filler_cache=app.state.filler_caches,
            macros=macros,
            state_fetch=_make_state_fetch(app.state.plugin_manager),
            pending_runs_fetch=_make_pending_runs_fetch(app.state.workflow_repo),
            session_recorder=session_recorder,
            workflow_tool_host=app.state.workflow_tool_host,
            tool_owners=app.state.plugin_manager.owners_of_bare_name,
            brain_turn_timeout_s=config.brain.turn_timeout_s,
            state_timeout_ms=config.brain.state_timeout_ms,
            state_domains=config.brain.state_domains,
        )
    )
    app.state.background_turns.add(task)
    task.add_done_callback(app.state.background_turns.discard)

    return WebrtcAnswerPayload(**answer)


async def _run_webrtc_turn(
    app: FastAPI,
    transport: WebrtcTransport,
    source: Any,
    turn_started_event: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Run one turn against `source`, then close `transport`'s peer
    connection.

    `source` is `transport` wrapped in `ObserverPublishingSource` (D-08);
    `transport` itself is kept as a separate parameter because the wrapper
    deliberately forwards only the `AudioSource` protocol plus `barge_in`,
    and `close()` is neither -- closing the peer connection is this
    function's own responsibility, not something to route through a
    fan-out wrapper. The degraded-slot refusal is asked of `transport` for
    the same reason `/ws/turn` asks it of the unwrapped source: a refused
    turn is not a turn, and must not open a card on `/live`.

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
    `app` is the first parameter for one reason: CR-03's degraded-slot
    refusal has to happen inside the `try`/`finally` below, so a refused
    turn still closes its peer connection the way a run one does.
    """
    try:
        if await _refuse_turn_if_any_slot_is_degraded(app, transport):
            return
        app.state.observer_registry.publish(turn_started_event)
        # Plan 09-04 (Task 3): built fresh for this turn -- `WEBRTC_SOURCE_NAME`
        # is the same label `turn_started_event["source"]` above carries.
        await run_turn(
            source, *args, handoff_context=build_handoff_context(app, WEBRTC_SOURCE_NAME), **kwargs
        )
    finally:
        await transport.close()


@app.websocket(
    "/ws/turn",
    dependencies=[Depends(require_setup_complete), Depends(require_role(Role.OPERATOR))],
)
async def turn_ws(websocket: WebSocket) -> None:
    """Behind `require_role(Role.OPERATOR)` as of plan 03-05 (T-03-32) --
    the same guarantee `/webrtc/offer` above carries, applied to this
    project's other turn-starting surface. `require_role`'s own dependency
    chain is typed on `HTTPConnection` specifically so it resolves
    correctly here: a `Request`-typed dependency raises a bare `TypeError`
    when FastAPI tries to solve it against a WebSocket connection
    (confirmed directly against the installed `fastapi==0.141.1`), which
    would have made this route's own tests the only way this gap was ever
    found.
    """
    await websocket.accept()
    source = WebSocketAudioSource(websocket)
    # CR-03 (code review): before this, the default clean-clone Compose
    # deployment (no XAI_API_KEY, so all three slots degrade) closed this
    # socket with no message at all, on an `AttributeError` escaping into
    # Starlette's WebSocket handling. Now the operator is told which slot
    # did not start and why, in the slot's own words.
    if await _refuse_turn_if_any_slot_is_degraded(websocket.app, source):
        return
    config: Config = websocket.app.state.config
    timings = TurnTimings()
    # A live read, not `config.macros` (D-12, T-04-28) -- see
    # `_current_macros`'s own docstring for why this call site reads the
    # repository directly rather than trusting an app-state snapshot.
    macros = await _current_macros(websocket.app)
    # IN-02 (code review): no `speech_lock=` here, for the identical
    # reason `webrtc_offer` above carries the same omission -- this turn
    # speaks through `WebSocketAudioSource`, never through `app.state.
    # camera_source`, so there is nothing for a scheduled step's own
    # utterance (guarded by `app.state.speaker_lock`) to interleave with.
    # See that route's own comment for what would make this unsafe.
    #
    # WEB-06/D-08: this turn appears on the observer feed too, labelled as
    # the browser-microphone source -- 08-UI-SPEC.md's own unresolved
    # question resolved yes (see this plan's SUMMARY): all sources appear
    # uniformly, an operator's own browser turn included.
    #
    # `session_recorder` is built here, not inline in the `run_turn(...)`
    # call below, so its real directory name (`session_id`) is on hand for
    # the published event -- see `_make_run_turn_for_source`'s own comment
    # for why the bare `turn_id` alone cannot resolve `/live`'s "View full
    # session ->" link.
    session_recorder = SessionRecorder(config.session, timings)
    websocket.app.state.observer_registry.publish(
        {
            "type": "turn.started",
            "source": BROWSER_MIC_SOURCE_NAME,
            "turn_id": timings.turn_id,
            "session_id": session_recorder.directory.name,
        }
    )
    source = ObserverPublishingSource(source, BROWSER_MIC_SOURCE_NAME, websocket.app.state.observer_registry)
    await run_turn(
        source,
        websocket.app.state.stt,
        websocket.app.state.brain,
        websocket.app.state.tts,
        websocket.app.state.tool_host_lookup,
        websocket.app.state.tools_schema,
        websocket.app.state.catalog_prompt,
        config.brain.max_tool_rounds,
        timings,
        config.stt.max_utterance_s,
        tiers=websocket.app.state.tier_brains,
        filler_after_ms=config.brain.filler_after_ms,
        filler_cache=websocket.app.state.filler_cache,
        macros=macros,
        state_fetch=_make_state_fetch(websocket.app.state.plugin_manager),
        pending_runs_fetch=_make_pending_runs_fetch(websocket.app.state.workflow_repo),
        session_recorder=session_recorder,
        workflow_tool_host=websocket.app.state.workflow_tool_host,
        tool_owners=websocket.app.state.plugin_manager.owners_of_bare_name,
        brain_turn_timeout_s=config.brain.turn_timeout_s,
        state_timeout_ms=config.brain.state_timeout_ms,
        state_domains=config.brain.state_domains,
        # Plan 09-04 (Task 3): built fresh for this turn.
        handoff_context=build_handoff_context(websocket.app, BROWSER_MIC_SOURCE_NAME),
    )


@app.websocket(
    "/ws/listen",
    dependencies=[Depends(require_setup_complete), Depends(require_role(Role.OPERATOR))],
)
async def listen_ws(websocket: WebSocket) -> None:
    """An always-on browser microphone: the page streams 16 kHz PCM16 for
    as long as it stays open, and each connection gets its own
    `SourceRunner` and its own wake detector -- the same pipeline the
    camera runs. VOICE-03 holds here for the same reason it holds for the
    camera: nothing reaches speech-to-text until this connection's own
    detector fires and the gate allows it.

    A fresh detector per connection, never `app.state.wake_detector`: a
    Vosk recognizer carries decode state across chunks, and two streams
    fed into one would corrupt both. The threshold starts at the camera
    runner's live value, so a tuned threshold applies here too.
    """
    await websocket.accept()
    app_ = websocket.app
    source = WebSocketAudioSource(websocket)
    if await _refuse_turn_if_any_slot_is_degraded(app_, source):
        return
    config: Config = app_.state.config
    # Loading a wake model blocks for up to a second -- off the event loop.
    detector = await asyncio.to_thread(_build_wake_detector, config.wake.resolve(LISTEN_SOURCE_NAME))
    runner = SourceRunner(
        LISTEN_SOURCE_NAME,
        source,
        detector,
        lambda chunk: chunk,  # already 16 kHz PCM16, the detector's own format
        _make_run_turn_for_source(app_, config, LISTEN_SOURCE_NAME, room_speaker=False),
        wake_config=config.wake,
        gate_config=config.gate,
        # ponytail: reuses the camera's pre-roll length, add a listener key if they ever need to differ.
        preroll=PrerollBuffer(source.source_format(), config.camera.preroll_ms),
        wake_event_repo=getattr(app_.state, "wake_event_repo", None),
        # Plan 09-06/09-07: the browser listener gets the identical
        # follow-up window the camera does -- both run through
        # `SourceRunner`, both reading the same live `app.state.
        # follow_up_window_s` (resolved once at boot, overwritten by
        # `PUT /api/settings/follow-up-window`).
        follow_up_window_s=lambda: websocket.app.state.follow_up_window_s,
        follow_up_echo_tail_s=config.follow_up.echo_tail_ms / 1000,
    )
    camera_runners = getattr(app_.state, "source_runners", None) or []
    if camera_runners:
        runner.set_wake_threshold(camera_runners[0].wake_threshold)
    await source.send_event({"type": "listen.ready", "wake_phrase": config.wake.phrase})
    try:
        await runner.run()
    finally:
        detector.close()


@app.websocket(
    "/ws/edge",
    dependencies=[Depends(require_setup_complete)],
)
async def edge_ws(websocket: WebSocket, device: EdgeDevice = Depends(require_edge_device)) -> None:
    """The Pi's own connection (D-01, D-02). Authenticated by a hashed
    edge device token, never a user session -- `require_edge_device`
    (`auth/edge_tokens.py`) reads the `Authorization` header and refuses
    an unknown or revoked token before this route ever calls `accept()`
    (T-10-01). `/ws/edge` is exempt from `require_role` for that reason:
    a device token authenticates the caller here, not a cookie session
    (see `tests/test_auth_roles.py`'s `_ROLE_EXEMPT_PATHS` entry).

    When `app.state.edge_source` is `None` -- `audio_source` did not
    resolve to `edge` at boot -- this accepts and closes at once with
    4003 (`CLOSE_NOT_CONFIGURED`): a Pi with a valid token but a server
    still pointed at the camera gets a named refusal, not a silent hang.
    """
    edge_source = websocket.app.state.edge_source
    await websocket.accept()
    if edge_source is None:
        await websocket.close(code=CLOSE_NOT_CONFIGURED)
        return

    # Plan 10-04 (D-03): record the connect for the admin device list's
    # own `last_connected_at` column. A store error is logged by device
    # id only (T-10-02: never token material) and never ends the
    # connection -- a store that is down must not stop a house listening.
    edge_device_repo = getattr(websocket.app.state, "edge_device_repo", None)
    if edge_device_repo is not None:
        try:
            await edge_device_repo.mark_connected(device.id, at=datetime.now(timezone.utc))
        except Exception:
            logger.exception("edge device %s: mark_connected failed", device.id)

    await edge_source.serve(websocket, device)


@app.websocket(
    "/ws/sessions/live",
    dependencies=[Depends(require_setup_complete), Depends(require_role(Role.OPERATOR))],
)
async def observer_ws(websocket: WebSocket) -> None:
    """The read-only observer feed (WEB-06, D-05) -- a fan-out of the same
    stream `/ws/turn` and the camera path already publish to, not a second
    data source.

    Declared as a top-level route here, exactly like `/ws/turn` above,
    never nested inside a wrapping `APIRouter`: verified directly against
    the installed `fastapi==0.141.1` that an `APIWebSocketRoute` added
    through `include_router` loses its own `path` field, which would break
    `tests/test_auth_roles.py`'s `_flatten_routes` walk (that test's own
    docstring records the identical finding for `_IncludedRouter`). The
    same dependency pair `/ws/turn` carries -- `require_setup_complete`
    plus `require_role(Role.OPERATOR)` -- keeps this at the same trust
    level, per D-03.

    This connection carries no audio in either direction and cannot start a
    turn (D-05, T-08-21): it subscribes and forwards, nothing else. A
    binary frame closes the connection immediately; a text frame is read
    and discarded, never interpreted -- there is no path here by which a
    client message reaches a provider or a source. Two tasks run
    concurrently for exactly this reason -- one forwarding published events
    out, one watching the socket's own inbound frames for that binary
    case -- and whichever finishes first (a disconnect, or a binary frame)
    cancels the other; `finally` unsubscribes the queue from the registry
    so a closed tab never leaves one accumulating (T-08-23).
    """
    await websocket.accept()
    config: Config = websocket.app.state.config
    registry: ObserverRegistry = websocket.app.state.observer_registry

    await websocket.send_text(
        json.dumps(
            {
                "type": "observer.opened",
                "wake_phrase": config.wake.phrase,
                "sources": list(OBSERVED_SOURCE_NAMES),
            }
        )
    )

    queue = registry.subscribe()
    try:
        async def _forward_published_events() -> None:
            while True:
                event = await queue.get()
                await websocket.send_text(json.dumps(event))

        async def _watch_for_binary_or_disconnect() -> None:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    # D-05: an observer never carries audio in. Closing here
                    # is the enforcement, not a convention the handler could
                    # forget to honor.
                    #
                    # The value, not the key. The installed uvicorn emits
                    # only one of `text`/`bytes`, but an ASGI server that
                    # emits `{"bytes": None, "text": "..."}` would have
                    # closed every observer connection on its first text
                    # frame -- frames this handler's own comment below says
                    # it reads and discards.
                    await websocket.close()
                    return
                # A text frame is read and discarded -- this connection is
                # an observer, not a participant.

        forward_task = asyncio.ensure_future(_forward_published_events())
        watch_task = asyncio.ensure_future(_watch_for_binary_or_disconnect())
        try:
            await asyncio.wait({forward_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            forward_task.cancel()
            watch_task.cancel()
            with suppress(asyncio.CancelledError):
                await forward_task
            with suppress(asyncio.CancelledError):
                await watch_task
    finally:
        registry.unsubscribe(queue)


if __name__ == "__main__":
    import uvicorn

    _cfg = load_config(CONFIG_PATH)
    uvicorn.run(app, host=_cfg.server.bind_host, port=_cfg.server.port)
