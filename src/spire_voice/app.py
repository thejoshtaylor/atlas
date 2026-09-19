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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from zoneinfo import ZoneInfo

import httpx
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from spire_voice.audio.ring import PrerollBuffer
from spire_voice.auth.dependencies import Role, require_role, require_setup_complete
from spire_voice.auth.tokens import validate_secret_key_strength
from spire_voice.calibration.record import EchoCalibration
from spire_voice.calibration.runner import find_latest_calibration, run_echo_calibration
from spire_voice.config import (
    MACROS_KEY_REJECTED_ERROR,
    MCP_KEY_REJECTED_ERROR,
    SAFETY_KEY_REJECTED_ERROR,
    Config,
    ConfigError,
    DatabaseConfig,
    SecurityConfig,
    WakeConfig,
    load_config,
    load_raw_config,
)
from spire_voice.crypto.credentials import CredentialSlot, resolve_credential_value
from spire_voice.db.engine import build_engine, get_current_revision, run_migrations
from spire_voice.db.postgres import (
    PostgresAccountRepository,
    PostgresCredentialRepository,
    PostgresMacroRepository,
    PostgresPluginRepository,
    PostgresPolicyRepository,
    PostgresSettingsRepository,
    PostgresSetupRepository,
    PostgresWorkflowRepository,
)
from spire_voice.db.repository import (
    CredentialRepository,
    MacroRepository,
    PluginRepository,
    SettingsRepository,
    WorkflowRepository,
)
from spire_voice.mcp_client import McpToolHost, McpToolHostLookup, UnknownToolError, mcp_tools_to_openai_tools
from spire_voice.plugins.manager import PluginManager
from spire_voice.policy_snapshot import safety_block_from_policy
from spire_voice.providers.stt_xai import XaiStt
from spire_voice.providers.tier_reply import FILLER_TEXT
from spire_voice.providers.tts_cache import CachedTts, precache_all
from spire_voice.providers.tts_xai import XaiTts
from spire_voice.routes import register_routers
from spire_voice.routes.wizard import resolve_audio_source
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
from spire_voice.turn.controller import _speak, run_turn
from spire_voice.wake.base import WakeDetector, WakeError
from spire_voice.wake.vosk_engine import VoskWakeDetector
from spire_voice.workflow.scheduler import WorkflowScheduler
from spire_voice.workflow.steps import execute_step
from spire_voice.workflow.summary import summarise_pending_runs
from spire_voice.workflow.tool import WorkflowToolHost

logger = logging.getLogger("spire_voice.app")

CONFIG_PATH = os.environ.get("SPIRE_CONFIG", "config/config.example.yaml")
# The repository's own `mcp/` directory -- three levels up from this file
# (src/spire_voice/app.py -> src/spire_voice -> src -> repo root -> mcp).
MCP_ROOT = Path(__file__).resolve().parents[2] / "mcp"
# `web/vite.config.ts`'s own `build.outDir` -- that file's own comment
# names this exact path as the consumer a rename there would break. Three
# levels up from this file, the same computation `MCP_ROOT` above uses,
# since `web/` lives at the repo root beside `src/`, not under it.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "web" / "dist"


@dataclass(frozen=True)
class _LegacyConfigKey:
    """One retired top-level configuration key `lifespan`'s two-stage boot
    still tolerates on exactly one boot -- the boot whose migration just
    seeded it -- before rejecting it by name on every boot after (CR-01
    fix, generalized plan 04-05, D-16).

    `name` is the raw top-level key (`"safety"`, `"macros"`); `rejected_error`
    is the exact `ConfigError` message this project has always raised for
    it, lifted to module level in `spire_voice.config` so `Config.from_config`
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

# D-01 (phase 4): the zone `_state_message` reads the current time and date
# against. `lifespan` sets this from `config.server.timezone`; `None` --
# the default, and every module-load before `lifespan` has run a single
# time -- means the process's own local zone, resolved fresh on every call
# by `_current_moment()` below rather than baked in once at import time.
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


def _state_message(states: dict[str, str], pending_runs: "tuple[Any, ...]" = ()) -> str:
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
        "Current state:",
    ]
    for entity_id, state in states.items():
        lines.append(f"- {entity_id}: {state}")
    lines.append(summarise_pending_runs(pending_runs, now))
    return "\n".join(lines)


def _make_state_fetch(tool_host: "McpToolHost | None") -> Callable[[], Any]:
    """Build the per-turn `state_fetch` factory `run_turn` awaits
    concurrently with the operator still speaking (D-15).

    This is a read, gated by `allow_read` inside the MCP child -- a denied
    entity's state is still returned and injected into the prompt. SAFE-02
    is deliberate here, not an oversight: a question about a denied entity
    is exactly the case that must keep working. Reuses `_tool_result_json`
    rather than re-implementing the MCP payload walk a second time.

    Plan 06-01: `tool_host` is `None` when the plugin that enforces the
    house policy (`app.state.tool_host`, `PluginManager.enforcing_host`)
    is disabled or failed to start -- an absent host returns an empty
    entity list rather than raising an attribute error, the same
    "unknown reads as nothing known" posture `_state_message` already
    gives an empty `states` mapping.
    """

    async def _fetch() -> list[dict[str, Any]]:
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
        # Plan 05-01: scheduled workflow runs and steps (D-01 .. D-04).
        "workflow_repo": PostgresWorkflowRepository(
            sessionmaker,
            max_attempts=config.workflow.max_attempts,
            retry_backoff_s=config.workflow.retry_backoff_s,
            claim_recovery_after_s=config.workflow.claim_recovery_after_s,
        ),
    }


async def _resolve_and_log_credential(
    slot: CredentialSlot, config: Config, credential_repo: CredentialRepository
) -> str:
    """`resolve_credential_value`, plus the one required log line: which
    source won, by slot name only, never by value (T-03-43). The one
    place `lifespan` asks "what is this credential, really" -- see
    `spire_voice.crypto.credentials`'s own module docstring for why the
    database-vs-environment decision itself lives there, not here."""
    value, source = await resolve_credential_value(slot, config, credential_repo)
    logger.info("credential slot %s resolved from %s", slot.value, source)
    return value


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

    `Macro` (`spire_voice.db.repository`) is duck-type compatible with
    `spire_voice.config.MacroConfig` -- `match()`/`fire_macro()`
    (`turn/macros.py`) need no change to accept either.
    """
    macro_repo: MacroRepository = app.state.macro_repo
    return tuple(await macro_repo.list_macros())


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
            app.state.tool_host_lookup,
            app.state.tools_schema,
            app.state.catalog_prompt,
            config.brain.max_tool_rounds,
            timings,
            config.stt.max_utterance_s,
            tiers=app.state.tier_brains,
            filler_after_ms=config.brain.filler_after_ms,
            filler_cache=app.state.filler_cache,
            macros=await _current_macros(app),
            state_fetch=_make_state_fetch(app.state.tool_host),
            pending_runs_fetch=_make_pending_runs_fetch(app.state.workflow_repo),
            session_recorder=session_recorder,
            speech_lock=app.state.speaker_lock,
            workflow_tool_host=app.state.workflow_tool_host,
            tool_owners=app.state.plugin_manager.owners_of_bare_name,
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

    # D-01 (phase 4): resolve the zone `_state_message` reads the current
    # time and date against, and log which one won, by name -- an
    # unlogged, implicit container zone is how a house ends up told the
    # wrong time with nothing to point at. `ServerConfig.from_config`
    # already validated a configured name against `zoneinfo`, so this
    # `ZoneInfo(...)` call cannot raise here.
    global _resolved_timezone
    if config.server.timezone:
        _resolved_timezone = ZoneInfo(config.server.timezone)
        logger.info("resolved timezone: %s (from server.timezone)", config.server.timezone)
    else:
        _resolved_timezone = None
        logger.info(
            "resolved timezone: %s (server.timezone not set; using the process's own zone)",
            _resolved_timezone_name(),
        )
    # Plan 05-04: the one process-wide reading of the house's own
    # configured zone, exposed on `app.state` so `routes/workflows.py`
    # can hand it to `workflow.schedule.resolve_schedule` for a
    # zone-less "Run at" string, without resolving a zone name itself
    # (that module's own acceptance criterion forbids it from importing
    # `ZoneInfo`). The same `_resolved_timezone` `WorkflowToolHost` below
    # is already constructed with -- one resolved zone, two readers.
    app.state.server_timezone = _resolved_timezone

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

    # Every provider credential is resolved exactly once, here, in the one
    # order D-07 states: the database wins when an operator has saved a
    # value through the webapp, the configuration file's own environment
    # expansion wins otherwise, and an unset slot resolves to an empty
    # string -- never a fresh env read, never a per-provider conditional
    # (see `spire_voice.crypto.credentials`'s own module docstring for why
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
    resolved_ha_token = await _resolve_and_log_credential(
        CredentialSlot.HOME_ASSISTANT, config, credential_repo
    )

    app.state.stt = XaiStt(replace(config.stt, api_key=resolved_stt_key))
    tier_brains = brain_race.build_tiers(replace(config.brain, api_key=resolved_brain_key))
    app.state.tier_brains = tier_brains
    # Nothing else in this file reads `app.state.brain` today, but it stays
    # pointed at the top tier's `XaiBrain` so any future reader keeps seeing
    # the one that actually reaches Home Assistant.
    app.state.brain = tier_brains[-1].brain
    app.state.tts = XaiTts(replace(config.tts, api_key=resolved_tts_key))

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

    plugin_manager = PluginManager(
        plugin_repo,
        mcp_root=MCP_ROOT,
        security=security_config,
        safety_block_provider=_current_safety_block,
        plugins_config=config.plugins,
        on_rebuild=_publish_tool_view,
    )
    app.state.plugin_manager = plugin_manager
    # Publishes an empty view first, so every attribute above exists even
    # if `start_all()` raises, and then the real one from `start_all()`'s
    # own closing `rebuild()`.
    _publish_tool_view()
    await plugin_manager.start_all()
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
    # against the browser sink -- Phase 1's only consumer. A macro reply
    # missing from this list would raise at turn time instead of here
    # (Pitfall 4, 01.1-RESEARCH.md), the exact silent-REST-call regression
    # this precache step exists to prevent -- unchanged by the move from
    # `config.macros` to the database (D-12): the precache source changed,
    # the guarantee it gives did not. A failure here propagates uncaught,
    # matching this function's existing posture toward `tool_host.start()`:
    # a broken startup should stop the process, not start it
    # half-configured with a filler (or macro-reply) path that will fall
    # over on the first turn.
    filler_phrases = [*FILLER_TEXT.values(), *config.tts.precache, *(m.reply for m in seeded_macros)]
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
    speaker_writer = FifoWriter(config.speaker.fifo_path, reopen_timeout_s=config.speaker.reopen_timeout_s)
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
    camera_source = CameraAudioSource(config.camera, speaker_writer, on_reconnect=ffmpeg_supervisor.handle_reconnect)
    camera_source.start()
    app.state.camera_source = camera_source

    # Plan 02-11 Task 3: the single in-flight guard the run route below
    # checks before calling run_echo_calibration against these same
    # camera_source/speaker_writer resources -- two probes playing at once
    # would measure each other (T-02-51).
    app.state.calibration_in_progress = False

    # CR-01 fix (code review): every one of these used to be omitted, which
    # left `SourceRunner.__init__`'s own "absent configuration" fallbacks in
    # force for the one runner the application actually builds -- no
    # refractory window, no gate, no pre-roll replay, and barge-in forced to
    # `BargeInConfig(enabled=False)` regardless of what `config.example.yaml`
    # said. `wake_config`/`gate_config`/`barge_in_config` are the *global*
    # `Config` sections, not pre-resolved -- `SourceRunner.__init__` itself
    # calls `.resolve("camera")` on each, the same way `wake_detector` above
    # already resolves `config.wake` for engine selection. `is_media_playing`
    # is left at its default (`None`, resolving to "nothing is ever playing"
    # inside `WakeGate`): no real Home-Assistant-backed implementation exists
    # yet, and `config.example.yaml`'s own `gate.mute_when_playing` ships
    # empty, so the callable is never actually reached with the shipped
    # default (`wake/gate.py`'s own short-circuit). Wiring a live one is
    # future work, not something this fix pass invents untested.
    preroll = PrerollBuffer(camera_source.source_format(), config.camera.preroll_ms)

    # Plan 02-12 Task 3: the startup refusal CR-02 left as a gap. Turning
    # `correlation_enabled` on for a source with no valid, non-stale
    # calibration on file must stop the process by name, never fall back to
    # the guard-window-plus-floor gate the code review already found
    # cannot tell the assistant's own voice from the operator's -- that
    # fallback is exactly the silent failure this refusal exists to
    # replace with a loud one. `find_latest_calibration` never raises for
    # "no calibration directory yet" or "directory exists but empty" (its
    # own docstring): both read as `None` here, the same "missing" case.
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

    camera_runner = SourceRunner(
        "camera",
        camera_source,
        wake_detector,
        camera_source.decode_for_detector,
        _make_run_turn_for_source(app, config),
        wake_config=config.wake,
        gate_config=config.gate,
        barge_in_config=config.barge_in,
        preroll=preroll,
        calibration=camera_calibration,
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
        speaking_tts = (
            CachedTts(app.state.filler_cache) if text in app.state.filler_cache else app.state.tts
        )
        await _speak(
            app.state.camera_source,
            speaking_tts,
            TurnTimings(),
            text,
            kind="answer",
            speech_lock=app.state.speaker_lock,
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

    yield

    # CR-03 fix (code review): `SourceRunner.run()` now contains a
    # per-chunk exception rather than letting it end the task (see that
    # method's own docstring), but a task can still end with a stored
    # exception from somewhere this loop cannot anticipate (a bug in
    # `run_turn` itself, say). `return_exceptions=True` is what keeps that
    # possibility from mattering here: `task.cancel()` on an already-done
    # task is a no-op, and `await task` on one that ended with a
    # non-cancellation exception used to re-raise it, aborting every
    # cleanup call below (`camera_source.close()`, `ffmpeg_supervisor.
    # stop()`, `retention_scheduler.stop()`, `speaker_writer.close()`,
    # `plugin_manager.stop_all()`) and leaving the ffmpeg child, every MCP
    # child, and the FIFO's open handle behind uncleanly on process exit.
    for task in app.state.source_runner_tasks:
        task.cancel()
    await asyncio.gather(*app.state.source_runner_tasks, return_exceptions=True)
    await camera_source.close()
    wake_detector.close()
    await ffmpeg_supervisor.stop()
    await retention_scheduler.stop()
    await workflow_scheduler.stop()
    await speaker_http_client.aclose()
    await speaker_writer.close()
    # Plan 06-01: the manager owns every plugin child's teardown now --
    # one call, not a per-host `aclose()` for however many plugins happen
    # to be running.
    await plugin_manager.stop_all()
    await db_engine.dispose()


# `require_setup_complete` (WEB-01, D-08) is registered here, as an
# application-level dependency, rather than repeated on each route below --
# a route added in a later phase inherits the gate by default this way,
# which is the whole point: a gate a future route can forget to add is not
# a gate. `SETUP_GATE_EXEMPT_PATHS` (auth/dependencies.py) is the complete,
# named exception list; every other route in this process, present or
# future, is behind it. It is typed on `HTTPConnection`, not `Request`
# (see that module's own docstring), so it applies uniformly to the
# WebSocket turn route below as well as every HTTP route.
app = FastAPI(lifespan=lifespan, dependencies=[Depends(require_setup_complete)])
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
# must never raise merely because the directory does not exist yet.
app.frontend("/", directory=str(FRONTEND_DIR), check_dir=False)


@app.get("/health")
async def health() -> dict[str, str]:
    """Exempt from the setup gate by name (`SETUP_GATE_EXEMPT_PATHS`) --
    a gate that blocks the one route naming whether the process is even up
    is a lockout, not a safeguard. Carries no role requirement either: a
    health check answering only to an authenticated caller is not a health
    check a container orchestrator or a load balancer can use."""
    return {"status": "ok"}


@app.get("/transport")
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
    It still sits behind the application-level setup gate above, like
    every other route not named in `SETUP_GATE_EXEMPT_PATHS`.
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
    }


@app.get("/calibration/echo-path", dependencies=[Depends(require_role(Role.OPERATOR))])
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


@app.post("/calibration/echo-path/run", dependencies=[Depends(require_role(Role.OPERATOR))])
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
    if app.state.calibration_in_progress:
        raise HTTPException(status_code=409, detail="a calibration run is already in progress")

    app.state.calibration_in_progress = True
    try:
        result = await run_echo_calibration(
            app.state.camera_source,
            app.state.speaker_writer,
            config.camera,
            config.calibration,
            payload.placement_note,
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


@app.post("/webrtc/offer", dependencies=[Depends(require_role(Role.OPERATOR))])
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
    task = asyncio.create_task(
        _run_webrtc_turn(
            transport,
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
            filler_cache=app.state.filler_cache,
            macros=macros,
            state_fetch=_make_state_fetch(app.state.tool_host),
            pending_runs_fetch=_make_pending_runs_fetch(app.state.workflow_repo),
            session_recorder=SessionRecorder(config.session, timings),
            workflow_tool_host=app.state.workflow_tool_host,
            tool_owners=app.state.plugin_manager.owners_of_bare_name,
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


@app.websocket("/ws/turn", dependencies=[Depends(require_role(Role.OPERATOR))])
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
        state_fetch=_make_state_fetch(websocket.app.state.tool_host),
        pending_runs_fetch=_make_pending_runs_fetch(websocket.app.state.workflow_repo),
        session_recorder=SessionRecorder(config.session, timings),
        workflow_tool_host=websocket.app.state.workflow_tool_host,
        tool_owners=websocket.app.state.plugin_manager.owners_of_bare_name,
    )


if __name__ == "__main__":
    import uvicorn

    _cfg = load_config(CONFIG_PATH)
    uvicorn.run(app, host=_cfg.server.bind_host, port=_cfg.server.port)
