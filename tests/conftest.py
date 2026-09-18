"""Shared test fixtures: fake providers, a fake audio source, a fake Home Assistant.

Every turn-pipeline test in this phase runs against these fakes instead of a
real network call. `fake_ha` is the one Home Assistant this project's tests
ever talk to; every entity id in it is invented, following the rule
`safety.py`'s own self-check states: no real house appears in this file.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, AsyncIterator, Sequence

import httpx
import pytest
import pytest_asyncio

from spire_mcp.safety import Policy
from spire_voice.db.repository import (
    Credential,
    Invite,
    PolicyRule,
    RefreshToken,
    Setting,
    SetupStep,
    User,
)
from spire_voice.transports.base import SourceFormat


@dataclass
class PartialTranscript:
    """A speech-to-text event carrying an in-progress transcript."""

    text: str


@dataclass
class FinalTranscript:
    """A speech-to-text event carrying the finished transcript.

    `text == ""` is the empty-transcript case (VOICE-08): the operator
    toggled the microphone and said nothing intelligible.
    """

    text: str


@dataclass
class ToolCall:
    """One tool call the language model asked for."""

    name: str
    arguments: dict


@dataclass
class BrainReply:
    """One `chat()` result: zero or more tool calls, and/or reply text."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""


class FakeStt:
    """Replays a scripted sequence of transcript events, then stops -- or hangs.

    Three constructible modes, all through the same `events`/`hang` pair:
    scripted events ending in a real `FinalTranscript` (the happy path),
    scripted events ending in an empty `FinalTranscript` (VOICE-08's first
    case -- something arrived, but it decoded to nothing), and `hang=True`
    (VOICE-08's second case -- the provider never sends a
    `transcript.partial`/`transcript.done` event at all). `events=()` alone
    still exhausts immediately rather than hanging; only `hang=True` actually
    suspends forever, which is what makes it a faithful stand-in for a real
    socket that a client-side timeout -- not a server event -- must close.
    """

    def __init__(self, events: Sequence[object] = (), hang: bool = False) -> None:
        self._events = list(events)
        self._hang = hang

    async def stream(self, frames, source_format: SourceFormat | None = None) -> AsyncIterator[object]:
        # `frames` and `source_format` are accepted and ignored: this fake
        # replays its scripted events regardless of what audio it was handed.
        for event in self._events:
            yield event
        if self._hang:
            # Never resolves on its own -- only cancellation (the
            # controller's timeout guard) or garbage collection ends this.
            await asyncio.Event().wait()


@pytest.fixture
def fake_stt():
    """Factory: `fake_stt(events=[...])` builds a scripted `FakeStt`."""
    return FakeStt


class RecordingFakeStt:
    """Like `FakeStt`, but actually drains `frames` and records every chunk
    it received, byte-identical, before yielding its scripted events.

    `FakeStt` "accepts and ignores" whatever audio it is handed -- it can
    never prove VOICE-03 (nothing reaches the transcriber before a wake
    hit) or PROV-07 (what it does receive is the camera's raw bytes, not a
    decoded form), because it never touches the frames it is given.
    `tests/test_room_tracer.py` uses this instead, precisely because it
    IS a real consumer of `frames`.
    """

    def __init__(self, events: Sequence[object] = ()) -> None:
        self._events = list(events)
        self.received: list[bytes] = []

    async def stream(self, frames, source_format: SourceFormat | None = None) -> AsyncIterator[object]:
        async for chunk in frames:
            self.received.append(chunk)
        for event in self._events:
            yield event


@pytest.fixture
def recording_fake_stt():
    """Factory: `recording_fake_stt(events=[...])` builds a `RecordingFakeStt`."""
    return RecordingFakeStt


@dataclass
class FakeWakeDetector:
    """A `WakeDetector` that fires on a configurable chunk index.

    `fire_at_call=0` (the default) fires on the very first chunk it is
    handed -- the shape `test_room_tracer.py` wants, so nearly the whole
    fixture stream is left for `RecordingFakeStt` to receive and prove
    byte identity against. `score` is the fixed value every hit reports.
    """

    fire_at_call: int = 0
    score: float = 1.0
    calls: int = field(default=0, init=False)
    closed: bool = field(default=False, init=False)

    def process(self, chunk: bytes) -> "FakeWakeHit | None":
        call_index = self.calls
        self.calls += 1
        if call_index != self.fire_at_call:
            return None
        return FakeWakeHit(score=self.score)

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeWakeHit:
    """Structurally identical to `spire_voice.wake.base.WakeHit`."""

    score: float


class FakeBrain:
    """A streaming, tool-calling language model fake.

    Each call to `chat()` consumes and returns the next scripted
    `BrainReply`. Calling `chat()` more times than replies were scripted
    raises, so a test proving the `max_tool_rounds` cap can construct a
    reply list shorter than the rounds it expects the pipeline to attempt.
    """

    def __init__(self, replies: Sequence[BrainReply] = ()) -> None:
        self._replies = list(replies)
        self.call_count = 0

    async def chat(self, messages, tools=None) -> BrainReply:
        if self.call_count >= len(self._replies):
            raise AssertionError("FakeBrain.chat called more times than scripted")
        reply = self._replies[self.call_count]
        self.call_count += 1
        return reply


@pytest.fixture
def fake_brain():
    """Factory: `fake_brain(replies=[...])` builds a scripted `FakeBrain`."""
    return FakeBrain


class FakeTts:
    """Yields a scripted list of audio chunks; records the text it received."""

    def __init__(self, chunks: Sequence[bytes] = ()) -> None:
        self._chunks = list(chunks)
        self.received_text: list[str] = []

    async def synthesize(self, text_deltas) -> AsyncIterator[bytes]:
        async for delta in text_deltas:
            self.received_text.append(delta)
        for chunk in self._chunks:
            yield chunk


@pytest.fixture
def fake_tts():
    """Factory: `fake_tts(chunks=[...])` builds a scripted `FakeTts`."""
    return FakeTts


class FakeAudioSource:
    """Satisfies the transport protocol: yields a fixed list of PCM16 frames.

    Transport-independent, per D-02 -- a test drives the turn pipeline
    through this fake without caring whether WebSocket or WebRTC is live.
    """

    def __init__(self, frames: Sequence[bytes] = ()) -> None:
        self._frames = list(frames)
        self.sent_audio: list[bytes] = []

    async def frames(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    def source_format(self) -> SourceFormat:
        """Every existing test scripts 16 kHz mono PCM16 frames; this fake
        declares exactly that, matching both real browser transports."""
        return SourceFormat("pcm", 16000)


@pytest.fixture
def fake_audio_source():
    """Factory: `fake_audio_source(frames=[...])` builds a `FakeAudioSource`."""
    return FakeAudioSource


class FakeEnvelopeClient:
    """A fake `instructor`-wrapped client for a `brain_race.TierBrain`.

    Only `.chat.completions.create(...)` exists -- the one method
    `run_triage_tier`/`run_top_tier` actually call. Returns a scripted
    `TierReply`, optionally after a real `asyncio.sleep`, so a test can make
    one tier "return later" than another without a real network call.
    Records every call's keyword arguments, so a test can assert a `tools`
    keyword never reached a triage tier's client.
    """

    def __init__(self, reply: object, delay_s: float = 0.0) -> None:
        self._reply = reply
        self._delay_s = delay_s
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> object:
        # D-05's structural guarantee, enforced here rather than only by
        # inspection: an envelope call -- triage or top tier -- never
        # carries a `tools` keyword. `run_top_tier`'s tool rounds go through
        # `tool_host.call_tool`, never through this client.
        assert "tools" not in kwargs, "an envelope call must never carry a 'tools' keyword (D-05)"
        self.calls.append(kwargs)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return self._reply


@pytest.fixture
def fake_envelope_client():
    """Factory: `fake_envelope_client(reply=..., delay_s=...)` builds one."""
    return FakeEnvelopeClient


# Every entity id below is invented. No real house appears in this file,
# matching the convention `mcp/spire_mcp/safety.py::_demo` already states.
_FAKE_STATES: dict[str, dict] = {
    "switch.example_fan": {
        "entity_id": "switch.example_fan",
        "state": "off",
        "attributes": {},
        "last_changed": "2026-01-01T00:00:00+00:00",
        "last_updated": "2026-01-01T00:00:00+00:00",
    },
    "switch.example_server_socket": {
        "entity_id": "switch.example_server_socket",
        "state": "on",
        "attributes": {},
        "last_changed": "2026-01-01T00:00:00+00:00",
        "last_updated": "2026-01-01T00:00:00+00:00",
    },
    "sensor.example_server_power": {
        "entity_id": "sensor.example_server_power",
        "state": "42.0",
        "attributes": {"unit_of_measurement": "W"},
        "last_changed": "2026-01-01T00:00:00+00:00",
        "last_updated": "2026-01-01T00:00:00+00:00",
    },
    "light.example_lamp": {
        "entity_id": "light.example_lamp",
        "state": "off",
        "attributes": {},
        "last_changed": "2026-01-01T00:00:00+00:00",
        "last_updated": "2026-01-01T00:00:00+00:00",
    },
}


class FakeHomeAssistant:
    """An `httpx.MockTransport` handler plus the `AsyncClient` built on it.

    Serves `GET /api/states`, `GET /api/states/{entity_id}`, and
    `POST /api/services/{domain}/{service}` with the response shapes
    RESEARCH.md section 5 documents. `requests` records every request this
    fake received, so a test can assert its length is 0 for a denied call.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.client = httpx.AsyncClient(
            base_url="http://ha.invalid",
            transport=httpx.MockTransport(self._handle),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if request.method == "GET" and path == "/api/states":
            return httpx.Response(200, json=list(_FAKE_STATES.values()))

        if request.method == "GET" and path.startswith("/api/states/"):
            entity_id = path.removeprefix("/api/states/")
            state = _FAKE_STATES.get(entity_id)
            if state is None:
                return httpx.Response(404, json={"message": "Entity not found"})
            return httpx.Response(200, json=state)

        if request.method == "POST" and path.startswith("/api/services/"):
            _, _, _, domain, service = path.split("/", 4)
            body = request.read()
            payload = json.loads(body) if body else {}
            raw_entity_id = payload.get("entity_id")
            # `handle_call_service` posts the whole checked entity id list,
            # not a single id (plan 03-04: an expanded area target can
            # carry several) -- accept both shapes so a single-entity call
            # and an expanded, multi-entity call both get a real reply
            # rather than a `dict.get` on an unhashable list.
            if isinstance(raw_entity_id, list):
                requested_ids = raw_entity_id
            elif raw_entity_id:
                requested_ids = [raw_entity_id]
            else:
                requested_ids = []
            states = [_FAKE_STATES[e] for e in requested_ids if e in _FAKE_STATES]
            return httpx.Response(200, json=states)

        return httpx.Response(404, json={"message": "not found"})

    async def aclose(self) -> None:
        await self.client.aclose()


@pytest_asyncio.fixture
async def fake_ha():
    ha = FakeHomeAssistant()
    yield ha
    await ha.aclose()


class FakePolicyRepository:
    """An in-memory `PolicyRepository` (`spire_voice.db.repository`) --
    the one Postgres-free implementation D-04 ("the suite runs with no
    Postgres reachable") requires.

    Structurally satisfies the `PolicyRepository` protocol without
    inheriting from it. Constructed with a starting `mode` and four rule
    lists, mirroring `Policy.from_config`'s own raw-dict keyword shape so a
    test can build one from the same literal it would hand to
    `Policy.from_config`. `record_audit` appends to `audit_log` rather than
    discarding the call, so a test can assert on what was recorded.
    """

    def __init__(
        self,
        *,
        mode: str = "allow_all_except_denylist",
        deny_entities: Sequence[str] = (),
        deny_patterns: Sequence[str] = (),
        allow_entities: Sequence[str] = (),
        allow_patterns: Sequence[str] = (),
    ) -> None:
        self.mode = mode
        self._next_rule_id = 1
        self.rules: list[PolicyRule] = []
        self._add_rules("deny_entity", deny_entities)
        self._add_rules("deny_pattern", deny_patterns)
        self._add_rules("allow_entity", allow_entities)
        self._add_rules("allow_pattern", allow_patterns)
        self.audit_log: list[dict] = []

    def _add_rules(self, kind: str, values: Sequence[str]) -> None:
        for value in values:
            self.rules.append(
                PolicyRule(
                    id=self._next_rule_id,
                    kind=kind,
                    value=value,
                    note=None,
                    created_at=datetime.now(timezone.utc),
                    created_by_user_id=None,
                )
            )
            self._next_rule_id += 1

    async def load_policy(self) -> Policy:
        return Policy.from_db_rows(
            mode=self.mode,
            deny_entities=[r.value for r in self.rules if r.kind == "deny_entity"],
            deny_patterns=[r.value for r in self.rules if r.kind == "deny_pattern"],
            allow_entities=[r.value for r in self.rules if r.kind == "allow_entity"],
            allow_patterns=[r.value for r in self.rules if r.kind == "allow_pattern"],
        )

    async def list_rules(self) -> list[PolicyRule]:
        return list(self.rules)

    async def record_audit(self, action: str, detail: dict, actor_user_id: int | None) -> None:
        self.audit_log.append(
            {"action": action, "detail": detail, "actor_user_id": actor_user_id}
        )

    async def add_rule(
        self, *, kind: str, value: str, note: str | None, created_by_user_id: int | None
    ) -> PolicyRule:
        rule = PolicyRule(
            id=self._next_rule_id,
            kind=kind,
            value=value,
            note=note,
            created_at=datetime.now(timezone.utc),
            created_by_user_id=created_by_user_id,
        )
        self.rules.append(rule)
        self._next_rule_id += 1
        return rule

    async def remove_rule(self, rule_id: int) -> None:
        self.rules = [r for r in self.rules if r.id != rule_id]

    async def set_mode(self, mode: str, *, updated_by_user_id: int | None) -> None:
        self.mode = mode


@pytest.fixture
def fake_policy_repository():
    """Factory: `fake_policy_repository(mode=..., deny_entities=[...])`
    builds a scripted `FakePolicyRepository`."""
    return FakePolicyRepository


class FakeAccountRepository:
    """An in-memory `AccountRepository` (`spire_voice.db.repository`) --
    the Postgres-free implementation D-04's "the suite runs with no
    Postgres reachable" requires, matching `FakePolicyRepository`'s own
    precedent above exactly.

    Structurally satisfies the `AccountRepository` protocol without
    inheriting from it. `rotate_refresh_token`/`revoke_refresh_chain`
    reimplement the same forward-walk-through-`rotated_to_id` chain-revoke
    behavior `PostgresAccountRepository` uses, in memory, so a test against
    this fake exercises the same replay-detection semantics a test against
    real Postgres would.
    """

    def __init__(self) -> None:
        self._next_user_id = 1
        self.users: dict[int, User] = {}
        self._next_invite_id = 1
        self.invites: dict[int, Invite] = {}
        self._next_refresh_id = 1
        self.refresh_tokens: dict[int, RefreshToken] = {}

    async def any_user_exists(self) -> bool:
        return bool(self.users)

    async def create_user(
        self, *, email: str, display_name: str, password_hash: str, role: str
    ) -> User:
        user = User(
            id=self._next_user_id,
            email=email,
            display_name=display_name,
            password_hash=password_hash,
            role=role,
            created_at=datetime.now(timezone.utc),
            disabled_at=None,
        )
        self.users[user.id] = user
        self._next_user_id += 1
        return user

    async def create_user_if_no_user_exists(
        self, *, email: str, display_name: str, password_hash: str, role: str
    ) -> User | None:
        # WR-03 fix (code review): matches `PostgresAccountRepository`'s
        # atomicity contract structurally, not merely by return shape --
        # this method has no `await` between the check and the insert, so
        # under asyncio's single-threaded cooperative scheduling nothing
        # can interleave mid-body regardless (a fake never needs the real
        # implementation's lock to be equally race-free; it is race-free
        # by construction).
        if self.users:
            return None
        return await self.create_user(
            email=email, display_name=display_name, password_hash=password_hash, role=role
        )

    async def get_user_by_email(self, email: str) -> User | None:
        for user in self.users.values():
            if user.email == email:
                return user
        return None

    async def get_user_by_id(self, user_id: int) -> User | None:
        return self.users.get(user_id)

    async def list_users(self) -> list[User]:
        return list(self.users.values())

    async def disable_user(self, user_id: int) -> None:
        user = self.users.get(user_id)
        if user is not None and user.disabled_at is None:
            self.users[user_id] = replace(user, disabled_at=datetime.now(timezone.utc))

    async def create_invite(
        self,
        *,
        token_hash: str,
        role: str,
        email: str | None,
        expires_at: datetime,
        created_by_user_id: int,
    ) -> Invite:
        invite = Invite(
            id=self._next_invite_id,
            token_hash=token_hash,
            role=role,
            email=email,
            expires_at=expires_at,
            created_by_user_id=created_by_user_id,
            accepted_at=None,
            accepted_by_user_id=None,
        )
        self.invites[invite.id] = invite
        self._next_invite_id += 1
        return invite

    async def get_invite_by_token_hash(self, token_hash: str) -> Invite | None:
        for invite in self.invites.values():
            if invite.token_hash == token_hash:
                return invite
        return None

    async def claim_invite(self, invite_id: int, *, now: datetime) -> bool:
        # WR-03 fix (code review): same "no await between check and
        # write, so nothing can interleave" reasoning as
        # `create_user_if_no_user_exists` above -- race-free by
        # construction under asyncio's cooperative scheduling, matching
        # `PostgresAccountRepository.claim_invite`'s atomicity contract.
        invite = self.invites.get(invite_id)
        if invite is None or invite.accepted_at is not None or invite.expires_at <= now:
            return False
        self.invites[invite_id] = replace(invite, accepted_at=now)
        return True

    async def record_invite_acceptor(self, invite_id: int, *, accepted_by_user_id: int) -> None:
        invite = self.invites.get(invite_id)
        if invite is not None:
            self.invites[invite_id] = replace(invite, accepted_by_user_id=accepted_by_user_id)

    async def revoke_invite(self, invite_id: int, *, revoked_at: datetime) -> None:
        invite = self.invites.get(invite_id)
        if invite is not None and invite.accepted_at is None:
            self.invites[invite_id] = replace(invite, expires_at=revoked_at)

    async def list_invites(self) -> list[Invite]:
        return list(self.invites.values())

    async def store_refresh_token(
        self, *, user_id: int, token_hash: str, issued_at: datetime, expires_at: datetime
    ) -> RefreshToken:
        token = RefreshToken(
            id=self._next_refresh_id,
            user_id=user_id,
            token_hash=token_hash,
            issued_at=issued_at,
            expires_at=expires_at,
            revoked_at=None,
            rotated_to_id=None,
        )
        self.refresh_tokens[token.id] = token
        self._next_refresh_id += 1
        return token

    async def rotate_refresh_token(
        self,
        old_token_hash: str,
        *,
        new_token_hash: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> RefreshToken | None:
        old = self._find_refresh_by_hash(old_token_hash)
        if old is None:
            return None
        if old.revoked_at is not None:
            self._revoke_chain(old)
            return None

        new_token = RefreshToken(
            id=self._next_refresh_id,
            user_id=old.user_id,
            token_hash=new_token_hash,
            issued_at=issued_at,
            expires_at=expires_at,
            revoked_at=None,
            rotated_to_id=None,
        )
        self.refresh_tokens[new_token.id] = new_token
        self._next_refresh_id += 1
        self.refresh_tokens[old.id] = replace(
            old, revoked_at=datetime.now(timezone.utc), rotated_to_id=new_token.id
        )
        return new_token

    async def revoke_refresh_chain(self, token_hash: str) -> None:
        row = self._find_refresh_by_hash(token_hash)
        if row is not None:
            self._revoke_chain(row)

    def _find_refresh_by_hash(self, token_hash: str) -> RefreshToken | None:
        for token in self.refresh_tokens.values():
            if token.token_hash == token_hash:
                return token
        return None

    def _revoke_chain(self, row: RefreshToken) -> None:
        now = datetime.now(timezone.utc)
        current: RefreshToken | None = row
        while current is not None:
            if current.revoked_at is None:
                current = replace(current, revoked_at=now)
                self.refresh_tokens[current.id] = current
            next_id = current.rotated_to_id
            current = self.refresh_tokens.get(next_id) if next_id is not None else None


@pytest.fixture
def fake_account_repository():
    """Factory: `fake_account_repository()` builds an empty
    `FakeAccountRepository` -- every test populates it itself
    (`create_user`, `create_invite`, ...), matching the rest of this
    file's factory-fixture convention even though this one fake takes no
    constructor arguments."""
    return FakeAccountRepository


class FakeCredentialRepository:
    """An in-memory `CredentialRepository` (`spire_voice.db.repository`) --
    the Postgres-free implementation D-04's "the suite runs with no
    Postgres reachable" requires, matching `FakePolicyRepository`'s and
    `FakeAccountRepository`'s own precedent above exactly. Holds
    ciphertext only, the same as the real implementation -- nothing here
    ever decrypts a value.
    """

    def __init__(self) -> None:
        self.credentials: dict[str, Credential] = {}

    async def get_credential(self, slot: str) -> Credential | None:
        return self.credentials.get(slot)

    async def list_credentials(self) -> list[Credential]:
        return list(self.credentials.values())

    async def upsert_credential(
        self,
        slot: str,
        *,
        ciphertext: bytes,
        key_version: int,
        updated_by_user_id: int | None,
    ) -> Credential:
        credential = Credential(
            slot=slot,
            ciphertext=ciphertext,
            key_version=key_version,
            updated_at=datetime.now(timezone.utc),
            updated_by_user_id=updated_by_user_id,
        )
        self.credentials[slot] = credential
        return credential


@pytest.fixture
def fake_credential_repository():
    """Factory: `fake_credential_repository()` builds an empty
    `FakeCredentialRepository` -- every test populates it itself,
    matching `fake_account_repository`'s own convention."""
    return FakeCredentialRepository


class FakeSetupRepository:
    """An in-memory `SetupRepository` (`spire_voice.db.repository`) --
    the Postgres-free implementation D-04's "the suite runs with no
    Postgres reachable" requires, matching every other `Fake*Repository`
    in this file. Seeds the same five named rows the real migration
    (`0004_setup_state.py`) seeds, so a test never has to special-case "a
    step nobody has touched yet" between the fake and the real thing.
    """

    _STEP_NAMES = ("admin_account", "hub", "provider_set", "audio_source", "room")

    def __init__(self) -> None:
        self._completed_at: datetime | None = None
        self._next_step_id = 1
        self.steps: dict[str, SetupStep] = {}
        for name in self._STEP_NAMES:
            self.steps[name] = SetupStep(
                id=self._next_step_id, name=name, completed_at=None, detail=None
            )
            self._next_step_id += 1

    async def is_setup_complete(self) -> bool:
        return self._completed_at is not None

    async def mark_setup_complete(self, *, completed_at: datetime) -> None:
        self._completed_at = completed_at

    async def get_step(self, name: str) -> SetupStep | None:
        return self.steps.get(name)

    async def list_steps(self) -> list[SetupStep]:
        return list(self.steps.values())

    async def complete_step(self, name: str, *, detail: dict, completed_at: datetime) -> SetupStep:
        existing = self.steps.get(name)
        step_id = existing.id if existing is not None else self._next_step_id
        if existing is None:
            self._next_step_id += 1
        step = SetupStep(id=step_id, name=name, completed_at=completed_at, detail=detail)
        self.steps[name] = step
        return step


@pytest.fixture
def fake_setup_repository():
    """Factory: `fake_setup_repository()` builds a `FakeSetupRepository`
    pre-seeded with the same five named, incomplete steps a fresh
    migration seeds -- matching `fake_account_repository`'s own
    factory-fixture convention."""
    return FakeSetupRepository


class FakeSettingsRepository:
    """An in-memory `SettingsRepository` (`spire_voice.db.repository`) --
    the Postgres-free implementation D-04 requires, matching every other
    `Fake*Repository` in this file."""

    def __init__(self) -> None:
        self._next_id = 1
        self.settings: dict[str, Setting] = {}

    async def get_setting(self, key: str) -> Setting | None:
        return self.settings.get(key)

    async def set_setting(
        self, key: str, value, *, updated_by_user_id: int | None, updated_at: datetime
    ) -> Setting:
        existing = self.settings.get(key)
        setting_id = existing.id if existing is not None else self._next_id
        if existing is None:
            self._next_id += 1
        setting = Setting(
            id=setting_id,
            key=key,
            value=value,
            updated_at=updated_at,
            updated_by_user_id=updated_by_user_id,
        )
        self.settings[key] = setting
        return setting


@pytest.fixture
def fake_settings_repository():
    """Factory: `fake_settings_repository()` builds an empty
    `FakeSettingsRepository` -- every test populates it itself, matching
    `fake_credential_repository`'s own convention."""
    return FakeSettingsRepository
