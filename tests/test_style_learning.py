"""Plan 09-09, Task 1: a profile, up-to-five samples, and a Gmail signature
learned from one account's own Sent mail (D-19, D-20, D-22) -- proven
against `FakeGoogleAccountRepository`, `FakeGoogle`, and
`RecordingFakeBrain`, the same "primitives in isolation" shape
`tests/test_google_token_refresh.py` already uses for `GoogleTokenService`.

Task 2 extends this file with the admin-only style routes
(`GET/PUT /api/google/accounts/{id}/style`,
`POST /api/google/accounts/{id}/style/relearn`).

Every token/secret literal here is shorter than eight characters
(`"at-1"`, `"rt-1"`, `"cs-1"`) -- `tests/test_repo_hygiene.py::_CREDENTIAL_RE`
scans every tracked file and all of git history, and a mistake here is
permanent.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from google_fakes import FakeGoogle
from google_repo_fakes import FakeGoogleAccountRepository

from atlas_mcp.google_tools import REQUIRED_SCOPES

from atlas.auth.tokens import issue_access_token
from atlas.config import SecurityConfig
from atlas.crypto.credentials import encrypt_credential
from atlas.google.style import STYLE_PROFILE_INSTRUCTION, SENT_MESSAGES_TO_SCAN, learn_style
from atlas.google.token_service import GoogleTokenService
from atlas.plugins.manager import PluginState
from atlas.providers.base import BrainReply
from atlas.routes.google_accounts import router as google_accounts_router

from brain_fakes import RecordingFakeBrain

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


@pytest.fixture(autouse=True)
def _secret_key_env(monkeypatch):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)


async def _repo_with_account(security: SecurityConfig, *, label: str = "work", email: str = "work@example.com"):
    repo = FakeGoogleAccountRepository()
    secret_ciphertext, secret_key_version = encrypt_credential("cs-1", security)
    await repo.set_oauth_client(
        client_id="cid-1",
        client_secret_ciphertext=secret_ciphertext,
        key_version=secret_key_version,
        updated_by_user_id=None,
        updated_at=datetime.now(timezone.utc),
    )
    ciphertext, key_version = encrypt_credential("rt-1", security)
    account = await repo.insert_account(
        label=label,
        email=email,
        refresh_token_ciphertext=ciphertext,
        key_version=key_version,
        granted_scopes=" ".join(REQUIRED_SCOPES),
        refresh_token_expires_at=None,
        linked_by_user_id=None,
        linked_at=datetime.now(timezone.utc),
    )
    return repo, account


def _learn_style(repo, account, security, fake: FakeGoogle, brain, *, access_token: str = "at-1"):
    fake.add_refresh_token("rt-1", access_token)
    token_service = GoogleTokenService(repo, security, fake.client)
    return learn_style(
        account=account,
        repo=repo,
        token_service=token_service,
        http_client=fake.client,
        brain=brain,
        now=datetime.now(timezone.utc),
    )


# --- profile: cleaning, paging, capping ------------------------------------


async def test_cleaning_removes_quoted_and_signature_text():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    body_text = "Sounds good, see you then.\n\n> On Tue, Dana wrote:\n> What time works?\n-- \nAlex"
    fake.add_gmail_messages("at-1", ["m1"])
    fake.add_gmail_full("at-1", "m1", headers={}, text=body_text)
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    await _learn_style(repo, account, security, fake, brain)

    [call] = brain.calls
    assert call.messages[0] == {"role": "system", "content": STYLE_PROFILE_INSTRUCTION}
    assert call.messages[1] == {"role": "user", "content": "Sounds good, see you then."}
    assert call.tools is None


async def test_pages_up_to_two_hundred_sent_messages_when_more_exist():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    ids = [f"m{i}" for i in range(250)]
    fake.add_gmail_messages("at-1", ids)
    for message_id in ids:
        fake.add_gmail_full("at-1", message_id, headers={}, text="short reply, thanks!")
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    style = await _learn_style(repo, account, security, fake, brain)

    full_fetches = [
        r
        for r in fake.requests
        if "/messages/" in r.url.path and r.url.params.get("format") == "full"
    ]
    assert len(full_fetches) == SENT_MESSAGES_TO_SCAN == 200
    assert style.messages_scanned == 200
    assert style.status == "ready"


async def test_profile_round_uses_only_the_newest_hundred_bodies():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    ids = [f"m{i}" for i in range(150)]
    fake.add_gmail_messages("at-1", ids)
    for i, message_id in enumerate(ids):
        # Short bodies here, deliberately: total joined length stays well
        # under the 30,000-character cap, so this test proves the
        # newest-100 slice alone, uncomplicated by truncation.
        fake.add_gmail_full("at-1", message_id, headers={}, text=f"body-{i:04d} is a short reply.")
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    await _learn_style(repo, account, security, fake, brain)

    [call] = brain.calls
    profile_input = call.messages[1]["content"]
    assert "body-0000" in profile_input
    assert "body-0099" in profile_input
    assert "body-0100" not in profile_input


async def test_profile_input_is_capped_at_thirty_thousand_characters():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    ids = [f"m{i}" for i in range(150)]
    fake.add_gmail_messages("at-1", ids)
    for i, message_id in enumerate(ids):
        # Every body is long enough that 100 of them together exceed the
        # 30,000-character cap.
        fake.add_gmail_full("at-1", message_id, headers={}, text=f"body-{i:04d} " + "x" * 320)
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    await _learn_style(repo, account, security, fake, brain)

    [call] = brain.calls
    profile_input = call.messages[1]["content"]
    assert len(profile_input) <= 30000
    assert "body-0000" in profile_input
    assert "body-0100" not in profile_input


# --- samples -----------------------------------------------------------


async def test_samples_are_the_newest_up_to_five_bodies_in_range():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    # Newest first: too short, five good ones, then a sixth good one and
    # one too long -- only the newest five in-range bodies are kept.
    bodies = [
        "hi",  # too short (< 40 chars)
        "reply number one is a good length for a sample here.",  # 55 chars
        "reply number two is also a good length for a sample.",  # 55 chars
        "reply number three is also a good length for a sample.",
        "reply number four is also a good length for a sample here.",
        "reply number five is also a good length for a sample here!",
        "reply number six would also qualify but is the sixth in range.",
        "y" * 800,  # too long (> 700 chars)
    ]
    ids = [f"m{i}" for i in range(len(bodies))]
    fake.add_gmail_messages("at-1", ids)
    for message_id, body in zip(ids, bodies):
        fake.add_gmail_full("at-1", message_id, headers={}, text=body)
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    style = await _learn_style(repo, account, security, fake, brain)

    assert list(style.samples) == bodies[1:6]
    assert all(40 <= len(s) <= 700 for s in style.samples)


# --- signature -----------------------------------------------------------


async def test_signature_from_the_default_send_as_entry():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    fake.add_gmail_messages("at-1", [])
    fake.add_send_as(
        "at-1",
        [
            {"sendAsEmail": "other@example.com", "isPrimary": True, "signature": "<b>Other</b>"},
            {"sendAsEmail": "work@example.com", "isDefault": True, "signature": "<i>Alex</i>"},
        ],
    )
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    style = await _learn_style(repo, account, security, fake, brain)

    assert style.signature_html == "<i>Alex</i>"
    assert style.signature_text == "Alex"


async def test_signature_falls_back_to_primary_when_no_default():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    fake.add_gmail_messages("at-1", [])
    fake.add_send_as(
        "at-1",
        [{"sendAsEmail": "work@example.com", "isPrimary": True, "signature": "<i>Primary Alex</i>"}],
    )
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    style = await _learn_style(repo, account, security, fake, brain)

    assert style.signature_html == "<i>Primary Alex</i>"
    assert style.signature_text == "Primary Alex"


async def test_empty_signature_stores_nulls():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    fake.add_gmail_messages("at-1", [])
    fake.add_send_as("at-1", [{"sendAsEmail": "work@example.com", "isDefault": True, "signature": ""}])
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    style = await _learn_style(repo, account, security, fake, brain)

    assert style.signature_html is None
    assert style.signature_text is None


async def test_no_send_as_entries_stores_no_signature():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    fake.add_gmail_messages("at-1", [])
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    style = await _learn_style(repo, account, security, fake, brain)

    assert style.signature_html is None
    assert style.signature_text is None


# --- status transitions and failures --------------------------------------


async def test_status_moves_learning_to_ready_with_scanned_count_and_learned_at():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    await repo.set_style_status(account.id, "learning", None, datetime.now(timezone.utc))
    fake = FakeGoogle()
    fake.add_gmail_messages("at-1", ["m1"])
    fake.add_gmail_full("at-1", "m1", headers={}, text="a reasonably normal length reply body here.")
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    style = await _learn_style(repo, account, security, fake, brain)

    assert style.status == "ready"
    assert style.status_detail is None
    assert style.messages_scanned == 1
    assert style.learned_at is not None


async def test_no_language_model_ends_failed_with_a_detail_naming_that():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()

    style = await _learn_style(repo, account, security, fake, brain=None)

    assert style.status == "failed"
    assert "language model" in (style.status_detail or "")


async def test_a_revoked_token_ends_failed_and_never_raises():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    # No refresh token registered with the fake -> the token endpoint
    # answers invalid_grant, same as `GoogleTokenService`'s own tests.
    token_service = GoogleTokenService(repo, security, fake.client)
    brain = RecordingFakeBrain()

    style = await learn_style(
        account=account,
        repo=repo,
        token_service=token_service,
        http_client=fake.client,
        brain=brain,
        now=datetime.now(timezone.utc),
    )

    assert style.status == "failed"
    assert style.status_detail
    assert brain.call_count == 0


async def test_an_empty_model_reply_ends_failed():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    fake.add_gmail_messages("at-1", ["m1"])
    fake.add_gmail_full("at-1", "m1", headers={}, text="a reasonably normal length reply body here.")
    brain = RecordingFakeBrain(replies=[BrainReply(text="   ")])

    style = await _learn_style(repo, account, security, fake, brain)

    assert style.status == "failed"


async def test_a_gmail_list_failure_ends_failed_and_never_raises():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake = FakeGoogle()
    fake.add_refresh_token("rt-1", "at-1")
    fake.fail_gmail_list("at-1", status=500)
    token_service = GoogleTokenService(repo, security, fake.client)
    brain = RecordingFakeBrain()

    style = await learn_style(
        account=account,
        repo=repo,
        token_service=token_service,
        http_client=fake.client,
        brain=brain,
        now=datetime.now(timezone.utc),
    )

    assert style.status == "failed"
    assert style.status_detail


# --- get_style / get_style_by_label ----------------------------------------


async def test_get_style_for_a_never_learned_account_is_not_learned():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)

    style = await repo.get_style(account.id)

    assert style.status == "not_learned"
    assert style.profile == ""
    assert style.samples == ()
    assert style.signature_html is None


async def test_get_style_by_label_finds_the_same_row():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security, label="work")
    await repo.save_learned_style(
        account.id,
        profile="a friendly, brief style.",
        samples=("hello there, thanks for reaching out today.",),
        signature_html="<i>Alex</i>",
        signature_text="Alex",
        messages_scanned=5,
        learned_at=datetime.now(timezone.utc),
    )

    by_label = await repo.get_style_by_label("work")
    by_id = await repo.get_style(account.id)

    assert by_label == by_id
    assert by_label.status == "ready"


async def test_get_style_by_label_for_an_unknown_label_is_none():
    security = SecurityConfig()
    repo, _account = await _repo_with_account(security)

    assert await repo.get_style_by_label("nope") is None


# =========================================================================
# Task 2: learned on link, re-learned on request, edited by the admin
# =========================================================================


class _FakePluginManagerForGoogle:
    """`tests/test_google_account_routes.py::_FakePluginManagerForGoogle`'s
    own sibling here -- duplicated rather than imported, matching this
    codebase's own per-test-file fake convention."""

    def __init__(self) -> None:
        self.states: dict[str, PluginState] = {}
        self.start_calls: list[int] = []
        self.stop_calls: list[int] = []
        self.respawn_calls: list[int] = []

    def state_for(self, slug):
        return self.states.get(slug)

    async def start_one(self, plugin):
        self.start_calls.append(plugin.id)
        self.states[plugin.slug] = PluginState.RUNNING

    async def stop_one(self, plugin):
        self.stop_calls.append(plugin.id)
        self.states[plugin.slug] = PluginState.DISABLED

    async def request_respawn(self, plugin_id, safety_block):
        self.respawn_calls.append(plugin_id)


def _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google: FakeGoogle, *, brain=None):
    app = FastAPI()
    app.state.config = SimpleNamespace(security=security)
    app.state.account_repo = account_repo
    app.state.google_account_repo = google_repo
    app.state.google_http_client = fake_google.client
    app.state.google_token_service = GoogleTokenService(google_repo, security, fake_google.client)
    app.state.plugin_repo = plugin_repo
    app.state.plugin_manager = _FakePluginManagerForGoogle()
    app.state.brain = brain
    # Plan 09-09: the same detached-task keep-alive set `app.py`'s real
    # `lifespan` builds (`app.state.background_turns`) -- required by
    # `POST .../style/relearn` and the OAuth callback's own new-account
    # scheduling.
    app.state.background_turns = set()
    app.include_router(google_accounts_router)
    return app


def _client_as_role(app, security, account_repo, *, role: str) -> TestClient:
    user = asyncio.run(
        account_repo.create_user(
            email=f"{role}@example.invalid",
            display_name=f"A {role.title()}",
            password_hash="not-checked-by-this-test",
            role=role,
        )
    )
    token = issue_access_token(user_id=user.id, role=role, security=security)
    return TestClient(app, cookies={security.cookie_name: token}, follow_redirects=False)


def _drain_background(client: TestClient, app: FastAPI) -> None:
    """Deterministically wait for every task `app.state.background_turns`
    holds -- `client.portal` is the single event loop every request made
    through `with TestClient(app) as client:` shares (Starlette's own
    `_portal_factory` reuses `self.portal` once entered), so running an
    async waiter on that same portal lets a test wait for a route's own
    `asyncio.create_task(...)` work without racing it. A plain (not
    `async def`) function: `client.portal.call(...)` itself runs the
    coroutine synchronously, so callers -- sync test functions included --
    never need to `await` this."""

    async def _wait() -> None:
        tasks = list(app.state.background_turns)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    client.portal.call(_wait)


def _seed_style_account(google_repo, security, *, label="work", email="work@example.com"):
    ciphertext, key_version = encrypt_credential("rt-1", security)
    return asyncio.run(
        google_repo.insert_account(
            label=label,
            email=email,
            refresh_token_ciphertext=ciphertext,
            key_version=key_version,
            granted_scopes=" ".join(REQUIRED_SCOPES),
            refresh_token_expires_at=None,
            linked_by_user_id=None,
            linked_at=datetime.now(timezone.utc),
        )
    )


def _seed_style_oauth_client(google_repo, security):
    ciphertext, key_version = encrypt_credential("cs-1", security)
    asyncio.run(
        google_repo.set_oauth_client(
            client_id="cid-1",
            client_secret_ciphertext=ciphertext,
            key_version=key_version,
            updated_by_user_id=None,
            updated_at=datetime.now(timezone.utc),
        )
    )


def test_get_style_before_any_learning_is_not_learned(monkeypatch, fake_account_repository, fake_plugin_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()
    _seed_style_oauth_client(google_repo, security)
    account = _seed_style_account(google_repo, security)

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google)
    with TestClient(app) as client:
        admin = _client_as_role(app, security, account_repo, role="admin")
        response = admin.get(f"/api/google/accounts/{account.id}/style")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "not_learned"
    assert body["profile"] == ""
    assert body["samples"] == []


def test_get_style_for_unknown_account_is_404(monkeypatch, fake_account_repository, fake_plugin_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google)
    with TestClient(app) as client:
        admin = _client_as_role(app, security, account_repo, role="admin")
        response = admin.get("/api/google/accounts/999/style")

    assert response.status_code == 404


def test_put_style_stores_the_profile(monkeypatch, fake_account_repository, fake_plugin_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()
    _seed_style_oauth_client(google_repo, security)
    account = _seed_style_account(google_repo, security)

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google)
    with TestClient(app) as client:
        admin = _client_as_role(app, security, account_repo, role="admin")
        put_response = admin.put(
            f"/api/google/accounts/{account.id}/style", json={"profile": "brief and friendly."}
        )
        assert put_response.status_code == 200, put_response.text
        get_response = admin.get(f"/api/google/accounts/{account.id}/style")

    assert get_response.json()["profile"] == "brief and friendly."


def test_put_style_refuses_a_profile_over_four_thousand_characters(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()
    _seed_style_oauth_client(google_repo, security)
    account = _seed_style_account(google_repo, security)

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google)
    with TestClient(app) as client:
        admin = _client_as_role(app, security, account_repo, role="admin")
        response = admin.put(
            f"/api/google/accounts/{account.id}/style", json={"profile": "x" * 4001}
        )

    assert response.status_code == 400


def test_relearn_answers_202_learning_then_finishes_ready_in_the_background(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()
    _seed_style_oauth_client(google_repo, security)
    account = _seed_style_account(google_repo, security)
    fake_google.add_refresh_token("rt-1", "at-1")
    fake_google.add_gmail_messages("at-1", ["m1"])
    fake_google.add_gmail_full("at-1", "m1", headers={}, text="a reasonably normal length reply here.")
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google, brain=brain)
    with TestClient(app) as client:
        admin = _client_as_role(app, security, account_repo, role="admin")
        relearn_response = admin.post(f"/api/google/accounts/{account.id}/style/relearn")
        assert relearn_response.status_code == 202, relearn_response.text
        assert relearn_response.json()["status"] == "learning"

        _drain_background(client, app)

        get_response = admin.get(f"/api/google/accounts/{account.id}/style")

    body = get_response.json()
    assert body["status"] == "ready"
    assert body["profile"] == "a short, friendly style."
    assert body["messages_scanned"] == 1


def test_relearn_while_learning_is_refused_with_409(monkeypatch, fake_account_repository, fake_plugin_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()
    _seed_style_oauth_client(google_repo, security)
    account = _seed_style_account(google_repo, security)
    # Set the row `"learning"` directly, sidestepping any race with the
    # background task a first `POST .../relearn` would itself schedule
    # (its own body could finish before a second request is even issued)
    # -- this test's own job is the 409 refusal, not the scheduling race.
    asyncio.run(google_repo.set_style_status(account.id, "learning", None, datetime.now(timezone.utc)))
    brain = RecordingFakeBrain()

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google, brain=brain)
    with TestClient(app) as client:
        admin = _client_as_role(app, security, account_repo, role="admin")
        response = admin.post(f"/api/google/accounts/{account.id}/style/relearn")

    assert response.status_code == 409
    assert brain.call_count == 0


def test_relearn_with_no_language_model_ends_failed_with_a_detail(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()
    _seed_style_oauth_client(google_repo, security)
    account = _seed_style_account(google_repo, security)
    fake_google.add_refresh_token("rt-1", "at-1")

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google, brain=None)
    with TestClient(app) as client:
        admin = _client_as_role(app, security, account_repo, role="admin")
        admin.post(f"/api/google/accounts/{account.id}/style/relearn")
        _drain_background(client, app)
        response = admin.get(f"/api/google/accounts/{account.id}/style")

    body = response.json()
    assert body["status"] == "failed"
    assert "language model" in (body["status_detail"] or "")


def test_style_routes_are_403_for_viewer_and_operator(monkeypatch, fake_account_repository, fake_plugin_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()
    _seed_style_oauth_client(google_repo, security)
    account = _seed_style_account(google_repo, security)

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google)
    with TestClient(app) as client:
        for role in ("viewer", "operator"):
            non_admin = _client_as_role(app, security, account_repo, role=role)
            assert non_admin.get(f"/api/google/accounts/{account.id}/style").status_code == 403
            assert (
                non_admin.put(f"/api/google/accounts/{account.id}/style", json={"profile": "x"}).status_code
                == 403
            )
            assert non_admin.post(f"/api/google/accounts/{account.id}/style/relearn").status_code == 403


def test_a_new_link_schedules_exactly_one_background_learn_style(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()
    _seed_style_oauth_client(google_repo, security)
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short, friendly style.")])

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google, brain=brain)
    with TestClient(app) as client:
        admin = _client_as_role(app, security, account_repo, role="admin")

        start_response = admin.post(
            "/api/google/oauth/start", json={"label": "work"}, headers={"Origin": "https://atlas.example.com"}
        )
        assert start_response.status_code == 200, start_response.text
        import urllib.parse

        parsed = urllib.parse.urlsplit(start_response.json()["authorization_url"])
        raw_state = urllib.parse.parse_qs(parsed.query)["state"][0]

        fake_google.add_code("code-1", access_token="at-1", refresh_token="rt-1")
        # `add_code` seeds the authorization-code exchange only -- the
        # background `learn_style` task refreshes with the now-stored
        # refresh token afterwards, over the plain `grant_type=refresh_token`
        # path `add_refresh_token` seeds.
        fake_google.add_refresh_token("rt-1", "at-1")
        fake_google.add_profile("at-1", "work@example.com")
        fake_google.add_calendar_list("at-1", [])
        fake_google.add_gmail_messages("at-1", [])

        callback_response = admin.get("/api/google/oauth/callback", params={"code": "code-1", "state": raw_state})
        assert callback_response.status_code == 303, callback_response.text
        account_id = int(callback_response.headers["location"].split("/")[-1].split("?")[0])

        _drain_background(client, app)
        style_response = admin.get(f"/api/google/accounts/{account_id}/style")

    assert style_response.json()["status"] == "ready"
    assert brain.call_count == 1


def test_a_relink_of_an_existing_address_schedules_no_learn_style(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    fake_google = FakeGoogle()
    _seed_style_oauth_client(google_repo, security)
    account = _seed_style_account(google_repo, security, label="work", email="work@example.com")
    brain = RecordingFakeBrain()

    app = _build_style_app(security, account_repo, google_repo, plugin_repo, fake_google, brain=brain)
    with TestClient(app) as client:
        admin = _client_as_role(app, security, account_repo, role="admin")

        start_response = admin.post(
            "/api/google/oauth/start",
            json={"label": "work", "relink_account_id": account.id},
            headers={"Origin": "https://atlas.example.com"},
        )
        assert start_response.status_code == 200, start_response.text
        import urllib.parse

        parsed = urllib.parse.urlsplit(start_response.json()["authorization_url"])
        raw_state = urllib.parse.parse_qs(parsed.query)["state"][0]

        fake_google.add_code("code-1", access_token="at-1", refresh_token="rt-1")
        fake_google.add_profile("at-1", "work@example.com")
        fake_google.add_calendar_list("at-1", [])

        callback_response = admin.get("/api/google/oauth/callback", params={"code": "code-1", "state": raw_state})
        assert callback_response.status_code == 303, callback_response.text

        _drain_background(client, app)
        style_response = admin.get(f"/api/google/accounts/{account.id}/style")

    assert style_response.json()["status"] == "not_learned"
    assert brain.call_count == 0
