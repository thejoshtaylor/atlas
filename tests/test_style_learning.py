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
