"""Provider choice over HTTP (D-01 .. D-04, PROV-01).

Same "primitives in isolation" shape `tests/test_plugin_routes.py` already
uses: a throwaway `FastAPI()` app carrying only `routes/providers.py`'s own
router, driven against `FakeProviderSelectionRepository` and
`FakeCredentialRepository` -- no real Postgres.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig, SttConfig
from spire_voice.providers.boot import ProviderSlotStatus
from spire_voice.routes.providers import router as providers_router

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


def _build_providers_app(
    security,
    account_repo,
    provider_selection_repo,
    credential_repo,
    *,
    provider_slots: "dict[str, ProviderSlotStatus] | None" = None,
    stt_api_key: str = "",
) -> FastAPI:
    app = FastAPI()
    app.state.config = SimpleNamespace(
        security=security, stt=SttConfig(api_key=stt_api_key)
    )
    app.state.account_repo = account_repo
    app.state.credential_repo = credential_repo
    app.state.provider_selection_repo = provider_selection_repo
    app.state.provider_slots = provider_slots if provider_slots is not None else {
        "stt": ProviderSlotStatus(
            slot="stt", selected="xai", active="xai", state="running", reason=None, wrapped=False
        )
    }
    app.include_router(providers_router)
    return app


def _admin_client(app, security, account_repo):
    user = asyncio.run(
        account_repo.create_user(
            email="admin@example.invalid",
            display_name="An Admin",
            password_hash="not-checked-by-this-test",
            role="admin",
        )
    )
    token = issue_access_token(user_id=user.id, role="admin", security=security)
    return TestClient(app, cookies={security.cookie_name: token})


def test_get_providers_reports_the_stt_slot_with_its_options(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    assert response.status_code == 200, response.text
    body = response.json()

    [slot] = body["slots"]
    assert slot["slot"] == "stt"
    assert slot["label"] == "Speech to text"
    assert slot["selected"] == "xai"
    assert slot["active"] == "xai"
    assert slot["state"] == "running"
    assert slot["reason"] is None
    assert slot["wrapped"] is False
    assert slot["measured_ms"] is None
    assert slot["settings"] == {}
    [option] = slot["options"]
    assert option["name"] == "xai"
    assert option["label"] == "xAI"
    assert option["requires_credential"] is True


def test_get_providers_reports_credential_set_from_the_credential_repository(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    from spire_voice.crypto.credentials import encrypt_credential

    ciphertext, key_version = encrypt_credential("a-real-secret-token", security)
    asyncio.run(
        credential_repo.upsert_credential(
            "stt_api_key", ciphertext=ciphertext, key_version=key_version, updated_by_user_id=None
        )
    )

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    [slot] = response.json()["slots"]
    [option] = slot["options"]
    assert option["credential_set"] is True
    assert "a-real-secret-token" not in response.text


def test_get_providers_reports_credential_unset_when_nothing_is_stored(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    [slot] = response.json()["slots"]
    [option] = slot["options"]
    assert option["credential_set"] is False


def test_put_providers_saves_a_new_choice_and_a_reload_finds_it_still_chosen(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.put("/api/providers", json={"slots": {"stt": {"provider_name": "xai"}}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["applies_live"] is False
    [slot] = body["slots"]
    assert slot["selected"] == "xai"

    reload_response = client.get("/api/providers")
    [reloaded_slot] = reload_response.json()["slots"]
    assert reloaded_slot["selected"] == "xai"


def test_put_providers_is_admin_only(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)

    user = asyncio.run(
        account_repo.create_user(
            email="operator@example.invalid",
            display_name="An Operator",
            password_hash="not-checked-by-this-test",
            role="operator",
        )
    )
    token = issue_access_token(user_id=user.id, role="operator", security=security)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get("/api/providers")
    assert response.status_code == 403


def test_get_providers_reports_selected_and_active_diverging_after_a_stored_change(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """D-02: a row changed after boot must report `selected` as the new
    stored name and `active` as the name the process actually built its
    client from at last boot -- the divergence itself is what a "needs
    restart" badge is built from, and this route must never paper over it
    by reporting only one of the two facts."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    # The database row now says something the boot never saw -- there is
    # only one registered "stt" provider today ("xai"), so this simulates
    # the shape a second registered provider would produce once plan 07-02
    # adds one, without waiting on that plan to prove the route's own
    # selected/active honesty.
    asyncio.run(
        provider_selection_repo.set_selection(
            "stt", "a-name-the_boot_never_saw", {}, updated_by_user_id=None, updated_at=datetime.now(timezone.utc)
        )
    )

    app = _build_providers_app(
        security,
        account_repo,
        provider_selection_repo,
        credential_repo,
        provider_slots={
            "stt": ProviderSlotStatus(
                slot="stt", selected="xai", active="xai", state="running", reason=None, wrapped=False
            )
        },
    )
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    [slot] = response.json()["slots"]
    assert slot["selected"] == "a-name-the_boot_never_saw", "selected must be a fresh repository read"
    assert slot["active"] == "xai", "active must be what the process actually built, unchanged by the new row"


def test_get_providers_reports_a_degraded_slot_honestly(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(
        security,
        account_repo,
        provider_selection_repo,
        credential_repo,
        provider_slots={
            "stt": ProviderSlotStatus(
                slot="stt",
                selected="xai",
                active=None,
                state="degraded",
                reason="xAI needs an API key -- add one in Settings.",
                wrapped=False,
            )
        },
    )
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    [slot] = response.json()["slots"]
    assert slot["state"] == "degraded"
    assert slot["active"] is None
    assert slot["reason"] == "xAI needs an API key -- add one in Settings."
