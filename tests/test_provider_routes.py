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
from spire_voice.config import BrainConfig, SecurityConfig, SttConfig, TtsConfig
from spire_voice.providers.boot import ProviderSlotStatus
from spire_voice.routes.providers import router as providers_router

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"

# Plan 07-02: every slot runs, unwrapped except tts, with no client
# object at all -- the shape a normal boot leaves behind when no test
# needs to read `measured_ms` off a real client.
_DEFAULT_PROVIDER_SLOTS: "dict[str, ProviderSlotStatus]" = {
    "stt": ProviderSlotStatus(
        slot="stt", selected="xai", active="xai", state="running", reason=None, wrapped=False
    ),
    "tts": ProviderSlotStatus(
        slot="tts", selected="xai", active="xai", state="running", reason=None, wrapped=True
    ),
    "brain": ProviderSlotStatus(
        slot="brain", selected="xai", active="xai", state="running", reason=None, wrapped=False
    ),
}


def _build_providers_app(
    security,
    account_repo,
    provider_selection_repo,
    credential_repo,
    *,
    provider_slots: "dict[str, ProviderSlotStatus] | None" = None,
    stt_api_key: str = "",
    tts_api_key: str = "",
    brain_api_key: str = "",
    tts_client: object = None,
) -> FastAPI:
    app = FastAPI()
    app.state.config = SimpleNamespace(
        security=security,
        stt=SttConfig(api_key=stt_api_key),
        tts=TtsConfig(api_key=tts_api_key),
        brain=BrainConfig(api_key=brain_api_key),
    )
    app.state.account_repo = account_repo
    app.state.credential_repo = credential_repo
    app.state.provider_selection_repo = provider_selection_repo
    app.state.provider_slots = (
        provider_slots if provider_slots is not None else dict(_DEFAULT_PROVIDER_SLOTS)
    )
    # `_measured_ms_for` reads `request.app.state.tts` live -- `None` here
    # (every test that does not care about `measured_ms`) is the same
    # "no client at all" state a degraded slot leaves behind, so it reads
    # as an honest `measured_ms: None` rather than erroring.
    app.state.tts = tts_client
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


def _slot(body: dict, slot_name: str) -> dict:
    [match] = [slot for slot in body["slots"] if slot["slot"] == slot_name]
    return match


def test_get_providers_reports_all_three_slots_in_the_fixed_order(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """07-UI-SPEC.md: speech to text, text to speech, language model --
    this order never changes, so returning to the page after a restart
    lands the admin's eye on the same slot in the same place."""
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

    assert [slot["slot"] for slot in body["slots"]] == ["stt", "tts", "brain"]
    assert [slot["label"] for slot in body["slots"]] == [
        "Speech to text",
        "Text to speech",
        "Language model",
    ]


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

    slot = _slot(body, "stt")
    assert slot["label"] == "Speech to text"
    assert slot["selected"] == "xai"
    assert slot["active"] == "xai"
    assert slot["state"] == "running"
    assert slot["reason"] is None
    assert slot["wrapped"] is False
    assert slot["measured_ms"] is None
    assert slot["settings"] == {}
    # 07-03-PLAN.md Task 2 adds a second stt option (`faster-whisper`,
    # local) alongside `xai` -- assert on the xai option specifically
    # rather than assuming the slot has exactly one.
    options_by_name = {option["name"]: option for option in slot["options"]}
    assert options_by_name.keys() == {"xai", "faster-whisper"}
    assert options_by_name["xai"]["label"] == "xAI"
    assert options_by_name["xai"]["requires_credential"] is True
    assert options_by_name["faster-whisper"]["requires_credential"] is False


def test_get_providers_reports_the_tts_slot_wrapped_with_no_measured_figure_yet(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """D-08: a wrapped slot with nothing synthesized since the last
    restart reports `wrapped: true` and `measured_ms: null` -- an honest
    absence, not an invented zero."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    body = response.json()

    slot = _slot(body, "tts")
    assert slot["label"] == "Text to speech"
    assert slot["wrapped"] is True
    assert slot["measured_ms"] is None
    # 07-03-PLAN.md Task 3 adds a second tts option (`piper`, local)
    # alongside `xai` -- both declare batch=True on their registry entry.
    assert {option["name"] for option in slot["options"]} == {"xai", "piper"}
    assert all(option["wrapped"] is True for option in slot["options"]), (
        "both tts registry entries declare batch=True"
    )


def test_get_providers_reports_a_real_measured_figure_off_the_live_tts_client(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """D-08: `measured_ms` is a live read off `app.state.tts` at
    response-build time, never a stored copy -- a client that has
    synthesized something reports the real figure."""
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
        tts_client=SimpleNamespace(last_synthesis_ms=1467.3),
    )
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    slot = _slot(response.json(), "tts")
    assert slot["measured_ms"] == 1467.3


def test_get_providers_reports_the_brain_slot(
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
    slot = _slot(response.json(), "brain")
    assert slot["label"] == "Language model"
    assert slot["selected"] == "xai"
    assert slot["wrapped"] is False
    # 07-04-PLAN.md Task 1 adds a second (local) brain option -- assert by
    # name rather than assuming exactly one, matching the fix already
    # applied for stt/tts in this file (07-02/07-03).
    options_by_name = {option["name"]: option for option in slot["options"]}
    assert options_by_name["xai"]["name"] == "xai"


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
    options_by_name = {option["name"]: option for option in _slot(response.json(), "stt")["options"]}
    assert options_by_name["xai"]["credential_set"] is True
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
    options_by_name = {option["name"]: option for option in _slot(response.json(), "stt")["options"]}
    assert options_by_name["xai"]["credential_set"] is False


def test_get_providers_reports_tts_and_brain_credential_set_from_their_own_slot(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """T-07-11-adjacent honesty check: each slot's `credential_set` reads
    its own credential slot (`CredentialSlot.TTS`/`CredentialSlot.BRAIN`),
    never the speech-to-text one, even though all three share the same
    registered provider name."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    from spire_voice.crypto.credentials import encrypt_credential

    ciphertext, key_version = encrypt_credential("a-real-secret-token", security)
    asyncio.run(
        credential_repo.upsert_credential(
            "tts_api_key", ciphertext=ciphertext, key_version=key_version, updated_by_user_id=None
        )
    )

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    body = response.json()
    assert _slot(body, "tts")["options"][0]["credential_set"] is True
    assert _slot(body, "stt")["options"][0]["credential_set"] is False
    assert _slot(body, "brain")["options"][0]["credential_set"] is False


def test_get_providers_reports_needs_server_url_and_measured_note_from_the_registry(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """07-06-PLAN.md Task 1: `ProviderEntry.needs_server_url`/`.measured_note`
    (07-04-PLAN.md) reach the wire unchanged -- the flag the screen uses to
    reveal a "Server URL" field and the local-set latency caption, from
    data rather than a hardcoded provider name."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    body = response.json()

    brain_options = {option["name"]: option for option in _slot(body, "brain")["options"]}
    assert brain_options["xai"]["needs_server_url"] is False
    assert brain_options["local"]["needs_server_url"] is True
    assert brain_options["xai"]["measured_note"] is None

    stt_options = {option["name"]: option for option in _slot(body, "stt")["options"]}
    assert stt_options["xai"]["needs_server_url"] is False
    assert stt_options["xai"]["measured_note"] is None
    assert stt_options["faster-whisper"]["measured_note"] is not None
    assert "Measured on this project's CPU-only host" in stt_options["faster-whisper"]["measured_note"]

    tts_options = {option["name"]: option for option in _slot(body, "tts")["options"]}
    assert tts_options["piper"]["measured_note"] is not None
    assert tts_options["piper"]["needs_server_url"] is False


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
    assert _slot(body, "stt")["selected"] == "xai"

    reload_response = client.get("/api/providers")
    assert _slot(reload_response.json(), "stt")["selected"] == "xai"


def test_put_providers_accepts_all_three_slots_in_one_request(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """07-UI-SPEC.md's probe addendum: one request carrying all three
    choices, one outcome -- no per-slot save and no partial-success
    state to render."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.put(
        "/api/providers",
        json={
            "slots": {
                "stt": {"provider_name": "xai"},
                "tts": {"provider_name": "xai"},
                "brain": {"provider_name": "xai"},
            }
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert _slot(body, "stt")["selected"] == "xai"
    assert _slot(body, "tts")["selected"] == "xai"
    assert _slot(body, "brain")["selected"] == "xai"


def test_put_providers_refuses_an_unrecognized_provider_name_and_leaves_the_row_unchanged(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """T-07-02: a save carrying a name no registry key matches is refused,
    naming the value and the known set, and the stored row is untouched
    afterwards -- a partial write across a save is the state 07-UI-SPEC.md's
    probe addendum explicitly refuses to render."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.put(
        "/api/providers", json={"slots": {"stt": {"provider_name": "not-a-real-provider"}}}
    )
    assert response.status_code == 400
    assert "not-a-real-provider" in response.json()["detail"]
    assert "xai" in response.json()["detail"]

    reload_response = client.get("/api/providers")
    assert _slot(reload_response.json(), "stt")["selected"] == "xai", (
        "the stored row must be unchanged after a refused save"
    )


def test_put_providers_refuses_an_unknown_slot(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.put("/api/providers", json={"slots": {"video": {"provider_name": "xai"}}})
    assert response.status_code == 400
    assert "video" in response.json()["detail"]


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
    # the shape a second registered provider would produce.
    asyncio.run(
        provider_selection_repo.set_selection(
            "stt", "a-name-the_boot_never_saw", {}, updated_by_user_id=None, updated_at=datetime.now(timezone.utc)
        )
    )

    app = _build_providers_app(security, account_repo, provider_selection_repo, credential_repo)
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    slot = _slot(response.json(), "stt")
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

    provider_slots = dict(_DEFAULT_PROVIDER_SLOTS)
    provider_slots["stt"] = ProviderSlotStatus(
        slot="stt",
        selected="xai",
        active=None,
        state="degraded",
        reason="xAI needs an API key -- add one in Settings.",
        wrapped=False,
    )

    app = _build_providers_app(
        security,
        account_repo,
        provider_selection_repo,
        credential_repo,
        provider_slots=provider_slots,
    )
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    slot = _slot(response.json(), "stt")
    assert slot["state"] == "degraded"
    assert slot["active"] is None
    assert slot["reason"] == "xAI needs an API key -- add one in Settings."


def test_get_providers_reports_a_degraded_tts_slot_with_no_measured_figure(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """A degraded tts slot has no client at all -- `measured_ms` must
    read `None` rather than erroring on a missing attribute."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    provider_slots = dict(_DEFAULT_PROVIDER_SLOTS)
    provider_slots["tts"] = ProviderSlotStatus(
        slot="tts",
        selected="xai",
        active=None,
        state="degraded",
        reason="Missing an API key for xAI. Add one in Settings.",
        wrapped=False,
    )

    app = _build_providers_app(
        security,
        account_repo,
        provider_selection_repo,
        credential_repo,
        provider_slots=provider_slots,
        tts_client=None,
    )
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/providers")
    slot = _slot(response.json(), "tts")
    assert slot["state"] == "degraded"
    assert slot["measured_ms"] is None


# --- WR-08 (code review): a degraded slot the admin has just fixed ------


def test_a_degraded_slot_whose_settings_changed_since_boot_reports_it(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """WR-08, walked exactly as the review describes it.

    The language-model slot is set to `local` with no server URL, so boot
    degrades it: `state="degraded"`, `active=None`, and a reason naming
    the missing URL. The admin types the URL and saves. `selected` is
    still `"local"` and `active` is still `None`, so neither
    `selected != active` nor anything else the response carried could
    say that something had changed -- the screen showed no restart badge
    and rendered the OLD reason beside the field they had just filled in.

    `selection_changed_since_boot` is the fact that was missing: the
    server knows what configuration the boot actually used, and can
    compare it against a fresh read.
    """
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    reason = "Missing a server URL for Self-hosted (local). Add one in Settings."
    provider_slots = dict(_DEFAULT_PROVIDER_SLOTS)
    provider_slots["brain"] = ProviderSlotStatus(
        slot="brain",
        selected="local",
        active=None,
        state="degraded",
        reason=reason,
        wrapped=False,
        booted_settings={},
    )
    asyncio.run(
        provider_selection_repo.set_selection(
            "brain", "local", {}, updated_by_user_id=None, updated_at=datetime.now(timezone.utc)
        )
    )

    app = _build_providers_app(
        security,
        account_repo,
        provider_selection_repo,
        credential_repo,
        provider_slots=provider_slots,
    )
    client = _admin_client(app, security, account_repo)

    before = _slot(client.get("/api/providers").json(), "brain")
    assert before["selection_changed_since_boot"] is False, (
        "nothing has been changed yet -- a slot that boots degraded and is then left "
        "alone must not claim a restart would help"
    )
    assert before["reason"] == reason

    # The admin fills in the server URL and saves.
    response = client.put(
        "/api/providers",
        json={"slots": {"brain": {"provider_name": "local", "settings": {"server_url": "http://localhost:8000/v1"}}}},
    )
    assert response.status_code == 200, response.text

    after = _slot(client.get("/api/providers").json(), "brain")
    assert after["selected"] == "local"
    assert after["active"] is None
    assert after["state"] == "degraded"
    assert after["settings"] == {"server_url": "http://localhost:8000/v1"}
    assert after["selection_changed_since_boot"] is True
    # The reason is still reported byte for byte -- the fix is that the
    # screen now knows it describes a configuration that is no longer
    # stored. Withdrawing it from the response would lose the ability to
    # show it at all.
    assert after["reason"] == reason


def test_a_provider_name_changed_since_boot_reports_it_too(
    monkeypatch, fake_account_repository, fake_provider_selection_repository, fake_credential_repository
):
    """The running case still works through the same field, so a slot
    that is not degraded reports the change from both directions."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    provider_selection_repo = fake_provider_selection_repository()
    credential_repo = fake_credential_repository()

    app = _build_providers_app(
        security, account_repo, provider_selection_repo, credential_repo
    )
    client = _admin_client(app, security, account_repo)

    assert _slot(client.get("/api/providers").json(), "stt")["selection_changed_since_boot"] is False

    response = client.put(
        "/api/providers", json={"slots": {"stt": {"provider_name": "faster-whisper"}}}
    )
    assert response.status_code == 200, response.text

    after = _slot(client.get("/api/providers").json(), "stt")
    assert after["selection_changed_since_boot"] is True
    assert after["active"] == "xai", "what is actually running is unchanged by a save (D-02)"
