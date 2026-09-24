"""The wizard reaches a working assistant and proves the microphone and
speaker work before it lets the operator call setup finished (WEB-02,
WEB-03).

Wizard step state is server-side (Postgres), specifically so an abandoned
wizard resumes where it was left rather than forcing a re-run from step one
-- a stranger who closes the tab partway through a first-run setup, per
PROJECT.md's own requirement that a stranger can deploy and configure this
project from a browser, must not lose their progress. The second test is
this phase's inherited obligation from Phase 2 (STATE.md's handoff, named
in 03-CONTEXT.md's domain section): camera barge-in (VOICE-07) ships
disabled because no real echo-path calibration exists, and the wizard is
where the operator chose to close that gap. A wizard that lets "finish" be
clicked before the microphone/speaker test (Phase 2's
`/calibration/echo-path/run`) has actually run would ship a wizard that
*looks* like it closed VOICE-07 without actually running the one check that
does.

Every route this file drives sits on a throwaway `FastAPI()` app carrying
the real routers (`auth_router`, `setup_router`, `wizard_router`) over
in-memory fakes -- the same "primitives in isolation, not the whole
lifespan" shape `test_auth_roles.py`'s own `_build_test_app` and
`test_credentials_crypto.py`'s own `_build_full_app` already use.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import conftest
from atlas.auth.tokens import issue_access_token
from atlas.calibration.record import EchoCalibration
from atlas.config import SecurityConfig
from atlas.crypto.credentials import CredentialSlot, encrypt_credential
from atlas.db.repository import Plugin, PluginConfigValue
from atlas.routes.auth import router as auth_router, setup_router
from atlas.routes.wizard import router as wizard_router
import atlas.routes.wizard as wizard_module

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"
_HA_URL = "http://ha.invalid:8123"
_NO_SUCH_CALIBRATION_DIR = "/tmp/atlas-test-no-such-calibration-dir-wizard-flow"


def _fake_config(*, calibration_dir: str = _NO_SUCH_CALIBRATION_DIR) -> SimpleNamespace:
    """A `Config`-shaped stand-in carrying only what `routes/wizard.py`'s
    own per-step checkers read: the three provider credential slots'
    environment fallbacks (empty, matching a fresh install) and the
    calibration directory. Plan 06-01: no `mcp_servers` any more -- the
    hub step reads `HA_URL`/`HA_TOKEN` from a `PluginRepository` now
    (`_ha_plugin_repo` below), never from `Config`.
    """
    return SimpleNamespace(
        stt=SimpleNamespace(api_key=""),
        brain=SimpleNamespace(api_key=""),
        tts=SimpleNamespace(api_key=""),
        calibration=SimpleNamespace(dir=calibration_dir),
    )


def _ha_plugin_repo(
    *,
    ha_url: str | None = _HA_URL,
    ha_token: str | None = None,
    security: SecurityConfig | None = None,
) -> conftest.FakePluginRepository:
    """A `PluginRepository` carrying only the `ha` plugin row `routes/
    wizard.py::_ha_plugin_connection_info` reads (plan 06-01, D-01):
    `HA_URL` as a plain config value, and `HA_TOKEN` -- when given --
    encrypted through the real `encrypt_credential` under `security`, the
    same "a database-stored value is the only path that reaches
    `decrypt_credential`" shape this file's own pre-06-01 tests already
    relied on for `CredentialSlot.HOME_ASSISTANT`.
    """
    now = datetime.now(timezone.utc)
    plugin = Plugin(
        id=1,
        slug="ha",
        display_name="Home Assistant",
        transport="stdio",
        args=("-m", "atlas_mcp.ha"),
        url=None,
        enabled=True,
        builtin=True,
        enforces_policy=True,
        timeout_ms=5000,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )
    config_values: list[PluginConfigValue] = []
    if ha_url:
        config_values.append(
            PluginConfigValue(key="HA_URL", secret=False, value=ha_url, ciphertext=None, key_version=None)
        )
    if ha_token:
        assert security is not None, "ha_token given with no security to encrypt it under"
        ciphertext, key_version = encrypt_credential(ha_token, security)
        config_values.append(
            PluginConfigValue(
                key="HA_TOKEN", secret=True, value=None, ciphertext=ciphertext, key_version=key_version
            )
        )
    return conftest.FakePluginRepository(plugins=[plugin], config_values={1: config_values})


def _build_wizard_app(
    security: SecurityConfig,
    account_repo,
    credential_repo,
    setup_repo,
    settings_repo,
    *,
    config: SimpleNamespace | None = None,
    plugin_repo: conftest.FakePluginRepository | None = None,
    ha_http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """The wizard's own routers, plus `auth_router`/`setup_router` (the
    create-admin and setup-status routes the wizard's own tests need to
    drive a sign-in and to assert `GET /api/setup/status`'s own shape),
    on one throwaway app -- never the whole `lifespan`."""
    app = FastAPI()
    app.state.config = SimpleNamespace(security=security, **(config or _fake_config()).__dict__)
    app.state.account_repo = account_repo
    app.state.credential_repo = credential_repo
    app.state.setup_repo = setup_repo
    app.state.settings_repo = settings_repo
    app.state.plugin_repo = plugin_repo if plugin_repo is not None else _ha_plugin_repo()
    if ha_http_client is not None:
        app.state.ha_http_client = ha_http_client
    app.include_router(auth_router)
    app.include_router(setup_router)
    app.include_router(wizard_router)
    return app


def _create_admin(client: TestClient) -> dict:
    response = client.post(
        "/api/auth/create-admin",
        json={
            "email": "wizard-admin@example.invalid",
            "display_name": "Wizard Admin",
            "password": "a-plainly-fictional-wizard-password",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _admin_cookie(security: SecurityConfig, user_id: int) -> str:
    return issue_access_token(user_id=user_id, role="admin", security=security)


def _make_calibration(taken_at: datetime, *, source: str = "camera") -> EchoCalibration:
    return EchoCalibration(
        schema_version=1,
        probe_format_version=1,
        probe_seed=12345,
        source=source,
        delay_s=0.12,
        confidence=0.9,
        echo_level=0.3,
        gain=1.0,
        agc_verdict="fixed",
        segment_levels=(0.1, 0.2, 0.3),
        encoding="alaw",
        sample_rate=8000,
        channels=1,
        placement_note="test placement",
        taken_at=taken_at,
    )


def test_an_abandoned_wizard_resumes_at_the_first_unfinished_step(monkeypatch):
    """Closing the wizard partway through and returning later must resume
    at the first step that was not yet completed, not restart from the
    beginning."""
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = conftest.FakeAccountRepository()
    credential_repo = conftest.FakeCredentialRepository()
    setup_repo = conftest.FakeSetupRepository()
    settings_repo = conftest.FakeSettingsRepository()

    app = _build_wizard_app(security, account_repo, credential_repo, setup_repo, settings_repo)
    client = TestClient(app)

    admin = _create_admin(client)
    client.cookies.set(security.cookie_name, _admin_cookie(security, admin["id"]))

    # Nothing else done yet: admin_account is the only complete step.
    first_read = client.get("/api/wizard")
    assert first_read.status_code == 200, first_read.text
    assert first_read.json()["first_unfinished_step"] == "hub"

    # Complete the hub step for real, through its own route.
    fake_ha = conftest.FakeHomeAssistant()
    app.state.ha_http_client = fake_ha.client
    hub_response = client.post("/api/wizard/steps/hub/check")
    assert hub_response.status_code == 200, hub_response.text
    assert hub_response.json()["complete"] is True

    # Drop the session entirely -- a fresh client with no cookies at all,
    # the same "closed the tab" scenario the wizard exists to survive.
    del app.state.ha_http_client
    anonymous = TestClient(app)
    anonymous.cookies.set(security.cookie_name, _admin_cookie(security, admin["id"]))

    resumed = anonymous.get("/api/wizard")
    assert resumed.status_code == 200, resumed.text
    body = resumed.json()
    steps_by_name = {s["name"]: s["complete"] for s in body["steps"]}
    assert steps_by_name["admin_account"] is True
    assert steps_by_name["hub"] is True
    assert steps_by_name["provider_set"] is False
    # The first unfinished step is now provider_set, not admin_account or
    # hub -- the read resumes past what was already completed.
    assert body["first_unfinished_step"] == "provider_set"


def test_the_wizard_cannot_finish_before_the_microphone_and_speaker_test_runs(monkeypatch):
    """The wizard's final "finish" step must be unreachable until the
    microphone/speaker test (Phase 2's echo-path calibration) has actually
    run and returned a result -- every other step complete, room still
    missing, finish refused naming exactly `room`."""
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = conftest.FakeAccountRepository()
    credential_repo = conftest.FakeCredentialRepository()
    setup_repo = conftest.FakeSetupRepository()
    settings_repo = conftest.FakeSettingsRepository()

    app = _build_wizard_app(security, account_repo, credential_repo, setup_repo, settings_repo)
    client = TestClient(app)

    admin = _create_admin(client)
    client.cookies.set(security.cookie_name, _admin_cookie(security, admin["id"]))

    # Complete hub.
    fake_ha = conftest.FakeHomeAssistant()
    app.state.ha_http_client = fake_ha.client
    assert client.post("/api/wizard/steps/hub/check").status_code == 200
    del app.state.ha_http_client

    # Complete provider_set (every slot, from the environment this time --
    # a database write is not the only way a slot becomes set).
    for slot, env_attr in (
        (CredentialSlot.STT, "stt"),
        (CredentialSlot.BRAIN, "brain"),
        (CredentialSlot.TTS, "tts"),
    ):
        setattr(getattr(app.state.config, env_attr), "api_key", "env-value-not-a-real-key")

    # Complete audio_source.
    assert client.put("/api/wizard/audio-source", json={"source": "camera"}).status_code == 200

    # room is still missing -- no calibration on disk. Finish must refuse.
    still_incomplete = client.post("/api/wizard/finish")
    assert still_incomplete.status_code == 409, still_incomplete.text
    assert "room" in still_incomplete.json()["detail"]

    read_before = client.get("/api/wizard")
    assert read_before.json()["first_unfinished_step"] == "room"

    # Now make a calibration exist, exactly as Phase 2's own run route
    # would leave one -- the wizard reads it, never runs a second one.
    monkeypatch.setattr(
        wizard_module,
        "find_latest_calibration",
        lambda _dir: _make_calibration(datetime.now(timezone.utc)),
    )

    finished = client.post("/api/wizard/finish")
    assert finished.status_code == 200, finished.text
    assert finished.json()["complete"] is True

    status = client.get("/api/setup/status")
    assert status.status_code == 200, status.text
    assert status.json()["complete"] is True


# ---------------------------------------------------------------------
# Per-step condition tests (plan 03-09 Task 2)
# ---------------------------------------------------------------------


def _authed_app(
    monkeypatch,
    *,
    config: SimpleNamespace | None = None,
    plugin_repo: conftest.FakePluginRepository | None = None,
    ha_http_client=None,
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = conftest.FakeAccountRepository()
    credential_repo = conftest.FakeCredentialRepository()
    setup_repo = conftest.FakeSetupRepository()
    settings_repo = conftest.FakeSettingsRepository()
    app = _build_wizard_app(
        security, account_repo, credential_repo, setup_repo, settings_repo,
        config=config, plugin_repo=plugin_repo, ha_http_client=ha_http_client,
    )
    client = TestClient(app)
    admin = _create_admin(client)
    client.cookies.set(security.cookie_name, _admin_cookie(security, admin["id"]))
    return app, client, security, admin, credential_repo, setup_repo, settings_repo


def test_the_hub_step_completes_only_after_a_successful_call_and_records_when(monkeypatch):
    app, client, *_ = _authed_app(monkeypatch)
    fake_ha = conftest.FakeHomeAssistant()
    app.state.ha_http_client = fake_ha.client

    before = client.get("/api/wizard")
    assert before.json()["steps"][1]["name"] == "hub"
    assert before.json()["steps"][1]["complete"] is False

    response = client.post("/api/wizard/steps/hub/check")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["complete"] is True
    assert "checked_at" in body["detail"]
    assert len(fake_ha.requests) == 1

    after = client.get("/api/wizard")
    hub_step = next(s for s in after.json()["steps"] if s["name"] == "hub")
    assert hub_step["complete"] is True
    assert hub_step["detail"]["checked_at"] == body["detail"]["checked_at"]


def test_the_hub_check_route_never_returns_the_decrypted_home_assistant_token(monkeypatch):
    """WR-02 (code review): `check_hub_step` is the second call site
    `resolve_credential_value`/`decrypt_credential` reach, alongside
    `app.py`'s `lifespan` -- `crypto/credentials.py`'s own docstring now
    names both and states the guarantee that actually matters: the
    decrypted value is used only for a server-side outbound call and
    never included in a response body. This is that guarantee, checked
    directly against this specific route (the review's own suggested
    follow-up to `test_credentials_crypto.py::
    test_a_saved_credential_never_comes_back_from_any_route`, which never
    walks a POST route and so never exercised this one).

    A real, database-stored (not environment-fallback) Home Assistant
    token is used here specifically -- `resolve_credential_value` only
    reaches `decrypt_credential` at all when a database row exists to
    decrypt; the environment-fallback path this file's other hub tests
    use never calls it, and would pass this assertion vacuously.
    """
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    # The one placeholder value `tests/test_repo_hygiene.py`'s own
    # credential-literal check already allowlists (see that file's
    # `_ALLOWED_CREDENTIAL_VALUES`) -- an invented value here, not a real
    # Home Assistant token shape, is the point either way.
    stored_value = "test-key"

    # Plan 06-01: `HA_TOKEN` is a plugin config value now, not a
    # `CredentialSlot.HOME_ASSISTANT` provider credential -- `_ha_plugin_repo`
    # is what makes this a genuinely database-stored (encrypted, decrypted
    # only by `_ha_plugin_connection_info`) token, the same "only a real
    # ciphertext row reaches decrypt_credential" shape this test's own
    # docstring already required of the retired mechanism.
    app, client, _security, _admin, _credential_repo, _setup_repo, _settings_repo = _authed_app(
        monkeypatch,
        config=_fake_config(),
        plugin_repo=_ha_plugin_repo(ha_token=stored_value, security=security),
    )

    fake_ha = conftest.FakeHomeAssistant()
    app.state.ha_http_client = fake_ha.client

    response = client.post("/api/wizard/steps/hub/check")
    assert response.status_code == 200, response.text
    assert stored_value not in response.text, (
        "check_hub_step's response body must never contain the decrypted "
        "Home Assistant token -- WR-02's whole point is that this route "
        "reaches decrypt_credential without becoming a second way to read "
        "a credential back out"
    )
    assert set(response.json()) == {"name", "complete", "detail"}


def test_a_refused_connection_is_reported_as_unreachable(monkeypatch):
    def _raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    app, client, *_ = _authed_app(monkeypatch)
    client_with_refused_transport = httpx.AsyncClient(transport=httpx.MockTransport(_raise_connect_error))
    app.state.ha_http_client = client_with_refused_transport

    response = client.post("/api/wizard/steps/hub/check")
    assert response.status_code == 502, response.text
    assert response.json()["detail"].startswith("unreachable:")

    after = client.get("/api/wizard")
    hub_step = next(s for s in after.json()["steps"] if s["name"] == "hub")
    assert hub_step["complete"] is False


def test_an_authentication_failure_is_reported_as_unauthorized(monkeypatch):
    def _unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "invalid auth token"})

    app, client, *_ = _authed_app(monkeypatch)
    app.state.ha_http_client = httpx.AsyncClient(transport=httpx.MockTransport(_unauthorized))

    response = client.post("/api/wizard/steps/hub/check")
    assert response.status_code == 502, response.text
    assert response.json()["detail"].startswith("unauthorized:")


def test_an_unparseable_response_is_reported_as_unparseable(monkeypatch):
    def _garbage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    app, client, *_ = _authed_app(monkeypatch)
    app.state.ha_http_client = httpx.AsyncClient(transport=httpx.MockTransport(_garbage))

    response = client.post("/api/wizard/steps/hub/check")
    assert response.status_code == 502, response.text
    assert response.json()["detail"].startswith("unparseable:")


def test_the_provider_step_refuses_with_the_blank_slots_named(monkeypatch):
    app, client, security, admin, credential_repo, *_ = _authed_app(monkeypatch)

    # Nothing set anywhere yet -- every slot is blank.
    status = client.get("/api/wizard")
    provider_step = next(s for s in status.json()["steps"] if s["name"] == "provider_set")
    assert provider_step["complete"] is False
    assert sorted(provider_step["detail"]["missing"]) == sorted(
        [CredentialSlot.STT.value, CredentialSlot.BRAIN.value, CredentialSlot.TTS.value]
    )

    # Set two of three through the real credential-encryption path --
    # the wizard never writes credentials through a second path.
    for slot in (CredentialSlot.STT, CredentialSlot.BRAIN):
        ciphertext, key_version = encrypt_credential("a-plainly-fictional-value", security)
        import asyncio

        asyncio.run(
            credential_repo.upsert_credential(
                slot.value, ciphertext=ciphertext, key_version=key_version, updated_by_user_id=admin["id"]
            )
        )

    still_incomplete = client.get("/api/wizard")
    provider_step = next(s for s in still_incomplete.json()["steps"] if s["name"] == "provider_set")
    assert provider_step["complete"] is False
    assert provider_step["detail"]["missing"] == [CredentialSlot.TTS.value]


def test_a_slot_satisfied_only_by_the_environment_counts_as_set(monkeypatch):
    """An existing deployment with keys in the environment (never touching
    the database) must not be told its provider set is incomplete."""
    # "test-key" is this repository's one allowlisted credential-shaped
    # placeholder (tests/test_repo_hygiene.py).
    config = _fake_config()
    config.stt.api_key = "test-key"
    config.brain.api_key = "test-key"
    config.tts.api_key = "test-key"

    app, client, *_ = _authed_app(monkeypatch, config=config)

    response = client.get("/api/wizard")
    provider_step = next(s for s in response.json()["steps"] if s["name"] == "provider_set")
    assert provider_step["complete"] is True
    assert "missing" not in provider_step["detail"]


def test_the_audio_source_setting_is_stored_and_read_back_reporting_a_restart(monkeypatch):
    app, client, *_ = _authed_app(monkeypatch)

    before = client.get("/api/wizard")
    source_step = next(s for s in before.json()["steps"] if s["name"] == "audio_source")
    assert source_step["complete"] is False

    write = client.put("/api/wizard/audio-source", json={"source": "camera"})
    assert write.status_code == 200, write.text
    assert write.json()["complete"] is True
    assert write.json()["detail"]["applies_live"] is False
    assert write.json()["detail"]["source"] == "camera"

    after = client.get("/api/wizard")
    source_step = next(s for s in after.json()["steps"] if s["name"] == "audio_source")
    assert source_step["complete"] is True
    assert source_step["detail"]["resolved_from"] == "database"


def test_an_unknown_audio_source_is_refused(monkeypatch):
    app, client, *_ = _authed_app(monkeypatch)
    response = client.put("/api/wizard/audio-source", json={"source": "not-a-real-source"})
    assert response.status_code == 400, response.text


def test_the_room_step_reads_the_stored_calibration_and_never_runs_one(monkeypatch):
    # The room step must never be able to reach a real calibration run --
    # asserted structurally: no attribute of this module names
    # `run_echo_calibration` at all, so there is nothing in the module's
    # own namespace a call could even resolve to.
    assert not hasattr(wizard_module, "run_echo_calibration")

    monkeypatch.setattr(wizard_module, "find_latest_calibration", lambda _dir: None)
    app, client, *_ = _authed_app(monkeypatch)

    before = client.get("/api/wizard")
    room_step = next(s for s in before.json()["steps"] if s["name"] == "room")
    assert room_step["complete"] is False
    assert room_step["detail"] is None

    taken_at = datetime.now(timezone.utc)
    monkeypatch.setattr(wizard_module, "find_latest_calibration", lambda _dir: _make_calibration(taken_at))

    after = client.get("/api/wizard")
    room_step = next(s for s in after.json()["steps"] if s["name"] == "room")
    assert room_step["complete"] is True
    assert room_step["detail"]["taken_at"] == taken_at.isoformat()


def test_finishing_succeeds_and_status_reports_complete_end_to_end(monkeypatch):
    """With all five conditions met, finishing succeeds and `GET
    /api/setup/status` reports complete."""
    app, client, security, admin, credential_repo, *_ = _authed_app(monkeypatch)

    fake_ha = conftest.FakeHomeAssistant()
    app.state.ha_http_client = fake_ha.client
    assert client.post("/api/wizard/steps/hub/check").status_code == 200
    del app.state.ha_http_client

    for slot in (CredentialSlot.STT, CredentialSlot.BRAIN, CredentialSlot.TTS):
        ciphertext, key_version = encrypt_credential("a-plainly-fictional-value", security)
        import asyncio

        asyncio.run(
            credential_repo.upsert_credential(
                slot.value, ciphertext=ciphertext, key_version=key_version, updated_by_user_id=admin["id"]
            )
        )

    assert client.put("/api/wizard/audio-source", json={"source": "camera"}).status_code == 200

    monkeypatch.setattr(
        wizard_module, "find_latest_calibration", lambda _dir: _make_calibration(datetime.now(timezone.utc))
    )

    finished = client.post("/api/wizard/finish")
    assert finished.status_code == 200, finished.text

    status = client.get("/api/setup/status")
    assert status.status_code == 200, status.text
    body = status.json()
    assert body["complete"] is True
    assert all(step["complete"] for step in body["steps"])


def test_setup_status_leaks_no_house_data(monkeypatch):
    """`GET /api/setup/status` returns step names and booleans only --
    never a hub address, an entity id, or an email, even once the hub step
    has actually recorded a real check."""
    app, client, *_ = _authed_app(monkeypatch)
    fake_ha = conftest.FakeHomeAssistant()
    app.state.ha_http_client = fake_ha.client
    assert client.post("/api/wizard/steps/hub/check").status_code == 200

    response = client.get("/api/setup/status")
    assert response.status_code == 200, response.text
    body_text = response.text
    assert _HA_URL not in body_text
    assert "@example.invalid" not in body_text
    assert "wizard-admin" not in body_text
    parsed = response.json()
    assert set(parsed.keys()) == {"complete", "steps"}
    for step in parsed["steps"]:
        assert set(step.keys()) == {"name", "complete"}


def test_wizard_routes_answer_while_setup_is_incomplete_and_are_not_locked_out(monkeypatch):
    """The wizard's own routes (and the status route) must never 503 with
    "setup incomplete" -- they may refuse for other reasons (401 with no
    session, 409 while steps are outstanding), but the setup gate itself
    must never be what stops them, which is exactly what the lockout WEB-02
    forbids would look like."""
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = conftest.FakeAccountRepository()
    credential_repo = conftest.FakeCredentialRepository()
    setup_repo = conftest.FakeSetupRepository()
    settings_repo = conftest.FakeSettingsRepository()
    app = _build_wizard_app(security, account_repo, credential_repo, setup_repo, settings_repo)
    client = TestClient(app)

    # No admin exists yet -- the setup gate is not wired onto this
    # throwaway app (it is an application-level dependency on the real
    # `app.py::app`, asserted separately in `test_auth_setup.py`), so this
    # asserts the narrower, route-local claim: every wizard route and the
    # status route answer with something other than a hand-rolled "setup
    # incomplete" 503 of their own.
    status = client.get("/api/setup/status")
    assert status.status_code == 200, status.text

    for method, path, json_body in (
        ("GET", "/api/wizard", None),
        ("POST", "/api/wizard/steps/hub/check", None),
        ("PUT", "/api/wizard/audio-source", {"source": "camera"}),
        ("POST", "/api/wizard/finish", None),
    ):
        response = client.request(method, path, json=json_body)
        assert response.status_code != 503, f"{method} {path} reported setup incomplete: {response.text}"
