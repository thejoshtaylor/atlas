"""Provider credentials are encrypted at rest and write-only from the
browser (PROV-04).

CONTEXT.md is explicit: after a credential is saved, the webapp shows a
mask and never receives the plaintext back. A route that returns a stored
credential's value -- even to an authenticated admin, even for a
"show/hide" convenience toggle -- defeats the reason this data is encrypted
at all: the whole point of write-only is that a compromised admin session,
or a bug in a response serializer that includes one field too many, cannot
leak the plaintext back out over the same channel that would leak the leak.
The second test guards the storage layer directly: a row read straight out
of the database, with no `ATLAS_SECRET_KEY`, must not be interpretable as
the original credential -- proving the encryption is real, not a
base64-shaped no-op.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cryptography.fernet import InvalidToken

from atlas.auth.tokens import _CREDENTIAL_ENCRYPTION_INFO, _JWT_SIGNING_INFO, _derive_key
from atlas.auth.tokens import issue_access_token
from atlas.config import SecurityConfig, read_secret_key
from atlas.crypto.credentials import (
    CredentialSlot,
    decrypt_credential,
    encrypt_credential,
    resolve_credential_source,
    resolve_credential_value,
)
from atlas.routes.accounts import router as accounts_router
from atlas.routes.auth import router as auth_router, setup_router
from atlas.routes.credentials import router as credentials_router
from atlas.routes.policy import router as policy_router

import conftest
from test_auth_roles import _flatten_routes
from test_auth_setup import _fill_path_params

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"
_OTHER_SECRET_KEY = "a-different-plainly-fictional-secret-value"
# Plainly fictional -- not a real API key of any provider's, shaped
# distinctly enough (a nonsense phrase) that it cannot be mistaken for one.
_REAL_SECRET_VALUE = "correct horse battery staple, definitely not a real key"


def _fake_config() -> SimpleNamespace:
    """A `Config`-shaped stand-in carrying only what
    `atlas.crypto.credentials.env_value_for_slot` reads -- every
    provider slot's environment fallback is empty, and the Home Assistant
    slot has no declared server block, matching a fresh install with
    nothing in the environment either."""
    return SimpleNamespace(
        stt=SimpleNamespace(api_key=""),
        brain=SimpleNamespace(api_key=""),
        tts=SimpleNamespace(api_key=""),
        mcp_servers={},
        # Plan 03-09: `GET /api/setup/status` (included below) now
        # recomputes every wizard step live, including `room`
        # (`find_latest_calibration(config.calibration.dir)`) -- a
        # directory that does not exist reads as "no calibration yet"
        # (`find_latest_calibration`'s own docstring), never an error.
        calibration=SimpleNamespace(dir="/tmp/atlas-test-no-such-calibration-dir"),
    )


def _build_full_app(security, account_repo, policy_repo, credential_repo, tool_host=None) -> FastAPI:
    """Every Phase 3 router this application registers, on one throwaway
    `FastAPI()` app -- the same "primitives in isolation, not the whole
    lifespan" shape `tests/test_auth_roles.py`'s own `_build_test_app`
    uses, extended to cover every router a credential could plausibly
    surface from."""
    app = FastAPI()
    app.state.config = SimpleNamespace(security=security, **_fake_config().__dict__)
    app.state.account_repo = account_repo
    app.state.policy_repo = policy_repo
    app.state.credential_repo = credential_repo
    # Plan 03-09: `GET /api/setup/status` (included below via
    # `setup_router`) now reads `setup_repo`/`settings_repo` too.
    app.state.setup_repo = conftest.FakeSetupRepository()
    app.state.settings_repo = conftest.FakeSettingsRepository()
    # WR-06 (code review): the `ha_token` slot is stored as the Home
    # Assistant plugin's own `HA_TOKEN` configuration value -- the one
    # place anything reads that token from -- so the credentials routes
    # read `plugin_repo` for that one slot. Seeded empty here: this file
    # is about the three provider slots and the write-only property they
    # all share.
    app.state.plugin_repo = conftest.FakePluginRepository()
    app.state.tool_host = tool_host
    app.state.safety_block = None
    app.include_router(auth_router)
    app.include_router(setup_router)
    app.include_router(accounts_router)
    app.include_router(policy_router)
    app.include_router(credentials_router)
    return app


def test_a_saved_credential_never_comes_back_from_any_route(
    monkeypatch, fake_account_repository, fake_policy_repository, fake_credential_repository
):
    """Enumerate every route this throwaway application registers, call
    each one a credential could plausibly surface from, and assert no
    response body anywhere contains the stored plaintext -- an
    enumeration, not a sample, for the same reason plan 03-05's own
    route-coverage test is one: the failure mode is the one route
    somebody forgot.
    """
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()
    credential_repo = fake_credential_repository()

    app = _build_full_app(security, account_repo, policy_repo, credential_repo)
    client = TestClient(app)

    created = client.post(
        "/api/auth/create-admin",
        json={
            "email": "admin@example.invalid",
            "display_name": "The Admin",
            "password": "a-plainly-fictional-test-password",
        },
    )
    assert created.status_code == 201, created.text
    admin_id = created.json()["id"]
    token = issue_access_token(user_id=admin_id, role="admin", security=security)
    client.cookies.set(security.cookie_name, token)

    write_response = client.put(
        "/api/credentials/stt_api_key", json={"value": _REAL_SECRET_VALUE}
    )
    assert write_response.status_code == 200, write_response.text
    assert _REAL_SECRET_VALUE not in write_response.text, (
        "the write route's own response must not echo the value back"
    )

    flat_routes = _flatten_routes(app.routes)
    checked_any = False
    for route in flat_routes:
        path = getattr(route, "path", None) or getattr(route, "path_format", None)
        if path is None:
            continue
        methods = getattr(route, "methods", None)
        if not methods or "GET" not in methods:
            continue
        checked_any = True
        response = client.get(_fill_path_params(path))
        assert _REAL_SECRET_VALUE not in response.text, (
            f"GET {path} returned the stored plaintext credential in its response body"
        )

    assert checked_any, "no GET route was walked -- this test would pass vacuously"

    # The listing route specifically: no ciphertext, no plaintext, no
    # partial value -- not a prefix, not a suffix, not a length hint.
    listing = client.get("/api/credentials")
    assert listing.status_code == 200, listing.text
    entries = {e["slot"]: e for e in listing.json()}
    stt_entry = entries["stt_api_key"]
    assert set(stt_entry) == {"slot", "label", "is_set", "updated_at", "source", "applies_live"}, (
        f"the listing entry carries an unexpected field: {stt_entry!r}"
    )
    assert stt_entry["is_set"] is True
    assert stt_entry["source"] == "database"
    assert stt_entry["applies_live"] is False, (
        "a provider credential needs a restart to take effect -- the providers "
        "it feeds are constructed once in lifespan, and this field is the "
        "server-stated fact the webapp's badge reads, never a browser guess"
    )


def test_a_stored_credential_is_unreadable_without_the_key(monkeypatch):
    """A stored ciphertext cannot be decrypted with a different secret,
    and a corrupted ciphertext raises rather than returning something
    that looks like it worked."""
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    ciphertext, key_version = encrypt_credential(_REAL_SECRET_VALUE, security)

    assert _REAL_SECRET_VALUE.encode("utf-8") not in ciphertext, (
        "the ciphertext must not contain the plaintext in any recognizable form -- "
        "this is not a base64-shaped no-op"
    )

    # Round trip with the correct key.
    assert decrypt_credential(ciphertext, key_version, security) == _REAL_SECRET_VALUE

    # A different secret cannot decrypt it.
    monkeypatch.setenv("ATLAS_SECRET_KEY", _OTHER_SECRET_KEY)
    different_security = SecurityConfig()
    with pytest.raises(InvalidToken):
        decrypt_credential(ciphertext, key_version, different_security)

    # A corrupted ciphertext raises rather than returning something.
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    corrupted = ciphertext[:-1] + (b"\x00" if ciphertext[-1:] != b"\x00" else b"\x01")
    with pytest.raises(InvalidToken):
        decrypt_credential(corrupted, key_version, security)


def test_the_credential_key_and_the_jwt_signing_key_derive_independently(monkeypatch):
    """One operator-supplied `ATLAS_SECRET_KEY` yields two different
    derived values -- the credential-encryption key
    (`atlas.crypto.credentials`) and the JWT signing key
    (`atlas.auth.tokens`) -- under two distinct HKDF labels, so
    neither can be used to attack the other."""
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    secret = read_secret_key(security)

    credential_key = _derive_key(secret, _CREDENTIAL_ENCRYPTION_INFO)
    jwt_key = _derive_key(secret, _JWT_SIGNING_INFO)

    assert credential_key != jwt_key


def test_a_write_to_an_unknown_slot_is_refused(
    monkeypatch, fake_account_repository, fake_policy_repository, fake_credential_repository
):
    """The slot set is closed and defined in code -- a write to a name
    outside `CredentialSlot` is refused before it ever reaches
    `provider_credentials`."""
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    policy_repo = fake_policy_repository()
    credential_repo = fake_credential_repository()

    app = _build_full_app(security, account_repo, policy_repo, credential_repo)
    client = TestClient(app)
    created = client.post(
        "/api/auth/create-admin",
        json={
            "email": "admin2@example.invalid",
            "display_name": "The Admin",
            "password": "a-plainly-fictional-test-password",
        },
    )
    assert created.status_code == 201, created.text
    token = issue_access_token(user_id=created.json()["id"], role="admin", security=security)
    client.cookies.set(security.cookie_name, token)

    response = client.put(
        "/api/credentials/not_a_real_slot", json={"value": "whatever"}
    )
    assert response.status_code == 400
    assert "not_a_real_slot" in response.json()["detail"]
    assert credential_repo.credentials == {}, "an unknown slot must never be stored"


def test_a_missing_secret_key_raises_configerror_naming_the_variable(monkeypatch):
    """A missing `ATLAS_SECRET_KEY` raises `ConfigError` naming the
    variable at the moment a credential is read or written, never a
    silently generated ephemeral key -- an encryption key that changes on
    every restart would make every previously-encrypted credential
    permanently unreadable.

    `ConfigError` is read off `atlas.config` fresh, here, rather
    than the name this file imported at collection time --
    `tests/test_config.py::test_config_and_turn_macros_import_in_either_order`
    reloads `atlas.config`, which mints a *new* `ConfigError` class
    distinct from the one a module-level `from atlas.config import
    ConfigError` bound before that reload ran. `read_secret_key` itself
    always raises whatever class is currently bound in `config.py`'s own
    namespace, so a stale import here would build a `pytest.raises` that
    can never match it once that reload has run earlier in the same test
    session (the exact gotcha `tests/test_startup_smoke.py` documents for
    `app_module.ConfigError`)."""
    import atlas.config as config_module

    monkeypatch.delenv("ATLAS_SECRET_KEY", raising=False)
    security = SecurityConfig()

    with pytest.raises(config_module.ConfigError, match="ATLAS_SECRET_KEY"):
        encrypt_credential(_REAL_SECRET_VALUE, security)

    with pytest.raises(config_module.ConfigError, match="ATLAS_SECRET_KEY"):
        decrypt_credential(b"not-a-real-ciphertext", 1, security)


# --- Task 3: the resolution order, unit-level (D-07) ------------------------


def _config_with_env_value(value: str) -> SimpleNamespace:
    """A `Config`-shaped stand-in whose `stt.api_key` holds `value` --
    stands in for what `config.py`'s `${XAI_API_KEY}` expansion would
    have already produced by the time `lifespan` reads it."""
    return SimpleNamespace(
        stt=SimpleNamespace(api_key=value),
        brain=SimpleNamespace(api_key=""),
        tts=SimpleNamespace(api_key=""),
        mcp_servers={},
    )


async def test_a_value_in_both_places_the_database_wins(monkeypatch, fake_credential_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    repo = fake_credential_repository()
    ciphertext, key_version = encrypt_credential("the-database-value", security)
    await repo.upsert_credential(
        CredentialSlot.STT.value,
        ciphertext=ciphertext,
        key_version=key_version,
        updated_by_user_id=None,
    )
    config = SimpleNamespace(
        security=security, **_config_with_env_value("the-environment-value").__dict__
    )

    is_set, source, updated_at = await resolve_credential_source(CredentialSlot.STT, repo, config)
    assert is_set is True
    assert source == "database"
    assert updated_at is not None

    value, value_source = await resolve_credential_value(CredentialSlot.STT, config, repo)
    assert value == "the-database-value"
    assert value_source == "database"


async def test_a_value_in_only_the_environment_is_used_and_reported(
    monkeypatch, fake_credential_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    repo = fake_credential_repository()  # empty -- no database row
    config = SimpleNamespace(
        security=security, **_config_with_env_value("the-environment-value").__dict__
    )

    is_set, source, updated_at = await resolve_credential_source(CredentialSlot.STT, repo, config)
    assert is_set is True
    assert source == "environment"
    assert updated_at is None

    value, value_source = await resolve_credential_value(CredentialSlot.STT, config, repo)
    assert value == "the-environment-value"
    assert value_source == "environment"


async def test_neither_source_is_unset(monkeypatch, fake_credential_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    repo = fake_credential_repository()
    config = SimpleNamespace(security=security, **_config_with_env_value("").__dict__)

    is_set, source, updated_at = await resolve_credential_source(CredentialSlot.STT, repo, config)
    assert is_set is False
    assert source == "unset"
    assert updated_at is None

    value, value_source = await resolve_credential_value(CredentialSlot.STT, config, repo)
    assert value == ""
    assert value_source == "unset"


def test_the_home_assistant_token_is_written_where_the_assistant_reads_it(
    monkeypatch, fake_account_repository, fake_policy_repository, fake_credential_repository
):
    """WR-06 (code review): plan 06-01 moved the Home Assistant token into
    `plugin_config_values` (D-01) and left the `ha_token` credential slot
    behind. `GET /api/credentials` still listed it, `PUT /api/credentials/
    ha_token` still encrypted and stored a row, `resolve_credential_source`
    still called it `database` once written -- and nothing anywhere read
    that row again. An operator rotating the token on the settings screen,
    or typing it into the setup wizard's own hub step (which posts to
    exactly this route), saved it into a table no code consults and was
    told a restart would apply it. No restart would.

    The slot's storage is the Home Assistant plugin's own `HA_TOKEN`
    configuration value now. This asserts the write lands where the
    readers actually look: `routes/wizard.py`'s hub check reads it back
    through its own `_ha_plugin_connection_info`, which is the same
    decryption `PluginManager` performs when it spawns the child.
    """
    import asyncio
    from datetime import datetime, timezone

    from atlas.db.repository import Plugin, PluginConfigValue
    from atlas.routes.wizard import _ha_plugin_connection_info

    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    now = datetime.now(timezone.utc)
    ha = Plugin(
        id=1, slug="ha", display_name="Home Assistant", transport="stdio",
        args=("-m", "atlas_mcp.ha"), url=None, enabled=True, builtin=True,
        enforces_policy=True, timeout_ms=5000, created_at=now, updated_at=now,
        created_by_user_id=None,
    )
    plugin_repo = conftest.FakePluginRepository(
        plugins=[ha],
        config_values={
            1: [
                PluginConfigValue(
                    key="HA_URL", secret=False, value="http://hub.invalid:8123",
                    ciphertext=None, key_version=None,
                )
            ]
        },
    )

    app = _build_full_app(security, account_repo, fake_policy_repository(), fake_credential_repository())
    app.state.plugin_repo = plugin_repo
    client = TestClient(app)

    created = client.post(
        "/api/auth/create-admin",
        json={
            "email": "admin@example.invalid",
            "display_name": "The Admin",
            "password": "a-plainly-fictional-test-password",
        },
    )
    assert created.status_code == 201, created.text
    client.cookies.set(
        security.cookie_name,
        issue_access_token(user_id=created.json()["id"], role="admin", security=security),
    )

    before = {entry["slot"]: entry for entry in client.get("/api/credentials").json()}
    assert before["ha_token"]["is_set"] is False
    assert before["ha_token"]["source"] == "unset"

    written = client.put("/api/credentials/ha_token", json={"value": _REAL_SECRET_VALUE})
    assert written.status_code == 200, written.text
    assert _REAL_SECRET_VALUE not in written.text
    assert written.json()["is_set"] is True
    assert written.json()["source"] == "database"

    # The one thing the old slot could not do: reach a reader.
    base_url, token = asyncio.run(_ha_plugin_connection_info(plugin_repo, security))
    assert base_url == "http://hub.invalid:8123", "the other configuration values survive the write"
    assert token == _REAL_SECRET_VALUE

    # Still write-only, and still encrypted at rest.
    listing = client.get("/api/credentials")
    assert _REAL_SECRET_VALUE not in listing.text
    [stored] = [v for v in plugin_repo.config_values[1] if v.key == "HA_TOKEN"]
    assert stored.secret is True
    assert stored.value is None
    assert _REAL_SECRET_VALUE.encode("utf-8") not in stored.ciphertext
