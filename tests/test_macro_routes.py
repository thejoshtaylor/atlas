"""Macro authoring over HTTP (MACRO-03): the routes give an operator a
control surface over the macros plan 04-05 moved into the database, and
flag a conflict with the running safety policy at authoring time, using
the exact same check the fire path uses (D-10, Pitfall 5).

Every test here builds a small, throwaway `FastAPI()` app carrying only
`spire_voice.routes.macros`'s own router -- the same "primitives in
isolation" shape `tests/test_policy_routes.py` already uses, since this
file's whole point is the macro routes themselves, not the rest of the
application.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig
from spire_voice.providers.tts_xai import SinkFormat
from spire_voice.routes.macros import router as macros_router

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


def _build_macro_app(
    security,
    account_repo,
    macro_repo,
    *,
    policy_repo=None,
    tool_host=None,
    tts=None,
    filler_cache=None,
    cache_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.config = SimpleNamespace(
        security=security,
        tts=SimpleNamespace(
            cache_dir=str(cache_dir) if cache_dir is not None else "/tmp/spire-test-tts-cache",
            voice_id="eve",
        ),
    )
    app.state.account_repo = account_repo
    app.state.macro_repo = macro_repo
    app.state.policy_repo = policy_repo
    app.state.tool_host = tool_host
    app.state.tts = tts
    app.state.filler_cache = filler_cache if filler_cache is not None else {}
    app.include_router(macros_router)
    return app


def _issue_cookie(security, account_repo, *, role: str):
    import asyncio

    user = asyncio.run(
        account_repo.create_user(
            email=f"{role}@example.invalid",
            display_name=f"A {role.title()}",
            password_hash="not-checked-by-this-test",
            role=role,
        )
    )
    return user


class _EntityListingToolHost:
    """A fake `tool_host` whose `ha_list_entities` reports a fixed live
    catalog -- the shape `routes/macros.py::_known_entity_ids` reads."""

    def __init__(self, known_entity_ids: list[str]) -> None:
        self._known_entity_ids = known_entity_ids
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append(name)
        assert name == "ha_list_entities"
        return SimpleNamespace(
            structuredContent=[{"entity_id": e, "friendly_name": e} for e in self._known_entity_ids],
            content=[],
        )


class _UnreachableToolHost:
    """`call_tool` always raises -- simulates Home Assistant being
    unreachable, so `_known_entity_ids` must return `None`, not an empty
    catalog."""

    async def call_tool(self, name: str, arguments: dict):
        raise RuntimeError("simulated home assistant unreachable")


class _RaisingPolicyRepository:
    """`load_policy` always raises -- simulates a policy read failure
    independent of the catalog, so `_load_policy_or_none` must return
    `None`."""

    async def load_policy(self):
        raise RuntimeError("simulated policy read failure")


class _FakeMacroTts:
    """A `precache_all`-compatible fake: `browser_sink()` plus a
    `synthesize(text_deltas, sink=...)` that records the text it was
    asked to render and either succeeds with scripted bytes or raises."""

    def __init__(self, chunks: tuple[bytes, ...] = (b"synthesized-audio",), fail: bool = False) -> None:
        self._chunks = chunks
        self._fail = fail
        self.synthesized_texts: list[str] = []
        self.sinks: list = []

    def browser_sink(self) -> SinkFormat:
        return SinkFormat(codec="pcm", sample_rate=24000)

    async def synthesize(self, text_deltas, sink=None):
        text = "".join([delta async for delta in text_deltas])
        self.synthesized_texts.append(text)
        self.sinks.append(sink)
        if self._fail:
            raise RuntimeError("simulated synthesis failure")
        for chunk in self._chunks:
            yield chunk


class _NoOpMacroToolHost:
    """A `tool_host` a fired macro action can call against -- always
    succeeds, records nothing else. Distinct from the conflict-check
    `tool_host` `app.state` carries, matching how `fire_macro` takes its
    own `tool_host` argument in `run_turn`, independent of the route."""

    async def call_tool(self, name: str, arguments: dict):
        from types import SimpleNamespace

        return SimpleNamespace(isError=False, content=[])


def _macro_kwargs(
    phrase: str = "good night",
    aliases: tuple[str, ...] = (),
    reply: str = "good night",
    actions: tuple[tuple[str, dict], ...] = (
        ("ha_call_service", {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"}),
    ),
) -> tuple[str, tuple[str, ...], str, tuple[tuple[str, dict], ...]]:
    return phrase, aliases, reply, actions


# --- Task 1: list, read, and the three-state conflict annotation --------


def test_listing_macros_returns_phrase_aliases_reply_and_ordered_actions(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(
        macros=[
            _macro_kwargs(
                phrase="good night",
                aliases=("goodnight",),
                reply="good night",
                actions=(
                    ("ha_call_service", {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_a"}),
                    ("ha_call_service", {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_b"}),
                ),
            )
        ]
    )
    policy_repo = fake_policy_repository()
    tool_host = _EntityListingToolHost(["switch.example_a", "switch.example_b"])

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo, tool_host=tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get("/api/macros")
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 1
    macro = body[0]
    assert macro["phrase"] == "good night"
    assert macro["aliases"] == ["goodnight"]
    assert macro["reply"] == "good night"
    assert [a["arguments"]["entity_id"] for a in macro["actions"]] == [
        "switch.example_a",
        "switch.example_b",
    ]


def test_reading_one_macro_by_id(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(macros=[_macro_kwargs(phrase="good night")])
    policy_repo = fake_policy_repository()
    tool_host = _EntityListingToolHost(["switch.example_fan"])

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo, tool_host=tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    macro_id = next(iter(macro_repo.macros))
    response = client.get(f"/api/macros/{macro_id}")
    assert response.status_code == 200, response.text
    assert response.json()["phrase"] == "good night"


def test_reading_an_unknown_macro_id_returns_a_named_404(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get("/api/macros/999")
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_an_action_targeting_a_denied_entity_is_annotated_denied(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(
        macros=[
            _macro_kwargs(
                actions=(
                    (
                        "ha_call_service",
                        {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_denied"},
                    ),
                )
            )
        ]
    )
    policy_repo = fake_policy_repository(deny_entities=["switch.example_denied"])
    tool_host = _EntityListingToolHost(["switch.example_denied"])

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo, tool_host=tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get("/api/macros")
    assert response.status_code == 200, response.text
    assert response.json()[0]["actions"][0]["conflict"] == "denied"


def test_an_action_targeting_an_entity_absent_from_the_catalog_is_annotated_not_found(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(
        macros=[
            _macro_kwargs(
                actions=(
                    (
                        "ha_call_service",
                        {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_gone"},
                    ),
                )
            )
        ]
    )
    policy_repo = fake_policy_repository()
    # A non-empty catalog that simply does not carry this entity -- an
    # empty catalog is indistinguishable from "could not fetch" through
    # `_tool_result_json`'s own falsy-payload handling (matching
    # `routes/policy.py`'s identical behavior), so this test uses a
    # catalog with an unrelated entity present instead.
    tool_host = _EntityListingToolHost(["switch.example_unrelated"])

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo, tool_host=tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get("/api/macros")
    assert response.status_code == 200, response.text
    assert response.json()[0]["actions"][0]["conflict"] == "not_found"


def test_the_unknown_annotation_is_returned_when_the_catalog_cannot_be_read_and_differs_from_no_conflict(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(
        macros=[
            _macro_kwargs(
                actions=(
                    (
                        "ha_call_service",
                        {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"},
                    ),
                )
            )
        ]
    )
    policy_repo = fake_policy_repository()
    tool_host = _UnreachableToolHost()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo, tool_host=tool_host)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get("/api/macros")
    assert response.status_code == 200, response.text
    conflict = response.json()[0]["actions"][0]["conflict"]
    assert conflict == "unknown"
    assert conflict != "ok", "an unknown check must never render as no conflict"


def test_the_unknown_annotation_is_returned_when_the_policy_cannot_be_read(
    monkeypatch, fake_account_repository, fake_macro_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(
        macros=[
            _macro_kwargs(
                actions=(
                    (
                        "ha_call_service",
                        {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"},
                    ),
                )
            )
        ]
    )
    tool_host = _EntityListingToolHost(["switch.example_fan"])

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(
        security, account_repo, macro_repo, policy_repo=_RaisingPolicyRepository(), tool_host=tool_host
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.get("/api/macros")
    assert response.status_code == 200, response.text
    assert response.json()[0]["actions"][0]["conflict"] == "unknown"


def test_reading_and_listing_reject_an_unauthenticated_request_and_a_viewer(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(macros=[_macro_kwargs()])
    policy_repo = fake_policy_repository()

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    anonymous_client = TestClient(app)
    assert anonymous_client.get("/api/macros").status_code == 401

    viewer = _issue_cookie(security, account_repo, role="viewer")
    token = issue_access_token(user_id=viewer.id, role="viewer", security=security)
    viewer_client = TestClient(app, cookies={security.cookie_name: token})
    assert viewer_client.get("/api/macros").status_code == 403


# --- Task 2: write, refused by the same rules the file parser uses ------


def _action_json(tool: str = "ha_call_service", **arguments) -> dict:
    return {"tool": tool, "arguments": arguments or {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"}}


def test_creating_a_macro_with_a_phrase_reply_and_actions_returns_it_stored_in_order(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros",
        json={
            "phrase": "good night",
            "aliases": ["goodnight"],
            "reply": "good night",
            "actions": [
                _action_json(entity_id="switch.example_a"),
                _action_json(entity_id="switch.example_b"),
            ],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["phrase"] == "good night"
    assert [a["arguments"]["entity_id"] for a in body["actions"]] == [
        "switch.example_a",
        "switch.example_b",
    ]
    assert len(macro_repo.macros) == 1


def test_creating_a_macro_with_no_actions_is_refused(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros", json={"phrase": "no actions", "aliases": [], "reply": "ok", "actions": []}
    )
    assert response.status_code == 400
    assert "no actions" in response.json()["detail"]
    assert not macro_repo.macros


def test_creating_a_macro_with_a_blank_phrase_is_refused(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    """MED-01 fix (phase 4 code review): `MacroConfig.from_config` (the
    file parser) has always refused a blank `phrase` -- this route did
    not, so an operator could save a macro `normalize("")` matches no
    transcript against, silently dead with no error at save time."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros",
        json={"phrase": "", "aliases": [], "reply": "ok", "actions": [_action_json()]},
    )
    assert response.status_code == 400
    assert "phrase" in response.json()["detail"]
    assert not macro_repo.macros


def test_creating_a_macro_with_a_blank_reply_is_refused(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    """MED-01 fix (phase 4 code review): the file parser's identical
    refusal for a blank `reply` -- a blank reply is handed to
    `precache_all` on save, attempting to synthesize zero-length speech."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros",
        json={"phrase": "blank reply", "aliases": [], "reply": "", "actions": [_action_json()]},
    )
    assert response.status_code == 400
    assert "reply" in response.json()["detail"]
    assert not macro_repo.macros


def test_updating_a_macro_with_a_blank_phrase_or_reply_is_refused(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    """MED-01 fix: the same two checks apply to `update_macro`, not only
    `create_macro` -- an operator clearing the Phrase or Reply field while
    editing an existing macro must be refused the same way."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(macros=[_macro_kwargs(phrase="good night")])
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})
    [macro_id] = list(macro_repo.macros.keys())

    blank_phrase = client.put(
        f"/api/macros/{macro_id}",
        json={"phrase": "", "aliases": [], "reply": "ok", "actions": [_action_json()]},
    )
    assert blank_phrase.status_code == 400
    assert "phrase" in blank_phrase.json()["detail"]

    blank_reply = client.put(
        f"/api/macros/{macro_id}",
        json={"phrase": "good night", "aliases": [], "reply": "", "actions": [_action_json()]},
    )
    assert blank_reply.status_code == 400
    assert "reply" in blank_reply.json()["detail"]

    # Neither refused write must have touched the stored macro.
    assert macro_repo.macros[macro_id].phrase == "good night"


def test_creating_a_macro_whose_phrase_collides_with_an_existing_one_is_refused_naming_both(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(macros=[_macro_kwargs(phrase="Good Night")])
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros",
        json={
            "phrase": "good night",  # normalizes to the same key as "Good Night"
            "aliases": [],
            "reply": "ok",
            "actions": [_action_json()],
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Good Night" in detail and "good night" in detail
    assert len(macro_repo.macros) == 1, "the colliding macro must never be stored"


def test_creating_a_macro_action_with_no_tool_name_is_refused(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository()
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros",
        json={
            "phrase": "no tool",
            "aliases": [],
            "reply": "ok",
            "actions": [{"tool": "", "arguments": {}}],
        },
    )
    assert response.status_code == 400
    assert not macro_repo.macros


def test_updating_a_macro_replaces_actions_wholesale_and_a_reorder_persists(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(
        macros=[
            _macro_kwargs(
                phrase="good night",
                actions=(
                    ("ha_call_service", {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_a"}),
                    ("ha_call_service", {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_b"}),
                ),
            )
        ]
    )
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    macro_id = next(iter(macro_repo.macros))
    response = client.put(
        f"/api/macros/{macro_id}",
        json={
            "phrase": "good night",
            "aliases": [],
            "reply": "good night",
            "actions": [
                _action_json(entity_id="switch.example_b"),
                _action_json(entity_id="switch.example_a"),
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert [a["arguments"]["entity_id"] for a in response.json()["actions"]] == [
        "switch.example_b",
        "switch.example_a",
    ]


def test_updating_a_macro_to_a_phrase_that_collides_with_a_different_macro_is_refused(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(
        macros=[
            _macro_kwargs(phrase="good morning"),
            _macro_kwargs(phrase="good night"),
        ]
    )
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    macro_ids = list(macro_repo.macros)
    good_night_id = next(m.id for m in macro_repo.macros.values() if m.phrase == "good night")

    response = client.put(
        f"/api/macros/{good_night_id}",
        json={"phrase": "good morning", "aliases": [], "reply": "ok", "actions": [_action_json()]},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "good morning" in detail
    assert macro_repo.macros[good_night_id].phrase == "good night", "the update must not have landed"


def test_updating_a_macro_to_its_own_existing_phrase_succeeds(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(macros=[_macro_kwargs(phrase="good night")])
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    macro_id = next(iter(macro_repo.macros))
    response = client.put(
        f"/api/macros/{macro_id}",
        json={
            "phrase": "good night",  # unchanged -- self-collision must be legal
            "aliases": [],
            "reply": "updated reply",
            "actions": [_action_json()],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["reply"] == "updated reply"


def test_deleting_a_macro_removes_it_and_deleting_again_is_not_an_error(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(macros=[_macro_kwargs(phrase="good night")])
    policy_repo = fake_policy_repository()

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    macro_id = next(iter(macro_repo.macros))
    response = client.delete(f"/api/macros/{macro_id}")
    assert response.status_code == 204
    assert macro_id not in macro_repo.macros

    # Deleting it again is not an error.
    response = client.delete(f"/api/macros/{macro_id}")
    assert response.status_code == 204


def test_every_write_route_rejects_a_viewer(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(macros=[_macro_kwargs(phrase="good night")])
    policy_repo = fake_policy_repository()

    viewer = _issue_cookie(security, account_repo, role="viewer")
    token = issue_access_token(user_id=viewer.id, role="viewer", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    macro_id = next(iter(macro_repo.macros))
    assert (
        client.post(
            "/api/macros",
            json={"phrase": "x", "aliases": [], "reply": "x", "actions": [_action_json()]},
        ).status_code
        == 403
    )
    assert (
        client.put(
            f"/api/macros/{macro_id}",
            json={"phrase": "good night", "aliases": [], "reply": "x", "actions": [_action_json()]},
        ).status_code
        == 403
    )
    assert client.delete(f"/api/macros/{macro_id}").status_code == 403
    assert macro_id in macro_repo.macros, "a viewer's delete attempt must not have landed"


# --- Task 3: the save's side effect, awaited, and a synthesis failure ---
# that does not lose the edit -------------------------------------------


def test_creating_a_macro_synthesizes_its_reply_before_returning_and_merges_it_into_the_turn_cache(
    monkeypatch, tmp_path, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository()
    policy_repo = fake_policy_repository()
    tts = _FakeMacroTts(chunks=(b"good-night-audio",))
    filler_cache: dict = {}

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(
        security,
        account_repo,
        macro_repo,
        policy_repo=policy_repo,
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros",
        json={
            "phrase": "good night",
            "aliases": [],
            "reply": "good night, sleeping now",
            "actions": [_action_json()],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["reply_synthesis_degraded"] is False
    assert body["reply_cached"] is True
    assert tts.synthesized_texts == ["good night, sleeping now"]
    # The same dict object the app's filler_cache started as -- proves the
    # merge happened in place, the exact cache a turn reads from.
    assert filler_cache["good night, sleeping now"] == b"good-night-audio"


def test_creating_a_macro_also_caches_its_reply_for_the_camera_sink(
    monkeypatch, tmp_path, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    """A macro saved after boot must be speakable on the camera, not only
    in the browser: the reply is synthesized for every sink in
    `app.state.filler_caches`, each in that sink's own format."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    tts = _FakeMacroTts(chunks=(b"reply-audio",))
    filler_cache: dict = {}
    camera_cache: dict = {}

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    app = _build_macro_app(
        security,
        account_repo,
        fake_macro_repository(),
        policy_repo=fake_policy_repository(),
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    app.state.filler_caches = {None: filler_cache, ("alaw", 8000): camera_cache}
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros",
        json={"phrase": "good night", "aliases": [], "reply": "good night, sleeping now", "actions": [_action_json()]},
    )
    assert response.status_code == 201, response.text
    assert "good night, sleeping now" in filler_cache
    assert "good night, sleeping now" in camera_cache
    assert SinkFormat(codec="alaw", sample_rate=8000) in tts.sinks


def test_updating_a_macro_with_an_unchanged_reply_does_not_resynthesize(
    monkeypatch, tmp_path, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(
        macros=[_macro_kwargs(phrase="good night", reply="good night, sleeping now")]
    )
    policy_repo = fake_policy_repository()
    tts = _FakeMacroTts()
    filler_cache: dict = {"good night, sleeping now": b"already-cached-audio"}

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(
        security,
        account_repo,
        macro_repo,
        policy_repo=policy_repo,
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    macro_id = next(iter(macro_repo.macros))
    response = client.put(
        f"/api/macros/{macro_id}",
        json={
            "phrase": "good night",
            "aliases": [],
            "reply": "good night, sleeping now",  # unchanged
            "actions": [_action_json(entity_id="switch.example_reordered")],
        },
    )
    assert response.status_code == 200, response.text
    assert tts.synthesized_texts == [], "an unchanged reply must never trigger a re-synthesis call"
    assert response.json()["reply_cached"] is True


def test_a_synthesis_failure_returns_a_degraded_success_and_the_macro_is_still_readable(
    monkeypatch, tmp_path, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository()
    policy_repo = fake_policy_repository()
    tts = _FakeMacroTts(fail=True)
    filler_cache: dict = {}

    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(
        security,
        account_repo,
        macro_repo,
        policy_repo=policy_repo,
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros",
        json={
            "phrase": "good night",
            "aliases": [],
            "reply": "this reply will fail to synthesize",
            "actions": [_action_json()],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["reply_synthesis_degraded"] is True
    assert body["reply_synthesis_message"]
    assert "this reply will fail to synthesize" not in filler_cache

    # The macro is still readable, with its new content -- the failed
    # synthesis did not lose the operator's edit.
    macro_id = body["id"]
    read_back = client.get(f"/api/macros/{macro_id}")
    assert read_back.status_code == 200
    assert read_back.json()["reply"] == "this reply will fail to synthesize"


async def test_a_macro_saved_through_the_route_matches_on_the_next_turn_with_no_restart(
    monkeypatch,
    tmp_path,
    fake_account_repository,
    fake_macro_repository,
    fake_policy_repository,
    fake_audio_source,
    fake_stt,
    fake_brain,
    fake_tts,
):
    """T-04-33's own proof: a macro saved through the route is matched by
    a real `run_turn` call, in the same test, with no restart in between
    -- driven from the exact `Macro` the repository now holds and the
    exact `filler_cache` dict the save route updated in place."""
    from spire_voice.providers.base import FinalTranscript
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository()
    policy_repo = fake_policy_repository()
    tts = _FakeMacroTts(chunks=(b"good-night-live-audio",))
    filler_cache: dict = {}

    # `_issue_cookie` calls `asyncio.run(...)`, which cannot run inside
    # this test's own already-running event loop (this test is `async
    # def`, unlike its sync siblings above) -- await the repository call
    # directly instead.
    operator = await account_repo.create_user(
        email="operator@example.invalid",
        display_name="An Operator",
        password_hash="not-checked-by-this-test",
        role="operator",
    )
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(
        security,
        account_repo,
        macro_repo,
        policy_repo=policy_repo,
        tts=tts,
        filler_cache=filler_cache,
        cache_dir=tmp_path,
    )
    client = TestClient(app, cookies={security.cookie_name: token})

    response = client.post(
        "/api/macros",
        json={
            "phrase": "good night",
            "aliases": [],
            "reply": "good night, sleeping now",
            "actions": [_action_json(entity_id="switch.example_lamp")],
        },
    )
    assert response.status_code == 201, response.text
    macro_id = response.json()["id"]

    # No restart: read the macro straight back through the same live
    # repository -- exactly what `_current_macros` (`app.py`) would hand
    # the very next turn.
    saved_macro = await macro_repo.get_macro(macro_id)

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="good night")])
    brain = fake_brain(replies=[])  # never called -- a macro hit skips the brain
    live_tts = fake_tts(chunks=[])  # never called -- the reply is cacheable
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        live_tts,
        _NoOpMacroToolHost(),
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        macros=(saved_macro,),
        # 260922-cts: `run_turn`'s own `filler_cache` is now keyed by
        # `(codec, sample_rate)`, with `None` for the browser default --
        # `source` here (`fake_audio_source`) declares no `sink_format`,
        # so it resolves to `None`, the same key `app.state.filler_cache`
        # (the flat, browser-only dict the route under test just
        # populated) belongs under.
        filler_cache={None: app.state.filler_cache},
    )

    assert timings.turn_outcome == "macro"
    assert source.sent_audio == [b"good-night-live-audio"]
    assert len(live_tts.received_text) == 0


async def test_deleting_a_macro_removes_it_from_what_the_next_turn_can_match(
    monkeypatch, fake_account_repository, fake_macro_repository, fake_policy_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    macro_repo = fake_macro_repository(macros=[_macro_kwargs(phrase="good night")])
    policy_repo = fake_policy_repository()

    operator = await account_repo.create_user(
        email="operator@example.invalid",
        display_name="An Operator",
        password_hash="not-checked-by-this-test",
        role="operator",
    )
    token = issue_access_token(user_id=operator.id, role="operator", security=security)

    app = _build_macro_app(security, account_repo, macro_repo, policy_repo=policy_repo)
    client = TestClient(app, cookies={security.cookie_name: token})

    macro_id = next(iter(macro_repo.macros))
    response = client.delete(f"/api/macros/{macro_id}")
    assert response.status_code == 204

    from spire_voice.turn.macros import match

    remaining = await macro_repo.list_macros()
    assert match(remaining, "good night") is None
