"""Tests for `atlas_mcp.ha_spotify`: the spoken-name matcher and
`handle_play_spotify_playlist`, driven against a scripted Home Assistant
(ordered HTTP responses) and a fake websocket browser.

Every entity, speaker, and playlist below is invented.
"""

import json

import httpx
import pytest

import atlas_mcp.ha as ha_module
from atlas_mcp.ha_spotify import handle_play_spotify_playlist, match_spoken_name
from atlas_mcp.registry import RegistryUnavailableError
from atlas_mcp.safety import Denied, Policy

_ENTITY = "media_player.example_spotify"
_ACTIVE_FEATURES = 512 | 131072 | 16384
_IDLE_FEATURES = 2048

_PLAYLISTS = [
    {
        "title": "Love Refined",
        "media_content_id": "spotify:playlist:example1",
        "media_content_type": "spotify://playlist",
        "can_play": True,
    },
    {
        "title": "Morning Run",
        "media_content_id": "spotify:playlist:example2",
        "media_content_type": "spotify://playlist",
        "can_play": True,
    },
    {
        "title": "Dinner Jazz",
        "media_content_id": "spotify:playlist:example3",
        "media_content_type": "spotify://playlist",
        "can_play": True,
    },
]


class _ScriptedHa:
    """Answers a fixed, ordered script of responses, one per request. A
    request past the end of the script raises, so a runaway retry fails the
    test instead of looping."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected extra request to home assistant")
        return self._responses.pop(0)

    @property
    def posts(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST"]


class _FakeBrowser:
    def __init__(self, result=None, raises: Exception | None = None) -> None:
        self.commands: list[dict] = []
        self._result = result if result is not None else {"children": _PLAYLISTS}
        self._raises = raises

    async def run_command(self, command):
        self.commands.append(dict(command))
        if self._raises is not None:
            raise self._raises
        return self._result


def _state(features: int, *, source_list=None, source=None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "entity_id": _ENTITY,
            "state": "idle" if features == _IDLE_FEATURES else "playing",
            "attributes": {
                "supported_features": features,
                "source_list": source_list if source_list is not None else [],
                "source": source,
            },
        },
    )


def _ok() -> httpx.Response:
    return httpx.Response(200, json=[{"entity_id": _ENTITY, "state": "playing"}])


async def _play(scripted: _ScriptedHa, browser, name="love refined", *, entity=_ENTITY, source=None, policy=None, default_source=None):
    try:
        return await handle_play_spotify_playlist(
            policy or Policy.from_config(None),
            scripted.client,
            "http://ha.invalid",
            "test-token",
            browser,
            entity,
            name,
            source=source,
            default_source=default_source,
        )
    finally:
        await scripted.client.aclose()


_TITLES = ["Love Refined", "Morning Run", "Dinner Jazz"]


# -- matcher ---------------------------------------------------------------


@pytest.mark.parametrize(
    "spoken",
    ["love refined", "my Love Refined playlist.", "lof refined", "My Love"],
)
def test_matcher_picks_love_refined(spoken):
    assert match_spoken_name(spoken, _TITLES, what="playlist") == "Love Refined"


def test_matcher_ties_name_both_titles():
    with pytest.raises(Denied) as exc:
        match_spoken_name("love", ["Love Refined", "Love Songs"], what="playlist")
    assert "Love Refined" in exc.value.reason
    assert "Love Songs" in exc.value.reason


def test_matcher_no_match_names_the_closest():
    titles = ["One", "Two", "Three", "Four", "Morning Run"]
    with pytest.raises(Denied) as exc:
        match_spoken_name("banana", titles, what="playlist")
    assert exc.value.reason.startswith("i couldn't find a spotify playlist called banana")
    named = [t for t in titles if t in exc.value.reason]
    assert 1 <= len(named) <= 3


def test_matcher_empty_candidates():
    with pytest.raises(Denied, match="i can't see any spotify playlists right now"):
        match_spoken_name("anything", [], what="playlist")


def test_matcher_identical_normalized_titles_are_a_tie():
    with pytest.raises(Denied):
        match_spoken_name("love refined", ["Love Refined", "love  refined!"], what="playlist")


# -- handler ---------------------------------------------------------------


async def test_active_player_browses_then_plays():
    scripted = _ScriptedHa([_state(_ACTIVE_FEATURES, source="Example Speaker"), _ok()])
    browser = _FakeBrowser()

    result = await _play(scripted, browser)

    assert browser.commands == [
        {
            "type": "media_player/browse_media",
            "entity_id": _ENTITY,
            "media_content_type": "spotify://current_user_playlists",
            "media_content_id": "current_user_playlists",
        }
    ]
    assert [r.method for r in scripted.requests] == ["GET", "POST"]
    play = scripted.posts[0]
    assert play.url.path == "/api/services/media_player/play_media"
    assert json.loads(play.content) == {
        "entity_id": [_ENTITY],
        "media_content_id": "spotify:playlist:example1",
        "media_content_type": "playlist",
    }
    assert result["playlist"] == "Love Refined"
    assert result["changed"] == [{"entity_id": _ENTITY, "state": "playing"}]


async def test_a_child_that_is_not_a_playlist_is_never_matched():
    children = [
        {
            "title": "Love Refined",
            "media_content_id": "spotify:album:example9",
            "media_content_type": "spotify://album",
            "can_play": True,
        },
        _PLAYLISTS[1],
    ]
    scripted = _ScriptedHa([_state(_ACTIVE_FEATURES)])
    with pytest.raises(Denied):
        await _play(scripted, _FakeBrowser({"children": children}))
    assert scripted.posts == []


async def test_idle_player_selects_the_only_speaker_first():
    scripted = _ScriptedHa(
        [_state(_IDLE_FEATURES, source_list=["Example Speaker"]), _ok(), _ok()]
    )
    browser = _FakeBrowser()

    await _play(scripted, browser)

    assert [r.method for r in scripted.requests] == ["GET", "POST", "POST"]
    select, play = scripted.posts
    assert select.url.path == "/api/services/media_player/select_source"
    assert json.loads(select.content) == {"entity_id": [_ENTITY], "source": "Example Speaker"}
    assert play.url.path == "/api/services/media_player/play_media"
    assert len(browser.commands) == 1


async def test_idle_player_with_two_speakers_and_no_source_is_refused_quietly():
    scripted = _ScriptedHa(
        [_state(_IDLE_FEATURES, source_list=["Example Speaker", "Example Speaker Two"])]
    )
    browser = _FakeBrowser()

    with pytest.raises(Denied) as exc:
        await _play(scripted, browser)

    assert "Example Speaker" in exc.value.reason
    assert "Example Speaker Two" in exc.value.reason
    assert scripted.posts == []
    assert browser.commands == []


async def test_idle_player_with_no_speaker_is_refused():
    scripted = _ScriptedHa([_state(_IDLE_FEATURES, source_list=[])])
    with pytest.raises(Denied, match="can't see any speaker"):
        await _play(scripted, _FakeBrowser())
    assert scripted.posts == []


async def test_source_is_matched_loosely_against_the_speaker_list():
    scripted = _ScriptedHa(
        [
            _state(_IDLE_FEATURES, source_list=["Example Speaker", "Example Speaker Two"]),
            _ok(),
            _ok(),
        ]
    )
    await _play(scripted, _FakeBrowser(), source="example speaker two")
    assert json.loads(scripted.posts[0].content)["source"] == "Example Speaker Two"


async def test_active_player_with_a_different_named_source_selects_first():
    scripted = _ScriptedHa(
        [
            _state(_ACTIVE_FEATURES, source_list=["Example Speaker", "Example Speaker Two"], source="Example Speaker"),
            _ok(),
            _ok(),
        ]
    )
    await _play(scripted, _FakeBrowser(), source="Example Speaker Two")
    select, play = scripted.posts
    assert select.url.path.endswith("/select_source")
    assert json.loads(select.content)["source"] == "Example Speaker Two"
    assert play.url.path.endswith("/play_media")


async def test_active_player_with_no_source_does_not_select():
    scripted = _ScriptedHa(
        [_state(_ACTIVE_FEATURES, source_list=["Example Speaker"], source="Example Speaker"), _ok()]
    )
    await _play(scripted, _FakeBrowser())
    assert [p.url.path.rsplit("/", 1)[1] for p in scripted.posts] == ["play_media"]


async def test_idle_select_then_browse_failure_says_try_again():
    scripted = _ScriptedHa([_state(_IDLE_FEATURES, source_list=["Example Speaker"]), _ok()])
    browser = _FakeBrowser(raises=RegistryUnavailableError("not supported"))

    with pytest.raises(Denied) as exc:
        await _play(scripted, browser)

    assert exc.value.reason == (
        "spotify isn't ready to play on Example Speaker yet -- try again in a moment"
    )
    assert len(scripted.posts) == 1  # the select only, no play


async def test_a_denied_entity_causes_no_requests_at_all():
    scripted = _ScriptedHa([])
    browser = _FakeBrowser()
    policy = Policy.from_config({"deny_entities": [_ENTITY]})

    with pytest.raises(Denied, match="that one is off limits"):
        await _play(scripted, browser, policy=policy)

    assert scripted.requests == []
    assert browser.commands == []


async def test_a_non_media_player_entity_causes_no_requests_at_all():
    scripted = _ScriptedHa([])
    browser = _FakeBrowser()

    with pytest.raises(Denied):
        await _play(scripted, browser, entity="switch.example_fan")

    assert scripted.requests == []
    assert browser.commands == []


async def test_play_media_500_is_one_post_and_a_speakable_refusal():
    scripted = _ScriptedHa([_state(_ACTIVE_FEATURES), httpx.Response(500, text="boom")])

    with pytest.raises(Denied) as exc:
        await _play(scripted, _FakeBrowser())

    assert "Love Refined" in exc.value.reason
    assert len(scripted.posts) == 1


async def test_no_match_plays_nothing():
    scripted = _ScriptedHa([_state(_ACTIVE_FEATURES)])
    with pytest.raises(Denied):
        await _play(scripted, _FakeBrowser(), name="banana")
    assert scripted.posts == []


async def test_a_player_with_no_features_is_refused():
    scripted = _ScriptedHa([_state(0)])
    with pytest.raises(Denied, match="that spotify player can't be controlled from here"):
        await _play(scripted, _FakeBrowser())
    assert scripted.posts == []


async def test_an_empty_name_is_refused_before_any_request():
    scripted = _ScriptedHa([])
    with pytest.raises(Denied, match="which playlist do you want me to play"):
        await _play(scripted, _FakeBrowser(), name="   ")
    assert scripted.requests == []


# -- wiring ----------------------------------------------------------------


async def test_tool_is_registered_with_the_expected_schema():
    tools = {tool.name: tool for tool in await ha_module.mcp_server.list_tools()}
    tool = tools["ha_play_spotify_playlist"]
    schema = tool.input_schema if hasattr(tool, "input_schema") else tool.inputSchema
    assert set(schema["properties"]) == {"entity_id", "name", "source"}
    assert set(schema["required"]) == {"entity_id", "name"}


# -- default speaker -------------------------------------------------------

_TWO = ["Example Speaker", "Example Pi"]


async def test_default_source_is_selected_on_an_idle_player():
    scripted = _ScriptedHa([_state(_IDLE_FEATURES, source_list=_TWO), _ok(), _ok()])
    await _play(scripted, _FakeBrowser(), default_source="Example Pi")
    assert [r.method for r in scripted.requests] == ["GET", "POST", "POST"]
    select, play = scripted.posts
    assert select.url.path.endswith("/select_source")
    assert json.loads(select.content)["source"] == "Example Pi"
    assert play.url.path.endswith("/play_media")


async def test_default_source_moves_playback_off_another_speaker():
    scripted = _ScriptedHa(
        [_state(_ACTIVE_FEATURES, source_list=_TWO, source="Example Speaker"), _ok(), _ok()]
    )
    await _play(scripted, _FakeBrowser(), default_source="Example Pi")
    select, play = scripted.posts
    assert json.loads(select.content)["source"] == "Example Pi"
    assert play.url.path.endswith("/play_media")


async def test_default_source_already_playing_only_plays():
    scripted = _ScriptedHa(
        [_state(_ACTIVE_FEATURES, source_list=_TWO, source="Example Pi"), _ok()]
    )
    await _play(scripted, _FakeBrowser(), default_source="Example Pi")
    assert [p.url.path.rsplit("/", 1)[1] for p in scripted.posts] == ["play_media"]


async def test_a_named_source_wins_over_the_default():
    scripted = _ScriptedHa(
        [_state(_ACTIVE_FEATURES, source_list=_TWO, source="Example Pi"), _ok(), _ok()]
    )
    await _play(scripted, _FakeBrowser(), source="Example Speaker", default_source="Example Pi")
    assert json.loads(scripted.posts[0].content)["source"] == "Example Speaker"


async def test_a_default_missing_from_the_source_list_is_refused_with_no_post():
    scripted = _ScriptedHa([_state(_IDLE_FEATURES, source_list=["Example Speaker"])])
    with pytest.raises(Denied) as exc:
        await _play(scripted, _FakeBrowser(), default_source="Example Pi")
    assert "Example Pi" in exc.value.reason
    assert scripted.posts == []


@pytest.mark.parametrize("blank", [None, "", "   "])
async def test_a_blank_default_keeps_todays_behavior(blank):
    scripted = _ScriptedHa(
        [_state(_ACTIVE_FEATURES, source_list=_TWO, source="Example Speaker"), _ok()]
    )
    await _play(scripted, _FakeBrowser(), default_source=blank)
    assert [p.url.path.rsplit("/", 1)[1] for p in scripted.posts] == ["play_media"]


# -- default speaker: ha.py wiring -------------------------------------------


async def test_startup_stores_the_stripped_default_source(monkeypatch):
    monkeypatch.setenv("HA_URL", "http://ha.invalid")
    monkeypatch.setenv("HA_TOKEN", "test-token")
    monkeypatch.setattr(ha_module, "_http_client", None)
    monkeypatch.setattr(ha_module, "_registry_client", None)
    monkeypatch.setattr(ha_module, "_default_spotify_source", None)
    monkeypatch.setenv("SPOTIFY_DEFAULT_SOURCE", "  Example Pi ")
    ha_module._startup()
    try:
        assert ha_module._default_spotify_source == "Example Pi"
        monkeypatch.setenv("SPOTIFY_DEFAULT_SOURCE", "   ")
        await ha_module._http_client.aclose()
        ha_module._startup()
        assert ha_module._default_spotify_source is None
        monkeypatch.delenv("SPOTIFY_DEFAULT_SOURCE")
        await ha_module._http_client.aclose()
        ha_module._startup()
        assert ha_module._default_spotify_source is None
    finally:
        await ha_module._http_client.aclose()


async def test_the_tool_passes_the_stored_default_source(monkeypatch):
    seen = {}

    async def fake_handle(*args, **kwargs):
        seen.update(kwargs)
        return {"playlist": "x"}

    client = httpx.AsyncClient()
    monkeypatch.setattr(ha_module, "handle_play_spotify_playlist", fake_handle)
    monkeypatch.setattr(ha_module, "_http_client", client)
    monkeypatch.setattr(ha_module, "_default_spotify_source", "Example Pi")
    try:
        await ha_module.ha_play_spotify_playlist(_ENTITY, "love refined")
    finally:
        await client.aclose()
    assert seen == {"source": None, "default_source": "Example Pi"}
