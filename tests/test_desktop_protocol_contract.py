"""The shared wire contract in `desktop/protocol/v1/` against the Python models
(Phase 14, D-09, D-21).

`swift test` reads the same files for the Mac half. This file walks every
fixture from disk, so a file added without a matching model rule fails here.
An empty directory fails at collection, so an empty glob can never pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.desktop import protocol

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "desktop" / "protocol" / "v1"

_MESSAGE_FILES = sorted((PROTOCOL_DIR / "messages").glob("*.json"))
_INVALID_FILES = sorted((PROTOCOL_DIR / "invalid").glob("*.json"))
assert _MESSAGE_FILES, f"no message fixtures under {PROTOCOL_DIR / 'messages'}"
assert _INVALID_FILES, f"no invalid fixtures under {PROTOCOL_DIR / 'invalid'}"


def _load(path: Path) -> dict:
    fixture = json.loads(path.read_text())
    assert fixture["direction"] in ("client_to_server", "server_to_client"), path.name
    return fixture


def _by_direction(files: list[Path], direction: str, *, unknown: bool) -> list[Path]:
    return [
        path
        for path in files
        if _load(path)["direction"] == direction and bool(_load(path).get("unknown")) is unknown
    ]


_CLIENT_MODELS = {
    "hello": protocol.DesktopHello,
    "ping": protocol.DesktopPing,
    "pong": protocol.DesktopPong,
}

# What each server frame's builder takes, read off the fixture's own fields.
_SERVER_BUILDERS = {
    "hello.ack": lambda m: protocol.build_hello_ack(m["device_id"]),
    "ping": lambda m: protocol.build_ping(m["id"]),
    "pong": lambda m: protocol.build_pong(m["id"]),
    "error": lambda m: protocol.build_error(m["code"], m["detail"]),
}


@pytest.mark.parametrize(
    "path",
    _by_direction(_MESSAGE_FILES, "client_to_server", unknown=False),
    ids=lambda p: p.name,
)
def test_client_fixture_parses_to_its_model_and_round_trips(path):
    message = _load(path)["message"]
    parsed = protocol.parse_client_message(json.dumps(message))
    assert type(parsed) is _CLIENT_MODELS[message["type"]]
    assert parsed.model_dump() == message


@pytest.mark.parametrize(
    "path",
    _by_direction(_MESSAGE_FILES, "server_to_client", unknown=False),
    ids=lambda p: p.name,
)
def test_server_fixture_validates_and_its_builder_produces_it(path):
    message = _load(path)["message"]
    assert protocol.SERVER_MESSAGE_ADAPTER.validate_python(message).model_dump() == message
    built = json.loads(_SERVER_BUILDERS[message["type"]](message))
    assert built == message


@pytest.mark.parametrize(
    "path",
    _by_direction(_MESSAGE_FILES, "client_to_server", unknown=True),
    ids=lambda p: p.name,
)
def test_an_unknown_client_type_is_ignored_not_an_error(path):
    message = _load(path)["message"]
    parsed = protocol.parse_client_message(json.dumps(message))
    assert isinstance(parsed, protocol.UnknownMessage)
    assert parsed.type == message["type"]


@pytest.mark.parametrize(
    "path",
    _by_direction(_MESSAGE_FILES, "server_to_client", unknown=True),
    ids=lambda p: p.name,
)
def test_an_unknown_server_type_is_not_a_frame_this_server_sends(path):
    """The app, not the server, ignores it. This pins that the server's own
    frame set does not contain the type, so the fixture stays "unknown"."""
    with pytest.raises(ValidationError):
        protocol.SERVER_MESSAGE_ADAPTER.validate_python(_load(path)["message"])


@pytest.mark.parametrize("path", _INVALID_FILES, ids=lambda p: p.name)
def test_every_invalid_fixture_is_rejected_in_its_direction(path):
    fixture = _load(path)
    assert fixture.get("description"), f"{path.name} must say what is wrong"
    if fixture["direction"] == "client_to_server":
        with pytest.raises(protocol.DesktopProtocolError):
            protocol.parse_client_message(json.dumps(fixture["message"]))
    else:
        with pytest.raises(ValidationError):
            protocol.SERVER_MESSAGE_ADAPTER.validate_python(fixture["message"])


@pytest.mark.parametrize("text", ["not json", "[]", "3", '"hello"', "null"])
def test_text_that_is_not_a_json_object_is_rejected(text):
    with pytest.raises(protocol.DesktopProtocolError):
        protocol.parse_client_message(text)


def test_close_codes_fixture_equals_the_python_constants():
    fixture = json.loads((PROTOCOL_DIR / "close_codes.json").read_text())
    constants = {
        name: value
        for name, value in vars(protocol).items()
        if name.startswith("CLOSE_") and isinstance(value, int)
    }
    assert {int(code) for code in fixture} == set(constants.values())
    for code, entry in fixture.items():
        assert constants[f"CLOSE_{entry['name'].upper()}"] == int(code)
        assert entry["client_action"]


def test_constants_fixture_equals_the_python_constants():
    fixture = json.loads((PROTOCOL_DIR / "constants.json").read_text())
    expected = {
        "protocol": protocol.PROTOCOL_VERSION,
        "ping_interval_s": protocol.PING_INTERVAL_S,
        "pong_timeout_s": protocol.PONG_TIMEOUT_S,
        "hello_timeout_s": protocol.HELLO_TIMEOUT_S,
        "idle_timeout_s": protocol.IDLE_TIMEOUT_S,
        "max_text_frame_bytes": protocol.MAX_TEXT_FRAME_BYTES,
        "max_invalid_messages": protocol.MAX_INVALID_MESSAGES,
        "test_timeout_s": protocol.TEST_TIMEOUT_S,
    }
    assert fixture == expected


def test_refusal_fixture_equals_the_python_constants():
    fixture = json.loads((PROTOCOL_DIR / "refusal.json").read_text())
    assert fixture == {
        "status": protocol.REFUSAL_STATUS,
        "header": protocol.REFUSAL_HEADER,
        "token_value": protocol.REFUSAL_TOKEN,
    }
