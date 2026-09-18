"""The weather MCP server: a plugin from birth, not built-in code awaiting
migration.

This file proves the properties CONTEXT.md D-02 and SAFE-09 name directly:
an unreachable upstream degrades to a short spoken sentence rather than a
crash or a guess (never `UpstreamUnreachableError` propagating raw text an
operator would find odd to hear, and never a partially-parsed forecast); a
malformed upstream is a distinguishable fact from an unreachable one; a
repeated question inside the cache window costs no second outbound call; and
the server holds no credential and imports no policy at all.

Fixture coordinates throughout are 0.0, 0.0 (the Gulf of Guinea) --
deliberately not the author's house, per PROJECT.md's public-repo
constraint. That constraint applies to a test fixture as much as to source.
"""

from __future__ import annotations

import os
import subprocess
import sys

import httpx
import pytest

from spire_mcp.open_meteo import (
    OpenMeteoClient,
    UpstreamMalformedError,
    UpstreamUnreachableError,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_FAKE_LATITUDE = 0.0
_FAKE_LONGITUDE = 0.0

# The exact shape a real call to api.open-meteo.com/v1/forecast returned
# this session (see 04-02-SUMMARY.md for the verbatim top-level keys),
# reproduced here with the same field names and value types so the parser
# is tested against a response that arrived, not one that was remembered.
_CURRENT_BODY = {
    "latitude": 0.035149384,
    "longitude": 0.0,
    "generationtime_ms": 0.17797946929931,
    "utc_offset_seconds": 0,
    "timezone": "Etc/GMT",
    "timezone_abbreviation": "GMT",
    "elevation": 0.0,
    "current_units": {
        "time": "iso8601",
        "interval": "seconds",
        "temperature_2m": "°C",
        "weather_code": "wmo code",
        "is_day": "",
    },
    "current": {
        "time": "2026-09-18T16:00",
        "interval": 900,
        "temperature_2m": 24.4,
        "weather_code": 3,
        "is_day": 1,
    },
}

_FORECAST_BODY = {
    "latitude": 0.035149384,
    "longitude": 0.0,
    "timezone": "Etc/GMT",
    "daily_units": {
        "time": "iso8601",
        "temperature_2m_max": "°C",
        "temperature_2m_min": "°C",
        "weather_code": "wmo code",
    },
    "daily": {
        "time": ["2026-09-18", "2026-09-19", "2026-09-20"],
        "temperature_2m_max": [24.5, 24.7, 25.0],
        "temperature_2m_min": [23.5, 23.9, 24.0],
        "weather_code": [51, 3, 51],
    },
}


class _Clock:
    """A fake monotonic clock a test can move forward without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _RecordingTransport:
    """An `httpx.MockTransport` handler that counts and can be reconfigured."""

    def __init__(self, response_body: dict, status_code: int = 200) -> None:
        self.calls = 0
        self.response_body = response_body
        self.status_code = status_code

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(self.status_code, json=self.response_body)


def _client_for(transport_handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))


# ---------------------------------------------------------------------------
# Task 1: the open-meteo client
# ---------------------------------------------------------------------------


async def test_current_conditions_returns_a_plain_dict_with_temperature_and_condition():
    recorder = _RecordingTransport(_CURRENT_BODY)
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client, ttl_seconds=600.0)
        result = await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)

    assert type(result) is dict
    assert result["temperature_c"] == 24.4
    assert result["condition"] == "overcast"
    assert result["local_time"] == "2026-09-18T16:00"
    assert result["timezone"] == "Etc/GMT"


async def test_second_fetch_inside_the_ttl_window_makes_no_outbound_request():
    recorder = _RecordingTransport(_CURRENT_BODY)
    clock = _Clock()
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client, ttl_seconds=600.0, clock=clock)
        await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)
        clock.advance(60.0)
        await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)

    assert recorder.calls == 1, "a repeated question inside the cache window must cost no second call"


async def test_fetch_after_the_ttl_window_makes_a_new_outbound_request():
    recorder = _RecordingTransport(_CURRENT_BODY)
    clock = _Clock()
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client, ttl_seconds=600.0, clock=clock)
        await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)
        clock.advance(600.1)
        await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)

    assert recorder.calls == 2, "a fetch after the TTL window must reach the upstream again"


async def test_connection_error_raises_unreachable_with_a_speakable_sentence():
    def _raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client_for(_raise_connect_error) as http_client:
        client = OpenMeteoClient(http_client)
        with pytest.raises(UpstreamUnreachableError) as excinfo:
            await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)

    assert str(excinfo.value), "the exception must carry a non-empty speakable sentence"


async def test_timeout_raises_unreachable_with_a_speakable_sentence():
    def _raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    async with _client_for(_raise_timeout) as http_client:
        client = OpenMeteoClient(http_client)
        with pytest.raises(UpstreamUnreachableError) as excinfo:
            await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)

    assert str(excinfo.value)


async def test_non_2xx_status_raises_unreachable_with_a_speakable_sentence():
    recorder = _RecordingTransport({"error": "internal"}, status_code=503)
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client)
        with pytest.raises(UpstreamUnreachableError) as excinfo:
            await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)

    assert str(excinfo.value)


async def test_200_with_missing_fields_raises_malformed_with_a_different_sentence():
    incomplete_body = {"latitude": 0.0, "longitude": 0.0, "timezone": "Etc/GMT"}
    recorder = _RecordingTransport(incomplete_body)
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client)
        with pytest.raises(UpstreamMalformedError) as malformed_excinfo:
            await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)

    async with _client_for(_RecordingTransport({}, status_code=500)) as http_client:
        client = OpenMeteoClient(http_client)
        with pytest.raises(UpstreamUnreachableError) as unreachable_excinfo:
            await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)

    assert str(malformed_excinfo.value) != str(unreachable_excinfo.value), (
        "an unreachable upstream and a malformed one must be distinguishable in what "
        "the operator hears"
    )


async def test_forecast_returns_bounded_days_with_high_low_and_condition():
    recorder = _RecordingTransport(_FORECAST_BODY)
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client)
        result = await client.forecast(_FAKE_LATITUDE, _FAKE_LONGITUDE, days=3)

    assert len(result["days"]) == 3
    first = result["days"][0]
    assert first["date"] == "2026-09-18"
    assert first["high_c"] == 24.5
    assert first["low_c"] == 23.5
    assert first["condition"] == "light drizzle"


@pytest.mark.integration
async def test_live_current_conditions_has_the_fields_the_parser_reads():
    """Calls the real endpoint. A schema change upstream must fail loudly
    here, not silently at turn time. Fixed, non-house coordinates only."""
    async with httpx.AsyncClient(timeout=15.0) as http_client:
        client = OpenMeteoClient(http_client)
        result = await client.current_conditions(_FAKE_LATITUDE, _FAKE_LONGITUDE)

    assert isinstance(result["temperature_c"], (int, float))
    assert isinstance(result["condition"], str) and result["condition"]
    assert isinstance(result["local_time"], str) and result["local_time"]
    assert isinstance(result["timezone"], str) and result["timezone"]


# ---------------------------------------------------------------------------
# Task 2: the two tool handlers
# ---------------------------------------------------------------------------


async def test_current_conditions_handler_returns_a_speech_ready_dict():
    from spire_mcp.weather import handle_weather_current

    recorder = _RecordingTransport(_CURRENT_BODY)
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client)
        result = await handle_weather_current(client, _FAKE_LATITUDE, _FAKE_LONGITUDE)

    assert result["temperature_c"] == 24.4
    assert result["condition"] == "overcast"
    assert "local_time" in result


async def test_forecast_handler_returns_bounded_days_and_refuses_an_out_of_range_count():
    from spire_mcp.weather import handle_weather_forecast

    recorder = _RecordingTransport(_FORECAST_BODY)
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client)
        result = await handle_weather_forecast(client, _FAKE_LATITUDE, _FAKE_LONGITUDE, days=3)
        assert len(result["days"]) == 3
        for day in result["days"]:
            assert set(day) >= {"date", "high_c", "low_c", "condition"}

        with pytest.raises(ValueError):
            await handle_weather_forecast(client, _FAKE_LATITUDE, _FAKE_LONGITUDE, days=0)
        with pytest.raises(ValueError):
            await handle_weather_forecast(client, _FAKE_LATITUDE, _FAKE_LONGITUDE, days=17)


async def test_the_decorated_forecast_tool_refuses_an_out_of_range_count_as_a_tool_error():
    """LOW-01 fix (phase 4 code review): `handle_weather_forecast` raising
    a bare `ValueError` (proven above) is only half the property -- the
    decorated tool `weather_forecast` must translate it into `ToolError`,
    the same "boundary's own words, nothing reworded" doctrine every other
    refusal in this module already follows
    (`weather_current`'s own `except (UpstreamUnreachableError,
    UpstreamMalformedError)` clause). Before this fix, `ValueError` was not
    named in `weather_forecast`'s own `except` clause, so it propagated
    uncaught through the MCP SDK's generic error path instead -- still
    surfaced as an error, but in a different voice than every other
    refusal in this turn."""
    from mcp.server.mcpserver.exceptions import ToolError
    from spire_mcp import weather

    recorder = _RecordingTransport(_FORECAST_BODY)
    async with _client_for(recorder) as http_client:
        weather._open_meteo_client = OpenMeteoClient(http_client)
        weather._latitude = _FAKE_LATITUDE
        weather._longitude = _FAKE_LONGITUDE
        try:
            with pytest.raises(ToolError) as excinfo:
                await weather.weather_forecast(days=17)
            assert "16" in str(excinfo.value), "the refused count's own bound must reach the caller"
        finally:
            weather._open_meteo_client = None


async def test_handlers_take_their_client_as_a_parameter_and_propagate_upstream_errors():
    from spire_mcp.weather import handle_weather_current

    def _raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client_for(_raise_connect_error) as http_client:
        client = OpenMeteoClient(http_client)
        with pytest.raises(UpstreamUnreachableError):
            await handle_weather_current(client, _FAKE_LATITUDE, _FAKE_LONGITUDE)


def test_handler_docstrings_are_non_empty_tool_descriptions():
    from spire_mcp import weather

    assert weather.handle_weather_current.__doc__
    assert weather.handle_weather_forecast.__doc__


def test_no_handler_body_references_a_module_level_client_name():
    import inspect

    from spire_mcp import weather

    for name in ("handle_weather_current", "handle_weather_forecast"):
        source = inspect.getsource(getattr(weather, name))
        assert "_http_client" not in source
        assert "_open_meteo_client" not in source


# ---------------------------------------------------------------------------
# Task 3: the stdio server
# ---------------------------------------------------------------------------


def _spawn_weather_child(env_extra: dict[str, str]) -> subprocess.Popen:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{REPO}/mcp",
        **env_extra,
    }
    return subprocess.Popen(
        [sys.executable, "-m", "spire_mcp.weather"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=f"{REPO}/mcp",
        text=True,
    )


async def test_the_real_child_lists_exactly_the_two_weather_tools():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "spire_mcp.weather"],
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": f"{REPO}/mcp",
            "WEATHER_LATITUDE": "0.0",
            "WEATHER_LONGITUDE": "0.0",
        },
        cwd=f"{REPO}/mcp",
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()

    names = sorted(tool.name for tool in listed.tools)
    assert names == ["weather_current", "weather_forecast"]
    for tool in listed.tools:
        assert tool.description


def test_a_child_started_with_no_coordinate_variables_refuses_to_start():
    proc = _spawn_weather_child({})
    try:
        _, stderr = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr = proc.communicate()
        raise
    assert proc.returncode != 0, "a child with no coordinates must not serve a default location"
    assert "WEATHER_LATITUDE" in stderr or "WEATHER_LONGITUDE" in stderr


def test_weather_module_imports_no_policy_and_holds_no_credential_shaped_name():
    import re

    weather_path = os.path.join(REPO, "mcp", "spire_mcp", "weather.py")
    open_meteo_path = os.path.join(REPO, "mcp", "spire_mcp", "open_meteo.py")

    for path in (weather_path, open_meteo_path):
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        assert not re.search(r"from spire_mcp\.safety|import safety|allow_call|allow_read", source)
        assert not re.search(r"HA_TOKEN|HA_URL|XAI_API_KEY|api_key", source)
