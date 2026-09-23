"""Tests for resolving a spoken place name to weather coordinates.

Fixtures use generic public places -- Paris, Springfield, Portland, London
-- and synthetic coordinates only. The house location stays out of this
public repository, the same constraint tests/test_weather_tool.py already
follows with its 0.0, 0.0 fixture coordinates.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from spire_mcp.open_meteo import OpenMeteoClient

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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

_PARIS_GEOCODE_BODY = {
    "results": [
        {
            "name": "Paris",
            "latitude": 48.85341,
            "longitude": 2.3488,
            "country": "France",
            "country_code": "FR",
            "admin1": "Île-de-France",
            "population": 2138551,
        },
        {
            "name": "Paris",
            "latitude": 33.66094,
            "longitude": -95.55551,
            "country": "United States",
            "country_code": "US",
            "admin1": "Texas",
            "population": 25171,
        },
        {
            "name": "Paris",
            "latitude": 36.30200,
            "longitude": -88.32670,
            "country": "United States",
            "country_code": "US",
            "admin1": "Tennessee",
            "population": 10156,
        },
    ]
}


_SPRINGFIELD_GEOCODE_BODY = {
    "results": [
        {
            "name": "Springfield",
            "latitude": 39.80172,
            "longitude": -89.64371,
            "country": "United States",
            "country_code": "US",
            "admin1": "Illinois",
            "population": 116250,
        },
        {
            "name": "Springfield",
            "latitude": 37.21533,
            "longitude": -93.29824,
            "country": "United States",
            "country_code": "US",
            "admin1": "Missouri",
            "population": 169176,
        },
        {
            "name": "Springfield",
            "latitude": 42.10148,
            "longitude": -72.58981,
            "country": "United States",
            "country_code": "US",
            "admin1": "Massachusetts",
            "population": 153606,
        },
    ]
}

_PORTLAND_GEOCODE_BODY = {
    "results": [
        {
            "name": "Portland",
            "latitude": 45.52345,
            "longitude": -122.67621,
            "country": "United States",
            "country_code": "US",
            "admin1": "Oregon",
            "population": 652503,
        },
        {
            "name": "Portland",
            "latitude": 50.54759,
            "longitude": -2.4562,
            "country": "United Kingdom",
            "country_code": "GB",
            "admin1": "England",
            "population": 12800,
        },
    ]
}


class _RoutingTransport:
    """Routes a request by host: the geocoding host gets `geocode_body`,
    everything else gets `other_body`. Records every request it answers."""

    def __init__(self, geocode_body: dict[str, Any], other_body: dict[str, Any]) -> None:
        self.geocode_body = geocode_body
        self.other_body = other_body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json=self.geocode_body)
        return httpx.Response(200, json=self.other_body)


def _client_for(transport_handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))


# ---------------------------------------------------------------------------
# Task 1: end-to-end "weather_current for 'Paris, France'" -- one path only
# ---------------------------------------------------------------------------


async def test_weather_current_for_a_qualified_place_reports_the_resolved_location():
    from spire_mcp import weather

    router = _RoutingTransport(_PARIS_GEOCODE_BODY, _CURRENT_BODY)
    async with _client_for(router) as http_client:
        weather._open_meteo_client = OpenMeteoClient(http_client)
        weather._latitude = 0.0
        weather._longitude = 0.0
        try:
            result = await weather.weather_current(place="Paris, France")
        finally:
            weather._open_meteo_client = None

    assert result["location"] == "Paris, Île-de-France, France"

    geocode_requests = [r for r in router.requests if r.url.host == "geocoding-api.open-meteo.com"]
    assert len(geocode_requests) == 1
    assert geocode_requests[0].url.params["name"] == "Paris"

    forecast_requests = [r for r in router.requests if r.url.host != "geocoding-api.open-meteo.com"]
    assert len(forecast_requests) == 1
    assert float(forecast_requests[0].url.params["latitude"]) == 48.85341


async def test_weather_current_without_a_place_reports_home():
    from spire_mcp import weather

    router = _RoutingTransport(_PARIS_GEOCODE_BODY, _CURRENT_BODY)
    async with _client_for(router) as http_client:
        weather._open_meteo_client = OpenMeteoClient(http_client)
        weather._latitude = 0.0
        weather._longitude = 0.0
        try:
            result = await weather.weather_current()
        finally:
            weather._open_meteo_client = None

    assert result["location"] == "home"
    forecast_requests = [r for r in router.requests if r.url.host != "geocoding-api.open-meteo.com"]
    assert len(forecast_requests) == 1
    assert float(forecast_requests[0].url.params["latitude"]) == 0.0
    geocode_requests = [r for r in router.requests if r.url.host == "geocoding-api.open-meteo.com"]
    assert geocode_requests == []


async def test_weather_current_in_fahrenheit_sends_temperature_unit_and_labels_f(monkeypatch):
    from spire_mcp import weather

    router = _RoutingTransport(_PARIS_GEOCODE_BODY, _CURRENT_BODY)
    async with _client_for(router) as http_client:
        weather._open_meteo_client = OpenMeteoClient(http_client)
        weather._latitude = 0.0
        weather._longitude = 0.0
        monkeypatch.setattr(weather, "_units", "fahrenheit")
        try:
            result = await weather.weather_current()
        finally:
            weather._open_meteo_client = None

    non_geocoding_requests = [r for r in router.requests if r.url.host != "geocoding-api.open-meteo.com"]
    assert len(non_geocoding_requests) == 1
    assert non_geocoding_requests[0].url.params["temperature_unit"] == "fahrenheit"
    assert result["temperature"] == 24.4
    assert result["unit"] == "F"
    assert not any(key.endswith("_c") for key in result)


async def test_weather_current_default_units_sends_no_temperature_unit_param():
    from spire_mcp import weather

    router = _RoutingTransport(_PARIS_GEOCODE_BODY, _CURRENT_BODY)
    async with _client_for(router) as http_client:
        weather._open_meteo_client = OpenMeteoClient(http_client)
        weather._latitude = 0.0
        weather._longitude = 0.0
        try:
            result = await weather.weather_current()
        finally:
            weather._open_meteo_client = None

    non_geocoding_requests = [r for r in router.requests if r.url.host != "geocoding-api.open-meteo.com"]
    assert len(non_geocoding_requests) == 1
    assert "temperature_unit" not in non_geocoding_requests[0].url.params
    assert result["unit"] == "C"


# ---------------------------------------------------------------------------
# Task 2: locale-biased selection, the global fallback, and a named refusal
# ---------------------------------------------------------------------------


def test_split_place_splits_on_the_first_comma_and_strips_both_parts():
    from spire_mcp.weather import split_place

    assert split_place("Paris, France") == ("Paris", "France")
    assert split_place("  Springfield ,  Illinois ") == ("Springfield", "Illinois")
    assert split_place("Paris") == ("Paris", None)
    assert split_place("Paris,") == ("Paris", None)


def test_haversine_km_matches_the_known_paris_to_london_distance():
    from spire_mcp.weather import haversine_km

    distance = haversine_km(48.8566, 2.3522, 51.5074, -0.1278)
    assert abs(distance - 343.5) < 5


def test_qualifier_filter_narrows_to_the_matching_country_or_admin1_or_none():
    from spire_mcp.weather import select_place

    matches = _PARIS_GEOCODE_BODY["results"]

    texas = select_place(
        matches, qualifier="texas", home_latitude=0.0, home_longitude=0.0, local_radius_km=300.0
    )
    assert texas["admin1"] == "Texas"

    france = select_place(
        matches, qualifier=" fr ", home_latitude=0.0, home_longitude=0.0, local_radius_km=300.0
    )
    assert france["country_code"] == "FR"

    assert (
        select_place(
            matches,
            qualifier="Germany",
            home_latitude=0.0,
            home_longitude=0.0,
            local_radius_km=300.0,
        )
        is None
    )


def test_qualifier_alias_usa_keeps_only_the_us_match():
    from spire_mcp.weather import select_place

    matches = _PORTLAND_GEOCODE_BODY["results"]
    result = select_place(
        matches, qualifier="USA", home_latitude=0.0, home_longitude=0.0, local_radius_km=300.0
    )
    assert result["country_code"] == "US"
    assert result["admin1"] == "Oregon"


def test_qualifier_then_nearby_picks_the_nearer_qualifying_match():
    from spire_mcp.weather import select_place

    matches = _PARIS_GEOCODE_BODY["results"]
    tennessee_latitude, tennessee_longitude = 36.30200, -88.32670

    result = select_place(
        matches,
        qualifier="United States",
        home_latitude=tennessee_latitude,
        home_longitude=tennessee_longitude,
        local_radius_km=300.0,
    )
    assert result["admin1"] == "Tennessee"


def test_qualifier_then_population_picks_the_more_populous_qualifying_match():
    from spire_mcp.weather import select_place

    matches = _PARIS_GEOCODE_BODY["results"]
    result = select_place(
        matches,
        qualifier="United States",
        home_latitude=0.0,
        home_longitude=0.0,
        local_radius_km=300.0,
    )
    assert result["admin1"] == "Texas"


def test_nearby_wins_over_a_more_populous_farther_match():
    from spire_mcp.weather import select_place

    matches = _SPRINGFIELD_GEOCODE_BODY["results"]
    illinois_latitude, illinois_longitude = 39.80172, -89.64371

    result = select_place(
        matches,
        qualifier=None,
        home_latitude=illinois_latitude,
        home_longitude=illinois_longitude,
        local_radius_km=300.0,
    )
    assert result["admin1"] == "Illinois"


def test_nearest_of_several_nearby_wins_even_over_a_far_more_populous_one():
    from spire_mcp.weather import select_place

    matches = [
        {"name": "A", "latitude": 10.0, "longitude": 10.0, "population": 1000000},
        {"name": "B", "latitude": 10.5, "longitude": 10.0, "population": 10},
    ]
    result = select_place(
        matches, qualifier=None, home_latitude=10.4, home_longitude=10.0, local_radius_km=300.0
    )
    assert result["name"] == "B"


def test_population_fallback_picks_the_most_populous_match_when_none_are_nearby():
    from spire_mcp.weather import select_place

    matches = _PARIS_GEOCODE_BODY["results"]
    result = select_place(
        matches, qualifier=None, home_latitude=0.0, home_longitude=0.0, local_radius_km=300.0
    )
    assert result["country_code"] == "FR"


def test_missing_population_counts_as_zero():
    from spire_mcp.weather import select_place

    matches = [
        {"name": "Alpha", "latitude": 80.0, "longitude": 80.0, "population": None},
        {"name": "Beta", "latitude": 80.0, "longitude": 80.0, "population": 1},
        {"name": "Gamma", "latitude": 80.0, "longitude": 80.0},
    ]
    result = select_place(
        matches, qualifier=None, home_latitude=0.0, home_longitude=0.0, local_radius_km=300.0
    )
    assert result["name"] == "Beta"


def test_empty_match_list_returns_none():
    from spire_mcp.weather import select_place

    assert (
        select_place([], qualifier=None, home_latitude=0.0, home_longitude=0.0, local_radius_km=300.0)
        is None
    )


def test_format_location_skips_empty_parts():
    from spire_mcp.weather import format_location

    assert format_location({"name": "Paris", "admin1": "", "country": "France"}) == "Paris, France"
    assert format_location({"name": "Paris", "admin1": "", "country": ""}) == "Paris"


async def test_the_decorated_weather_current_names_the_unknown_place_in_a_tool_error():
    from mcp.server.mcpserver.exceptions import ToolError
    from spire_mcp import weather

    no_match_body = {"generationtime_ms": 0.4}
    router = _RoutingTransport(no_match_body, _CURRENT_BODY)
    async with _client_for(router) as http_client:
        weather._open_meteo_client = OpenMeteoClient(http_client)
        weather._latitude = 0.0
        weather._longitude = 0.0
        try:
            with pytest.raises(ToolError) as excinfo:
                await weather.weather_current(place="Atlantis")
        finally:
            weather._open_meteo_client = None

    assert "Atlantis" in str(excinfo.value)


def test_read_local_radius_km_default_and_valid_value(monkeypatch):
    from spire_mcp.weather import _read_local_radius_km

    monkeypatch.delenv("WEATHER_LOCAL_RADIUS_KM", raising=False)
    assert _read_local_radius_km() == 300.0

    monkeypatch.setenv("WEATHER_LOCAL_RADIUS_KM", "  ")
    assert _read_local_radius_km() == 300.0

    monkeypatch.setenv("WEATHER_LOCAL_RADIUS_KM", "50")
    assert _read_local_radius_km() == 50.0


def test_read_local_radius_km_rejects_unparseable_negative_and_infinite_values(monkeypatch):
    from spire_mcp.weather import _read_local_radius_km

    for bad_value in ("abc", "-1", "inf"):
        monkeypatch.setenv("WEATHER_LOCAL_RADIUS_KM", bad_value)
        with pytest.raises(SystemExit) as excinfo:
            _read_local_radius_km()
        assert "WEATHER_LOCAL_RADIUS_KM" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Task 3: forecast for a named place, geocode parsing edges, and the schema
# ---------------------------------------------------------------------------


class _Clock:
    """A fake monotonic clock a test can move forward without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _RecordingTransport:
    """An `httpx.MockTransport` handler that counts and records requests."""

    def __init__(self, response_body, status_code: int = 200) -> None:
        self.calls = 0
        self.requests: list[httpx.Request] = []
        self.response_body = response_body
        self.status_code = status_code

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.requests.append(request)
        return httpx.Response(self.status_code, json=self.response_body)


async def test_weather_forecast_for_a_qualified_place_reports_its_location():
    from spire_mcp import weather

    router = _RoutingTransport(_SPRINGFIELD_GEOCODE_BODY, _FORECAST_BODY)
    async with _client_for(router) as http_client:
        weather._open_meteo_client = OpenMeteoClient(http_client)
        weather._latitude = 0.0
        weather._longitude = 0.0
        try:
            result = await weather.weather_forecast(days=2, place="Springfield, Illinois")
        finally:
            weather._open_meteo_client = None

    assert result["location"] == "Springfield, Illinois, United States"
    assert isinstance(result["days"], list) and result["days"]


async def test_weather_forecast_refuses_a_bad_day_count_before_any_request():
    from mcp.server.mcpserver.exceptions import ToolError
    from spire_mcp import weather

    router = _RoutingTransport(_PARIS_GEOCODE_BODY, _FORECAST_BODY)
    async with _client_for(router) as http_client:
        weather._open_meteo_client = OpenMeteoClient(http_client)
        weather._latitude = 0.0
        weather._longitude = 0.0
        try:
            with pytest.raises(ToolError) as excinfo:
                await weather.weather_forecast(days=17, place="Paris")
        finally:
            weather._open_meteo_client = None

    assert "16" in str(excinfo.value)
    assert router.requests == []


async def test_geocode_sends_the_expected_request_shape():
    recorder = _RecordingTransport(_PARIS_GEOCODE_BODY)
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client)
        await client.geocode("Paris")

    assert recorder.calls == 1
    request = recorder.requests[0]
    assert request.url.host == "geocoding-api.open-meteo.com"
    assert request.url.path == "/v1/search"
    assert request.url.params["name"] == "Paris"
    assert request.url.params["count"] == "10"
    assert request.url.params["language"] == "en"
    assert request.url.params["format"] == "json"


async def test_geocode_returns_the_normalized_fields_with_a_missing_admin1_as_empty():
    body = {
        "results": [
            {
                "name": "Nowhere",
                "latitude": 1.0,
                "longitude": 2.0,
                "country": "Testland",
                "country_code": "TL",
                "population": 5,
            }
        ]
    }
    recorder = _RecordingTransport(body)
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client)
        result = await client.geocode("Nowhere")

    assert result == [
        {
            "name": "Nowhere",
            "latitude": 1.0,
            "longitude": 2.0,
            "country": "Testland",
            "country_code": "TL",
            "admin1": "",
            "population": 5,
        }
    ]


async def test_geocode_with_no_results_key_or_an_empty_list_gives_no_matches():
    for body in ({"generationtime_ms": 0.3}, {"results": []}):
        recorder = _RecordingTransport(body)
        async with _client_for(recorder) as http_client:
            client = OpenMeteoClient(http_client)
            result = await client.geocode("Nowhere")
        assert result == []


async def test_geocode_with_an_unreadable_body_raises_malformed():
    from spire_mcp.open_meteo import UpstreamMalformedError

    for body in ({"results": [{"name": "Paris"}]}, {"results": "nope"}):
        recorder = _RecordingTransport(body)
        async with _client_for(recorder) as http_client:
            client = OpenMeteoClient(http_client)
            with pytest.raises(UpstreamMalformedError):
                await client.geocode("Paris")


async def test_geocode_with_a_503_raises_unreachable():
    from spire_mcp.open_meteo import UpstreamUnreachableError

    recorder = _RecordingTransport({"error": "internal"}, status_code=503)
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client)
        with pytest.raises(UpstreamUnreachableError):
            await client.geocode("Paris")


async def test_a_second_geocode_inside_the_ttl_makes_one_request_in_total():
    recorder = _RecordingTransport(_PARIS_GEOCODE_BODY)
    clock = _Clock()
    async with _client_for(recorder) as http_client:
        client = OpenMeteoClient(http_client, ttl_seconds=600.0, clock=clock)
        await client.geocode("Paris")
        clock.advance(60.0)
        await client.geocode("Paris")

    assert recorder.calls == 1


async def test_both_tools_advertise_an_optional_place_in_their_schema():
    from spire_mcp import weather

    listed = await weather.mcp_server.list_tools()
    tools_by_name = {tool.name: tool for tool in listed}

    for name in ("weather_current", "weather_forecast"):
        schema = tools_by_name[name].input_schema
        assert "place" in schema["properties"]
        assert "place" not in schema.get("required", [])
