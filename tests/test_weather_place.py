"""Tests for resolving a spoken place name to weather coordinates.

Fixtures use generic public places -- Paris, Springfield, Portland, London
-- and synthetic coordinates only. The house location stays out of this
public repository, the same constraint tests/test_weather_tool.py already
follows with its 0.0, 0.0 fixture coordinates.
"""

from __future__ import annotations

from typing import Any

import httpx

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
