"""The stdio MCP tool server for outdoor weather, from open-meteo.

This module imports nothing from `spire_mcp.safety`, unlike `ha.py`. Weather
controls nothing -- there is no entity to deny, so there is no denylist
concept for this plugin at all. That absence is the correct design here, not
a missing check: a handler that reads a public forecast has nothing to gate.

Every handler below takes the open-meteo client as a parameter and reads no
module-level global. Neither handler catches the client's own exceptions --
translating `UpstreamUnreachableError`/`UpstreamMalformedError` into the
SDK's tool error is the decorated wrapper's single job, so that translation
happens in exactly one place.

Each handler's docstring is the description the language model sees when
choosing between this server's tools and the Home Assistant server's. Both
say plainly that they answer about *outdoor* weather from an *external*
service, so a question about a room's own temperature sensor is not routed
here by mistake.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from spire_mcp.open_meteo import OpenMeteoClient, UpstreamMalformedError, UpstreamUnreachableError

_MIN_FORECAST_DAYS = 1
_MAX_FORECAST_DAYS = 16
_DEFAULT_FORECAST_DAYS = 3


async def handle_weather_current(
    client: OpenMeteoClient, latitude: float, longitude: float
) -> dict[str, Any]:
    """Answer a question about the current outdoor weather, from an
    external service (open-meteo) -- never a room's own temperature sensor.

    Returns a dict already close to speech: a temperature in Celsius, a
    short condition phrase, and the location's local time. Rounded to one
    decimal place, the precision a person says out loud.
    """
    result = await client.current_conditions(latitude, longitude)
    return {
        "temperature_c": round(result["temperature_c"], 1),
        "condition": result["condition"],
        "local_time": result["local_time"],
    }


async def handle_weather_forecast(
    client: OpenMeteoClient, latitude: float, longitude: float, days: int
) -> dict[str, Any]:
    """Answer a question about the outdoor weather forecast, from an
    external service (open-meteo) -- never a room's own temperature sensor.

    Returns up to `days` days, each with a high, a low (Celsius), and a
    short condition phrase. `days` must be between 1 and 16 -- the range
    open-meteo itself accepts -- and a count outside that range is refused
    rather than clamped, so a mistaken request never silently answers a
    different question than the one asked.
    """
    if not (_MIN_FORECAST_DAYS <= days <= _MAX_FORECAST_DAYS):
        raise ValueError(
            f"a forecast day count must be between {_MIN_FORECAST_DAYS} and "
            f"{_MAX_FORECAST_DAYS}, got {days}"
        )
    result = await client.forecast(latitude, longitude, days)
    return {
        "days": [
            {
                "date": day["date"],
                "high_c": round(day["high_c"], 1),
                "low_c": round(day["low_c"], 1),
                "condition": day["condition"],
            }
            for day in result["days"]
        ],
    }


mcp_server = MCPServer("spire-weather")

_http_client: httpx.AsyncClient | None = None
_open_meteo_client: OpenMeteoClient | None = None
_latitude: float = 0.0
_longitude: float = 0.0


@mcp_server.tool()
async def weather_current() -> dict[str, Any]:
    """Answer a question about the current outdoor weather, from an
    external service (open-meteo) -- never a room's own temperature sensor.

    Returns the current outdoor temperature and condition at the house's
    configured location.
    """
    assert _open_meteo_client is not None, "weather_current invoked before startup"
    try:
        return await handle_weather_current(_open_meteo_client, _latitude, _longitude)
    except (UpstreamUnreachableError, UpstreamMalformedError) as exc:
        # The exception's own sentence crosses this boundary unchanged --
        # the same "boundary's own words, nothing reworded" doctrine the
        # Home Assistant server already follows for its own refusals.
        raise ToolError(str(exc)) from exc


@mcp_server.tool()
async def weather_forecast(days: int = _DEFAULT_FORECAST_DAYS) -> dict[str, Any]:
    """Answer a question about the outdoor weather forecast, from an
    external service (open-meteo) -- never a room's own temperature sensor.

    Returns up to `days` days (1-16) of high, low, and condition at the
    house's configured location.
    """
    assert _open_meteo_client is not None, "weather_forecast invoked before startup"
    try:
        return await handle_weather_forecast(_open_meteo_client, _latitude, _longitude, days)
    except (UpstreamUnreachableError, UpstreamMalformedError, ValueError) as exc:
        # LOW-01 fix (phase 4 code review): `handle_weather_forecast` raises
        # a bare `ValueError` for a day count outside 1-16 -- previously not
        # named in this `except` clause, so it propagated as a raw,
        # uncaught exception through the MCP SDK's own generic error path
        # instead of this module's "boundary's own words, nothing reworded"
        # doctrine, the same swallowing Phase 3's CR-01 fix addressed for
        # `Denied`. `str(exc)` on a `ValueError` is exactly the message
        # `handle_weather_forecast` raised, so this crosses the boundary in
        # the same voice as every other refusal in this module.
        raise ToolError(str(exc)) from exc


def _read_coordinate(name: str) -> float:
    """Read one coordinate from this process's own environment.

    A location pinpoints the physical house, which is exactly the kind of
    detail this repository's public-repo constraint keeps out of git -- so
    it arrives here as an environment value and never as a literal or a
    default in this file. Missing or unparseable makes this child refuse to
    start, naming the variable, the same refuse-rather-than-guess posture
    the Home Assistant child already takes toward a malformed policy.
    """
    raw = os.environ.get(name)
    if raw is None:
        raise SystemExit(f"{name} is required and was not set")
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} is not a valid number: {raw!r}") from exc


def _startup() -> None:
    global _http_client, _open_meteo_client, _latitude, _longitude
    _latitude = _read_coordinate("WEATHER_LATITUDE")
    _longitude = _read_coordinate("WEATHER_LONGITUDE")
    _http_client = httpx.AsyncClient()
    _open_meteo_client = OpenMeteoClient(_http_client)


async def _run() -> None:
    _startup()
    try:
        await mcp_server.run_stdio_async()
    finally:
        if _http_client is not None:
            await _http_client.aclose()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_run())
