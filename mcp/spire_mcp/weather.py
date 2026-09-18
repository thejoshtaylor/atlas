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

from typing import Any

from spire_mcp.open_meteo import OpenMeteoClient

_MIN_FORECAST_DAYS = 1
_MAX_FORECAST_DAYS = 16


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
