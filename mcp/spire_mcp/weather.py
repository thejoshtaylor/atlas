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

Both tools also take an optional `place`. The geocoder resolves a spoken
place with a local-first rule and a global fallback (`select_place`), and
every result names its `location`, so a missing place still says "home"
rather than staying silent about where the answer is for.
"""

from __future__ import annotations

import math
import os
from typing import Any

import httpx

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from spire_mcp.open_meteo import OpenMeteoClient, UpstreamMalformedError, UpstreamUnreachableError

_MIN_FORECAST_DAYS = 1
_MAX_FORECAST_DAYS = 16
_DEFAULT_FORECAST_DAYS = 3
_DEFAULT_LOCAL_RADIUS_KM = 300.0

# WEATHER_UNITS aliases, matched case-insensitively with surrounding spaces
# stripped. Kept as two closed sets, the same "add an entry, never guess"
# posture _COUNTRY_ALIASES already takes -- any value outside both refuses
# to start rather than being interpreted as one or the other.
_FAHRENHEIT_UNIT_ALIASES = {"f", "fahrenheit", "imperial", "us"}
_CELSIUS_UNIT_ALIASES = {"c", "celsius", "metric"}

# Common spoken aliases for a country, mapped to open-meteo's own
# `country_code`. Kept small and closed on purpose -- see
# `_qualifier_matches`. This is not a general alias system; add an entry
# only for a country a person is likely to name by an alias rather than
# by its own name.
_COUNTRY_ALIASES: dict[str, str] = {
    "usa": "US",
    "us": "US",
    "america": "US",
    "united states of america": "US",
    "uk": "GB",
    "britain": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
}


def split_place(place: str) -> tuple[str, str | None]:
    """Split a spoken place into its name and an optional qualifier.

    Splits on the first comma only, so "Springfield, Illinois, USA" keeps
    "Illinois, USA" together as one qualifier. Both parts are stripped. A
    qualifier that is missing, or empty once stripped, becomes None.
    """
    stripped = place.strip()
    name, _, rest = stripped.partition(",")
    name = name.strip()
    qualifier = rest.strip()
    return name, (qualifier or None)


def _qualifier_matches(qualifier_key: str, match: dict[str, Any]) -> bool:
    """Decide whether a stripped, casefolded qualifier picks out `match`.

    A qualifier matches when it equals the match's country, country_code,
    or admin1 (case-insensitive, trimmed), or when the alias map sends the
    qualifier to the match's country_code.
    """
    country_code = str(match.get("country_code", "")).strip()
    fields = {
        str(match.get("country", "")).strip().casefold(),
        country_code.casefold(),
        str(match.get("admin1", "")).strip().casefold(),
    }
    if qualifier_key in fields:
        return True
    alias_code = _COUNTRY_ALIASES.get(qualifier_key)
    return alias_code is not None and country_code != "" and alias_code == country_code.upper()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """The great-circle distance between two points, in kilometers.

    Uses the haversine formula with an earth radius of 6371.0 km.
    """
    earth_radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


def select_place(
    matches: list[dict[str, Any]],
    *,
    qualifier: str | None,
    home_latitude: float,
    home_longitude: float,
    local_radius_km: float,
) -> dict[str, Any] | None:
    """Pick one geocoding match from several matches sharing a name.

    The qualifier, when given, narrows the candidates first: a candidate
    survives only when `_qualifier_matches` accepts it. Among the
    survivors, the nearest one within `local_radius_km` of home wins. When
    none is that close, the one with the highest population wins,
    treating a missing population as 0. `min` and `max` both keep the
    geocoder's own first result on a tie, so an exact tie favors the
    geocoder's own ranking. Returns None when no candidate is left.
    """
    candidates = matches
    if qualifier is not None:
        qualifier_key = qualifier.strip().casefold()
        candidates = [match for match in candidates if _qualifier_matches(qualifier_key, match)]
    if not candidates:
        return None

    def _distance_from_home(match: dict[str, Any]) -> float:
        return haversine_km(home_latitude, home_longitude, match["latitude"], match["longitude"])

    nearby = [match for match in candidates if _distance_from_home(match) <= local_radius_km]
    if nearby:
        return min(nearby, key=_distance_from_home)
    return max(candidates, key=lambda match: match.get("population") or 0)


def format_location(match: dict[str, Any]) -> str:
    """Join a match's name, admin1, and country into one spoken phrase.

    Empty parts are skipped, so a match with no admin1 gives "Paris,
    France" rather than "Paris, , France".
    """
    parts = [
        str(match.get("name", "")).strip(),
        str(match.get("admin1", "")).strip(),
        str(match.get("country", "")).strip(),
    ]
    return ", ".join(part for part in parts if part)


async def resolve_place(
    client: OpenMeteoClient,
    place: str | None,
    *,
    home_latitude: float,
    home_longitude: float,
    local_radius_km: float,
) -> tuple[float, float, str]:
    """Resolve a spoken place to coordinates and a spoken location name.

    A missing or blank `place` resolves to the house's own coordinates and
    the string "home", with no geocoding call. Otherwise the name before
    any qualifier is sent to the geocoder, and `select_place` picks one
    match. Raises `ValueError` naming the place when nothing is left to
    resolve to -- the decorated tools turn that into a `ToolError`, the
    same way they already turn a bad day count into one.
    """
    if place is None or not place.strip():
        return home_latitude, home_longitude, "home"
    name, qualifier = split_place(place)
    match = None
    if name:
        matches = await client.geocode(name)
        match = select_place(
            matches,
            qualifier=qualifier,
            home_latitude=home_latitude,
            home_longitude=home_longitude,
            local_radius_km=local_radius_km,
        )
    if match is None:
        raise ValueError(f"I could not find a place called {place.strip()}.")
    return match["latitude"], match["longitude"], format_location(match)


async def handle_weather_current(
    client: OpenMeteoClient,
    latitude: float,
    longitude: float,
    *,
    place: str | None = None,
    local_radius_km: float = _DEFAULT_LOCAL_RADIUS_KM,
    units: str = "celsius",
) -> dict[str, Any]:
    """Answer a question about the current outdoor weather, from an
    external service (open-meteo) -- never a room's own temperature sensor.

    Resolves `place` first (see `resolve_place`), then returns a dict
    already close to speech: a temperature in the configured unit (`unit`
    names it), a short condition phrase, the location's local time, and
    the resolved `location`. Rounded to one decimal place, the precision a
    person says out loud.

    Reads the unit only from its own `units` parameter, never from the
    `_units` module global -- keeps this handler pure, the same property
    `test_no_handler_body_references_a_module_level_client_name` already
    proves for the client.
    """
    resolved_latitude, resolved_longitude, location = await resolve_place(
        client,
        place,
        home_latitude=latitude,
        home_longitude=longitude,
        local_radius_km=local_radius_km,
    )
    result = await client.current_conditions(
        resolved_latitude, resolved_longitude, temperature_unit=units
    )
    return {
        "temperature": round(result["temperature"], 1),
        "unit": _unit_symbol(units),
        "condition": result["condition"],
        "local_time": result["local_time"],
        "location": location,
    }


async def handle_weather_forecast(
    client: OpenMeteoClient,
    latitude: float,
    longitude: float,
    days: int,
    *,
    place: str | None = None,
    local_radius_km: float = _DEFAULT_LOCAL_RADIUS_KM,
) -> dict[str, Any]:
    """Answer a question about the outdoor weather forecast, from an
    external service (open-meteo) -- never a room's own temperature sensor.

    Checks `days` before resolving `place`, so a bad day count makes no
    network call at all. Returns up to `days` days, each with a high, a
    low (Celsius), and a short condition phrase, plus the resolved
    `location`. `days` must be between 1 and 16 -- the range open-meteo
    itself accepts -- and a count outside that range is refused rather
    than clamped, so a mistaken request never silently answers a different
    question than the one asked.
    """
    if not (_MIN_FORECAST_DAYS <= days <= _MAX_FORECAST_DAYS):
        raise ValueError(
            f"a forecast day count must be between {_MIN_FORECAST_DAYS} and "
            f"{_MAX_FORECAST_DAYS}, got {days}"
        )
    resolved_latitude, resolved_longitude, location = await resolve_place(
        client,
        place,
        home_latitude=latitude,
        home_longitude=longitude,
        local_radius_km=local_radius_km,
    )
    result = await client.forecast(resolved_latitude, resolved_longitude, days)
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
        "location": location,
    }


mcp_server = MCPServer("spire-weather")

_http_client: httpx.AsyncClient | None = None
_open_meteo_client: OpenMeteoClient | None = None
_latitude: float = 0.0
_longitude: float = 0.0
_local_radius_km: float = _DEFAULT_LOCAL_RADIUS_KM
_units: str = "celsius"


@mcp_server.tool()
async def weather_current(place: str | None = None) -> dict[str, Any]:
    """Answer a question about the current outdoor weather, from an
    external service (open-meteo) -- never a room's own temperature sensor.

    Leave out `place` for the weather at home. Set `place` to the place
    name as the user said it. If the user also said a country or a
    region, add it after a comma, for example "Paris, France" or
    "Springfield, Illinois". Do not add a qualifier the user did not say.
    The result's `location` names the place the answer is for, or "home".
    Say that place in the reply. The result's `temperature` is in the unit
    its `unit` field names: "F" means Fahrenheit and "C" means Celsius.
    Say that unit in the reply.
    """
    assert _open_meteo_client is not None, "weather_current invoked before startup"
    try:
        return await handle_weather_current(
            _open_meteo_client,
            _latitude,
            _longitude,
            place=place,
            local_radius_km=_local_radius_km,
            units=_units,
        )
    except (UpstreamUnreachableError, UpstreamMalformedError, ValueError) as exc:
        # The exception's own sentence crosses this boundary unchanged --
        # the same "boundary's own words, nothing reworded" doctrine the
        # Home Assistant server already follows for its own refusals.
        raise ToolError(str(exc)) from exc


@mcp_server.tool()
async def weather_forecast(
    days: int = _DEFAULT_FORECAST_DAYS, place: str | None = None
) -> dict[str, Any]:
    """Answer a question about the outdoor weather forecast, from an
    external service (open-meteo) -- never a room's own temperature sensor.

    Returns up to `days` days (1-16) of high, low, and condition. Leave
    out `place` for the forecast at home. Set `place` to the place name as
    the user said it. If the user also said a country or a region, add it
    after a comma, for example "Paris, France" or "Springfield, Illinois".
    Do not add a qualifier the user did not say. The result's `location`
    names the place the answer is for, or "home". Say that place in the
    reply.
    """
    assert _open_meteo_client is not None, "weather_forecast invoked before startup"
    try:
        return await handle_weather_forecast(
            _open_meteo_client,
            _latitude,
            _longitude,
            days,
            place=place,
            local_radius_km=_local_radius_km,
        )
    except (UpstreamUnreachableError, UpstreamMalformedError, ValueError) as exc:
        # LOW-01 fix (phase 4 code review): `handle_weather_forecast` raises
        # a bare `ValueError` for a day count outside 1-16 -- previously not
        # named in this `except` clause, so it propagated as a raw,
        # uncaught exception through the MCP SDK's own generic error path
        # instead of this module's "boundary's own words, nothing reworded"
        # doctrine, the same swallowing Phase 3's CR-01 fix addressed for
        # `Denied`. `str(exc)` on a `ValueError` is exactly the message
        # `handle_weather_forecast` raised, so this crosses the boundary in
        # the same voice as every other refusal in this module. The same
        # clause also carries `resolve_place`'s unknown-place `ValueError`.
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


def _read_local_radius_km() -> float:
    """Read WEATHER_LOCAL_RADIUS_KM from this process's own environment.

    Missing or blank gives the default of 300 km. A value that will not
    parse, is not finite, or is negative stops startup, naming the
    variable -- the same refuse-rather-than-guess posture
    `_read_coordinate` already takes. The plugin manager builds a
    plugin's child environment only from its declared config keys
    (`_env_from_config_values` in `manager.py`), so in a real deployment
    this key is absent and the default applies; a value of 0 is valid and
    turns off the local preference entirely.
    """
    raw = os.environ.get("WEATHER_LOCAL_RADIUS_KM")
    if raw is None or not raw.strip():
        return _DEFAULT_LOCAL_RADIUS_KM
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"WEATHER_LOCAL_RADIUS_KM is not a valid number: {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise SystemExit(
            f"WEATHER_LOCAL_RADIUS_KM must be a finite, non-negative number: {raw!r}"
        )
    return value


def _read_units() -> str:
    """Read WEATHER_UNITS from this process's own environment.

    Missing or blank means celsius -- a catalog install stores an empty
    string and migration 0013 seeds the literal "celsius", and both mean
    the same thing. Matching ignores case and surrounding spaces. Any
    value outside the two closed alias sets stops startup, naming the
    variable and the raw value -- the same refuse-rather-than-guess
    posture `_read_coordinate` already takes.
    """
    raw = os.environ.get("WEATHER_UNITS")
    if raw is None or not raw.strip():
        return "celsius"
    normalized = raw.strip().casefold()
    if normalized in _FAHRENHEIT_UNIT_ALIASES:
        return "fahrenheit"
    if normalized in _CELSIUS_UNIT_ALIASES:
        return "celsius"
    raise SystemExit(f"WEATHER_UNITS must be celsius or fahrenheit, got {raw!r}")


def _unit_symbol(units: str) -> str:
    """The one-letter label a spoken reply names -- "F" or "C".

    "fahrenheit" gives "F"; any other value gives "C". The one definition
    of this label, so a handler's returned `unit` and the request
    parameter it sent can never disagree.
    """
    return "F" if units == "fahrenheit" else "C"


def _startup() -> None:
    global _http_client, _open_meteo_client, _latitude, _longitude, _local_radius_km, _units
    _latitude = _read_coordinate("WEATHER_LATITUDE")
    _longitude = _read_coordinate("WEATHER_LONGITUDE")
    _local_radius_km = _read_local_radius_km()
    _units = _read_units()
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
