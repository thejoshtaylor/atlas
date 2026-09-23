"""A pure open-meteo client: fetch, cache for a bounded window, and fail in
two distinguishable ways.

This module imports nothing from `spire_mcp.safety`. Weather controls
nothing -- there is no entity to deny, so there is no denylist concept here.
That absence is deliberate, the same way `handle_get_state` in `ha.py` notes
"a read is never denied": here there is not even anything read-shaped to
gate, so the boundary this module needs is a smaller one than `ha.py`'s.

This module also reads no environment variable and holds no credential.
open-meteo's free, non-commercial forecast endpoint needs no key, which is
the whole point: a plugin scoped to exactly what it needs holds nothing
else, and the absence of any environment lookup anywhere in this file is
that scoping made checkable by `grep`. The geocoding endpoint this module
also calls needs no key either, so this remains true across both.

The `httpx.AsyncClient` and the cache TTL both arrive as constructor
arguments, never constructed internally -- dependency injection over
subclassing, so a test drives this exact class through `httpx.MockTransport`
rather than a fake stand-in. The cache's clock is injectable too, defaulting
to `time.monotonic`, so a test can move time forward without sleeping.

Two exception classes carry the two facts a caller needs to distinguish:
`UpstreamUnreachableError` for a network that would not answer or a status
open-meteo itself calls an error, and `UpstreamMalformedError` for a 200
response whose body the parser cannot read. They are separate because they
call for different operator actions -- a down network and a changed upstream
schema are not the same problem -- and because a parser that quietly
swallowed a missing field into a partial, half-guessed dict would be worse
than telling the operator nothing was answered at all.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# WMO weather interpretation codes, as documented at open-meteo.com/en/docs.
# An unrecognised code is not a malformed response -- open-meteo may add a
# code this table does not yet know about, and that is a labelling gap, not
# an unreadable body -- so it falls back to a generic phrase rather than
# raising.
_WEATHER_CODE_CONDITIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _condition_for(weather_code: Any) -> str:
    try:
        code = int(weather_code)
    except (TypeError, ValueError):
        return "changing conditions"
    return _WEATHER_CODE_CONDITIONS.get(code, "changing conditions")


class UpstreamUnreachableError(Exception):
    """open-meteo could not be reached, or answered with a non-2xx status.

    Raised for a connection error, a timeout, or any response whose status
    is not in the 2xx range. `str(self)` is a short sentence fit to be
    spoken, following the same "boundary's own words, nothing reworded"
    doctrine the Home Assistant refusal path already follows.
    """


class UpstreamMalformedError(Exception):
    """open-meteo answered 200 with a body this parser cannot read.

    Raised when a field the parser needs is missing or shaped unexpectedly.
    A different exception from `UpstreamUnreachableError` on purpose: a
    network that is down and a provider that changed its response shape are
    different facts, and conflating them would tell the operator the wrong
    thing to do next.
    """


class OpenMeteoClient:
    """An async open-meteo client with a short, bounded-window cache.

    The cache exists to keep a repeated question out of the turn's latency
    budget, not to stay under open-meteo's free-tier rate limit -- a single
    household will never approach that limit. Ten to fifteen minutes is the
    right size for forecast data that itself only changes on the order of
    minutes.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        ttl_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}

    async def current_conditions(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Fetch (or serve from cache) the current temperature and condition.

        Returns a plain dict -- no SDK object, no ORM row -- carrying a
        temperature, a condition, the location's local time, and its
        timezone name.
        """
        key = ("current", latitude, longitude, 0)
        cached = self._cached(key)
        if cached is not None:
            return cached
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code,is_day",
            "timezone": "auto",
        }
        body = await self._fetch(params)
        result = self._parse_current(body)
        self._store(key, result)
        return result

    async def forecast(self, latitude: float, longitude: float, days: int) -> dict[str, Any]:
        """Fetch (or serve from cache) a `days`-day forecast.

        `days` is passed straight through to open-meteo's own
        `forecast_days` parameter; bounding it to a range a person would
        actually ask for is the handler's job (`weather.py`), not this
        client's -- this client's job is fetching and parsing, nothing more.
        """
        key = ("forecast", latitude, longitude, days)
        cached = self._cached(key)
        if cached is not None:
            return cached
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "forecast_days": days,
            "timezone": "auto",
        }
        body = await self._fetch(params)
        result = self._parse_forecast(body)
        self._store(key, result)
        return result

    async def geocode(self, name: str) -> list[dict[str, Any]]:
        """Look up open-meteo's geocoding matches for a spoken place name.

        Returns a list of plain dicts, one for each candidate, in the
        geocoder's own order. An empty list means no match -- that is not
        an error. `select_place` in `weather.py` picks one of them.
        """
        key = ("geocode", name)
        cached = self._cached(key)
        if cached is not None:
            return cached
        params = {
            "name": name,
            "count": 10,
            "language": "en",
            "format": "json",
        }
        body = await self._fetch(params, url=_GEOCODING_URL)
        result = self._parse_geocode(body)
        self._store(key, result)
        return result

    def _cached(self, key: tuple[Any, ...]) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        cached_at, value = entry
        if self._clock() - cached_at >= self._ttl_seconds:
            return None
        return value

    def _store(self, key: tuple[Any, ...], value: Any) -> None:
        self._cache[key] = (self._clock(), value)

    async def _fetch(self, params: dict[str, Any], *, url: str = _FORECAST_URL) -> dict[str, Any]:
        try:
            response = await self._client.get(url, params=params)
        except httpx.RequestError as exc:
            raise UpstreamUnreachableError("I can't reach the weather service right now.") from exc
        if response.status_code // 100 != 2:
            raise UpstreamUnreachableError(
                f"The weather service answered with an error (status {response.status_code})."
            )
        try:
            return response.json()
        except ValueError as exc:
            # A 200 whose body is not even JSON is unreadable, not merely
            # missing a field -- the same malformed-upstream fact either way.
            raise UpstreamMalformedError(
                "The weather service sent back something I couldn't read."
            ) from exc

    def _parse_current(self, body: Any) -> dict[str, Any]:
        try:
            current = body["current"]
            temperature_c = current["temperature_2m"]
            weather_code = current["weather_code"]
            local_time = current["time"]
            timezone = body["timezone"]
        except (KeyError, TypeError) as exc:
            raise UpstreamMalformedError(
                "The weather service sent back something I couldn't read."
            ) from exc
        return {
            "temperature_c": temperature_c,
            "condition": _condition_for(weather_code),
            "local_time": local_time,
            "timezone": timezone,
        }

    def _parse_forecast(self, body: Any) -> dict[str, Any]:
        try:
            daily = body["daily"]
            dates = daily["time"]
            highs = daily["temperature_2m_max"]
            lows = daily["temperature_2m_min"]
            codes = daily["weather_code"]
            timezone = body["timezone"]
        except (KeyError, TypeError) as exc:
            raise UpstreamMalformedError(
                "The weather service sent back something I couldn't read."
            ) from exc
        if not (len(dates) == len(highs) == len(lows) == len(codes)):
            raise UpstreamMalformedError(
                "The weather service sent back something I couldn't read."
            )
        days = [
            {
                "date": date,
                "high_c": high,
                "low_c": low,
                "condition": _condition_for(code),
            }
            for date, high, low, code in zip(dates, highs, lows, codes)
        ]
        return {"days": days, "timezone": timezone}

    def _parse_geocode(self, body: Any) -> list[dict[str, Any]]:
        if not isinstance(body, dict):
            raise UpstreamMalformedError(
                "The weather service sent back something I couldn't read."
            )
        results = body.get("results")
        if results is None:
            return []
        if not isinstance(results, list):
            raise UpstreamMalformedError(
                "The weather service sent back something I couldn't read."
            )
        matches: list[dict[str, Any]] = []
        for entry in results:
            try:
                name = entry["name"]
                if not isinstance(name, str):
                    raise TypeError("a geocoding match's name must be a string")
                latitude = float(entry["latitude"])
                longitude = float(entry["longitude"])
            except (KeyError, TypeError, ValueError) as exc:
                raise UpstreamMalformedError(
                    "The weather service sent back something I couldn't read."
                ) from exc
            matches.append(
                {
                    "name": name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "country": entry.get("country") or "",
                    "country_code": entry.get("country_code") or "",
                    "admin1": entry.get("admin1") or "",
                    "population": entry.get("population"),
                }
            )
        return matches
