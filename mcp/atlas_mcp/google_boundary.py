"""The `GOOGLE_ACCOUNTS_JSON` wire shape, and the boundary that decides
which linked account(s) a request reaches -- the per-calendar and
per-account equivalent of `atlas_mcp.safety`'s entity denylist, D-05
("code enforces this at the tool boundary, in the same way as the entity
denylist. A prompt is not an enforcement point").

`CalendarGrant`/`AccountGrant` are the child's own in-memory view of
exactly what `src/atlas/google/env.py::GoogleEnvBuilder` wrote --
never the refresh token or the OAuth client secret (T-09-01):
`parse_accounts_env` is this file's only reader of the raw environment
string, and nothing downstream of it ever sees anything wider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from atlas_mcp.safety import Denied

# The one env key this whole shape lives behind -- kept as a plain string
# here (not imported from `atlas_mcp.google_tools`) so this module has no
# import-time dependency on that one; both name the identical value, and
# `google.py`'s own `_startup()` imports the canonical constant from
# `google_tools` directly.
_ACCOUNTS_KEY = "accounts"
_CALENDARS_KEY = "calendars"


@dataclass(frozen=True)
class CalendarGrant:
    """One calendar an account carries into the child -- already filtered
    to "not off" by `GoogleEnvBuilder` (D-05): a calendar the operator has
    not enabled never reaches this dataclass, or the child, at all."""

    calendar_id: str
    name: str
    primary: bool
    access: str


@dataclass(frozen=True)
class AccountGrant:
    """One linked account as the child sees it -- `access_token` is
    `None` exactly when `unreachable_reason` names why (an account
    `GoogleTokenService` could not refresh, GOOG-12); such an account is
    still present here (so it can be named in `unreachable_accounts`),
    just with nothing to call Google with."""

    label: str
    email: str
    is_default: bool
    access_token: str | None
    unreachable_reason: str | None
    calendars: tuple[CalendarGrant, ...]


def parse_accounts_env(raw: str | None) -> tuple[AccountGrant, ...]:
    """Parse `GOOGLE_ACCOUNTS_JSON`'s raw string into `AccountGrant`s.

    Absent or empty (no accounts linked, or the key was not set at all)
    returns `()` -- an empty tuple is a real, supported state, never an
    error. Anything present but malformed -- invalid JSON, the wrong
    top-level shape, or a field missing its expected type -- raises
    `ValueError` naming the field, so `_startup()` can turn it into a
    loud `SystemExit` (the same fail-closed posture `ha.py::_load_policy`
    already takes toward a malformed `ATLAS_SAFETY`) rather than starting
    with a silently empty or partial account list.
    """
    if raw is None or not raw.strip():
        return ()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GOOGLE_ACCOUNTS_JSON is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"GOOGLE_ACCOUNTS_JSON must be a JSON object, got {type(body).__name__}")
    raw_accounts = body.get(_ACCOUNTS_KEY, [])
    if not isinstance(raw_accounts, list):
        raise ValueError(f"GOOGLE_ACCOUNTS_JSON.{_ACCOUNTS_KEY} must be a list")

    accounts: list[AccountGrant] = []
    for index, raw_account in enumerate(raw_accounts):
        if not isinstance(raw_account, dict):
            raise ValueError(f"GOOGLE_ACCOUNTS_JSON.{_ACCOUNTS_KEY}[{index}] must be an object")
        try:
            label = str(raw_account["label"])
            email = str(raw_account["email"])
            is_default = bool(raw_account["is_default"])
            access_token = raw_account.get("access_token")
            unreachable_reason = raw_account.get("unreachable_reason")
        except KeyError as exc:
            raise ValueError(
                f"GOOGLE_ACCOUNTS_JSON.{_ACCOUNTS_KEY}[{index}] is missing field {exc}"
            ) from exc
        raw_calendars = raw_account.get(_CALENDARS_KEY, [])
        if not isinstance(raw_calendars, list):
            raise ValueError(
                f"GOOGLE_ACCOUNTS_JSON.{_ACCOUNTS_KEY}[{index}].{_CALENDARS_KEY} must be a list"
            )
        calendars: list[CalendarGrant] = []
        for cal_index, raw_calendar in enumerate(raw_calendars):
            if not isinstance(raw_calendar, dict):
                raise ValueError(
                    f"GOOGLE_ACCOUNTS_JSON.{_ACCOUNTS_KEY}[{index}].{_CALENDARS_KEY}[{cal_index}] "
                    "must be an object"
                )
            try:
                calendars.append(
                    CalendarGrant(
                        calendar_id=str(raw_calendar["calendar_id"]),
                        name=str(raw_calendar["name"]),
                        primary=bool(raw_calendar["primary"]),
                        access=str(raw_calendar["access"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"GOOGLE_ACCOUNTS_JSON.{_ACCOUNTS_KEY}[{index}].{_CALENDARS_KEY}[{cal_index}] "
                    f"is missing field {exc}"
                ) from exc
        accounts.append(
            AccountGrant(
                label=label,
                email=email,
                is_default=is_default,
                access_token=(str(access_token) if access_token is not None else None),
                unreachable_reason=(
                    str(unreachable_reason) if unreachable_reason is not None else None
                ),
                calendars=tuple(calendars),
            )
        )
    return tuple(accounts)


def resolve_accounts(
    accounts: tuple[AccountGrant, ...], label: str | None
) -> tuple[AccountGrant, ...]:
    """D-04: a request naming no account reads every account; a request
    naming one (matched case-insensitively) reads that one.

    Raises `Denied` naming every linked label when `label` matches none of
    them -- never a guess, the same "distinguishable, spoken reason"
    discipline `ha.py::_resolve_target` already establishes for an unknown
    area/device/label target.
    """
    if label is None:
        return accounts
    normalized = label.strip().casefold()
    for account in accounts:
        if account.label.casefold() == normalized:
            return (account,)
    linked = ", ".join(sorted(a.label for a in accounts)) or "none linked"
    raise Denied(f"i don't have a google account called {label!r} -- linked accounts: {linked}")
