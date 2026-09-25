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


@dataclass(frozen=True)
class Clarification:
    """D-04: what to ask the operator when a write proposal's target is
    ambiguous -- which account, or (once an account is settled) which of
    its read-write calendars. `about` names which question this is
    (`"account"` or `"calendar"`); `candidates` are the labels or calendar
    names to speak, sorted so the same ambiguity always produces the same
    question.
    """

    about: str
    candidates: tuple[str, ...]


def _resolve_calendar_for_account(
    account: AccountGrant, calendar_name: "str | None"
) -> "tuple[AccountGrant, CalendarGrant] | Clarification":
    """D-05: the one calendar a proposal against `account` may target --
    the calendar named (case-insensitive), else the account's primary
    calendar when it is writable, else its only writable calendar, else a
    `Clarification(about="calendar")` over every writable calendar's name.

    A calendar that is off never reaches `account.calendars` at all
    (`GoogleEnvBuilder`'s own upstream filter) -- this function only ever
    tells a read-write calendar apart from a read-only one.
    """
    if calendar_name is not None:
        normalized_cal = calendar_name.strip().casefold()
        match = next((c for c in account.calendars if c.name.casefold() == normalized_cal), None)
        if match is None:
            raise Denied(
                f"the {calendar_name} calendar isn't turned on for {account.label} -- "
                "turn it on in the google accounts screen"
            )
        if match.access == "read_only":
            raise Denied(
                f"the {match.name} calendar in {account.label} is read only -- "
                "change that in the google accounts screen"
            )
        return (account, match)

    primary = next((c for c in account.calendars if c.primary), None)
    if primary is not None and primary.access == "read_write":
        return (account, primary)

    read_write_calendars = [c for c in account.calendars if c.access == "read_write"]
    if len(read_write_calendars) == 1:
        return (account, read_write_calendars[0])
    if not read_write_calendars:
        raise Denied(
            f"turn on a calendar for read and write on {account.label} in the google accounts screen"
        )
    return Clarification(about="calendar", candidates=tuple(sorted(c.name for c in read_write_calendars)))


def _resolve_unnamed_write_target(
    accounts: "tuple[AccountGrant, ...]",
) -> "tuple[AccountGrant, CalendarGrant] | Clarification":
    """D-04: no account named in the proposal -- the one account with a
    writable calendar, else the operator's default account, else a
    `Clarification(about="account")` over every writable account's label
    (sorted, so the same ambiguity always asks the same question).

    A candidate is any linked account carrying at least one read-write
    calendar -- an account with no writable calendar at all is never a
    candidate, the same way it is never reachable by name either
    (`_resolve_calendar_for_account`'s own "turn on a calendar for read and
    write" refusal).

    R2-WR-08: the target is chosen from the writable accounts first, and
    reachability is checked on the chosen account after. An unreachable
    chosen account (GOOG-12, `access_token is None`) is refused by name,
    up front -- never replaced by another account in silence (D-04 sends
    a write with no signal to the default), and never read back only to
    fail at execute time (B1-WR-02). A clarification names every writable
    account. If the operator then names an unreachable one, the named path
    (`resolve_write_target` below) refuses it the same way.

    Once exactly one account is settled on, its own calendar is resolved
    the identical way a named account's would be
    (`_resolve_calendar_for_account` with no calendar named) -- there is
    no second calendar-selection mechanism for the unnamed path.
    """
    writable = [account for account in accounts if any(c.access == "read_write" for c in account.calendars)]
    if not writable:
        raise Denied(
            "turn on a calendar for read and write on a google account in the google accounts screen"
        )

    chosen: "AccountGrant | None" = None
    if len(writable) == 1:
        chosen = writable[0]
    else:
        defaults = [account for account in writable if account.is_default]
        if len(defaults) == 1:
            chosen = defaults[0]

    if chosen is None:
        return Clarification(about="account", candidates=tuple(sorted(a.label for a in writable)))
    if chosen.access_token is None:
        raise Denied(f"i can't reach your {chosen.label} account right now")
    return _resolve_calendar_for_account(chosen, None)


def require_writable(
    accounts: "tuple[AccountGrant, ...]", account_label: str, calendar_id: str
) -> "tuple[AccountGrant, CalendarGrant]":
    """T-09-28: the one gate every EXECUTING calendar-write tool runs
    first, re-checked against this env at the moment it actually runs --
    never trusting a proposal-time `resolve_write_target` call from
    earlier in the turn, since the operator may have turned the calendar
    off or made it read-only in the time between the readback and the
    confirmation.

    Raises `Denied`, with a spoken reason, before any HTTP request exists:
    an account this env does not carry, an account whose access token
    could not be refreshed (GOOG-12), a calendar id this account does not
    carry (turned off, or never existed), and a read-only calendar.
    """
    normalized = account_label.strip().casefold()
    account = next((a for a in accounts if a.label.casefold() == normalized), None)
    if account is None:
        linked = ", ".join(sorted(a.label for a in accounts)) or "none linked"
        raise Denied(f"i don't have a google account called {account_label!r} -- linked accounts: {linked}")
    if account.access_token is None:
        raise Denied(f"i can't reach your {account.label} account right now")
    calendar = next((c for c in account.calendars if c.calendar_id == calendar_id), None)
    if calendar is None:
        raise Denied(
            f"that calendar isn't turned on for {account.label} -- turn it on in the google accounts screen"
        )
    if calendar.access == "read_only":
        raise Denied(
            f"the {calendar.name} calendar in {account.label} is read only -- "
            "change that in the google accounts screen"
        )
    return account, calendar


def resolve_write_target(
    accounts: "tuple[AccountGrant, ...]", account_label: "str | None", calendar_name: "str | None"
) -> "tuple[AccountGrant, CalendarGrant] | Clarification":
    """D-04/D-05: the write target for a calendar proposal -- an exact
    `(AccountGrant, CalendarGrant)` pair, or a `Clarification` naming what
    to ask the operator.

    A named account resolves its calendar through `_resolve_calendar_for_account`;
    an account whose token could not be refreshed (`access_token is None`,
    GOOG-12) is refused with a spoken reason before its calendar is ever
    resolved. No account named resolves through `_resolve_unnamed_write_target`:
    the only account with a writable calendar, else the operator's default
    account, else a clarifying question over the candidates.
    """
    if account_label is None:
        return _resolve_unnamed_write_target(accounts)
    normalized = account_label.strip().casefold()
    account = next((a for a in accounts if a.label.casefold() == normalized), None)
    if account is None:
        linked = ", ".join(sorted(a.label for a in accounts)) or "none linked"
        raise Denied(f"i don't have a google account called {account_label!r} -- linked accounts: {linked}")
    if account.access_token is None:
        raise Denied(f"i can't reach your {account.label} account right now")
    return _resolve_calendar_for_account(account, calendar_name)
