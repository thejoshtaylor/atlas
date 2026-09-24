"""The curated plugin list, shipped as a static file in the repository
(D-13, PLUG-03): `config/plugin-catalog.json`, read once per call, never
fetched over the network. This module performs no network access of any
kind -- an install screen that depended on a live registry would break the
moment the house is offline, and would put a third party in a position to
decide what a house can install (D-13's own reasoning, `06-CONTEXT.md`).

Per entry: a name, a description, a transport, its arguments or its URL,
and the configuration keys it needs, each with a human label and a secret
flag (Task 1's own schema, D-16) -- no icon or logo field, per
`06-UI-SPEC.md`'s own stated assumption that this document specifies a
text-only catalog row.

Read with the same discipline `alembic/versions/0005_macro_tables.py`'s
`_read_macros_block` already uses for a file a feature depends on: an
unreadable or malformed file raises `CatalogError` naming the path, rather
than silently serving an empty list an admin would read as "there is
nothing available." A file with zero entries is a different, legitimate
state -- it parses cleanly and serves `()`, matching 06-UI-SPEC.md's own
"From catalog" empty-collection state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# IN-01 (code review): derived from this file's own location, the same
# computation `app.py`'s `MCP_ROOT`/`FRONTEND_DIR` use (src/atlas/
# plugins/catalog.py -> plugins -> atlas -> src -> repo root), with
# an environment override for a deployment that mounts the catalog
# somewhere else. It used to be the relative string
# `"config/plugin-catalog.json"`, which resolved against the process's
# current working directory: started from anywhere but the repository
# root, `GET /api/plugins/catalog` raised `CatalogError` and the plugins
# screen showed a raw 500 rather than one of this codebase's own named
# refusals.
DEFAULT_CATALOG_PATH = os.environ.get(
    "ATLAS_PLUGIN_CATALOG",
    str(Path(__file__).resolve().parents[3] / "config" / "plugin-catalog.json"),
)

# The two transports D-02 ships -- an entry names exactly one of them, and
# the args-or-url exclusivity below is enforced against this same set.
_VALID_TRANSPORTS = ("stdio", "remote")


class CatalogError(Exception):
    """The shipped catalog file could not be read, or does not parse into
    the shape this module requires -- always names the path, per the
    module docstring's read-or-raise discipline. Raised at load time, not
    discovered later when an admin tries to install from a broken entry."""


@dataclass(frozen=True)
class CatalogConfigKey:
    """One configuration key a catalog entry declares -- its own name, a
    human label for the install form, and whether it is secret (D-16). No
    entry ever carries a *value* here, only the declaration of what a key
    is and whether it is secret -- what any of them actually contain lives
    in `plugin_config_values`, encrypted, never in this file."""

    key: str
    label: str
    secret: bool


@dataclass(frozen=True)
class CatalogEntry:
    """One curated plugin, exactly as `config/plugin-catalog.json` shipped
    it -- a name, a description, a transport, its stdio `args` or its
    remote `url` (mutually exclusive, enforced at load time), and the
    configuration keys it declares."""

    name: str
    description: str
    transport: str
    args: "tuple[str, ...]"
    url: "str | None"
    config_keys: "tuple[CatalogConfigKey, ...]"


def _malformed_error(path: "str | os.PathLike[str]", reason: str) -> CatalogError:
    return CatalogError(f"the plugin catalog at {os.fspath(path)!r} is malformed: {reason}")


def _parse_config_key(path: "str | os.PathLike[str]", entry_name: str, raw: object) -> CatalogConfigKey:
    if not isinstance(raw, dict):
        raise _malformed_error(path, f"entry {entry_name!r} has a config key that is not an object")
    key = raw.get("key")
    label = raw.get("label")
    secret = raw.get("secret")
    if not isinstance(key, str) or not key:
        raise _malformed_error(path, f"entry {entry_name!r} has a config key with no 'key' name")
    if not isinstance(label, str) or not label:
        raise _malformed_error(path, f"entry {entry_name!r}'s config key {key!r} has no 'label'")
    if not isinstance(secret, bool):
        raise _malformed_error(
            path, f"entry {entry_name!r}'s config key {key!r} has no boolean 'secret' flag"
        )
    return CatalogConfigKey(key=key, label=label, secret=secret)


def _parse_entry(path: "str | os.PathLike[str]", raw: object) -> CatalogEntry:
    if not isinstance(raw, dict):
        raise _malformed_error(path, "an entry is not an object")
    name = raw.get("name")
    description = raw.get("description")
    transport = raw.get("transport")
    args_raw = raw.get("args") or []
    url = raw.get("url")
    config_keys_raw = raw.get("config_keys") or []

    if not isinstance(name, str) or not name:
        raise _malformed_error(path, "an entry has no 'name'")
    if not isinstance(description, str) or not description:
        raise _malformed_error(path, f"entry {name!r} has no 'description'")
    if transport not in _VALID_TRANSPORTS:
        raise _malformed_error(
            path, f"entry {name!r} has transport {transport!r} -- expected one of {_VALID_TRANSPORTS!r}"
        )
    if not isinstance(args_raw, list) or not all(isinstance(a, str) for a in args_raw):
        raise _malformed_error(path, f"entry {name!r}'s 'args' must be a list of strings")
    if url is not None and not isinstance(url, str):
        raise _malformed_error(path, f"entry {name!r}'s 'url' must be a string or null")

    has_args = bool(args_raw)
    has_url = bool(url)
    if has_args and has_url:
        raise _malformed_error(
            path, f"entry {name!r} declares both 'args' and 'url' -- a plugin is one transport, not two"
        )
    if transport == "stdio" and not has_args:
        raise _malformed_error(path, f"entry {name!r} has transport 'stdio' but no 'args'")
    if transport == "remote" and not has_url:
        raise _malformed_error(path, f"entry {name!r} has transport 'remote' but no 'url'")

    if not isinstance(config_keys_raw, list):
        raise _malformed_error(path, f"entry {name!r}'s 'config_keys' must be a list")
    config_keys = tuple(_parse_config_key(path, name, raw_key) for raw_key in config_keys_raw)

    return CatalogEntry(
        name=name,
        description=description,
        transport=transport,
        args=tuple(args_raw),
        url=url,
        config_keys=config_keys,
    )


def load_catalog(path: "str | os.PathLike[str]" = DEFAULT_CATALOG_PATH) -> "tuple[CatalogEntry, ...]":
    """Read and validate the shipped catalog file, returning every entry
    it declares -- `()` for a file that genuinely declares zero entries
    (a real state, per the module docstring), never for a file this
    function could not read or parse, which raises `CatalogError` naming
    `path` instead.

    Every entry is validated here, at read time -- including the
    args-or-url exclusivity -- so a malformed catalog is caught once, at
    the moment this is first called, rather than at the moment an admin
    tries to install from whichever entry happens to be broken.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            raw_text = fh.read()
    except OSError as exc:
        raise CatalogError(
            f"the plugin catalog could not be read at {os.fspath(path)!r} -- refusing to serve an "
            "empty list silently. Either make the file readable at that path, or point the catalog "
            "loader at the file that holds the curated plugin list."
        ) from exc

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"the plugin catalog at {os.fspath(path)!r} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict) or "entries" not in raw:
        raise _malformed_error(path, "expected a JSON object with an 'entries' list")
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, list):
        raise _malformed_error(path, "'entries' must be a list")

    return tuple(_parse_entry(path, entry_raw) for entry_raw in entries_raw)


def find_entry(entries: "Sequence[CatalogEntry]", name: str) -> "CatalogEntry | None":
    """The one entry in `entries` named `name`, or `None` -- catalog
    entries are looked up by their own `name` field (the only identifier
    this file's schema carries; catalog entries have no separate id)."""
    for entry in entries:
        if entry.name == name:
            return entry
    return None
