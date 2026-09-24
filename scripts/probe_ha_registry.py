#!/usr/bin/env python3
"""Ask a real, running Home Assistant whether its area/device/label/entity
registries answer over the WebSocket API -- the transport SAFE-03's
expansion in `mcp/atlas_mcp/ha.py` depends on, and the one claim in
03-RESEARCH.md never confirmed against a live instance (MEDIUM confidence:
corroborated by official documentation and by the community, but not
falsified against a real house).

Run this before trusting the expansion logic against real Home Assistant
data. `docs/runbooks/ha-registry-expansion.md` walks through what a
healthy answer looks like and what to do if this script reports a
rejection instead.

Follows `scripts/calibrate_echo_path.py`'s own shape: a thin caller of the
real client (`atlas_mcp.registry.HaRegistryClient`) and nothing else. This
script reads no environment variable it did not already need --
`HA_URL`/`HA_TOKEN` are the same two the running child (`atlas_mcp.ha`)
already requires, read directly from the environment the same way
`atlas_mcp.ha._startup()` does. Run it through
`scripts/dev-probe-ha-registry.sh`, which sources `.env` the way
`dev-run.sh`/`dev-calibrate-echo.sh` do.

Prints counts and shapes for every registry fetched. Prints entity ids
only for the one area `--area` explicitly names -- this is the operator's
own terminal, not a log line this project ships anywhere, but house data
still has no reason to appear on screen beyond what was asked for.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# `atlas_mcp` lives under `mcp/`, not `src/` -- it is the package the MCP
# child process runs, and this probe is a second, standalone caller of its
# registry client. Inserted here so the script also runs bare
# (`.venv/bin/python scripts/probe_ha_registry.py`), not only through
# `scripts/dev-probe-ha-registry.sh`'s `PYTHONPATH=mcp` wrapper.
_MCP_ROOT = Path(__file__).resolve().parents[1] / "mcp"
if str(_MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_ROOT))

from atlas_mcp.registry import (
    HaRegistryClient,
    RegistryAuthError,
    RegistryUnavailableError,
    UnknownRegistryTargetError,
    expand_area,
    ws_url_from_http,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to a real Home Assistant over the WebSocket API and report "
            "whether its area/device/label/entity registries answer, including how "
            "many entities inherit their area from a device rather than carrying "
            "their own -- the case a naive expansion silently drops."
        )
    )
    parser.add_argument(
        "--area",
        default=None,
        help=(
            "an area id you actually have -- prints the entity ids it expands to. "
            "Omit to only report registry counts and shapes."
        ),
    )
    return parser


def _missing_env_vars() -> list[str]:
    return [name for name in ("HA_URL", "HA_TOKEN") if not os.environ.get(name)]


def _print_counts(snapshot) -> None:
    own_area = sum(1 for entity in snapshot.entities if entity.area_id is not None)
    inherited = sum(
        1
        for entity in snapshot.entities
        if entity.area_id is None
        and entity.device_id is not None
        and snapshot.devices.get(entity.device_id) is not None
        and snapshot.devices[entity.device_id].area_id is not None
    )
    print(f"areas: {len(snapshot.areas)}")
    print(f"devices: {len(snapshot.devices)}")
    print(f"labels: {len(snapshot.labels)}")
    print(f"entities: {len(snapshot.entities)}")
    print(f"entities with their own area set: {own_area}")
    print(f"entities inheriting their area from a device: {inherited}")


async def _run(area: str | None) -> int:
    missing = _missing_env_vars()
    if missing:
        print(f"unset in the environment: {', '.join(missing)} -- see .env.example", file=sys.stderr)
        return 1

    ha_url = os.environ["HA_URL"]
    ha_token = os.environ["HA_TOKEN"]

    client = HaRegistryClient(ws_url_from_http(ha_url), ha_token)
    try:
        snapshot = await client.get_snapshot()
    except RegistryAuthError as exc:
        print(f"authentication failed: {exc}", file=sys.stderr)
        print("the registry commands were never reached -- check HA_TOKEN", file=sys.stderr)
        return 1
    except RegistryUnavailableError as exc:
        print(f"registry commands were rejected or unreachable: {exc}", file=sys.stderr)
        print(
            "this is the central assumption docs/runbooks/ha-registry-expansion.md "
            "and 03-RESEARCH.md's Pitfall 3 rest on -- read that runbook's "
            "'if the registry commands are rejected' section before going further",
            file=sys.stderr,
        )
        return 1

    _print_counts(snapshot)

    if area is not None:
        try:
            entity_ids = expand_area(snapshot, area)
        except UnknownRegistryTargetError as exc:
            print(f"area {area!r} not found: {exc}", file=sys.stderr)
            return 1
        print(f"area {area!r} expands to {len(entity_ids)} entities:")
        for entity_id in sorted(entity_ids):
            print(f"  {entity_id}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args.area))


if __name__ == "__main__":
    sys.exit(main())
