"""Speaker volume on the Pi, through `amixer` (ALSA).

Stdlib only. `amixer` always runs through `asyncio.create_subprocess_exec`
with an argument list, never through a shell. The level in a `sset` call is
an integer this module formats itself. The card and control names come from
`config.toml`, and `load_config` refuses a name that starts with "-", so
config text can never become an `amixer` option.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Awaitable, Callable, Sequence

from atlas_edge.protocol import Volume

logger = logging.getLogger("atlas_edge.volume")

# One `amixer` call must not stop the session for longer than this.
AMIXER_TIMEOUT_S = 2.0

_LEVEL_RE = re.compile(r"\[(\d{1,3})%\]")
_STDERR_CHARS = 200


class VolumeError(Exception):
    """An `amixer` call failed or its output has no level."""


def resolve_target(request: Volume, current: "int | None") -> int:
    """The level to set: the absolute `request.level`, or `current` plus or
    minus `request.step_percent`. The result is clamped into the request's
    limits and into 0..100."""
    if request.level is not None:
        base = request.level
    else:
        if current is None:
            raise VolumeError("a relative volume request needs the current level")
        step = request.step_percent or 0
        base = current + step if request.direction == "up" else current - step
    return max(max(0, request.min_percent), min(min(100, request.max_percent), base))


def parse_level(output: str) -> int:
    """The first `[N%]` in `amixer` output. Raises `VolumeError` when there
    is none, or when N is more than 100."""
    match = _LEVEL_RE.search(output)
    if match is None:
        raise VolumeError("amixer output has no volume level")
    level = int(match.group(1))
    if level > 100:
        raise VolumeError(f"amixer reported a level over 100: {level}")
    return level


async def run_amixer(args: Sequence[str], *, timeout_s: float = AMIXER_TIMEOUT_S) -> str:
    """Run `amixer` with `args` and return its stdout. Raises `VolumeError`
    when `amixer` is not installed, does not finish within `timeout_s`, or
    exits with a non-zero code."""
    try:
        process = await asyncio.create_subprocess_exec(
            "amixer", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise VolumeError("amixer is not installed on this device") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise VolumeError(f"amixer did not finish within {timeout_s:g} seconds") from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[:_STDERR_CHARS].strip()
        raise VolumeError(f"amixer exited with code {process.returncode}: {detail}")
    return stdout.decode("utf-8", errors="replace")


class VolumeControl:
    """Sets the level of one ALSA mixer control. `apply` is the `on_volume`
    callback the session calls for each `volume` message."""

    def __init__(
        self,
        card: str,
        control: str,
        *,
        run: Callable[[Sequence[str]], Awaitable[str]] = run_amixer,
    ) -> None:
        self._card = card
        self._control = control
        self._run = run

    async def apply(self, request: Volume) -> int:
        """Set the level the request asks for and return the level `amixer`
        reports after the change. A relative request reads the current
        level first."""
        current = None
        if request.level is None:
            current = parse_level(await self._run(["-c", self._card, "sget", self._control]))
        target = resolve_target(request, current)
        output = await self._run(["-c", self._card, "sset", self._control, f"{target}%"])
        level = parse_level(output)
        logger.info("speaker volume is now %d percent (card %s, control %s)", level, self._card, self._control)
        return level
