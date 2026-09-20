"""Boot-time provider slot resolution (D-02, D-04, PROV-01).

`resolve_slot` is the one function `lifespan` calls per provider slot: read
the stored choice, fall back to a shipped default when nothing is stored
yet, build the client through the slot's own registry `builder`, and report
back a small, closed-set fact about what happened -- never a pair of loose
booleans (`spire_voice.plugins.manager.PluginState`'s own discipline,
applied here to a one-shot boot outcome rather than a supervised respawn
loop, because D-02 means there is no live provider swap to supervise).

Generic from the first line: nothing here names "speech to text" or
"xAI" -- the three slots plan 07-02 adds all resolve through this same
function, passed a different `builder`/`config`/`api_key` each time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from spire_voice.db.repository import ProviderSelectionRepository
from spire_voice.providers import batch_tts_adapter


class ProviderUnavailable(Exception):
    """Raised by a `builder` (a registry `build_*` function, or a small
    wrapper around one) when the selected provider cannot be constructed
    right now -- today, exclusively a missing credential (D-04). The
    message is carried byte for byte into `ProviderSlotStatus.reason`: it
    is what the webapp shows an admin, never paraphrased (the CMD-07/
    VOICE-02 lesson, restated once more for this phase's own screen)."""


@dataclass(frozen=True)
class ProviderSlotStatus:
    """One slot's boot-time outcome, read fresh by `routes/providers.py` at
    response-build time -- never a stored copy.

    `selected` is what the database row said at the moment this was
    computed (repository read, not stored on this dataclass past that
    instant); `active` is the name this process actually built a client
    from, or `None` when the slot is degraded (D-02's whole reason to
    exist: the two can diverge, and the divergence itself is what a
    "needs restart" badge reports). `state` is exactly `"running"` or
    `"degraded"` -- a closed two-member set, never a boolean pair.
    `reason` is `None` for a running slot and the builder's own message,
    unchanged, for a degraded one. `wrapped` mirrors the built client's own
    `wrapped` attribute when it declares one (plan 07-02's batch-to-
    streaming wrap, `is_batch` below) and is `False` for a client with no
    such attribute.
    """

    slot: str
    selected: str
    active: "str | None"
    state: str
    reason: "str | None"
    wrapped: bool


async def resolve_slot(
    slot: str,
    repo: ProviderSelectionRepository,
    default_name: str,
    builder: "Callable[[str, Any, str, dict], Any]",
    config: Any,
    api_key: str,
    *,
    is_batch: "Callable[[str], bool] | None" = None,
) -> "tuple[ProviderSlotStatus, Any | None]":
    """Read `slot`'s stored choice (or fall back to `default_name` when no
    row exists yet), build a client through `builder`, and return
    `(status, client)`.

    `builder` is expected to be one of `registry.build_stt`/`build_tts`/
    `build_brain` (or a thin wrapper around one) -- called as
    `builder(selected, config, api_key, options)`, where `options` is the
    stored selection row's own `options` dict (`{}` when no row exists yet
    or the row carries none). Every existing builder ignores it; the local
    language-model entry (07-04-PLAN.md Task 1) is the first reader -- a
    server URL saved through the route is what the next boot's factory
    call actually receives. A `ProviderUnavailable` it raises
    yields a degraded status and a `None` client; this boot continues
    (D-04: an admin locked out of the very screen that would fix the
    configuration is the worst available outcome). Any other exception
    (an unrecognized provider name, for instance) propagates uncaught and
    stops the boot -- running a provider nobody chose is worse than not
    starting (T-07-05).

    `is_batch` (plan 07-02, D-05, D-07, D-08), given, is called with the
    resolved name after a successful build; when it answers `True`, the
    built client is wrapped in the batch-to-streaming adapter
    (`providers/batch_tts_adapter.py`) before it is returned, and
    `wrapped` reports `True`. This is the one place that wrap ever
    happens -- never inside `registry.py`, never inside a provider
    module. Omitted (every caller that predates it, and every slot whose
    registry never marks an entry batch), `wrapped` falls back to reading
    the built client's own `wrapped` attribute, unchanged from before.
    """
    selection = await repo.get_selection(slot)
    selected = selection.provider_name if selection is not None else default_name
    options = selection.options if selection is not None else {}

    try:
        client = builder(selected, config, api_key, options)
    except ProviderUnavailable as exc:
        return (
            ProviderSlotStatus(
                slot=slot,
                selected=selected,
                active=None,
                state="degraded",
                reason=str(exc),
                wrapped=False,
            ),
            None,
        )

    if is_batch is not None and is_batch(selected):
        client = batch_tts_adapter.BatchTtsAdapter(client)
        wrapped = True
    else:
        wrapped = bool(getattr(client, "wrapped", False))

    return (
        ProviderSlotStatus(
            slot=slot,
            selected=selected,
            active=selected,
            state="running",
            reason=None,
            wrapped=wrapped,
        ),
        client,
    )
