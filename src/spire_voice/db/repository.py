"""The policy repository protocol: what `lifespan` and the policy routes
need from storage, with no commitment to how storage answers it.

Follows `spire_voice.providers.base`'s `Protocol`-typed shape exactly --
`PolicyRepository` is a `typing.Protocol`, not an abstract base class, so
`PostgresPolicyRepository` (real) and `FakePolicyRepository`
(`tests/conftest.py`) both satisfy it structurally, the same
dependency-injection-over-subclassing convention `FfmpegSupervisor` and
`CameraAudioSource` already use for their own real dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from spire_mcp.safety import Policy


@dataclass(frozen=True)
class PolicyRule:
    """One row from `policy_rules`, carried as a plain value object rather
    than the ORM row itself -- a caller outside `src/spire_voice/db/` has
    no reason to hold a SQLAlchemy-mapped instance open past its session."""

    id: int
    kind: str
    value: str
    note: str | None
    created_at: datetime
    created_by_user_id: int | None


class PolicyRepository(Protocol):
    """What the safety policy's storage layer must answer.

    Three members only: `load_policy` builds the frozen `Policy`
    `mcp/spire_mcp/safety.py` enforces, `list_rules` is the raw rows the
    webapp's policy editor renders, and `record_audit` is the one write
    path plan 03-07's mode switch uses. Nothing here returns an ORM row --
    a `Policy` and a `PolicyRule` are both plain value objects.
    """

    async def load_policy(self) -> Policy:
        """Build the current `Policy` from the database's rows."""
        ...

    async def list_rules(self) -> Sequence[PolicyRule]:
        """Every `policy_rules` row, for the webapp's editor."""
        ...

    async def record_audit(
        self, action: str, detail: dict, actor_user_id: int | None
    ) -> None:
        """Write one `audit_log` row."""
        ...
