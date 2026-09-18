"""The real `PolicyRepository`, backed by the three tables in
`spire_voice.db.models`.

Takes its `async_sessionmaker` as a constructor argument with no default,
matching `FfmpegSupervisor`'s and `CameraAudioSource`'s own
dependency-injection-over-subclassing convention -- this class never opens
its own engine or session factory, and a test drives it against whatever
sessionmaker it is handed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from spire_mcp.safety import Policy
from spire_voice.db.models import AuditRow, PolicyRuleRow, SafetyPolicyRow
from spire_voice.db.repository import PolicyRule

# `safety_policy` is a single-row table -- `id` is always this value, never
# generated, so `load_policy`/a future write path never has to discover it.
_SINGLETON_POLICY_ID = 1


class PostgresPolicyRepository:
    """`PolicyRepository`, implemented against a real Postgres.

    Structurally satisfies `spire_voice.db.repository.PolicyRepository`
    (a `typing.Protocol`) -- there is no base class to inherit from.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def load_policy(self) -> Policy:
        """Build the current `Policy` from `safety_policy` and
        `policy_rules` -- the sibling of `Policy.from_config`, fed by rows
        instead of a config block."""
        async with self._sessionmaker() as session:
            mode_row = await session.get(SafetyPolicyRow, _SINGLETON_POLICY_ID)
            mode = mode_row.mode if mode_row is not None else "allow_all_except_denylist"

            rules = (await session.execute(select(PolicyRuleRow))).scalars().all()
            deny_entities = [r.value for r in rules if r.kind == "deny_entity"]
            deny_patterns = [r.value for r in rules if r.kind == "deny_pattern"]
            allow_entities = [r.value for r in rules if r.kind == "allow_entity"]
            allow_patterns = [r.value for r in rules if r.kind == "allow_pattern"]

        return Policy.from_db_rows(
            mode=mode,
            deny_entities=deny_entities,
            deny_patterns=deny_patterns,
            allow_entities=allow_entities,
            allow_patterns=allow_patterns,
        )

    async def list_rules(self) -> list[PolicyRule]:
        async with self._sessionmaker() as session:
            rows = (await session.execute(select(PolicyRuleRow))).scalars().all()
            return [
                PolicyRule(
                    id=row.id,
                    kind=row.kind,
                    value=row.value,
                    note=row.note,
                    created_at=row.created_at,
                    created_by_user_id=row.created_by_user_id,
                )
                for row in rows
            ]

    async def record_audit(
        self, action: str, detail: dict, actor_user_id: int | None
    ) -> None:
        async with self._sessionmaker() as session:
            session.add(
                AuditRow(
                    at=datetime.now(timezone.utc),
                    actor_user_id=actor_user_id,
                    action=action,
                    detail=detail,
                )
            )
            await session.commit()
