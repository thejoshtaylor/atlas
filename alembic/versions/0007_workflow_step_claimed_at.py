"""workflow step claimed_at

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-18

Adds `workflow_steps.claimed_at` -- the column CR-01's code-review fix
(05-REVIEW.md) needs to make a claimed-but-not-yet-terminal step a durable,
recoverable fact rather than a state indistinguishable from "never
attempted."

Before this migration, `claim_and_execute_next_due_step` claimed a due
step's row lock and ran its side effect (for `call_service`, a real HTTP
call to Home Assistant) *inside* the same transaction as the eventual
terminal write. A process killed between the side effect landing and that
transaction's own `commit()` rolled the whole attempt back: `status` was
still `"pending"`, `attempts` still `0`, byte-identical to a step never
claimed at all. The next poll reclaimed and re-executed it -- silently, for
a `call_service` step whose second invocation is not always safe to make
(this project's own prior incident: a single gesture cutting power to Home
Assistant, Frigate, and the recognizer together).

The fix is a two-phase claim: transaction one now marks a step `"claimed"`
(a new step-level status this migration does not need a `CHECK` constraint
for -- `status` has always been a plain `Text` column, `db/models.py`'s own
docstring names the full closed set in prose, not a database-enforced
enum) and commits, releasing the row lock *before* the side effect runs.
Transaction two, opened only after the side effect returns, re-locks the
same row by id and writes its terminal (or retried) state. `claimed_at` is
what makes a step found `"claimed"` on a later poll -- because the process
that claimed it never reached transaction two, not because it is still
genuinely in flight -- a decidable fact rather than a guess: stale once
`claimed_at` is older than `WorkflowConfig.claim_recovery_after_s`, and
only then eligible for `db/postgres.py`'s own recovery-claim query.

Nullable, with no `server_default` and no backfill: every existing step row
predates this column's own meaning (a step claimed under the old,
single-transaction design was never left `"pending"` after a crash -- the
whole point of this fix is that a *future* claim will now say so
durably), so `NULL` here means exactly what it should for every row that
exists before this migration ever runs: "never claimed under the two-phase
design," not "claimed, timestamp unknown."
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "workflow_steps",
        sa.Column("claimed_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("workflow_steps", "claimed_at")
