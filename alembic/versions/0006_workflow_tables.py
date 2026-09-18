"""workflow tables

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-18

Creates `workflow_runs` and `workflow_steps`, the two tables FLOW-01
through FLOW-10 and SAFE-08 build on: a spoken or webapp-authored plan (a
run) made of ordered steps (`wait`, `call_service`, `speak`), each with its
own absolute, naive-UTC `due_at`, claimed one at a time by the poller
`workflow/scheduler.py` starts in `lifespan` (05-CONTEXT.md D-01 .. D-04).

This migration has no seed step, and deliberately looks nothing like
`0005_macro_tables.py` in that one respect: a run or a step is a
database-only concept from the moment this phase adds it (D-07, D-08) --
there was never an operator-editable `workflow_runs`/`workflow_steps`
equivalent in the configuration file for a migration to carry forward, the
way `0001`'s `safety:` block and `0005`'s `macros:` block each once were.
`spire_voice.config.WorkflowConfig` (added the same plan as this
migration) is a sibling, unrelated concern -- it declares the poller's own
*schedule* (how often it wakes, how many steps it drains per tick), never
a run or a step, so there is nothing of that shape for this migration to
read either. `upgrade()` below is table creation only: no environment
variable is read, no legacy key is rejected on a later boot, and no
operator's configuration file is opened at all. A reader checking that
this migration carries no house-specific data forward should find nothing
here to check.

Three column-shape decisions were confirmed at plan-review time, one-way
because `downgrade()` (real, below) drops both tables along with every
pending run in them once an operator has any:

1. `arguments` is a JSON column *per step*, not per run -- D-07 rejected a
   JSON column for per-step *state*: `kind`, `status`, `attempts`, `due_at`
   and `fired_at` are all real columns a constraint and a query can reach.
   Only the per-kind payload (a service call's domain/service/entity id, a
   wait's duration, or the words a `speak` step says) is JSON, the same
   split `macro_actions.arguments` already uses for its own per-action
   payload.
2. There is no `run.due_at`. A run's own schedule is its first step's
   `due_at`; a second copy would be a second source of truth about when
   the house acts, the same duplication both Phase 3 and Phase 4 spent a
   Critical finding on.
3. `summary` is stored, not derived. FLOW-04 and FLOW-06 match the
   operator's own words against it; deriving it at read time would let a
   later change to the composer silently change which run "the lights"
   now means, and the operator would learn about it by cancelling the
   wrong one.

`ix_workflow_steps_status_due_at` on `(status, due_at)` is the poller's own
hot predicate -- `claim_and_execute_next_due_step`'s `WHERE status =
'pending' AND due_at <= now()` reads through this index on every poll.
`uq_workflow_steps_run_id_position` on `(run_id, position)` is what the
claim predicate's own `NOT EXISTS`-on-earlier-pending-sibling check relies
on to keep one run's steps unambiguous by position -- see
`db/postgres.py::PostgresWorkflowRepository.claim_and_execute_next_due_step`
for where that predicate is actually written.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("origin", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="fk_workflow_runs_created_by_user_id",
        ),
    )
    op.create_table(
        "workflow_steps",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column("due_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("result_detail", sa.JSON(), nullable=True),
        sa.Column("fired_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["workflow_runs.id"],
            name="fk_workflow_steps_run_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_workflow_steps_status_due_at",
        "workflow_steps",
        ["status", "due_at"],
    )
    op.create_unique_constraint(
        "uq_workflow_steps_run_id_position",
        "workflow_steps",
        ["run_id", "position"],
    )


def downgrade() -> None:
    op.drop_table("workflow_steps")
    op.drop_table("workflow_runs")
