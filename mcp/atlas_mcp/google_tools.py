"""The one contract file both processes import -- the backend
(`src/atlas/google/`) and the `atlas_mcp.google` child alike -- so no
later plan in Phase 9 has to guess a tool name, an env key, or which
scopes the OAuth consent needs (09-01-PLAN.md's own action text).

This module declares no behavior of its own. `MODEL_TOOL_NAMES` are tools
a language-model round may call; `CODE_ONLY_TOOL_NAMES` are actions this
codebase's own turn controller invokes directly, never offered to a model
round, because they are only ever reached after a spoken confirmation
(D-06 .. D-09) whose exact arguments code already holds. Later plans in
this phase implement the tools these names promise -- this plan implements
`calendar_list_events` only (Task 1/2) and states the rest of the contract
up front.
"""

from __future__ import annotations

# The stdio child module name -- `plugins.host.module_for_stdio_args`'s
# own `["-m", "<module>"]` shape names this exact string, and
# `PluginManager.custom_env_builders` (`src/atlas/plugins/manager.py`) is
# keyed by it.
GOOGLE_PLUGIN_MODULE = "atlas_mcp.google"

# The one environment key `GoogleEnvBuilder` writes and `atlas_mcp.google`'s
# own `_startup()` reads -- never a per-account key, so the env shape stays
# fixed regardless of how many accounts are linked.
GOOGLE_ACCOUNTS_ENV = "GOOGLE_ACCOUNTS_JSON"

# Reserved keys inside a tool result's own JSON shape -- named here so a
# later plan's turn-controller wiring and this plan's own child agree on
# the spelling without either importing the other's module.
HANDOFF_KEY = "atlas_handoff"
UNREACHABLE_KEY = "unreachable_accounts"

# GOOG-11/D-13: calendar and email tools never use the turn controller's
# "done" shortcut (`_DONE_SHORTCUT_TOOLS` in `src/atlas/turn/controller.py`
# names only `ha_call_service`) -- every tool name below must stay disjoint
# from that set, proved directly by a later plan's own
# `tests/test_done_shortcut_exclusion.py`.
MODEL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "calendar_list_events",
        "calendar_propose_event",
        "calendar_propose_delete",
        "gmail_list_unread",
        "gmail_search",
        "gmail_read",
        "gmail_draft_reply",
    }
)

# Actions only code invokes directly -- after a spoken confirmation whose
# exact arguments code already holds (D-08) -- never offered to a model
# round's own tool schema.
CODE_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "calendar_insert_event",
        "calendar_delete_event",
        "gmail_fetch_body",
        "gmail_create_draft",
    }
)

ALL_TOOL_NAMES: frozenset[str] = MODEL_TOOL_NAMES | CODE_ONLY_TOOL_NAMES

# A-CR-02 fix: the only tools a proposal-restricted turn
# (`turn/controller.py`'s own `restrict_tools_to_proposals`) may call that
# lead to a change. Both members only ever build a fresh `pending_action`
# handoff (D-08) -- neither one executes anything directly -- so any change
# the operator describes in that no-wake-word window ("yes, but make it 4")
# can only ever become a new proposal with its own readback and confirm,
# never an action that runs with no confirmation step at all.
CALENDAR_PROPOSAL_TOOL_NAMES: frozenset[str] = frozenset(
    {"calendar_propose_event", "calendar_propose_delete"}
)

# R2-WR-05: every tool a proposal-restricted turn may call -- the two
# proposal tools, plus the read-only `calendar_list_events`. A delete
# proposal needs an event id, and the continuation messages carry no
# earlier tool results. Without the list tool, "no, the one on tuesday"
# could never become a new delete proposal. The list tool changes nothing
# in Google Calendar, and D-18 restricts only email bodies.
PROPOSAL_TURN_TOOL_NAMES: frozenset[str] = CALENDAR_PROPOSAL_TOOL_NAMES | frozenset({"calendar_list_events"})

# D-03: one consent per account covers both the Calendar and Gmail scopes
# this whole phase needs -- named here so the OAuth authorize-url builder
# (a later plan) and any documentation both read from one list.
REQUIRED_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.settings.basic",
)
