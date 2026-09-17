"""Wave-0 scaffolds for the tier race (VOICE-02) -- turned green by plan 01.1-04.

Both tests are intentionally red until 01.1-04 lands `turn/brain_race.py`.
Every plan in this phase scopes its own suite run to explicit files for
exactly this reason.
"""


def test_first_confident_tier_wins_and_losers_are_cancelled():
    """The first tier to return `confident=True` is spoken; every other
    in-flight tier call is cancelled at that moment (CONTEXT.md: "All tiers
    are dispatched concurrently... the losing calls are cancelled")."""
    raise AssertionError("Wave 0 scaffold - turned green by plan 01.1-04")


def test_a_one_entry_tier_list_still_resolves_through_the_list():
    """The companion invariant from the assumption-delta decision: a turn
    resolves through the tier list for every configured list length,
    including one, so a future single-model assumption goes red immediately."""
    raise AssertionError("Wave 0 scaffold - turned green by plan 01.1-04")
