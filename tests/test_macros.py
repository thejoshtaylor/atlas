"""Wave-0 scaffolds for macros (MACRO-01, MACRO-02, CMD-07) -- turned green
by plan 01.1-05.

All three tests are intentionally red until 01.1-05 lands `turn/macros.py`.
Plan 01.1-05 is the first plan in this phase that can run the bare full
suite green.
"""


def test_macro_actions_all_pass_allow_call():
    """Every macro action passes `allow_call` at fire time, exactly like a
    model-issued call -- a macro is a shortcut past the model, never past
    the safety boundary (Pattern 4, 01.1-PATTERNS.md)."""
    raise AssertionError("Wave 0 scaffold - turned green by plan 01.1-05")


def test_macro_hit_skips_brain():
    """A phrase matching a configured macro skips the tier race entirely --
    a macro is a latency mechanism, not only a convenience (MACRO-01)."""
    raise AssertionError("Wave 0 scaffold - turned green by plan 01.1-05")


def test_macro_partial_failure_speaks_a_reason():
    """A macro's precached confirmation must never play unless every one of
    its actions returned success; a failure speaks a synthesized reason
    naming which part failed (CMD-07)."""
    raise AssertionError("Wave 0 scaffold - turned green by plan 01.1-05")
