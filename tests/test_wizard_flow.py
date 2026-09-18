"""The wizard reaches a working assistant and proves the microphone and
speaker work before it lets the operator call setup finished (WEB-02,
WEB-03).

Wizard step state is server-side (Postgres), specifically so an abandoned
wizard resumes where it was left rather than forcing a re-run from step one
-- a stranger who closes the tab partway through a first-run setup, per
PROJECT.md's own requirement that a stranger can deploy and configure this
project from a browser, must not lose their progress. The second test is
this phase's inherited obligation from Phase 2 (STATE.md's handoff, named
in 03-CONTEXT.md's domain section): camera barge-in (VOICE-07) ships
disabled because no real echo-path calibration exists, and the wizard is
where the operator chose to close that gap. A wizard that lets "finish" be
clicked before the microphone/speaker test (Phase 2's
`/calibration/echo-path/run`) has actually run would ship a wizard that
*looks* like it closed VOICE-07 without actually running the one check that
does.
"""

from __future__ import annotations


def test_an_abandoned_wizard_resumes_at_the_first_unfinished_step():
    """Closing the wizard partway through and returning later must resume
    at the first step that was not yet completed, not restart from the
    beginning -- plan 03-09 fills this in."""
    raise AssertionError(
        "plan 03-09 fills this in (WEB-02: an abandoned wizard resumes at the first unfinished step)"
    )


def test_the_wizard_cannot_finish_before_the_microphone_and_speaker_test_runs():
    """The wizard's final "finish" step must be unreachable until the
    microphone/speaker test (Phase 2's echo-path calibration) has actually
    run and returned a result -- plan 03-10 fills this in."""
    raise AssertionError(
        "plan 03-10 fills this in (WEB-03: the wizard cannot finish before the mic/speaker test runs)"
    )
