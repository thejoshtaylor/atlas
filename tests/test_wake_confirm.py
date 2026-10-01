"""`WakeConfirmation`: the server's own one-time confirmation of a wake
(Phase 15, D-05). The rule is `strip_wake_phrase`: a bare phrase confirms,
a lead word alone or a television sentence does not."""

from __future__ import annotations

import pytest

from atlas.turn.wake_confirm import WakeConfirmation

PHRASE = "hey atlas"


class _Recorder:
    def __init__(self) -> None:
        self.calls = 0

    async def emit(self) -> None:
        self.calls += 1


@pytest.mark.parametrize("text", ["hey atlas", "Hey Atlas, what time is it?", "hey atlas what time"])
async def test_a_transcript_that_opens_with_the_wake_phrase_confirms(text) -> None:
    recorder = _Recorder()
    confirmation = WakeConfirmation(PHRASE, recorder.emit)

    await confirmation.observe(text)

    assert recorder.calls == 1
    assert confirmation.confirmed is True


@pytest.mark.parametrize("text", ["hey", "You always say AM in the morning.", "", "turn on the lights"])
async def test_a_transcript_that_does_not_open_with_the_phrase_does_not_confirm(text) -> None:
    recorder = _Recorder()
    confirmation = WakeConfirmation(PHRASE, recorder.emit)

    await confirmation.observe(text)

    assert recorder.calls == 0
    assert confirmation.confirmed is False


async def test_a_second_confirming_text_does_not_emit_again() -> None:
    recorder = _Recorder()
    confirmation = WakeConfirmation(PHRASE, recorder.emit)

    await confirmation.observe("hey")
    await confirmation.observe("hey atlas")
    await confirmation.observe("hey atlas what time")
    await confirmation.observe("hey atlas what time is it")

    assert recorder.calls == 1


async def test_a_confirmation_stays_confirmed_through_a_text_that_no_longer_matches() -> None:
    recorder = _Recorder()
    confirmation = WakeConfirmation(PHRASE, recorder.emit)

    await confirmation.observe("hey atlas")
    await confirmation.observe("turn on the lights")

    assert recorder.calls == 1
    assert confirmation.confirmed is True
