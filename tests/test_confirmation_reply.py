"""`handle_confirmation_reply` and `run_confirmation_round`: the reply to a
stored calendar readback (D-08, D-09, D-11).

Every account, calendar, and event name below is invented.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from atlas.turn.follow_up import FollowUpRequest
from atlas.turn.handoff import HandoffContext
from atlas.turn.pending_action import CANCELLED_REPLY, CONFIRMED_CREATE_REPLY, handle_confirmation_reply

import test_pending_action as tpa
from google_fakes import FakeGoogle
from pending_action_fakes import FakePendingActionRepository

_READBACK = "add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?"


async def _stored_create(pending_actions):
    return await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback=_READBACK,
        created_at=tpa._NOW,
        expires_at=tpa._NOW + timedelta(seconds=60),
    )


def _incoming(created) -> FollowUpRequest:
    return FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )


async def _setup(brain, *, pending_actions=None):
    pending_actions = pending_actions if pending_actions is not None else FakePendingActionRepository()
    created = await _stored_create(pending_actions)
    fake_google = FakeGoogle()
    tool_host = tpa._GoogleToolHost((tpa._home_account(),), google_client=fake_google.client)
    ctx = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=brain,
        now=tpa._NOW + timedelta(seconds=5),
    )
    return ctx, created, pending_actions, fake_google


@pytest.mark.parametrize("transcript", ["yeah", "Yeah.", "okay", "OK", "yeah okay", "uh yeah"])
async def test_a_short_agreement_reaches_the_confirmation_round_and_confirms(transcript):
    """R2-WR-02: "yeah" and "okay" are the most common spoken answers to a
    yes/no question. They go to the restricted round, not to the
    filler-only silence path."""
    brain = tpa._RecordingConfirmationBrain("confirm")
    ctx, created, pending_actions, fake_google = await _setup(brain)

    outcome = await handle_confirmation_reply(ctx, _incoming(created), transcript, timeout_s=5.0)

    assert len(brain.calls) == 1
    assert outcome.turn_outcome == "confirmed"
    assert outcome.reply_text == CONFIRMED_CREATE_REPLY
    assert len(fake_google.inserted) == 1
    assert (await pending_actions.get(created.id)).status == "executed"


@pytest.mark.parametrize("transcript", ["", "   ", "\n"])
async def test_an_empty_or_blank_transcript_is_silence(transcript):
    brain = tpa._RecordingConfirmationBrain("confirm")
    ctx, created, pending_actions, _ = await _setup(brain)

    outcome = await handle_confirmation_reply(ctx, _incoming(created), transcript, timeout_s=5.0)

    assert brain.calls == []
    assert outcome.turn_outcome == "follow_up_silence"
    assert outcome.reply_text == CANCELLED_REPLY
    assert (await pending_actions.get(created.id)).status == "expired"
