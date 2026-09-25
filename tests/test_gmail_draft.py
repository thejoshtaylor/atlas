"""Plan 09-09, Task 3: "reply to Dana that I'll be 15 minutes late" -- a
threaded draft, in the right account, written in the operator's style,
never sent (D-08, D-12, D-18, D-21, D-23).

Drives `atlas_mcp.google`'s own handlers directly (`handle_gmail_create_draft`,
`handle_gmail_draft_reply`) and `atlas.turn.email_handoff.handle_email_draft`
against `FakeGoogle`, `FakeGoogleAccountRepository`, and
`RecordingFakeBrain` -- the same "primitives in isolation" shape
`tests/test_last_email_list.py` already uses, extended end-to-end through
`run_turn` for the plan's own required test names.

Every token/address/name here is invented -- no real house appears.
"""

from __future__ import annotations

import base64
import email
import email.policy
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from google_fakes import FakeGoogle
from google_repo_fakes import FakeGoogleAccountRepository
from pending_action_fakes import FakePendingActionRepository

from atlas_mcp.google import handle_gmail_create_draft, handle_gmail_draft_reply
from atlas_mcp.google_boundary import AccountGrant
from atlas_mcp.safety import Denied

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.email_handoff import DRAFT_SAVED_CARRIER, handle_email_draft
from atlas.turn.email_memory import EmailListItem, EmailListMemory
from atlas.turn.handoff import Handoff, HandoffContext

from brain_fakes import RecordingFakeBrain


def _account(label: str, access_token: str) -> AccountGrant:
    return AccountGrant(
        label=label, email=f"{label}@example.com", is_default=(label == "work"),
        access_token=access_token, unreachable_reason=None, calendars=(),
    )


def _decode_raw(raw: str) -> "email.message.EmailMessage":
    padded = raw + "=" * (-len(raw) % 4)
    return email.message_from_bytes(base64.urlsafe_b64decode(padded), policy=email.policy.default)


# --- handle_gmail_create_draft (the code-only, header-reading half) --------


async def test_draft_references_original_message_id():
    fake = FakeGoogle()
    fake.add_gmail_metadata(
        "at-work",
        "m1",
        headers={
            "From": "Dana Example <dana@example.com>",
            "Subject": "Lunch tomorrow",
            "Message-Id": "<orig-1@example.com>",
            "References": "<thread-0@example.com>",
        },
        thread_id="thread-99",
    )
    accounts = (_account("work", "at-work"),)

    result = await handle_gmail_create_draft(
        accounts, fake.client, account="work", message_id="m1", body_text="sounds good, see you then."
    )

    [draft] = fake.drafts
    assert draft["thread_id"] == "thread-99"
    message = _decode_raw(draft["raw"])
    assert message["In-Reply-To"] == "<orig-1@example.com>"
    assert message["References"] == "<thread-0@example.com> <orig-1@example.com>"
    assert message["Subject"] == "Re: Lunch tomorrow"
    assert "From" not in message
    assert result["draft"]["to_address"] == "dana@example.com"


async def test_subject_already_carrying_re_is_not_doubled():
    fake = FakeGoogle()
    fake.add_gmail_metadata(
        "at-work",
        "m1",
        headers={
            "From": "Dana Example <dana@example.com>",
            "Subject": "Re: Lunch tomorrow",
            "Message-Id": "<orig-1@example.com>",
            "References": "",
        },
    )
    accounts = (_account("work", "at-work"),)

    await handle_gmail_create_draft(
        accounts, fake.client, account="work", message_id="m1", body_text="sounds good."
    )

    [draft] = fake.drafts
    message = _decode_raw(draft["raw"])
    assert message["Subject"] == "Re: Lunch tomorrow"


async def test_recipient_prefers_reply_to_over_from():
    fake = FakeGoogle()
    fake.add_gmail_metadata(
        "at-work",
        "m1",
        headers={
            "From": "Dana Example <dana@example.com>",
            "Reply-To": "Dana Support <support@example.com>",
            "Subject": "Lunch",
            "Message-Id": "<orig-1@example.com>",
            "References": "",
        },
    )
    accounts = (_account("work", "at-work"),)

    result = await handle_gmail_create_draft(
        accounts,
        fake.client,
        account="work",
        message_id="m1",
        # A body naming a different address changes nothing about the
        # recipient -- code, never model output, decides who this goes to.
        body_text="tell someone@else.example I said hi.",
    )

    assert result["draft"]["to_address"] == "support@example.com"
    [draft] = fake.drafts
    message = _decode_raw(draft["raw"])
    assert message["To"] == "Dana Support <support@example.com>"


async def test_recipient_falls_back_to_from_when_no_reply_to():
    fake = FakeGoogle()
    fake.add_gmail_metadata(
        "at-work",
        "m1",
        headers={
            "From": "Dana Example <dana@example.com>",
            "Subject": "Lunch",
            "Message-Id": "<orig-1@example.com>",
            "References": "",
        },
    )
    accounts = (_account("work", "at-work"),)

    result = await handle_gmail_create_draft(
        accounts, fake.client, account="work", message_id="m1", body_text="sounds good."
    )

    assert result["draft"]["to_address"] == "dana@example.com"


async def test_draft_uses_the_access_token_of_the_receiving_account():
    fake = FakeGoogle()
    fake.add_gmail_metadata(
        "at-home",
        "m1",
        headers={"From": "Dana <dana@example.com>", "Subject": "Lunch", "Message-Id": "<o1@example.com>"},
    )
    accounts = (_account("work", "at-work"), _account("home", "at-home"))

    await handle_gmail_create_draft(
        accounts, fake.client, account="home", message_id="m1", body_text="ok."
    )

    [draft] = fake.drafts
    assert draft["access_token"] == "at-home"


async def test_draft_is_multipart_alternative_with_signature():
    fake = FakeGoogle()
    fake.add_gmail_metadata(
        "at-work",
        "m1",
        headers={"From": "Dana <dana@example.com>", "Subject": "Lunch", "Message-Id": "<o1@example.com>"},
    )
    accounts = (_account("work", "at-work"),)

    await handle_gmail_create_draft(
        accounts,
        fake.client,
        account="work",
        message_id="m1",
        body_text="sounds good",
        signature_text="Alex",
        signature_html="<i>Alex</i>",
    )

    [draft] = fake.drafts
    message = _decode_raw(draft["raw"])
    assert message.is_multipart()
    assert message.get_content_type() == "multipart/alternative"
    plain_part, html_part = message.get_payload()
    assert plain_part.get_content_type() == "text/plain"
    assert plain_part.get_content().strip() == "sounds good\n\nAlex"
    assert html_part.get_content_type() == "text/html"
    assert "sounds good" in html_part.get_content()
    assert "<i>Alex</i>" in html_part.get_content()


async def test_header_value_with_a_line_break_is_refused_and_no_draft_created():
    fake = FakeGoogle()
    fake.add_gmail_metadata(
        "at-work",
        "m1",
        headers={
            "From": "Dana <dana@example.com>",
            "Subject": "Lunch\r\nBcc: evil@example.com",
            "Message-Id": "<o1@example.com>",
        },
    )
    accounts = (_account("work", "at-work"),)

    try:
        await handle_gmail_create_draft(
            accounts, fake.client, account="work", message_id="m1", body_text="ok"
        )
        raised = False
    except Denied:
        raised = True

    assert raised
    assert fake.drafts == []


async def test_unknown_account_is_refused():
    fake = FakeGoogle()
    accounts = ()

    try:
        await handle_gmail_create_draft(
            accounts, fake.client, account="nope", message_id="m1", body_text="ok"
        )
        raised = False
    except Denied:
        raised = True

    assert raised
    assert fake.drafts == []


# --- handle_gmail_draft_reply (the model-callable proposal half) -----------


async def test_gmail_draft_reply_makes_no_request_and_returns_a_handoff():
    result = await handle_gmail_draft_reply(position=1, instructions="say I'll be 15 minutes late")
    assert result["atlas_handoff"] == {
        "kind": "email_draft",
        "position": 1,
        "sender": None,
        "instructions": "say I'll be 15 minutes late",
    }


async def test_gmail_draft_reply_refuses_both_position_and_sender():
    try:
        await handle_gmail_draft_reply(position=1, sender="dana", instructions="ok")
        raised = False
    except Denied:
        raised = True
    assert raised


async def test_gmail_draft_reply_refuses_empty_instructions():
    try:
        await handle_gmail_draft_reply(position=1, instructions="   ")
        raised = False
    except Denied:
        raised = True
    assert raised


async def test_gmail_draft_reply_refuses_instructions_over_five_hundred_characters():
    try:
        await handle_gmail_draft_reply(position=1, instructions="x" * 501)
        raised = False
    except Denied:
        raised = True
    assert raised


# --- handle_email_draft: the quarantine round, no pending action -----------


_DANA = EmailListItem(
    position=1, account="work", message_id="m1", thread_id="m1",
    from_name="Dana Example", from_address="dana@example.com", subject="Lunch",
)


def _handoff(*, position=None, sender=None, instructions="say I'll be 15 minutes late") -> Handoff:
    return Handoff(
        kind="email_draft",
        payload={"position": position, "sender": sender, "instructions": instructions},
    )


class _StubToolHost:
    """Scripted `gmail_fetch_body`/`gmail_create_draft` answers, recording
    every call for the round's own `tools=None` and no-pending-action
    assertions below."""

    def __init__(self, *, fetch_body: dict, draft: dict) -> None:
        self._fetch_body = fetch_body
        self._draft = draft
        self.calls: "list[tuple[str, dict]]" = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, dict(arguments)))
        if name == "gmail_fetch_body":
            body = self._fetch_body
        elif name == "gmail_create_draft":
            body = self._draft
        else:
            raise AssertionError(f"unknown tool: {name}")
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(body))])


async def test_drafting_round_is_tools_none_and_no_pending_action_is_stored():
    memory = EmailListMemory()
    memory.store("camera", (_DANA,))
    tool_host = _StubToolHost(
        fetch_body={
            "body": "running behind, can we push to 1?",
            "truncated": False,
            "from_name": "Dana Example",
            "from_address": "dana@example.com",
            "subject": "Lunch",
        },
        draft={"draft": {"draft_id": "draft-1", "to_name": "Dana Example", "to_address": "dana@example.com", "account": "work"}},
    )
    brain = RecordingFakeBrain(
        replies=[BrainReply(text="GIST: running 15 minutes late.\nBODY: I'll be about 15 minutes late, sorry!")]
    )
    pending_actions = FakePendingActionRepository()
    ctx = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, email_memory=memory
    )

    outcome = await handle_email_draft(_handoff(position=1), ctx)

    [call] = brain.calls
    assert call.tools is None
    assert pending_actions._rows == {}
    assert outcome.follow_up is None
    assert outcome.turn_outcome == "email_draft"
    assert outcome.reply_text == DRAFT_SAVED_CARRIER.format(
        first_name="Dana", label="work", gist="running 15 minutes late."
    )


async def test_output_without_markers_saves_whole_text_and_speaks_first_sentence():
    memory = EmailListMemory()
    memory.store("camera", (_DANA,))
    tool_host = _StubToolHost(
        fetch_body={"body": "running behind.", "truncated": False, "from_name": "Dana", "from_address": "dana@example.com", "subject": "Lunch"},
        draft={"draft": {"draft_id": "draft-1", "to_name": "Dana", "to_address": "dana@example.com", "account": "work"}},
    )
    brain = RecordingFakeBrain(replies=[BrainReply(text="I'll be 15 minutes late. See you soon!")])
    ctx = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=brain, email_memory=memory
    )

    outcome = await handle_email_draft(_handoff(position=1), ctx)

    assert outcome.reply_text == DRAFT_SAVED_CARRIER.format(
        first_name="Dana", label="work", gist="I'll be 15 minutes late."
    )
    saved_body = tool_host.calls[-1][1]["body_text"]
    assert saved_body == "I'll be 15 minutes late. See you soon!"


async def test_a_failed_create_speaks_the_tools_own_error_text():
    memory = EmailListMemory()
    memory.store("camera", (_DANA,))
    tool_host = _StubToolHost(
        fetch_body={"body": "running behind.", "truncated": False, "from_name": "Dana", "from_address": "dana@example.com", "subject": "Lunch"},
        draft={},
    )

    class _FailingCreateHost(_StubToolHost):
        async def call_tool(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            if name == "gmail_fetch_body":
                return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(self._fetch_body))])
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text="i can't reach your work account right now")])

    failing_host = _FailingCreateHost(fetch_body=tool_host._fetch_body, draft={})
    brain = RecordingFakeBrain(replies=[BrainReply(text="GIST: late.\nBODY: running late.")])
    ctx = HandoffContext(
        source_name="camera", tool_host=failing_host, pending_actions=None, brain=brain, email_memory=memory
    )

    outcome = await handle_email_draft(_handoff(position=1), ctx)

    assert outcome.reply_text == "i can't reach your work account right now"
    assert outcome.turn_outcome == "email_draft_failed"


async def test_style_profile_and_samples_reach_the_drafting_round():
    memory = EmailListMemory()
    memory.store("camera", (_DANA,))
    tool_host = _StubToolHost(
        fetch_body={"body": "running behind.", "truncated": False, "from_name": "Dana", "from_address": "dana@example.com", "subject": "Lunch"},
        draft={"draft": {"draft_id": "draft-1", "to_name": "Dana", "to_address": "dana@example.com", "account": "work"}},
    )
    brain = RecordingFakeBrain(replies=[BrainReply(text="GIST: late.\nBODY: running late, sorry!")])
    style_repo = FakeGoogleAccountRepository()
    account = await style_repo.insert_account(
        label="work", email="work@example.com", refresh_token_ciphertext=b"x", key_version=1,
        granted_scopes="", refresh_token_expires_at=None, linked_by_user_id=None,
        linked_at=datetime.now(timezone.utc),
    )
    await style_repo.save_learned_style(
        account.id, profile="brief and warm.", samples=("thanks so much, talk soon!",),
        signature_html=None, signature_text=None, messages_scanned=3, learned_at=datetime.now(timezone.utc),
    )
    ctx = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=brain, email_memory=memory,
        style_repo=style_repo,
    )

    await handle_email_draft(_handoff(position=1), ctx)

    [call] = brain.calls
    prompt = call.messages[1]["content"]
    assert "brief and warm." in prompt
    assert "thanks so much, talk soon!" in prompt


# --- end to end through run_turn --------------------------------------------


class _GmailToolHost:
    def __init__(self, accounts, client) -> None:
        self._accounts = accounts
        self._client = client
        self.calls: "list[tuple[str, dict]]" = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, dict(arguments)))
        try:
            if name == "gmail_draft_reply":
                result = await handle_gmail_draft_reply(**arguments)
            elif name == "gmail_fetch_body":
                from atlas_mcp.google import handle_gmail_fetch_body

                result = await handle_gmail_fetch_body(self._accounts, self._client, **arguments)
            elif name == "gmail_create_draft":
                result = await handle_gmail_create_draft(self._accounts, self._client, **arguments)
            else:
                raise AssertionError(f"unknown tool: {name}")
        except Denied as exc:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text=str(exc))])
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(result))])


async def test_reply_to_dana_end_to_end_is_a_threaded_never_sent_draft(fake_audio_source, fake_stt, fake_tts):
    accounts = (_account("work", "at-work"),)
    fake = FakeGoogle()
    fake.add_gmail_full(
        "at-work", "m1", headers={"From": "Dana Example <dana@example.com>", "Subject": "Lunch"},
        text="running behind, can we push to one?",
    )
    fake.add_gmail_metadata(
        "at-work", "m1",
        headers={
            "From": "Dana Example <dana@example.com>", "Subject": "Lunch",
            "Message-Id": "<orig-1@example.com>", "References": "",
        },
        thread_id="thread-1",
    )
    memory = EmailListMemory()
    memory.store("camera", (_DANA,))

    tool_host = _GmailToolHost(accounts, fake.client)
    top_brain = RecordingFakeBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="gmail_draft_reply",
                        arguments={"position": 1, "instructions": "say I'll be 15 minutes late"},
                    )
                ]
            )
        ]
    )
    drafting_brain = RecordingFakeBrain(
        replies=[BrainReply(text="GIST: running 15 minutes late.\nBODY: I'll be about 15 minutes late, sorry!")]
    )
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=drafting_brain, email_memory=memory
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="reply to Dana that I'll be 15 minutes late")])
    tts = fake_tts(chunks=[b"\x01"])
    timings = TurnTimings()

    await run_turn(
        source, stt, top_brain, tts, tool_host,
        tools_schema=[], system_prompt="you manage email", max_tool_rounds=3, timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == [
        "draft to Dana saved in work: running 15 minutes late."
    ]
    assert timings.turn_outcome == "email_draft"
    [draft] = fake.drafts
    assert draft["thread_id"] == "thread-1"
    message = _decode_raw(draft["raw"])
    assert message["In-Reply-To"] == "<orig-1@example.com>"
    assert message["Subject"] == "Re: Lunch"
