"""Plan 09-08 Task 2: "any new email?" answered per D-14/D-15 across every
linked account -- a summary each for one or two unread messages, the
count and senders grouped by account for more, and bodies only ever
reach a quarantine round (`turn/quarantine.py`), never the tool round.

Drives the real `run_turn` through a fake tool host that calls the
child's own Gmail handlers in-process, following
`tests/test_pending_action.py::_GoogleToolHost`'s pattern. Every account,
address, and name below is invented -- no real house appears here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from atlas_mcp.google import handle_gmail_fetch_body, handle_gmail_list_unread, handle_gmail_search
from atlas_mcp.google_boundary import AccountGrant
from atlas_mcp.safety import Denied

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.email_memory import EmailListMemory
from atlas.turn.handoff import HandoffContext

from brain_fakes import RecordingFakeBrain
from google_fakes import FakeGoogle


def _account(label: str, access_token: str | None, *, unreachable_reason: str | None = None) -> AccountGrant:
    return AccountGrant(
        label=label,
        email=f"{label}@example.com",
        is_default=(label == "work"),
        access_token=access_token,
        unreachable_reason=unreachable_reason,
        calendars=(),
    )


class _GmailToolHost:
    """Calls straight into `atlas_mcp.google`'s own handler functions --
    the same functional-double shape `tests/test_pending_action.py::_GoogleToolHost`
    already establishes."""

    def __init__(self, accounts, client) -> None:
        self._accounts = accounts
        self._client = client
        self.calls: "list[tuple[str, dict]]" = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, dict(arguments)))
        try:
            if name == "gmail_list_unread":
                result = await handle_gmail_list_unread(self._accounts, self._client, **arguments)
            elif name == "gmail_search":
                result = await handle_gmail_search(self._accounts, self._client, **arguments)
            elif name == "gmail_fetch_body":
                result = await handle_gmail_fetch_body(self._accounts, self._client, **arguments)
            else:
                raise AssertionError(f"unknown tool: {name}")
        except Denied as exc:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text=str(exc))])
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(result))])


class _ContentAwareBrain:
    """A quarantine-round-only double: returns whichever scripted reply's
    key is a substring of the outgoing user content, so two summarization
    tasks running concurrently (`asyncio.gather` in
    `email_handoff.handle_email_list`) each get their own reply
    regardless of which one's `chat()` call happens to land first --
    `RecordingFakeBrain`'s own strict call-order script cannot make that
    guarantee against real concurrency."""

    def __init__(self, replies_by_substring: "dict[str, str]") -> None:
        self._replies = dict(replies_by_substring)
        self.calls: "list[tuple[list[dict], list[dict] | None]]" = []

    async def chat(self, messages, tools=None) -> BrainReply:
        self.calls.append((list(messages), tools))
        content = messages[-1]["content"]
        for substring, text in self._replies.items():
            if substring in content:
                return BrainReply(text=text)
        raise AssertionError(f"no scripted reply matches content: {content!r}")


async def _run(fake_audio_source, fake_stt, fake_tts, *, brain, tool_host, handoff_context, transcript):
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text=transcript)])
    tts = fake_tts(chunks=[b"\x01"])
    timings = TurnTimings()
    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage email",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )
    return tts, timings


async def test_more_than_two_groups_by_sender(fake_audio_source, fake_stt, fake_tts):
    accounts = (_account("work", "at-work"), _account("home", "at-home"))
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["w1", "w2", "w3", "w4"])
    fake.add_gmail_metadata("at-work", "w1", headers={"From": "GitHub <notify@example.com>"}, internal_date="6000")
    fake.add_gmail_metadata("at-work", "w2", headers={"From": "GitHub <notify@example.com>"}, internal_date="5000")
    fake.add_gmail_metadata("at-work", "w3", headers={"From": "GitHub <notify@example.com>"}, internal_date="4000")
    fake.add_gmail_metadata(
        "at-work", "w4", headers={"From": "Dana Example <dana@example.com>"}, internal_date="1000"
    )
    fake.add_gmail_messages("at-home", ["h1"])
    fake.add_gmail_metadata("at-home", "h1", headers={"From": "Lee Example <lee@example.com>"}, internal_date="3000")

    tool_host = _GmailToolHost(accounts, fake.client)
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[ToolCall(name="gmail_list_unread", arguments={})])])
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=None,
        brain=brain,
        email_memory=EmailListMemory(),
    )

    tts, timings = await _run(
        fake_audio_source,
        fake_stt,
        fake_tts,
        brain=brain,
        tool_host=tool_host,
        handoff_context=handoff_context,
        transcript="any new email?",
    )

    assert tts.received_text == [
        "you have 5 new emails. work: 3 from GitHub, 1 from Dana Example. home: 1 from Lee Example."
    ]
    assert not any(name == "gmail_fetch_body" for name, _ in tool_host.calls)
    assert not any(r.url.params.get("format") == "full" for r in fake.requests)
    assert brain.call_count == 1
    assert timings.turn_outcome == "email_list"


async def test_two_unread_summarizes_each_through_quarantine(fake_audio_source, fake_stt, fake_tts):
    accounts = (_account("work", "at-work"), _account("home", "at-home"))
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["w1"])
    fake.add_gmail_metadata(
        "at-work", "w1", headers={"From": "Dana Example <dana@example.com>"}, internal_date="2000"
    )
    fake.add_gmail_full("at-work", "w1", headers={"From": "Dana Example <dana@example.com>"}, text="Body one text.")
    fake.add_gmail_messages("at-home", ["h1"])
    fake.add_gmail_metadata(
        "at-home", "h1", headers={"From": "Lee Example <lee@example.com>"}, internal_date="1000"
    )
    fake.add_gmail_full("at-home", "h1", headers={"From": "Lee Example <lee@example.com>"}, text="Body two text.")

    tool_host = _GmailToolHost(accounts, fake.client)
    top_brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[ToolCall(name="gmail_list_unread", arguments={})])])
    quarantine_brain = _ContentAwareBrain({"Body one text.": "summary one", "Body two text.": "summary two"})
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=None,
        brain=quarantine_brain,
        email_memory=EmailListMemory(),
    )

    tts, timings = await _run(
        fake_audio_source,
        fake_stt,
        fake_tts,
        brain=top_brain,
        tool_host=tool_host,
        handoff_context=handoff_context,
        transcript="any new email?",
    )

    assert tts.received_text == ["work: Dana Example: summary one. home: Lee Example: summary two."]
    fetch_calls = [args for name, args in tool_host.calls if name == "gmail_fetch_body"]
    assert len(fetch_calls) == 2
    for call in quarantine_brain.calls:
        assert call[1] is None


async def test_zero_unread_speaks_no_new_email(fake_audio_source, fake_stt, fake_tts):
    accounts = (_account("work", "at-work"),)
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", [])

    tool_host = _GmailToolHost(accounts, fake.client)
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[ToolCall(name="gmail_list_unread", arguments={})])])
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=brain, email_memory=EmailListMemory()
    )

    tts, _timings = await _run(
        fake_audio_source,
        fake_stt,
        fake_tts,
        brain=brain,
        tool_host=tool_host,
        handoff_context=handoff_context,
        transcript="any new email?",
    )

    assert tts.received_text == ["no new email."]


async def test_search_with_no_results_speaks_the_search_fallback(fake_audio_source, fake_stt, fake_tts):
    accounts = (_account("work", "at-work"),)
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", [])

    tool_host = _GmailToolHost(accounts, fake.client)
    brain = RecordingFakeBrain(
        replies=[
            BrainReply(
                tool_calls=[ToolCall(name="gmail_search", arguments={"query": "from:nobody"})]
            )
        ]
    )
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=brain, email_memory=EmailListMemory()
    )

    tts, _timings = await _run(
        fake_audio_source,
        fake_stt,
        fake_tts,
        brain=brain,
        tool_host=tool_host,
        handoff_context=handoff_context,
        transcript="find email from nobody",
    )

    assert tts.received_text == ["i didn't find any email matching that."]


async def test_search_with_more_than_two_groups_the_same_way(fake_audio_source, fake_stt, fake_tts):
    accounts = (_account("work", "at-work"),)
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["w1", "w2", "w3"])
    for i in (1, 2, 3):
        fake.add_gmail_metadata(
            "at-work", f"w{i}", headers={"From": "Dana Example <dana@example.com>"}, internal_date=str(4000 - i)
        )

    tool_host = _GmailToolHost(accounts, fake.client)
    brain = RecordingFakeBrain(
        replies=[BrainReply(tool_calls=[ToolCall(name="gmail_search", arguments={"query": "from:dana"})])]
    )
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=brain, email_memory=EmailListMemory()
    )

    tts, _timings = await _run(
        fake_audio_source,
        fake_stt,
        fake_tts,
        brain=brain,
        tool_host=tool_host,
        handoff_context=handoff_context,
        transcript="find email from dana",
    )

    assert tts.received_text == ["i found 3 emails. work: 3 from Dana Example."]


async def test_unreachable_account_note_appended_to_the_zero_unread_reply(fake_audio_source, fake_stt, fake_tts):
    accounts = (_account("work", "at-work"), _account("home", None, unreachable_reason="needs_relink"))
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", [])

    tool_host = _GmailToolHost(accounts, fake.client)
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[ToolCall(name="gmail_list_unread", arguments={})])])
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=brain, email_memory=EmailListMemory()
    )

    tts, _timings = await _run(
        fake_audio_source,
        fake_stt,
        fake_tts,
        brain=brain,
        tool_host=tool_host,
        handoff_context=handoff_context,
        transcript="any new email?",
    )

    assert tts.received_text == ["no new email. i can't reach your home account right now."]


async def test_spoken_list_is_stored_in_memory_in_spoken_order(fake_audio_source, fake_stt, fake_tts):
    accounts = (_account("work", "at-work"),)
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["w1", "w2"])
    fake.add_gmail_metadata(
        "at-work", "w1", headers={"From": "Dana Example <dana@example.com>"}, internal_date="2000"
    )
    fake.add_gmail_full("at-work", "w1", headers={"From": "Dana Example <dana@example.com>"}, text="Body one.")
    fake.add_gmail_metadata(
        "at-work", "w2", headers={"From": "Lee Example <lee@example.com>"}, internal_date="1000"
    )
    fake.add_gmail_full("at-work", "w2", headers={"From": "Lee Example <lee@example.com>"}, text="Body two.")

    tool_host = _GmailToolHost(accounts, fake.client)
    top_brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[ToolCall(name="gmail_list_unread", arguments={})])])
    quarantine_brain = _ContentAwareBrain({"Body one.": "summary one", "Body two.": "summary two"})
    memory = EmailListMemory()
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=quarantine_brain, email_memory=memory
    )

    await _run(
        fake_audio_source,
        fake_stt,
        fake_tts,
        brain=top_brain,
        tool_host=tool_host,
        handoff_context=handoff_context,
        transcript="any new email?",
    )

    stored = memory.get("camera")
    assert stored is not None
    assert [(item.position, item.message_id) for item in stored] == [(1, "w1"), (2, "w2")]
