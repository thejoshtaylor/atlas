"""Plan 09-08 Task 3: a later wake refers to the last spoken email list by
position or by sender, resolved entirely by code (`EmailListMemory`) --
never a second Gmail search and never a model guess (D-16, GOOG-08).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from atlas_mcp.google import handle_gmail_fetch_body, handle_gmail_read
from atlas_mcp.google_boundary import AccountGrant
from atlas_mcp.safety import Denied

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.email_handoff import VERBATIM_READ_CAP, handle_email_read
from atlas.turn.email_memory import EmailListItem, EmailListMemory
from atlas.turn.handoff import Handoff, HandoffContext

from brain_fakes import RecordingFakeBrain
from google_fakes import FakeGoogle

_DANA = EmailListItem(
    position=1,
    account="work",
    message_id="m1",
    thread_id="m1",
    from_name="Dana Example",
    from_address="dana@example.com",
    subject="Lunch",
)
_LEE = EmailListItem(
    position=2,
    account="home",
    message_id="m2",
    thread_id="m2",
    from_name="Lee Example",
    from_address="lee@example.com",
    subject="Weekend",
)
_ALEX = EmailListItem(
    position=3,
    account="work",
    message_id="m3",
    thread_id="m3",
    from_name="Alex Example",
    from_address="alex@example.com",
    subject="Invoice",
)


class _StubToolHost:
    """A tool host with one scripted `gmail_fetch_body` answer -- either a
    real result or a raised `Denied`, mirroring
    `_GmailToolHost.call_tool`'s own `Denied`-to-error-shaped-result
    translation."""

    def __init__(self, *, body: "dict | None" = None, denied: "Denied | None" = None) -> None:
        self._body = body
        self._denied = denied
        self.calls: "list[tuple[str, dict]]" = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, dict(arguments)))
        if self._denied is not None:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text=str(self._denied))])
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(self._body))])


class _AssertNeverCalledBrain:
    async def chat(self, messages, tools=None):
        raise AssertionError("brain.chat must not be called for a word-for-word read")


def _handoff(*, position=None, sender=None, word_for_word=False) -> Handoff:
    return Handoff(
        kind="email_read",
        payload={"position": position, "sender": sender, "word_for_word": word_for_word},
    )


def _memory_with(*items: EmailListItem, clock=None) -> EmailListMemory:
    memory = EmailListMemory(clock=clock) if clock is not None else EmailListMemory()
    memory.store("camera", items)
    return memory


# --- position/sender resolution, direct calls ------------------------------


async def test_position_resolves_the_exact_stored_item():
    memory = _memory_with(_DANA, _LEE, _ALEX)
    tool_host = _StubToolHost(
        body={"body": "see you then.", "truncated": False, "from_name": "Lee Example", "from_address": "lee@example.com", "subject": "Weekend"}
    )
    brain = RecordingFakeBrain(replies=[BrainReply(text="a short summary.")])
    ctx = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=brain, email_memory=memory
    )

    outcome = await handle_email_read(_handoff(position=2), ctx)

    assert tool_host.calls == [("gmail_fetch_body", {"account": "home", "message_id": "m2"})]
    assert outcome.reply_text == "home: Lee Example: a short summary."
    assert outcome.turn_outcome == "email_read"


async def test_sender_resolves_to_the_lowest_position_match():
    dana_again = EmailListItem(
        position=4, account="home", message_id="m4", thread_id="m4",
        from_name="Dana Example", from_address="dana@work-example.com", subject="Follow-up",
    )
    memory = _memory_with(_DANA, _LEE, _ALEX, dana_again)
    tool_host = _StubToolHost(
        body={"body": "lunch works.", "truncated": False, "from_name": "Dana Example", "from_address": "dana@example.com", "subject": "Lunch"}
    )
    brain = RecordingFakeBrain(replies=[BrainReply(text="lunch confirmed.")])
    ctx = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=brain, email_memory=memory
    )

    outcome = await handle_email_read(_handoff(sender="dana"), ctx)

    assert tool_host.calls == [("gmail_fetch_body", {"account": "work", "message_id": "m1"})]
    assert outcome.reply_text == "work: Dana Example: lunch confirmed."


async def test_unknown_sender_speaks_the_no_match_fallback():
    memory = _memory_with(_DANA, _LEE)
    ctx = HandoffContext(
        source_name="camera", tool_host=_StubToolHost(), pending_actions=None, brain=None, email_memory=memory
    )

    outcome = await handle_email_read(_handoff(sender="quinn"), ctx)

    assert outcome.reply_text == "i don't see an email from quinn in the last list."
    assert outcome.turn_outcome == "email_read"


async def test_position_past_the_end_names_the_real_count():
    memory = _memory_with(_DANA, _LEE, _ALEX)
    ctx = HandoffContext(
        source_name="camera", tool_host=_StubToolHost(), pending_actions=None, brain=None, email_memory=memory
    )

    outcome = await handle_email_read(_handoff(position=7), ctx)

    assert outcome.reply_text == "there were only 3 in the last list."
    assert outcome.turn_outcome == "email_read"


async def test_no_list_at_all_asks_the_operator_to_ask_whats_new():
    memory = EmailListMemory()
    ctx = HandoffContext(
        source_name="camera", tool_host=_StubToolHost(), pending_actions=None, brain=None, email_memory=memory
    )

    outcome = await handle_email_read(_handoff(position=1), ctx)

    assert outcome.reply_text == "i don't have a recent email list -- ask me what's new first."


async def test_a_list_older_than_ten_minutes_is_treated_as_no_list():
    now = [0.0]
    memory = _memory_with(_DANA, clock=lambda: now[0])
    now[0] = 601.0
    ctx = HandoffContext(
        source_name="camera", tool_host=_StubToolHost(), pending_actions=None, brain=None, email_memory=memory
    )

    outcome = await handle_email_read(_handoff(position=1), ctx)

    assert outcome.reply_text == "i don't have a recent email list -- ask me what's new first."


async def test_another_sources_list_is_never_used():
    memory = EmailListMemory()
    memory.store("browser", (_DANA,))
    ctx = HandoffContext(
        source_name="camera", tool_host=_StubToolHost(), pending_actions=None, brain=None, email_memory=memory
    )

    outcome = await handle_email_read(_handoff(position=1), ctx)

    assert outcome.reply_text == "i don't have a recent email list -- ask me what's new first."


async def test_empty_cleaned_body_speaks_the_nothing_to_read_fallback():
    memory = _memory_with(_DANA)
    tool_host = _StubToolHost(
        body={"body": "", "truncated": False, "from_name": "Dana Example", "from_address": "dana@example.com", "subject": "Lunch"}
    )
    ctx = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=None, email_memory=memory
    )

    outcome = await handle_email_read(_handoff(position=1), ctx)

    assert outcome.reply_text == "that email has no text i can read."
    assert outcome.turn_outcome == "email_read"


async def test_fetch_failure_speaks_the_tools_own_error_text_and_fails_the_turn():
    memory = _memory_with(_DANA)
    tool_host = _StubToolHost(denied=Denied("i can't reach your work account right now"))
    ctx = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=None, email_memory=memory
    )

    outcome = await handle_email_read(_handoff(position=1), ctx)

    assert outcome.reply_text == "i can't reach your work account right now"
    assert outcome.turn_outcome == "email_read_failed"


async def test_word_for_word_reads_the_cleaned_text_with_no_model_call():
    memory = _memory_with(_DANA)
    tool_host = _StubToolHost(
        body={"body": "let's do lunch friday.", "truncated": False, "from_name": "Dana Example", "from_address": "dana@example.com", "subject": "Lunch"}
    )
    ctx = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=None,
        brain=_AssertNeverCalledBrain(),
        email_memory=memory,
    )

    outcome = await handle_email_read(_handoff(position=1, word_for_word=True), ctx)

    assert outcome.reply_text == "Dana Example wrote: let's do lunch friday."
    assert outcome.turn_outcome == "email_read"


async def test_word_for_word_cuts_at_the_cap_and_says_so():
    long_text = "a" * (VERBATIM_READ_CAP + 200)
    memory = _memory_with(_DANA)
    tool_host = _StubToolHost(
        body={"body": long_text, "truncated": False, "from_name": "Dana Example", "from_address": "dana@example.com", "subject": "Lunch"}
    )
    ctx = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=None,
        brain=_AssertNeverCalledBrain(),
        email_memory=memory,
    )

    outcome = await handle_email_read(_handoff(position=1, word_for_word=True), ctx)

    assert outcome.reply_text.startswith("Dana Example wrote: ")
    assert outcome.reply_text.endswith(" ... that's where i stop reading.")
    assert len(outcome.reply_text) < len(long_text) + 100


# --- end-to-end through run_turn, per the plan's own required test names ---


def _account(label: str, access_token: str) -> AccountGrant:
    return AccountGrant(
        label=label, email=f"{label}@example.com", is_default=(label == "work"),
        access_token=access_token, unreachable_reason=None, calendars=(),
    )


class _GmailToolHost:
    def __init__(self, accounts, client) -> None:
        self._accounts = accounts
        self._client = client
        self.calls: "list[tuple[str, dict]]" = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, dict(arguments)))
        try:
            if name == "gmail_read":
                result = await handle_gmail_read(**arguments)
            elif name == "gmail_fetch_body":
                result = await handle_gmail_fetch_body(self._accounts, self._client, **arguments)
            else:
                raise AssertionError(f"unknown tool: {name}")
        except Denied as exc:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text=str(exc))])
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(result))])


async def test_ordinal_reference_resolves_from_memory(fake_audio_source, fake_stt, fake_tts):
    """After a turn that spoke three emails, a new turn whose brain calls
    `gmail_read(position=2)` fetches exactly the second spoken item's
    message id -- no `messages.list` request at all -- summarizes it
    through quarantine, and speaks "<label>: <sender>: <summary>"."""
    accounts = (_account("work", "at-work"), _account("home", "at-home"))
    fake = FakeGoogle()
    fake.add_gmail_full(
        "at-home", "m2", headers={"From": "Lee Example <lee@example.com>"}, text="see you saturday."
    )
    memory = EmailListMemory()
    memory.store("camera", (_DANA, _LEE, _ALEX))

    tool_host = _GmailToolHost(accounts, fake.client)
    brain = RecordingFakeBrain(
        replies=[
            BrainReply(tool_calls=[ToolCall(name="gmail_read", arguments={"position": 2})]),
            BrainReply(text="plans for saturday."),
        ]
    )
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=None, brain=brain, email_memory=memory
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="read the second one")])
    tts = fake_tts(chunks=[b"\x01"])
    timings = TurnTimings()

    await run_turn(
        source, stt, brain, tts, tool_host,
        tools_schema=[], system_prompt="you manage email", max_tool_rounds=3, timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == ["home: Lee Example: plans for saturday."]
    assert not any(r.url.path.endswith("/messages") for r in fake.requests)
    fetch_calls = [(name, args) for name, args in tool_host.calls if name == "gmail_fetch_body"]
    assert fetch_calls == [("gmail_fetch_body", {"account": "home", "message_id": "m2"})]
    assert timings.turn_outcome == "email_read"


async def test_word_for_word_makes_zero_brain_calls_after_the_tool_round(fake_audio_source, fake_stt, fake_tts):
    """A test asserts a word-for-word read makes zero brain calls after
    the tool round."""
    accounts = (_account("work", "at-work"),)
    fake = FakeGoogle()
    fake.add_gmail_full(
        "at-work", "m1", headers={"From": "Dana Example <dana@example.com>"}, text="let's do lunch friday."
    )
    memory = EmailListMemory()
    memory.store("camera", (_DANA,))

    tool_host = _GmailToolHost(accounts, fake.client)
    top_brain = RecordingFakeBrain(
        replies=[BrainReply(tool_calls=[ToolCall(name="gmail_read", arguments={"position": 1, "word_for_word": True})])]
    )
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=None,
        brain=_AssertNeverCalledBrain(),
        email_memory=memory,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="read it word for word")])
    tts = fake_tts(chunks=[b"\x01"])
    timings = TurnTimings()

    await run_turn(
        source, stt, top_brain, tts, tool_host,
        tools_schema=[], system_prompt="you manage email", max_tool_rounds=3, timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == ["Dana Example wrote: let's do lunch friday."]
    assert top_brain.call_count == 1
