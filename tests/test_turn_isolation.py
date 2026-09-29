"""Phase 12 D-12: nothing crosses between two turns that run at the same time.

Three pieces of per-source state used to be shared by accident:

- the workflow run-id scope on `WorkflowToolHost` (Research Pitfall 6),
- the pending-action supersede key (Research Pitfall 5),
- the local speech models (Research Pitfall 11).

Each section below covers one of them.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from atlas.workflow.tool import WorkflowToolHost


def _host(repo) -> WorkflowToolHost:
    fixed_now = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)
    return WorkflowToolHost(repo, zone=None, clock=lambda: fixed_now)


async def _schedule_one(host: WorkflowToolHost) -> int:
    result = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [
                {
                    "kind": "call_service",
                    "arguments": {
                        "domain": "switch",
                        "service": "turn_off",
                        "entity_id": "switch.example_fan",
                    },
                }
            ],
            "delay_seconds": 300,
            "summary": "turn off the fan later",
        },
    )
    return result.structured_content["run_id"]


# ---------------------------------------------------------------------------
# Task 1: the workflow run-id scope is per task
# ---------------------------------------------------------------------------


async def test_two_concurrent_turns_each_keep_their_own_run_id_scope(fake_workflow_repository):
    repo = fake_workflow_repository()
    host = _host(repo)
    run_a = await _schedule_one(host)
    run_b = await _schedule_one(host)

    a_has_set = asyncio.Event()
    b_has_set = asyncio.Event()

    async def turn_a():
        host.set_current_turn_run_ids({run_a})
        a_has_set.set()
        await b_has_set.wait()  # turn B has now set its own ids
        own = await host.call_tool("cancel_workflow_run", {"run_id": run_a})
        other = await host.call_tool("cancel_workflow_run", {"run_id": run_b})
        return own, other

    async def turn_b():
        await a_has_set.wait()
        host.set_current_turn_run_ids({run_b})
        b_has_set.set()
        return await host.call_tool("cancel_workflow_run", {"run_id": run_a})

    (a_own, a_other), b_foreign = await asyncio.gather(
        asyncio.create_task(turn_a()), asyncio.create_task(turn_b())
    )

    assert not getattr(a_own, "is_error", False)
    assert a_other.is_error
    assert "not in this turn's own pending-run list" in a_other.content[0].text
    assert b_foreign.is_error
    assert "not in this turn's own pending-run list" in b_foreign.content[0].text
    # Turn B never lost its own run to turn A's cancel.
    assert (await repo.get_run(run_b)).status == "pending"


async def test_a_task_that_never_set_run_ids_refuses_every_run_id(fake_workflow_repository):
    repo = fake_workflow_repository()
    host = _host(repo)
    run_id = await _schedule_one(host)

    async def setter():
        host.set_current_turn_run_ids({run_id})

    await asyncio.create_task(setter())  # another task's scope must not leak here

    cancel = await host.call_tool("cancel_workflow_run", {"run_id": run_id})
    append = await host.call_tool(
        "append_workflow_steps",
        {"run_id": run_id, "steps": [{"kind": "speak", "arguments": {"text": "hi"}}]},
    )

    assert cancel.is_error
    assert append.is_error


async def test_a_child_task_sees_the_ids_its_parent_turn_set(fake_workflow_repository):
    repo = fake_workflow_repository()
    host = _host(repo)
    run_id = await _schedule_one(host)
    other_id = await _schedule_one(host)
    host.set_current_turn_run_ids({run_id})

    async def child(target: int):
        return await host.call_tool("cancel_workflow_run", {"run_id": target})

    shown, unshown = await asyncio.gather(child(run_id), child(other_id))

    assert not getattr(shown, "is_error", False)
    assert unshown.is_error


# ---------------------------------------------------------------------------
# Task 2: the pending-action key is per run
# ---------------------------------------------------------------------------

import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from atlas.config import SttConfig, TtsConfig  # noqa: E402
from atlas.providers import stt_faster_whisper, tts_piper  # noqa: E402
from atlas.providers.stt_faster_whisper import FasterWhisperStt  # noqa: E402
from atlas.providers.tts_piper import PiperTts  # noqa: E402
from atlas.transports.base import SourceFormat  # noqa: E402
from atlas.turn.email_handoff import handle_email_read  # noqa: E402
from atlas.turn.email_memory import EmailListItem, EmailListMemory  # noqa: E402
from atlas.turn.handoff import Handoff, HandoffContext, dispatch_handoff  # noqa: E402
from atlas.google.turn_context import build_handoff_context  # noqa: E402

from pending_action_fakes import FakePendingActionRepository  # noqa: E402

_NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def _calendar_handoff(title: str = "Dentist") -> Handoff:
    return Handoff(
        kind="pending_action",
        payload={
            "kind": "pending_action",
            "action": "calendar_create",
            "account": "home",
            "calendar_id": "cal-home-primary",
            "calendar_name": "Home",
            "calendar_primary": True,
            "title": title,
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
    )


def _ctx(repo, *, pending_source_key=None, **extra) -> HandoffContext:
    return HandoffContext(
        source_name="edge",
        tool_host=None,
        pending_actions=repo,
        brain=None,
        now=_NOW,
        pending_source_key=pending_source_key,
        **extra,
    )


async def _propose(ctx: HandoffContext, title: str = "Dentist"):
    outcome = await dispatch_handoff(
        _calendar_handoff(title),
        ctx,
        transcript=f"add {title}",
        follow_up_available=True,
        chain_depth=1,
    )
    assert outcome.follow_up is not None
    return outcome.follow_up.pending_action_id


async def test_two_runs_on_one_source_each_keep_their_awaiting_action():
    repo = FakePendingActionRepository()

    first = await _propose(_ctx(repo, pending_source_key="edge:1"), "Dentist")
    second = await _propose(_ctx(repo, pending_source_key="edge:2"), "Haircut")

    assert (await repo.get(first)).status == "awaiting"
    assert (await repo.get(second)).status == "awaiting"


async def test_an_amendment_inside_one_run_still_supersedes_its_earlier_row():
    repo = FakePendingActionRepository()
    ctx = _ctx(repo, pending_source_key="edge:1")

    first = await _propose(ctx, "Dentist")
    second = await _propose(ctx, "Dentist at four")

    assert (await repo.get(first)).status == "superseded"
    assert (await repo.get(second)).status == "awaiting"


async def test_no_pending_source_key_stores_under_the_source_name():
    repo = FakePendingActionRepository()

    first = await _propose(_ctx(repo), "Dentist")
    second = await _propose(_ctx(repo), "Haircut")

    assert (await repo.get(first)).source == "edge"
    assert (await repo.get(first)).status == "superseded"
    assert (await repo.get(second)).status == "awaiting"


async def test_the_pending_action_row_carries_the_per_run_key_as_its_source():
    repo = FakePendingActionRepository()

    action_id = await _propose(_ctx(repo, pending_source_key="edge:7"))

    assert (await repo.get(action_id)).source == "edge:7"


async def test_email_list_memory_stays_keyed_by_the_source_name():
    item = EmailListItem(
        position=1,
        account="work",
        message_id="m1",
        thread_id="m1",
        from_name="Dana Example",
        from_address="dana@example.com",
        subject="Lunch",
    )
    memory = EmailListMemory()
    memory.store("edge", (item,))
    tool_host = SimpleNamespace(calls=[])

    async def _call_tool(name, arguments):
        tool_host.calls.append((name, dict(arguments)))
        body = {
            "body": "see you then.",
            "truncated": False,
            "from_name": "Dana Example",
            "from_address": "dana@example.com",
            "subject": "Lunch",
        }
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(body))])

    tool_host.call_tool = _call_tool
    ctx = HandoffContext(
        source_name="edge",
        tool_host=tool_host,
        pending_actions=None,
        brain=None,
        email_memory=memory,
        pending_source_key="edge:9",
    )

    outcome = await handle_email_read(
        Handoff(
            kind="email_read",
            payload={"position": 1, "sender": None, "word_for_word": True},
        ),
        ctx,
    )

    assert tool_host.calls == [("gmail_fetch_body", {"account": "work", "message_id": "m1"})]
    assert outcome.turn_outcome == "email_read"


def test_build_handoff_context_passes_the_pending_source_key_through():
    app = SimpleNamespace(state=SimpleNamespace())

    keyed = build_handoff_context(app, "edge", pending_source_key="edge:3")
    plain = build_handoff_context(app, "edge")

    assert keyed.pending_source_key == "edge:3"
    assert keyed.source_name == "edge"
    assert plain.pending_source_key is None


# ---------------------------------------------------------------------------
# Task 2: one decode or render at a time per local provider
# ---------------------------------------------------------------------------


class _OverlapCounter:
    """Counts how many calls run at once inside worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.calls = 0

    def enter(self) -> None:
        with self._lock:
            self.calls += 1
            self.running += 1
            self.max_running = max(self.max_running, self.running)

    def leave(self) -> None:
        with self._lock:
            self.running -= 1


async def _frames(*chunks: bytes):
    for chunk in chunks:
        yield chunk


async def test_two_concurrent_decodes_on_one_stt_provider_never_overlap(tmp_path, monkeypatch):
    counter = _OverlapCounter()

    def slow_transcribe(model, audio, language):
        counter.enter()
        time.sleep(0.05)
        counter.leave()
        return []

    monkeypatch.setattr(stt_faster_whisper, "_run_transcribe", slow_transcribe)
    model_dir = tmp_path / "faster-whisper"
    model_dir.mkdir()
    stt = FasterWhisperStt(
        SttConfig(local_model_dir=str(model_dir)), load_model=lambda config: object()
    )

    async def one_turn():
        return [
            event
            async for event in stt.stream(
                _frames(np.zeros(160, dtype="<i2").tobytes()), SourceFormat("pcm", 16000)
            )
        ]

    await asyncio.gather(one_turn(), one_turn())

    assert counter.calls == 2
    assert counter.max_running == 1


async def test_two_concurrent_renders_on_one_tts_provider_never_overlap(tmp_path, monkeypatch):
    counter = _OverlapCounter()

    def slow_render(voice, text, sample_rate, codec):
        counter.enter()
        time.sleep(0.05)
        counter.leave()
        return b"\x00\x00"

    monkeypatch.setattr(tts_piper, "_render_for_sink", slow_render)
    voice_path = tmp_path / "voice.onnx"
    voice_path.write_bytes(b"fake-voice-weights")
    config_path = tmp_path / "voice.onnx.json"
    config_path.write_text("{}")
    tts = PiperTts(
        TtsConfig(piper_voice_path=str(voice_path), piper_config_path=str(config_path)),
        load_voice=lambda config: object(),
    )

    await asyncio.gather(tts.synthesize_once("one"), tts.synthesize_once("two"))

    assert counter.calls == 2
    assert counter.max_running == 1
