"""The brain-turn intent log (quick task 261001-mp8): the recorder, the
bounded writer, and `run_turn` writing one entry per brain turn."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas.db.brain_turn_repository import BrainTurnEntry
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.brain_turn_log import (
    MAX_PENDING_BRAIN_TURN_WRITES,
    BrainTurnLog,
    RecordingToolHost,
    TurnBrainRecord,
    classify_tool_call,
)

# --- test doubles ----------------------------------------------------------


class _FakeRepo:
    def __init__(self) -> None:
        self.entries: list[BrainTurnEntry] = []

    async def record_brain_turn(self, entry: BrainTurnEntry) -> None:
        self.entries.append(entry)


class _RaisingRepo:
    async def record_brain_turn(self, entry: BrainTurnEntry) -> None:
        raise RuntimeError(f"insert failed with parameters ({entry.transcript!r},)")


class _HangingRepo:
    def __init__(self) -> None:
        self.started = 0

    async def record_brain_turn(self, entry: BrainTurnEntry) -> None:
        self.started += 1
        await asyncio.Event().wait()


def _entry(turn_id: str = "t1", transcript: str = "secret words") -> BrainTurnEntry:
    return BrainTurnEntry(
        turn_id=turn_id,
        created_at=datetime.now(timezone.utc),
        transcript=transcript,
        normalized_transcript=transcript,
        transcript_fingerprint="f",
        continuation=False,
        tool_calls=(),
        reply_text=None,
        tier_index=None,
        tier_model=None,
        brain_latency_ms=None,
        outcome="completed",
    )


def _ok_result(payload: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(payload or {"ok": True}))])


# --- classify_tool_call ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arguments", "kind"),
    [
        ("ha_call_service", {"service": "turn_on", "entity_id": "light.example_lamp"}, "write"),
        ("ha_call_service", {"service": "get_forecasts"}, "read"),
        ("ha_get_state", {"entity_id": "light.example_lamp"}, "read"),
        ("weather_forecast", {}, "read"),
        ("calendar_list_events", {}, "read"),
        ("plugin__ha_get_state", {}, "read"),
        ("set_timer", {"seconds": 5}, "write"),
        ("schedule_workflow", {}, "write"),
        ("frobnicate", {}, "write"),
    ],
)
def test_classify_tool_call(name, arguments, kind):
    assert classify_tool_call(name, arguments) == kind


# --- TurnBrainRecord -------------------------------------------------------


def test_the_record_normalizes_and_fingerprints_the_command():
    text = "Hey Atlas, turn on the Example Lamp!"

    record = TurnBrainRecord.start(turn_id="t1", transcript=text, command_text=text, continuation=False)

    assert record.normalized_transcript == "turn on the example lamp"
    assert record.transcript_fingerprint == hashlib.sha256(b"turn on the example lamp").hexdigest()
    assert record.transcript == text


def test_caps_hold_for_transcript_arguments_calls_and_errors():
    record = TurnBrainRecord.start(
        turn_id="t1", transcript="x" * 10_000, command_text="y" * 10_000, continuation=False
    )
    assert len(record.transcript) == 2000
    assert len(record.normalized_transcript) == 2000

    huge = record.begin_call("tool", {"blob": "z" * 5000})
    assert huge["arguments"] == {"_truncated": True}

    odd = record.begin_call("tool", {"when": datetime(2026, 1, 1, tzinfo=timezone.utc)})
    assert isinstance(odd["arguments"]["when"], str)
    json.dumps(odd["arguments"])

    for _ in range(100):
        record.begin_call("tool", {})
    assert len(record.calls) == 32

    record.note_reply("r" * 9000)
    assert len(record.reply_text) == 2000


def test_finish_picks_the_outcome_and_the_latency():
    record = TurnBrainRecord.start(turn_id="t1", transcript="a", command_text="a", continuation=True)
    done_at = record.started_at + 1.2504

    completed = record.finish(outcome="completed", failed=False, cancelled=False, tool_rounds_done_at=done_at)
    assert completed.outcome == "completed"
    assert completed.brain_latency_ms == 1250
    assert completed.continuation is True
    assert completed.created_at.tzinfo is not None

    assert record.finish(outcome="completed", failed=True, cancelled=True, tool_rounds_done_at=None).outcome == "failed"
    cancelled = record.finish(outcome="completed", failed=False, cancelled=True, tool_rounds_done_at=None)
    assert cancelled.outcome == "cancelled"
    assert cancelled.brain_latency_ms is None


# --- RecordingToolHost -----------------------------------------------------


class _InnerHost:
    tools = ["a-tool"]

    def __init__(self, behaviors) -> None:
        self._behaviors = behaviors

    async def call_tool(self, name, arguments):
        behavior = self._behaviors[name]
        if isinstance(behavior, Exception):
            raise behavior
        await asyncio.sleep(behavior.get("delay", 0))
        return behavior["result"]


async def test_the_recording_host_keeps_model_order_and_records_outcomes():
    ok = _ok_result()
    error = SimpleNamespace(isError=True, content=[SimpleNamespace(text="e" * 500)])
    inner = _InnerHost(
        {
            "slow_read": {"delay": 0.05, "result": ok},
            "fast_write": {"delay": 0, "result": error},
            "boom": RuntimeError("secret detail"),
        }
    )
    record = TurnBrainRecord.start(turn_id="t1", transcript="a", command_text="a", continuation=False)
    host = RecordingToolHost(inner, record)

    assert host.inner is inner
    assert host.tools == ["a-tool"]
    results = await asyncio.gather(
        host.call_tool("slow_read", {}),
        host.call_tool("fast_write", {}),
        host.call_tool("boom", {}),
        return_exceptions=True,
    )

    assert results[0] is ok
    assert results[1] is error
    assert isinstance(results[2], RuntimeError)
    assert [c["name"] for c in record.calls] == ["slow_read", "fast_write", "boom"]
    assert [c["ok"] for c in record.calls] == [True, False, False]
    assert record.calls[0]["error"] is None
    assert record.calls[1]["error"] == "e" * 200
    assert record.calls[2]["error"] == "RuntimeError"


# --- BrainTurnLog ----------------------------------------------------------


async def test_a_raising_repository_is_contained_and_the_log_has_no_transcript(caplog):
    log = BrainTurnLog(_RaisingRepo())

    with caplog.at_level(logging.DEBUG, logger="atlas.turn.brain_turn_log"):
        log.submit(_entry(turn_id="turn-abc", transcript="secret words"))
        await log.drain()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "turn-abc" in errors[0].getMessage()
    assert "secret" not in caplog.text
    assert errors[0].exc_info is None


async def test_a_hanging_repository_bounds_pending_writes_and_drain_cancels(caplog):
    repo = _HangingRepo()
    log = BrainTurnLog(repo)

    with caplog.at_level(logging.WARNING, logger="atlas.turn.brain_turn_log"):
        for index in range(MAX_PENDING_BRAIN_TURN_WRITES + 10):
            log.submit(_entry(turn_id=f"t{index}"))
        await asyncio.sleep(0)
        assert repo.started == MAX_PENDING_BRAIN_TURN_WRITES
        full_warnings = [r for r in caplog.records if "already in flight" in r.getMessage()]
        assert len(full_warnings) == 1

        await log.drain(timeout=0.01)

    assert not log._pending


async def test_submit_never_raises_even_with_no_running_loop_problem():
    log = BrainTurnLog(_FakeRepo())
    log.submit(_entry())  # a running loop exists here
    await log.drain()


# --- run_turn --------------------------------------------------------------


class _ScriptedHost:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append(name)
        return _ok_result()


_SCHEMA = [
    {"type": "function", "function": {"name": "ha_get_state"}},
    {"type": "function", "function": {"name": "ha_call_service"}},
]


async def _run(
    fake_audio_source,
    fake_stt,
    fake_brain,
    fake_tts,
    *,
    text="Hey Atlas, turn on the kitchen light.",
    replies=None,
    log=None,
    macros=(),
    filler_cache=None,
):
    from atlas.turn.controller import run_turn

    if replies is None:
        replies = [
            BrainReply(tool_calls=[ToolCall(name="ha_get_state", arguments={"entity_id": "light.example_kitchen"})]),
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="ha_call_service",
                        arguments={"domain": "light", "service": "turn_on", "entity_id": "light.example_kitchen"},
                    )
                ]
            ),
            BrainReply(text="Done."),
        ]
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    kwargs = {}
    if log is not None:
        kwargs["brain_turn_log"] = log
    await run_turn(
        fake_audio_source(frames=[b"\x00\x01"]),
        fake_stt(events=[FinalTranscript(text=text)]),
        fake_brain(replies=replies),
        tts,
        _ScriptedHost(),
        tools_schema=_SCHEMA,
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        macros=macros,
        filler_cache=filler_cache,
        **kwargs,
    )
    return tts, timings


async def test_a_brain_turn_writes_one_entry_with_ordered_calls(fake_audio_source, fake_stt, fake_brain, fake_tts):
    repo = _FakeRepo()
    log = BrainTurnLog(repo)

    tts, timings = await _run(fake_audio_source, fake_stt, fake_brain, fake_tts, log=log)
    await log.drain()

    assert len(repo.entries) == 1
    entry = repo.entries[0]
    assert entry.turn_id == timings.turn_id
    assert entry.transcript == "Hey Atlas, turn on the kitchen light."
    assert entry.normalized_transcript == "turn on the kitchen light"
    assert entry.transcript_fingerprint == hashlib.sha256(b"turn on the kitchen light").hexdigest()
    assert [(c["name"], c["kind"], c["ok"]) for c in entry.tool_calls] == [
        ("ha_get_state", "read", True),
        ("ha_call_service", "write", True),
    ]
    assert entry.tool_calls[0]["arguments"] == {"entity_id": "light.example_kitchen"}
    assert entry.tool_calls[1]["arguments"]["service"] == "turn_on"
    assert entry.reply_text == "".join(tts.received_text)
    assert entry.tier_index == 0
    assert entry.outcome == "completed"
    assert entry.continuation is False
    assert isinstance(entry.brain_latency_ms, int) and entry.brain_latency_ms >= 0


async def test_a_failing_store_never_touches_the_spoken_reply(fake_audio_source, fake_stt, fake_brain, fake_tts):
    log = BrainTurnLog(_RaisingRepo())

    tts, timings = await _run(fake_audio_source, fake_stt, fake_brain, fake_tts, log=log)
    await log.drain()

    assert "".join(tts.received_text) == "Done."
    assert timings.turn_outcome == "completed"


async def test_a_hanging_store_does_not_delay_the_turn(fake_audio_source, fake_stt, fake_brain, fake_tts):
    log = BrainTurnLog(_HangingRepo())

    tts, timings = await asyncio.wait_for(
        _run(fake_audio_source, fake_stt, fake_brain, fake_tts, log=log), timeout=5
    )

    assert timings.turn_outcome == "completed"
    await log.drain(timeout=0.01)


async def test_a_macro_turn_writes_no_entry(fake_audio_source, fake_stt, fake_brain, fake_tts):
    from atlas.config import MacroActionConfig, MacroConfig

    macro = MacroConfig(
        phrase="good night",
        aliases=(),
        reply="good night",
        actions=(MacroActionConfig(tool="ha_get_state", arguments={}),),
    )
    repo = _FakeRepo()
    log = BrainTurnLog(repo)

    _tts, timings = await _run(
        fake_audio_source,
        fake_stt,
        fake_brain,
        fake_tts,
        text="good night",
        replies=[],
        log=log,
        macros=(macro,),
        filler_cache={None: {"good night": b"\x01\x02"}},
    )
    await log.drain()

    assert timings.turn_outcome == "macro"
    assert repo.entries == []


async def test_a_turn_without_a_log_behaves_as_before(fake_audio_source, fake_stt, fake_brain, fake_tts):
    tts, timings = await _run(fake_audio_source, fake_stt, fake_brain, fake_tts)

    assert "".join(tts.received_text) == "Done."
    assert timings.turn_outcome == "completed"


# --- D-14: the log never touches the macro module --------------------------


def _imported_modules(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append(module)
            names.extend(f"{module}.{alias.name}" for alias in node.names)
    return names


def test_neither_this_test_nor_the_log_module_imports_the_macro_module():
    for path in (Path(__file__), Path(__file__).resolve().parents[1] / "src/atlas/turn/brain_turn_log.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = [name for name in _imported_modules(tree) if "macros" in name]
        assert offenders == [], f"{path.name} imports {offenders}"
