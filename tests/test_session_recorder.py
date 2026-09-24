"""Real assertions for the per-turn session recorder (DBG-01, DBG-02) and,
from Task 2 onward, the merged timeline it feeds (`session/timeline.py`).

Every case builds its own `SessionConfig` pointed at `tmp_path` -- nothing
here writes outside the pytest temporary directory, and nothing here is a
real house (`tests/test_repo_hygiene.py` guards that claim mechanically).

Replaces the Wave-0 scaffold from plan 02-01.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from atlas.config import SessionConfig
from atlas.session import timeline as timeline_module
from atlas.session.recorder import SessionRecorder
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn


def _session_config(tmp_path: Path, *, record_audio: bool = True) -> SessionConfig:
    return SessionConfig(dir=str(tmp_path), record_audio=record_audio)


def test_two_turns_starting_in_the_same_second_get_distinct_directories(tmp_path):
    """CD-5: the directory name carries the turn id, not the timestamp
    alone, so two turns starting in the same second cannot collide.
    """
    config = _session_config(tmp_path)
    recorder_a = SessionRecorder(config, TurnTimings())
    recorder_b = SessionRecorder(config, TurnTimings())

    assert recorder_a.directory != recorder_b.directory
    assert recorder_a.directory.exists()
    assert recorder_b.directory.exists()


def test_turn_that_produced_no_audio_still_writes_its_directory(tmp_path):
    """A turn that produced no audio still writes its folder, holding the
    events and the timing record -- an absent folder would make a failed
    turn indistinguishable from a turn that never happened (T-02-22).
    """
    config = _session_config(tmp_path)
    timings = TurnTimings(turn_outcome="empty_transcript")
    timings.mark_turn_started()
    recorder = SessionRecorder(config, timings)

    recorder.record_event({"type": "reply.text", "text": "sorry, i didn't catch that"})
    recorder.close(timings)

    assert recorder.directory.exists()
    assert (recorder.directory / "events.jsonl").exists()
    assert (recorder.directory / "timing.json").exists()
    assert list(recorder.directory.glob("audio.*")) == []


def test_turn_timings_survive_serialization_with_an_unset_stage_as_null(tmp_path):
    """A stage never reached must serialize as absent/null, never as a
    fabricated zero -- a stage that took no time and a stage that never
    happened are different facts.
    """
    config = _session_config(tmp_path)
    timings = TurnTimings()
    timings.mark_turn_started()
    timings.mark_stt_socket_open()
    # Deliberately never reaches first_partial_at, stt_final_at, and later.
    recorder = SessionRecorder(config, timings)
    recorder.close(timings)

    payload = json.loads((recorder.directory / "timing.json").read_text(encoding="utf-8"))
    assert payload["turn_started_at"] is not None
    assert payload["first_partial_at"] is None
    assert payload["first_partial_at"] != 0
    assert payload["stt_final_at"] is None


def test_two_equal_timestamp_stages_both_appear_with_their_own_equal_values(tmp_path):
    """`first_audio_at`/`answer_audio_at` are deliberately bit-equal on a
    turn with no filler -- neither may be collapsed into the other nor
    given a fabricated separation.
    """
    config = _session_config(tmp_path)
    timings = TurnTimings()
    timings.mark_turn_started()
    timings.mark_stt_final()
    now = time.monotonic()
    timings.first_audio_at = now
    timings.answer_audio_at = now
    recorder = SessionRecorder(config, timings)
    recorder.close(timings)

    payload = json.loads((recorder.directory / "timing.json").read_text(encoding="utf-8"))
    assert payload["first_audio_at"] == payload["answer_audio_at"] == now


def test_written_audio_bytes_equal_the_bytes_the_turn_drained(tmp_path):
    """The audio a session folder holds is the raw source bytes exactly as
    captured (D-13) -- a second tap or a re-encode would fail this.
    """
    config = _session_config(tmp_path)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)
    recorder.set_audio_format("alaw", 8000)
    drained = [b"\x01\x02", b"\x03\x04\x05"]
    for chunk in drained:
        recorder.record_audio_chunk(chunk)
    recorder.close(timings)

    audio_path = recorder.directory / "audio.alaw"
    assert audio_path.read_bytes() == b"".join(drained)


def test_record_audio_false_omits_the_audio_file_but_keeps_events_and_timing(tmp_path):
    """The record-audio flag omits only the audio -- the events and the
    timing record are the cheap part and stay written either way.
    """
    config = _session_config(tmp_path, record_audio=False)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)
    recorder.set_audio_format("alaw", 8000)
    recorder.record_audio_chunk(b"\x01\x02")
    recorder.record_event({"type": "transcript.partial", "text": "turn on"})
    recorder.close(timings)

    assert list(recorder.directory.glob("audio.*")) == []
    assert (recorder.directory / "events.jsonl").exists()
    assert (recorder.directory / "timing.json").exists()


def test_events_are_written_one_json_object_per_line_in_order(tmp_path):
    config = _session_config(tmp_path)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)
    recorder.record_event({"type": "transcript.partial", "text": "turn on"})
    recorder.record_event({"type": "reply.text", "text": "turned on the fan"})
    recorder.close(timings)

    lines = (recorder.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [event["text"] for event in parsed] == ["turn on", "turned on the fan"]


def test_close_is_idempotent(tmp_path):
    """Every exit path in `run_turn` calls `close()` from one `finally`
    block -- a second call must be a no-op, not a second write.
    """
    config = _session_config(tmp_path)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)
    recorder.close(timings)
    recorder.close(timings)


def test_the_timeline_is_written_alongside_the_other_three_artifacts(tmp_path):
    config = _session_config(tmp_path)
    timings = TurnTimings()
    timings.mark_turn_started()
    recorder = SessionRecorder(config, timings)
    recorder.record_event({"type": "reply.text", "text": "turned on the fan"})
    recorder.close(timings)

    assert (recorder.directory / "timeline.jsonl").exists()


def test_timeline_renders_events_in_the_order_the_jsonl_wrote_them(tmp_path):
    config = _session_config(tmp_path)
    timings = TurnTimings()
    timings.mark_turn_started()
    timings.mark_stt_final()
    recorder = SessionRecorder(config, timings)
    recorder.record_event({"type": "transcript.partial", "text": "turn on"})
    recorder.record_event({"type": "reply.text", "text": "turned on the fan"})
    recorder.close(timings)

    lines = (recorder.directory / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    rendered = [json.loads(line) for line in lines]
    event_texts = [entry["text"] for entry in rendered if entry["kind"] == "event"]
    assert event_texts == ["turn on", "turned on the fan"]
    stage_labels = [entry["stage"] for entry in rendered if entry["kind"] == "stage"]
    assert stage_labels == ["turn_started_at", "stt_final_at"]


def test_timeline_regenerates_byte_identical_after_deletion(tmp_path):
    """The property that makes 'derived, never the source of truth' a fact
    rather than a comment: delete the rendered timeline, rebuild it from
    the JSONL and the timing record alone, and it must come back
    byte-identical.
    """
    config = _session_config(tmp_path)
    timings = TurnTimings()
    timings.mark_turn_started()
    timings.mark_stt_final()
    recorder = SessionRecorder(config, timings)
    recorder.record_event({"type": "transcript.partial", "text": "turn on"})
    recorder.record_event({"type": "reply.text", "text": "turned on the fan"})
    recorder.close(timings)

    timeline_path = recorder.directory / "timeline.jsonl"
    original = timeline_path.read_bytes()

    timeline_path.unlink()
    assert not timeline_path.exists()

    timeline_module.regenerate_timeline(recorder.directory)

    assert timeline_path.read_bytes() == original


def test_timeline_preserves_written_order_for_events_whose_timestamps_tie():
    """A pure test of `render_timeline` itself: two events sharing the
    exact same `recorded_at` value must keep the order they were written
    in, not an order re-derived from a second clock.
    """
    tied_ts = 100.0
    events = [
        {"recorded_at": tied_ts, "type": "transcript.partial", "text": "first"},
        {"recorded_at": tied_ts, "type": "transcript.partial", "text": "second"},
    ]

    rendered = timeline_module.render_timeline(events, {})

    assert [entry["text"] for entry in rendered] == ["first", "second"]


async def test_a_full_turn_through_run_turn_produces_all_four_artifacts(
    fake_audio_source, recording_fake_stt, fake_brain, fake_tts, tmp_path
):
    """The integration case: a real `run_turn`, wired to a `SessionRecorder`,
    leaves a directory holding all four artifacts -- not the recorder
    exercised in isolation.

    `recording_fake_stt`, not `fake_stt`, because `FakeStt` accepts and
    ignores whatever `frames` it is handed -- it never actually drains the
    iterator, so it can never prove this plan's tap did either.
    """
    from atlas.providers.base import BrainReply, FinalTranscript

    config = _session_config(tmp_path)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)

    source = fake_audio_source(frames=[b"\x00\x01", b"\x02\x03"])
    stt = recording_fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = fake_brain(replies=[BrainReply(text="turned on the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host=None,
        tools_schema=[],
        system_prompt="",
        max_tool_rounds=3,
        timings=timings,
        session_recorder=recorder,
    )

    assert (recorder.directory / "events.jsonl").exists()
    assert (recorder.directory / "timing.json").exists()
    assert (recorder.directory / "timeline.jsonl").exists()
    audio_path = recorder.directory / "audio.pcm"
    assert audio_path.exists()
    assert audio_path.read_bytes() == b"\x00\x01\x02\x03"


async def test_the_recorded_audio_byte_count_equals_what_the_turn_drained(
    fake_audio_source, recording_fake_stt, fake_brain, fake_tts, tmp_path
):
    """A second tap on the source would double this count -- this test
    would fail it.
    """
    from atlas.providers.base import BrainReply, FinalTranscript

    config = _session_config(tmp_path)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)

    frames = [b"\x00\x01", b"\x02\x03", b"\x04\x05"]
    source = fake_audio_source(frames=frames)
    stt = recording_fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = fake_brain(replies=[BrainReply(text="turned on the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host=None,
        tools_schema=[],
        system_prompt="",
        max_tool_rounds=3,
        timings=timings,
        session_recorder=recorder,
    )

    audio_path = recorder.directory / "audio.pcm"
    assert len(audio_path.read_bytes()) == sum(len(chunk) for chunk in frames)


async def test_a_turn_ending_on_an_empty_transcript_still_writes_its_directory(fake_audio_source, fake_stt, fake_brain, fake_tts, tmp_path):
    from atlas.providers.base import FinalTranscript

    config = _session_config(tmp_path)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[])

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host=None,
        tools_schema=[],
        system_prompt="",
        max_tool_rounds=3,
        timings=timings,
        session_recorder=recorder,
    )

    assert timings.turn_outcome == "empty_transcript"
    assert recorder.directory.exists()
    assert (recorder.directory / "events.jsonl").exists()
    assert (recorder.directory / "timing.json").exists()


async def test_record_audio_off_keeps_the_directory_events_and_timing_but_not_audio(
    fake_audio_source, fake_stt, fake_brain, fake_tts, tmp_path
):
    from atlas.providers.base import BrainReply, FinalTranscript

    config = _session_config(tmp_path, record_audio=False)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = fake_brain(replies=[BrainReply(text="turned on the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host=None,
        tools_schema=[],
        system_prompt="",
        max_tool_rounds=3,
        timings=timings,
        session_recorder=recorder,
    )

    assert recorder.directory.exists()
    assert (recorder.directory / "events.jsonl").exists()
    assert (recorder.directory / "timing.json").exists()
    assert list(recorder.directory.glob("audio.*")) == []


def test_timeline_module_holds_no_state_between_calls():
    """`render_timeline` computes its view fresh every call -- calling it
    twice on the same inputs must produce the same output, not a cached
    or mutated one.
    """
    events = [{"recorded_at": 1.0, "type": "reply.text", "text": "hello"}]
    timing_payload = {"turn_started_at": 0.5}

    first = timeline_module.render_timeline(events, timing_payload)
    second = timeline_module.render_timeline(events, timing_payload)

    assert first == second
