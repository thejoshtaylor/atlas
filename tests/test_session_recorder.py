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

from spire_voice.config import SessionConfig
from spire_voice.session.recorder import SessionRecorder
from spire_voice.timing import TurnTimings


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
