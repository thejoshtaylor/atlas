"""The reply cursor: where a reply is in time on the Pi's playback queue
(Phase 13, RESEARCH Finding 1). Pure, no asyncio, plain float times."""

from __future__ import annotations

import pytest

from atlas.providers.tts_xai import SinkFormat
from atlas.sources.reply_timing import ReplyCursor, bytes_per_second

_PCM_16K = SinkFormat("pcm", 16000)
_ALAW_8K = SinkFormat("alaw", 8000)


def test_bytes_per_second_follows_the_codec():
    assert bytes_per_second(_PCM_16K) == 32000
    assert bytes_per_second(_ALAW_8K) == 8000
    assert bytes_per_second(None) is None


def test_no_sink_means_no_op():
    cursor = ReplyCursor()
    cursor.note_audio_written(32000, 1.0, None, utterance_start=True)
    assert cursor.playing_until is None
    cursor.note_utterance_end("text")
    assert cursor.utterances == []


def test_no_bytes_means_no_op():
    cursor = ReplyCursor()
    cursor.note_audio_written(0, 1.0, _PCM_16K, utterance_start=True)
    assert cursor.playing_until is None


def test_one_second_of_pcm_at_16k_is_32000_bytes():
    cursor = ReplyCursor()
    cursor.note_audio_written(16000, 10.0, _PCM_16K, utterance_start=True)
    cursor.note_audio_written(16000, 10.0, _PCM_16K, utterance_start=False)
    assert cursor.playing_until == pytest.approx(11.0)


def test_a_second_utterance_queues_behind_one_that_still_plays():
    cursor = ReplyCursor()
    cursor.note_audio_written(32000, 10.0, _PCM_16K, utterance_start=True)  # filler, ends at 11.0
    cursor.note_utterance_end("filler")
    cursor.note_audio_written(32000, 10.2, _PCM_16K, utterance_start=True)  # answer, queued
    assert cursor.playing_until == pytest.approx(12.0)
    cursor.note_utterance_end("answer")
    assert cursor.utterances[-1].start == pytest.approx(11.0)
    assert cursor.utterances[-1].end == pytest.approx(12.0)


def test_an_utterance_after_the_queue_ended_starts_at_now():
    cursor = ReplyCursor()
    cursor.note_audio_written(32000, 10.0, _PCM_16K, utterance_start=True)
    cursor.note_utterance_end("first")
    cursor.note_audio_written(16000, 20.0, _PCM_16K, utterance_start=True)
    cursor.note_utterance_end("second")
    assert cursor.utterances[-1].start == pytest.approx(20.0)
    assert cursor.utterances[-1].end == pytest.approx(20.5)


def test_note_utterance_end_records_text_start_and_end():
    cursor = ReplyCursor()
    cursor.note_audio_written(32000, 5.0, _PCM_16K, utterance_start=True)
    cursor.note_utterance_end("hello there")
    (utterance,) = cursor.utterances
    assert (utterance.text, utterance.start, utterance.end) == ("hello there", 5.0, 6.0)


def test_note_utterance_end_with_nothing_written_records_nothing():
    cursor = ReplyCursor()
    cursor.note_utterance_end("nothing")
    assert cursor.utterances == []


def test_spoken_s_clamps_to_the_utterance_length():
    cursor = ReplyCursor()
    assert cursor.spoken_s(3.0) == 0.0
    cursor.note_audio_written(32000, 5.0, _PCM_16K, utterance_start=True)
    cursor.note_utterance_end("hello")
    assert cursor.spoken_s(4.0) == 0.0
    assert cursor.spoken_s(5.25) == pytest.approx(0.25)
    assert cursor.spoken_s(99.0) == pytest.approx(1.0)
