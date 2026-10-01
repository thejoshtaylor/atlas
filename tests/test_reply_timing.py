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


# --- D-02: where the reply's own wake word falls in time --------------------


from atlas.sources.reply_timing import ReplyUtterance, in_any_window, wake_word_windows  # noqa: E402


def _windows(text, *, start=10.0, end=11.0, phrase="atlas", margin=0.5, echo=0.0):
    return wake_word_windows(
        [ReplyUtterance(text, start, end)], wake_phrase=phrase, margin_s=margin, echo_delay_s=echo
    )


def test_a_reply_with_no_wake_word_has_no_windows():
    assert _windows("the lights are on") == ()


def test_a_window_follows_the_words_character_position():
    text = "hello, I am Atlas, here to help"
    match_start = text.index("Atlas")
    match_end = match_start + len("Atlas")
    ((low, high),) = _windows(text)
    assert low == pytest.approx(10.0 + match_start / len(text) - 0.5)
    assert high == pytest.approx(10.0 + match_end / len(text) + 0.5)


def test_the_echo_delay_widens_the_high_edge_only():
    (plain,) = _windows("hello Atlas")
    (delayed,) = _windows("hello Atlas", echo=0.25)
    assert delayed[0] == pytest.approx(plain[0])
    assert delayed[1] == pytest.approx(plain[1] + 0.25)


@pytest.mark.parametrize("text", ["ATLAS is here", "atlas is here", "Hey, Atlas!"])
def test_matching_ignores_case_and_punctuation(text):
    assert len(_windows(text)) == 1


@pytest.mark.parametrize("text", ["the atlases are heavy", "a fatlas appeared", "atlas2 is a model"])
def test_matching_needs_a_whole_word(text):
    assert _windows(text) == ()


def test_two_words_make_two_windows_and_two_replies_make_windows_from_both():
    assert len(_windows("Atlas here, Atlas there")) == 2
    both = wake_word_windows(
        [ReplyUtterance("I am Atlas", 1.0, 2.0), ReplyUtterance("Atlas again", 5.0, 6.0)],
        wake_phrase="atlas",
        margin_s=0.1,
        echo_delay_s=0.0,
    )
    assert len(both) == 2
    assert both[0][1] < 3.0 < 4.0 < both[1][0] + 1.0


def test_a_two_word_phrase_matches_on_its_last_word_once():
    assert len(_windows("hey Atlas, welcome", phrase="hey atlas")) == 1


def test_an_empty_phrase_has_no_windows():
    assert _windows("Atlas", phrase="") == ()
    assert _windows("Atlas", phrase="  ") == ()


def test_in_any_window_is_inclusive_at_both_edges():
    windows = ((1.0, 2.0), (5.0, 6.0))
    assert in_any_window(1.0, windows)
    assert in_any_window(2.0, windows)
    assert in_any_window(5.5, windows)
    assert not in_any_window(2.5, windows)
    assert not in_any_window(0.9, windows)
    assert in_any_window(1.0, ()) is False
