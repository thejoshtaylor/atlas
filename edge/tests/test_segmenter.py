"""RED for Segmenter/VadEvent/AudioOut/frames_for_ms (10-08-PLAN.md Task 1)."""

from __future__ import annotations

from atlas_edge.segmenter import AudioOut, Segmenter, VadEvent, frames_for_ms


def _silence(n: int, start_t: float = 0.0) -> list[tuple[bytes, bool, float]]:
    return [(f"s{i}".encode(), False, start_t + i * 0.016) for i in range(n)]


def _speech(n: int, start_t: float) -> list[tuple[bytes, bool, float]]:
    return [(f"v{i}".encode(), True, start_t + i * 0.016) for i in range(n)]


def test_only_non_speech_emits_nothing_ever():
    seg = Segmenter(pre_roll_frames=3, tail_frames=2)
    out = []
    for frame, is_speech, t in _silence(50):
        out.extend(seg.push(frame, is_speech, t))
    assert out == []


def test_first_speech_frame_emits_start_then_preroll_then_live():
    seg = Segmenter(pre_roll_frames=3, tail_frames=2)
    pre = _silence(5)  # more than pre_roll_frames -- only the last 3 kept
    for frame, is_speech, t in pre:
        assert seg.push(frame, is_speech, t) == []

    speech_frame, _, speech_t = ("v0".encode(), True, 1.0)
    out = seg.push(speech_frame, True, speech_t)

    assert out[0] == VadEvent("vad.start", 0)
    # Only the last 3 pre-roll frames survive (maxlen=3), oldest first.
    expected_preroll = pre[-3:]
    for item, (expected_frame, _, expected_t) in zip(out[1:4], expected_preroll):
        assert item == AudioOut(expected_frame, False, expected_t)
    assert out[4] == AudioOut(speech_frame, True, speech_t)
    assert len(out) == 5


def test_vad_end_fires_at_once_then_tail_then_stops():
    seg = Segmenter(pre_roll_frames=3, tail_frames=2)
    for frame, is_speech, t in _silence(3):
        seg.push(frame, is_speech, t)
    for frame, is_speech, t in _speech(2, start_t=1.0):
        seg.push(frame, is_speech, t)

    end_frame, end_t = "sil0".encode(), 2.0
    out = seg.push(end_frame, False, end_t)
    assert out[0] == VadEvent("vad.end", 0)
    assert out[1] == AudioOut(end_frame, True, end_t)
    assert len(out) == 2

    tail2_frame, tail2_t = "sil1".encode(), 2.016
    out2 = seg.push(tail2_frame, False, tail2_t)
    assert out2 == [AudioOut(tail2_frame, True, tail2_t)]

    # tail_frames (2) is exhausted -- output stops.
    out3 = seg.push("sil2".encode(), False, 2.032)
    assert out3 == []


def test_speech_resumes_inside_tail_with_no_preroll_replay():
    seg = Segmenter(pre_roll_frames=3, tail_frames=5)
    for frame, is_speech, t in _silence(3):
        seg.push(frame, is_speech, t)
    for frame, is_speech, t in _speech(2, start_t=1.0):
        seg.push(frame, is_speech, t)
    seg.push("end".encode(), False, 2.0)  # vad.end seq=0, tail_remaining=4

    resume_frame, resume_t = "resume".encode(), 2.016
    out = seg.push(resume_frame, True, resume_t)
    assert out == [VadEvent("vad.start", 1), AudioOut(resume_frame, True, resume_t)]


def test_audio_out_carries_its_own_captured_at():
    seg = Segmenter(pre_roll_frames=2, tail_frames=1)
    seg.push(b"p0", False, 10.0)
    seg.push(b"p1", False, 10.016)
    out = seg.push(b"v0", True, 10.032)
    # vad.start, then pre-roll p0/p1 with their own captured_at, then v0.
    assert out[1] == AudioOut(b"p0", False, 10.0)
    assert out[2] == AudioOut(b"p1", False, 10.016)
    assert out[3] == AudioOut(b"v0", True, 10.032)


def test_frames_for_ms_rounds_up():
    # 256 samples at 16kHz is 16ms/frame.
    assert frames_for_ms(1100, 256, 16000) == 69  # 1100/16 = 68.75 -> 69
    assert frames_for_ms(1300, 256, 16000) == 82  # 1300/16 = 81.25 -> 82
    assert frames_for_ms(16, 256, 16000) == 1
    assert frames_for_ms(0, 256, 16000) == 0
