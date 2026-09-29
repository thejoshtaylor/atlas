"""Tests for MusicMixer: the FIFO-to-array music path. No audio device is
opened. The FIFO tests use a real named pipe under tmp_path."""

from __future__ import annotations

import logging
import os
import stat
import time

import numpy as np
import pytest

from atlas_edge.capture import Capture
from atlas_edge.music import LIBRESPOT_RATE, MusicMixer

OUT_RATE = 16000
BLOCK = 256


def _tone_bytes(freq: float, seconds: float, amp: float = 0.5) -> bytes:
    n = int(LIBRESPOT_RATE * seconds)
    t = np.arange(n) / LIBRESPOT_RATE
    mono = (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype("<i2")
    return np.column_stack([mono, mono]).tobytes()


def _drain(mixer: MusicMixer) -> np.ndarray:
    with mixer._lock:
        parts = list(mixer._buffer)
        mixer._buffer.clear()
        mixer._buffered = 0
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _mixer(**kwargs) -> MusicMixer:
    kwargs.setdefault("duck_active", lambda: False)
    return MusicMixer("/nonexistent/music.fifo", **kwargs)


def _block_from_mono(values: np.ndarray) -> bytes:
    return np.column_stack([values, values]).astype("<i2").tobytes()


def _left(block: bytes) -> np.ndarray:
    return np.frombuffer(block, dtype="<i2")[0::2]


# --- resampler -----------------------------------------------------------


def test_resampler_keeps_a_1khz_tone_at_1khz_and_full_amplitude():
    mixer = _mixer()
    raw = _tone_bytes(1000, 1.0)
    mixer.feed(raw)
    out = _drain(mixer)

    n_in = len(raw) // 4
    assert abs(len(out) - n_in * OUT_RATE / LIBRESPOT_RATE) <= 1

    steady = out[400:-400]
    spectrum = np.abs(np.fft.rfft(steady * np.hanning(len(steady))))
    peak_hz = np.argmax(spectrum) * OUT_RATE / len(steady)
    assert abs(peak_hz - 1000) <= OUT_RATE / len(steady)
    assert np.max(np.abs(steady)) == pytest.approx(0.5, rel=0.05)


def test_resampler_is_chunk_invariant():
    raw = _tone_bytes(1000, 0.5)
    whole = _mixer()
    whole.feed(raw)
    expected = _drain(whole)

    pieces = _mixer()
    pos = 0
    for size in (3, 37, 4001, 5, 100000, 7):
        pieces.feed(raw[pos : pos + size])
        pos += size
    pieces.feed(raw[pos:])
    got = _drain(pieces)

    assert len(got) == len(expected)
    assert np.max(np.abs(got - expected)) < 1e-4


def test_resampler_removes_a_12khz_tone():
    ref = _mixer()
    ref.feed(_tone_bytes(1000, 0.5))
    high = _mixer()
    high.feed(_tone_bytes(12000, 0.5))
    rms_ref = np.sqrt(np.mean(_drain(ref)[400:-400] ** 2))
    rms_high = np.sqrt(np.mean(_drain(high)[400:-400] ** 2))
    assert rms_high < 0.05 * rms_ref


# --- mix: pass-through, unity, saturation -----------------------------------


def test_mix_with_nothing_buffered_returns_the_block_unchanged():
    mixer = _mixer()
    block = bytes(range(256)) * 4
    assert mixer.mix(block, False) == block
    assert mixer.mix(block, True) == block


def test_mix_adds_music_to_both_channels_at_unity():
    mixer = _mixer()
    mixer._push(np.full(BLOCK, 0.25, dtype=np.float32))
    reply = _block_from_mono(np.full(BLOCK, 1000))
    out = np.frombuffer(mixer.mix(reply, False), dtype="<i2")
    assert np.array_equal(out[0::2], out[1::2])
    assert np.all(np.abs(out[0::2] - (1000 + 0.25 * 32767)) <= 1)


def test_mix_saturates_instead_of_wrapping():
    mixer = _mixer()
    mixer._push(np.full(BLOCK, 0.9, dtype=np.float32))
    reply = _block_from_mono(np.full(BLOCK, 30000))
    out = np.frombuffer(mixer.mix(reply, False), dtype="<i2")
    assert np.all(out == 32767)
    mixer._push(np.full(BLOCK, -0.9, dtype=np.float32))
    reply = _block_from_mono(np.full(BLOCK, -30000))
    out = np.frombuffer(mixer.mix(reply, False), dtype="<i2")
    assert np.all(out <= -32767)


# --- ducking -----------------------------------------------------------------


def _ducking_mixer(active):
    # ramp of two blocks: 512 samples at 16 kHz is 32 ms.
    return _mixer(duck_active=lambda: active["on"], duck_level=0.2, ramp_ms=32)


def test_duck_reaches_the_level_by_the_end_of_the_ramp_and_restores():
    active = {"on": True}
    mixer = _ducking_mixer(active)
    silence = bytes(BLOCK * 4)
    m = 0.5
    for _ in range(2):
        mixer._push(np.full(BLOCK, m, dtype=np.float32))
        last = mixer.mix(silence, False)
    assert abs(int(_left(last)[-1]) - 0.2 * m * 32767) <= 1
    assert mixer.gain == pytest.approx(0.2)

    active["on"] = False
    for _ in range(2):
        mixer._push(np.full(BLOCK, m, dtype=np.float32))
        last = mixer.mix(silence, False)
    assert abs(int(_left(last)[-1]) - m * 32767) <= 1
    assert mixer.gain == pytest.approx(1.0)


def test_reply_active_ducks_on_its_own():
    mixer = _ducking_mixer({"on": False})
    for _ in range(2):
        mixer._push(np.full(BLOCK, 0.5, dtype=np.float32))
        mixer.mix(bytes(BLOCK * 4), True)
    assert mixer.gain == pytest.approx(0.2)


def test_gain_tracks_even_with_no_music_buffered():
    mixer = _ducking_mixer({"on": True})
    for _ in range(3):
        assert mixer.mix(bytes(BLOCK * 4), False) == bytes(BLOCK * 4)
    assert mixer.gain == pytest.approx(0.2)
    mixer._push(np.full(BLOCK, 0.5, dtype=np.float32))
    out = _left(mixer.mix(bytes(BLOCK * 4), False))
    assert abs(int(out[0]) - 0.2 * 0.5 * 32767) <= 1


def test_zero_ramp_is_instant():
    mixer = _mixer(duck_active=lambda: True, ramp_ms=0)
    mixer.mix(bytes(BLOCK * 4), False)
    assert mixer.gain == pytest.approx(0.05)


# --- start(): fail-safe -------------------------------------------------------


def test_start_in_a_missing_directory_warns_once_and_never_raises(tmp_path, caplog):
    mixer = MusicMixer(str(tmp_path / "no" / "such" / "music.fifo"), duck_active=lambda: False)
    with caplog.at_level(logging.WARNING):
        mixer.start()
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    block = bytes(range(256)) * 4
    assert mixer.mix(block, False) == block
    mixer.stop()


def test_start_on_a_regular_file_leaves_it_untouched(tmp_path, caplog):
    path = tmp_path / "music.fifo"
    path.write_bytes(b"keep me")
    mixer = MusicMixer(str(path), duck_active=lambda: False)
    with caplog.at_level(logging.WARNING):
        mixer.start()
    assert path.read_bytes() == b"keep me"
    assert stat.S_ISREG(os.stat(path).st_mode)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    block = bytes(range(256)) * 4
    assert mixer.mix(block, False) == block
    mixer.stop()
    assert path.read_bytes() == b"keep me"


# --- end to end through a real FIFO and Capture --------------------------------


def _poll(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = predicate()
        if result is not None:
            return result
        time.sleep(0.01)
    return None


def test_fifo_bytes_come_out_of_the_capture_output_callback(tmp_path):
    fifo = tmp_path / "music.fifo"
    mixer = MusicMixer(str(fifo), duck_active=lambda: False)
    mixer.start()
    try:
        assert stat.S_ISFIFO(os.stat(fifo).st_mode)
        assert stat.S_IMODE(os.stat(fifo).st_mode) == 0o620

        writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        try:
            raw = _tone_bytes(1000, 0.2)
            view = memoryview(raw)
            while view:
                try:
                    written = os.write(writer, view)
                    view = view[written:]
                except BlockingIOError:
                    time.sleep(0.005)
        finally:
            os.close(writer)

        capture = Capture("reSpeaker", stream_factory=lambda kind, **kw: None)
        capture.mixer = mixer.mix
        needed = BLOCK * 2 * 2

        def _nonzero():
            outdata = bytearray(needed)
            capture._output_callback(outdata, BLOCK, None, None)
            samples = np.frombuffer(bytes(outdata), dtype="<i2")
            return samples if np.any(samples) else None

        samples = _poll(_nonzero)
        assert samples is not None
        assert np.array_equal(samples[0::2], samples[1::2])
    finally:
        mixer.stop()
    assert mixer._thread is None or not mixer._thread.is_alive()
