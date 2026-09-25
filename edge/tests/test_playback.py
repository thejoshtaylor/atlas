"""RED for Playback (10-09-PLAN.md Task 2, per the orchestrator directive:
Playback never opens its own RawOutputStream -- it writes through a fake
`enqueue` sink, the same seam Capture.enqueue_playback provides in
production). No real audio device anywhere in this file."""

from __future__ import annotations

import pytest

from atlas_edge.playback import Playback


def _fake_clock(values):
    it = iter(values)

    def _clock():
        return next(it)

    return _clock


@pytest.mark.asyncio
async def test_mono_samples_are_duplicated_into_both_channels() -> None:
    sunk: "list[bytes]" = []
    playback = Playback(sunk.append, clock=_fake_clock([0.0] * 10))

    # Two 2-byte PCM16 "samples": a=0x0102, b=0x0304.
    mono = bytes([0x01, 0x02, 0x03, 0x04])
    await playback.write(mono)

    assert b"".join(sunk) == bytes([0x01, 0x02, 0x01, 0x02, 0x03, 0x04, 0x03, 0x04])


@pytest.mark.asyncio
async def test_an_odd_trailing_byte_is_kept_for_the_next_write() -> None:
    sunk: "list[bytes]" = []
    playback = Playback(sunk.append, clock=_fake_clock([0.0] * 10))

    await playback.write(bytes([0x01, 0x02, 0x03]))  # one full sample + 1 leftover byte
    assert b"".join(sunk) == bytes([0x01, 0x02, 0x01, 0x02])

    sunk.clear()
    await playback.write(bytes([0x04]))  # completes the leftover sample: [0x03, 0x04]
    assert b"".join(sunk) == bytes([0x03, 0x04, 0x03, 0x04])


@pytest.mark.asyncio
async def test_a_buffer_past_max_buffer_s_drops_the_oldest_audio_and_counts_it() -> None:
    sunk: "list[bytes]" = []
    # A frozen clock -- no decay between writes -- so the buffer only grows.
    playback = Playback(
        sunk.append,
        sample_rate=16000,
        channels=2,
        max_buffer_s=0.001,  # tiny cap: 16000*2*2*0.001 = 64 bytes
        clock=_fake_clock([0.0] * 100),
    )

    mono_chunk = bytes(range(64)) * 4  # 256 mono bytes -> 512 stereo bytes, far over cap
    await playback.write(mono_chunk)

    assert playback.dropped_bytes > 0
    assert len(b"".join(sunk)) <= 64


@pytest.mark.asyncio
async def test_the_buffer_estimate_decays_over_elapsed_time() -> None:
    sunk: "list[bytes]" = []
    bytes_per_second = 16000 * 2 * 2
    max_buffer_s = 1.0
    playback = Playback(
        sunk.append,
        sample_rate=16000,
        channels=2,
        max_buffer_s=max_buffer_s,
        # First write establishes last_update_at at t=0. Second write is at
        # t=2.0s later -- long enough to fully drain any prior estimate.
        clock=_fake_clock([0.0, 2.0]),
    )

    small = bytes([0x00, 0x01])  # 2 mono bytes -> 4 stereo bytes
    await playback.write(small)
    await playback.write(small)

    assert playback.dropped_bytes == 0


@pytest.mark.asyncio
async def test_start_and_stop_are_harmless_no_ops() -> None:
    playback = Playback(lambda data: None)
    playback.start()
    playback.stop()
