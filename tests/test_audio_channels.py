"""Real assertions for `SourceFormat`'s channel validation
(`transports/base.py`) and the channel selector
(`atlas.audio.channels`) -- 10-02-PLAN.md Task 2, closing the
`<assumption_delta_decision>` invariant: every shipped source's own
`SourceFormat` has `0 <= asr_channel < channels`, and `stt_view` always
hands speech-to-text a one-channel format.
"""

from __future__ import annotations

import pytest

from atlas.audio.channels import select_channel, stt_view
from atlas.config import CameraConfig, EdgeSourceConfig
from atlas.transports.base import SourceFormat
from atlas.transports.camera import CameraAudioSource
from atlas.transports.edge import EdgeAudioSource
from atlas.transports.websocket import WebSocketAudioSource


def test_zero_channels_raises():
    with pytest.raises(ValueError):
        SourceFormat("pcm", 16000, channels=0)


def test_asr_channel_at_or_past_channels_raises():
    with pytest.raises(ValueError):
        SourceFormat("pcm", 16000, channels=2, asr_channel=2)
    with pytest.raises(ValueError):
        SourceFormat("pcm", 16000, channels=2, asr_channel=5)


def test_negative_asr_channel_raises():
    with pytest.raises(ValueError):
        SourceFormat("pcm", 16000, channels=2, asr_channel=-1)


def test_every_existing_positional_construction_still_works():
    fmt = SourceFormat("alaw", 8000)
    assert fmt.channels == 1
    assert fmt.asr_channel == 0


def test_select_channel_on_a_chunk_that_is_not_a_whole_number_of_frames_raises():
    with pytest.raises(ValueError):
        select_channel(b"\x00\x01\x02", 2, 0)  # 3 bytes, 2 channels -> not a whole frame.


def test_select_channel_picks_the_named_channel():
    # Two frames: (ch0=1, ch1=2), (ch0=3, ch1=4), little-endian int16.
    chunk = bytes([1, 0, 2, 0, 3, 0, 4, 0])
    assert select_channel(chunk, 2, 0) == bytes([1, 0, 3, 0])
    assert select_channel(chunk, 2, 1) == bytes([2, 0, 4, 0])


async def test_stt_view_on_a_one_channel_format_returns_the_same_iterator_object():
    async def _frames():
        yield b"\x00\x01"

    frames = _frames()
    fmt = SourceFormat("pcm", 16000)

    returned_frames, returned_fmt = stt_view(frames, fmt)

    assert returned_frames is frames
    assert returned_fmt == fmt


async def test_stt_view_on_a_two_channel_pcm_format_de_interleaves_to_one_channel():
    async def _frames():
        yield bytes([1, 0, 2, 0])  # one frame: ch0=1, ch1=2

    frames, fmt = stt_view(_frames(), SourceFormat("pcm", 16000, channels=2, asr_channel=1))

    assert fmt == SourceFormat("pcm", 16000)
    chunks = [chunk async for chunk in frames]
    assert chunks == [bytes([2, 0])]


async def test_stt_view_on_a_multichannel_non_pcm_format_raises():
    async def _frames():
        yield b"\x00"

    with pytest.raises(ValueError):
        stt_view(_frames(), SourceFormat("alaw", 8000, channels=2, asr_channel=1))


def test_every_shipped_sources_own_format_holds_the_asr_channel_invariant():
    """0 <= asr_channel < channels for every source this codebase ships,
    and `stt_view` always hands speech-to-text a one-channel format --
    the `<assumption_delta_decision>` invariant test."""
    camera_source = CameraAudioSource(CameraConfig(rtsp_url="rtsp://camera.invalid/stream1"), _NoopSpeaker())
    ws_source = WebSocketAudioSource(object())
    edge_source = EdgeAudioSource(
        EdgeSourceConfig(sample_rate=16000, channels=2, asr_channel=1, pre_roll_ms=200, tail_ms=300)
    )

    for source in (camera_source, ws_source, edge_source):
        fmt = source.source_format()
        assert 0 <= fmt.asr_channel < fmt.channels

        async def _one_frame():
            yield b"\x00" * (2 * fmt.channels if fmt.encoding == "pcm" else 1)

        stt_frames, stt_fmt = stt_view(_one_frame(), fmt)
        assert stt_fmt.channels == 1


class _NoopSpeaker:
    async def write(self, chunk: bytes) -> None:
        return None
