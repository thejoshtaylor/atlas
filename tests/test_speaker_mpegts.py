"""Golden-byte tests for `speaker/mpegts.py`.

Every assertion here is frozen against go2rtc's own known-good output,
captured live from the proven prototype (`reference/tapo_talk.py`) this
session. Pure and deterministic -- no camera, no pytapo, no network.
"""

from __future__ import annotations

from spire_voice.speaker.mpegts import PesState, build_header, get_payload

PAT_PREFIX_HEX = "474000100000b00d0001c100000001f0002ab104b2"
PMT_PREFIX_HEX = "475000100002b0120001c10000fffff00090e100f0005631e8cd"


def test_build_header_is_two_188_byte_ts_packets():
    header = build_header()
    assert len(header) == 376


def test_build_header_packets_both_start_with_the_sync_byte():
    header = build_header()
    assert header[0] == 0x47
    assert header[188] == 0x47


def test_build_header_pat_matches_go2rtcs_known_good_bytes():
    """PAT on PID 0, program 1 mapping to PMT PID 0x1000, section CRC
    0x2ab104b2 -- captured from the proven prototype's live run."""
    header = build_header()
    assert header[:21].hex() == PAT_PREFIX_HEX


def test_build_header_pmt_matches_go2rtcs_known_good_bytes():
    """PMT on PID 0x1000, stream type 0x90 (PCMATapo) on elementary PID
    0x100, section CRC 0x5631e8cd -- captured from the proven prototype's
    live run."""
    header = build_header()
    assert header[188:214].hex() == PMT_PREFIX_HEX


def test_build_header_padding_after_each_prefix_is_zero():
    """Everything past the meaningful prefix within each 188-byte packet
    is fixed zero padding -- proves the tail-padding contract, not just
    the prefix bytes above."""
    header = build_header()
    pat_packet = header[:188]
    pmt_packet = header[188:376]
    assert pat_packet[21:] == b"\x00" * (188 - 21)
    assert pmt_packet[26:] == b"\x00" * (188 - 26)


def test_get_payload_returns_one_188_byte_packet_for_a_short_frame():
    pes = PesState()
    payload = bytes([0xD5] * 160)
    packet = get_payload(pes, 0, payload)
    assert len(packet) == 188
    assert packet[0] == 0x47


def test_get_payload_carries_the_pes_start_code_and_audio_stream_id():
    """The PES header's start code (00 00 01) plus the audio stream id
    (0xC0) must appear right after the four-byte TS header this frame's
    payload-unit-start-indicator packet carries."""
    pes = PesState()
    payload = bytes([0xD5] * 160)
    packet = get_payload(pes, 0, payload)
    assert b"\x00\x00\x01\xc0" in packet
