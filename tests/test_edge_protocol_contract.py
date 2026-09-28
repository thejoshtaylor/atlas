"""Pins edge protocol v1's two independent restatements against each
other (10-02-PLAN.md Task 2): `atlas.transports.edge` (server) and
`atlas_edge.protocol` (Pi, stdlib-only, never importing `atlas`). Neither
module imports the other -- this test is the one place that imports both
and proves they agree.
"""

from __future__ import annotations

import atlas_edge.protocol as pi_protocol

from atlas.config import EdgeSourceConfig
from atlas.transports import edge as server_protocol


def test_protocol_version_and_frame_samples_match():
    assert server_protocol.PROTOCOL_VERSION == pi_protocol.PROTOCOL_VERSION
    assert server_protocol.FRAME_SAMPLES == pi_protocol.FRAME_SAMPLES


def test_message_type_constants_match():
    assert server_protocol.MSG_HELLO == pi_protocol.MSG_HELLO
    assert server_protocol.MSG_PING == pi_protocol.MSG_PING
    assert server_protocol.MSG_PONG == pi_protocol.MSG_PONG
    assert server_protocol.MSG_VAD_START == pi_protocol.MSG_VAD_START
    assert server_protocol.MSG_VAD_END == pi_protocol.MSG_VAD_END
    assert server_protocol.MSG_DOA == pi_protocol.MSG_DOA
    assert server_protocol.MSG_LATENCY == pi_protocol.MSG_LATENCY
    assert server_protocol.MSG_LED == pi_protocol.MSG_LED


def test_every_pi_builder_produces_text_the_server_accepts():
    for text in (
        pi_protocol.vad_start(1),
        pi_protocol.vad_end(2),
        pi_protocol.doa([10.0, 20.0], [0.5, 0.1]),
        pi_protocol.doa([359.9]),  # speech_energy omitted -- defaults to [].
        pi_protocol.latency(100.0, 150.0, 200.0, 42),
        pi_protocol.pong(9, 123456),
    ):
        parsed = server_protocol.parse_edge_event(text)
        assert parsed["type"] in {
            server_protocol.MSG_VAD_START,
            server_protocol.MSG_VAD_END,
            server_protocol.MSG_DOA,
            server_protocol.MSG_LATENCY,
            server_protocol.MSG_PONG,
        }


def test_the_servers_hello_parses_on_the_pi_side():
    config = EdgeSourceConfig(sample_rate=16000, channels=2, asr_channel=1, pre_roll_ms=200, tail_ms=300)
    hello_text = server_protocol.build_hello(config, device_id=7)

    parsed = pi_protocol.parse_server_message(hello_text)

    assert isinstance(parsed, pi_protocol.Hello)
    assert parsed.protocol == pi_protocol.PROTOCOL_VERSION
    assert parsed.device_id == 7
    assert parsed.sample_rate == 16000
    assert parsed.channels == 2
    assert parsed.asr_channel == 1
    assert parsed.pre_roll_ms == 200
    assert parsed.tail_ms == 300
    assert parsed.frame_samples == server_protocol.FRAME_SAMPLES


def test_a_server_ping_parses_on_the_pi_side_and_the_pong_it_builds_parses_back():
    import json

    ping_text = json.dumps({"type": pi_protocol.MSG_PING, "id": 3, "server_t_ms": 555})
    parsed_ping = pi_protocol.parse_server_message(ping_text)
    assert isinstance(parsed_ping, pi_protocol.Ping)
    assert parsed_ping.id == 3
    assert parsed_ping.server_t_ms == 555

    pong_text = pi_protocol.pong(parsed_ping.id, parsed_ping.server_t_ms)
    parsed_pong = server_protocol.parse_edge_event(pong_text)
    assert parsed_pong == {"type": server_protocol.MSG_PONG, "id": 3, "server_t_ms": 555}


def test_led_states_match_in_value_and_order():
    assert server_protocol.LED_STATES == pi_protocol.LED_STATES
    assert server_protocol.LED_STATES == ("idle", "listening", "thinking", "replying")


def test_every_server_led_message_parses_on_the_pi_side():
    for state in server_protocol.LED_STATES:
        parsed = pi_protocol.parse_server_message(server_protocol.build_led(state))
        assert parsed == pi_protocol.Led(state)


def test_a_bad_led_message_is_refused_on_both_sides():
    import json

    import pytest

    with pytest.raises(ValueError):
        server_protocol.build_led("bogus")
    for raw in (
        {"type": "led", "state": "bogus"},
        {"type": "led"},
        {"type": "led", "state": 3},
    ):
        with pytest.raises(pi_protocol.ProtocolError):
            pi_protocol.parse_server_message(json.dumps(raw))
