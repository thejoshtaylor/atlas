"""The WebRTC transport: Opus in, the same 16 kHz mono PCM16 out.

This is the second implementation of `AudioSource` (transports/base.py),
proved equivalent to `WebSocketAudioSource` (transports/websocket.py) by
`tests/test_transports.py`'s shared expected-bytes constant, not merely
assumed equivalent. Opus decode happens inside `aiortc` and PyAV -- this
module never touches the codec itself (RESEARCH.md "Don't Hand-Roll":
reimplementing DTLS-SRTP or a media decoder is a security liability, not a
scoping win).

Per D-04, the peer connection is built with an explicitly empty ICE server
list: a browser and a server on the same LAN or the same Tailscale network
need host candidates alone, and provisioning a STUN server would be
infrastructure this deployment does not need. `aiortc`'s own default
(`RTCConfiguration()` with no `iceServers` argument) resolves to
`stun.l.google.com:19302` -- confirmed by reading
`aiortc.rtcicetransport.RTCIceGatherer.getDefaultIceServers()` directly this
session (RESEARCH.md Open Question 3 / Assumption A3) -- so the empty list
must be explicit, never left as the library default. Every candidate this
peer connection gathers is logged by type, turning the "host candidates are
enough" assumption (RESEARCH.md Pitfall 6 / Assumption A2, a single
unverified report) into an observation on each real connection rather than
a silent success.

Reply audio and events (the partial transcript, the reply text, the timing
line) go back over one `RTCDataChannel` the browser creates alongside its
offer -- the same three-message shape `WebSocketAudioSource` already sends
over its one socket, just carried on a data channel instead of a raw
WebSocket. This is why the WebRTC path introduces no second signalling
connection beyond the one stateless `POST /webrtc/offer` exchange: the data
channel is negotiated inside the same SDP offer/answer, not a parallel
connection the WebSocket transport would also have to grow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator

from aiortc import (
    MediaStreamError,
    RTCConfiguration,
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
)
from av.audio.resampler import AudioResampler

from spire_voice.transports.base import SourceFormat

logger = logging.getLogger("spire_voice.transports.webrtc")

_TARGET_FORMAT = "s16"
_TARGET_LAYOUT = "mono"
_TARGET_SAMPLE_RATE = 16000

# Matches an SDP `a=candidate:` line and captures its `typ <type>` field --
# e.g. "a=candidate:1 1 udp 2130706431 192.0.2.1 5000 typ host". Parsing the
# negotiated SDP text directly is simpler and more representative of what a
# real peer actually agreed on than walking aiortc's internal ICE-gatherer
# objects, and it needs no access to anything not already returned by
# `pc.localDescription`.
_CANDIDATE_TYPE_RE = re.compile(r"^a=candidate:\S+ \d+ \S+ \d+ \S+ \d+ typ (\S+)", re.MULTILINE)


class WebrtcTransport:
    """Satisfies `AudioSource` over one `aiortc` peer connection.

    `frames()` yields 16 kHz mono PCM16 regardless of what the inbound
    track actually negotiated -- every `av.AudioFrame` the track consumer
    receives is resampled before its bytes reach the queue, because the
    seam's contract is one format and Opus negotiation does not guarantee
    it (see `_frame_to_pcm16`).
    """

    def __init__(self) -> None:
        self._pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._resampler = AudioResampler(
            format=_TARGET_FORMAT, layout=_TARGET_LAYOUT, rate=_TARGET_SAMPLE_RATE
        )
        self._data_channel: RTCDataChannel | None = None
        self._consumer_task: asyncio.Task[None] | None = None

        @self._pc.on("track")
        def _on_track(track: Any) -> None:
            # One long-lived consumer task per track, created once -- never
            # a task per received frame (RESEARCH.md Open Question 3 flags
            # the per-frame-task shape as the leak to avoid).
            if track.kind == "audio" and self._consumer_task is None:
                self._consumer_task = asyncio.create_task(self._consume_track(track))

        @self._pc.on("datachannel")
        def _on_datachannel(channel: RTCDataChannel) -> None:
            self._data_channel = channel

    async def frames(self) -> AsyncIterator[bytes]:
        """Yield 16 kHz mono PCM16 frames until the inbound track ends."""
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk

    async def send_audio(self, chunk: bytes) -> None:
        self._send(chunk, "reply audio chunk")

    async def send_event(self, event: dict[str, Any]) -> None:
        self._send(json.dumps(event), "event")

    def source_format(self) -> SourceFormat:
        """`_frame_to_pcm16` always resamples to 16 kHz mono PCM16, whatever the
        inbound track actually negotiated."""
        return SourceFormat("pcm", _TARGET_SAMPLE_RATE)

    async def close(self) -> None:
        """Close the peer connection.

        T-1-13 accepts the DoS risk of repeated offers on the basis that "a
        peer connection is closed when its turn ends" -- this method is
        what makes that statement true rather than aspirational. The route
        that owns this transport's lifetime (`app.py`) calls it once the
        turn's `run_turn` coroutine finishes, success or failure alike.
        """
        if self._consumer_task is not None:
            self._consumer_task.cancel()
        await self._pc.close()

    def _send(self, payload: bytes | str, what: str) -> None:
        if self._data_channel is None or self._data_channel.readyState != "open":
            logger.warning("dropping %s: WebRTC data channel is not open", what)
            return
        self._data_channel.send(payload)

    async def _consume_track(self, track: Any) -> None:
        """The one long-lived task that drains `track` into `self._queue`."""
        try:
            while True:
                frame = await track.recv()
                for chunk in self._frame_to_pcm16(frame):
                    await self._queue.put(chunk)
        except MediaStreamError:
            pass
        except Exception:
            logger.exception("WebRTC inbound track consumer failed")
        finally:
            await self._queue.put(None)

    def _frame_to_pcm16(self, frame: Any) -> list[bytes]:
        """Resample one `av.AudioFrame` to 16 kHz mono PCM16 raw bytes.

        A resampled frame's `.planes[0]` buffer can be larger than its
        valid sample count (FFmpeg over-allocates for alignment) -- slicing
        to `samples * format.bytes * channels` is what keeps this from
        leaking padding bytes into the audio the turn controller consumes.
        Verified against the installed `av`/`aiortc` versions this session:
        a same-rate, same-layout frame round-trips byte-identical through
        this resampler with this slicing; only the slice, not the resample
        call, is what makes that true.
        """
        chunks: list[bytes] = []
        for resampled in self._resampler.resample(frame):
            valid_length = resampled.samples * resampled.format.bytes * len(resampled.layout.channels)
            chunks.append(bytes(resampled.planes[0])[:valid_length])
        return chunks


async def create_offer_answer(transport: WebrtcTransport, offer: dict[str, str]) -> dict[str, str]:
    """Apply `offer` to `transport`'s peer connection and answer it.

    One stateless call: set the remote description, create and set the
    local answer, and log the type of every ICE candidate the answer
    negotiated -- turning RESEARCH.md's flagged assumption about host-only
    gathering into a recorded observation on this connection.
    """
    pc = transport._pc
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    _log_ice_candidate_types(pc.localDescription.sdp)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


def _log_ice_candidate_types(sdp: str) -> None:
    """Log every gathered ICE candidate's type.

    Per D-04 and RESEARCH.md Pitfall 6: a `host`-typed candidate is the
    expected, silent-success case. Any other type appearing with no STUN or
    TURN server configured is the finding Pitfall 6's single unverified
    report predicted -- logged at `warning` so it surfaces rather than
    passing unnoticed, and recorded in the plan summary either way.
    """
    for candidate_type in _CANDIDATE_TYPE_RE.findall(sdp):
        if candidate_type == "host":
            logger.info("gathered ICE candidate type=%s", candidate_type)
        else:
            logger.warning("gathered non-host ICE candidate type=%s", candidate_type)
