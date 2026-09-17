// The WebRTC half of the browser transport switch.
//
// Same microphone constraints, same partial-transcript/reply-text/gapless-
// playback renderers as the WebSocket path in index.html -- this file only
// supplies a different way to move the bytes. No STUN or TURN server is
// configured here either, per D-04: host candidates on the same LAN or the
// same Tailscale network are enough, and this dev harness never reaches a
// browser outside the house.
//
// Reply audio and events arrive over one RTCDataChannel this file creates
// alongside the offer, so the signalling exchange stays the single
// stateless POST /webrtc/offer the server answers -- no second persistent
// connection the WebSocket path would also have to grow.

async function startWebrtcListening() {
  const mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
      sampleRate: 16000,
    },
  });

  const pc = new RTCPeerConnection({ iceServers: [] });
  const dataChannel = pc.createDataChannel("events");
  dataChannel.binaryType = "arraybuffer";
  dataChannel.onmessage = handleTransportMessage;

  // GaplessPlayer is declared once, in index.html's inline script, and
  // shared by both transports -- this file does not define its own copy.
  window.player = new GaplessPlayer(new AudioContext(), 24000);

  for (const track of mediaStream.getAudioTracks()) {
    pc.addTrack(track, mediaStream);
  }

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitForIceGatheringComplete(pc);

  const response = await fetch("/webrtc/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
  });
  const answer = await response.json();
  await pc.setRemoteDescription(answer);

  return {
    close: () => {
      dataChannel.close();
      pc.close();
      mediaStream.getTracks().forEach((track) => track.stop());
    },
  };
}

// A single non-trickle exchange: wait for gathering to finish so the one
// POST already carries every candidate, rather than opening a second
// channel to trickle candidates in after the fact.
function waitForIceGatheringComplete(pc) {
  if (pc.iceGatheringState === "complete") {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    function checkState() {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", checkState);
        resolve();
      }
    }
    pc.addEventListener("icegatheringstatechange", checkState);
  });
}

window.startWebrtcListening = startWebrtcListening;
