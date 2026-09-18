// AudioWorklet processor: converts Float32 capture frames to 16-bit PCM16
// and posts the result by transfer, not by copy, so the main thread never
// shares a buffer that this thread might still be writing.
//
// This runs off the main thread, unlike the older, now-deprecated
// main-thread audio-processing node it replaces.
class PCMWorklet extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0][0];
    if (input) {
      const pcm16 = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const s = Math.max(-1, Math.min(1, input[i]));
        pcm16[i] = s < 0 ? s * 32768 : s * 32767;
      }
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
