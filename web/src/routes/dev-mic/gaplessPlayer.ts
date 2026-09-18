// Multi-buffer scheduling against `AudioContext.currentTime` -- an
// `AudioBufferSourceNode` can't be appended to once assigned, so gapless
// playback means scheduling each new chunk right after the last one ends.
// Ported unchanged from `web/public/dev-mic/webrtc.js`'s own `GaplessPlayer`
// (carried into the SPA as a route behind admin authentication, D-19); the
// logic is untouched, only the file it lives in and the language it is
// written in.
export class GaplessPlayer {
  private readonly ctx: AudioContext
  private readonly sampleRate: number
  private nextStartTime = 0

  constructor(ctx: AudioContext, sampleRate: number) {
    this.ctx = ctx
    this.sampleRate = sampleRate
  }

  playChunk(int16Array: Int16Array): void {
    const float32 = Float32Array.from(int16Array, (s) => (s < 0 ? s / 32768 : s / 32767))
    const buffer = this.ctx.createBuffer(1, float32.length, this.sampleRate)
    buffer.copyToChannel(float32, 0)
    const src = this.ctx.createBufferSource()
    src.buffer = buffer
    src.connect(this.ctx.destination)
    const startAt = Math.max(this.ctx.currentTime, this.nextStartTime)
    src.start(startAt)
    this.nextStartTime = startAt + buffer.duration
  }
}
