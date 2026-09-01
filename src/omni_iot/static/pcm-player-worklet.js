class PcmStreamPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.ended = false;
    this.playingReported = false;
    this.drainedReported = false;
    this.port.onmessage = (event) => {
      if (event.data?.type === "push" && event.data.samples) {
        this.queue.push(event.data.samples);
      } else if (event.data?.type === "end") {
        this.ended = true;
      } else if (event.data?.type === "reset") {
        this.queue = [];
        this.offset = 0;
        this.ended = true;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0]?.[0];
    if (!output) {
      return true;
    }
    output.fill(0);
    let outputOffset = 0;
    while (outputOffset < output.length && this.queue.length > 0) {
      const current = this.queue[0];
      const count = Math.min(output.length - outputOffset, current.length - this.offset);
      output.set(current.subarray(this.offset, this.offset + count), outputOffset);
      outputOffset += count;
      this.offset += count;
      if (this.offset >= current.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (outputOffset > 0 && !this.playingReported) {
      this.playingReported = true;
      this.port.postMessage({ type: "playing" });
    }
    if (this.ended && this.queue.length === 0 && !this.drainedReported) {
      this.drainedReported = true;
      this.port.postMessage({ type: "drained" });
    }
    return true;
  }
}

registerProcessor("pcm-stream-player", PcmStreamPlayerProcessor);
