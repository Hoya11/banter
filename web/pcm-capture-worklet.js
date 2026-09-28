const CHUNK_SAMPLES = 2400;

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pending = new Float32Array(CHUNK_SAMPLES);
    this.pendingLength = 0;
    this.sealed = false;
    this.port.onmessage = (event) => {
      if (event.data?.type !== 'flush') return;
      if (!this.sealed) {
        this.sealed = true;
        this.flush();
      }
      this.port.postMessage({ type: 'flushed' });
    };
  }

  emitFullChunk() {
    const samples = this.pending;
    this.pending = new Float32Array(CHUNK_SAMPLES);
    this.pendingLength = 0;
    this.port.postMessage({ type: 'audio', samples }, [samples.buffer]);
  }

  flush() {
    if (!this.pendingLength) return;
    const samples = this.pending.slice(0, this.pendingLength);
    this.pendingLength = 0;
    this.port.postMessage({ type: 'audio', samples }, [samples.buffer]);
  }

  process(inputs) {
    if (this.sealed) return true;
    const input = inputs[0]?.[0];
    if (!input?.length) return true;
    let offset = 0;
    while (offset < input.length) {
      const count = Math.min(
        input.length - offset,
        CHUNK_SAMPLES - this.pendingLength,
      );
      this.pending.set(input.subarray(offset, offset + count), this.pendingLength);
      this.pendingLength += count;
      offset += count;
      if (this.pendingLength === CHUNK_SAMPLES) this.emitFullChunk();
    }
    return true;
  }
}

registerProcessor('pcm-capture-processor', PcmCaptureProcessor);
