// GPU timestamp timers, one begin/end pair per pass. Readbacks rotate through a
// small pool so the CPU never waits on the GPU; a frame without a free buffer
// simply goes untimed. Without the timestamp-query feature only the extra
// readback (visible splat count) runs.

const POOL = 3;
const EXTRA_BYTES = 16;

export class GpuTimers {
  constructor(device, names) {
    this.device = device;
    this.names = names;
    this.enabled = device.features.has('timestamp-query');
    this.ms = Object.fromEntries(names.map(n => [n, 0]));
    this.extra = new Uint32Array(4);
    const queryBytes = names.length * 2 * 8;
    this.extraOffset = Math.ceil(queryBytes / 256) * 256;
    if (this.enabled) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: names.length * 2 });
      this.resolveBuf = device.createBuffer({ size: queryBytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    this.free = Array.from({ length: POOL }, () => device.createBuffer({
      size: this.extraOffset + EXTRA_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }));
    this.current = null;
  }

  // Call before encoding; decides whether this frame is timed.
  begin() {
    this.current = this.free.pop() ?? null;
    return this.current !== null;
  }

  writes(name) {
    if (!this.enabled || !this.current) return undefined;
    const i = this.names.indexOf(name);
    return { querySet: this.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
  }

  // Encode the resolve and copies; `extraSrc` is a 16 B buffer to read back too.
  end(encoder, extraSrc) {
    const buf = this.current;
    if (!buf) return null;
    if (this.enabled) {
      encoder.resolveQuerySet(this.querySet, 0, this.names.length * 2, this.resolveBuf, 0);
      encoder.copyBufferToBuffer(this.resolveBuf, 0, buf, 0, this.names.length * 16);
    }
    if (extraSrc) encoder.copyBufferToBuffer(extraSrc, 0, buf, this.extraOffset, EXTRA_BYTES);
    this.current = null;
    return buf;
  }

  // Call after submit with what end() returned.
  collect(buf, onUpdate) {
    if (!buf) return;
    buf.mapAsync(GPUMapMode.READ).then(() => {
      if (this.enabled) {
        const t = new BigInt64Array(buf.getMappedRange(0, this.names.length * 16));
        this.names.forEach((n, i) => {
          const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6;
          // Exponential average; drop the occasional bogus (negative) sample.
          if (ms >= 0 && ms < 1000) this.ms[n] = this.ms[n] ? this.ms[n] * 0.9 + ms * 0.1 : ms;
        });
      }
      this.extra.set(new Uint32Array(buf.getMappedRange(this.extraOffset, EXTRA_BYTES)));
      buf.unmap();
      this.free.push(buf);
      onUpdate?.();
    }).catch(() => { /* device lost; the main loop reports it */ });
  }
}
