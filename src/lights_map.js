// Opacity maps for every light (runtime/wgsl/lights_map.wgsl; ?lights=unified, David 2026-10-29): a layer per light and
// part, splatted from the light's root centre, re-splatted only when the part or the light moves (or a light is
// switched); each frame something changed, lm_lookup writes every receiver's per-light shadow slots. Replaces the
// unified walk (walk_emit + ema_light, archived 2026-10-29).

const WG = 64;
const B = GPUBufferUsage;
const PARAMS_BYTES = 96;   // lights_map.wgsl LMParams
const BINDS = {
  lm_clear: [70, 72, 73],
  lm_splat: [0, 1, 2, 4, 5, 35, 52, 53, 70, 71, 72, 73],
  lm_resolve: [70, 72, 73, 74],
  lm_down: [70, 72, 74, 75],
  lm_lookup: [0, 4, 5, 6, 52, 70, 71, 74],
};

export class LightsMap {
  constructor(device, sources, lighting) {
    this.device = device;
    this.lighting = lighting;
    const lights = lighting.lightSet.lights;
    this.L = lights.length;
    this.P = lighting.partCount;
    this.layers = this.L * this.P;
    // the largest power-of-two resolution whose maps stay within ~120 MB per buffer (WebGPU's binding limit is 128 MB)
    let res = 512;
    while (res > 32 && (this.layers * res * res * 12 > 120e6 || this.layers * res * res * 16 * 4 / 3 > 120e6)) res >>= 1;
    this.res = res;
    this.levels = [];
    let off = 0;
    for (let k = res; k >= 1; k >>= 1) { this.levels.push([off, k]); off += k * k; }
    this.layerTexels = off;
    const buf = (label, size, usage = B.STORAGE) => device.createBuffer({ label, size: Math.max(16, size), usage: usage | B.COPY_DST });
    this.ub = buf('lights map params', PARAMS_BYTES, B.UNIFORM);
    this.partBuf = buf('lights map light info', this.L * 16);   // lights_map.wgsl LightInfo, written in encode()
    this.dirtyBuf = buf('lights map dirty layers', this.layers * 4);
    this.dirty = new Uint32Array(this.layers);
    this.acc = buf('lights map (fixed point)', this.layers * res * res * 12);
    this.mip = buf('lights map mips', this.layers * off * 16);
    this.lv = this.levels.map(([, rk], k) => { const b = buf(`lights map level ${k}`, 16, B.UNIFORM); device.queue.writeBuffer(b, 0, new Uint32Array([k, rk, 0, 0])); return b; });
    const Lb = lighting.buffers;
    const res_ = { 0: Lb.uniform, 1: lighting.blur ? Lb.gaussians_blur : Lb.gaussians, 2: lighting.blur ? Lb.nodes_blur : Lb.nodes,
      4: Lb.parts, 5: Lb.lamps, 6: Lb.lamp_state, 35: Lb.node_parent, 52: Lb.gaussians, 53: Lb.nodes, 70: this.ub,
      71: this.partBuf, 72: this.dirtyBuf, 73: this.acc, 74: this.mip };
    const module = device.createShaderModule({ label: 'lights_map', code: sources.common + '\n' + sources.lights_map });
    this.k = {};
    for (const [entry, binds] of Object.entries(BINDS)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      this.k[entry] = { pipe, groups: (entry === 'lm_down' ? this.lv : [null]).map(lvb => device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: binds.map(b => ({ binding: b, resource: { buffer: b === 75 ? lvb : res_[b] } })) })) };
    }
    this.lightKey = null;
    this.first = true;
  }

  run(encoder, entry, threads, k = 0) {
    const pass = encoder.beginComputePass({ label: entry });
    pass.setPipeline(this.k[entry].pipe); pass.setBindGroup(0, this.k[entry].groups[k]);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(threads / WG))); pass.end();
  }

  // movedMask: parts that moved this frame (lighting.js; with motion blur, last frame's movers too)
  encode(encoder, movedMask) {
    const lights = this.lighting.lightSet.lights;
    const key = lights.map(l => `${l.on}:${l.gain}`).join();
    const switched = key !== this.lightKey;
    this.lightKey = key;
    this.dirty.fill(0);
    let any = false;
    lights.forEach((l, i) => {
      const lightMoved = l.part < 32 && ((movedMask >>> l.part) & 1);
      for (let p = 0; p < this.P; p++) {
        if (this.first || switched || lightMoved || (p < 32 && ((movedMask >>> p) & 1))) { this.dirty[i * this.P + p] = 1; any = true; }
      }
    });
    this.first = false;
    if (!any) return;
    const u = new Uint32Array(PARAMS_BYTES / 4), f = new Float32Array(u.buffer);
    u.set([this.L, this.P, this.res, this.levels.length, this.layerTexels, this.lighting.nodeCount], 0);
    f[6] = this.lighting.blur ? this.lighting.blurShutter : 0;
    f[7] = this.lighting.blurCut ?? 0;
    this.levels.forEach(([o], k) => { u[8 + k] = o; });
    this.device.queue.writeBuffer(this.ub, 0, u);
    this.device.queue.writeBuffer(this.dirtyBuf, 0, this.dirty);
    // each light's part, and a one-sided light's world emitting axis (its rest tree's agreed normal, rotated by its part)
    const info = new Float32Array(this.L * 4), iu = new Uint32Array(info.buffer), xf = this.lighting.partXf;
    lights.forEach((l, i) => {
      iu[i * 4 + 3] = l.part;
      const n = l.tree?.nrm;
      if (!n || !xf) return;
      const m = xf.subarray(l.part * 32, l.part * 32 + 12);
      info.set([m[0] * n[0] + m[1] * n[1] + m[2] * n[2], m[4] * n[0] + m[5] * n[1] + m[6] * n[2], m[8] * n[0] + m[9] * n[1] + m[10] * n[2]], i * 4);
      iu[i * 4 + 3] = l.part | (1 << 16);
    });
    this.device.queue.writeBuffer(this.partBuf, 0, info);
    const n = this.res * this.res * 12;
    for (let l = 0; l < this.layers; l++) if (this.dirty[l]) encoder.clearBuffer(this.acc, l * n, n);
    this.run(encoder, 'lm_splat', this.lighting.nodeCount);
    this.run(encoder, 'lm_resolve', this.layers * this.res * this.res);
    for (let k = 1; k < this.levels.length; k++) this.run(encoder, 'lm_down', this.layers * this.levels[k][1] ** 2, k);
    this.run(encoder, 'lm_lookup', this.lighting.count);
  }
}
