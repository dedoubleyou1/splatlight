// One-frame shadows from distant lights (runtime/wgsl/sun_map.wgsl; David, 2026-10-29). Every part gets orthographic
// opacity maps per distant light: a small part one map fitted to its bounding sphere, a large one (the hill's ground and
// grass) cascades centred ahead of the camera plus a map of the whole part. A layer re-splats only when its part moves,
// the light moves (the sun slider), a cascade re-centres (the camera moved a quarter of it), or on load; every frame
// anything changed, sm_lookup writes env_T for every receiver. No per-receiver walk (env_shadow, archived
// 2026-10-29). Environment (environment.js) owns the schedule; this owns the maps.

const WG = 64;
const B = GPUBufferUsage;
const PARAMS_BYTES = 432;   // sun_map.wgsl SunParams
const LAYER_BYTES = 48;     // SunLayer
const DESC_BYTES = 32;      // LayerDesc
const CASCADES = [8, 32];   // half sizes (m) of the camera cascades, for parts with a larger bound
const BINDS = {
  sm_frame: [4, 60, 61, 66],
  sm_clear: [60, 62, 66],
  sm_splat: [0, 1, 2, 4, 35, 52, 53, 60, 61, 62, 64, 66],
  sm_resolve: [60, 62, 63, 66],
  sm_down: [60, 63, 65, 66],
  sm_lookup: [0, 4, 41, 52, 60, 61, 63, 66],
};

export class SunMap {
  // env: Environment (its base lights); returns with .ok false when there is nothing to map
  constructor(device, sources, lighting, env, asset) {
    this.device = device;
    this.lighting = lighting;
    this.env = env;
    const q = new URLSearchParams(globalThis.location?.search ?? '');
    const parts = asset.header.parts;
    this.K = Math.min(env.base.lights.length, 4);
    this.P = Math.min(parts.length, 32);
    this.ok = this.K > 0 && sources.sun_map;
    if (!this.ok) return;
    // the parts' rest bounds (centre, radius), for the layer plan and the penumbra margin
    this.bound = parts.slice(0, this.P).map((p, i) => Array.from(new Float32Array(asset.raw.parts, i * 128 + 96, 4)));
    const ext = Math.max(...this.bound.map(b => Math.hypot(b[0], b[1], b[2]) + b[3]));
    const sigMax = Math.max(...env.base.lights.slice(0, this.K).map(L => L.std));
    this.margin = 3 * sigMax * 2 * ext;
    // layers, ordered by part, then light, then finest first (sm_lookup uses the first one a receiver lies in)
    this.desc = [];
    this.partRange = [];
    for (let p = 0; p < this.P; p++) {
      const first = this.desc.length, big = this.bound[p][3] > CASCADES[0] && parts[p].mobility === 'static';   // movers: one fitted map
      if (parts[p].gaussians?.[1] > parts[p].gaussians?.[0]) {
        for (let k = 0; k < this.K; k++) {
          if (big) CASCADES.forEach((h, c) => { if (h < this.bound[p][3] * 1.5) this.desc.push({ p, k, kind: 1, c, half: h, center: null }); });
          this.desc.push({ p, k, kind: 0 });
        }
      }
      this.partRange.push([first, this.desc.length - first]);
    }
    this.layers = this.desc.length;
    let res = Math.max(64, Number(q.get('sunmapres')) || 512);
    while (res > 64 && (this.layers * res * res * 12 > 120e6 || this.layers * res * res * 16 * 4 / 3 > 120e6)) res >>= 1;
    this.res = res;
    this.levels = [];
    let off = 0;
    for (let k = this.res; k >= 1; k >>= 1) { this.levels.push([off, k]); off += k * k; }
    this.layerTexels = off;
    // each part's runs of record indices (leaves and internal nodes, for the motion-blur cut)
    const nu = new Uint32Array(asset.raw.nodes), N = nu.length / 24;
    this.runs = Array.from({ length: this.P }, () => []);
    for (let i = 0; i < N;) {
      const p = nu[i * 24 + 15];
      let j = i + 1;
      while (j < N && nu[j * 24 + 15] === p) j++;
      if (p < this.P) this.runs[p].push([i, j]);
      i = j;
    }
    this.maxRuns = Math.max(1, this.runs.reduce((n, r) => n + r.length, 0));
    const buf = (label, size, usage = B.STORAGE) => device.createBuffer({ label, size: Math.max(16, size), usage: usage | B.COPY_DST });
    this.params = new ArrayBuffer(PARAMS_BYTES);
    this.ub = buf('sun map params', PARAMS_BYTES, B.UNIFORM);
    this.layerBuf = buf('sun map layers', this.layers * LAYER_BYTES);
    this.descBuf = buf('sun map layer plan', this.layers * DESC_BYTES);
    this.acc = buf('sun map (fixed point)', this.layers * this.res * this.res * 12);
    this.mip = buf('sun map mips', this.layers * off * 16);
    this.rangeBuf = buf('sun map ranges', this.maxRuns * 16);
    this.lv = this.levels.map(([, rk], k) => { const b = buf(`sun map level ${k}`, 16, B.UNIFORM); device.queue.writeBuffer(b, 0, new Uint32Array([k, rk, 0, 0])); return b; });
    const L = lighting.buffers;
    const res_ = { 0: L.uniform, 1: lighting.blur ? L.gaussians_blur : L.gaussians, 2: lighting.blur ? L.nodes_blur : L.nodes,
      4: L.parts, 35: L.node_parent, 41: L.env_T, 52: L.gaussians, 53: L.nodes, 60: this.ub, 61: this.layerBuf,
      62: this.acc, 63: this.mip, 64: this.rangeBuf, 66: this.descBuf };
    const module = device.createShaderModule({ label: 'sun_map', code: sources.common + '\n' + sources.sun_map });
    this.k = {};
    for (const [entry, binds] of Object.entries(BINDS)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      const groups = (entry === 'sm_down' ? this.lv : [null]).map(lvb => device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
        entries: binds.map(b => ({ binding: b, resource: { buffer: b === 65 ? lvb : res_[b] } })) }));
      this.k[entry] = { pipe, groups };
    }
    this.lightKey = null;
  }

  // threads past 65535 x 64 spill into y (sm_clear / sm_resolve / sm_down index gid.y * nw.x * 64 + gid.x)
  run(encoder, entry, threads, k = 0) {
    const pass = encoder.beginComputePass({ label: entry });
    pass.setPipeline(this.k[entry].pipe); pass.setBindGroup(0, this.k[entry].groups[k]);
    const g = Math.max(1, Math.ceil(threads / WG)), gy = Math.ceil(g / 65535);
    pass.dispatchWorkgroups(Math.ceil(g / gy), gy); pass.end();
  }

  // movedMask: parts that moved this frame; writes env_T for every receiver when anything changed
  encode(encoder, movedMask) {
    const lights = this.env.lights();   // current directions (the sun follows the UI)
    const key = lights.map(L => [...L.dir, L.std].join()).join('|') + (this.env.on ? '' : 'off');
    const lightsChanged = key !== this.lightKey;
    this.lightKey = key;
    // cascade centres: a point ahead of the camera at the cascade's half size, across each light; re-centred (and
    // re-splatted) once it drifts a quarter of the cascade, so the camera moving a little costs nothing
    const eye = this.lighting.camEye, fwd = this.lighting.camFwd;
    let any = false;
    const dirtyFlags = this.desc.map(d => {
      let dirty = lightsChanged || (d.p < 32 && ((movedMask >>> d.p) & 1) === 1);
      if (d.kind === 1) {
        const L = lights[d.k]?.dir ?? [0, 0, 1];
        const t1 = basisK(L), t2 = cross(L, t1);
        const h = d.half, at = eye && fwd ? eye.map((v, i) => v + fwd[i] * h * 0.75) : this.bound[d.p].slice(0, 3);
        const c = [dot(at, t1), dot(at, t2)];
        if (!d.center || lightsChanged || Math.hypot(c[0] - d.center[0], c[1] - d.center[1]) > h / 4) { d.center = c; dirty = true; }
      }
      if (dirty) any = true;
      return dirty;
    });
    if (!any) return;
    const descU = new Uint32Array(this.layers * 8), descF = new Float32Array(descU.buffer);
    this.desc.forEach((d, l) => {
      descU.set([d.p, d.k, d.kind, dirtyFlags[l] ? 1 : 0], l * 8);
      if (d.kind === 1) descF.set([d.center[0], d.center[1], d.half], l * 8 + 4);
    });
    this.device.queue.writeBuffer(this.descBuf, 0, descU);
    const f = new Float32Array(this.params), u = new Uint32Array(this.params);
    const list = [];
    let threads = 0;
    for (let p = 0; p < this.P; p++) {
      const [a0, n] = this.partRange[p];
      if (!dirtyFlags.slice(a0, a0 + n).some(Boolean)) continue;
      for (const [a, b] of this.runs[p]) { list.push(a, b, threads, p); threads += b - a; }
    }
    u.set([this.layers, this.res, this.levels.length, this.layerTexels, this.K, this.P, 0, list.length / 4], 0);
    f.set([this.margin, this.lighting.blur ? this.lighting.blurShutter : 0, this.lighting.blurCut ?? 0], 8);
    u[11] = this.lighting.nodeCount;
    for (let k = 0; k < 4; k++) {
      const L = lights[k], cast = this.env.base.lights[k]?.shadows;
      f.set(L ? [...L.dir, cast && this.env.on ? L.std : -1] : [0, 0, 1, -1], 12 + 4 * k);
    }
    this.partRange.forEach(([a, n], p) => { u[28 + 2 * p] = a; u[28 + 2 * p + 1] = n; });
    this.levels.forEach(([o], k) => { u[92 + k] = o; });
    this.device.queue.writeBuffer(this.ub, 0, this.params);
    this.run(encoder, 'sm_frame', this.layers);
    if (list.length) this.device.queue.writeBuffer(this.rangeBuf, 0, new Uint32Array(list));
    const n = this.res * this.res * 12;
    dirtyFlags.forEach((d, l) => { if (d) encoder.clearBuffer(this.acc, l * n, n); });
    if (threads) this.run(encoder, 'sm_splat', threads);
    this.run(encoder, 'sm_resolve', this.layers * this.res * this.res);
    for (let k = 1; k < this.levels.length; k++) this.run(encoder, 'sm_down', this.layers * this.levels[k][1] ** 2, k);
    this.run(encoder, 'sm_lookup', this.lighting.count);
  }
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
// common.wgsl basis_k: a unit vector perpendicular to w (the same one the shaders build)
function basisK(w) {
  const a = Math.abs(w[2]) >= 0.999 ? [1, 0, 0] : [0, 0, 1];
  const c = cross(a, w), l = Math.hypot(...c);
  return c.map(v => v / l);
}
