// One-frame shadows from distant lights for movable parts (runtime/wgsl/sun_map.wgsl; David, 2026-10-29). The sun's
// walk (environment.wgsl env_shadow) then covers static occluders only, refreshed in slices when the light moves; each
// movable part gets an orthographic opacity map per distant light, re-splatted when it or the light moves, and every
// frame anything changed sm_lookup sets env_T = walked static T x the movable parts' map T. ?sunmap=0: the old walk of
// every occluder. Environment (environment.js) owns the schedule; this owns the maps.

const WG = 64;
const B = GPUBufferUsage;
const PARAMS_BYTES = 240;   // sun_map.wgsl SunParams
const LAYER_BYTES = 48;     // SunLayer
const BINDS = {
  sm_frame: [4, 60, 61],
  sm_clear: [60, 62],
  sm_splat: [0, 1, 2, 4, 35, 52, 53, 60, 61, 62, 64],
  sm_resolve: [60, 62, 63],
  sm_down: [60, 63, 65],
  sm_lookup: [0, 4, 41, 52, 60, 61, 63],
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
    const movable = parts.map((p, i) => i).filter(i => parts[i].mobility !== 'static' && parts[i].gaussians?.[1] > parts[i].gaussians?.[0]);
    this.M = Math.min(movable.length, 16, Math.floor(32 / Math.max(this.K, 1)));
    this.ok = q.get('sunmap') !== '0' && this.K > 0 && this.M > 0 && sources.sun_map;
    if (!this.ok) return;
    this.movable = movable.slice(0, this.M);
    this.layers = this.M * this.K;
    this.res = Math.max(64, Number(q.get('sunmapres')) || 512);
    this.levels = [];
    let off = 0;
    for (let k = this.res; k >= 1; k >>= 1) { this.levels.push([off, k]); off += k * k; }
    this.layerTexels = off;
    // penumbrae reach past a part by up to ~3 sigma x the scene's depth
    const ext = Math.max(...parts.map((p, i) => {
      const b = new Float32Array(asset.raw.parts, i * 128 + 96, 4);
      return Math.hypot(b[0], b[1], b[2]) + b[3];
    }));
    const sigMax = Math.max(...env.base.lights.slice(0, this.K).map(L => L.std));
    this.margin = 3 * sigMax * 2 * ext;
    // each movable part's runs of record indices (leaves and internal nodes), as lighting.js blur_records
    const G = lighting.count, nu = new Uint32Array(asset.raw.nodes), N = nu.length / 24;
    this.runs = this.movable.map(() => []);
    for (let i = 0; i < N;) {
      const p = nu[i * 24 + 15];
      let j = i + 1;
      while (j < N && nu[j * 24 + 15] === p) j++;
      const m = this.movable.indexOf(p);
      if (m >= 0) this.runs[m].push([i, j]);
      i = j;
    }
    this.maxRuns = Math.max(1, this.runs.reduce((n, r) => n + r.length, 0));
    const buf = (label, size, usage = B.STORAGE) => device.createBuffer({ label, size: Math.max(16, size), usage: usage | B.COPY_DST });
    this.params = new ArrayBuffer(PARAMS_BYTES);
    this.ub = buf('sun map params', PARAMS_BYTES, B.UNIFORM);
    this.layerBuf = buf('sun map layers', this.layers * LAYER_BYTES);
    this.acc = buf('sun map (fixed point)', this.layers * this.res * this.res * 12);
    this.mip = buf('sun map mips', this.layers * off * 16);
    this.rangeBuf = buf('sun map ranges', this.maxRuns * 16);
    this.lv = this.levels.map(([, rk], k) => { const b = buf(`sun map level ${k}`, 16, B.UNIFORM); device.queue.writeBuffer(b, 0, new Uint32Array([k, rk, 0, 0])); return b; });
    const L = lighting.buffers;
    const res = { 0: L.uniform, 1: lighting.blur ? L.gaussians_blur : L.gaussians, 2: lighting.blur ? L.nodes_blur : L.nodes,
      4: L.parts, 35: L.node_parent, 41: L.env_T, 52: L.gaussians, 53: L.nodes, 60: this.ub, 61: this.layerBuf,
      62: this.acc, 63: this.mip, 64: this.rangeBuf };
    const module = device.createShaderModule({ label: 'sun_map', code: sources.common + '\n' + sources.sun_map });
    this.k = {};
    for (const [entry, binds] of Object.entries(BINDS)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      const groups = (entry === 'sm_down' ? this.lv : [null]).map(lvb => device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
        entries: binds.map(b => ({ binding: b, resource: { buffer: b === 65 ? lvb : res[b] } })) }));
      this.k[entry] = { pipe, groups };
    }
    this.lightKey = null;
  }

  run(encoder, entry, threads, k = 0) {
    const pass = encoder.beginComputePass({ label: entry });
    pass.setPipeline(this.k[entry].pipe); pass.setBindGroup(0, this.k[entry].groups[k]);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(threads / WG))); pass.end();
  }

  // movedMask: parts that moved this frame; staticChanged: the walked static T changed (a slice ran);
  // returns nothing; writes env_T for every receiver when anything changed
  encode(encoder, movedMask, staticChanged) {
    const lights = this.env.lights();   // current directions (the sun follows the UI)
    const key = lights.map(L => [...L.dir, L.std].join()).join('|') + (this.env.on ? '' : 'off');
    const lightsChanged = key !== this.lightKey;
    this.lightKey = key;
    let dirty = 0;
    this.movable.forEach((p, m) => {
      if (lightsChanged || (p < 32 && (movedMask >>> p) & 1)) for (let k = 0; k < this.K; k++) dirty |= 1 << (m * this.K + k);
    });
    dirty >>>= 0;
    if (!dirty && !staticChanged) return;
    const f = new Float32Array(this.params), u = new Uint32Array(this.params);
    const list = [];
    let threads = 0;
    this.movable.forEach((p, m) => {
      if (!((dirty >>> (m * this.K)) & ((1 << this.K) - 1))) return;
      for (const [a, b] of this.runs[m]) { list.push(a, b, threads, m); threads += b - a; }
    });
    u.set([this.layers, this.res, this.levels.length, this.layerTexels, this.K, this.M, dirty, list.length / 4], 0);
    f.set([this.margin, this.lighting.blur ? this.lighting.blurShutter : 0, this.lighting.blurCut ?? 0], 8);
    u[11] = this.lighting.nodeCount;
    for (let k = 0; k < 4; k++) {
      const L = lights[k], cast = this.env.base.lights[k]?.shadows;
      f.set(L ? [...L.dir, cast && this.env.on ? L.std : -1] : [0, 0, 1, -1], 12 + 4 * k);
    }
    this.movable.forEach((p, m) => { u[28 + m] = p; });
    this.levels.forEach(([o], k) => { u[44 + k] = o; });
    this.device.queue.writeBuffer(this.ub, 0, this.params);
    this.run(encoder, 'sm_frame', this.layers);
    if (dirty) {
      if (list.length) this.device.queue.writeBuffer(this.rangeBuf, 0, new Uint32Array(list));
      const n = this.res * this.res * 12;
      for (let l = 0; l < this.layers; l++) if ((dirty >>> l) & 1) encoder.clearBuffer(this.acc, l * n, n);
      if (threads) this.run(encoder, 'sm_splat', threads);
      this.run(encoder, 'sm_resolve', this.layers * this.res * this.res);
      for (let k = 1; k < this.levels.length; k++) this.run(encoder, 'sm_down', this.layers * this.levels[k][1] ** 2, k);
    }
    this.run(encoder, 'sm_lookup', this.lighting.count);
  }
}
