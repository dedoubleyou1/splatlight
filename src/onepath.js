// One-path lighting (M45 E9; experiments/M45_unified/E9_gpu): ?lighting=onepath replaces the lighting kernels with
// one gather over the scene's own Gaussians (lamp, emitters and bounce alike): per receiver cluster (10 cm, one part)
// a light cut, one sender leaf per cut node, one shared bundle ray each (theta-16 walk), per-member geometry; the
// direction-independent B converges through a per-Gaussian count history. Parts move by their transforms (boxes, the
// lamp): once the scene has been still a while the history becomes a cache, and while parts move each frame estimates
// only the change against it (control variate), spending the samples on the clusters the motion touches. Needs the asset's onepath.bin / onepath.json (bake_onepath.py) and a
// true-density V grid (cornell_sdf_room_plain). Display fill (E2) on by default, ?fill=0 off.

const WG = 64;
const SLOT = 5120;               // uniform slot per dispatch (>= the Params struct, 256-aligned)
const PSIZE = 4944;   // + soft (16): soft cluster membership, ?soft=N
// was 4928;   // + sched (16) + view-projection (64): progressive scheduling, ?budget=N
// was 4848;              // onepath.wgsl Params: 16 scalars, roots, pbound, dtot, mdelta, xf / xfp / xfc, masks, knobs, env
const ENV_BIT = 0x80000000;      // masks: the environment changed
const NBINS = 64;                // sky bins (M45 E5: 64 Fibonacci bins, sampled with jitter over each bin)
const SKY_RAYS = 8;              // sky rays per cluster and frame (shared T over the far parts, like the picks)
const SUN_RAYS = 1;              // sun rays per receiver and frame (its own walk: the sun's shadow edge is sharp)

// SPEC 2.9 SH order (envlib.sh_basis)
function shBasis(x, y, z) {
  return [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x,
    1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y),
    0.590044 * y * (3 * x * x - y * y), 2.890611 * x * y * z, 0.457046 * y * (5 * z * z - 1),
    0.373176 * z * (5 * z * z - 3), 0.457046 * x * (5 * z * z - 1), 1.445306 * z * (x * x - y * y),
    0.590044 * x * (x * x - 3 * y * y)];
}
function fibonacci(n) {
  const out = [];
  for (let k = 0; k < n; k++) {
    const z = 1 - 2 * (k + 0.5) / n, ph = Math.PI * (1 + Math.sqrt(5)) * (k + 0.5), r = Math.sqrt(1 - z * z);
    out.push([r * Math.cos(ph), r * Math.sin(ph), z]);
  }
  return out;
}
// NBINS Fibonacci bins: each bin's L Omega = the SH sky integrated over its Voronoi cell (dense samples), so the bins
// hold the sky's total (envlib.sky_bins); returns Float32Array [bins (dir, om) | brad (rgb, luminance CDF)]
function skyBins(sh, dense = 40000) {
  const w = fibonacci(NBINS), d = fibonacci(dense);
  const rad = w.map(() => [0, 0, 0]), om = new Array(NBINS).fill(0);
  for (const [x, y, z] of d) {
    let best = -2, bi = 0;
    for (let b = 0; b < NBINS; b++) { const c = x * w[b][0] + y * w[b][1] + z * w[b][2]; if (c > best) { best = c; bi = b; } }
    const Y = shBasis(x, y, z);
    for (let ch = 0; ch < 3; ch++) {
      let L = 0; sh.forEach((c, k) => { L += Y[k] * c[ch]; });
      rad[bi][ch] += Math.max(L, 0) * 4 * Math.PI / dense;
    }
    om[bi] += 4 * Math.PI / dense;
  }
  const lum = rad.map(r => 0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2]);
  const tot = lum.reduce((a, b) => a + b, 0) || 1;
  const f = new Float32Array(NBINS * 8);
  let cdf = 0;
  for (let b = 0; b < NBINS; b++) {
    cdf += lum[b] / tot;
    f.set([...w[b], om[b]], 4 * b); f.set([...rad[b], b === NBINS - 1 ? 1 : cdf], 4 * (NBINS + b));
  }
  return f;
}
const IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
const KMAX = 64;                 // onepath.wgsl: picks per cluster, at most

// soft membership: each Gaussian's n nearest clusters on its own part (rest frame) and their weights, the Gaussian kernel
// of the distance to the cluster centroid (std sigma) minus that of the (n + 1)-th nearest, so a cluster's weight reaches
// 0 as it leaves the set and the blend stays continuous. The Gaussian's own cluster is always slot 0 (gather_soft skips
// the receiver when it was not gathered). Returns Uint32Array [G x (4 clusters, 4 weights as f32 bits)]
function softNeighbours(meta, lists, gaussBytes, n, sigma) {
  const gf = new Float32Array(gaussBytes), gu = new Uint32Array(gaussBytes);
  const G = meta.G, cell = meta.cluster;
  const cen = new Float32Array(meta.ncl * 3), cpart = new Uint32Array(meta.ncl), own = new Int32Array(G).fill(-1);
  const grid = new Map();
  for (let c = 0; c < meta.ncl; c++) {
    const start = lists[meta.cl_off + 2 * c], count = lists[meta.cl_off + 2 * c + 1];
    let x = 0, y = 0, z = 0;
    for (let m = 0; m < count; m++) {
      const g = lists[start + m]; own[g] = c;
      x += gf[g * 16]; y += gf[g * 16 + 1]; z += gf[g * 16 + 2];
    }
    cen.set([x / count, y / count, z / count], 3 * c);
    cpart[c] = gu[lists[start] * 16 + 11] & 0xffff;
    const key = `${cpart[c]},${Math.floor(x / count / cell)},${Math.floor(y / count / cell)},${Math.floor(z / count / cell)}`;
    (grid.get(key) ?? grid.set(key, []).get(key)).push(c);
  }
  const out = new Uint32Array(G * 8), wf = new Float32Array(out.buffer);
  const k = d2 => Math.exp(-d2 / (2 * sigma * sigma));
  for (let g = 0; g < G; g++) {
    const oc = own[g];
    if (oc < 0) continue;
    const p = cpart[oc], x = gf[g * 16], y = gf[g * 16 + 1], z = gf[g * 16 + 2];
    const ix = Math.floor(x / cell), iy = Math.floor(y / cell), iz = Math.floor(z / cell);
    const near = [];
    for (let a = -2; a <= 2; a++) for (let b = -2; b <= 2; b++) for (let e = -2; e <= 2; e++) {
      const list = grid.get(`${p},${ix + a},${iy + b},${iz + e}`);
      if (!list) continue;
      for (const c of list) {
        const dx = cen[3 * c] - x, dy = cen[3 * c + 1] - y, dz = cen[3 * c + 2] - z;
        near.push([dx * dx + dy * dy + dz * dz, c]);
      }
    }
    near.sort((u, v) => u[0] - v[0]);
    const floor = near.length > n ? k(near[n][0]) : 0;
    const set = near.slice(0, n);
    if (!set.some(([, c]) => c === oc)) set[set.length - 1] = near.find(([, c]) => c === oc) ?? [Infinity, oc];
    set.sort((u, v) => (v[1] === oc) - (u[1] === oc));          // own cluster first
    set.forEach(([d2, c], q) => { out[g * 8 + q] = c; wf[g * 8 + 4 + q] = Math.max(k(d2) - floor, 0); });
  }
  return out;
}

export class OnePath {
  static async load(device, assetName, runtimeShaders, lighting) {
    const base = `runtime/assets/${assetName}/onepath`;
    const meta = await fetch(`${base}.json`).then(r => { if (!r.ok) throw new Error(`${base}.json missing (bake_onepath.py)`); return r.json(); });
    const bin = await fetch(`${base}.bin`).then(r => r.arrayBuffer());
    const src = await fetch('runtime/wgsl/onepath.wgsl').then(r => r.text());
    const csrc = await fetch('runtime/wgsl/onepath_colour.wgsl').then(r => r.text());
    return new OnePath(device, meta, bin, runtimeShaders.common + '\n' + src, runtimeShaders.common + '\n' + csrc, lighting);
  }

  constructor(device, meta, bin, src, csrc, lighting) {
    this.device = device; this.meta = meta; this.lighting = lighting;
    const B = GPUBufferUsage;
    const mk = (label, bytes, extra = 0) => {
      const b = device.createBuffer({ label, size: Math.max(16, bytes.byteLength), usage: B.STORAGE | B.COPY_DST | extra });
      device.queue.writeBuffer(b, 0, bytes); return b;
    };
    const slice = s => new Uint8Array(bin, s.offset, s.bytes);
    this.lists = mk('onepath lists', slice(meta.lists));
    // soft cluster membership (?soft=N, 1-4 clusters per receiver; prototype 2026-10-28): gather_m stores each cluster's
    // picks with 1 / N of the budget, gather_soft blends each receiver's N nearest clusters (?softsigma=<m>, kernel std)
    const qs = new URLSearchParams(globalThis.location?.search ?? '');
    this.soft = Math.max(0, Math.min(4, Number(qs.get('soft')) || 0));
    this.softDiv = this.soft ? Math.max(1, Number(qs.get('softdiv')) || this.soft) : 1;
    this.softR = qs.has('softr') ? Number(qs.get('softr')) : Infinity;   // debug: gather_soft over the first N receivers only
    this.softHdr = meta.ncl * KMAX * 2; this.softAcc = this.softHdr + meta.ncl;
    this.sb = device.createBuffer({ label: 'onepath soft picks', size: this.soft ? (this.softAcc + 2 * meta.G) * 16 : 16, usage: B.STORAGE });
    const nb = this.soft ? softNeighbours(meta, new Uint32Array(bin, meta.lists.offset, meta.lists.bytes / 4),
      lighting.gaussianBytes, this.soft, Number(qs.get('softsigma')) || 0.5 * meta.cluster) : new Uint32Array(4);
    this.snb = mk('onepath soft neighbours', nb);
    this.tree = mk('onepath tree', slice(meta.tree));
    // one extra record after the Gaussians: sched_cut's cutoff bucket (gext[G].Le_w.x)
    this.gextInit = new Uint8Array(meta.gext.bytes + 80);
    this.gextInit.set(slice(meta.gext));
    const lampOff = new URLSearchParams(globalThis.location?.search ?? '').get('oplamp') === '0';   // ?oplamp=0: emitters dark (sky + sun only)
    { const g = new Float32Array(this.gextInit.buffer);
      for (let i = 0; i < meta.G; i++) {
        g[i * 20 + 7] = 0;                                                                    // counts start at 0
        if (lampOff) for (const o of [0, 4, 8, 16]) g.fill(0, i * 20 + o, i * 20 + o + 3);   // Le, B, B_frame, cache
      } }
    this.gext = mk('onepath gext', this.gextInit, B.COPY_SRC);   // COPY_SRC: gsDebug.readBuffer
    this.nsh = device.createBuffer({ label: 'onepath node SH', size: meta.N * 27 * 4, usage: B.STORAGE });
    this.colour = device.createBuffer({ label: 'onepath colour', size: meta.G * 2 * 16, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC });   // COPY_SRC: gsDebug.readBuffer
    this.fill = mk('onepath fill', slice(meta.fill));
    this.zeroBounce = device.createBuffer({ label: 'onepath (no bounce)', size: meta.G * 15 * 16, usage: B.STORAGE });
    this.params = device.createBuffer({ label: 'onepath params', size: SLOT * 16, usage: B.UNIFORM | B.COPY_DST });
    const lb = lighting.buffers;
    const BT = t => ({ type: t });
    this.layout = device.createBindGroupLayout({ label: 'onepath', entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: BT('uniform') },
      ...[1, 2, 3, 4].map(b => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: BT('read-only-storage') })),
      ...[5, 6, 7].map(b => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: BT('storage') })),
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: BT('read-only-storage') }] });
    // group 2, gather_m and gather_soft: the soft picks (onepath_spec.wgsl prepends onepath.wgsl and owns group 1)
    this.softLayout = device.createBindGroupLayout({ label: 'onepath soft', entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: BT('storage') },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: BT('read-only-storage') }] });
    this.softGroup = device.createBindGroup({ layout: this.softLayout, entries: [
      { binding: 0, resource: { buffer: this.sb } }, { binding: 1, resource: { buffer: this.snb } }] });
    this.emptyGroup = device.createBindGroup({ layout: device.createBindGroupLayout({ entries: [] }), entries: [] });
    const pl2 = device.createPipelineLayout({ bindGroupLayouts: [this.layout, device.createBindGroupLayout({ entries: [] }), this.softLayout] });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const module = device.createShaderModule({ label: 'onepath', code: src });
    this.pipes = Object.fromEntries(['nb_leaf', 'nb_up', 'gather_m', 'history_x', 'rebase', 'sched_cut', 'gather_soft'].map(e =>
      [e, device.createComputePipeline({ label: e, layout: e.startsWith('gather') ? pl2 : pl, compute: { module, entryPoint: e } })]));
    this.groups = Array.from({ length: 16 }, (_, k) => device.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.params, offset: k * SLOT, size: PSIZE } },
      { binding: 1, resource: { buffer: lb.gaussians } }, { binding: 2, resource: { buffer: lb.nodes } },
      { binding: 3, resource: { buffer: lb.kids } }, { binding: 4, resource: { buffer: lb.vgrid } },
      { binding: 5, resource: { buffer: this.tree } }, { binding: 6, resource: { buffer: this.nsh } },
      { binding: 7, resource: { buffer: this.gext } }, { binding: 8, resource: { buffer: this.lists } }] }));
    // colour copy
    const cmod = device.createShaderModule({ label: 'onepath colour', code: csrc });
    this.cpipe = device.createComputePipeline({ label: 'op_colour', layout: 'auto', compute: { module: cmod, entryPoint: 'op_colour' } });
    this.cparams = device.createBuffer({ size: 16, usage: B.UNIFORM | B.COPY_DST });
    this.fillOn = new URLSearchParams(globalThis.location?.search ?? '').get('fill') !== '0';   // ?fill=0: no display fill
    device.queue.writeBuffer(this.cparams, 0, new Uint32Array([meta.G, this.fillOn ? 1 : 0, 0, 0]));
    this.cgroup = device.createBindGroup({ layout: this.cpipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.cparams } }, { binding: 1, resource: { buffer: this.gext } },
      { binding: 2, resource: { buffer: this.colour } }, { binding: 3, resource: { buffer: this.fill } }] });
    this.frame = 0;
    this.nmax = 256;                // history count cap (still); 4 where a moving part touches > 5% of the light. A low cap
                                    // (64) left the change noise near a moved box as blotches, then the rebase froze them
    this.kAct = 32; this.kIdle = 8; this.trickle = 64;
    // progressive scheduling (?budget=N clusters a frame, prototype 2026-10-28): converging clusters gather by priority
    // (on-screen size x distance from converged) at K sched, counting K / K idle samples; 0 = off (today's behaviour)
    this.budget = Math.max(0, Number(new URLSearchParams(globalThis.location?.search ?? '').get('budget')) || 0);
    this.kSched = 32;
    this.vp = new Float32Array(16);
    this.P = meta.part_count;
    this.xf = new Float32Array(16 * 12); this.xfp = new Float32Array(16 * 12); this.xfc = new Float32Array(16 * 12);
    for (let p = 0; p < 16; p++) { this.xf.set(IDENT, p * 12); this.xfp.set(IDENT, p * 12); this.xfc.set(IDENT, p * 12); }
    this.cacheOk = false;
    this.noCV = false;              // debug: full estimate everywhere
    this.env = null;                // distant senders: { sun, sunE, bins } or null (environment off)
    this.envKey = '';
    this.envCacheKey = '';          // the environment the cache was solved with
    this.still = 0;                 // frames since a part last moved
    this.slot = SLOT; this.psize = PSIZE;   // for onepath_det.js (its own uniform slots, the same Params layout)
    this.idleAfter = 3 * this.nmax; // stop rendering this long after the last change (the trickle would keep refining)
  }

  // Params (onepath.wgsl): u32 / f32 words
  paramBytes({ R = 0, frame = 0, lvl = [0, 0], alpha = 1, K = this.kAct, masks = [0, 0, 0, 0] }) {
    const m = this.meta;
    const u = new Uint32Array(PSIZE / 4), f = new Float32Array(u.buffer);
    u[0] = m.G; u[1] = m.N; u[2] = R; u[3] = K; u[4] = frame; u[5] = m.part_count; f[6] = 0.1; f[7] = alpha;
    u[8] = lvl[0]; u[9] = lvl[1]; u[10] = m.senders; u[11] = m.lsh_off;
    f[12] = m.theta; u[13] = m.cl_off; u[14] = m.ncl; u[15] = 0xffffffff;
    m.roots.forEach((r, i) => { u[16 + i] = r; });
    m.pbound.forEach((b, i) => f.set(b, 32 + 4 * i));
    f.set(this.xf, 104); f.set(this.xfp, 296); f.set(this.xfc, 488);
    u.set(masks, 680); f[684] = this.nmax; f[685] = this.trickle; f[686] = 0.05; f[687] = this.noCV ? 1 : 0;
    const e = this.env;
    if (e) {
      f.set(e.sun, 688); f.set(e.sunE, 692);
      u[696] = NBINS; u[697] = e.sky ? SKY_RAYS : 0; u[698] = e.sunE[0] + e.sunE[1] + e.sunE[2] > 0 ? SUN_RAYS : 0;
      f.set(e.bins, 700);
    }
    u.set([this.budget, this.kSched, 0, 0], 1212);   // sched
    f.set(this.vp, 1216);                              // vp
    u.set([this.soft, this.softHdr, this.softAcc, this.softDiv], 1232);   // soft
    return u;
  }

  reset() { this.frame = 0; this.cacheOk = false; this.still = 0; this.device.queue.writeBuffer(this.gext, 0, this.gextInit); }

  // the environment (environment.js: sky SH with its gain, the first distant light) as distant senders; true if changed
  setEnv(environment) {
    let e = null;
    if (environment?.on) {
      const L = environment.lights()[0];
      const sh = environment.base.sh.map(c => c.map(v => v * environment.skyGain));
      e = { sun: L ? [...L.dir, L.std] : [0, 0, 1, 0], sunE: L ? [...L.irradiance, 0] : [0, 0, 0, 0], sh, sky: sh.length > 0 };
    }
    const key = e ? JSON.stringify([e.sun, e.sunE, e.sh]) : '';
    if (key === this.envKey) return false;
    this.envKey = key;
    if (e) e.bins = e.sky ? skyBins(e.sh) : new Float32Array(NBINS * 8);
    this.env = e;
    return true;
  }

  get converging() { return this.still < this.idleAfter; }

  // the renderer's part records (main.js partBytes: 32 floats per part, xf rows at 0..11)
  readParts(pf) {
    let since = 0, frame = 0;
    for (let p = 0; p < this.P; p++) {
      const x = pf.subarray(p * 32, p * 32 + 12);
      for (let k = 0; k < 12; k++) {
        if (x[k] !== this.xf[p * 12 + k]) frame |= 1 << p;
      }
      this.xfp.set(this.xf.subarray(p * 12, p * 12 + 12), p * 12);
      this.xf.set(x, p * 12);
      for (let k = 0; k < 12; k++) if (x[k] !== this.xfc[p * 12 + k]) since |= 1 << p;
    }
    return [since, frame];
  }

  // one lighting frame: sender aggregates, the cluster gather (control variate against the cache once there is one),
  // the count history, a cache rebase after a still spell, then the colour copy
  // vp: the camera's view-projection (column-major), for progressive scheduling
  encode(encoder, partXf, environment = null, vp = null) {
    if (vp) this.vp.set(vp);
    const m = this.meta, f = this.frame;
    let [since, moved] = partXf ? this.readParts(partXf) : [0, 0];
    if (this.setEnv(environment)) moved |= ENV_BIT;
    if (this.envKey !== this.envCacheKey) since |= ENV_BIT;
    since >>>= 0; moved >>>= 0;
    this.still = moved ? 0 : this.still + 1;
    // first convergence: after 2 n_max frames the history becomes the cache; later, after a still spell
    const rebase = this.still === 2 * this.nmax && (!this.cacheOk || since !== 0);
    if (this.converging) {
      const masks = [since, moved, this.cacheOk ? 1 : 0, this.kIdle];
      const steps = [['nb_leaf', m.senders, { lvl: [m.send_off, 0] }],
        ...m.levels.map(([o, c]) => ['nb_up', c, { lvl: [o, c] }]),
        ...(this.budget > 0 ? [['sched_cut', 1, { masks }]] : []),
        ['gather_m', m.ncl, { R: m.receivers, frame: 50000 + f, masks }],
        ...(this.soft ? [['gather_soft', Math.min(m.receivers, this.softR), { R: Math.min(m.receivers, this.softR), frame: 50000 + f, masks }]] : []),
        ['history_x', m.receivers, { R: m.receivers, frame: f, masks }]];
      if (rebase) steps.push(['rebase', m.G, {}]);
      steps.forEach(([, , p], k) => this.device.queue.writeBuffer(this.params, k * SLOT, this.paramBytes(p)));
      steps.forEach(([e, n], k) => {
        const pass = encoder.beginComputePass({ label: `onepath ${e}` });
        pass.setPipeline(this.pipes[e]); pass.setBindGroup(0, this.groups[k]);
        if (e.startsWith('gather')) { pass.setBindGroup(1, this.emptyGroup); pass.setBindGroup(2, this.softGroup); }
        pass.dispatchWorkgroups(e === 'gather_m' || e === 'sched_cut' ? n : Math.ceil(n / WG)); pass.end();
      });
      if (rebase) { this.cacheOk = true; this.xfc.set(this.xf); this.envCacheKey = this.envKey; }
      this.frame++;
    }
    const pass = encoder.beginComputePass({ label: 'onepath colour' });
    pass.setPipeline(this.cpipe); pass.setBindGroup(0, this.cgroup); pass.dispatchWorkgroups(Math.ceil(m.G / WG)); pass.end();
  }
}
