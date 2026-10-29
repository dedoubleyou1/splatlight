// Light-side deterministic shadows (runtime/wgsl/light_shadow.wgsl), in place of walk_emit + ema (A is the renderer's
// default since 2026-10-25, David; main.js `shadowMethod`):
//   'map'  A: per-part opacity map splatted from the light (lsm_splat -> lsm_resolve -> lsm_down per level -> lsm_lookup)
//   'lobe' B: Gaussian-lobe light map (lobe_clear -> lobe_cut -> lobe_lookup)
//   'slice' C: B simplified: one cut per receiver-distance slice, lobes evaluated in the light's direction space, bins
//          per (slice, tile, part) compacted (lc_clear -> lc_cut -> lc_scan -> lc_scatter -> lc_lookup)
// Both write the other-part transmittance into lamp_state slot 0 every frame, with no history, so shade_emit and the
// bounce read it unchanged. A re-splats only the layers of parts that moved (all of them when the light changed);
// B and C rebuild their cuts when anything moved. All three are deterministic, so with nothing moved the lookup is
// skipped and lamp_state keeps last frame's T. Mirrors renderer/tools/light_shadow.py (the native harness).

const WG = 64;
const B = GPUBufferUsage;
const NONE = 0xffffffff;
const BINDS = {
  lsm_splat: [0, 1, 2, 4, 35, 40, 41, 52, 53, 54],
  lsm_resolve: [40, 41, 42],
  lsm_down: [40, 42, 46],
  lsm_lookup: [0, 1, 4, 6, 40, 42],
  lobe_clear: [40, 44],
  lobe_cut: [0, 1, 2, 4, 35, 40, 43, 44, 45],
  lobe_lookup: [0, 1, 2, 4, 6, 9, 40, 43, 44, 45],
  lc_clear: [40, 48],
  lc_cut: [0, 1, 2, 4, 35, 40, 47, 48],
  lc_scan: [40, 48, 49, 50],
  lc_scatter: [40, 47, 48, 50, 51],
  lc_lookup: [0, 1, 4, 6, 40, 47, 48, 49, 51],
};
// C: lobes, bin items (compacted), lobes evaluated per receiver at most
const OPT_C = { maxLobes: 1 << 19, items: 1 << 23, budget: 4096 };
// light_shadow.py defaults (cut 0.25 rad, refine 8, unwidened flat tops summed per part; 512 map)
const OPT = { res: 512, tiles: 32, tileCap: 1024, cutTheta: 0.25, refine: 8, budget: 256, maxLobes: 1 << 17, slices: 10, d0: 0.25, d1: 3.5, combine: 2 };

// Per part, the mean number of its surfaces a straight line crosses (light_shadow.py part_layers): centres binned
// across 13 directions, runs along each direction more than 4 cells apart counted. A wall ~1, a closed box ~2.
function partLayers(g, parts) {
  const dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]];
  const out = new Float32Array(32).fill(1);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = a => { const l = Math.hypot(...a); return a.map(v => v / l); };
  parts.forEach((p, pi) => {
    const [g0, g1] = p.gaussians, n = g1 - g0;
    if (n < 8 || pi >= 32) return;
    const mid = Array.from({ length: n }, (_, i) => g[(g0 + i) * 16 + 9]).sort((a, b) => a - b);
    const cell = 2 * mid[n >> 1];
    let acc = 0;
    for (const d0 of dirs) {
      const d = unit(d0), u = unit(cross(d, Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0])), v = cross(d, u);
      const rows = [];
      for (let i = 0; i < n; i++) {
        const o = (g0 + i) * 16, m = [g[o], g[o + 1], g[o + 2]];
        const dot = a => a[0] * m[0] + a[1] * m[1] + a[2] * m[2];
        rows.push([Math.floor(dot(u) / cell) * 1000003 + Math.floor(dot(v) / cell), dot(d)]);
      }
      rows.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      let cells = 0, runs = 0;
      rows.forEach((r, i) => {
        if (i === 0 || r[0] !== rows[i - 1][0]) cells++;
        else if (r[1] - rows[i - 1][1] > 4 * cell) runs++;
      });
      acc += (cells + runs) / cells;
    }
    out[pi] = Math.max(acc / dirs.length, 1);
  });
  return out;
}

export class LightShadow {
  constructor(device, sources, lighting, asset) {
    this.device = device;
    this.lighting = lighting;
    this.asset = asset;
    this.method = null;           // null = today's walk
    this.module = device.createShaderModule({ label: 'light_shadow', code: sources.common + '\n' + sources.light_shadow });
    this.layers = partLayers(new Float32Array(asset.gaussians), asset.header.parts);
    this.ready = {};
    this.dirtyAll = true;
  }

  // null, 'map' or 'lobe'; buffers and pipelines are made on first use
  setMethod(m) {
    if (m === this.method) return;
    this.method = m;
    this.dirtyAll = true;
    if (m && !this.ready[m]) this.init(m);
  }

  init(m) {
    const d = this.device, L = this.lighting, P = L.partCount;
    const buf = (label, size, usage = B.STORAGE) => d.createBuffer({ label, size: Math.max(16, size), usage: usage | B.COPY_DST | B.COPY_SRC });
    if (!this.params) {
      this.params = buf('light shadow params', 336, B.UNIFORM);
      this.ls = new ArrayBuffer(336);
    }
    const res = {
      0: L.buffers.uniform, 1: L.buffers.gaussians, 2: L.buffers.nodes, 4: L.buffers.parts, 6: L.buffers.lamp_state,
      9: L.buffers.kids, 35: L.buffers.node_parent, 40: this.params,
      52: L.buffers.gaussians, 53: L.buffers.nodes,   // rest records (lsm_splat's 1 and 2 are the motion-blur copies)
    };
    // motion-blur shadows from a smear-sized cut (lighting.js blurCut: ?blurcut=<k>, default 1, David 2026-10-29;
    // light_shadow.wgsl lsm_splat; 0 = stretch every Gaussian, the earlier rule)
    this.blurCut = L.blur ? L.blurCut : 0;
    const bc = new ArrayBuffer(16);
    new Float32Array(bc).set([L.blurShutter ?? 0, this.blurCut]);
    new Uint32Array(bc)[2] = this.blurCut > 0 ? L.nodeCount : 0;
    res[54] = buf('blur cut params', 16, B.UNIFORM);
    d.queue.writeBuffer(res[54], 0, bc);
    const pipe = (entry, groups) => {
      const p = d.createComputePipeline({ label: entry, layout: 'auto', compute: { module: this.module, entryPoint: entry } });
      return { pipe: p, groups: groups.map(g => d.createBindGroup({ layout: p.getBindGroupLayout(0),
        entries: BINDS[entry].map(b => ({ binding: b, resource: { buffer: g[b] ?? res[b] } })) })) };
    };
    if (m === 'map') {
      let r = OPT.res;
      while (r > 64 && Math.ceil(P * r * r / WG) > 65535) r >>= 1;   // one dispatch over every layer's texels
      this.res = r;
      this.levels = [];
      let off = 0;
      for (let k = r; k >= 1; k >>= 1) { this.levels.push([off, k]); off += k * k; }
      this.layerTexels = off;
      this.acc = buf('light shadow map (fixed point)', P * r * r * 12);
      this.mip = buf('light shadow map mips', P * off * 16);
      this.lv = this.levels.map(([, rk], k) => { const b = buf(`light shadow level ${k}`, 16, B.UNIFORM); d.queue.writeBuffer(b, 0, new Uint32Array([k, rk, 0, 0])); return b; });
      Object.assign(res, { 41: this.acc, 42: this.mip });
      this.map = {
        // motion blur (lighting.js): moving parts splat their stretched records
        lsm_splat: pipe('lsm_splat', [L.blur ? { 1: L.buffers.gaussians_blur, 2: L.buffers.nodes_blur } : {}]), lsm_resolve: pipe('lsm_resolve', [{}]),
        lsm_down: pipe('lsm_down', this.lv.map(b => ({ 46: b }))), lsm_lookup: pipe('lsm_lookup', [{}]),
      };
    } else if (m === 'slice') {
      const nb = OPT.slices * OPT.tiles * OPT.tiles * P;
      this.nbins = nb;
      Object.assign(res, {
        47: buf('light-space lobes', OPT_C.maxLobes * 80), 48: buf('light-space bin counts', (nb + 2) * 4),
        49: buf('light-space bin starts', nb * 4), 50: buf('light-space bin cursors', nb * 4),
        51: buf('light-space bin items', OPT_C.items * 4),
      });
      this.slice = Object.fromEntries(['lc_clear', 'lc_cut', 'lc_scan', 'lc_scatter', 'lc_lookup'].map(e => [e, pipe(e, [{}])]));
    } else {
      const bins = OPT.slices * OPT.tiles * OPT.tiles;
      this.lobes = buf('light lobes', OPT.maxLobes * 64);
      this.tileCount = buf('light lobe bins', (bins + 1) * 4);
      this.tileItems = buf('light lobe bin items', bins * OPT.tileCap * 4);
      Object.assign(res, { 43: this.lobes, 44: this.tileCount, 45: this.tileItems });
      this.lobe = { lobe_clear: pipe('lobe_clear', [{}]), lobe_cut: pipe('lobe_cut', [{}]), lobe_lookup: pipe('lobe_lookup', [{}]) };
    }
    this.ready[m] = true;
  }

  // The light as one point and an isotropic spread: the emitter tree root's mean and its ASG covariance (lighting.js
  // keeps the last tree in emitTree / emitLight). Called whenever the emitters change.
  setLight() {
    this.dirtyAll = true;
  }

  writeParams(dirty) {
    const L = this.lighting, t = L.emitTree, e = L.emitLight;
    if (!t || !e) return false;
    const f = new Float32Array(this.ls), u = new Uint32Array(this.ls);
    const nf = t.nodes, c = t.cov.subarray(t.root * 8, t.root * 8 + 8);
    const center = [nf[t.root * 16], nf[t.root * 16 + 1], nf[t.root * 16 + 2]];
    let n = e.normal; const nl = Math.hypot(...n); n = n.map(v => v / nl);
    const a = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let t1 = [n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0]];
    const tl = Math.hypot(...t1); t1 = t1.map(v => v / tl);
    const t2 = [n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]];
    // isotropic light std: the two larger eigenvalues of the root's covariance (trace minus the smallest)
    const ev = eig3([c[0], c[1], c[2], c[4], c[5], c[6]]);
    const sigma = Math.sqrt(0.5 * (ev[1] + ev[2]));
    f.set(center, 0); u[3] = this.res ?? OPT.res;
    f.set(n, 4); u[7] = this.levels?.length ?? 1;
    f.set(t1, 8); u[11] = this.layerTexels ?? 0;
    f.set(t2, 12); u[15] = dirty;
    f[16] = sigma; u[17] = e.skipPart; u[18] = e.oneSided ? 1 : 0; u[19] = L.partCount;
    const C = this.method === 'slice';
    f[20] = OPT.cutTheta; u[21] = OPT.tiles; u[22] = C ? OPT_C.items : OPT.tileCap; f[23] = OPT.refine;
    f.set([c[0], c[1], c[2]], 24); u[27] = C ? OPT_C.maxLobes : OPT.maxLobes;
    f.set([c[4], c[5], c[6]], 28); u[31] = C ? OPT_C.budget : OPT.budget;
    u[32] = OPT.slices; f[33] = OPT.d0; f[34] = Math.log(OPT.d1 / OPT.d0) / OPT.slices; u[35] = OPT.combine;
    (this.levels ?? []).forEach(([o], k) => { u[36 + k] = o; });
    f.set(this.layers, 52);
    this.device.queue.writeBuffer(this.params, 0, this.ls);
    return true;
  }

  // In place of shadow_cull + walk_emit + ema. movedMask: bit p = part p moved this frame.
  encode(encoder, timers, movedMask, changed) {
    const L = this.lighting, P = L.partCount;
    const run = (p, entry, threads, k = 0, timer) => {
      const { pipe, groups } = p[entry];
      const pass = encoder.beginComputePass({ label: entry, timestampWrites: timer ? timers.writes(timer) : undefined });
      pass.setPipeline(pipe); pass.setBindGroup(0, groups[k]);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(threads / WG))); pass.end();
    };
    // deterministic: with nothing moved, last frame's T in lamp_state is still right
    const look = this.dirtyAll || changed || movedMask !== 0;
    if (this.method === 'map') {
      const dirty = this.dirtyAll ? (2 ** P - 1) : movedMask;
      if (!this.writeParams(dirty >>> 0)) return false;
      if (dirty) {
        const n = this.res * this.res * 12;
        for (let p = 0; p < P; p++) if ((dirty >>> p) & 1) encoder.clearBuffer(this.acc, p * n, n);
        run(this.map, 'lsm_splat', this.blurCut > 0 ? L.nodeCount : L.count);   // ?blurcut: tree nodes too
        run(this.map, 'lsm_resolve', P * this.res * this.res);
        for (let k = 1; k < this.levels.length; k++) run(this.map, 'lsm_down', P * this.levels[k][1] ** 2, k);
      }
      if (look) run(this.map, 'lsm_lookup', L.count, 0, 'walk_lamp');
    } else if (this.method === 'slice') {
      if (!this.writeParams(0)) return false;
      if (look) {
        run(this.slice, 'lc_clear', this.nbins + 2);
        run(this.slice, 'lc_cut', L.nodeCount * OPT.slices);
        const pass = encoder.beginComputePass({ label: 'lc_scan' });
        pass.setPipeline(this.slice.lc_scan.pipe); pass.setBindGroup(0, this.slice.lc_scan.groups[0]);
        pass.dispatchWorkgroups(1); pass.end();
        run(this.slice, 'lc_scatter', OPT_C.maxLobes);
        run(this.slice, 'lc_lookup', L.count, 0, 'walk_lamp');
      }
    } else {
      if (!this.writeParams(0)) return false;
      if (this.dirtyAll || changed) {
        run(this.lobe, 'lobe_clear', OPT.slices * OPT.tiles * OPT.tiles + 1);
        run(this.lobe, 'lobe_cut', L.nodeCount);
      }
      if (look) run(this.lobe, 'lobe_lookup', L.count, 0, 'walk_lamp');
    }
    this.dirtyAll = false;
    return true;
  }
}

// eigenvalues of a symmetric 3x3 [xx yy zz xy xz yz], ascending (closed form)
function eig3([a, d, f, b, c, e]) {
  const p1 = b * b + c * c + e * e;
  const q = (a + d + f) / 3;
  const p2 = (a - q) ** 2 + (d - q) ** 2 + (f - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  if (p < 1e-30) return [q, q, q];
  const B = [(a - q) / p, (d - q) / p, (f - q) / p, b / p, c / p, e / p];
  const r = (B[0] * (B[1] * B[2] - B[5] * B[5]) - B[3] * (B[3] * B[2] - B[5] * B[4]) + B[4] * (B[3] * B[5] - B[1] * B[4])) / 2;
  const phi = Math.acos(Math.min(1, Math.max(-1, r))) / 3;
  const e1 = q + 2 * p * Math.cos(phi), e3 = q + 2 * p * Math.cos(phi + 2 * Math.PI / 3);
  return [e3, 3 * q - e1 - e3, e1];
}
