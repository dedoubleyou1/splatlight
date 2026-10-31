// Screen-space LOD for drawing (?lod=<px>, 2026-10-28; runtime/wgsl/lod.wgsl). The asset's tree is the LOD: each
// internal node is drawn as its moment-matched Gaussian in place of its subtree when it is under <px> on screen.
// nodeGaussians() turns the internal nodes into Gaussian records (appended to the renderer's Gaussian buffer, so the
// projection pass draws them like leaves); Lod.encode() runs, after the lighting, lod_colour (bottom-up: the lit
// colour of every node) and lod_select (top-down: the cut), one dispatch per tree depth.

const NODE = 96, GAUSS = 64;

// Symmetric 3x3 eigen-decomposition (Jacobi): returns [values ascending, vectors as columns].
function eig3(A) {
  const a = A.map(r => r.slice()), V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 20; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) off += a[p][q] * a[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) {
      if (Math.abs(a[p][q]) < 1e-30) continue;
      const th = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]), c = Math.cos(th), s = Math.sin(th);
      for (let k = 0; k < 3; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
    }
  }
  const idx = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  return [idx.map(i => a[i][i]), idx.map(i => [V[0][i], V[1][i], V[2][i]])];   // vectors[k] = k-th column
}

// rotation with columns c0, c1, c2 -> quaternion (x, y, z, w), the convention of quat_to_mat (columns = axes)
function quatFromColumns(c) {
  const m00 = c[0][0], m11 = c[1][1], m22 = c[2][2];
  const m10 = c[0][1], m20 = c[0][2], m01 = c[1][0], m21 = c[1][2], m02 = c[2][0], m12 = c[2][1];   // m_rc = c[col][row]
  const tr = m00 + m11 + m22;
  let x, y, z, w;
  if (tr > 0) { const s = 2 * Math.sqrt(tr + 1); w = s / 4; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s; }
  else if (m00 > m11 && m00 > m22) { const s = 2 * Math.sqrt(1 + m00 - m11 - m22); x = s / 4; w = (m21 - m12) / s; y = (m01 + m10) / s; z = (m02 + m20) / s; }
  else if (m11 > m22) { const s = 2 * Math.sqrt(1 + m11 - m00 - m22); y = s / 4; w = (m02 - m20) / s; x = (m01 + m10) / s; z = (m12 + m21) / s; }
  else { const s = 2 * Math.sqrt(1 + m22 - m00 - m11); z = s / 4; w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; }
  const l = Math.hypot(x, y, z, w) || 1;
  return [x / l, y / l, z / l, w / l];
}

// Internal nodes (G..N-1) as Gaussian records: the node's mean, kappa and covariance (inverted from cinv), its part,
// the mass-weighted albedo of its leaves and the material of its first leaf; and each node's own-part V grid, the
// mass-weighted mean of its children's (for lighting a node directly, ?lodlight=1). Returns { records, vgrid, N, G, maxDepth }.
// withV false: skip the node V grids (only lit nodes, ?lodlight, read them; most of the cost on big scenes)
export function nodeGaussians(asset, withV = true) {
  const G = asset.count, nb = asset.raw.nodes, N = nb.byteLength / NODE;
  const nf = new Float32Array(nb), nu = new Uint32Array(nb), kids = new Uint32Array(asset.raw.kids);
  const gf = new Float32Array(asset.gaussians), gu = new Uint32Array(asset.gaussians);
  const out = new ArrayBuffer(Math.max(1, N - G) * GAUSS), of = new Float32Array(out), ou = new Uint32Array(out);
  const mass = new Float64Array(N), alb = new Float64Array(N * 3), mat = new Uint32Array(N);
  const leafV = new Uint8Array(asset.raw.vgrid), V = withV ? new Float32Array(N * 128) : null;   // V grid cells, 0-255
  if (withV) for (let i = 0; i < G * 128; i++) V[i] = leafV[i];
  const cov = n => {
    const o = n * 24, a = nf.subarray(o + 4, o + 7), b = nf.subarray(o + 8, o + 11);   // cinv diag (xx yy zz), off (xy xz yz)
    const M = [[a[0], b[0], b[1]], [b[0], a[1], b[2]], [b[1], b[2], a[2]]];
    const det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
      + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    const inv = [[M[1][1] * M[2][2] - M[1][2] * M[2][1], M[0][2] * M[2][1] - M[0][1] * M[2][2], M[0][1] * M[1][2] - M[0][2] * M[1][1]],
      [M[1][2] * M[2][0] - M[1][0] * M[2][2], M[0][0] * M[2][2] - M[0][2] * M[2][0], M[0][2] * M[1][0] - M[0][0] * M[1][2]],
      [M[1][0] * M[2][1] - M[1][1] * M[2][0], M[0][1] * M[2][0] - M[0][0] * M[2][1], M[0][0] * M[1][1] - M[0][1] * M[1][0]]];
    return [inv.map(r => r.map(v => v / det)), det];
  };
  let maxDepth = 0;
  const order = [];
  for (let n = 0; n < N; n++) { const d = nu[n * 24 + 21]; maxDepth = Math.max(maxDepth, d); if (n >= G) order.push(n); }
  for (let g = 0; g < G; g++) {
    mass[g] = nf[g * 24 + 7] / Math.sqrt(Math.max(cov(g)[1], 1e-30));
    alb.set([gf[g * 16 + 12], gf[g * 16 + 13], gf[g * 16 + 14]], 3 * g); mat[g] = gu[g * 16 + 15] & 0xffff;
  }
  order.sort((a, b) => nu[b * 24 + 21] - nu[a * 24 + 21]);              // deepest first: children before parents
  for (const n of order) {
    const lo = nu[n * 24 + 19], hi = nu[n * 24 + 20];
    let m = 0, c = [0, 0, 0];
    for (let k = lo; k < hi; k++) { const ch = kids[k]; m += mass[ch]; for (let j = 0; j < 3; j++) c[j] += mass[ch] * alb[3 * ch + j]; }
    mass[n] = m; for (let j = 0; j < 3; j++) alb[3 * n + j] = c[j] / Math.max(m, 1e-30);
    if (withV) for (let k = lo; k < hi; k++) { const ch = kids[k], w = mass[ch] / Math.max(m, 1e-30); for (let q = 0; q < 128; q++) V[n * 128 + q] += w * V[ch * 128 + q]; }
    mat[n] = hi > lo ? mat[kids[lo]] : 0;
    const [C] = cov(n), [vals, vecs] = eig3(C);
    const cols = vecs.slice();
    const det = cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1]) - cols[1][0] * (cols[0][1] * cols[2][2] - cols[0][2] * cols[2][1])
      + cols[2][0] * (cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1]);
    if (det < 0) cols[2] = cols[2].map(v => -v);
    const r = n - G;
    of.set([nf[n * 24], nf[n * 24 + 1], nf[n * 24 + 2], nf[n * 24 + 7]], r * 16);             // mu, kappa
    of.set(quatFromColumns(cols), r * 16 + 4);
    of.set(vals.map(v => Math.sqrt(Math.max(v, 1e-12))), r * 16 + 8);                        // scale, ascending
    ou[r * 16 + 11] = nu[n * 24 + 15] & 0xffff;                                                // part_surface: the part
    of.set([alb[3 * n], alb[3 * n + 1], alb[3 * n + 2]], r * 16 + 12);
    ou[r * 16 + 15] = mat[n];
  }
  let vgrid = null;
  if (withV) {
    const vg = new Uint8Array(Math.max(1, N - G) * 128);
    for (let i = 0; i < (N - G) * 128; i++) vg[i] = Math.min(255, Math.round(V[G * 128 + i]));
    vgrid = vg.buffer;
  }
  return { records: out, vgrid, N, G, maxDepth };
}

const LOD_U_BYTES = 128;   // lod.wgsl LodU: view (64) + 4 u32 + 8 f32 + 4 u32

export class Lod {
  // lighting: its buffers (nodes, kids, colour, lo_b); renderer: parts and the flag buffer project.wgsl reads
  constructor(device, sources, asset, lighting, renderer, info) {
    this.device = device; this.info = info; this.lighting = lighting; this.flags = renderer.lodBuf;   // flags: inspection
    const module = device.createShaderModule({ label: 'lod', code: sources.common + '\n' + sources.lod });
    const b = lighting.buffers;
    this.uniforms = Array.from({ length: info.maxDepth + 1 }, () =>
      device.createBuffer({ label: 'lod params', size: LOD_U_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    const emitPart = asset.header.emitters ? asset.header.emitters.part : asset.header.parts.findIndex(p => p.name === 'lamp');
    this.emitPart = emitPart >= 0 ? emitPart : 0xffffffff;
    // per depth: that depth's internal nodes, then its leaves, so each pass dispatches over one level only (a binary
    // tree has ~25 levels; sweeping every node per level cost ~3x more, 2026-10-29)
    const nu = new Uint32Array(asset.raw.nodes), { N, G } = info;
    const byDepth = Array.from({ length: info.maxDepth + 1 }, () => [[], []]);
    for (let n = 0; n < N; n++) byDepth[nu[n * 24 + 21]][n < G ? 1 : 0].push(n);
    const order = new Uint32Array(N);
    let off = 0;
    this.levels = byDepth.map(([inner, leaves]) => {
      order.set(inner, off); order.set(leaves, off + inner.length);
      const l = { off, nInt: inner.length, nAll: inner.length + leaves.length };
      off += l.nAll;
      return l;
    });
    this.levelBuf = device.createBuffer({ label: 'lod nodes by depth', size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.levelBuf, 0, order);
    this.pipes = {};
    this.cut = lighting.lodCut;
    // open by the projected footprint's area (lod.wgsl screen_cov), the default since 2026-10-29 (David): long thin
    // things collapse once they are thin on screen; ?lodsize=radius the bounding-radius rule
    this.sizeRule = new URLSearchParams(globalThis.location?.search ?? '').get('lodsize') === 'radius' ? 0 : 1;   // ?lodlight=2: the cut is chosen before the lighting and only it is shaded
    const entries = { lod_colour: [1, 2, 4, 5, 6, 9, 10, 11], lod_select: [1, 2, 3, 6, 11] };
    if (this.cut) Object.assign(entries, { lod_list: [6, 7, 8], lod_args: [7] });
    for (const [e, binds] of Object.entries(entries)) {
      const pipe = device.createComputePipeline({ label: e, layout: 'auto', compute: { module, entryPoint: e } });
      const res = { 1: b.nodes, 2: b.kids, 3: renderer.partBuf, 4: b.colour, 5: b.lo_b, 6: renderer.lodBuf,
        7: lighting.lodList, 8: lighting.lodNeed, 9: b.lamp_state, 10: b.env_T, 11: this.levelBuf };
      this.pipes[e] = { pipe, groups: this.uniforms.map(u => device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
        entries: [...(e === 'lod_args' ? [] : [{ binding: 0, resource: { buffer: u } }]), ...binds.map(k => ({ binding: k, resource: { buffer: res[k] } }))] })) };
    }
  }

  // view: world -> view (column-major 16 floats); threshold: node footprint in px to open at; gain: bounce gain
  // shutter, blurLod: motion blur variant D (?blur=1&blurlod=<k>): a moving part opens only nodes over k x its smear
  // ?lodlight=2: main.js calls this before the lighting (it needs only the camera) and encode() skips it
  // moved: a part moved this frame (main.js writeParts). The cut is kept when nothing it depends on changed: the
  // camera, the settings and every part's transform (2026-10-31: re-selecting a still view cost ~4 ms on hill).
  encodeSelect(encoder, view, threshold, focal, near, gain, shutter = 0, blurLod = 0, moved = true) {
    const { N, G, maxDepth } = this.info;
    this.last = { threshold, focal, near, gain, shutter, blurLod };   // inspection (gsDebug.lod)
    if (!(threshold > 0) && !this.lighting.lodLight) return;
    const key = [...view, threshold, focal, near, shutter, blurLod, this.openParts?.() ?? 0, this.colourAll ? 1 : 0, this.lighting.lodLight ? 1 : 0].join();
    const blurSel = blurLod > 0 && shutter > 0;   // blur LOD: a part that just stopped still has last frame's smear
    if (!moved && !this.cut && !blurSel && key === this.selectKey) return;
    // same camera and settings: re-select only the parts that moved (lighting.movedMask, set by main.js writeParts)
    const mask = (this.lighting.movedMask ?? 0) >>> 0;
    const only = key === this.selectKey && !this.cut && !blurSel && mask !== 0 ? mask : 0xffffffff;
    this.selectKey = key;
    for (let d = 0; d <= maxDepth; d++) {
      const ab = new ArrayBuffer(LOD_U_BYTES), f = new Float32Array(ab), u = new Uint32Array(ab);
      f.set(view, 0); u.set([G, N, only, this.emitPart], 16); f.set([threshold, focal, near, gain, shutter, blurLod, this.lighting.lodLight ? 1 : 0, 0], 20);
      u[27] = this.openParts?.() ?? 0;   // LodU.open_parts (main.js)
      const l = this.levels[d]; u.set([l.off, l.nInt, l.nAll, this.sizeRule | (this.colourAll ? 2 : 0)], 28);   // bit 1: lod_colour colours every node
      this.device.queue.writeBuffer(this.uniforms[d], 0, ab);
    }
    for (let d = 0; d <= maxDepth; d++) this.run(encoder, 'lod_select', d, this.levels[d].nAll);
    if (this.cut) {
      this.lighting.encodeFillNeed(encoder);
      encoder.clearBuffer(this.lighting.lodList, 0, 16);   // the two counts
      this.run(encoder, 'lod_list', 0, N);
      this.run(encoder, 'lod_args', 0, 1);
    }
  }

  // node colours bottom-up from this frame's leaf colours (after encodeSelect: only the drawn subtrees unless colourAll)
  encodeColour(encoder) {
    for (let d = this.info.maxDepth; d >= 0; d--) this.run(encoder, 'lod_colour', d, this.levels[d].nInt);
  }

  run(encoder, e, d, threads) {
    const pass = encoder.beginComputePass({ label: `${e} ${d}` });
    pass.setPipeline(this.pipes[e].pipe); pass.setBindGroup(0, this.pipes[e].groups[d]);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(threads / 64))); pass.end();
  }

  // colourAll: colour every node, not only the drawn subtrees (main.js: an explosion, whose lit colours stay while
  // its cut follows the camera)
  encode(encoder, view, threshold, focal, near, gain, shutter = 0, blurLod = 0, moved = true, colourAll = false) {
    if (!(threshold > 0) && !this.lighting.lodLight) return;   // ?lod=0 (the sender cut alone keeps the tree): nothing drawn from it
    this.colourAll = colourAll || !!this.debugColourAll;   // debugColourAll: smoke_renderer.mjs lod_colour_cut
    if (!this.cut) this.encodeSelect(encoder, view, threshold, focal, near, gain, shutter, blurLod, moved);
    this.encodeColour(encoder);
    this.lighting.encodeNodeDirect(encoder);   // ?lodlight=1/2: drawn nodes lit directly (+ their leaves' mean bounce)
  }
}
