// The runtime lighting kernels (runtime/wgsl, SPEC v0.2 §4) in the browser
// frame: lamp segment walk, per-Gaussian EMA, then M36 shading into
// colour[g], which the project pass reads in the lit view. Kernel sources are
// shared with the native runner (runtime/run_kernels.py), unchanged.

import { Lights, MAX_LIGHTS } from './lights.js';

export const KERNELS = [['walk_lamp', 'walk_lamp'], ['ema', 'ema'], ['shade', 'shade']];
// Probe-loop bounce (runtime/wgsl/bounce.wgsl): entry points in dispatch order, and the bindings each one uses.
const BOUNCE = {
  near_vis: [0, 1, 2, 3, 4, 9, 10, 12, 27],
  direct_dirs: [0, 1, 2, 3, 4, 5, 6, 7, 10, 13],
  bounce_dirs: [0, 1, 2, 3, 4, 7, 10, 11, 12, 14],
  node_leaf: [0, 1, 2, 3, 4, 10, 13, 14, 15],
  node_up: [0, 2, 9, 10, 15, 17, 29],
  node_sh2: [10, 15, 16],
  probe_gather: [0, 2, 3, 4, 5, 9, 10, 11, 16, 40],   // 40: the environment (sky for rays that leave the scene)
  probe_valid: [0, 4, 10, 11],
  // M39 stage 2 (SH2 probes)
  iso_xfer: [0, 1, 2, 3, 4, 10, 12, 26],
  bounce_sh: [0, 1, 2, 4, 7, 10, 11, 14, 25, 26],
  probe_sh2: [10, 11, 25],
  // near-field reflection (off by default; setNearReflection)
  near_refl: [0, 1, 2, 7, 8, 10, 14, 16, 27, 28],
  // near-field cones (prototype, off by default; setNearCones): replaces near_vis, same output
  near_cones: [0, 1, 2, 3, 4, 9, 10, 12, 27],
};
export const BOUNCE_TIMERS = ['near_vis', 'direct_dirs', 'bounce_dirs', 'node_leaf', 'probe_gather', 'iso_xfer', 'bounce_sh', 'probe_sh2'];
// Emissive Gaussian lights (runtime/wgsl/emissive.wgsl): entry -> bindings, and the timer slot each one reports into.
const EMISSIVE = {
  walk_emit: [[0, 1, 2, 3, 4, 9, 18, 19, 20, 33], 'walk_lamp'],
  shade_emit: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19], 'shade'],
  direct_dirs_emit: [[0, 1, 2, 3, 4, 6, 7, 13, 18, 19], 'direct_dirs'],
};
// Option B field shadows (?shadow=pfd, emissive.wgsl pfd_T): the emissive module with shade_emitters' two lamp_state
// readers replaced (unified per-light T -> pfd_T, the single lamp's T -> 1). Throws if the source no longer matches.
// mode 'rcd' / 'pxd' (occluder proxies, emissive.wgsl rcd_tau / pxd_tau) swaps the field lookup in pfd_T_core;
// 'rcc' / 'pxc' do the same per receiver cluster (cluster_T, pxc_T); colour.w then holds cluster hits x 1024 +
// direct T evaluations (fallbacks).
export function pfdSource(sources, split = true, mode = 'pfd') {
  let src = sources.emissive;
  if (!split) src = src.replace('var<private> pfd_nsel: u32;', 'var<private> pfd_nsel: u32 = PFD_ALL;');
  const tauFn = { rcd: 'rcd', pxd: 'pxd', rcc: 'rcd', pxc: 'pxd' }[mode];
  const clustered = mode === 'rcc' || mode === 'pxc';
  const swap = mode === 'pfd' ? [] : [['tau += max(pf_tau(p, xf_point_inv(xf, P), xf_dir_inv(xf, w), dist), 0.0);',
    `tau += ${tauFn}_tau(p, xf_point_inv(xf, P), xf_dir_inv(xf, w), dist);`]];
  if (clustered) swap.push(['let T = pfd_T_calc(g, P, e, ni);', 'let T = pxc_T(g, P, e, ni);'],
    ['const PXC_NEAR: bool = false;', `const PXC_NEAR: bool = ${new URLSearchParams(globalThis.location?.search ?? '').get('pxcmiss') !== 'walk'};`]);
  for (const [from, to] of [...swap,
    ['light_T(g, e.tag >> 16u)', 'pfd_T(g, r.P, e, ni)'],
    ['select(lamp_transmittance(g, 0u), 1.0, EP.lights != 0u)', '1.0'],
    // the strongest-lights split: pick the nodes that get lookups before the cut walk
    ['-> array<vec3f, 15> {\n    let nd = nodes[g];\n', '-> array<vec3f, 15> {\n    let nd = nodes[g];\n' + (split ? '    pfd_select(g, r, nout);\n' : '')],
    // colour.w (not read downstream), for inspection: parts in the receiver's mask x 65536 + field T evaluations x 1024
    // + parts that passed the per-node world cull
    ['colour[g] = vec4f(c[0], 1.0);', clustered ? 'colour[g] = vec4f(c[0], f32(pxc_hit) * 1024.0 + f32(pfd_nlook));'
      : 'colour[g] = vec4f(c[0], f32(countOneBits(pfd_mask)) * 65536.0 + f32(pfd_nlook) * 1024.0 + f32(pfd_npass));']]) {
    if (src.split(from).length !== 2) throw new Error(`pfd: expected one '${from}' in emissive.wgsl`);
    src = src.replace(from, to);
  }
  return sources.common + '\n' + sources.material_ext + '\n' + src;
}
// Hierarchical shadow walk (runtime/wgsl/shadow_cull.wgsl): entry -> bindings; 38 is the per-depth level uniform.
const SHADOW_CULL = {
  hs_box_leaf: [0, 1, 34],
  hs_box_up: [0, 2, 9, 34, 35, 36, 38],
  shadow_cull: [0, 2, 4, 9, 33, 34, 35, 36, 37, 38],
};
const HS_BUDGET = 4096;   // occluder nodes one hull test may visit before it gives up (the node then walks)
const MAX_MATERIALS = 32;
// Progressive tree lighting: entry -> bindings (progressive.wgsl), worklist size, fade-in frames.
const PROGRESSIVE = {
  lp_schedule: [0, 2, 4, 10, 21, 23, 24],
  lp_finish: [21, 24],
  lp_shade: [0, 2, 4, 9, 10, 11, 18, 19, 21, 22, 23, 24],
  lp_transfer: [0, 1, 3, 4, 7, 8, 14, 19, 21, 22, 23],
  lp_transfer_dirs: [0, 1, 3, 4, 7, 13, 14, 19, 21, 22, 23],
  lp_iso_init: [0, 1, 3, 4, 7, 22],
};
const LP_MAX_BUDGET = 65536, LP_BLEND = 8;
// f32 -> IEEE half bits (for pack2x16float fields written from the host)
// pfield.json's fingerprint (pf_export.py) against the one loadAsset took (asset.fingerprint)
const sameFingerprint = (a, b) => !!a && a.count === b.count && a.bytes === b.bytes && a.hash === b.hash
  && JSON.stringify(a.parts) === JSON.stringify(b.parts);

function fromHalf(h) {
  const e = (h >> 10) & 31, m = h & 1023, s = h & 0x8000 ? -1 : 1;
  return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15);
}

function toHalf(v) {
  const f = new Float32Array([v]), i = new Uint32Array(f.buffer)[0];
  const s = (i >>> 16) & 0x8000, e = ((i >>> 23) & 0xff) - 112, m = i & 0x7fffff;
  if (e <= 0) return s;
  if (e >= 31) return s | 0x7c00;
  return s | (e << 10) | (m >>> 13);
}
const MAX_EMITTERS = 16384, EMITTERS_TARGET = 300, EMIT_THETA = 4;
// Ein(3) = int_0^3 (1 - e^-x) / x dx: the footprint factor of a Gaussian at optical depth 3.
const EIN3 = Array.from({ length: 3000 }, (_, i) => { const x = (i + 0.5) * 0.001; return (1 - Math.exp(-x)) / x * 0.001; })
  .reduce((acc, v) => acc + v, 0);

// ---- emitter tree (host) -------------------------------------------------------------------------------------
// 3x3 symmetric matrices as [xx, yy, zz, xy, xz, yz].
// GPU EmitNode (emissive.wgsl): the 16-float node record, then its 8-float ASG covariance [xx yy zz 0 xy xz yz 0]
function packEmitNodes(nodes, cov) {
  const n = nodes.length / 16, out = new Float32Array(n * 24);
  for (let i = 0; i < n; i++) { out.set(nodes.subarray(i * 16, i * 16 + 16), i * 24); out.set(cov.subarray(i * 8, i * 8 + 8), i * 24 + 16); }
  return out;
}
const ASG_WIDEN = EIN3 / (1 - Math.exp(-3));   // P3: flat-top footprint at optical depth 3 -> one Gaussian
const symDet = ([a, d, f, b, c, e]) => a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
function symInv(m) {
  const [a, d, f, b, c, e] = m, det = symDet(m);
  return [(d * f - e * e) / det, (a * f - c * c) / det, (a * d - b * b) / det,
    (c * e - b * f) / det, (b * e - c * d) / det, (b * c - a * e) / det];
}
const symQuad = ([a, d, f, b, c, e], [x, y, z]) => a * x * x + d * y * y + f * z * z + 2 * (b * x * y + c * x * z + e * y * z);
// Eigenvectors of a symmetric 3x3 by Jacobi rotations (the node's principal axes).
function symAxes(m) {
  const A = [[m[0], m[3], m[4]], [m[3], m[1], m[5]], [m[4], m[5], m[2]]];
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 12; sweep++) {
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(A[p][q]) < 1e-30) continue;
      const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
      const c = Math.cos(th), s = Math.sin(th);
      for (let k = 0; k < 3; k++) { const x = A[k][p], y = A[k][q]; A[k][p] = c * x - s * y; A[k][q] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = A[p][k], y = A[q][k]; A[p][k] = c * x - s * y; A[q][k] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
    }
  }
  return [0, 1, 2].map(k => [V[0][k], V[1][k], V[2][k]]);
}
const lum = c => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

// Binary tree over emitter Gaussian records (16 floats each), median split on the longest axis. A node's radiant
// intensity is Le sqrt(w^T M w): for a leaf M = (2 pi Ein(3))^2 det C C^-1, which is its footprint at optical depth 3;
// a parent's M is diagonal in the frame of its moment-matched covariance, set so its intensity along those three
// axes equals the children's sum, and its Le is their intensity-weighted mean. Returns EmitNode records (16 floats)
// and, per node, the covariance the ASG specular reads (emissive.wgsl EmitCov, 8 floats): a leaf's C widened by
// Ein(3) / (1 - e^-3) (its flat-topped footprint as one Gaussian of the same peak and integral), a parent's the
// children's moment match.
export function buildEmitterTree(rec, n) {
  if (!(n >= 1)) throw new Error(`emitter tree needs at least one emitter (got ${n})`);   // build([]) recurses forever
  const f = new Float32Array((2 * n - 1) * 16), u = new Uint32Array(f.buffer);
  const cov = new Float32Array((2 * n - 1) * 8);
  const info = [];
  let next = 0;
  const leaf = i => {
    const o = i * 16, mu = [rec[o], rec[o + 1], rec[o + 2]];
    const [x, y, z, w] = [rec[o + 4], rec[o + 5], rec[o + 6], rec[o + 7]];
    const R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
    const s2 = [rec[o + 8] ** 2, rec[o + 9] ** 2, rec[o + 10] ** 2];
    const c = (r, k) => R[r][0] * R[k][0] * s2[0] + R[r][1] * R[k][1] * s2[1] + R[r][2] * R[k][2] * s2[2];
    const C = [c(0, 0), c(1, 1), c(2, 2), c(0, 1), c(0, 2), c(1, 2)];
    const g = (2 * Math.PI * EIN3) ** 2 * symDet(C);
    const M = symInv(C).map(v => v * g);
    const mass = rec[o + 3] * (2 * Math.PI) ** 1.5 * Math.sqrt(symDet(C));
    return { mu, C, M, mass, Cs: C.map(v => v * ASG_WIDEN), Le: [rec[o + 12], rec[o + 13], rec[o + 14]],
      radius: 3 * Math.max(rec[o + 8], rec[o + 9], rec[o + 10]), lo: 0xffffffff, hi: i };
  };
  const write = nd => {
    const id = next++, o = id * 16;
    f.set([...nd.mu, nd.radius, nd.M[0], nd.M[1], nd.M[2], 0, nd.M[3], nd.M[4], nd.M[5]], o);
    u[o + 11] = nd.lo;
    f.set(nd.Le, o + 12);
    u[o + 15] = nd.hi;
    cov.set([nd.Cs[0], nd.Cs[1], nd.Cs[2], 0, nd.Cs[3], nd.Cs[4], nd.Cs[5], 0], id * 8);
    info[id] = nd;
    return id;
  };
  const build = idx => {
    if (idx.length === 1) return write(leaf(idx[0]));
    const pos = i => [rec[i * 16], rec[i * 16 + 1], rec[i * 16 + 2]];
    const lo = [0, 1, 2].map(k => Math.min(...idx.map(i => pos(i)[k])));
    const hi = [0, 1, 2].map(k => Math.max(...idx.map(i => pos(i)[k])));
    const ax = [0, 1, 2].reduce((m, k) => (hi[k] - lo[k] > hi[m] - lo[m] ? k : m), 0);
    const sorted = [...idx].sort((p, q) => pos(p)[ax] - pos(q)[ax]);
    const half = sorted.length >> 1;
    const ia = build(sorted.slice(0, half)), ib = build(sorted.slice(half));
    const kids = [info[ia], info[ib]], m = kids[0].mass + kids[1].mass;
    const mu = [0, 1, 2].map(k => kids.reduce((acc, ch) => acc + ch.mass * ch.mu[k], 0) / m);
    const C = [0, 1, 2, 3, 4, 5].map(j => {
      const [r, k] = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]][j];
      return kids.reduce((acc, ch) => acc + ch.mass * (ch.C[j] + (ch.mu[r] - mu[r]) * (ch.mu[k] - mu[k])), 0) / m;
    });
    const Cs = [0, 1, 2, 3, 4, 5].map(j => {
      const [r, k] = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]][j];
      return kids.reduce((acc, ch) => acc + ch.mass * (ch.Cs[j] + (ch.mu[r] - mu[r]) * (ch.mu[k] - mu[k])), 0) / m;
    });
    const axes = symAxes(C);
    // children's luminance intensity along each axis; the parent matches their sum there
    const along = (ch, w) => lum(ch.Le) * Math.sqrt(Math.max(symQuad(ch.M, w), 0));
    const tot = axes.map(w => kids.reduce((acc, ch) => acc + along(ch, w), 0));
    const wsum = tot.reduce((x, y) => x + y, 0);
    const Le = [0, 1, 2].map(c => kids.reduce((acc, ch) => acc + ch.Le[c] * axes.reduce((s2, w) => s2 + Math.sqrt(Math.max(symQuad(ch.M, w), 0)), 0), 0)
      / kids.reduce((acc, ch) => acc + axes.reduce((s2, w) => s2 + Math.sqrt(Math.max(symQuad(ch.M, w), 0)), 0), 0));
    const L = Math.max(lum(Le), 1e-12);
    const M = [0, 0, 0, 0, 0, 0];
    axes.forEach((w, k) => {
      const lam = (tot[k] / L) ** 2;
      const idx2 = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]];
      idx2.forEach(([r, c], j) => { M[j] += lam * w[r] * w[c]; });
    });
    const radius = Math.max(...kids.map(ch => Math.hypot(...ch.mu.map((v, k) => v - mu[k])) + ch.radius));
    return write({ mu, C, Cs, M, mass: m, Le, radius, lo: ia, hi: ib, wsum });
  };
  const root = build(Array.from({ length: n }, (_, i) => i));
  return { nodes: f, root, cov };
}

// Rotation matrix columns (c0, c1, c2) -> quaternion x, y, z, w.
function quatFromColumns(c0, c1, c2) {
  const [m00, m10, m20] = c0, [m01, m11, m21] = c1, [m02, m12, m22] = c2;
  const t = m00 + m11 + m22;
  let x, y, z, w;
  if (t > 0) { const k = 0.5 / Math.sqrt(t + 1); w = 0.25 / k; x = (m21 - m12) * k; y = (m02 - m20) * k; z = (m10 - m01) * k; }
  else if (m00 > m11 && m00 > m22) { const k = 2 * Math.sqrt(1 + m00 - m11 - m22); w = (m21 - m12) / k; x = 0.25 * k; y = (m01 + m10) / k; z = (m02 + m20) / k; }
  else if (m11 > m22) { const k = 2 * Math.sqrt(1 + m11 - m00 - m22); w = (m02 - m20) / k; x = (m01 + m10) / k; y = 0.25 * k; z = (m12 + m21) / k; }
  else { const k = 2 * Math.sqrt(1 + m22 - m00 - m11); w = (m10 - m01) / k; x = (m02 + m20) / k; y = (m12 + m21) / k; z = 0.25 * k; }
  const l = Math.hypot(x, y, z, w);
  return [x / l, y / l, z / l, w / l];
}
// ProbeParams: 832 B, then dims (16), levels (64), ground (4096), static volumes (512 + 16) (bounce.wgsl; SPEC 2.7)
// + node probes (256 vec4f, ?nodeprobes=1)
const PROBE_PARAMS_BYTES = 832 + 16 + 64 + 4096 + 528 + 4096;
// Node probes (David, 2026-10-28): a movable part's probes at the patches of its own split instead of a 2x2x2 cube
// inside it (bounce.wgsl NODE_PROBES). ?nodeprobes=1 (prototype; progressive tree lighting still assumes 8 per part)
const NODE_PROBES = new URLSearchParams(globalThis.location?.search ?? '').get('nodeprobes') === '1';
const PPK = NODE_PROBES ? 16 : 8;   // probes per movable part
export function nodeProbeSource(bounce) {
  return NODE_PROBES ? bounce.replace('const PART_PROBES_PER: u32 = 8u;', `const PART_PROBES_PER: u32 = ${PPK}u;`)
    .replace('const NODE_PROBES: bool = false;', 'const NODE_PROBES: bool = true;') : bounce;
}

// Per part, PPK probes in its rest frame: the part's Gaussian centres split (largest-variance axis, at the median)
// until PPK groups; per group the flattest direction of its spread, oriented away from the part's centre, and the
// probe 3 cm beyond the group's furthest centre along it; w = the blend radius (0.6 x the mean distance to the
// nearest other probe of the part). Returns Float32Array [parts x PPK x 4].
function nodeProbes(asset, k = PPK) {
  const gf = new Float32Array(asset.gaussians), parts = asset.header.parts;
  const out = new Float32Array(256 * 4);
  parts.forEach((p, q) => {
    if (q * k + k > 256) return;
    const [g0, g1] = p.gaussians;
    if (g1 - g0 < k) return;
    const idx = Array.from({ length: g1 - g0 }, (_, i) => g0 + i);
    const P = i => [gf[i * 16], gf[i * 16 + 1], gf[i * 16 + 2]];
    const mean = ids => { const m = [0, 0, 0]; for (const i of ids) { const x = P(i); m[0] += x[0]; m[1] += x[1]; m[2] += x[2]; } return m.map(v => v / ids.length); };
    const cov = (ids, m) => { const c = [0, 0, 0, 0, 0, 0]; for (const i of ids) { const x = P(i), d = [x[0] - m[0], x[1] - m[1], x[2] - m[2]];
      c[0] += d[0] * d[0]; c[1] += d[1] * d[1]; c[2] += d[2] * d[2]; c[3] += d[0] * d[1]; c[4] += d[0] * d[2]; c[5] += d[1] * d[2]; } return c.map(v => v / ids.length); };
    // first by facing: each Gaussian's short axis, signed away from the part's centre, binned into the 6 directions of the
    // part's own principal axes (so a box's faces fall into separate groups, not corner pieces across its edges)
    const pc0 = mean(idx), cp = cov(idx, pc0);
    const Mp = [[cp[0], cp[3], cp[4]], [cp[3], cp[1], cp[5]], [cp[4], cp[5], cp[2]]];
    const axes = [];
    for (let a = 0; a < 3; a++) {                                   // principal axes by power iteration + deflation
      let v = [[1, 0.3, 0.2], [0.2, 1, 0.3], [0.3, 0.2, 1]][a];
      for (let it = 0; it < 60; it++) {
        let w = [0, 1, 2].map(r => Mp[r][0] * v[0] + Mp[r][1] * v[1] + Mp[r][2] * v[2]);
        for (const b of axes) { const d = w[0] * b[0] + w[1] * b[1] + w[2] * b[2]; w = w.map((x, c) => x - d * b[c]); }
        const l = Math.hypot(...w) || 1; v = w.map(x => x / l);
      }
      axes.push(v);
    }
    const nrm = i => {
      const q = [gf[i * 16 + 4], gf[i * 16 + 5], gf[i * 16 + 6], gf[i * 16 + 7]], [x, y, z, w] = q;
      const n = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)];
      const d = [gf[i * 16] - pc0[0], gf[i * 16 + 1] - pc0[1], gf[i * 16 + 2] - pc0[2]];
      return n[0] * d[0] + n[1] * d[1] + n[2] * d[2] < 0 ? n.map(v => -v) : n;
    };
    const bins = Array.from({ length: 6 }, () => []);
    for (const i of idx) {
      const n = nrm(i); let best = -2, bi = 0;
      axes.forEach((a, j) => { const d = n[0] * a[0] + n[1] * a[1] + n[2] * a[2]; if (d > best) { best = d; bi = 2 * j; } if (-d > best) { best = -d; bi = 2 * j + 1; } });
      bins[bi].push(i);
    }
    const groups = bins.filter(b => b.length >= Math.max(8, idx.length / (4 * k)));
    for (const b of bins) if (b.length && !groups.includes(b)) groups[0]?.push(...b);   // tiny bins join the first
    while (groups.length > k) groups.sort((a, b) => b.length - a.length).splice(-2, 2, [...groups.at(-2), ...groups.at(-1)]);
    while (groups.length < k) {
      let gi = 0; groups.forEach((g, i) => { if (g.length > groups[gi].length) gi = i; });
      const g = groups[gi], m = mean(g), c = cov(g, m);
      const ax = [c[0], c[1], c[2]].indexOf(Math.max(c[0], c[1], c[2]));
      g.sort((a, b) => gf[a * 16 + ax] - gf[b * 16 + ax]);
      groups.splice(gi, 1, g.slice(0, g.length >> 1), g.slice(g.length >> 1));
    }
    const pc = mean(idx), pos = [];
    for (const g of groups) {
      const m = mean(g), c = cov(g, m);
      const M = [[c[0], c[3], c[4]], [c[3], c[1], c[5]], [c[4], c[5], c[2]]];
      // flattest direction: power iteration on (tr I - M)
      const tr = c[0] + c[1] + c[2];
      let v = [1, 0.7, 0.3];
      for (let it = 0; it < 60; it++) {
        const w = [0, 1, 2].map(r => tr * v[r] - (M[r][0] * v[0] + M[r][1] * v[1] + M[r][2] * v[2]));
        const l = Math.hypot(...w) || 1; v = w.map(x => x / l);
      }
      // facing: the group's mean signed short axis (falls back to the flattest direction, signed away from the centre)
      const nm = [0, 0, 0]; for (const i of g) { const n = nrm(i); nm[0] += n[0]; nm[1] += n[1]; nm[2] += n[2]; }
      const ln = Math.hypot(...nm);
      if (ln > 0.3 * g.length) v = nm.map(x => x / ln);
      else {
        const away = [m[0] - pc[0], m[1] - pc[1], m[2] - pc[2]];
        if (v[0] * away[0] + v[1] * away[1] + v[2] * away[2] < 0) v = v.map(x => -x);
      }
      let ext = 0;
      for (const i of g) { const x = P(i); ext = Math.max(ext, (x[0] - m[0]) * v[0] + (x[1] - m[1]) * v[1] + (x[2] - m[2]) * v[2]); }
      pos.push(m.map((x, a) => x + v[a] * (ext + 0.03)));
    }
    pos.forEach((x, j) => {
      let dmin = Infinity;
      pos.forEach((y, l) => { if (l !== j) dmin = Math.min(dmin, Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2])); });
      out.set([...x, 0], 4 * (q * k + j));
      out[4 * (q * k + j) + 3] = 0.6 * (isFinite(dmin) ? dmin : 0.3);
    });
  });
  // one blend radius per part (the mean), so neighbouring probes overlap evenly
  parts.forEach((p, q) => {
    if (q * k + k > 256) return;
    let s = 0; for (let j = 0; j < k; j++) s += out[4 * (q * k + j) + 3];
    for (let j = 0; j < k; j++) out[4 * (q * k + j) + 3] = s / k;
  });
  return out;
}
const PROBES_AXIS = 8, NCELL = 128, HYSTERESIS = 0.5, PROBE_THETA = 4, TMAX = 6, NEAR_R = 0.3;
// Lighting converges within a frame or two of a change (David, 2026-10-28: the one-frame convergence rule): every probe
// and every bounce_dirs Gaussian refreshes every frame (?probesubset=4 = the old quarter a frame, ~2-5 ms cheaper on the
// M4, 8 frames instead of 2 to settle a box jump), and each probe gather is taken as is (the gather is deterministic, so
// a blend only delays). ?probeblend=ema the old EMA with HYSTERESIS, ?probeblend=fir a tapered blend of the last 3
// gathers (bounce.wgsl FIR_TAPS). renderer/results/converge/README.md
const SUBSET = Math.max(1, Number(new URLSearchParams(globalThis.location?.search ?? '').get('probesubset')) || 1);
const PROBE_BLEND = new URLSearchParams(globalThis.location?.search ?? '').get('probeblend') ?? 'none';
const probeAlpha = first => first ? 1 : PROBE_BLEND === 'fir' ? -1 : PROBE_BLEND === 'none' ? 1 : 1 - HYSTERESIS;
const FULL_ROUNDS = 8;   // ...after the first rounds, which refresh everything so the probes settle in a few frames
const NEAR_CHUNK = 16384;   // with near-field reflection a forced near_vis refresh runs this many Gaussians a frame (~0.3 s on the M4)

// D14 (6 axes + 8 diagonals) and the least-squares SH2 fit of values on them (loop_surf.py's pinv(sh2(D14))).
// The vertical axes lean 1e-4 rad off the pole: vgrid_lookup's atan2(0, 0) is NaN on Metal.
const D14 = [[1, 0, 0], [0, 1, 0], [1e-4, 0, 1], [-1, 0, 0], [0, -1, 0], [1e-4, 0, -1],
  ...[1, -1].flatMap(x => [1, -1].flatMap(y => [1, -1].map(z => [x, y, z].map(v => v / Math.sqrt(3)))))];
const sh2 = ([x, y, z]) => [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z,
  0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)];
function pinvD14() {
  const A = D14.map(sh2);                       // 14 x 9
  const M = Array.from({ length: 9 }, (_, i) => [...Array.from({ length: 9 }, (_, j) => A.reduce((s, r) => s + r[i] * r[j], 0)),
    ...Array.from({ length: 9 }, (_, j) => +(i === j))]);
  for (let c = 0; c < 9; c++) {                 // Gauss-Jordan on [AtA | I]
    let piv = c;
    for (let r = c + 1; r < 9; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c];
    for (let j = 0; j < 18; j++) M[c][j] /= d;
    for (let r = 0; r < 9; r++) if (r !== c) { const f = M[r][c]; for (let j = 0; j < 18; j++) M[r][j] -= f * M[c][j]; }
  }
  const inv = M.map(r => r.slice(9));
  return Array.from({ length: 9 }, (_, m) => A.map(row => row.reduce((s, v, j) => s + inv[m][j] * v, 0)));   // 9 x 14
}
const WG = 64;
const B = GPUBufferUsage;
const EMPTY_SLOT = 0xffff;
// M35 / scenarios.py defaults.
const THETA = 8, ALPHA_MIN = 1 / 16, ALPHA_PART = 0.5, STRATUM_STEP = 7;

// SPEC 2.7 probe layout. Room (Cornell, asset.json `room`): an 8^3 lattice inset 0.125 m from the walls, i.e. ground 0
// and evenly spaced levels. Terrain (asset.json `probes.layout = "terrain"`, or an asset with `bounds` and no `room`):
// dims x / y columns over the bounds, each with dims z probes at fixed heights above the terrain under it (the highest
// Gaussian centre of the terrain parts in its cell), so probes hug the ground and thin out upward; no probes in open
// air. The terrain parts are `probes.terrain_parts`, else the static part with the widest footprint.
export function probeLayout(asset) {
  const h = asset.header, pr = h.probes ?? {};
  const terrain = pr.layout === 'terrain' || (!h.room && !!h.bounds && pr.layout !== 'room');
  const box = h.room ?? (h.bounds ? { x: [h.bounds.lo[0], h.bounds.hi[0]], y: [h.bounds.lo[1], h.bounds.hi[1]], z: [h.bounds.lo[2], h.bounds.hi[2]] } : null);
  if (!box) throw new Error('asset.json has neither room nor bounds (probe lattice)');
  const ax = ['x', 'y', 'z'].map(k => box[k]);
  // probe rays reach TMAX, or across the whole scene when it has an environment (rays that leave it see the sky)
  const tmax = h.environment ? Math.max(TMAX, Math.hypot(...ax.map(([a, b]) => b - a))) : TMAX;
  if (!terrain) {
    const n = PROBES_AXIS;
    const cen = ax.map(([a, b]) => (a + b) / 2), half = ax.map(([a, b]) => (b - a) / 2 - 0.125);
    const lo = cen.map((c, k) => c - half[k]), step = half.map(v => 2 * v / (n - 1));
    return { terrain: false, dims: [n, n, n], lo, step, levels: Array.from({ length: n }, (_, k) => lo[2] + k * step[2]),
      ground: new Float32Array(n * n), tmax, vols: [] };
  }
  const dims = [...(pr.dims ?? [16, 16, 5])];
  if (dims[0] * dims[1] > 1024 || dims[2] > 16) throw new Error('probe terrain layout: at most 1024 columns and 16 levels');
  const levels = pr.heights_m ?? [0.3, 1, 3, 8, 20];
  dims[2] = levels.length;
  const step = [0, 1].map(k => (ax[k][1] - ax[k][0]) / dims[k]);
  const lo = [ax[0][0] + step[0] / 2, ax[1][0] + step[1] / 2, 0];
  // terrain parts: named, else the static part with the widest footprint
  const parts = h.parts;
  let tp = (pr.terrain_parts ?? []).map(n => parts.findIndex(p => p.name === n)).filter(i => i >= 0);
  if (!tp.length) {
    const gf = new Float32Array(asset.gaussians);
    let best = -1, bestA = -1;
    parts.forEach((p, i) => {
      if (p.mobility === 'movable') return;
      const [g0, g1] = p.gaussians;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let g = g0; g < g1; g += Math.max(1, (g1 - g0) >> 12)) {
        const x = gf[g * 16], y = gf[g * 16 + 1];
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
      const A = (x1 - x0) * (y1 - y0);
      if (A > bestA) { bestA = A; best = i; }
    });
    tp = best >= 0 ? [best] : [];
  }
  const gf = new Float32Array(asset.gaussians);
  const ground = new Float32Array(dims[0] * dims[1]).fill(-Infinity);
  for (const i of tp) {
    const [g0, g1] = parts[i].gaussians;
    for (let g = g0; g < g1; g++) {
      const ci = Math.floor((gf[g * 16] - ax[0][0]) / step[0]), cj = Math.floor((gf[g * 16 + 1] - ax[1][0]) / step[1]);
      if (ci < 0 || cj < 0 || ci >= dims[0] || cj >= dims[1]) continue;
      const q = ci * dims[1] + cj;
      ground[q] = Math.max(ground[q], gf[g * 16 + 2]);
    }
  }
  // empty columns take their nearest filled neighbour's height (rings outward), else the bounds' floor
  for (let pass = 0; pass < dims[0] + dims[1] && ground.some(v => v === -Infinity); pass++) {
    const prev = ground.slice();
    for (let i = 0; i < dims[0]; i++) for (let j = 0; j < dims[1]; j++) {
      if (prev[i * dims[1] + j] !== -Infinity) continue;
      let s = 0, c = 0;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const a = i + di, b = j + dj;
        if (a < 0 || b < 0 || a >= dims[0] || b >= dims[1] || prev[a * dims[1] + b] === -Infinity) continue;
        s += prev[a * dims[1] + b]; c++;
      }
      if (c) ground[i * dims[1] + j] = s / c;
    }
  }
  for (let q = 0; q < ground.length; q++) if (ground[q] === -Infinity) ground[q] = ax[2][0];
  return { terrain: true, dims, lo, step: [...step, 0], levels, ground, tmax, vols: probeVolumes(asset, tp, ax) };
}

// Probe volumes on static objects (SPEC 2.7; probes skip the object and serve only its Gaussians): asset.json
// `probes.volumes` [{ part, dims }], else (terrain layout) every
// static part that is not terrain and whose bounding radius is 1 m to a quarter of the scene's half-diagonal (a tree,
// a rock, a house; not the ground or a grass carpet): a grid of ~1.2 m spacing, 2-6 per axis.
function probeVolumes(asset, terrainParts, ax) {
  const h = asset.header, pr = h.probes ?? {};
  const pf = asset.parts ? new Float32Array(asset.parts) : null, pu = asset.parts ? new Uint32Array(asset.parts) : null;
  const out = [];
  if (new URLSearchParams(globalThis.location?.search ?? '').get('probevols') === '0') return out;   // comparisons
  if (pr.volumes) {
    for (const v of pr.volumes) {
      const q = h.parts.findIndex(p => p.name === v.part);
      if (q >= 0 && q < 32) out.push({ part: q, dims: v.dims });
    }
    return out;
  }
  if (!pf) return out;
  const halfDiag = 0.5 * Math.hypot(...ax.map(([a, b]) => b - a));
  h.parts.forEach((p, q) => {
    if (q >= 32 || terrainParts.includes(q)) return;
    const isStatic = p.mobility ? p.mobility !== 'movable' : (pu[q * 32 + 31] & 1) !== 0;
    const R = pf[q * 32 + 27];
    if (!isStatic || R < 1 || R > 0.25 * halfDiag) return;
    const n = Math.max(2, Math.min(6, Math.round(R / 1.2)));
    out.push({ part: q, dims: [n, n, n] });
  });
  return out;
}

export class Lighting {
  constructor(device, sources, asset, gaussBuf, partBuf) {
    this.device = device;
    this.count = asset.count;
    this.partCount = asset.partCount;
    this.lampCount = asset.header.buffers.lamps.count;
    this.materialCount = asset.header.buffers.materials.count + (asset.header.parts.filter(p => ['short_box', 'tall_box', 'floor'].includes(p.name)).length);
    this.lightSamplesAxis = asset.header.shading?.light_samples_axis ?? 4;
    this.frame = 0;
    const buf = (label, data, usage = B.STORAGE) => {
      const b = device.createBuffer({ label, size: Math.max(16, data.byteLength ?? data), usage: usage | B.COPY_DST | B.COPY_SRC });
      if (data.byteLength !== undefined) device.queue.writeBuffer(b, 0, data);
      return b;
    };
    const r = asset.raw;
    // Unified lights (?lights=unified, lights.js): every emissive object is a light, none special; lamps.bin then
    // holds one sphere record per light (room for MAX_LIGHTS)
    const q = new URLSearchParams(globalThis.location?.search ?? ''), lod = q.get('lightlod');
    this.unified = q.get('lights') === 'unified';
    if (this.unified) { this.lightSet = new Lights(asset, lod == null ? undefined : Number(lod)); this.lampCount = Math.max(this.lightSet.lights.length, 1); }
    this.lampBytes = new ArrayBuffer(Math.max(r.lamps.byteLength, this.unified ? MAX_LIGHTS * 96 : 0));
    new Uint8Array(this.lampBytes).set(new Uint8Array(r.lamps));
    this.baseLamps = r.lamps.slice(0);
    this.gaussianBytes = asset.gaussians;
    this.partRanges = asset.header.parts.map(p => p.gaussians);
    const em = asset.header.emitters;
    if (em) {
      const mf = new Uint32Array(r.materials)[em.material * 8 + 5];
      this.assetEmitters = { part: em.part, emission: em.emission, oneSided: (mf & 4) !== 0 };
    }
    this.emitOneSided = this.assetEmitters?.oneSided ?? false;
    // no lamp: the asset ships only a disabled placeholder (asset.json lamps[0].enabled false, e.g. hill: the sun
    // lights it). The lamp passes are skipped and the emitter panel is one dark emitter, so the light cut has a root.
    this.lampOff = !this.unified && !this.assetEmitters && asset.header.lamps?.[0]?.enabled === false;
    // Room for the per-box materials the renderer adds (see setBoxMaterial): the asset's, then one per box part.
    this.materialBytes = new ArrayBuffer(Math.max(MAX_MATERIALS, asset.header.buffers.materials.count + 3) * 32);
    new Uint8Array(this.materialBytes).set(new Uint8Array(r.materials));
    this.assetMaterialCount = asset.header.buffers.materials.count;
    this.materialInfo = asset.header.materials ?? [];
    // materials asset.json marks `ui: true` get live shine / roughness sliders; start from what the asset stores
    const mf = new Float32Array(r.materials), mu = new Uint32Array(r.materials);
    const fromHalf = h => { const e = (h >> 10) & 31, m = h & 1023, s = h & 0x8000 ? -1 : 1;
      return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15); };
    this.uiMaterials = this.materialInfo.filter(mi => mi.ui).map(mi => ({
      index: mi.index, name: mi.name ?? `material ${mi.index}`,
      shine: mu[mi.index * 8 + 5] & 8 ? fromHalf(mu[mi.index * 8 + 6] & 0xffff) : 0,
      // M41 transmission as the asset stores it (asset.json translucency / tint, else flags bit 4 + pad1 as setMaterialTransmit writes)
      transl: mi.translucency ?? (mu[mi.index * 8 + 5] & 16 ? 1 - fromHalf(mu[mi.index * 8 + 7] >>> 16) : 0),
      tint: mi.tint ?? (mu[mi.index * 8 + 5] & 16 ? fromHalf(mu[mi.index * 8 + 7] & 0xffff) : 0),
      rough: Math.sqrt(Math.max(mf[mi.index * 8 + 1], 1e-4)),
    }));
    // what the asset's own materials use (main.js turns the matching View toggles on): transmission (flags bit 4,
    // not emitters) and reflection coverage (bit 5), or asset.json `features`
    const amf = Array.from({ length: this.assetMaterialCount }, (_, i) => mu[i * 8 + 5]);
    this.assetFeatures = {
      transmit: amf.some(f => (f & 16) && !(f & 1)),
      coverage: amf.some(f => f & 32) || !!asset.header.features?.coverage,
      fresnel: amf.some(f => f & 32) || !!asset.header.features?.fresnel,
    };
    this.boxMaterial = {};
    for (const name of ['short_box', 'tall_box', 'floor']) {
      const p = asset.header.parts.findIndex(q => q.name === name);
      if (p < 0) continue;
      const m = this.assetMaterialCount + Object.keys(this.boxMaterial).length;
      new Uint8Array(this.materialBytes, m * 32, 32).set(new Uint8Array(r.materials, 0, 32));   // start from material 0
      this.boxMaterial[name] = m;
      // point the box's Gaussians at it (flags low 16 bits, SPEC 2.1 / 2.6)
      const [g0, g1] = asset.header.parts[p].gaussians;
      const gu = new Uint32Array(asset.gaussians);
      for (let g = g0; g < g1; g++) gu[g * 16 + 15] = (gu[g * 16 + 15] & 0xffff0000) | m;
    }
    // unified: Gaussian flags bit 16 marks emissive Gaussians (their material's flags bit 0) for walk_tau_light, so
    // walk_emit needs no materials binding (SPEC 2.1 keeps bits 16-31 reserved; every reader masks the low 16)
    if (this.unified) {
      const gu = new Uint32Array(asset.gaussians), mfl = new Uint32Array(this.materialBytes);
      for (let g = 0; g < this.count; g++) {
        const m = gu[g * 16 + 15] & 0xffff;
        gu[g * 16 + 15] = (mfl[m * 8 + 5] & 1 ? gu[g * 16 + 15] | 0x10000 : gu[g * 16 + 15] & ~0x10000) >>> 0;
      }
    }
    // screen-space LOD (lod.js; main.js sets asset.lodNodes): internal tree nodes become lit receivers too
    // (?lodlight=1): V grids, lamp_state and colour get room for them
    const lodN = asset.lodNodes ? asset.lodNodes.N - this.count : 0;
    this.lodN = lodN;
    this.lodFlagBuf = asset.lodFlagBuf ?? null;
    // ?lodlight=1: drawn nodes lit directly (their own geometry, their leaves' mean shadow T and bounce), every leaf
    // still shaded; 2: only the cut is shaded (lod.wgsl lod_list, indirect dispatch). Every light the emissive model
    // has: emitters with any shadow method, unified lights, and the environment's distant lights
    const lodQ = new URLSearchParams(globalThis.location?.search ?? '').get('lodlight');
    this.lodLight = lodN > 0 && (lodQ === '1' || lodQ === '2');
    this.lodCut = this.lodLight && lodQ === '2';
    let vgridBytes = r.vgrid;
    if (lodN) {
      const v = new Uint8Array(r.vgrid.byteLength + lodN * 128);
      v.set(new Uint8Array(r.vgrid)); v.set(new Uint8Array(asset.lodNodes.vgrid, 0, lodN * 128), r.vgrid.byteLength);
      vgridBytes = v.buffer;
    }
    this.buffers = {
      uniform: buf('frame uniforms', 64, B.UNIFORM),
      gaussians: gaussBuf, parts: partBuf,
      nodes: buf('nodes', r.nodes), kids: buf('kids', r.kids), vgrid: buf('vgrid', vgridBytes),
      lamps: buf('lamps', this.lampBytes), materials: buf('materials', this.materialBytes),
      samples: buf('samples', this.count * 16),
      lamp_state: buf('lamp_state', (this.count + lodN) * 32),
      // bit 0 shadow_cull's clear, bit 1 ema's walk-now for the amortized walk_emit (shadow_cull.wgsl, ema.wgsl)
      node_status: buf('shadow node status', asset.header.buffers.nodes.count * 4),
      // second half: the specular layer toward the eye with reflection coverage on (emissive.wgsl shade_emit, U.iso bit 4)
      // + two vec4 per internal tree node with ?lod= (lod.wgsl lod_colour: a node's shown colour at 2 G + (n - G), its
      // mean bounce at 2 G + (N - G) + (n - G))
      colour: buf('colour', (2 * this.count + 2 * lodN) * 16),
      // SPEC 2.9 environment (environment.js writes it; zeros = no sky, no distant lights) and its shadow T per Gaussian
      env: buf('environment', 416, B.UNIFORM),
      // second half: the static occluders' T, kept for static receivers while movable parts move (environment.wgsl)
      env_T: buf('distant-light shadow T', new Float32Array(this.count * 8 + lodN * 4).fill(1)),   // + LOD nodes' mean T (lod.wgsl)
    };
    if (this.lodCut) {   // lod.wgsl lod_list: header (counts, indirect sizes), then the leaf and node lists; fill's marks
      this.lodList = device.createBuffer({ label: 'lod shade list', size: (16 + this.count + lodN) * 4,
        usage: B.STORAGE | B.INDIRECT | B.COPY_DST | B.COPY_SRC });
      this.lodNeed = buf('lod fill sources', this.count * 4);
    }
    this.resetHistory();
    // the box Gaussians now point at their own materials, and unified marks emissive ones (above): re-upload the records
    if (Object.keys(this.boxMaterial).length || this.unified) device.queue.writeBuffer(gaussBuf, 0, asset.gaussians);
    // World-space motion blur for the lighting (?blur=1; ?blurlight=0 keeps it display-only; motion blur thread):
    // blur_records (motion_blur.wgsl) rewrites copies of the Gaussian and node records each frame, stretched along each
    // moving part's motion. What other parts see reads the copies: the opacity map's splat (light_shadow.js
    // lsm_splat), the shadow walks (walk_emit*) and the probe gather, so a moving part casts a smeared shadow and
    // bounce while its own shading keeps its rest shape (David, 2026-10-28). ?blurlight=all binds the copies to every
    // lighting kernel instead (the first prototype: a moving part's own shading terms see its smear too).
    const bq = new URLSearchParams(globalThis.location?.search ?? '');
    this.blurShutter = bq.get('blur') !== '0' && bq.get('blurlight') !== '0' ? Number(bq.get('shutter') ?? 0.5) : 0;   // on by default (David, 2026-10-29)
    // a moving part casts its shadows (opacity map, sun walk) from a cut of nodes about k x its smear across, each
    // keeping its coverage (light_shadow.wgsl lsm_splat, environment.wgsl walk_tau); default k 1 (David, 2026-10-29),
    // ?blurcut=0 stretches every Gaussian keeping its mass (darkens the swept strip of an opaque part)
    this.blurCut = Math.max(0, Number(bq.get('blurcut') ?? 1) || 0);
    if (this.blurShutter > 0 && sources.motion_blur) {
      const nodeCount = r.nodes.byteLength / 96;
      const rest = { gaussians: gaussBuf, nodes: this.buffers.nodes };
      this.buffers.gaussians_blur = buf('gaussians (motion blur)', asset.gaussians);
      this.buffers.nodes_blur = buf('nodes (motion blur)', r.nodes);
      if (bq.get('blurlight') === 'all') Object.assign(this.buffers, { gaussians: this.buffers.gaussians_blur, nodes: this.buffers.nodes_blur });
      const params = new ArrayBuffer(16);
      new Float32Array(params)[0] = this.blurShutter;
      new Uint32Array(params).set([this.count, nodeCount, 0], 1);
      const ub = buf('motion blur params', params, B.UNIFORM);
      // each part's runs of record indices (leaves are contiguous per part; internal nodes nearly so)
      const nuB = new Uint32Array(r.nodes), partRuns = [];
      for (let i = 0; i < nodeCount;) {
        const p = nuB[i * 24 + 15];
        let j = i + 1;
        while (j < nodeCount && nuB[j * 24 + 15] === p) j++;
        if (p !== 0xffffffff) (partRuns[p] ??= []).push([i, j]);
        i = j;
      }
      const maxRuns = Math.max(1, partRuns.reduce((n, l) => n + (l?.length ?? 0), 0));
      const ranges = buf('motion blur ranges', maxRuns * 16);
      const module = device.createShaderModule({ label: 'blur_records', code: sources.common + '\n' + sources.motion_blur });
      const pipe = device.createComputePipeline({ label: 'blur_records', layout: 'auto', compute: { module, entryPoint: 'blur_records' } });
      const group = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [ub, partBuf, rest.gaussians, rest.nodes,
        this.buffers.gaussians_blur, this.buffers.nodes_blur, ranges].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      this.blur = { pipe, group, ub, params, ranges, partRuns };
      this.prevMovedMask = 0;
      this.blurMask = 0;
    }

    const bindings = {
      walk_lamp: ['uniform', 'gaussians', 'nodes', 'kids', 'vgrid', 'parts', 'lamps', 'samples'],
      ema: ['uniform', 'gaussians', 'parts', 'lamps', 'samples', 'lamp_state', 'node_status'],
      shade: ['uniform', 'gaussians', 'nodes', 'vgrid', 'parts', 'lamps', 'lamp_state', 'materials', 'colour'],
    };
    this.pipelines = {};
    this.groups = {};
    for (const [file, entry] of KERNELS) {
      const module = device.createShaderModule({ label: entry, code: sources.common + '\n' + sources[file] });
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      this.pipelines[entry] = pipe;
      this.groups[entry] = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bindings[entry].map((name, binding) => ({ binding, resource: { buffer: this.buffers[name] } })),
      });
    }
    if (this.unified) {   // ema.wgsl ema_light: shadow history per light (unified lights), pooled over neighbours
      this.buffers.pool = buf('shadow pool', this.poolNeighbours(asset, q.get('shadowpool') !== '0'));
      bindings.ema_light = [...bindings.ema, 'pool'];
      const module = device.createShaderModule({ label: 'ema_light', code: sources.common + '\n' + sources.ema });
      const pipe = device.createComputePipeline({ label: 'ema_light', layout: 'auto', compute: { module, entryPoint: 'ema_light' } });
      this.pipelines.ema_light = pipe;
      this.groups.ema_light = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bindings.ema_light.map((name, binding) => ({ binding, resource: { buffer: this.buffers[name] } })),
      });
    }
    this.uniform = new ArrayBuffer(64);
    this.iso = true;    // M39 direction-independent transfer by default (David, 2026-10-24); the View panel switches it off
    this.shProbes = true;   // M39 stage 2: SH2 probes (only with iso)
    this.transmit = false;  // M41 transmission (only with iso), off by default
    this.specSplit = false; // reflection coverage / Fresnel (main.js): specular toward the eye in colour[G + g] and lo_b w lanes
    this.sheen = false;     // Fresnel's probe sheen for matte materials (bounce_sh, lo_b w lanes)
    this.initIso(device, sources);
    this.initBounce(device, sources, asset);
    this.initShadowCull(device, sources, asset);
    this.initEmissive(device, sources);
    this.initProgressive(device, sources, asset);
  }

  // M39: each leaf's iso_norm (exposed area from its V grid) into Node.pad0, once per asset (iso_init.wgsl), and the
  // M41 per-channel iso_norm of transmitting materials into Node.pad1 (again whenever setMaterialTransmit changes one).
  initIso(device, sources) {
    const module = device.createShaderModule({ label: 'iso_init', code: sources.common + '\n' + sources.iso_init });
    // one explicit layout for both entry points: a bind group made from an 'auto' layout only fits its own pipeline
    const C = GPUShaderStage.COMPUTE, ro = { type: 'read-only-storage' };
    const bgl = device.createBindGroupLayout({ label: 'iso_init', entries: [
      { binding: 0, visibility: C, buffer: { type: 'uniform' } }, { binding: 1, visibility: C, buffer: ro },
      { binding: 2, visibility: C, buffer: { type: 'storage' } }, { binding: 3, visibility: C, buffer: ro },
      { binding: 4, visibility: C, buffer: ro }] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const pipe = device.createComputePipeline({ label: 'iso_init', layout, compute: { module, entryPoint: 'iso_init' } });
    const group = device.createBindGroup({ layout: bgl, entries: ['uniform', 'gaussians', 'nodes', 'vgrid', 'materials']
      .map((name, binding) => ({ binding, resource: { buffer: this.buffers[name] } })) });
    this.isoInit = { pipe, group };
    if (this.lodN) this.isoInit.nodes = device.createComputePipeline({ label: 'iso_init_nodes', layout, compute: { module, entryPoint: 'iso_init_nodes' } });
    this.runIsoInit();
  }

  runIsoInit() {
    const u = new Uint32Array(16); u[1] = this.count; u[13] = this.materialCount;
    this.device.queue.writeBuffer(this.buffers.uniform, 0, u);   // the next frame's writeUniform restores the rest
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass({ label: 'iso_init' });
    pass.setPipeline(this.isoInit.pipe); pass.setBindGroup(0, this.isoInit.group); pass.dispatchWorkgroups(Math.ceil(this.count / WG));
    if (this.isoInit.nodes) { pass.setPipeline(this.isoInit.nodes); pass.dispatchWorkgroups(Math.ceil(this.lodN / WG)); }   // LOD nodes
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // M39 direction-independent transfer on or off; the bounce history restarts so the probes settle on the new rule.
  setIso(on) {
    this.iso = on;
    this.firstGather = true;
    this.round = 0;
    this.forceNear = true;   // stage 2: iso_xfer refreshes every Gaussian's t'
  }

  // M39 stage 2: the matte bounce through SH2 probes (bounce_sh) instead of the 128-cell gather (bounce_dirs).
  setShProbes(on) {
    this.shProbes = on;
    this.forceNear = true;
  }

  // Hierarchical shadow walk (shadow_cull.wgsl, off by default): the scene tree's nodes grouped by depth (part roots
  // first, the scene root left out), and once per asset every node's rest-frame box and parent.
  initShadowCull(device, sources, asset) {
    const buf = (label, size, usage = B.STORAGE) => device.createBuffer({ label, size, usage: usage | B.COPY_DST | B.COPY_SRC });
    const N = this.nodeCount, nu = new Uint32Array(asset.raw.nodes);
    const byDepth = [];
    for (let i = 0; i < N; i++) {
      if (nu[i * 24 + 15] === 0xffffffff) continue;
      (byDepth[nu[i * 24 + 21]] ??= []).push(i);
    }
    const order = new Uint32Array(N);
    this.hsLevels = [];
    let off = 0;
    for (let d = 1; d < byDepth.length; d++) {
      const list = byDepth[d] ?? [];
      if (!list.length) continue;
      order.set(list, off);
      const level = buf(`shadow level ${d}`, 16, B.UNIFORM);
      device.queue.writeBuffer(level, 0, new Uint32Array([off, list.length, d, 0]));
      this.hsLevels.push({ count: list.length, level });
      off += list.length;
    }
    Object.assign(this.buffers, {
      node_box: buf('node rest boxes', N * 32),
      node_parent: buf('node parents', N * 4), level_nodes: buf('nodes by depth', N * 4),
      hs_params: buf('shadow cull params', 32, B.UNIFORM),
    });
    // every node's parent (part roots and the scene root: none), filled here so readers other than shadow_cull
    // (light_shadow.wgsl lsm_splat's motion-blur cut) need not wait for its first pass
    const parent = new Uint32Array(N).fill(0xffffffff), kidsU = new Uint32Array(asset.raw.kids);
    for (let i = this.count; i < N; i++) {
      if (nu[i * 24 + 15] === 0xffffffff) continue;
      for (let j = nu[i * 24 + 19]; j < nu[i * 24 + 20]; j++) parent[kidsU[j]] = i;
    }
    device.queue.writeBuffer(this.buffers.node_parent, 0, parent);
    device.queue.writeBuffer(this.buffers.level_nodes, 0, order);
    this.hierShadow = true;  // on by default (David, 2026-10-25)
    this.hierStride = 1;     // amortized walk_emit: still Gaussians walk once every hierStride frames (1 = every frame; opt-in)
    this.hsDirty = true;     // shadow_cull reruns when the light or a part moved (encode's `changed`)
    const res = {
      0: this.buffers.uniform, 1: this.buffers.gaussians, 2: this.buffers.nodes, 4: this.buffers.parts, 9: this.buffers.kids,
      33: this.buffers.node_status, 34: this.buffers.node_box, 35: this.buffers.node_parent, 36: this.buffers.level_nodes,
      37: this.buffers.hs_params,
    };
    const module = device.createShaderModule({ label: 'shadow_cull', code: sources.common + '\n' + sources.shadow_cull });
    this.hs = {};
    for (const [entry, binds] of Object.entries(SHADOW_CULL)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      const group = level => device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: binds.map(b => ({ binding: b, resource: { buffer: b === 38 ? level : res[b] } })),
      });
      this.hs[entry] = { pipe, groups: binds.includes(38) ? this.hsLevels.map(l => group(l.level)) : [group(null)] };
    }
    const u = new Uint32Array(16); u[1] = this.count;
    device.queue.writeBuffer(this.buffers.uniform, 0, u);
    const encoder = device.createCommandEncoder();
    const run = (entry, threads, k = 0) => {
      const pass = encoder.beginComputePass({ label: entry });
      pass.setPipeline(this.hs[entry].pipe); pass.setBindGroup(0, this.hs[entry].groups[k]);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(threads / WG))); pass.end();
    };
    run('hs_box_leaf', this.count);
    for (let k = this.hsLevels.length - 1; k >= 0; k--) run('hs_box_up', this.hsLevels[k].count, k);
    device.queue.submit([encoder.finish()]);
  }

  // The world box every emitter_sample point can land in (shadow_cull.wgsl HP): each emitter leaf's mean +- radius / 3
  // per axis (the jitter cube), or +- sqrt(3) radius with the emitter-shaped jitter (a clamped 3-sigma draw).
  shadowLightBox(nodes, skipPart) {
    const f = new Float32Array(nodes.buffer, nodes.byteOffset, nodes.length), u = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length);
    const s = this.emitSmooth & 2 ? Math.sqrt(3) : 1 / 3;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let o = 0; o < f.length; o += 16) {
      if (u[o + 11] !== 0xffffffff) continue;
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], f[o + k] - s * f[o + 3]);
        hi[k] = Math.max(hi[k], f[o + k] + s * f[o + 3]);
      }
    }
    const hp = new ArrayBuffer(32), hf = new Float32Array(hp), hu = new Uint32Array(hp);
    hf.set(lo, 0); hu[3] = skipPart; hf.set(hi, 4); hu[7] = HS_BUDGET;
    this.device.queue.writeBuffer(this.buffers.hs_params, 0, hp);
  }

  // Hierarchical shadow walk on or off, and the amortized walk's stride (the walk on by default, the stride opt-in; main.js syncs them).
  setHierShadow(on, stride = this.hierStride) {
    this.hierShadow = on;
    this.hierStride = Math.max(1, stride | 0);
    this.hsDirty = true;
    this.emitKey = null;   // setEmitters rewrites EmitParams.hier / stride
  }

  initEmissive(device, sources) {
    const buf = (label, size, usage = B.STORAGE) => device.createBuffer({ label, size, usage: usage | B.COPY_DST | B.COPY_SRC });
    this.buffers.emitters = buf('emitters', MAX_EMITTERS * 64);
    this.buffers.emit_nodes = buf('emitter tree', (2 * MAX_EMITTERS - 1) * 96);   // EmitNode + its ASG covariance (packEmitNodes)
    this.buffers.emit_params = buf('emit params', 64, B.UNIFORM);
    this.emitterTarget = EMITTERS_TARGET;
    this.emitTheta = EMIT_THETA;
    this.emitMode = 1;
    this.emitterCount = 0;
    this.emitEdgeSoft = 0.2;   // one-sided emitters fade over |cos| < this around their plane (0 = hard cut)
    this.emitSmooth = 0;    // N5 shadow-sampling opt-ins: 1 golden pick, 2 emitter-shaped jitter, 4 progressive 4 samples + ease
    this.asgSpec = 1;       // P3 ASG specular (EmitParams.n5 bits 3-4), on by default (David, 2026-10-25): 1 lights as shapes in shiny highlights, 3 with the spec cut
    this.historyMode = 0;   // 0 EMA with the 1/16 floor (M35), 1 effective count (N5)
    const res = {
      0: this.buffers.uniform, 1: this.buffers.gaussians, 2: this.buffers.nodes, 3: this.buffers.vgrid,
      4: this.buffers.parts, 6: this.buffers.lamp_state, 7: this.buffers.materials, 8: this.buffers.colour,
      9: this.buffers.kids, 13: this.buffers.lo_dir, 18: this.buffers.emit_nodes, 19: this.buffers.emit_params,
      20: this.buffers.samples, 33: this.buffers.node_status, 61: this.lodFlagBuf, 62: this.lodList,
    };
    const module = device.createShaderModule({ label: 'emissive', code: sources.common + '\n' + sources.material_ext + '\n' + sources.emissive });
    this.emitModule = module;
    this.emitSources = sources;
    this.emitRes = res;
    this.emit = {};
    const kinds = this.lodCut ? { ...EMISSIVE, shade_emit_list: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19, 62], 'shade'],
      shade_emit_nodes_list: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19, 61, 62], 'shade'],
      shade_emit_nodes: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19, 61], 'shade'] }   // lodCut = false at run time: stage 1
      : this.lodLight ? { ...EMISSIVE, shade_emit_nodes: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19, 61], 'shade'] } : EMISSIVE;
    for (const [entry, [binds, timer]] of Object.entries(kinds)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      this.emit[entry] = { pipe, timer, groups: [device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        // motion blur: the shadow walks traverse other parts' stretched trees (the receiver's own records stay)
        entries: binds.map(b => ({ binding: b, resource: { buffer: b === 2 && this.blur && entry.startsWith('walk_emit') ? this.buffers.nodes_blur : res[b] } })),
      })] };
    }
  }

  // Projection-field shadows (prototype, ?lights=unified&shadow=pf): the asset's pfield.bin (M38 fields per part,
  // experiments/M42_many_lights/pf_export.py) in one storage buffer; walk_emit_pf (emissive.wgsl) then replaces
  // walk_emit, looking up the parts that have a field and walking the rest.
  // Returns null once the field is bound, or a warning when there is no usable field (missing, unreadable, or baked
  // from another version of the asset: pfield.json's fingerprint vs asset.fingerprint); the caller then keeps the
  // sampled path (walk_emit), which needs nothing from here.
  async loadField(url, direct = false, asset = null, name = '') {
    const fix = `run experiments/M42_many_lights/pf_export.py ${name}; using sampled shadows`;
    let bytes, meta;
    try {
      const [rb, rj] = await Promise.all([fetch(url), fetch(url.replace(/\.bin$/, '.json'))]);
      if (!rb.ok) return `no shadow fields for ${name}; ${fix}`;
      bytes = await rb.arrayBuffer();
      meta = rj.ok ? await rj.json() : null;
    } catch (e) {
      return `no shadow fields for ${name} (${e.message}); ${fix}`;
    }
    if (bytes.byteLength < 32 || new Uint32Array(bytes, 0, 1)[0] !== 0x444C4650) return `${url} is not a pfield file; ${fix}`;
    if (asset && !sameFingerprint(meta?.asset, asset.fingerprint)) return `shadow fields are stale for ${name}; re-${fix}`;
    const fq = new URLSearchParams(globalThis.location?.search ?? '');
    if (fq.get('pftab') === '0') new Uint32Array(bytes)[5] = 0;   // DEBUG: scan every axis instead of the axis table
    const B = GPUBufferUsage, device = this.device;
    // two regions after the file (header words 6 and 7, pfd only): each part's world-space bound (centre, radius;
    // radius -1 = no field), rewritten each frame from the parts' transforms (writePfBounds), and the per-Gaussian T
    // cache shade_emit_pfd shares with direct_dirs_emit_pfd (4 words each)
    const u = new Uint32Array(bytes), f = new Float32Array(bytes);
    const K = u[2], NP = u[3], words = u.length;
    this.pfDir = [];
    for (let p = 0; p < NP; p++) {
      const d = 8 + 12 * K + 8 * p;
      this.pfDir.push(u[d + 4] ? [f[d], f[d + 1], f[d + 2], f[d + 3]] : null);
    }
    if (direct) { u[6] = words; u[7] = words + 4 * NP; }
    this.pfBoundsWord = u[6];
    const size = (words + (direct ? 4 * NP + 4 * this.count : 0)) * 4;
    this.buffers.pfield = device.createBuffer({ label: 'projection fields', size, usage: B.STORAGE | B.COPY_DST });
    device.queue.writeBuffer(this.buffers.pfield, 0, bytes);
    const res = { ...this.emitRes, 50: this.buffers.pfield };
    const pipe = device.createComputePipeline({ label: 'walk_emit_pf', layout: 'auto', compute: { module: this.emitModule, entryPoint: 'walk_emit_pf' } });
    this.emit.walk_emit_pf = { pipe, timer: 'walk_lamp', groups: [device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [0, 1, 2, 4, 9, 18, 19, 20, 33, 50].map(b => ({ binding: b, resource: { buffer: res[b] } })),
    })] };
    this.pfield = true;
    if (!direct) return null;
    // option B (?shadow=pfd): shade_emitters' per-node T from the fields (emissive.wgsl pfd_T), no walk or history.
    // The variant module replaces the two lamp_state readers in shade_emitters, so its entry points bind the field
    // (50) in place of lamp_state (6) and stay within 8 storage buffers; the default module is untouched.
    const code = pfdSource(this.emitSources, fq.get('pfsplit') !== '0');   // DEBUG ?pfsplit=0: no strongest-lights split
    const mod = device.createShaderModule({ label: 'emissive pfd', code });
    for (const [entry, [binds, timer]] of Object.entries({ shade_emit: EMISSIVE.shade_emit, direct_dirs_emit: EMISSIVE.direct_dirs_emit })) {
      const p = device.createComputePipeline({ label: entry + '_pfd', layout: 'auto', compute: { module: mod, entryPoint: entry } });
      this.emit[entry + '_pfd'] = { pipe: p, timer, groups: [device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: binds.map(b => (b === 6 ? 50 : b)).map(b => ({ binding: b, resource: { buffer: res[b] } })),
      })] };
    }
    this.pfd = true;
    return null;
  }

  // ?shadow=pfd: each field part's bound in world space for pfd_T's cull (the field's rest-frame centre through the part's
  // current transform, partXf rows [R | t]; rigid, so the radius carries over)
  writePfBounds() {
    if (!this.pfBoundsWord || !this.partXf) return;
    const out = new Float32Array(this.pfDir.length * 4);
    this.pfDir.forEach((b, p) => {
      if (!b) { out[p * 4 + 3] = -1; return; }
      const m = this.partXf.subarray(p * 32, p * 32 + 12);
      for (let r = 0; r < 3; r++) out[p * 4 + r] = m[r * 4] * b[0] + m[r * 4 + 1] * b[1] + m[r * 4 + 2] * b[2] + m[r * 4 + 3];
      out[p * 4 + 3] = b[3];
    });
    this.device.queue.writeBuffer(this.buffers.pfield, this.pfBoundsWord * 4, out);
  }

  // Occluder-proxy shadows (prototype, ?lights=unified&shadow=rcd or pxd; experiments/M42_many_lights/proxy_stage1.py):
  // the pfd path (selection, T cache, part cull) with the field lookup swapped for emissive.wgsl rcd_tau or pxd_tau.
  // Both bind a host-built buffer at 50 in place of the field: header (word 1 data offset, 3 part count, 4 rmin, 6
  // world bounds, 7 T cache), then for rcd a copy of kids.bin (the entry points already hold 8 storage buffers, so
  // the walk can't bind kids itself), for pxd the asset's proxy.bin (proxy_export.py; fingerprint-checked as pfield).
  // Occluders as pfd's fields: parts with non-emissive Gaussians except the room shell (?rcdshell=1 walks it too);
  // for pxd, the parts proxy.bin lists. Bounds are parts.bin's rest-frame bound. Returns a warning or null.
  // rcc / pxc: the same per receiver cluster (emissive.wgsl cluster_T): each part's Gaussians on a ?rcl= grid (default
  // 0.03 m) in its rest frame, the cluster block appended to the buffer (word 5).
  async loadProxyShadow(mode, asset, name = '') {
    const q = new URLSearchParams(globalThis.location?.search ?? '');
    const NP = asset.partCount, B = GPUBufferUsage, device = this.device;
    let head;
    const tau = { rcc: 'rcd', pxc: 'pxd' }[mode] ?? mode, clustered = tau !== mode;
    if (tau === 'pxd') {
      const fix = `run experiments/M42_many_lights/proxy_export.py ${name}; using sampled shadows`;
      let bytes, meta;
      try {
        const base = `runtime/assets/${name}/proxy`;
        const [rb, rj] = await Promise.all([fetch(base + '.bin'), fetch(base + '.json')]);
        if (!rb.ok) return `no shadow proxies for ${name}; ${fix}`;
        bytes = await rb.arrayBuffer();
        meta = rj.ok ? await rj.json() : null;
      } catch (e) {
        return `no shadow proxies for ${name} (${e.message}); ${fix}`;
      }
      if (bytes.byteLength < 32 || new Uint32Array(bytes, 0, 1)[0] !== 0x59585250) return `proxy.bin is not a proxy file; ${fix}`;
      if (!sameFingerprint(meta?.asset, asset.fingerprint)) return `shadow proxies are stale for ${name}; re-${fix}`;
      head = new Uint32Array(bytes);
    } else {
      const kids = new Uint32Array(asset.raw.kids);
      head = new Uint32Array(8 + kids.length);
      head[0] = 0x44435252; head[1] = 8; head.set(kids, 8);
      new Float32Array(head.buffer)[4] = Number(q.get('rmin') ?? 0.04);
    }
    head[3] = NP;
    const pv = new DataView(asset.parts), gu = new Uint32Array(asset.gaussians);
    const shell = q.get('rcdshell') !== '1' ? /^(floor|ceiling|back|left_red|right_green)$/ : /^$/;
    this.pfDir = asset.header.parts.map((pt, p) => {
      let occ;
      if (tau === 'pxd') occ = head[9 + 2 * p] > 0;
      else {
        const [g0, g1] = pt.gaussians;
        occ = !shell.test(pt.name);
        if (occ) { occ = false; for (let g = g0; g < g1 && !occ; g++) occ = !(gu[g * 16 + 15] & 0x10000); }
      }
      return occ ? [0, 1, 2, 3].map(k => pv.getFloat32(p * 128 + 96 + 4 * k, true)) : null;
    });
    const words = head.length;
    let clu = null;
    if (clustered) {
      // clusters: per part, its Gaussians binned on a rcl grid in the rest frame; centre = the members' mean
      const gf = new Float32Array(asset.gaussians), rcl = Number(q.get('rcl') ?? 0.03);
      const gcl = new Uint32Array(this.count), list = [], cells = new Map();
      asset.header.parts.forEach((pt, p) => {
        const [g0, g1] = pt.gaussians;
        for (let g = g0; g < g1; g++) {
          const key = `${p},${Math.floor(gf[g * 16] / rcl)},${Math.floor(gf[g * 16 + 1] / rcl)},${Math.floor(gf[g * 16 + 2] / rcl)}`;
          let c = cells.get(key);
          if (c === undefined) { c = list.length; cells.set(key, c); list.push([0, 0, 0, 0, p]); }
          const e = list[c];
          e[0] += gf[g * 16]; e[1] += gf[g * 16 + 1]; e[2] += gf[g * 16 + 2]; e[3]++;
          gcl[g] = c;
        }
      });
      const lw = new Uint32Array(4 * list.length), lf = new Float32Array(lw.buffer);
      list.forEach((e, c) => { lf[4 * c] = e[0] / e[3]; lf[4 * c + 1] = e[1] / e[3]; lf[4 * c + 2] = e[2] / e[3]; lw[4 * c + 3] = e[4]; });
      clu = { n: list.length, gcl, list: lw };
      this.clusterCount = clu.n;
    }
    const cw = clu ? 4 + this.count + 4 * clu.n + 24 * clu.n : 0;
    const u = new Uint32Array(words + 4 * NP + 4 * this.count + cw);
    u.set(head);
    u[6] = words; u[7] = words + 4 * NP;
    if (clu) {
      const C = words + 4 * NP + 4 * this.count;
      u[5] = C; u[C] = clu.n; u[C + 1] = C + 4; u[C + 2] = C + 4 + this.count; u[C + 3] = C + 4 + this.count + 4 * clu.n;
      u.set(clu.gcl, C + 4); u.set(clu.list, C + 4 + this.count);
    }
    this.pfBoundsWord = u[6];
    this.buffers.pfield = device.createBuffer({ label: `shadow ${mode}`, size: u.byteLength, usage: B.STORAGE | B.COPY_DST });
    device.queue.writeBuffer(this.buffers.pfield, 0, u);
    const res = { ...this.emitRes, 50: this.buffers.pfield };
    const mod = device.createShaderModule({ label: `emissive ${mode}`, code: pfdSource(this.emitSources, q.get('pfsplit') !== '0', mode) });
    for (const [entry, [binds, timer]] of Object.entries({ shade_emit: EMISSIVE.shade_emit, direct_dirs_emit: EMISSIVE.direct_dirs_emit })) {
      const p = device.createComputePipeline({ label: `${entry}_${mode}`, layout: 'auto', compute: { module: mod, entryPoint: entry } });
      this.emit[entry + '_pfd'] = { pipe: p, timer, groups: [device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: binds.map(b => (b === 6 ? 50 : b)).map(b => ({ binding: b, resource: { buffer: res[b] } })),
      })] };
    }
    if (clustered) {
      const p = device.createComputePipeline({ label: `cluster_T_${mode}`, layout: 'auto', compute: { module: mod, entryPoint: 'cluster_T' } });
      this.emit.cluster_T = { pipe: p, timer: 'walk_lamp', groups: [device.createBindGroup({   // reports in walk_lamp's slot
        layout: p.getBindGroupLayout(0),
        entries: [0, 1, 2, 4, 18, 19, 50].map(b => ({ binding: b, resource: { buffer: res[b] } })),
      })] };
      this.cluster = true;
    }
    this.pfd = true;
    return null;
  }

  // The lamp as a panel of emissive Gaussians: the asset lamp's rectangle (scaled by `size`, centred on `center`) tiled
  // with about `emitterTarget` flat Gaussians (short axis along the lamp normal, optical depth 3 through the panel),
  // each showing the lamp's radiance, plus the emitter tree the light cut walks. Returns the count.
  setEmitters(center, size) {
    const key = [...center, size, this.emitterTarget, this.emitTheta, this.emitMode, this.emitOneSided, this.emitEdgeSoft, this.emitSmooth, this.asgSpec].join();
    if (this.unified) return this.setUnifiedEmitters();
    if (key === this.emitKey) return this.emitterDrawn;
    this.emitKey = key;
    if (this.assetEmitters) return this.setAssetEmitters(center);
    const lf = new Float32Array(this.baseLamps);
    const e1 = [...lf.subarray(4, 7)].map(v => v * size), e2 = [...lf.subarray(8, 11)].map(v => v * size);
    const nrm = [...lf.subarray(12, 15)], Le = [...lf.subarray(16, 19)];
    const a = 2 * Math.hypot(...e1), b = 2 * Math.hypot(...e2);
    // a disabled or degenerate lamp (hill's placeholder sphere: zero half-axes, so the panel had NaN emitters and the
    // tree build recursed until the stack overflowed): one dark emitter at the lamp, nothing drawn
    if (this.lampOff || !(a * b > 0)) return this.setDarkEmitter(center);
    const sp = Math.sqrt(a * b / this.emitterTarget);
    const nu = Math.max(1, Math.round(a / sp)), nv = Math.max(1, Math.round(b / sp));
    const n = Math.min(nu * nv, MAX_EMITTERS);
    const u1 = e1.map(v => v / Math.hypot(...e1)), u2 = e2.map(v => v / Math.hypot(...e2));
    const q = quatFromColumns(nrm, u1, u2);
    const sx = 0.2 * sp, sy = 0.6 * a / nu, sz = 0.6 * b / nv;
    const out = new Float32Array(n * 16);
    for (let k = 0; k < n; k++) {
      const i = k % nu, j = Math.floor(k / nu);
      const fa = (i + 0.5) / nu * 2 - 1, fb = (j + 0.5) / nv * 2 - 1;
      const o = k * 16;
      for (let c = 0; c < 3; c++) out[o + c] = center[c] + fa * e1[c] + fb * e2[c];
      out[o + 3] = 3 / (Math.sqrt(2 * Math.PI) * sx);
      out.set(q, o + 4);
      out.set([sx, sy, sz], o + 8);
      out.set(Le, o + 12);
    }
    this.device.queue.writeBuffer(this.buffers.emitters, 0, out);
    const { nodes, root, cov } = buildEmitterTree(out, n);
    this.emitTree = { nodes, root, cov };   // light_shadow.js reads the root as its light
    this.device.queue.writeBuffer(this.buffers.emit_nodes, 0, packEmitNodes(nodes, cov));
    this.shadowLightBox(nodes, 0xffffffff);
    // Overlap correction: panel area / sum of the emitters' footprints along the normal (2 pi sy sz Ein(3)).
    const scale = (a * b) / (n * 2 * Math.PI * sy * sz * EIN3);
    this.writeEmitParams(n, root, scale, nodes.length / 16, 0xffffffff, this.emitOneSided ? 1 : 0, nrm);
    this.emitterDrawn = n;
    return n;
  }

  // Unified lights: the whole scene's light tree (lights.js) from the parts' current transforms (main.js sets partXf)
  // and the live material flags; rewritten only when a light moved, switched or changed gain. No light is drawn here:
  // emitters are scene Gaussians.
  setUnifiedEmitters() {
    if (!this.partXf) return 0;
    const tree = this.lightSet.update(this.partXf, new Uint32Array(this.materialBytes, 0, this.materialCount * 8));
    const params = [this.emitTheta, this.emitMode, this.emitEdgeSoft, this.emitSmooth, this.asgSpec, this.hierShadow, this.hierStride].join();
    if (!tree && this.lightSet.key === this.unifiedKey) {
      // lights unchanged; EmitParams still follow the panel (setHierShadow clears emitKey)
      if (params !== this.unifiedParams || this.emitKey === null) {
        this.unifiedParams = params;
        this.writeUnifiedParams();
      }
      return 0;
    }
    this.unifiedKey = this.lightSet.key;
    this.unifiedParams = params;
    let packed;
    if (tree) packed = Lights.pack(tree);
    else {   // no light on: one dark leaf so the cut has a root
      const f = new Float32Array(24), u = new Uint32Array(f.buffer);
      u[11] = 0xffffffff; u[7] = 0xffff << 16; u[23] = 0xffffffff;
      packed = { nodes: f, count: 1, root: 0, leaves: [] };
    }
    this.device.queue.writeBuffer(this.buffers.emit_nodes, 0, packed.nodes);
    // the hierarchical shadow walk's light box: every leaf's jitter box, as shadowLightBox
    const s = this.emitSmooth & 2 ? Math.sqrt(3) : 1 / 3;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const nd of packed.leaves) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], nd.mu[k] - s * nd.radius); hi[k] = Math.max(hi[k], nd.mu[k] + s * nd.radius); }
    if (!packed.leaves.length) { lo.fill(0); hi.fill(0); }
    const hp = new ArrayBuffer(32), hf = new Float32Array(hp), hu = new Uint32Array(hp);
    hf.set(lo, 0); hu[3] = 0xffffffff; hf.set(hi, 4); hu[7] = HS_BUDGET;
    this.device.queue.writeBuffer(this.buffers.hs_params, 0, hp);
    this.unifiedPacked = { root: packed.root, count: packed.count, leaves: packed.leaves.length };
    this.writeUnifiedParams();
    return 0;
  }

  writeUnifiedParams() {
    const p = this.unifiedPacked;
    this.emitKey = `unified ${this.frame}`;   // main.js reads a changed key as "the light changed"
    this.writeEmitParams(p.leaves, p.root, 1, p.count, 0xffffffff, 0, [0, 0, 1]);
  }

  writeEmitParams(n, root, scale, nodeCount, skipPart, oneSided, normal) {
    const ep = new ArrayBuffer(64);
    const epu = new Uint32Array(ep), epf = new Float32Array(ep);
    epu[0] = n; epu[1] = root; epf[2] = scale; epf[3] = this.emitTheta; epu[4] = this.emitMode; epu[5] = nodeCount;
    epu[6] = skipPart; epu[7] = oneSided;
    epf.set(normal, 8); epu[11] = this.emitSmooth | (this.asgSpec << 3);   // N5 opt-ins + ASG bits (emissive.wgsl EmitParams.n5)
    epf[12] = this.emitEdgeSoft;
    epu[13] = (this.hierShadow ? 1 : 0) | (this.hierStride > 1 ? 2 : 0); epu[14] = this.hierStride;   // EmitParams.hier, stride
    // EmitParams.lights; bits 1-2 are a DEBUG switch, ?lightdbg=<bits> (emissive.wgsl shade_emitters), bit 3 self-shadow
    // (?selfshadow=1: a light's own part occludes it, emissive.wgsl walk_tau_light), unified only
    const eq = new URLSearchParams(globalThis.location?.search ?? '');
    epu[15] = this.unified ? 1 | (Number(eq.get('lightdbg') ?? 0) & 6) | (eq.get('selfshadow') === '1' ? 8 : 0) : 0;
    // ?shadow=pfd strongest-lights split (emissive.wgsl pfd_select): bits 8-11 K (?pfk=, 1-8), bits 16-23 share (?pfshare=)
    // (default off: no ?pfshare=, share bits 0 = every node gets a lookup)
    if (this.pfd) epu[15] |= (Math.min(Math.max(Number(eq.get('pfk') ?? 8), 1), 8) << 8) | (eq.has('pfshare') ? Math.round(Math.min(Math.max(Number(eq.get('pfshare')), 0.01), 1) * 255) << 16 : 0);
    this.hsDirty = true;   // the light moved or changed
    this.device.queue.writeBuffer(this.buffers.emit_params, 0, ep);
    this.emitterScale = scale;
    this.emitterCount = n;
    this.emitLight = { skipPart, oneSided, normal };
    this.lightShadow?.setLight();
  }

  setDarkEmitter(center) {
    const out = new Float32Array(16);
    out.set(center, 0); out[3] = 1; out.set([0, 0, 0, 1], 4); out.set([0.01, 0.01, 0.01], 8);   // Le (12-14) = 0
    this.device.queue.writeBuffer(this.buffers.emitters, 0, out);
    const { nodes, root, cov } = buildEmitterTree(out, 1);
    this.emitTree = { nodes, root, cov };
    this.device.queue.writeBuffer(this.buffers.emit_nodes, 0, packEmitNodes(nodes, cov));
    this.shadowLightBox(nodes, 0xffffffff);
    this.writeEmitParams(1, root, 0, nodes.length / 16, 0xffffffff, 0, [0, 0, 1]);
    this.emitterDrawn = 0;
    return 0;
  }

  // The asset's emissive part (SPEC 2.6, asset.json `emitters`): its Gaussians, moved so their rest centroid sits at
  // `center` (the renderer moves the part by the same offset), are the light set. They are drawn with the scene.
  setAssetEmitters(center) {
    const { part, emission } = this.assetEmitters;
    const [g0, g1] = this.partRanges[part];
    const n = g1 - g0;
    const src = new Float32Array(this.gaussianBytes, g0 * 64, n * 16);
    const rec = new Float32Array(src);
    const c0 = [0, 1, 2].map(k => { let a = 0; for (let i = 0; i < n; i++) a += src[i * 16 + k]; return a / n; });
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) rec[i * 16 + k] += center[k] - c0[k];
      rec.set(emission, i * 16 + 12);
    }
    const { nodes, root, cov } = buildEmitterTree(rec, n);
    this.emitTree = { nodes, root, cov };
    this.device.queue.writeBuffer(this.buffers.emit_nodes, 0, packEmitNodes(nodes, cov));
    this.shadowLightBox(nodes, part);
    // emitting normal: the first emitter's short axis R x (SPEC 2.6 one-sided convention)
    const [x, y, z, w] = rec.subarray(4, 8);
    const nrm = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)];
    // overlap correction as for the host panel: the panel's area over the emitters' summed footprint along the normal,
    // the area taken from the grid of emitter centres (distinct rows x distinct columns x spacing)
    const ax = [[2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]];
    const extent = ax.map(a => {
      const vals = Array.from({ length: n }, (_, i) => a[0] * rec[i * 16] + a[1] * rec[i * 16 + 1] + a[2] * rec[i * 16 + 2]);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const cols = new Set(vals.map(v => Math.round((v - lo) / (hi - lo + 1e-9) * 1000))).size;
      return (hi - lo) * cols / Math.max(cols - 1, 1);
    });
    let foot = 0;
    for (let i = 0; i < n; i++) foot += 2 * Math.PI * rec[i * 16 + 9] * rec[i * 16 + 10] * EIN3;
    this.writeEmitParams(n, root, extent[0] * extent[1] / foot, nodes.length / 16, part, this.emitOneSided ? 1 : 0, nrm);
    this.emitterDrawn = 0;
    return 0;
  }

  initBounce(device, sources, asset) {
    const h = asset.header, G = this.count, N = h.buffers.nodes.count;
    // the lattice (room or terrain layout, probeLayout), then 8 probes per part that move with it (bounce.wgsl
    // part_probe_frame; static parts' stay idle)
    const lay = probeLayout(asset);
    this.probeLayout = lay;
    const volProbes = (lay.vols ?? []).reduce((a, v) => a + v.dims[0] * v.dims[1] * v.dims[2], 0);
    const NP = lay.dims[0] * lay.dims[1] * lay.dims[2] + PPK * this.partCount + volProbes;
    const buf = (label, size, usage = B.STORAGE) => device.createBuffer({ label, size, usage: usage | B.COPY_DST | B.COPY_SRC });
    Object.assign(this.buffers, {
      probe_params: buf('probe params', PROBE_PARAMS_BYTES, B.UNIFORM),
      probe_data: buf('probe data', NP * NCELL * 96),
      nearvis: buf('near-field visibility', G * 128),
      lo_dir: buf('direct toward D14', G * 14 * 16),
      lo_b: buf('bounce toward D14 + eye', G * 15 * 16),
      node_acc: buf('node emitter sums', N * 14 * 16),
      node_sh: buf('node emitter SH2', N * 9 * 16),
      probe_sh: buf('probe SH2 (M39 stage 2)', NP * 9 * 16),
      xfer: buf('Gaussian transfer SH2 (M39 stage 2)', G * 3 * 16),
      blk: buf('near-field blockers', G * 128),
      lo_b_prev: buf('last frame bounce (near-field reflection)', G * 15 * 16),
    });
    // near-field reflection blockers, 16 (node, weight) per Gaussian: 0-7 own part from the asset's blockers.bin
    // (runtime/bake_blockers.py; none without it), 8-15 other parts, written by near_vis while reflection is on
    this.blkBytes = new Uint32Array(G * 32).fill(0xffffffff);
    for (let i = 1; i < G * 32; i += 2) this.blkBytes[i] = 0;
    if (asset.raw.blockers) {
      const own = new Uint32Array(asset.raw.blockers);
      for (let g = 0; g < G; g++) this.blkBytes.set(own.subarray(g * 16, g * 16 + 16), g * 32);
    }
    device.queue.writeBuffer(this.buffers.blk, 0, this.blkBytes);
    this.nearRefl = false;
    this.emitterBounceFix = false;
    this.nearCones = 8;   // near-field cone count (6 or 8) instead of the near_vis walk; 0 = the walk (main.js default: 8)
    this.nearNext = G;   // next Gaussian of a chunked forced near_vis refresh (G = none pending)
    this.nodeCount = N;
    this.probeCount = NP;
    this.maxDepth = h.walk?.max_depth ?? 6;
    // node_up per depth over that depth's internal nodes only (up_nodes[offset, offset + count)), not every node: a
    // binary tree has ~3x the levels (2026-10-29)
    const nuB = new Uint32Array(asset.raw.nodes), upBy = Array.from({ length: this.maxDepth + 1 }, () => []);
    for (let i = G; i < N; i++) {
      const d = nuB[i * 24 + 21];
      if (nuB[i * 24 + 15] !== 0xffffffff && d <= this.maxDepth) upBy[d].push(i);
    }
    let upOff = 0;
    this.levelCounts = upBy.map(l => l.length);
    this.levels = upBy.map((list, d) => {
      const b = buf(`level ${d}`, 16, B.UNIFORM);
      device.queue.writeBuffer(b, 0, new Uint32Array([d, upOff, list.length, 0]));
      upOff += list.length;
      return b;
    });
    const upNodes = new Uint32Array(Math.max(upOff, 1));
    upBy.reduce((o, list) => { upNodes.set(list, o); return o + list.length; }, 0);
    this.buffers.up_nodes = buf('node_up lists', upNodes.byteLength);
    device.queue.writeBuffer(this.buffers.up_nodes, 0, upNodes);

    this.params = new ArrayBuffer(PROBE_PARAMS_BYTES);   // ProbeParams: 80 B header, D14, pinv, refl, layout
    const f = new Float32Array(this.params), u = new Uint32Array(this.params);
    f.set(lay.lo, 0); u[3] = lay.dims[0];
    f.set(lay.step, 4); u[7] = N;
    f.set([0.5 * Math.hypot(...lay.step), PROBE_THETA, lay.tmax, NEAR_R], 8);
    u.set([...lay.dims, lay.terrain ? 1 : 0], 208);   // dims (bounce.wgsl ProbeParams.dims)
    f.set(lay.levels, 212);                            // levels[16]
    f.set(lay.ground, 228);                            // ground[1024]
    let off = PPK * this.partCount;                    // static volumes after the movable parts' PPK each
    for (const v of lay.vols ?? []) { u.set([off, ...v.dims], 1252 + 4 * v.part); off += v.dims[0] * v.dims[1] * v.dims[2]; }
    u[1380] = PPK * this.partCount;                    // vol_info.x
    if (NODE_PROBES) { this.nodeProbePos = nodeProbes(asset); f.set(this.nodeProbePos, 1384); }   // nodep
    f.set([1, 0, 0.02, 0.95], 12);              // alpha, force_near, vfloor, enclosed fraction
    // ?partprobes=0 (diagnosis, room layouts only): no probes riding with the parts, every Gaussian on the lattice
    const noPartProbes = !lay.terrain && new URLSearchParams(globalThis.location?.search ?? '').get('partprobes') === '0';
    u[16] = SUBSET; u[18] = (noPartProbes ? 0 : PPK * this.partCount) + volProbes;   // part_probes: movable parts' 8 each, then the volumes
    u[19] = 1;   // sh_full: bounce_sh every Gaussian every frame (< 0.5 ms); a quarter at a time sparkles while settling
    D14.forEach((d, k) => f.set([...d, 0], 20 + 4 * k));
    pinvD14().flat().forEach((v, i) => { f[76 + i] = v; });
    this.round = 0;
    this.forceNear = true;
    this.nearField = true;   // main.js turns it off on the first lit frame (own-part visibility is the base)
    this.firstGather = true;

    const module = device.createShaderModule({ label: 'bounce', code: sources.common + '\n' + sources.material_ext + '\n' + sources.bounce });
    const res = {
      0: this.buffers.uniform, 1: this.buffers.gaussians, 2: this.buffers.nodes, 3: this.buffers.vgrid,
      4: this.buffers.parts, 5: this.buffers.lamps, 6: this.buffers.lamp_state, 7: this.buffers.materials,
      9: this.buffers.kids, 10: this.buffers.probe_params, 11: this.buffers.probe_data, 12: this.buffers.nearvis,
      13: this.buffers.lo_dir, 14: this.buffers.lo_b, 15: this.buffers.node_acc, 16: this.buffers.node_sh,
      25: this.buffers.probe_sh, 26: this.buffers.xfer,
      8: this.buffers.colour, 27: this.buffers.blk, 28: this.buffers.lo_b_prev, 40: this.buffers.env,
      29: this.buffers.up_nodes,
    };
    this.bounce = {};
    const bounceQ = new URLSearchParams(globalThis.location?.search ?? '');
    for (const [entry, binds] of Object.entries(BOUNCE)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      const group = level => device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: binds.map(b => ({ binding: b, resource: { buffer: b === 17 ? this.levels[level]
          : b === 2 && entry === 'probe_gather' && this.blur && bounceQ.get('blurprobes') === '1' ? this.buffers.nodes_blur : res[b] } })),   // motion blur: the probes see moving parts unstretched (David, 2026-10-29: stretched-with-mass-kept senders gave dark splotches); ?blurprobes=1 the old stretch
      });
      this.bounce[entry] = { pipe, groups: entry === 'node_up' ? this.levels.map((_, d) => group(d)) : [group(0)] };
    }
    this.bounce.near_cones.timer = 'near_vis';
  }

  // Near-field visibility (bounce.wgsl near_vis) on or off. Off = other parts never block the bounce within near_r:
  // only each Gaussian's own-part V grid does (David's working assumption, 2026-10-23).
  setNearField(on) {
    this.nearField = on;
    if (on) this.forceNear = true;
    else {
      this.device.queue.writeBuffer(this.buffers.nearvis, 0, new Uint8Array(this.count * 128).fill(255));
      this.device.queue.writeBuffer(this.buffers.blk, 0, this.blkBytes);   // drop the other-part blockers
    }
  }

  // Emitters out of the bounce probes (bounce.wgsl node_leaf, PP.refl.w; off by default, M47 flush lamp): an emitter
  // Gaussian's reflected light is mostly its neighbours' emission caught at point-blank range, which the probes would
  // carry into the room as extra bounce.
  setEmitterBounceFix(on) {
    this.emitterBounceFix = on;
    this.firstGather = true;
    this.round = 0;
  }

  // Near-field reflection on or off (bounce.wgsl near_refl): the light visibility blocks comes back as the blockers'
  // colour, own-part blockers always, other parts' only with near-field visibility on. Off = the shipped bounce.
  setNearReflection(on) {
    this.nearRefl = on;
    this.forceNear = true;
    this.device.queue.writeBuffer(this.buffers.blk, 0, this.blkBytes);
  }

  // Near-field cones (bounce.wgsl near_cones; 8 is the default near-field method): with near-field visibility on, k wide cones (6 or 8) per
  // Gaussian instead of the 128-cell walk, for the transmittance and (with reflection) the other-part blockers. 0 = the walk.
  setNearCones(k) {
    this.nearCones = k === 6 || k === 8 ? k : 0;
    this.forceNear = true;
  }

  // Probe loop, one gather round per frame (M36 §3 with temporal blending).
  // Display fill (runtime/wgsl/fill.wgsl): the asset's baked table (onepath.bin `fill`, bake_onepath.py) for true-density
  // assets, whose hidden layers would otherwise show through as a felt texture; ?fill=0 off. Assets without a table skip it.
  async loadFill(assetName, sources) {
    if (new URLSearchParams(globalThis.location?.search ?? '').get('fill') === '0') return;
    const r = await fetch(`runtime/assets/${assetName}/onepath.json`);
    if (!r.ok) return;
    const meta = await r.json();
    if (!meta.fill || meta.G !== this.count) return;
    const bin = await fetch(`runtime/assets/${assetName}/onepath.bin`).then(x => x.arrayBuffer());
    const d = this.device;
    const table = d.createBuffer({ label: 'display fill', size: meta.fill.bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(table, 0, new Uint8Array(bin, meta.fill.offset, meta.fill.bytes));
    const u = d.createBuffer({ label: 'display fill params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(u, 0, new Uint32Array([this.count, 0, 0, 0]));
    const module = d.createShaderModule({ label: 'fill', code: sources.common + '\n' + sources.fill });
    const pipe = d.createComputePipeline({ label: 'display_fill', layout: 'auto', compute: { module, entryPoint: 'display_fill' } });
    this.fill = { pipe, group: d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [u, table, this.buffers.colour, this.buffers.lo_b]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) }) };
    if (this.lodCut) {   // ?lodlight=2: drawn leaves' fill sources get shaded too (fill.wgsl fill_need)
      const np = d.createComputePipeline({ label: 'fill_need', layout: 'auto', compute: { module, entryPoint: 'fill_need' } });
      this.fill.need = { pipe: np, group: d.createBindGroup({ layout: np.getBindGroupLayout(0), entries: [[0, u], [1, table],
        [4, this.lodFlagBuf], [5, this.lodNeed]].map(([binding, buffer]) => ({ binding, resource: { buffer } })) }) };
    }
  }

  // ?lodlight=2, after lod_select and before lod_list (lod.js encodeSelect)
  encodeFillNeed(encoder) {
    if (!this.fill?.need) return;
    const pass = encoder.beginComputePass({ label: 'fill_need' });
    pass.setPipeline(this.fill.need.pipe); pass.setBindGroup(0, this.fill.need.group);
    pass.dispatchWorkgroups(Math.ceil(this.count / 64)); pass.end();
  }

  encodeFill(encoder) {
    if (!this.fill) return;
    const pass = encoder.beginComputePass({ label: 'display_fill' });
    pass.setPipeline(this.fill.pipe); pass.setBindGroup(0, this.fill.group);
    pass.dispatchWorkgroups(Math.ceil(this.count / 64)); pass.end();
  }

  encodeBounce(encoder, timers, emissive = false) {
    const f = new Float32Array(this.params), u = new Uint32Array(this.params);
    f[12] = probeAlpha(this.firstGather);
    // with reflection the near_vis walk is ~2x dearer, so a forced refresh is spread over frames in NEAR_CHUNK pieces
    // (one submit for every Gaussian risks the GPU watchdog); force_near stays set until the last piece
    // (the cones are cheap enough to refresh every Gaussian in one submit)
    if (this.nearRefl && this.nearField && !this.nearCones && this.forceNear) this.nearNext = 0;
    const chunked = this.nearRefl && this.nearField && !this.nearCones && this.nearNext < this.count;
    u[13] = this.forceNear || chunked ? 1 : 0;
    u[205] = chunked ? this.nearNext : 0;   // PP.refl.y
    // For the first FULL_ROUNDS rounds refresh every ray and Gaussian each frame, so the probes settle quickly.
    const sub = this.round < FULL_ROUNDS ? 1 : SUBSET;
    u[16] = sub; u[17] = this.round % sub;
    u[204] = this.nearRefl ? 1 : 0;   // PP.refl.x (byte 816)
    u[206] = this.nearCones;          // PP.refl.z
    // PP.refl.w: the emitter part + 1, whose Gaussians send no reflected light into the probes (M47); 0 = off
    u[207] = this.emitterBounceFix && this.assetEmitters ? this.assetEmitters.part + 1 : 0;
    this.round++;
    this.device.queue.writeBuffer(this.buffers.probe_params, 0, this.params);
    this.forceNear = false;
    this.firstGather = false;
    const run = (entry, threads, group = 0) => {
      const { pipe, groups, timer = entry } = this.bounce[entry] ?? this.emit[entry];
      const pass = encoder.beginComputePass({ label: entry, timestampWrites: BOUNCE_TIMERS.includes(timer) ? timers.writes(timer) : undefined });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, groups[group]);
      pass.dispatchWorkgroups(Math.ceil(threads / 64));
      pass.end();
    };
    const sh = this.iso && this.shProbes;
    if (this.nearRefl) encoder.copyBufferToBuffer(this.buffers.lo_b, 0, this.buffers.lo_b_prev, 0, this.count * 15 * 16);
    if (this.nearField && this.nearCones) run('near_cones', this.count);
    else if (this.nearField) run('near_vis', chunked ? Math.min(NEAR_CHUNK, this.count - this.nearNext) : this.count);
    if (chunked) this.nearNext += NEAR_CHUNK;
    if (sh) run('iso_xfer', this.count);
    run(emissive ? (this.pfd && this.unified ? 'direct_dirs_emit_pfd' : 'direct_dirs_emit') : 'direct_dirs', sh ? Math.ceil(this.count / 4) : this.count);   // stage 2: 1/4 a frame
    this.env?.encodeDirs(encoder);   // distant lights toward the probes, for the same Gaussians (environment.js)
    // senders (this frame's direct light + last frame's bounce) -> the probes gather -> each Gaussian reads its
    // bounce from this frame's probes (David, 2026-10-29: the first bounce settles in the frame things move; it used
    // to read last frame's probes, a one-frame dark trail behind a fast mover). Later bounces still lag a frame each.
    const readBounce = () => {
      run('bounce_dirs', Math.ceil(this.count / u[16]));
      if (sh) run('bounce_sh', u[19] ? this.count : Math.ceil(this.count / u[16]));
      if (this.nearRefl) run('near_refl', this.count);
    };
    const oldOrder = new URLSearchParams(globalThis.location?.search ?? '').get('bounceorder') === 'old';   // debug: the pre-2026-10-29 order
    if (oldOrder) readBounce();
    run('node_leaf', this.count);
    for (let d = this.maxDepth - 1; d >= 1; d--) run('node_up', this.levelCounts[d], d);
    run('node_sh2', this.nodeCount);
    run('probe_gather', Math.ceil(this.probeCount * NCELL / u[16]));
    run('probe_valid', this.probeCount);
    if (sh) run('probe_sh2', this.probeCount);
    if (!oldOrder) readBounce();
  }

  // Shadow pooling (ema.wgsl ema_light): for each Gaussian, up to 3 leaf siblings under its parent node in the same
  // part (leaf node index == Gaussian index, SPEC 2.2), the 3 nearest by centre when the parent has more than 4 leaves.
  // u32 x 4 per Gaussian, 0xFFFFFFFF pads; all pads (no pooling) when off.
  poolNeighbours(asset, on) {
    const G = this.count, out = new Uint32Array(G * 4).fill(0xffffffff);
    if (!on) return out;
    const nu = new Uint32Array(asset.raw.nodes), nf = new Float32Array(asset.raw.nodes), kids = new Uint32Array(asset.raw.kids);
    const N = asset.header.buffers.nodes.count;
    const d2 = (a, b) => (nf[a * 24] - nf[b * 24]) ** 2 + (nf[a * 24 + 1] - nf[b * 24 + 1]) ** 2 + (nf[a * 24 + 2] - nf[b * 24 + 2]) ** 2;
    for (let i = G; i < N; i++) {
      const leaves = [];
      for (let j = nu[i * 24 + 19]; j < nu[i * 24 + 20]; j++) if (kids[j] < G) leaves.push(kids[j]);
      for (const g of leaves) {
        let nb = leaves.filter(h => h !== g && nu[h * 24 + 15] === nu[g * 24 + 15]);
        if (nb.length > 3) nb = nb.sort((a, b) => d2(g, a) - d2(g, b)).slice(0, 3);
        nb.forEach((h, j) => { out[g * 4 + j] = h; });
      }
    }
    return out;
  }

  // ---- progressive tree lighting (runtime/wgsl/progressive.wgsl) -------------------------------------------

  initProgressive(device, sources, asset) {
    const N = asset.header.buffers.nodes.count, G = this.count;
    const buf = (label, size, usage = B.STORAGE) => device.createBuffer({ label, size, usage: usage | B.COPY_DST | B.COPY_SRC });
    // parent of every node, from the kids lists (SPEC 2.2); a part root and the scene root point at themselves
    const nu = new Uint32Array(asset.raw.nodes), kids = new Uint32Array(asset.raw.kids);
    const state = new Uint32Array(N * 4);
    for (let i = 0; i < N; i++) { state[i * 4] = 0x3f800000; state[i * 4 + 3] = i; }   // T = 1.0, own parent
    for (let i = 0; i < N; i++) {
      for (let j = nu[i * 24 + 19]; j < nu[i * 24 + 20]; j++) {
        const c = kids[j];
        if (nu[c * 24 + 15] !== 0xffffffff && nu[i * 24 + 15] !== 0xffffffff) state[c * 4 + 3] = i;   // not under the scene root
      }
    }
    this.lpInitialState = state;
    Object.assign(this.buffers, {
      lp_params: buf('progressive params', 32, B.UNIFORM),
      node_light: buf('node incoming light', N * 18 * 16),
      node_state: buf('node lighting state', N * 16),
      work: buf('progressive worklist', (4 + LP_MAX_BUDGET) * 4),
    });
    this.lpReset();
    const res = {
      0: this.buffers.uniform, 1: this.buffers.gaussians, 2: this.buffers.nodes, 3: this.buffers.vgrid,
      4: this.buffers.parts, 7: this.buffers.materials, 8: this.buffers.colour, 9: this.buffers.kids,
      10: this.buffers.probe_params, 11: this.buffers.probe_data, 13: this.buffers.lo_dir, 14: this.buffers.lo_b,
      18: this.buffers.emit_nodes, 19: this.buffers.emit_params, 21: this.buffers.lp_params,
      22: this.buffers.node_light, 23: this.buffers.node_state, 24: this.buffers.work,
    };
    const module = device.createShaderModule({ label: 'progressive', code: sources.common + '\n' + sources.material_ext + '\n' + sources.progressive });
    this.lp = {};
    for (const [entry, binds] of Object.entries(PROGRESSIVE)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      this.lp[entry] = { pipe, group: device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: binds.map(b => ({ binding: b, resource: { buffer: res[b] } })),
      }) };
    }
    this.lpParams = new ArrayBuffer(32);
    // M39: each leaf's iso_norm for the progressive transfer, once (uniform: just the Gaussian count)
    const iu = new Uint32Array(16); iu[1] = G; iu[13] = this.materialCount;
    device.queue.writeBuffer(this.buffers.uniform, 0, iu);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass({ label: 'lp_iso_init' });
    pass.setPipeline(this.lp.lp_iso_init.pipe); pass.setBindGroup(0, this.lp.lp_iso_init.group);
    pass.dispatchWorkgroups(Math.ceil(G / WG)); pass.end();
    device.queue.submit([enc.finish()]);
  }

  // Forget all node light (every node unlit, histories cleared).
  lpReset() {
    this.device.queue.writeBuffer(this.buffers.node_state, 0, this.lpInitialState);
    const w = new Uint32Array(4);
    w[2] = 0x3f800000;                             // acceptance probability 1.0
    this.device.queue.writeBuffer(this.buffers.work, 0, w);
  }

  // One frame of progressive lighting: schedule, shade up to `budget` nodes, then every leaf's transfer to the eye
  // (and to the 14 probe directions for 1 / 8 of the leaves), then the probes' emitters and gather as before.
  // Motion blur records (motion_blur.wgsl) before any lighting kernel reads them. A part that stopped this frame was
  // stretched last frame, so it counts as moved once more (its opacity-map layer is re-splatted sharp); returns
  // whether any part's records changed.
  // the moving parts' records (and last frame's movers', back to rest); nothing while everything is still
  encodeBlur(encoder) {
    const moved = this.movedMask ?? 0;
    this.blurMask = moved | this.prevMovedMask;
    this.prevMovedMask = moved;
    if (!this.blurMask) return false;
    const { partRuns } = this.blur, list = [];
    let threads = 0;
    partRuns.forEach((runs, p) => {
      if (!runs || p > 31 || !(this.blurMask & (1 << p))) return;
      for (const [a, b] of runs) { list.push(a, b, threads, 0); threads += b - a; }
    });
    if (!threads) return true;
    new Uint32Array(this.blur.params)[3] = list.length / 4;
    this.device.queue.writeBuffer(this.blur.ub, 0, this.blur.params);
    this.device.queue.writeBuffer(this.blur.ranges, 0, new Uint32Array(list));
    const pass = encoder.beginComputePass({ label: 'blur_records' });
    pass.setPipeline(this.blur.pipe); pass.setBindGroup(0, this.blur.group);
    pass.dispatchWorkgroups(Math.ceil(threads / 64)); pass.end();
    return true;
  }

  encodeProgressive(encoder, eye, timers, { budget, minAng, refresh, changed, resetDepth }) {
    if (this.blur && this.encodeBlur(encoder)) changed = true;
    this.writeUniform(eye);
    const f = new Float32Array(this.lpParams), u = new Uint32Array(this.lpParams);
    u[0] = Math.min(budget, LP_MAX_BUDGET); f[1] = minAng; u[2] = refresh; u[3] = resetDepth;
    u[4] = changed ? 1 : 0; f[5] = LP_BLEND; u[6] = this.frame % 8;
    this.device.queue.writeBuffer(this.buffers.lp_params, 0, this.lpParams);
    const run = (entry, threads, timer) => {
      const { pipe, group } = this.lp[entry];
      const pass = encoder.beginComputePass({ label: entry, timestampWrites: timer ? timers.writes(timer) : undefined });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(threads / 64)));
      pass.end();
    };
    run('lp_schedule', this.nodeCount);
    run('lp_shade', u[0], 'lp_shade');
    run('lp_finish', 1);
    run('lp_transfer', this.count, 'lp_transfer');
    run('lp_transfer_dirs', Math.ceil(this.count / 8));
    // the probes' emitters and gather, unchanged (bounce.wgsl)
    const pf = new Float32Array(this.params), pu = new Uint32Array(this.params);
    pf[12] = probeAlpha(this.firstGather);
    pu[13] = 0;
    const sub = this.round < FULL_ROUNDS ? 1 : SUBSET;
    pu[16] = sub; pu[17] = this.round % sub;
    this.round++;
    this.firstGather = false;
    this.device.queue.writeBuffer(this.buffers.probe_params, 0, this.params);
    const runB = (entry, threads, group = 0) => {
      const { pipe, groups, timer = entry } = this.bounce[entry];
      const pass = encoder.beginComputePass({ label: entry, timestampWrites: BOUNCE_TIMERS.includes(timer) ? timers.writes(timer) : undefined });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, groups[group]);
      pass.dispatchWorkgroups(Math.ceil(threads / 64));
      pass.end();
    };
    runB('node_leaf', this.count);
    for (let d = this.maxDepth - 1; d >= 1; d--) runB('node_up', this.levelCounts[d], d);
    runB('node_sh2', this.nodeCount);
    runB('probe_gather', Math.ceil(this.probeCount * NCELL / sub));
    runB('probe_valid', this.probeCount);
  }

  // Every slot empty: (v = 0, lamp 0xFFFF, count 0), SPEC §3.3.
  resetHistory() {
    const init = new Uint32Array(this.count * 8);
    for (let i = 1; i < init.length; i += 2) init[i] = EMPTY_SLOT;
    this.device.queue.writeBuffer(this.buffers.lamp_state, 0, init);
  }

  // Host-side lamp record: centre and rectangle half-axes (SPEC §2.5).
  setLamp(index, center, size, prevCenter, emissive = false) {
    // unified lights: one sphere record per light for ema_light's motion test, from the parts' transforms (each
    // frame, so the last centre is last frame's)
    if (this.unified) {
      if (this.partXf) this.device.queue.writeBuffer(this.buffers.lamps, 0, this.lightSet.records(this.partXf));
      return;
    }
    const f = new Float32Array(this.lampBytes);
    const base = new Float32Array(this.baseLamps);
    const o = index * 24;
    // An emissive asset ships its analytic lamp disabled. It is enabled in both modes: the emitter light uses
    // lamps[0] as the shadow history's motion descriptor, and the analytic comparison mode lights with it. In that
    // mode it sits 1.5 cm in front of the emitter panel, so the panel's own Gaussians do not shadow it.
    const shift = this.assetEmitters && !emissive ? 0.015 : 0;
    const nrm = [...base.subarray(o + 12, o + 15)];
    f.set(center.map((v, k) => v + shift * nrm[k]), o);
    for (const k of [16, 32].map(b => o + b / 4)) for (let j = 0; j < 3; j++) f[k + j] = base[k + j] * size;
    f[o + 7] = base[o + 7] * size;            // sphere radius
    f.set(prevCenter.map((v, k) => v + shift * nrm[k]), o + 20);
    if (this.assetEmitters) new Uint32Array(this.lampBytes)[o + 15] = 1;
    this.device.queue.writeBuffer(this.buffers.lamps, 0, this.lampBytes);
  }

  // One-sided or two-sided emission (SPEC 2.6 flag bit 2) for every emissive material and for the light cut.
  setOneSided(on) {
    this.emitOneSided = on;
    const u = new Uint32Array(this.materialBytes);
    for (let m = 0; m < this.materialCount; m++) {
      if (u[m * 8 + 5] & 1) u[m * 8 + 5] = on ? u[m * 8 + 5] | 4 : u[m * 8 + 5] & ~4;
    }
    this.device.queue.writeBuffer(this.buffers.materials, 0, this.materialBytes);
  }

  // A box's own material: diffuse flakes mixed with specular flakes (material_ext.wgsl), `shine` = the specular
  // weight k, `rough` = the flake roughness r (S = diag(1, r^2, r^2) along the Gaussian's short axis).
  setBoxMaterial(name, shine, rough) {
    const m = this.boxMaterial[name];
    if (m !== undefined) this.setMaterialShine(m, shine, rough);
  }

  // Any material's specular mix (material_ext.wgsl): shine k and flake roughness r. `keepPhase` leaves the
  // material's diffuse phase (flat closed form or VNDF samples) as the asset set it.
  setMaterialShine(m, shine, rough, keepPhase = false) {
    const f = new Float32Array(this.materialBytes), u = new Uint32Array(this.materialBytes);
    const o = m * 8, a = Math.max(rough * rough, 1e-4);
    f.set([1, a, a], o);
    if (!keepPhase) u[o + 3] = 0;                  // diffuse part: the flat-flake closed form
    u[o + 5] = (u[o + 5] & ~8) | (shine > 0 ? 8 : 0);
    u[o + 6] = ((u[o + 6] & 0xffff0000) | toHalf(shine)) >>> 0;   // high half: reflection coverage (flags bit 5)
    this.device.queue.writeBuffer(this.buffers.materials, 0, this.materialBytes);
  }

  // Per material for project.wgsl's RenderFrame.mat_view (16 slots): coverage peak (flags bit 5, pad0's high half),
  // Fresnel F0, specular weight k, and flags (1 coverage set, 2 emissive: no view reflection). Reflection coverage,
  // experiments/reflection_coverage.
  materialView(f0 = 0.04, f0Coverage = f0) {   // f0Coverage: glass (coverage materials) only, e.g. a stylized 0.2
    const u = new Uint32Array(this.materialBytes), out = new Float32Array(64);
    for (let m = 0; m < Math.min(this.materialCount, 16); m++) {
      const fl = u[m * 8 + 5], p0 = u[m * 8 + 6], em = (fl & 1) !== 0;
      out.set([fl & 32 ? fromHalf(p0 >>> 16) : 0, fl & 32 ? f0Coverage : f0, !em && fl & 8 ? fromHalf(p0 & 0xffff) : 0, (fl & 32 ? 1 : 0) | (em ? 2 : 0)], 4 * m);
    }
    return out;
  }

  // M41 transmission (common.wgsl transmit_m): `transl` in [0, 0.95] lets light further into the material's own part
  // (k = 1 - transl), `tint` in [0, 1] colours it by the albedo (s). Both 0 = today's look. Emissive materials skip it.
  setMaterialTransmit(m, transl, tint) {
    const u = new Uint32Array(this.materialBytes);
    const o = m * 8;
    if (u[o + 5] & 1) return;
    const on = transl > 0 || tint > 0;
    u[o + 5] = (u[o + 5] & ~16) | (on ? 16 : 0);
    u[o + 7] = on ? (toHalf(tint) | (toHalf(1 - Math.min(transl, 0.95)) << 16)) >>> 0 : 0;
    this.device.queue.writeBuffer(this.buffers.materials, 0, this.materialBytes);
    this.runIsoInit();
  }

  setBoxTransmit(name, transl, tint) {
    const m = this.boxMaterial[name];
    if (m !== undefined) this.setMaterialTransmit(m, transl, tint);
  }

  // Flake shape for every material: S = diag(1, r^2, r^2) in the Gaussian's
  // frame (x = short axis). r near 0 keeps the closed-form flat-flake phase;
  // rougher flakes use 8 fixed VNDF samples.
  // The wall roughness slider. Skips emissive materials, materials that carry their own flakes (asset.json
  // `materials[i].own_flakes`) and the boxes' own materials (setBoxMaterial).
  setRoughness(r) {
    const f = new Float32Array(this.materialBytes), u = new Uint32Array(this.materialBytes);
    // the floor's own material follows the walls until it is made shiny (main.js floorCustom)
    const ms = [...Array(this.assetMaterialCount).keys(), ...(this.floorCustom || this.boxMaterial.floor === undefined ? [] : [this.boxMaterial.floor])];
    for (const m of ms) {
      const o = m * 8;
      if ((u[o + 5] & 1) || this.materialInfo[m]?.own_flakes) continue;
      const a = Math.max(r * r, 1e-4);
      f.set([1, a, a], o);
      u[o + 3] = r <= 0.011 ? 0 : 8;
    }
    this.device.queue.writeBuffer(this.buffers.materials, 0, this.materialBytes);
  }

  writeUniform(eye) {
    const f = new Float32Array(this.uniform), u = new Uint32Array(this.uniform);
    u[0] = this.frame++; u[1] = this.count; u[2] = this.partCount; u[3] = this.lampCount;
    f[4] = THETA; f[5] = ALPHA_MIN; f[6] = ALPHA_PART; u[7] = 1;      // one slot per lamp (SPEC §3.3 stub)
    f.set(eye, 8); u[11] = STRATUM_STEP;
    u[12] = this.lightSamplesAxis; u[13] = this.materialCount; u[14] = this.historyMode;   // 1: N5 count rule (SPEC 3.3)
    u[15] = (this.iso ? 1 : 0) | (this.iso && this.shProbes ? 2 : 0) | (this.iso && this.asgSpec ? 4 : 0) | (this.iso && this.transmit ? 8 : 0)   // M39 transfer (bit 0), SH2 probes (bit 1), P3 ASG probe cells toward the eye (bit 2), M41 transmission (bit 3)
      | (this.iso && this.specSplit ? 16 : 0) | (this.iso && this.sheen ? 32 : 0);   // reflection coverage: specular toward the eye kept apart (bit 4), view-only probe sheen (bit 5)
    this.device.queue.writeBuffer(this.buffers.uniform, 0, this.uniform);
  }

  // screen-space LOD with lit nodes (lod.js, ?lodlight=1/2): after lod_select and lod_colour (which leaves each drawn
  // node its leaves' mean shadow T and bounce), the nodes' direct light (emissive.wgsl shade_emit_nodes, then the
  // environment's distant lights)
  encodeNodeDirect(encoder) {
    const e = this.emit[this.lodCut ? 'shade_emit_nodes_list' : 'shade_emit_nodes'];
    if (!this.lodLight || !e) return;
    const pass = encoder.beginComputePass({ label: 'shade_emit_nodes' });
    pass.setPipeline(e.pipe); pass.setBindGroup(0, e.groups[0]);
    if (this.lodCut) pass.dispatchWorkgroupsIndirect(this.lodList, 32);   // lod_args: the node list's size
    else pass.dispatchWorkgroups(Math.ceil(this.lodN / WG));
    pass.end();
    this.env?.encodeNodes(encoder);   // the distant lights (sun) on top
  }

  // changed: a part or the light moved this frame (shadow_cull's answer only changes then; the camera does not matter)
  encode(encoder, eye, timers, emissive = false, changed = true) {
    this.writeUniform(eye);
    if (this.blur && this.encodeBlur(encoder)) changed = true;
    const groups = Math.ceil(this.count / WG);
    let entries = emissive ? [this.pfield && this.unified ? 'walk_emit_pf' : 'walk_emit', this.unified ? 'ema_light' : 'ema', 'shade_emit'] : KERNELS.map(k => k[1]);
    if (emissive && this.pfd && this.unified) { entries = this.cluster ? ['cluster_T', 'shade_emit_pfd'] : ['shade_emit_pfd']; this.writePfBounds(); }   // field shadows per cut node: no walk, cull or history
    if (changed) this.hsDirty = true;
    // no lamp: only the shade pass (its lamp term is zero), no lamp shadow walk or history over every Gaussian
    if (this.lampOff) entries = entries.filter(e => e.startsWith('shade') || e === 'cluster_T');
    // light-side shadows (light_shadow.js, off by default) in place of the cull, walk and history
    if (this.lampOff) { /* no lamp shadows */ }
    else if (emissive && !this.unified && this.lightShadow?.method && this.lightShadow.encode(encoder, timers, this.blur ? this.blurMask : this.movedMask ?? 0, changed)) {
      entries = ['shade_emit'];
    } else if (emissive && this.hierShadow && this.blur && this.blurMask && !(this.pfd && this.unified)) {
      // motion blur: shadow_cull's rest-frame node boxes do not cover a stretched part, so while one is stretched
      // (or just stopped) every Gaussian walks; the cull runs again on the next frame
      encoder.clearBuffer(this.buffers.node_status);
      this.hsDirty = true;
    } else if (emissive && this.hierShadow && this.hsDirty && !(this.pfd && this.unified)) {
      this.hsDirty = false;
      // shadow_cull per depth, part roots first, before walk_emit reads the leaves' status (untimed: one slot per pass)
      this.hsLevels.forEach((l, k) => {
        const pass = encoder.beginComputePass({ label: 'shadow_cull' });
        pass.setPipeline(this.hs.shadow_cull.pipe); pass.setBindGroup(0, this.hs.shadow_cull.groups[k]);
        pass.dispatchWorkgroups(Math.ceil(l.count / WG)); pass.end();
      });
    }
    if (this.lodCut) entries = entries.map(e => e === 'shade_emit' ? 'shade_emit_list' : e);   // ?lodlight=2: the cut only
    for (const entry of entries) {
      const e = this.emit[entry];
      const pass = encoder.beginComputePass({ label: entry, timestampWrites: timers.writes(e?.timer ?? (entry === 'ema_light' ? 'ema' : entry)) });   // ema_light reports into ema's slot
      pass.setPipeline(e?.pipe ?? this.pipelines[entry]);
      pass.setBindGroup(0, e?.groups[0] ?? this.groups[entry]);
      if (entry === 'shade_emit_list') pass.dispatchWorkgroupsIndirect(this.lodList, 16);   // lod_args: the leaf list
      else pass.dispatchWorkgroups(groups);
      pass.end();
    }
    this.env?.encodeDirect(encoder, timers, changed);   // distant lights (sun) on top of shade_emit (environment.js)
  }
}
