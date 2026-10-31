// The runtime lighting kernels (runtime/wgsl, SPEC v0.2 §4) in the browser
// frame: lamp segment walk, per-Gaussian EMA, then M36 shading into
// colour[g], which the project pass reads in the lit view. Kernel sources are
// shared with the native runner (runtime/run_kernels.py), unchanged.

import { Lights, MAX_LIGHTS } from './lights.js';
import { LightsMap } from './lights_map.js';

// per-Gaussian kernels built from their own files: none left. The analytic lamp (walk_lamp, shade) and the sampled
// shadow walk with its history (walk_emit, ema, shadow_cull) were removed 2026-10-29 (David); their WGSL stays for
// the native harness and experiments.
export const KERNELS = [];
// Probe-loop bounce (runtime/wgsl/bounce.wgsl): entry points in dispatch order, and the bindings each one uses.
const BOUNCE = {
  bounce_dirs: [0, 1, 2, 3, 4, 7, 10, 11, 12, 14],
  node_leaf: [0, 1, 2, 3, 4, 10, 13, 14, 15, 80],   // 80: the tree slots (sc_flag, bounce.wgsl tree_slot)
  node_up: [0, 2, 9, 10, 15, 17, 29, 80],
  node_sh2: [10, 15, 16, 80],
  probe_gather: [0, 2, 3, 4, 5, 9, 10, 11, 16, 40, 80],   // 40: the environment (sky for rays that leave the scene)
  probe_valid: [0, 4, 10, 11],
  // M39 stage 2 (SH2 probes)
  iso_xfer: [0, 1, 2, 3, 4, 10, 12, 26],
  bounce_sh: [0, 1, 2, 3, 4, 7, 10, 11, 14, 25, 26],
  probe_sh2: [10, 11, 25],
  // near-field reflection (off by default; setNearReflection)
  near_refl: [0, 1, 2, 7, 8, 10, 14, 16, 27, 28, 80],
  // near-field visibility (with nearField on, off by default): k wide cones per Gaussian (the 128-cell near_vis walk
  // was archived 2026-10-29; its WGSL stays)
  near_cones: [0, 1, 2, 3, 4, 9, 10, 12, 27],
};
export const BOUNCE_TIMERS = ['near_vis', 'direct_dirs', 'bounce_dirs', 'node_leaf', 'probe_gather', 'iso_xfer', 'bounce_sh', 'probe_sh2'];
// Emissive Gaussian lights (runtime/wgsl/emissive.wgsl): entry -> bindings, and the timer slot each one reports into.
const EMISSIVE = {
  shade_emit: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19], 'shade'],
  direct_dirs_emit: [[0, 1, 2, 3, 4, 6, 7, 13, 18, 19], 'direct_dirs'],
};
const MAX_MATERIALS = 32;
const LOB = 3;   // lo_b vec4s per Gaussian (common.wgsl LOB)
// f32 -> IEEE half bits (for pack2x16float fields written from the host)

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
const MAX_EMITTERS = 16384, EMIT_THETA = 4;
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
    // loops, not Math.min(...array): spreading a large array overflows the call stack (2026-10-31)
    const lo = [0, 1, 2].map(k => idx.reduce((m, i) => Math.min(m, pos(i)[k]), Infinity));
    const hi = [0, 1, 2].map(k => idx.reduce((m, i) => Math.max(m, pos(i)[k]), -Infinity));
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
// ProbeParams: 832 B, then dims (16), levels (64), ground (4096), static volumes (512 + 16) (bounce.wgsl; SPEC 2.7)
// + node probes (256 vec4f, archived; bounce.wgsl ProbeParams.nodep stays, zero)
const PROBE_PARAMS_BYTES = 832 + 16 + 64 + 4096 + 528 + 4096;
// probes per movable part: a 2x2x2 cube in its rest frame (bounce.wgsl part_probe_frame). Node probes, one per tree
// patch just outside the surface (?nodeprobes=1, 2026-10-28), were archived 2026-10-29 (David: revisit if the cube
// doesn't suffice for moving characters); bounce.wgsl keeps its NODE_PROBES branch, off.
const PPK = 8;
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
    this.lampOff = !this.unified && !this.assetEmitters;   // no emitters: no lamp (the analytic / generated panel lamp is gone, 2026-10-29)
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
    // sender LOD (bounce.wgsl sc_select): only the bounce senders some probe can resolve are lit; the rest of the tree
    // is lit as the nodes every probe sees whole. Needs the nodes' V grids (main.js) and SH2 probes. On by default
    // (David, 2026-10-29: hill_bin 505 -> 246 ms a frame on the M4, no visible change); ?sendercut=0 per-leaf senders
    this.senderCut = lodN > 0 && !!asset.lodNodes.vgrid && new URLSearchParams(globalThis.location?.search ?? '').get('sendercut') !== '0';
    let vgridBytes = r.vgrid;
    if (lodN && asset.lodNodes.vgrid) {   // nodes' own V grids: lit nodes only (?lodlight)
      const v = new Uint8Array(r.vgrid.byteLength + lodN * 128);
      v.set(new Uint8Array(r.vgrid)); v.set(new Uint8Array(asset.lodNodes.vgrid, 0, lodN * 128), r.vgrid.byteLength);
      vgridBytes = v.buffer;
    }
    this.buffers = {
      uniform: buf('frame uniforms', 64, B.UNIFORM),
      gaussians: gaussBuf, parts: partBuf,
      nodes: buf('nodes', r.nodes), kids: buf('kids', r.kids), vgrid: buf('vgrid', vgridBytes),
      lamps: buf('lamps', this.lampBytes), materials: buf('materials', this.materialBytes),
      lamp_state: buf('lamp_state', (this.count + lodN) * 32),
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
    if (this.senderCut) {
      const N = this.count + lodN, nu = new Uint32Array(r.nodes);
      this.scSlots = Math.min(lodN, 1 << 18);   // cut nodes lit per frame at most (sc_L); past it a node's children decide
      this.scList = device.createBuffer({ label: 'sender cut lists', size: (16 + N) * 4, usage: B.STORAGE | B.INDIRECT | B.COPY_DST | B.COPY_SRC });
      device.queue.writeBuffer(this.scList, 0, new Uint32Array([0, 0, 0, 0, 0, 0, 0, 0, this.scSlots, 0, 0, 0, 0, 0, 0, 0]));
      this.scStale = true;
      this.buffers.sc_flag = buf('sender cut flags', N * 4);
      // per slot x 8: each face's SH2 transfer and exposure (0-5, sc_xfer) and the direct light it catches (6-7, sc_L
      // in emissive.wgsl / environment.wgsl): one buffer, so node_cut stays within 10 storage buffers
      this.buffers.sc_X = buf('sender cut faces', this.scSlots * 8 * 16);
      // every node by depth (sc_select runs a depth at a time, parents before children)
      const byDepth = [];
      for (let n = 0; n < N; n++) (byDepth[nu[n * 24 + 21]] ??= []).push(n);
      const order = new Uint32Array(N);
      let off = 0;
      this.scLevels = [];
      byDepth.forEach((list, d) => {
        if (!list?.length) return;
        order.set(list, off);
        const lv = device.createBuffer({ label: `sender cut depth ${d}`, size: 16, usage: B.UNIFORM | B.COPY_DST });
        device.queue.writeBuffer(lv, 0, new Uint32Array([d, off, list.length, 0]));
        this.scLevels.push({ count: list.length, lv });
        off += list.length;
      });
      this.buffers.sc_nodes = buf('sender cut nodes by depth', order);
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

    this.uniform = new ArrayBuffer(64);
    // M39 direction-independent transfer with SH2 probes (stage 2) is the only lighting model since 2026-10-29 (David
    // archived M36's view-dependent flake shading and the 128-cell probe bounce; their WGSL branches stay for the
    // native tools). FrameUniforms.iso bits 0 and 1 are always set.
    this.transmit = false;  // M41 transmission (only with iso), off by default
    this.specSplit = false; // reflection coverage / Fresnel (main.js): specular toward the eye in colour[G + g] and lo_b lanes
    this.sheen = false;     // Fresnel's probe sheen for matte materials (bounce_sh, lo_b lanes)
    this.initIso(device, sources);
    this.initBounce(device, sources, asset);
    this.initNodeParents(device, asset);
    this.initEmissive(device, sources);
    if (this.senderCut) this.initSenderCut(device);
    // unified lights: an opacity map per light (lights_map.js), the only method since 2026-10-29 (David archived the
    // per-receiver walk)
    if (this.unified) this.lightsMap = new LightsMap(device, sources, this);
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
    if (this.lodN && (this.lodLight || this.senderCut)) this.isoInit.nodes = device.createComputePipeline({ label: 'iso_init_nodes', layout, compute: { module, entryPoint: 'iso_init_nodes' } });
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

  // every node's parent (part roots and the scene root: none), for the passes that walk up the tree: the opacity
  // maps' motion-blur cut (light_shadow.wgsl lsm_splat, sun_map.wgsl, lights_map.wgsl) and the sender cut
  initNodeParents(device, asset) {
    const N = this.nodeCount, nu = new Uint32Array(asset.raw.nodes), kids = new Uint32Array(asset.raw.kids);
    const parent = new Uint32Array(N).fill(0xffffffff);
    for (let i = this.count; i < N; i++) {
      if (nu[i * 24 + 15] === 0xffffffff) continue;
      for (let j = nu[i * 24 + 19]; j < nu[i * 24 + 20]; j++) parent[kids[j]] = i;
    }
    this.buffers.node_parent = device.createBuffer({ label: 'node parents', size: N * 4, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC });
    device.queue.writeBuffer(this.buffers.node_parent, 0, parent);
  }

  initEmissive(device, sources) {
    const buf = (label, size, usage = B.STORAGE) => device.createBuffer({ label, size, usage: usage | B.COPY_DST | B.COPY_SRC });
    this.buffers.emitters = buf('emitters', MAX_EMITTERS * 64);
    this.buffers.emit_nodes = buf('emitter tree', (2 * MAX_EMITTERS - 1) * 96);   // EmitNode + its ASG covariance (packEmitNodes)
    this.buffers.emit_params = buf('emit params', 64, B.UNIFORM);
    this.emitTheta = EMIT_THETA;
    this.emitMode = 1;
    this.emitterCount = 0;
    this.emitEdgeSoft = 0.2;   // one-sided emitters fade over |cos| < this around their plane (0 = hard cut)
    this.asgSpec = 1;       // P3 ASG specular (EmitParams.n5 bits 3-4), on by default (David, 2026-10-25): 1 lights as shapes in shiny highlights, 3 with the spec cut
    const res = {
      0: this.buffers.uniform, 1: this.buffers.gaussians, 2: this.buffers.nodes, 3: this.buffers.vgrid,
      4: this.buffers.parts, 6: this.buffers.lamp_state, 7: this.buffers.materials, 8: this.buffers.colour,
      9: this.buffers.kids, 13: this.buffers.lo_dir, 18: this.buffers.emit_nodes, 19: this.buffers.emit_params,
      61: this.lodFlagBuf, 62: this.lodList,
      81: this.scList, 83: this.buffers.sc_X,
    };
    const module = device.createShaderModule({ label: 'emissive', code: sources.common + '\n' + sources.material_ext + '\n' + sources.emissive });
    this.emitModule = module;
    this.emitSources = sources;
    this.emitRes = res;
    this.emit = {};
    let kinds = this.lodCut ? { ...EMISSIVE, shade_emit_list: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19, 62], 'shade'],
      shade_emit_nodes_list: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19, 61, 62], 'shade'],
      shade_emit_nodes: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19, 61], 'shade'] }   // lodCut = false at run time: stage 1
      : this.lodLight ? { ...EMISSIVE, shade_emit_nodes: [[0, 1, 2, 3, 4, 6, 7, 8, 18, 19, 61], 'shade'] } : EMISSIVE;
    if (this.senderCut) Object.assign(kinds === EMISSIVE ? (kinds = { ...EMISSIVE }) : kinds, {
      direct_dirs_emit_sc: [[0, 1, 2, 3, 4, 6, 7, 13, 18, 19, 81], 'direct_dirs'],
      direct_dirs_nodes_sc: [[0, 1, 2, 3, 4, 6, 7, 18, 19, 81, 83], 'sc'] });
    for (const [entry, [binds, timer]] of Object.entries(kinds)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module, entryPoint: entry } });
      this.emit[entry] = { pipe, timer, groups: [device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        // motion blur: the shadow walks traverse other parts' stretched trees (the receiver's own records stay)
        entries: binds.map(b => ({ binding: b, resource: { buffer: b === 2 && this.blur && entry.startsWith('walk_emit') ? this.buffers.nodes_blur : res[b] } })),
      })] };
    }
  }

  // The light: an asset's own emissive Gaussians (setAssetEmitters), or with ?lights=unified every emissive part
  // (setUnifiedEmitters). An asset without emitters has no lamp (one dark emitter, nothing lit directly): the panel
  // lamp generated from lamps.bin's rectangle, and its analytic twin, are gone (David, 2026-10-29). Returns the count.
  setEmitters(center) {
    const key = [...center, this.emitTheta, this.emitMode, this.emitOneSided, this.emitEdgeSoft, this.asgSpec].join();
    if (this.unified) return this.setUnifiedEmitters();
    if (key === this.emitKey) return this.emitterDrawn;
    this.emitKey = key;
    if (this.assetEmitters) return this.setAssetEmitters(center);
    return this.setDarkEmitter(center);
  }

  // Unified lights: the whole scene's light tree (lights.js) from the parts' current transforms (main.js sets partXf)
  // and the live material flags; rewritten only when a light moved, switched or changed gain. No light is drawn here:
  // emitters are scene Gaussians.
  setUnifiedEmitters() {
    if (!this.partXf) return 0;
    const tree = this.lightSet.update(this.partXf, new Uint32Array(this.materialBytes, 0, this.materialCount * 8));
    const params = [this.emitTheta, this.emitMode, this.emitEdgeSoft, this.asgSpec].join();
    if (!tree && this.lightSet.key === this.unifiedKey) {
      // lights unchanged; EmitParams still follow the panel
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
    epf.set(normal, 8); epu[11] = this.asgSpec << 3;   // ASG bits (emissive.wgsl EmitParams.n5; bits 0-2 were the walk's)
    epf[12] = this.emitEdgeSoft;   // EmitParams.hier / stride (13, 14): the archived walk's, 0
    // EmitParams.lights; bits 1-2 are a DEBUG switch, ?lightdbg=<bits> (emissive.wgsl shade_emitters), bit 3 self-shadow
    // (?selfshadow=1: a light's own part occludes it, emissive.wgsl walk_tau_light), unified only
    const eq = new URLSearchParams(globalThis.location?.search ?? '');
    epu[15] = this.unified ? 1 | (Number(eq.get('lightdbg') ?? 0) & 6) | (eq.get('selfshadow') === '1' ? 8 : 0) : 0;
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
    // count 0: the shaders skip the walk over this dark leaf (emissive.wgsl shade_leaf, ~13 ms a frame on hill)
    this.writeEmitParams(0, root, 0, nodes.length / 16, 0xffffffff, 0, [0, 0, 1]);
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
    // emitting normal: the first emitter's short axis R x (SPEC 2.6 one-sided convention)
    const [x, y, z, w] = rec.subarray(4, 8);
    const nrm = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)];
    // overlap correction as for the host panel: the panel's area over the emitters' summed footprint along the normal,
    // the area taken from the grid of emitter centres (distinct rows x distinct columns x spacing)
    const ax = [[2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]];
    const extent = ax.map(a => {
      const vals = Array.from({ length: n }, (_, i) => a[0] * rec[i * 16] + a[1] * rec[i * 16 + 1] + a[2] * rec[i * 16 + 2]);
      const lo = vals.reduce((m, v) => Math.min(m, v), Infinity), hi = vals.reduce((m, v) => Math.max(m, v), -Infinity);
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
    // tree slots (bounce.wgsl tree_slot; 2026-10-30): with sender LOD a start of 1/16 of the nodes (at least 64k),
    // grown when a cut needs more (the room's cut takes 40% of its nodes, hill_bin's 1.2%); without it one per node
    this.slotCap = this.senderCut ? Math.min(N, Math.max(65536, Math.ceil(N / 16))) : N;
    if (!this.senderCut) {   // identity slots (sc_flag = node << 2)
      const id = new Uint32Array(N);
      for (let n = 0; n < N; n++) id[n] = n << 2;
      this.buffers.sc_flag = device.createBuffer({ label: 'tree slots (one per node)', size: N * 4, usage: B.STORAGE | B.COPY_DST });
      device.queue.writeBuffer(this.buffers.sc_flag, 0, id);
    }
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
      // near-field visibility: a placeholder until near field is first switched on (setNearField; 142 MB on hill_bin,
      // off by default; bounce.wgsl near_byte reads 1 meanwhile)
      nearvis: buf('near-field visibility (placeholder)', 128),
      // common.wgsl LOB: one direct value per Gaussian, and the bounce toward the eye / the probes plus the specular
      // lanes (2026-10-30; were 14 + 15 vec4 per Gaussian, one per direction)
      lo_dir: buf('direct toward the probes', G * 16),
      lo_b: buf('bounce (eye, probes, lanes)', G * LOB * 16),
      // per tree slot, plus the null slot (bounce.wgsl tree_slot): with sender LOD only the nodes the probes reach
      // get a slot, re-handed when the cut is re-selected (growSlots when they don't fit); without it, slot = node
      node_acc: buf('node emitter sums', (this.slotCap + 1) * 14 * 16),
      node_sh: buf('node emitter SH2 (halves)', (this.slotCap + 1) * 14 * 4),
      probe_sh: buf('probe SH2 (M39 stage 2)', NP * 9 * 16),
      xfer: buf('Gaussian transfer SH2 (M39 stage 2)', G * 3 * 16),
      // near-field reflection's two buffers: placeholders until it is first switched on (setNearReflection; ~410 MB on
      // hill_bin, unused by default; 2026-10-29). near_cones binds blk but writes it only with reflection on.
      blk: buf('near-field blockers (placeholder)', 16),
      lo_b_prev: buf('last frame bounce (placeholder)', 16),
    });
    this.blockersRaw = asset.raw.blockers ?? null;
    this.blkBytes = null;
    this.nearRefl = false;
    this.emitterBounceFix = false;
    this.nearCones = 8;   // near-field cone count, 6 or 8 (main.js default: 8)
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
    u[1381] = new URLSearchParams(globalThis.location?.search ?? '').get('probemargin') === 'old' ? 1 : 0;   // vol_info.y: the block margin on every gather
    f.set([1, 0, 0.02, 0.95], 12);              // alpha, force_near, vfloor, enclosed fraction
    // ?partprobes=0 (diagnosis, room layouts only): no probes riding with the parts, every Gaussian on the lattice
    const noPartProbes = !lay.terrain && new URLSearchParams(globalThis.location?.search ?? '').get('partprobes') === '0';
    u[16] = SUBSET; u[18] = (noPartProbes ? 0 : PPK * this.partCount) + volProbes;   // part_probes: movable parts' 8 each, then the volumes
    u[19] = 1;   // sh_full: bounce_sh every Gaussian every frame (< 0.5 ms); a quarter at a time sparkles while settling
    D14.forEach((d, k) => f.set([...d, 0], 20 + 4 * k));
    pinvD14().flat().forEach((v, i) => { f[76 + i] = v; });
    this.round = 0;
    this.forceNear = true;
    this.nearField = false;   // own-part visibility is the base (David, 2026-10-23); main.js syncs the panel's box
    this.firstGather = true;

    const module = device.createShaderModule({ label: 'bounce', code: sources.common + '\n' + sources.material_ext + '\n' + sources.bounce });
    const res = {
      0: this.buffers.uniform, 1: this.buffers.gaussians, 2: this.buffers.nodes, 3: this.buffers.vgrid,
      4: this.buffers.parts, 5: this.buffers.lamps, 6: this.buffers.lamp_state, 7: this.buffers.materials,
      9: this.buffers.kids, 10: this.buffers.probe_params, 11: this.buffers.probe_data, 12: this.buffers.nearvis,
      13: this.buffers.lo_dir, 14: this.buffers.lo_b, 15: this.buffers.node_acc, 16: this.buffers.node_sh,
      25: this.buffers.probe_sh, 26: this.buffers.xfer,
      8: this.buffers.colour, 27: this.buffers.blk, 28: this.buffers.lo_b_prev, 40: this.buffers.env,
      29: this.buffers.up_nodes, 80: this.buffers.sc_flag,
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
      const k = this.bounce[entry] = { pipe };
      k.build = () => { k.groups = entry === 'node_up' ? this.levels.map((_, d) => group(d)) : [group(0)]; };   // again after growSlots
      k.build();
    }
    this.bounce.near_cones.timer = 'near_vis';   // its slot in the timer list
    this.bounceModule = module;
    this.bounceRes = res;
  }

  // sender LOD (bounce.wgsl sc_select ...): its bounce-module passes; per-depth groups for sc_select / node_up_sc
  initSenderCut(device) {
    const res = this.bounceRes;
    Object.assign(res, { 35: this.buffers.node_parent, 41: this.buffers.env_T,
      81: this.scList, 82: this.buffers.sc_nodes, 84: this.scList, 85: this.buffers.sc_X });
    const SC = {
      sc_select: [[0, 2, 4, 10, 15, 17, 35, 80, 81, 82], 'sc'],
      sc_args: [[81], 'sc'],
      sc_shadow: [[0, 2, 6, 9, 41, 84], 'sc'],
      node_leaf_list: [[1, 2, 3, 4, 10, 13, 14, 15, 80, 84], 'node_leaf'],
      sc_xfer: [[0, 1, 2, 3, 4, 84, 85], 'sc'],
      node_cut: [[0, 1, 2, 3, 4, 10, 11, 15, 25, 80, 84, 85], 'sc'],
      node_up_sc: [[0, 2, 9, 10, 15, 17, 29, 80], 'sc'],
      node_sh2_sc: [[10, 15, 16, 80], 'sc'],
    };
    for (const [entry, [binds, timer]] of Object.entries(SC)) {
      const pipe = device.createComputePipeline({ label: entry, layout: 'auto', compute: { module: this.bounceModule, entryPoint: entry } });
      const group = lv => device.createBindGroup({ layout: pipe.getBindGroupLayout(0),
        entries: binds.map(b => ({ binding: b, resource: { buffer: b === 17 ? lv : res[b] } })) });
      const k = this.bounce[entry] = { pipe, timer };
      k.build = () => { k.groups = entry === 'sc_select' ? this.scLevels.map(l => group(l.lv)) : entry === 'node_up_sc' ? this.levels.map(l => group(l)) : [group(null)]; };
      k.build();
    }
    this.slotRead = device.createBuffer({ label: 'tree slot counts (readback)', size: 8, usage: B.MAP_READ | B.COPY_DST });
    this.slotReadBusy = false;
  }

  // a re-selected cut handed out more tree slots than node_acc / node_sh hold (sender LOD; bounce.wgsl sc_mark):
  // bigger arrays, the passes rebound, and the cut selected again next frame. Until then the nodes past the end read
  // zero (a frame or two of missing bounce).
  growSlots(need) {
    const N = this.nodeCount, cap = Math.min(N, Math.max(this.slotCap * 2, Math.ceil(need * 1.25)));
    if (cap <= this.slotCap) return;
    this.slotCap = cap;
    const buf = (label, size) => this.device.createBuffer({ label, size, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC });
    this.buffers.node_acc.destroy(); this.buffers.node_sh.destroy();
    this.buffers.node_acc = buf('node emitter sums', (cap + 1) * 14 * 16);
    this.buffers.node_sh = buf('node emitter SH2 (halves)', (cap + 1) * 14 * 4);
    Object.assign(this.bounceRes, { 15: this.buffers.node_acc, 16: this.buffers.node_sh });
    for (const k of Object.values(this.bounce)) k.build?.();
    this.scStale = true;
    this.requestFrame?.();   // main.js: the render-on-demand loop may be idle by the time the readback lands
  }

  // after a re-selection: read back how many tree slots the cut wanted (sc_list words 10, 11), one read at a time
  checkSlots(encoder) {
    if (this.slotReadBusy) return;
    this.slotReadBusy = true;
    encoder.copyBufferToBuffer(this.scList, 40, this.slotRead, 0, 8);
    setTimeout(() => this.device.queue.onSubmittedWorkDone().then(() => this.slotRead.mapAsync(GPUMapMode.READ)).then(() => {
      const [used, over] = new Uint32Array(this.slotRead.getMappedRange().slice(0));
      this.slotRead.unmap();
      this.slotReadBusy = false;
      if (over > 0) this.growSlots(used);
    }).catch(() => { this.slotReadBusy = false; }), 0);
  }

  // Near-field visibility (bounce.wgsl near_vis) on or off. Off = other parts never block the bounce within near_r:
  // only each Gaussian's own-part V grid does (David's working assumption, 2026-10-23).
  setNearField(on) {
    if (on && this.buffers.nearvis.size < this.count * 128) {   // first use: the real buffer, all clear
      this.buffers.nearvis = this.device.createBuffer({ label: 'near-field visibility', size: this.count * 128, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC });
      this.device.queue.writeBuffer(this.buffers.nearvis, 0, new Uint8Array(this.count * 128).fill(255));
      this.bounceRes[12] = this.buffers.nearvis;
      for (const k of Object.values(this.bounce)) k.build?.();
    }
    this.nearField = on;
    this.forceNear = true;   // both ways: iso_xfer's transfer holds the near-field visibility it was built with (2026-10-31)
    if (!on) {
      if (this.buffers.nearvis.size >= this.count * 128) this.device.queue.writeBuffer(this.buffers.nearvis, 0, new Uint8Array(this.count * 128).fill(255));
      if (this.blkBytes) this.device.queue.writeBuffer(this.buffers.blk, 0, this.blkBytes);   // drop the other-part blockers
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
    if (on && !this.blkBytes) this.initNearRefl();
    this.nearRefl = on;
    this.forceNear = true;
    if (this.blkBytes) this.device.queue.writeBuffer(this.buffers.blk, 0, this.blkBytes);
  }

  // near-field reflection's buffers, on first use: 16 (node, weight) blockers per Gaussian, 0-7 own part from the
  // asset's blockers.bin (runtime/bake_blockers.py; none without it), 8-15 other parts (near_cones while reflection is
  // on), and last frame's bounce; the passes that bind them get new bind groups
  initNearRefl() {
    const G = this.count, device = this.device;
    const buf = (label, size) => device.createBuffer({ label, size, usage: B.STORAGE | B.COPY_DST | B.COPY_SRC });
    this.buffers.blk = buf('near-field blockers', G * 128);
    this.buffers.lo_b_prev = buf('last frame bounce (near-field reflection)', G * LOB * 16);
    this.blkBytes = new Uint32Array(G * 32).fill(0xffffffff);
    for (let i = 1; i < G * 32; i += 2) this.blkBytes[i] = 0;
    if (this.blockersRaw) {
      const own = new Uint32Array(this.blockersRaw);
      for (let g = 0; g < G; g++) this.blkBytes.set(own.subarray(g * 16, g * 16 + 16), g * 32);
    }
    device.queue.writeBuffer(this.buffers.blk, 0, this.blkBytes);
    Object.assign(this.bounceRes, { 27: this.buffers.blk, 28: this.buffers.lo_b_prev });
    for (const entry of ['near_cones', 'near_refl']) this.bounce[entry].build();
  }

  // Near-field cones (bounce.wgsl near_cones): with near-field visibility on, k wide cones (6 or 8) per Gaussian for
  // the transmittance and (with reflection) the other-part blockers
  setNearCones(k) {
    this.nearCones = k === 6 ? 6 : 8;
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

  // Does a part move this frame change any Gaussian's transfer (bounce.wgsl iso_xfer)? The transfer is world-frame SH
  // of a Gaussian's rest-frame V and footprint times its near-field visibility (near_byte). With near-field visibility
  // off that last factor is not refreshed, so only a part turning changes anything: a pure translation leaves every
  // transfer as it was. With it on, any move may (its cones follow positions). (2026-10-31: drifting clouds, which
  // only translate, re-ran iso_xfer over every Gaussian each frame for nothing.)
  isoMoved(mask) {
    if (!mask) return false;
    const pf = this.partXf;
    if (!pf || this.nearField || ((this.gpuMovedMask ?? 0) & mask) !== 0) return true;
    for (let p = 0; p < Math.min(this.partCount, 32); p++) {
      if (((mask >>> p) & 1) === 0) continue;
      const o = p * 32;
      for (let r = 0; r < 3; r++) for (let j = 0; j < 3; j++) if (pf[o + 4 * r + j] !== pf[o + 12 + 4 * r + j]) return true;
    }
    return false;
  }

  // lightStill (main.js): no lighting input changed for SETTLE_FRAMES frames, so the light toward the probes, the tree
  // and the probes hold their settled values; only the eye-dependent reads run (the camera may be moving)
  encodeBounce(encoder, timers, lightStill = false) {
    lightStill &&= !this.forceNear && !this.scStale && !this.firstGather;
    const f = new Float32Array(this.params), u = new Uint32Array(this.params);
    f[12] = probeAlpha(this.firstGather);
    u[13] = this.forceNear ? 1 : 0;
    u[205] = 0;   // PP.refl.y: the archived near_vis walk's chunk start
    // For the first FULL_ROUNDS rounds refresh every ray and Gaussian each frame, so the probes settle quickly.
    const sub = this.round < FULL_ROUNDS ? 1 : SUBSET;
    u[16] = sub; u[17] = this.round % sub;
    u[204] = this.nearRefl ? 1 : 0;   // PP.refl.x (byte 816)
    u[206] = this.nearCones;          // PP.refl.z
    // PP.refl.w: the emitter part + 1, whose Gaussians send no reflected light into the probes (M47); 0 = off
    u[207] = this.emitterBounceFix && this.assetEmitters ? this.assetEmitters.part + 1 : 0;
    this.round++;
    this.device.queue.writeBuffer(this.buffers.probe_params, 0, this.params);
    // iso_xfer rebuilds a Gaussian's transfer only near a part that moved, or on a forced refresh: with neither, skip
    // the dispatch (2026-10-31: its early-out over every Gaussian still cost ~2.4 ms a frame on hill)
    const isoStale = this.forceNear || this.isoMoved((this.movedMask ?? 0) >>> 0);
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
    if (this.nearRefl) encoder.copyBufferToBuffer(this.buffers.lo_b, 0, this.buffers.lo_b_prev, 0, this.count * LOB * 16);
    const readBounce = () => {
      run('bounce_dirs', Math.ceil(this.count / u[16]));
      run('bounce_sh', u[19] ? this.count : Math.ceil(this.count / u[16]));
      if (this.nearRefl) run('near_refl', this.count);
    };
    if (lightStill) { readBounce(); return; }
    if (this.nearField) run('near_cones', this.count);
    if (isoStale) run('iso_xfer', this.count);
    const sc = this.senderCut;   // sender LOD: the cut's leaves and nodes only (bounce.wgsl sc_select)
    const runInd = (entry, offset) => {
      const { pipe, groups, timer = entry } = this.bounce[entry] ?? this.emit[entry];
      const pass = encoder.beginComputePass({ label: entry, timestampWrites: BOUNCE_TIMERS.includes(timer) ? timers.writes(timer) : undefined });
      pass.setPipeline(pipe); pass.setBindGroup(0, groups[0]); pass.dispatchWorkgroupsIndirect(this.scList, offset); pass.end();
    };
    if (sc) {
      // the cut depends only on where the parts (and so their probes) are: re-selected when one moved
      const reselect = this.scStale || (this.movedMask ?? 0) !== 0;
      this.device.queue.writeBuffer(this.scList, 36, new Uint32Array([reselect ? 1 : 0]));   // word 9: every node slot refreshes
      if (reselect) {
        encoder.clearBuffer(this.scList, 0, 32);   // counts and dispatch sizes; word 8 (node slots) stays
        encoder.clearBuffer(this.scList, 40, 8);   // tree slots handed out / not fitting (sc_mark)
        this.scLevels.forEach((l, i) => run('sc_select', l.count, i));
        run('sc_args', 1);
        runInd('sc_xfer', 20);
        this.checkSlots(encoder);
        this.scStale = false;
      }
      runInd('sc_shadow', 20);
      runInd('direct_dirs_emit_sc', 8);
      runInd('direct_dirs_nodes_sc', 20);
      this.env?.encodeDirsSC(encoder);
    } else {
      run('direct_dirs_emit', Math.ceil(this.count / 4));   // stage 2: 1/4 a frame
      this.env?.encodeDirs(encoder);   // distant lights toward the probes, for the same Gaussians (environment.js)
    }
    // senders (this frame's direct light + last frame's bounce) -> the probes gather -> each Gaussian reads its
    // bounce from this frame's probes (David, 2026-10-29: the first bounce settles in the frame things move; it used
    // to read last frame's probes, a one-frame dark trail behind a fast mover). Later bounces still lag a frame each.
    const oldOrder = new URLSearchParams(globalThis.location?.search ?? '').get('bounceorder') === 'old';   // debug: the pre-2026-10-29 order
    if (oldOrder) readBounce();
    if (sc) {
      runInd('node_leaf_list', 8);
      runInd('node_cut', 20);
      for (let d = this.maxDepth - 1; d >= 1; d--) run('node_up_sc', this.levelCounts[d], d);
      run('node_sh2_sc', this.nodeCount);
    } else {
      run('node_leaf', this.count);
      for (let d = this.maxDepth - 1; d >= 1; d--) run('node_up', this.levelCounts[d], d);
      run('node_sh2', this.nodeCount);
    }
    run('probe_gather', Math.ceil(this.probeCount * NCELL / u[16]));
    run('probe_valid', this.probeCount);
    run('probe_sh2', this.probeCount);
    if (!oldOrder) readBounce();
  }

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

  // Every slot empty: (v = 0, lamp 0xFFFF, count 0), SPEC §3.3.
  resetHistory() {
    const init = new Uint32Array(this.count * 8);
    for (let i = 1; i < init.length; i += 2) init[i] = EMPTY_SLOT;
    this.device.queue.writeBuffer(this.buffers.lamp_state, 0, init);
  }

  // Host-side lamp record (SPEC §2.5): the emitter light's motion descriptor for the shadow history (centre and last
  // centre); the analytic lamp it once described as a light is gone (David, 2026-10-29)
  setLamp(index, center, prevCenter) {
    // unified lights: one sphere record per light for ema_light's motion test, from the parts' transforms (each
    // frame, so the last centre is last frame's)
    if (this.unified) {
      if (this.partXf) this.device.queue.writeBuffer(this.buffers.lamps, 0, this.lightSet.records(this.partXf));
      return;
    }
    const f = new Float32Array(this.lampBytes);
    const o = index * 24;
    f.set(center, o);
    f.set(prevCenter, o + 20);
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

  // Per material for project.wgsl's RenderFrame.mat_view (32 slots: mat_view, then mat_view_ext): coverage peak (flags bit 5, pad0's high half),
  // Fresnel F0, specular weight k, and flags (1 coverage set, 2 emissive: no view reflection). Reflection coverage,
  // experiments/reflection_coverage.
  materialView(f0 = 0.04, f0Coverage = f0) {   // f0Coverage: glass (coverage materials) only, e.g. a stylized 0.2
    const u = new Uint32Array(this.materialBytes), out = new Float32Array(128);
    for (let m = 0; m < Math.min(this.materialCount, 32); m++) {
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
    u[12] = this.lightSamplesAxis; u[13] = this.materialCount;
    u[15] = 1 | 2 | (this.asgSpec ? 4 : 0) | (this.transmit ? 8 : 0)   // M39 transfer (bit 0), SH2 probes (bit 1), P3 ASG probe cells toward the eye (bit 2), M41 transmission (bit 3)
      | (this.specSplit ? 16 : 0) | (this.sheen ? 32 : 0);   // reflection coverage: specular toward the eye kept apart (bit 4), view-only probe sheen (bit 5)
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
  // Direct light: the shadows (one opacity map per light for unified lights, lights_map.js; the emitter light's map
  // otherwise, light_shadow.js), then shade_emit (the LOD cut's leaves with ?lodlight=2), then the distant lights.
  // lightStill (main.js): nothing the shadow maps depend on changed for a while, so they are not touched
  encode(encoder, eye, timers, changed = true, lightStill = false) {
    this.writeUniform(eye);
    if (this.blur && this.encodeBlur(encoder)) changed = true;
    const moved = (this.blur ? this.blurMask : this.movedMask) ?? 0;
    if (this.lampOff || (lightStill && !changed && moved === 0)) { /* no lamp, or nothing moved: the maps hold */ }
    else if (this.unified) this.lightsMap.encode(encoder, moved);
    else this.lightShadow?.encode(encoder, timers, moved, changed);
    const entry = this.lodCut ? 'shade_emit_list' : 'shade_emit';
    const e = this.emit[entry];
    const pass = encoder.beginComputePass({ label: entry, timestampWrites: timers.writes(e.timer) });
    pass.setPipeline(e.pipe);
    pass.setBindGroup(0, e.groups[0]);
    if (this.lodCut) pass.dispatchWorkgroupsIndirect(this.lodList, 16);   // lod_args: the leaf list
    else pass.dispatchWorkgroups(Math.ceil(this.count / WG));
    pass.end();
    this.env?.encodeDirect(encoder, timers, changed, lightStill && !changed && moved === 0);   // distant lights (sun) on top of shade_emit (environment.js)
  }
}
