// Unified lights (?lights=unified): every emissive Gaussian in the scene is a light, and no light is special.
//
// A light is one object (asset part) with emissive Gaussians (material flags bit 0); the ceiling lamp is just one of
// them. Each light keeps its own emitter subtree in its part's rest frame, built once; when a light moves (its part's
// xf) or is switched or re-weighted, its subtree is carried to world space and a small top tree is merged over the
// lights' roots. Every receiver takes one light cut of the whole tree (emissive.wgsl), so the cost follows the cut,
// not the light count: far candles merge into one node.
//
// GPU records (emissive.wgsl EmitNode, 96 B): the 16-float node (mean, radius, M diag, tag, M off, child_lo, Le,
// child_hi), then its ASG covariance (C_s diag, nrm, C_s off, part).
//   tag   bit 1: one-sided, emitting toward nrm (an octahedral pack2x16snorm); bits 16-31: the light id (0xFFFF: a node
//         spanning several lights)
//   part  leaves: the emitter's part (its shadow rays skip it); else 0xFFFFFFFF
// Light records (lamps.bin's Lamp layout, LAMP_SPHERE, index = light id): world centre, radius and last centre of the
// light's root, for the shadow history's motion test (ema.wgsl ema_light).

const EIN3 = Array.from({ length: 3000 }, (_, i) => { const x = (i + 0.5) * 0.001; return (1 - Math.exp(-x)) / x * 0.001; })
  .reduce((acc, v) => acc + v, 0);
const ASG_WIDEN = EIN3 / (1 - Math.exp(-3));
const NO_CHILD = 0xffffffff, MIXED = 0xffff, TAG_ONE_SIDED = 2;
export const MAX_LIGHTS = 256;
// light LOD: a light-tree node smaller than this is used as a leaf; lights need not be finer than that (?lightlod=)
export const LIGHT_MIN_RADIUS = 0.015;
const IX = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]];   // symmetric [xx, yy, zz, xy, xz, yz]

const symDet = ([a, d, f, b, c, e]) => a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
function symInv(m) {
  const [a, d, f, b, c, e] = m, det = symDet(m);
  return [(d * f - e * e) / det, (a * f - c * c) / det, (a * d - b * b) / det,
    (c * e - b * f) / det, (b * e - c * d) / det, (b * c - a * e) / det];
}
const symQuad = ([a, d, f, b, c, e], [x, y, z]) => a * x * x + d * y * y + f * z * z + 2 * (b * x * y + c * x * z + e * y * z);
const symFull = s => [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]];
function symAxes(m) {   // eigenvectors by Jacobi rotations
  const A = symFull(m), V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
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
const halfToFloat = h => { const e = (h >> 10) & 31, m = h & 1023, s = h & 0x8000 ? -1 : 1;
  return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15); };
const quatRot = ([x, y, z, w]) => [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
const mul3 = (R, v) => R.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
function rotSym(R, s) {   // R S R^T
  const S = symFull(s), RS = R.map(r => [0, 1, 2].map(k => r[0] * S[0][k] + r[1] * S[1][k] + r[2] * S[2][k]));
  const o = (i, j) => RS[i][0] * R[j][0] + RS[i][1] * R[j][1] + RS[i][2] * R[j][2];
  return [o(0, 0), o(1, 1), o(2, 2), o(0, 1), o(0, 2), o(1, 2)];
}
const norm3 = v => { const l = Math.hypot(...v) || 1; return v.map(x => x / l); };
function packOct(n) {   // octahedral unit vector -> pack2x16snorm
  const s = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]);
  let x = n[0] / s, y = n[1] / s;
  if (n[2] < 0) { const ox = x; x = (1 - Math.abs(y)) * Math.sign(ox || 1); y = (1 - Math.abs(ox)) * Math.sign(y || 1); }
  const q = v => (Math.round(Math.max(-1, Math.min(1, v)) * 32767) & 0xffff);
  return (q(x) | (q(y) << 16)) >>> 0;
}

// One emitter Gaussian (a 16-float record: mu, kappa, quat, scale, -, Le) as a tree leaf: radiant intensity
// Le sqrt(w^T M w) with M = (2 pi Ein(3))^2 det C C^-1, its footprint at optical depth 3.
function leaf(rec, o, light, part, oneSided) {
  const R = quatRot([rec[o + 4], rec[o + 5], rec[o + 6], rec[o + 7]]);
  const s2 = [rec[o + 8] ** 2, rec[o + 9] ** 2, rec[o + 10] ** 2];
  const C = IX.map(([r, k]) => R[r][0] * R[k][0] * s2[0] + R[r][1] * R[k][1] * s2[1] + R[r][2] * R[k][2] * s2[2]);
  const det = symDet(C), g = (2 * Math.PI * EIN3) ** 2 * det;
  return { mu: [rec[o], rec[o + 1], rec[o + 2]], C, M: symInv(C).map(v => v * g), Cs: C.map(v => v * ASG_WIDEN),
    mass: rec[o + 3] * (2 * Math.PI) ** 1.5 * Math.sqrt(det), Le: [rec[o + 12], rec[o + 13], rec[o + 14]],
    radius: 3 * Math.max(rec[o + 8], rec[o + 9], rec[o + 10]), light, part,
    nrm: oneSided ? [R[0][0], R[1][0], R[2][0]] : null, kids: null };
}

// A parent of two nodes: moment-matched covariance; M diagonal in that frame, fitted so the intensity along its three
// axes equals the children's sum; Le their intensity-weighted mean. One-sided only when both children emit the same way.
function merge(a, b) {
  const kids = [a, b], m = a.mass + b.mass;
  const mu = [0, 1, 2].map(k => (a.mass * a.mu[k] + b.mass * b.mu[k]) / m);
  const mm = key => IX.map(([r, k], j) => kids.reduce((acc, ch) => acc + ch.mass * (ch[key][j] + (ch.mu[r] - mu[r]) * (ch.mu[k] - mu[k])), 0) / m);
  const C = mm('C'), Cs = mm('Cs'), axes = symAxes(C);
  const along = (ch, w) => Math.sqrt(Math.max(symQuad(ch.M, w), 0));
  const tot = axes.map(w => kids.reduce((acc, ch) => acc + lum(ch.Le) * along(ch, w), 0));
  const wk = kids.map(ch => axes.reduce((s, w) => s + along(ch, w), 0));
  const Le = [0, 1, 2].map(c => (a.Le[c] * wk[0] + b.Le[c] * wk[1]) / Math.max(wk[0] + wk[1], 1e-30));
  const L = Math.max(lum(Le), 1e-12), M = [0, 0, 0, 0, 0, 0];
  axes.forEach((w, k) => { const lam = (tot[k] / L) ** 2; IX.forEach(([r, c], j) => { M[j] += lam * w[r] * w[c]; }); });
  const radius = Math.max(...kids.map(ch => Math.hypot(...ch.mu.map((v, k) => v - mu[k])) + ch.radius));
  const same = a.nrm && b.nrm && a.nrm[0] * b.nrm[0] + a.nrm[1] * b.nrm[1] + a.nrm[2] * b.nrm[2] > 0.95;
  return { mu, C, Cs, M, mass: m, Le, radius, light: a.light === b.light ? a.light : MIXED, part: NO_CHILD,
    nrm: same ? norm3(a.nrm.map((v, k) => v + b.nrm[k])) : null, kids };
}

// Median split on the longest axis of the items' centres, down to single items.
function buildTree(items) {
  if (items.length === 1) return items[0];
  // loops, not Math.min(...array): spreading a large array overflows the call stack (2026-10-31)
  const lo = [0, 1, 2].map(k => items.reduce((m, it) => Math.min(m, it.mu[k]), Infinity));
  const hi = [0, 1, 2].map(k => items.reduce((m, it) => Math.max(m, it.mu[k]), -Infinity));
  const ax = [0, 1, 2].reduce((m, k) => (hi[k] - lo[k] > hi[m] - lo[m] ? k : m), 0);
  const sorted = [...items].sort((p, q) => p.mu[ax] - q.mu[ax]), half = sorted.length >> 1;
  return merge(buildTree(sorted.slice(0, half)), buildTree(sorted.slice(half)));
}

// Light LOD: a node with radius under rmin becomes a leaf of light part.
function collapse(nd, rmin, part) {
  if (!nd.kids) return;
  if (nd.radius < rmin) { nd.kids = null; nd.part = part; return; }
  nd.kids.forEach(ch => collapse(ch, rmin, part));
}

// A subtree carried by a rigid transform (R, t): covariances and M rotate, means and normals move.
function transformTree(nd, R, t) {
  const out = { ...nd, mu: mul3(R, nd.mu).map((v, k) => v + t[k]), C: rotSym(R, nd.C), Cs: rotSym(R, nd.Cs), M: rotSym(R, nd.M),
    nrm: nd.nrm && mul3(R, nd.nrm) };
  if (nd.kids) out.kids = nd.kids.map(ch => transformTree(ch, R, t));
  return out;
}

// One object's emitters summed as the cut sums them overcount: inner Gaussians hide behind outer ones and a thin
// flame is not at optical depth 3. The correction: the object's radiant intensity from a front-to-back composite of
// its emitters (each a thin layer at its centre, its real optical depth) on a 96^2 grid along +-x, +-y, +-z, over what
// the leaves would sum along the same axes, the axes weighted by their true intensity (a panel's edge-on views, where
// the summed leaves overcount most but little light goes, then barely count). Luminance only; one number per light.
export function lightScale(rec, n) {
  const RES = 96, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], g = [];
  for (let i = 0; i < n; i++) {
    const nd = leaf(rec, i * 16, 0, 0, false), r = 3 * Math.sqrt(Math.max(nd.C[0], nd.C[1], nd.C[2]));
    for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], nd.mu[k] - r); hi[k] = Math.max(hi[k], nd.mu[k] + r); }
    g.push({ ...nd, Ci: symInv(nd.C), kappa: rec[i * 16 + 3], L: lum(nd.Le) });
  }
  const S = [[0, 3, 4], [3, 1, 5], [4, 5, 2]];
  let num = 0, den = 0;
  for (let a = 0; a < 3; a++) {
    const [b, c] = [0, 1, 2].filter(k => k !== a), h = Math.max(hi[b] - lo[b], hi[c] - lo[c]) / RES;
    for (const sign of [1, -1]) {
      const T = new Float32Array(RES * RES).fill(1), L = new Float32Array(RES * RES);
      let model = 0;
      for (const e of [...g].sort((p, q) => sign * (q.mu[a] - p.mu[a]))) {   // nearest to the viewer first
        model += e.L * Math.sqrt(Math.max(e.M[S[a][a]], 0));
        const tau0 = e.kappa * Math.sqrt(2 * Math.PI / Math.max(e.Ci[S[a][a]], 1e-30));
        const sbb = e.C[S[b][b]], scc = e.C[S[c][c]], sbc = e.C[S[b][c]], d2 = sbb * scc - sbc * sbc;
        const ib = scc / d2, ic = sbb / d2, ibc = -sbc / d2, rad = 3 * Math.sqrt(Math.max(sbb, scc));
        const x0 = Math.max(0, Math.floor((e.mu[b] - rad - lo[b]) / h)), x1 = Math.min(RES - 1, Math.ceil((e.mu[b] + rad - lo[b]) / h));
        const y0 = Math.max(0, Math.floor((e.mu[c] - rad - lo[c]) / h)), y1 = Math.min(RES - 1, Math.ceil((e.mu[c] + rad - lo[c]) / h));
        for (let y = y0; y <= y1; y++) {
          const dy = lo[c] + (y + 0.5) * h - e.mu[c];
          for (let x = x0; x <= x1; x++) {
            const dx = lo[b] + (x + 0.5) * h - e.mu[b];
            const t = Math.exp(-tau0 * Math.exp(-0.5 * (ib * dx * dx + 2 * ibc * dx * dy + ic * dy * dy))), j = y * RES + x;
            L[j] += T[j] * e.L * (1 - t);
            T[j] *= t;
          }
        }
      }
      const truth = L.reduce((acc, v) => acc + v, 0) * h * h;
      num += truth * truth; den += truth * model;
    }
  }
  return den > 0 ? num / den : 1;
}

export class Lights {
  // asset: gs-runtime asset (header, raw, gaussians). Lights are found from the asset's own materials; the per-light
  // switch and gain start from asset.json parts[].casts_light (default true) and parts[].light_gain (default 1).
  // minRadius: the light LOD cap (0 = off).
  constructor(asset, minRadius = LIGHT_MIN_RADIUS) {
    this.minRadius = minRadius;
    const gu = new Uint32Array(asset.gaussians), gf = new Float32Array(asset.gaussians);
    const mu = new Uint32Array(asset.raw.materials), nm = asset.header.buffers.materials.count;
    this.lights = [];
    asset.header.parts.forEach((pt, p) => {
      const idx = [];
      for (let g = pt.gaussians[0]; g < pt.gaussians[1]; g++) {
        const m = gu[g * 16 + 15] & 0xffff;
        if (m < nm && (mu[m * 8 + 5] & 1)) idx.push(g);
      }
      if (!idx.length || this.lights.length >= MAX_LIGHTS) return;
      const rec = new Float32Array(idx.length * 16), mats = new Uint32Array(idx.length);
      idx.forEach((g, i) => {
        rec.set(gf.subarray(g * 16, g * 16 + 16), i * 16);
        const m = mats[i] = gu[g * 16 + 15] & 0xffff;
        rec.set([halfToFloat(mu[m * 8 + 6] & 0xffff), halfToFloat(mu[m * 8 + 6] >>> 16), halfToFloat(mu[m * 8 + 7] & 0xffff)], i * 16 + 12);
      });
      this.lights.push({ id: this.lights.length, part: p, name: pt.name, count: idx.length, rec, mats,
        scale: lightScale(rec, idx.length), on: pt.casts_light ?? true, gain: pt.light_gain ?? 1, prev: null });
    });
    this.key = null;
  }

  // Rest-frame subtree of a light at its current gain and one-sided flags (materialFlags: the live materials.bin u32
  // view), rebuilt only when those change.
  restTree(L, materialFlags) {
    const key = [L.gain, ...new Set(Array.from(L.mats, m => materialFlags[m * 8 + 5] & 4))].join();
    if (L.treeKey === key) return L.tree;
    const k = L.scale * L.gain, rec = new Float32Array(L.rec);
    const items = [];
    for (let i = 0; i < L.count; i++) {
      for (let c = 0; c < 3; c++) rec[i * 16 + 12 + c] *= k;
      items.push(leaf(rec, i * 16, L.id, L.part, (materialFlags[L.mats[i] * 8 + 5] & 4) !== 0));
    }
    L.tree = buildTree(items);
    if (this.minRadius > 0) collapse(L.tree, this.minRadius, L.part);
    L.treeKey = key;
    return L.tree;
  }

  // World tree over the lights that are on, from the parts' current transforms (partXf: parts.bin floats, xf = 3 rows
  // of 4 at 32 floats per part). Returns null when nothing changed since the last call.
  update(partXf, materialFlags) {
    const on = this.lights.filter(L => L.on && L.gain > 0);
    const xf = L => Array.from(partXf.subarray(L.part * 32, L.part * 32 + 12));
    const key = JSON.stringify(on.map(L => [L.id, L.gain, xf(L)])) + Array.from(materialFlags).join();
    if (key === this.key) return null;
    this.key = key;
    const roots = on.map(L => {
      const m = xf(L), R = [m.slice(0, 3), m.slice(4, 7), m.slice(8, 11)];
      return transformTree(this.restTree(L, materialFlags), R, [m[3], m[7], m[11]]);
    });
    return roots.length ? buildTree(roots) : null;
  }

  // Serialize a world tree into EmitNode records (24 floats each, packEmitNodes' layout); the root is last.
  static pack(root) {
    const list = [];
    const visit = nd => {   // post-order: children before parents
      if (nd.kids) { nd.kids.forEach(visit); nd.lo = nd.kids[0].id; nd.hi = nd.kids[1].id; }
      nd.id = list.length;
      list.push(nd);
    };
    visit(root);
    const f = new Float32Array(list.length * 24), u = new Uint32Array(f.buffer);
    list.forEach((nd, i) => {
      const o = i * 24;
      f.set([...nd.mu, nd.radius, nd.M[0], nd.M[1], nd.M[2]], o);
      u[o + 7] = (nd.nrm ? TAG_ONE_SIDED : 0) | (nd.light << 16);
      f.set([nd.M[3], nd.M[4], nd.M[5]], o + 8);
      u[o + 11] = nd.kids ? nd.lo : NO_CHILD;
      f.set(nd.Le, o + 12);
      u[o + 15] = nd.kids ? nd.hi : i;
      f.set([nd.Cs[0], nd.Cs[1], nd.Cs[2]], o + 16);
      u[o + 19] = nd.nrm ? packOct(nd.nrm) : 0;
      f.set([nd.Cs[3], nd.Cs[4], nd.Cs[5]], o + 20);
      u[o + 23] = nd.kids ? NO_CHILD : nd.part;
    });
    return { nodes: f, count: list.length, root: list.length - 1, leaves: list.filter(nd => !nd.kids) };
  }

  // Light records for ema_light's motion test (Lamp layout, 24 floats; kind 1 = sphere), index = light id. A light
  // that is off keeps flags 0 and strength 0.
  records(partXf) {
    const out = new Float32Array(Math.max(this.lights.length, 1) * 24), u = new Uint32Array(out.buffer);
    for (const L of this.lights) {
      const o = L.id * 24, m = partXf.subarray(L.part * 32, L.part * 32 + 12);
      const t = L.tree ?? buildTree([leaf(L.rec, 0, L.id, L.part, false)]);
      const c = [0, 1, 2].map(r => m[r * 4] * t.mu[0] + m[r * 4 + 1] * t.mu[1] + m[r * 4 + 2] * t.mu[2] + m[r * 4 + 3]);
      out.set(c, o); u[o + 3] = 1;
      out[o + 7] = t.radius;
      // pad0 (as f32): the light's mean radiant intensity (luminance), so its strength at a receiver is this / d^2; ema_light
      // keys the receiver's shadow slots on it
      out[o + 19] = L.on && L.tree ? lum(t.Le) * Math.sqrt(Math.max((t.M[0] + t.M[1] + t.M[2]) / 3, 0)) : 0;
      u[o + 15] = L.on ? 1 : 0;
      out.set(L.prev ?? c, o + 20);
      L.prev = c;
    }
    return out;
  }
}
