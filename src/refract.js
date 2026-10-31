// Faked refraction (?refract=1; experiments/refraction_fake/README.md, method A): each glass part's own Gaussians fit
// one ellipsoid lens (centre and principal axes of their centres, scaled to enclose them, so a solid fill and a skin
// both give the solid). The splat pass sorts every fragment into three "over" layers against the lenses (in front,
// inside, behind: splat.wgsl fs_layers) and refract.wgsl bends each lens pixel's ray through the ellipsoid with Snell
// at both interfaces, reading the inside and behind layers by a depth fixed point.

export const MAX_LENSES = 8;
export const LENS_BYTES = 48 + MAX_LENSES * 80;   // lens.wgsl `Lenses`
export const MODES = { A: 1, B: 2, C: 4, AC: 5, BC: 6, E: 9, EC: 13, EG: 25 };   // lens.wgsl REFR_* bits; ?refract=A|B|C|AC|BC|E|EC (1 = A); E = A's layers, bent by the exit map

// Symmetric 3x3 eigen-decomposition (cyclic Jacobi); returns { values, vectors (columns) }.
function eig3(a) {
  const A = a.map(r => r.slice()), V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 32; sweep++) {
    const off = Math.abs(A[0][1]) + Math.abs(A[0][2]) + Math.abs(A[1][2]);
    if (off < 1e-14) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(A[p][q]) < 1e-18) continue;
      const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
      const c = Math.cos(th), s = Math.sin(th);
      for (let k = 0; k < 3; k++) {   // A <- A J
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {   // A <- J^T A
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  return { values: [A[0][0], A[1][1], A[2][2]], vectors: V };
}

// Lenses in each part's rest frame: { part, c, M (rows; u = M (x - c) is the unit sphere), radius (bounding) }.
// Glass = Gaussians whose material (flags low 16 bits) sets a reflection coverage (asset.json materials[].coverage).
export function fitLenses(asset) {
  const { header } = asset;
  const glass = new Set((header.materials ?? []).filter(m => m.coverage !== undefined).map(m => m.index));
  if (!glass.size) return [];
  const f = new Float32Array(asset.gaussians), u = new Uint32Array(asset.gaussians);
  const lenses = [];
  header.parts.forEach((p, part) => {
    const [lo, hi] = p.gaussians;
    const pts = [];
    for (let i = lo; i < hi; i++) if (glass.has(u[i * 16 + 15] & 0xffff)) pts.push([f[i * 16], f[i * 16 + 1], f[i * 16 + 2]]);
    if (pts.length < 16 || lenses.length >= MAX_LENSES) return;
    const c = [0, 1, 2].map(k => pts.reduce((s, x) => s + x[k], 0) / pts.length);
    const S = [0, 1, 2].map(r => [0, 1, 2].map(k => pts.reduce((s, x) => s + (x[r] - c[r]) * (x[k] - c[k]), 0) / pts.length));
    const { values, vectors } = eig3(S);
    // whitened radius of every centre; the ellipsoid is scaled to the largest (glass skins sit on the surface)
    const w = pts.map(x => Math.hypot(...[0, 1, 2].map(j => (vectors[0][j] * (x[0] - c[0]) + vectors[1][j] * (x[1] - c[1]) + vectors[2][j] * (x[2] - c[2])) / Math.sqrt(Math.max(values[j], 1e-12)))));
    const k = Math.max(...w);
    const axes = values.map(v => k * Math.sqrt(Math.max(v, 1e-12)));
    const M = [0, 1, 2].map(j => [0, 1, 2].map(r => vectors[r][j] / axes[j]));
    lenses.push({ part, name: p.name, c, M, radius: Math.max(...axes) });
  });
  return lenses;
}

// Per frame: each lens in view space (centre c_v, N = M R^T V^T so u = N (x_v - c_v)) and its pixel rect.
// partXf: the part records (Float32Array, 32 floats per part, rows [R | t] first); view: column-major 4x4.
export function writeLenses(bytes, lenses, partXf, view, viewport, focal, opts) {
  const f32 = new Float32Array(bytes), u32 = new Uint32Array(bytes);
  f32.fill(0);
  u32[0] = lenses.length; u32[1] = opts.dispersion > 0 ? 1 : 0; u32[2] = opts.mode; u32[3] = opts.compact ? 1 : 0;
  f32.set([opts.ior, opts.dispersion, opts.absorb, opts.frontMargin], 4);
  // E miss fallback (refract.wgsl through_exit): probe radiance gain (?refrprobe=, 0 = off), confidence view (?refrdbg=1; 2 the probe radiance), per-pixel confidence only (?refrconf=px)
  const q = new URLSearchParams(globalThis.location?.search ?? '');
  f32.set([opts.bend, Number(q.get('refrprobe') ?? 1), Number(q.get('refrdbg') ?? 0), q.get('refrconf') === 'px' ? 1 : 0], 8);
  const V = (r, c) => view[c * 4 + r];
  lenses.forEach((L, i) => {
    const b = 12 + i * 20;   // floats: header 12, then 20 per lens (80 B)
    const X = (r, c) => partXf[L.part * 32 + r * 4 + c];
    const cw = [0, 1, 2].map(r => X(r, 0) * L.c[0] + X(r, 1) * L.c[1] + X(r, 2) * L.c[2] + X(r, 3));
    const cv = [0, 1, 2].map(r => V(r, 0) * cw[0] + V(r, 1) * cw[1] + V(r, 2) * cw[2] + V(r, 3));
    // N = M R^T Vr^T: N[a][b] = sum_j M[a][j] (Vr R)[b][j]
    const VR = [0, 1, 2].map(r => [0, 1, 2].map(j => V(r, 0) * X(0, j) + V(r, 1) * X(1, j) + V(r, 2) * X(2, j)));
    const N = [0, 1, 2].map(a => [0, 1, 2].map(bb => [0, 1, 2].reduce((s, j) => s + L.M[a][j] * VR[bb][j], 0)));
    f32.set([...cv, L.radius], b);
    N.forEach((row, a) => f32.set([...row, 0], b + 4 + 4 * a));
    // conservative pixel rect of the bounding sphere (whole screen when the camera is close)
    const z = -cv[2], r = L.radius;
    let rect = [0, 0, viewport[0], viewport[1]];
    if (z > 1.05 * r) {
      const s = r / (z - r);   // tangent half-angle bound, widened off axis below
      const px = 0.5 * viewport[0] + focal[0] * cv[0] / z, py = 0.5 * viewport[1] - focal[1] * cv[1] / z;
      const rx = focal[0] * s * (1 + (cv[0] / z) ** 2) + 2, ry = focal[1] * s * (1 + (cv[1] / z) ** 2) + 2;
      rect = [px - rx, py - ry, px + rx, py + ry];
    } else if (z < -r) rect = [1, 1, 0, 0];   // behind the camera: empty
    f32.set(rect, b + 16);
  });
}
