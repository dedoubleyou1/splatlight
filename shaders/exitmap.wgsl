// Faked refraction E (experiments/refraction_fake/fill/README.md, "David's exit map"): refraction traced per glass
// skin Gaussian, not per pixel. `prep` poses each glass Gaussian of a lens part in view space (centre, outward normal =
// its short axis, sigmas); `trace` refracts the eye ray through each front-facing one, sweeps the bent ray through the
// same lens's skin as discs (ray meets the tangent plane within 2 sigma), averages the nearest sheet it touches, and
// repeats in and out; it stores the exit point, exit direction and transmission per Gaussian. splat.wgsl vs_exit /
// fs_exit splat them into a screen map that refract.wgsl reads.
// The sweep narrows its candidates with the ray grid (hgrid.wgsl, raygrid.js): `prep` writes each skin disc into the
// grid's slots as posed this frame (no cached data, so LOD cuts and deforming glass work the same), the grid is built
// once for every consumer of the frame, and each level of the hashed grid is walked along the bent
// ray with a 3D DDA until a cell starts past the sheet found so far. A disc sits in every cell its reach touches, so a
// hit only counts in the cell that holds the hit point. (experiments/ray_intersect: brute force tested every disc of
// the lens, the asset tree walk was slower still, a cached skin-only grid was the baseline.)

@group(0) @binding(0) var<uniform> frame: RenderFrame;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> parts: array<Part>;
@group(0) @binding(3) var<uniform> LENSES: Lenses;
@group(0) @binding(4) var<storage, read> glass: array<u32>;           // [0] count, [1 + 2k] / [2 + 2k] lens k start / count (k < 8), [17 + k] lens k part, [25] widest sigma_t (f32), [26] ray grid base slot, [27 + s] = Gaussian index | lens << 24
@group(0) @binding(5) var<storage, read_write> skin: array<vec4f>;    // per Gaussian index: (centre_v, sigma_t), (normal_v outward, sigma_n); 0 when not glass
@group(0) @binding(6) var<storage, read_write> exits: array<vec4f>;   // per Gaussian index: (exit point_v, weight), (exit dir_v, transmission)

// G (gather): the inside ray per skin Gaussian, and the Gaussians inside each lens posed for the per-pixel gather
@group(0) @binding(7) var<storage, read_write> inray: array<vec4f>;   // per Gaussian index: (entry point - lens centre, leg 0 length), (leg 0 dir, 1), (leg 1 dir after a TIR bounce, leg 1 length)
@group(0) @binding(8) var<storage, read> inner_list: array<u32>;      // [0] count, [1 + 2k] / [2 + 2k] lens k start / count, [17] ray grid base slot, [18 + e] = Gaussian index | lens << 24
@group(0) @binding(9) var<storage, read_write> inner: array<vec4f>;   // per entry: (centre_v, kappa), (P xx xy xz, Gaussian index bits), (P yy yz zz, lens bits); P = view-space inverse covariance

const INNER_HEAD: u32 = 18u;

// the inside Gaussians of every lens, posed in view space this frame and filed in the ray grid (refract.wgsl
// gather_inside walks it per pixel)
@compute @workgroup_size(64)
fn prep_inner(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x;
  if (s >= inner_list[0]) { return; }
  let e = inner_list[INNER_HEAD + s];
  let gi = e & 0xffffffu;
  let g = gaussians[gi];
  let p = parts[min(g.part_surface & 0xffffu, MAX_PARTS - 1u)];
  let c = (frame.view * vec4f(part_apply(p, g.mu), 1.0)).xyz;
  let wr = mat3_of(frame.view) * (part_rotation(p) * quat_to_mat(g.quat));
  let is = 1.0 / max(g.scale * g.scale, vec3f(1e-12));
  let P = wr * mat3x3f(vec3f(is.x, 0.0, 0.0), vec3f(0.0, is.y, 0.0), vec3f(0.0, 0.0, is.z)) * transpose(wr);
  inner[3u * s] = vec4f(c, g.kappa);
  inner[3u * s + 1u] = vec4f(P[0][0], P[0][1], P[0][2], bitcast<f32>(gi));
  inner[3u * s + 2u] = vec4f(P[1][1], P[1][2], P[2][2], bitcast<f32>(e >> 24u));
  // its ray grid item: the gather counts a Gaussian where the ray passes within 4 sigma (q < 16) of it
  hg_items[inner_list[17] + s] = vec4f(c, 4.0 * max(max(g.scale.x, g.scale.y), g.scale.z));
  hg_keys[inner_list[17] + s] = hg_key(HG_TAG_INSIDE, s | ((e >> 24u) << 20u));   // entry (< 2^20) | lens (< 8) << 20
}

const GLASS_HEAD: u32 = 27u;
const MAX_EVENTS: u32 = 4u;
const HG_STEPS: u32 = 256u;    // cells per level per ray at most (a bound, not a tuning knob)

@compute @workgroup_size(64)
fn prep(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x;
  if (s >= glass[0]) { return; }
  let e = glass[GLASS_HEAD + s];
  let gi = e & 0xffffffu;
  let k = e >> 24u;
  let g = gaussians[gi];
  let p = parts[min(g.part_surface & 0xffffu, MAX_PARTS - 1u)];
  let c = (frame.view * vec4f(part_apply(p, g.mu), 1.0)).xyz;
  var n = normalize(mat3_of(frame.view) * (part_rotation(p) * quat_to_mat(g.quat))[0]);   // scales ascending: x is the short axis
  if (dot(n, c - LENSES.lens[k].c.xyz) < 0.0) { n = -n; }
  skin[2u * gi] = vec4f(c, sqrt(g.scale.y * g.scale.z));
  skin[2u * gi + 1u] = vec4f(n, g.scale.x);
  // its ray grid item: a hit lands within 2 sqrt(sigma_t^2 + beam^2) of the centre, and the widest beam is half the
  // widest sigma_t
  let st = sqrt(g.scale.y * g.scale.z);
  let beam = 0.5 * bitcast<f32>(glass[25]);
  hg_items[glass[26] + s] = vec4f(c, 2.0 * sqrt(st * st + beam * beam));
  hg_keys[glass[26] + s] = hg_key(HG_TAG_GLASS, gi | (k << 20u));   // Gaussian (< 2^20) | lens (< 8) << 20
}

fn schlick_e(cosi: f32, ior: f32) -> f32 {
  let f0 = (ior - 1.0) * (ior - 1.0) / ((ior + 1.0) * (ior + 1.0));
  return f0 + (1.0 - f0) * pow(1.0 - clamp(cosi, 0.0, 1.0), 5.0);
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x;
  if (s >= glass[0]) { return; }
  let e = glass[GLASS_HEAD + s];
  let gi = e & 0xffffffu;
  let k = e >> 24u;
  let a = skin[2u * gi];
  let nf = skin[2u * gi + 1u].xyz;
  let v = normalize(a.xyz);
  if (dot(v, nf) >= 0.0) { exits[2u * gi] = vec4f(0.0); exits[2u * gi + 1u] = vec4f(0.0); if ((LENSES.info.z & REFR_GATHER) != 0u) { inray[3u * gi] = vec4f(0.0); } return; }   // back-facing: not on the map
  let ior = LENSES.params.x;
  var thr = 1.0 - schlick_e(-dot(v, nf), ior);
  var t = refract(v, nf, 1.0 / ior);
  var o = a.xyz;
  var inside = true;
  let beam2 = 0.25 * a.w * a.w;
  let gap = 2.0 * a.w;
  let mask = hg_mask();
  // G: the inside legs (entry leg, and the leg after a TIR bounce off the first wall it meets)
  let t0 = t;
  var leg0 = 0.0; var t1 = vec3f(0.0); var leg1 = 0.0;
  for (var ev = 0u; ev < MAX_EVENTS; ev++) {
    // nearest sheet of touched discs facing the right way, Gaussian-weighted (online: restart when a nearer one appears)
    var best = 1e30; var ws = 0.0; var ss = 0.0; var nsum = vec3f(0.0);
    for (var L = 0u; L < HG_LEVELS; L++) {
      if ((mask & (1u << L)) == 0u) { continue; }
      var w = hg_begin(L, o, t);
      for (var it = 0u; it < HG_STEPS && w.se <= w.tout; it++) {
        if (w.se > best + gap) { break; }                           // every later hit on this level is past the sheet
        let r = hg_range(L, w);
        for (var q = r.x; q < r.y; q++) {
          let en = hg_refs[q];
          // a collision from another level, another system's item, or another lens
          if ((en >> 27u) != L || ((en >> HG_KEY_BITS) & 0xfu) != HG_TAG_GLASS || ((en >> 20u) & 7u) != k) { continue; }
          let j = en & 0xfffffu;
          let nj = skin[2u * j + 1u];
          let tn = dot(t, nj.xyz);
          if (select(tn > -0.05, tn < 0.05, inside)) { continue; }
          let bj = skin[2u * j];
          let sj = dot(bj.xyz - o, nj.xyz) / tn;
          if (sj <= 4.0 * nj.w + 1e-3) { continue; }
          let x = o + t * sj;
          if (!hg_owns(L, w.c, x)) { continue; }                     // another cell's copy (or a collision)
          let d = x - bj.xyz;
          let lat2 = dot(d, d) / (bj.w * bj.w + beam2);
          if (lat2 >= 4.0) { continue; }
          if (sj < best - gap) { best = sj; ws = 0.0; ss = 0.0; nsum = vec3f(0.0); }
          else if (sj > best + gap) { continue; }
          best = min(best, sj);
          let wt = exp(-0.5 * lat2);
          ws += wt; ss += wt * sj; nsum += wt * nj.xyz;
        }
        if (!hg_next(&w)) { break; }
      }
    }
    if (ws <= 0.0) { break; }
    let sh = ss / ws;
    let n = normalize(nsum);
    o = o + t * sh;
    if (ev == 0u) { leg0 = sh; }
    if (ev == 1u && inside) { leg1 = sh; }
    if (inside) {
      thr *= exp(-LENSES.params.z * sh);
      let c2 = abs(dot(t, n));
      let to = refract(t, -n, ior);
      if (dot(to, to) < 0.5) { t = reflect(t, -n); if (ev == 0u) { t1 = t; } }        // TIR: stay inside
      else { thr *= 1.0 - schlick_e(c2, ior); t = to; inside = false; }
    } else {
      thr *= 1.0 - schlick_e(abs(dot(t, n)), ior);
      t = refract(t, n, 1.0 / ior);
      inside = true;
    }
  }
  // miss confidence (refract.wgsl exit_look): the exit ray turning back toward the eye, still trapped inside, or
  // leaving the screen within 0.3 m; w > 0 keeps it on the map
  let pe = o + t * 0.3;
  let ze = max(-pe.z, 1e-3);
  let px = vec2f(0.5 * frame.viewport.x + frame.focal.x * pe.x / ze, 0.5 * frame.viewport.y - frame.focal.y * pe.y / ze);
  let edge = min(min(px.x, frame.viewport.x - px.x), min(px.y, frame.viewport.y - px.y));
  var conf = smoothstep(0.0, 0.3, -t.z) * clamp(edge / (0.04 * frame.viewport.y), 0.0, 1.0);
  if (inside) { conf *= 0.25; }
  if (pe.z > -1e-3) { conf = 0.0; }
  exits[2u * gi] = vec4f(o, max(conf, 1e-4));
  exits[2u * gi + 1u] = vec4f(t, thr);
  if ((LENSES.info.z & REFR_GATHER) != 0u) {   // G only: inray is a placeholder until G's first frame (renderer.js ensureExit)
    inray[3u * gi] = vec4f(a.xyz - LENSES.lens[k].c.xyz, leg0);
    inray[3u * gi + 1u] = vec4f(t0, 1.0);
    inray[3u * gi + 2u] = vec4f(t1, select(0.0, leg1, dot(t1, t1) > 0.5));
  }
}
