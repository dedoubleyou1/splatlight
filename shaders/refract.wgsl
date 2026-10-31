// Faked refraction composite (?refract=1; experiments/refraction_fake, method A). Runs after splat's fs_layers and
// writes the hdr target display reads. Lens pixels: the eye ray refracts into the nearest lens (Snell, Schlick), meets
// the inside layer at its depth (3-step fixed point along the inside ray), leaves through the far side (one internal
// bounce on TIR) and meets the behind layer the same way. The glass skin's own reflection rides in the front layer.
// Everything else is front over inside over behind, i.e. today's image.

@group(0) @binding(0) var<uniform> frame: RenderFrame;
@group(0) @binding(1) var<uniform> LENSES: Lenses;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var t_front: texture_2d<f32>;
@group(0) @binding(4) var t_inside: texture_2d<f32>;
@group(0) @binding(5) var t_behind: texture_2d<f32>;
@group(0) @binding(6) var z_inside: texture_2d<f32>;
@group(0) @binding(7) var z_behind: texture_2d<f32>;
@group(0) @binding(8) var lensmap: texture_2d<f32>;   // fs_lensmap's output
@group(0) @binding(9) var exit_dir: texture_2d<f32>;   // E: exit map (splat.wgsl fs_exit): direction * w, w
@group(0) @binding(10) var exit_pnt: texture_2d<f32>;  //    exit point * w, transmission * w
// E miss fallback: the lighting's probe lattice (bounce.wgsl ProbeParams, its prefix through ground) and SH2 radiance
struct RefrProbes {
  lo: vec4f, step: vec4f,
  pad: array<vec4f, 50>,
  dims: vec4u,                  // x, y, z counts (0: no probes); w = 1 terrain layout
  levels: array<vec4f, 4>,
  ground: array<vec4f, 256>,
}
@group(0) @binding(11) var<uniform> PR: RefrProbes;
@group(0) @binding(12) var<storage, read> probe_sh: array<vec4f>;   // per probe 9 x rgb, radiance arriving from w
@group(0) @binding(13) var exit_conf: texture_2d<f32>;   // E: miss confidence * w, w (splat.wgsl fs_exit)
// G: the inside ray map (splat.wgsl fs_exitin) and the posed inside Gaussians (exitmap.wgsl prep_inner); their colour
// is the splat's (project.wgsl), so a gathered Gaussian is lit as it would be drawn
@group(0) @binding(15) var in_pnt: texture_2d<f32>;    // (entry point - lens centre) * w, leg 0 length * w
@group(0) @binding(16) var in_dir0: texture_2d<f32>;   // leg 0 direction * w, w
@group(0) @binding(17) var in_dir1: texture_2d<f32>;   // leg 1 direction * w, leg 1 length * w
@group(0) @binding(18) var<storage, read> splats: array<Splat>;
@group(0) @binding(20) var<storage, read> inner: array<vec4f>;   // per inside entry (exitmap.wgsl prep_inner); found through the ray grid (@group(1), hgrid.wgsl)
// binding 14 probe_gain (array<vec4f>, rgb gain, w > 0 once set): read here, written by refract_calib (renderer.js
// declares it per module)

@vertex
fn vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((v << 1u) & 2u), f32(v & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn tap(t: texture_2d<f32>, p: vec2f) -> vec4f { return textureSampleLevel(t, lin, p / frame.viewport, 0.0); }

fn to_px(p: vec3f) -> vec2f {
  let z = max(-p.z, 1e-3);
  return vec2f(0.5 * frame.viewport.x + frame.focal.x * p.x / z, 0.5 * frame.viewport.y - frame.focal.y * p.y / z);
}

fn schlick(cosi: f32, ior: f32) -> f32 {
  let f0 = (ior - 1.0) * (ior - 1.0) / ((ior + 1.0) * (ior + 1.0));
  return f0 + (1.0 - f0) * pow(1.0 - clamp(cosi, 0.0, 1.0), 5.0);
}

// What the eye ray d sees through lens l entered at t_near: transmitted rgb, a = coverage of what it met.
fn through(l: Lens, d: vec3f, tn: f32, ior: f32) -> vec4f {
  let P = d * tn;
  let dn = normalize(d);
  let n = normalize(lens_nt(l, lens_n(l, P - l.c.xyz)));
  let F_in = schlick(-dot(dn, n), ior);
  let t1 = refract(dn, n, 1.0 / ior);
  let s1 = max(lens_hit(l, P, t1).y, 0.0);
  let Q = P + t1 * s1;
  let n2 = normalize(lens_nt(l, lens_n(l, Q - l.c.xyz)));
  var t2 = refract(t1, -n2, ior);
  var F_out = schlick(dot(t1, n2), ior);
  if (dot(t2, t2) < 0.5) { t2 = reflect(t1, -n2); F_out = 0.0; }   // TIR: one internal bounce, then out
  // inside layer: the inside ray's point at that layer's depth
  let pz = -P.z;
  let vz = max(-t1.z, 1e-3);
  var zs = pz + 0.5 * s1 * vz;
  var q = to_px(P + t1 * (0.5 * s1));
  let warped = (LENSES.info.z & REFR_WARP) != 0u;   // A + C: the inside Gaussians already sit at their images
  if (warped) { q = to_px(P); }
  for (var k = 0; k < select(3, 0, warped); k++) {
    q = to_px(P + t1 * clamp((zs - pz) / vz, 0.0, s1));
    let a = tap(t_inside, q).a;
    if (a > 1e-3) { zs = tap(z_inside, q).r / a; }
  }
  var inner = tap(t_inside, q);
  if (LENSES.info.w != 0u) { inner = vec4f(0.0); }   // compact A + C: the inside sits in the front layer
  // behind layer: the exit ray's point at that layer's depth (a ray turning back toward the eye goes 0.3 m)
  let qz = -Q.z;
  let wz = -t2.z;
  var zb = qz + 0.05;
  var r = q;
  for (var k = 0; k < 4; k++) {
    let s = select(0.3, max(zb - qz, 0.0) / wz, wz > 0.05);
    r = to_px(Q + t2 * s);
    let a = tap(t_behind, r).a;
    if (a > 1e-3) { zb = tap(z_behind, r).r / a; }
  }
  let back = tap(t_behind, r);
  let T = (1.0 - F_in) * (1.0 - F_out) * exp(-LENSES.params.z * s1);
  return vec4f((inner.rgb + (1.0 - inner.a) * back.rgb) * T, inner.a + (1.0 - inner.a) * back.a);
}

fn pr_level(k: u32) -> f32 { return PR.levels[k / 4u][k % 4u]; }
fn pr_ground(i: u32, j: u32) -> f32 { let q = i * PR.dims.y + j; return PR.ground[q / 4u][q % 4u]; }

// The probes' radiance arriving at world point P from world direction w: trilinear over the lattice cell around P
// (bounce.wgsl lat_cell, without its Chebyshev visibility), SH2 evaluated at w. Black without probes.
fn probe_radiance(P: vec3f, w: vec3f) -> vec3f {
  let dm = PR.dims.xyz;
  if (dm.x < 2u || dm.y < 2u || dm.z < 2u) { return vec3f(0.0); }
  let fxy = clamp((P.xy - PR.lo.xy) / PR.step.xy, vec2f(0.0), vec2f(dm.xy - vec2u(1u)));
  let ij = min(vec2u(floor(fxy)), dm.xy - vec2u(2u));
  let t = fxy - vec2f(ij);
  let g = mix(mix(pr_ground(ij.x, ij.y), pr_ground(ij.x, ij.y + 1u), t.y),
              mix(pr_ground(ij.x + 1u, ij.y), pr_ground(ij.x + 1u, ij.y + 1u), t.y), t.x);
  let h = P.z - g;
  var k = 0u;
  while (k + 2u < dm.z && h > pr_level(k + 1u)) { k += 1u; }
  let tz = clamp((h - pr_level(k)) / max(pr_level(k + 1u) - pr_level(k), 1e-6), 0.0, 1.0);
  let f = vec3f(t, tz);
  let y = array<f32, 9>(0.282095, 0.488603 * w.y, 0.488603 * w.z, 0.488603 * w.x, 1.092548 * w.x * w.y,
    1.092548 * w.y * w.z, 0.315392 * (3.0 * w.z * w.z - 1.0), 1.092548 * w.x * w.z, 0.546274 * (w.x * w.x - w.y * w.y));
  var L = vec3f(0.0);
  for (var c = 0u; c < 8u; c++) {
    let o = vec3u(c >> 2u, (c >> 1u) & 1u, c & 1u);
    let tw = select(1.0 - f, f, o == vec3u(1u));
    let p = ((ij.x + o.x) * dm.y + ij.y + o.y) * dm.z + k + o.z;
    for (var m = 0u; m < 9u; m++) { L += tw.x * tw.y * tw.z * y[m] * probe_sh[p * 9u + m].rgb; }
  }
  return max(L, vec3f(0.0));
}

// E: what the pixel sees through the glass, from the exit map: the behind layer along the mapped exit ray (same depth
// fixed point as `through`), with how far to trust that lookup and the probes' radiance for when not to
struct ExitLook { straight: vec4f, back: vec4f, Lp: vec3f, T: f32, conf: f32, w: f32 }

fn exit_look(ip: vec2i, p: vec2f) -> ExitLook {
  var o: ExitLook;
  let m0 = textureLoad(exit_dir, ip, 0);
  let m1 = textureLoad(exit_pnt, ip, 0);
  let mc = textureLoad(exit_conf, ip, 0);
  o.straight = textureLoad(t_behind, ip, 0);
  o.w = m0.w;
  if (m0.w < 1e-3) { return o; }
  let t2 = normalize(m0.xyz);
  let Q = m1.xyz / m0.w;
  o.T = m1.w / m0.w;
  let qz = -Q.z;
  let wz = -t2.z;
  var zb = qz + 0.05;
  var r = p;
  for (var k = 0; k < 4; k++) {
    let s = select(0.3, max(zb - qz, 0.0) / wz, wz > 0.05);
    r = to_px(Q + t2 * s);
    let a = tap(t_behind, r).a;
    if (a > 1e-3) { zb = tap(z_behind, r).r / a; }
  }
  o.back = tap(t_behind, r);
  // the exit ray can meet something nearer than the lens's far side (a sphere's lower half sees the floor in front of
  // it), which the layering put in the front layer: off the lens footprint, the front layer over the behind one
  let rc = clamp(vec2i(r), vec2i(0), vec2i(frame.viewport) - vec2i(1));
  if (textureLoad(lensmap, rc, 0).z < 0.5) {
    let fr = tap(t_front, r);
    o.back = vec4f(fr.rgb + (1.0 - fr.a) * o.back.rgb, fr.a + (1.0 - fr.a) * o.back.a);
  }
  // a miss: the exit ray turns back toward the eye or is still trapped, or its lookup leaves the screen (both per glass
  // Gaussian from the exit trace, blended by the map's splats), or the behind layer is empty there (per pixel)
  let vp = frame.viewport;
  let edge = min(min(r.x, vp.x - r.x), min(r.y, vp.y - r.y));
  o.conf = (mc.r / max(mc.a, 1e-6)) * clamp(edge / (0.04 * vp.y), 0.0, 1.0) * smoothstep(0.1, 0.7, o.back.a);
  if (LENSES.params2.w > 0.5) {   // ?refrconf=px: the per-pixel confidence alone (the first version)
    o.conf = smoothstep(0.05, 0.2, wz) * clamp(edge / (0.03 * vp.y), 0.0, 1.0) * clamp(o.back.a / 0.5, 0.0, 1.0);
  }
  let Rt = transpose(mat3x3f(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz));
  o.Lp = probe_radiance(Rt * (Q - frame.view[3].xyz), normalize(Rt * t2));
  return o;
}

fn through_exit(ip: vec2i, p: vec2f) -> vec4f {
  let o = exit_look(ip, p);
  if (o.w < 1e-3) { return o.straight; }
  // the probes, levelled to the screen by refract_calib's gain, fill in as the confidence falls
  let g = select(vec3f(1.0), probe_gain[0].rgb, probe_gain[0].w > 0.0);
  let Lp = o.Lp * g * LENSES.params2.y;
  var seen = mix(vec4f(Lp * o.T, 1.0), vec4f(o.back.rgb * o.T, o.back.a), o.conf);
  if (LENSES.params2.y <= 0.0) { seen = vec4f(o.back.rgb * o.T, o.back.a); }
  if (LENSES.params2.z > 0.5) { seen = vec4f(1.0 - o.conf, o.conf, 0.0, 1.0); }
  if (LENSES.params2.z > 1.5) { seen = vec4f(Lp, 1.0); }
  return mix(o.straight, seen, clamp(o.w / 0.5, 0.0, 1.0));
}

// G: a Gaussian's colour as the splat pass would draw it (diffuse + specular)
fn splat_rgb(i: u32) -> vec3f {
  let s = splats[i];
  if ((u32(frame.lamp_color.w + 0.5) & REFL_SPLIT) != 0u && frame.view_mode == 3u) {
    let a = unpack2x16float(s.color_depth.x);
    let b = unpack2x16float(s.color_depth.y);
    let c = unpack2x16float(s.color_depth.z);
    return vec3f(a, b.x) + vec3f(b.y, c);
  }
  return bitcast<vec3f>(s.color_depth.xyz);
}

fn erf_a(x: f32) -> f32 {   // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
  return select(-y, y, x >= 0.0);
}

const GATHER_K: u32 = 16u;
const GATHER_STEPS: u32 = 256u;   // cells per level per leg at most (a bound, as exitmap.wgsl HG_STEPS)

// G: the inside of lens k along the pixel's mapped inside ray, entry leg then the leg after a TIR bounce: each inside
// Gaussian the ray grid files along a leg (HG_TAG_INSIDE, lens k) adds its optical depth over the leg in closed form
// (tau = kappa exp(-q/2) sqrt(2 pi / a), clipped to the leg with erf); the GATHER_K nearest by their peak, composited
// front to back. rgb premultiplied, a = coverage. Each pixel follows its own ray, so a point seen along two paths
// (through a side face, or by TIR) shows twice.
fn gather_inside(ip: vec2i, k: u32) -> vec4f {
  let m0 = textureLoad(in_dir0, ip, 0);
  if (m0.w < 1e-3) { return vec4f(0.0); }
  let m1 = textureLoad(in_pnt, ip, 0);
  let m2 = textureLoad(in_dir1, ip, 0);
  let o0 = m1.xyz / m0.w + LENSES.lens[k].c.xyz;
  let L0 = m1.w / m0.w;
  let d0 = normalize(m0.xyz);
  let L1 = m2.w / m0.w;
  let d1 = select(vec3f(0.0), normalize(m2.xyz), dot(m2.xyz, m2.xyz) > 1e-12);
  let o1 = o0 + d0 * L0;
  var hs: array<vec2f, 16>;   // (distance along the path, alpha)
  var hi: array<u32, 16>;
  var n = 0u;
  let mask = hg_mask();
  for (var leg = 0u; leg < 2u; leg++) {
    let o = select(o0, o1, leg == 1u);
    let d = select(d0, d1, leg == 1u);
    let len = select(L0, L1, leg == 1u);
    if (len <= 0.0) { continue; }
    for (var lv = 0u; lv < HG_LEVELS; lv++) {
      if ((mask & (1u << lv)) == 0u) { continue; }
      var w = hg_begin(lv, o, d);
      for (var it = 0u; it < GATHER_STEPS && w.se <= w.tout && w.se <= len; it++) {
        let rg = hg_range(lv, w);
        for (var q = rg.x; q < rg.y; q++) {
          let en = hg_refs[q];
          // a collision from another level, another system's item, or another lens
          if ((en >> 27u) != lv || ((en >> HG_KEY_BITS) & 0xfu) != HG_TAG_INSIDE || ((en >> 20u) & 7u) != k) { continue; }
          let j = en & 0xfffffu;
          let A = inner[3u * j]; let B = inner[3u * j + 1u]; let C = inner[3u * j + 2u];
          let P = mat3x3f(vec3f(B.x, B.y, B.z), vec3f(B.y, C.x, C.y), vec3f(B.z, C.y, C.z));
          let r = A.xyz - o;
          let Pd = P * d;
          let a = dot(d, Pd);
          let b = dot(r, Pd);
          let sp = b / a;
          // counted once, in the cell holding its nearest point on the leg, and only within 4 sigma there
          let x = o + d * clamp(sp, 0.0, len);
          let rx = A.xyz - x;
          if (dot(rx, P * rx) > 16.0 || !hg_owns(lv, w.c, x)) { continue; }
          let qd = dot(r, P * r) - b * b / a;
          let h = sqrt(0.5 * a);
          let frac = 0.5 * (erf_a((len - sp) * h) - erf_a(-sp * h));
          let tau = A.w * exp(-0.5 * qd) * sqrt(6.2831853 / a) * frac;
          let alpha = 1.0 - exp(-tau);
          if (alpha < 1.0 / 255.0) { continue; }
          let key = clamp(sp, 0.0, len) + select(0.0, L0, leg == 1u);
          // insertion into the nearest GATHER_K by distance
          if (n == GATHER_K && key >= hs[GATHER_K - 1u].x) { continue; }
          var at = min(n, GATHER_K - 1u);
          while (at > 0u && hs[at - 1u].x > key) { hs[at] = hs[at - 1u]; hi[at] = hi[at - 1u]; at--; }
          hs[at] = vec2f(key, alpha); hi[at] = bitcast<u32>(B.w);
          n = min(n + 1u, GATHER_K);
        }
        if (!hg_next(&w)) { break; }
      }
    }
  }
  var acc = vec4f(0.0);
  for (var q = 0u; q < n; q++) {
    let al = hs[q].y * (1.0 - acc.a);
    acc += vec4f(splat_rgb(hi[q]) * al, al);
  }
  return acc * clamp(m0.w / 0.02, 0.0, 1.0);   // values are weight-normalised; fade only where no skin Gaussian reaches
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = pos.xy;
  let ip = vec2i(p);
  let f = textureLoad(t_front, ip, 0);
  if ((LENSES.info.z & REFR_LENS) == 0u) {
    // B: the behind layer read at the pixel moved by the summed glass gradient (inside target rg)
    let off = textureLoad(t_inside, ip, 0).xy * LENSES.params2.x * frame.viewport.y * frame.viewport.y * 1e-3;
    var b: vec4f;
    if (LENSES.info.y != 0u) {
      let s = LENSES.params.y * 10.0;   // dispersion: the offset spread per channel
      b = vec4f(tap(t_behind, p + off * (1.0 - s)).r, tap(t_behind, p + off).g, tap(t_behind, p + off * (1.0 + s)).b, tap(t_behind, p + off).a);
    } else { b = tap(t_behind, p + off); }
    return vec4f(f.rgb + (1.0 - f.a) * b.rgb, f.a + (1.0 - f.a) * b.a);
  }
  let d = eye_ray(p);
  let lm = textureLoad(lensmap, ip, 0);
  let best = lm.x;
  let hit = i32(lm.z + 0.5) - 1;
  var under: vec4f;
  if (hit < 0) {
    let i = textureLoad(t_inside, ip, 0);
    let b = textureLoad(t_behind, ip, 0);
    under = vec4f(i.rgb + (1.0 - i.a) * b.rgb, i.a + (1.0 - i.a) * b.a);
  } else {
    let l = LENSES.lens[hit];
    let ior = LENSES.params.x;
    if ((LENSES.info.z & REFR_EXIT) != 0u) {
      under = through_exit(ip, p);
      if ((LENSES.info.z & REFR_GATHER) != 0u) {
        let g = gather_inside(ip, u32(hit));
        under = vec4f(g.rgb + (1.0 - g.a) * under.rgb, g.a + (1.0 - g.a) * under.a);
      }
    } else if (LENSES.info.y != 0u) {
      // dispersion: one bent ray per channel, ior spread by params.y
      let s = LENSES.params.y;
      let r = through(l, d, best, ior - s);
      let g = through(l, d, best, ior);
      let b = through(l, d, best, ior + s);
      under = vec4f(r.r, g.g, b.b, g.a);
    } else {
      under = through(l, d, best, ior);
    }
  }
  return vec4f(f.rgb + (1.0 - f.a) * under.rgb, f.a + (1.0 - f.a) * under.a);
}

// Lens map (A, A + C): the nearest lens on each pixel's eye ray, once per pixel before the splat pass, so neither
// fs_layers nor the composite loops over lenses per fragment. (t_near, t_far, lens index + 1, 0); 0 = no lens.
@fragment
fn fs_lensmap(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = pos.xy;
  let d = eye_ray(p);
  var best = vec4f(0.0);
  for (var k = 0u; k < min(LENSES.info.x, 8u); k++) {
    let l = LENSES.lens[k];
    if (!in_rect(l, p)) { continue; }
    let t = lens_hit(l, vec3f(0.0), d);
    if (t.x <= t.y && t.x > 0.0 && (best.z == 0.0 || t.x < best.x)) { best = vec4f(t.x, t.y, f32(k + 1u), 0.0); }
  }
  return best;
}

// --- refract_calib: compiled only into the calib module (renderer.js), whose probe_gain is read_write ---
// E miss fallback, once per frame before the composite: the gain that levels the probes to the screen, from the
// pixels whose exit lookup is trusted (behind layer rgb over probe radiance along the same ray, per channel), eased
// over frames
var<workgroup> cal_b: array<vec4f, 256>;
var<workgroup> cal_p: array<vec3f, 256>;

@compute @workgroup_size(256)
fn refract_calib(@builtin(local_invocation_index) li: u32) {
  var sb = vec4f(0.0);
  var sp = vec3f(0.0);
  let vp = frame.viewport;
  for (var j = 0u; j < 16u; j++) {
    let c = li * 16u + j;   // 64 x 64 samples over the screen
    let p = (vec2f(f32(c % 64u), f32(c / 64u)) + 0.5) * vp / 64.0;
    let ip = vec2i(p);
    if (textureLoad(lensmap, ip, 0).z < 0.5) { continue; }
    let o = exit_look(ip, p);
    if (o.w < 0.5 || o.conf < 0.8) { continue; }
    sb += vec4f(o.back.rgb, 1.0);
    sp += o.Lp;
  }
  cal_b[li] = sb; cal_p[li] = sp;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (li < h) { cal_b[li] += cal_b[li + h]; cal_p[li] += cal_p[li + h]; }
    workgroupBarrier();
  }
  if (li != 0u || cal_b[0].w < 16.0) { return; }
  let ratio = clamp(cal_b[0].rgb / max(cal_p[0], vec3f(1e-6)), vec3f(0.25), vec3f(8.0));
  let old = probe_gain[0];
  probe_gain[0] = vec4f(select(ratio, mix(old.rgb, ratio, 0.25), old.w > 0.0), 1.0);
}

