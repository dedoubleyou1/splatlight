// Gaussian particles (?particles=sparks|smoke|dust|rain, src/particles.js): one compute pass a frame steps every
// particle and writes it as a world-space Gaussian (common.wgsl Gaussian, 64 B) into its own slot range, which
// project.wgsl draws after the scene's Gaussians, LOD nodes and emitters. The particles are sorted with the scene, so
// smoke wraps round the boxes and sparks pass in front of and behind them with no separate transparency pass.
//
// Colour is final radiance (drawn like an emitter): sparks emit (a cooling blackbody), smoke / dust / rain scatter the
// lighting's SH2 probe field through a Henyey-Greenstein phase evaluated on the probes' SH bands (no normal: the
// particle is lit as the fuzzy volume it is). Motion blur is built into the Gaussian: a Gaussian moving at v over the
// exposure is, to second order, one with covariance + v v^T / 12 (project.wgsl's motion blur), so a spark becomes an
// ellipsoid stretched along its velocity with its density lowered to keep its mass.

struct PUniform {
  t: f32, dt: f32, count: u32, preset: u32,      // preset: 0 sparks, 1 smoke, 2 dust, 3 rain
  origin: vec4f,                                 // emitter position xyz, w = emission rate scale
  room_lo: vec4f,                                // collision box (the room), w = shutter (s) for the velocity smear
  room_hi: vec4f,                                // w = brightness (sparks' emission scale, others' lighting gain)
  wind: vec4f,                                   // xyz wind m/s, w = turbulence strength
  look: vec4f,                                   // x size scale, y density scale, z phase g override (< -1: preset's), w seed
  eye: vec4f,                                    // camera position (the phase function needs the view direction)
  lamp_c: vec4f,                                 // rectangular lamp: centre xyz, area m^2 (0: no lamp)
  lamp_L: vec4f,                                 // its radiance rgb, w = 1: march the particle shadow grid toward it
}

@group(0) @binding(0) var<uniform> P: PUniform;
@group(0) @binding(1) var<storage, read_write> pstate: array<vec4f>;   // 2 per particle: (pos, age), (vel, life)
@group(0) @binding(2) var<storage, read_write> pout: array<Gaussian>;

// The lighting's probe lattice (bounce.wgsl ProbeParams prefix, as refract.wgsl reads it) and SH2 radiance
struct RefrProbes {
  lo: vec4f, step: vec4f,
  pad: array<vec4f, 50>,
  dims: vec4u,
  levels: array<vec4f, 4>,
  ground: array<vec4f, 256>,
}
@group(0) @binding(3) var<uniform> PR: RefrProbes;
@group(0) @binding(4) var<storage, read> probe_sh: array<vec4f>;
// Particle shadow grid: GRID^3 cells over the room box, each the extinction (1/m, fixed point x GRID_FIX) the
// scattering particles deposit this frame (step), read back by shade's march toward the lamp: smoke shades itself
@group(0) @binding(5) var<storage, read_write> grid: array<atomic<u32>>;
const GRID: u32 = 32u;
const GRID_FIX: f32 = 1000.0;

fn grid_cell(p: vec3f) -> vec3f { return (p - P.room_lo.xyz) / (P.room_hi.xyz - P.room_lo.xyz) * f32(GRID) - 0.5; }
fn grid_index(c: vec3i) -> u32 { return (u32(c.z) * GRID + u32(c.y)) * GRID + u32(c.x); }
fn grid_in(c: vec3i) -> bool { return all(c >= vec3i(0)) && all(c < vec3i(i32(GRID))); }

// add optical mass m (the extinction integral, m^2) at p, trilinear over 8 cells
fn deposit(p: vec3f, m: f32) {
  let cell = (P.room_hi.xyz - P.room_lo.xyz) / f32(GRID);
  let sigma = m / (cell.x * cell.y * cell.z);
  let f = grid_cell(p);
  let b = vec3i(floor(f));
  let t = f - floor(f);
  for (var c = 0u; c < 8u; c++) {
    let o = vec3u(c & 1u, (c >> 1u) & 1u, c >> 2u);
    let k = b + vec3i(o);
    if (!grid_in(k)) { continue; }
    let w3 = select(1.0 - t, t, o == vec3u(1u));
    atomicAdd(&grid[grid_index(k)], u32(sigma * w3.x * w3.y * w3.z * GRID_FIX + 0.5));
  }
}

fn grid_sigma(p: vec3f) -> f32 {
  let k = vec3i(floor(grid_cell(p) + 0.5));
  if (!grid_in(k)) { return 0.0; }
  return f32(atomicLoad(&grid[grid_index(k)])) / GRID_FIX;
}

// Henyey-Greenstein, normalised over the sphere
fn hg(cos_t: f32, g: f32) -> f32 {
  let d = 1.0 + g * g - 2.0 * g * cos_t;
  return (1.0 - g * g) / (12.5663706 * d * sqrt(d));
}

fn pr_level(k: u32) -> f32 { return PR.levels[k / 4u][k % 4u]; }
fn pr_ground(i: u32, j: u32) -> f32 { let q = i * PR.dims.y + j; return PR.ground[q / 4u][q % 4u]; }

// Light scattered toward the eye by a particle at P seen along view direction d (eye -> particle), phase HG(g):
// with incident radiance L(w) = sum c_lm Y_lm(w) (w toward the source), the HG convolution scales band l by g^l and
// forward scattering means light arriving from w = d: L_s = sum g^l c_lm Y_lm(d). g = 0: the mean incident radiance.
fn probe_scatter(Pw: vec3f, d: vec3f, g: f32) -> vec3f {
  let dm = PR.dims.xyz;
  if (dm.x < 2u || dm.y < 2u || dm.z < 2u) { return vec3f(0.05); }
  let fxy = clamp((Pw.xy - PR.lo.xy) / PR.step.xy, vec2f(0.0), vec2f(dm.xy - vec2u(1u)));
  let ij = min(vec2u(floor(fxy)), dm.xy - vec2u(2u));
  let t = fxy - vec2f(ij);
  let gz = mix(mix(pr_ground(ij.x, ij.y), pr_ground(ij.x, ij.y + 1u), t.y),
               mix(pr_ground(ij.x + 1u, ij.y), pr_ground(ij.x + 1u, ij.y + 1u), t.y), t.x);
  let h = Pw.z - gz;
  var k = 0u;
  while (k + 2u < dm.z && h > pr_level(k + 1u)) { k += 1u; }
  let tz = clamp((h - pr_level(k)) / max(pr_level(k + 1u) - pr_level(k), 1e-6), 0.0, 1.0);
  let f = vec3f(t, tz);
  let w = d;
  let g2 = g * g;
  let y = array<f32, 9>(0.282095, g * 0.488603 * w.y, g * 0.488603 * w.z, g * 0.488603 * w.x, g2 * 1.092548 * w.x * w.y,
    g2 * 1.092548 * w.y * w.z, g2 * 0.315392 * (3.0 * w.z * w.z - 1.0), g2 * 1.092548 * w.x * w.z, g2 * 0.546274 * (w.x * w.x - w.y * w.y));
  var L = vec3f(0.0);
  for (var c = 0u; c < 8u; c++) {
    let o = vec3u(c >> 2u, (c >> 1u) & 1u, c & 1u);
    let tw = select(1.0 - f, f, o == vec3u(1u));
    let p = ((ij.x + o.x) * dm.y + ij.y + o.y) * dm.z + k + o.z;
    for (var m = 0u; m < 9u; m++) { L += tw.x * tw.y * tw.z * y[m] * probe_sh[p * 9u + m].rgb; }
  }
  // SH radiance -> scattered radiance: the l = 0 term is c00 Y00 = the mean incident radiance (times 4 pi / 4 pi)
  return max(L, vec3f(0.0));
}

fn hash4(i: u32) -> vec4f {
  var h = i * 747796405u + 2891336453u;
  var r: vec4f;
  for (var k = 0; k < 4; k++) {
    h = h * 747796405u + 2891336453u;
    let w = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    r[k] = f32((w >> 22u) ^ w) / 4294967296.0;
  }
  return r;
}

// Divergence-free drift: the curl of a few travelling sine potentials (two octaves), m/s per unit strength
fn curl_noise(p: vec3f, t: f32) -> vec3f {
  var v = vec3f(0.0);
  var s = 1.0;
  var a = 1.0;
  for (var o = 0; o < 2; o++) {
    let x = p.x * s; let y = p.y * s; let z = p.z * s;
    let dAz_dy = 0.5 * 3.3 * cos(3.3 * y + 0.5 * t + 2.7);
    let dAy_dz = 1.9 * cos(1.9 * z + 0.8 * t + 2.1);
    let dAx_dz = 0.5 * 3.1 * cos(3.1 * z - 0.7 * t + 1.3);
    let dAz_dx = 2.3 * cos(2.3 * x - 0.7 * t + 0.4);
    let dAy_dx = 0.5 * 2.9 * cos(2.9 * x + 0.6 * t);
    let dAx_dy = 1.7 * cos(1.7 * y + 0.9 * t);
    v += a * vec3f(dAz_dy - dAy_dz, dAx_dz - dAz_dx, dAy_dx - dAx_dy);
    s *= 2.7; a *= 0.45;
  }
  return v * 0.25;
}

// Blackbody-ish colour for a cooling spark, T in kelvin (fit good to a few % over 1000-4000 K), luminance ~1 at 2500 K
fn ember(T: f32) -> vec3f {
  let x = clamp((T - 800.0) / 2700.0, 0.0, 1.5);
  let rgb = vec3f(1.0, clamp(0.08 + 0.75 * x * x, 0.0, 1.0), clamp(-0.15 + 0.9 * x * x * x, 0.0, 1.0));
  return rgb * pow(max(T / 2500.0, 0.0), 6.0);   // radiance falls steeply as it cools
}

// Rotation taking +z to unit vector a (shortest arc), as a quaternion x, y, z, w
fn quat_z_to(a: vec3f) -> vec4f {
  let c = a.z;
  if (c < -0.9999) { return vec4f(1.0, 0.0, 0.0, 0.0); }
  let ax = vec3f(-a.y, a.x, 0.0);   // cross(z, a)
  let q = vec4f(ax, 1.0 + c);
  return q / length(q);
}

fn spawn(i: u32, gen: u32) -> array<vec4f, 2> {
  let h = hash4(i * 9781u + gen * 6271u + u32(P.look.w));
  let h2 = hash4(i * 1307u + gen * 7919u + 17u);
  var pos = P.origin.xyz;
  var vel = vec3f(0.0);
  var life = 1.0;
  let lo = P.room_lo.xyz; let hi = P.room_hi.xyz;
  switch (P.preset) {
    case 0u: {   // sparks: a cone round +z from a point, 2.5-5 m/s
      let ph = 6.2831853 * h.x;
      let ct = mix(0.55, 1.0, h.y);
      let st = sqrt(1.0 - ct * ct);
      vel = vec3f(st * cos(ph), st * sin(ph), ct) * mix(2.0, 4.5, h.z);
      pos += (h2.xyz - 0.5) * 0.02;
      life = mix(0.8, 2.2, h.w);
    }
    case 1u: {   // smoke: a 4 cm disc, rising
      let r = 0.04 * sqrt(h.x); let ph = 6.2831853 * h.y;
      pos += vec3f(r * cos(ph), r * sin(ph), 0.0);
      vel = vec3f((h2.x - 0.5) * 0.1, (h2.y - 0.5) * 0.1, 0.35 + 0.2 * h.z);
      life = mix(5.0, 8.0, h.w);
    }
    case 2u: {   // dust: anywhere in the room
      pos = mix(lo + 0.05, hi - 0.05, h.xyz);
      vel = (h2.xyz - 0.5) * 0.02;
      life = mix(6.0, 12.0, h.w);
    }
    default: {   // rain: a sheet under the ceiling, falling at 6-7 m/s
      pos = vec3f(mix(lo.x, hi.x, h.x), mix(lo.y, hi.y, h.y), hi.z - 0.02);
      vel = vec3f(P.wind.x, P.wind.y, -mix(5.5, 7.0, h.z));
      life = 2.0;
    }
  }
  return array<vec4f, 2>(vec4f(pos, 0.0), vec4f(vel, life));
}

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.count) { return; }
  var s0 = pstate[2u * i];
  var s1 = pstate[2u * i + 1u];
  let dt = P.dt;
  var pos = s0.xyz;
  var age = s0.w + dt;
  var vel = s1.xyz;
  var life = s1.w;
  // age < 0: not born yet (the first fill is staggered so emission is steady); a dead slot respawns
  if (age >= life && life > 0.0) {
    let gen = u32(floor(P.t * 7.0)) + i;   // a different draw each respawn
    let s = spawn(i, gen);
    pos = s[0].xyz; vel = s[1].xyz; life = s[1].w;
    age = min(age - s1.w, dt);   // carry the remainder so slots stay spread out in time
    if (s1.w < 1e-5 && P.preset >= 2u) {
      // dust and rain start already in flight: a random age, and rain anywhere on its fall
      let hb = hash4(i * 4099u + 11u);
      age = hb.x * life * select(1.0, 0.0, P.preset == 3u);
      if (P.preset == 3u) { pos.z = mix(P.room_lo.z, P.room_hi.z, hb.y); }
    }
  }
  var out: Gaussian;
  out.part_surface = 0u;
  out.flags = 0u;
  out.kappa = 0.0;
  out.quat = vec4f(0.0, 0.0, 0.0, 1.0);
  out.scale = vec3f(1e-3);
  out.mu = pos;
  out.albedo = vec3f(0.0);
  if (age < 0.0 || life <= 0.0) {
    pstate[2u * i] = vec4f(pos, age);
    pstate[2u * i + 1u] = vec4f(vel, life);
    pout[i] = out;
    return;
  }
  let lo = P.room_lo.xyz; let hi = P.room_hi.xyz;
  let h = hash4(i * 31u + 5u);
  let tl = age / life;
  var size = 0.003;
  var tau_peak = 1.0;   // optical depth through the centre, at rest
  var colour = vec3f(0.0);
  var g = 0.0;
  var albedo = vec3f(0.8);
  var emissive = false;
  var stretch = true;
  switch (P.preset) {
    case 0u: {
      vel.z -= 9.8 * dt;
      vel *= exp(-0.6 * dt);
      let T = 3600.0 * exp(-age / 0.55) + 300.0;
      colour = ember(T) * P.room_hi.w;
      size = 0.0022;
      tau_peak = 3.0;
      emissive = true;
      if (T < 900.0) { tau_peak = 0.0; }   // gone dark: a cold spark is not worth drawing
    }
    case 1u: {
      // relax toward a rise that slows with age (hot smoke cools) plus the wind and the curl-noise turbulence
      let rise = vec3f(0.0, 0.0, 0.08 + 0.35 * exp(-age / 2.5));
      vel += 1.6 * (rise + P.wind.xyz + curl_noise(pos * 2.2, P.t) * 1.2 * P.wind.w - vel) * dt;
      size = 0.018 + 0.03 * age;   // spreads as it rises
      // optical mass thins as it grows (tau ~ 1 / size^1.4: a little mass is lost to the air), fading in and out
      tau_peak = 1.4 * pow(0.018 / size, 1.4) * smoothstep(0.0, 0.08, tl) * (1.0 - smoothstep(0.6, 1.0, tl));
      albedo = vec3f(0.62, 0.6, 0.58);
      g = 0.35;
      stretch = false;
    }
    case 2u: {
      vel += (P.wind.xyz * 0.05 + curl_noise(pos * 2.0, 0.3 * P.t) * 0.06 * max(P.wind.w, 0.2) - vel) * min(1.0, 2.0 * dt);
      size = 0.0015;
      tau_peak = 0.8 * smoothstep(0.0, 0.15, tl) * (1.0 - smoothstep(0.85, 1.0, tl));
      albedo = vec3f(0.9, 0.85, 0.75);
      g = 0.75;   // strongly forward: motes light up looking toward the lamp
      stretch = false;
    }
    default: {
      vel.z -= 9.8 * dt;
      vel = vec3f(P.wind.x, P.wind.y, max(vel.z, -7.5));
      size = 0.0012;
      tau_peak = 0.35;
      albedo = vec3f(0.9, 0.95, 1.0);
      g = 0.85;
    }
  }
  if (P.look.z >= -1.0) { g = P.look.z; }
  pos += vel * dt;
  // collisions with the room box: sparks bounce and skid, rain dies on the floor, smoke and dust slide
  for (var a = 0; a < 3; a++) {
    if (pos[a] < lo[a]) {
      pos[a] = lo[a];
      if (P.preset == 3u) { life = age; }
      vel[a] = abs(vel[a]) * select(0.05, 0.35, P.preset == 0u);
      if (P.preset == 0u) { let k = vec3f(f32(a != 0), f32(a != 1), f32(a != 2)); vel = vel * mix(vec3f(1.0), vec3f(0.55), k); }
    }
    if (pos[a] > hi[a]) {
      pos[a] = hi[a];
      vel[a] = -abs(vel[a]) * select(0.05, 0.35, P.preset == 0u);
    }
  }
  pstate[2u * i] = vec4f(pos, age);
  pstate[2u * i + 1u] = vec4f(vel, life);

  size *= P.look.x * mix(0.7, 1.3, h.x);
  tau_peak *= P.look.y;
  // isotropic Gaussian of std `size` with peak optical depth tau_peak: kappa sqrt(2 pi) size = tau_peak
  var scale = vec3f(size);
  var kappa = tau_peak / (2.5066283 * size);
  var quat = vec4f(0.0, 0.0, 0.0, 1.0);
  if (!stretch) {
    // smoke and dust: a little anisotropy and a random orientation per particle, so the blobs do not read as spheres
    scale = vec3f(size * mix(0.75, 1.0, h.y), size, size * mix(1.0, 1.35, h.z));
    let q = hash4(i * 77u + 3u) * 2.0 - 1.0;
    quat = q / max(length(q), 1e-4);
    kappa = tau_peak / (2.5066283 * scale.y);
  }
  // motion blur: covariance s^2 I + v v^T / 12 over the shutter; the long axis along v (the Gaussian's z, the
  // largest scale), density lowered by sqrt(1 + |v|^2 / (12 s^2)) to keep the mass (project.wgsl, determinant lemma)
  let shutter = P.room_lo.w;
  if (stretch && shutter > 0.0) {
    let sv = vel * shutter;
    let m2 = dot(sv, sv) / (12.0 * size * size);
    if (m2 > 1e-4) {
      scale = vec3f(size, size, size * sqrt(1.0 + m2));
      quat = quat_z_to(normalize(sv));
      kappa /= sqrt(1.0 + m2);
    }
  }
  var flags = 0u;
  if (!emissive && tau_peak > 0.0) {
    // lit in `shade` once every particle has deposited its optical mass into the shadow grid
    colour = albedo * P.room_hi.w;
    flags = 1u | (u32(clamp(g, -0.99, 0.99) * 1000.0 + 1000.0) << 1u);
    deposit(pos, kappa * 15.7496099 * scale.x * scale.y * scale.z);   // (2 pi)^1.5 kappa s1 s2 s3
  }
  out.mu = pos;
  out.kappa = select(0.0, kappa, tau_peak > 0.0);
  out.quat = quat;
  out.scale = scale;
  out.albedo = colour;
  out.flags = flags;
  pout[i] = out;
}

// Scattering particles: the probe field through the phase (ambient + bounce, smooth) plus the lamp, attenuated by the
// particles' own shadow grid (self-shadowing: the side of a plume away from the lamp goes dark). The scene does not
// shadow the particles here (the boxes do not block the lamp from the smoke): see the notes.
@compute @workgroup_size(64)
fn shade(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.count) { return; }
  var o = pout[i];
  if ((o.flags & 1u) == 0u || o.kappa <= 0.0) { return; }
  let g = f32(o.flags >> 1u) / 1000.0 - 1.0;
  let d = normalize(o.mu - P.eye.xyz);
  var L = probe_scatter(o.mu, d, g);
  if (P.lamp_c.w > 0.0) {
    let to_l = P.lamp_c.xyz - o.mu;
    let r2 = max(dot(to_l, to_l), 0.01);
    let r = sqrt(r2);
    let wl = to_l / r;
    let cos_l = max(-wl.z, 0.0);   // the panel faces down
    var tau = 0.0;
    if (P.lamp_L.w > 0.5) {
      let n = 24;
      let ds = r / f32(n);
      let j = hash4(i * 13u + u32(P.t * 60.0)).x;   // jittered start: banding becomes grain
      for (var k = 0; k < n; k++) { tau += grid_sigma(o.mu + wl * ds * (f32(k) + j)) * ds; }
      // the particle's own mass sits in the first cells: take off about half of it
      tau = max(tau - 0.5 * o.kappa * 2.5066283 * o.scale.y, 0.0);
    }
    // a small rectangle as a point: E = L A cos / r^2; scattered toward the eye by the phase (light travels -wl,
    // the eye ray d: cos between propagation directions is dot(-wl, -d)). Multiple scattering, approximated by
    // octaves with thinner extinction, a softer phase and less weight (Wrenninge et al., Oz: the Great and Volumetric):
    // without it a thick plume is single-scatter dark, about 4x darker than a white wall under the same lamp
    let E = P.lamp_L.rgb * (P.lamp_c.w * cos_l / r2);
    var ms = 0.0;
    var a = 1.0; var b = 1.0; var c = 1.0;
    for (var oct = 0; oct < 3; oct++) { ms += a * hg(dot(wl, d), g * c) * exp(-tau * b); a *= 0.6; b *= 0.35; c *= 0.4; }
    L += E * ms;
    L *= 1.0 + 0.8 * (1.0 - exp(-0.5 * o.kappa * 2.5066283 * o.scale.y));   // more in-scattering inside thick puffs
  }
  o.albedo *= L;
  o.flags = 0u;
  pout[i] = o;
}
