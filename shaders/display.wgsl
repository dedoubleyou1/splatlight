// Pass 5: resolve the linear HDR target to the swapchain (exposure, sRGB
// encode), or map the overdraw count to a heat ramp.

@group(0) @binding(0) var<uniform> frame: RenderFrame;
@group(0) @binding(1) var hdr: texture_2d<f32>;

// The environment (runtime/SPEC.md 2.9; runtime/wgsl/environment.wgsl): in the lit view the sky and the distant lights'
// discs fill what the splats leave uncovered (the target's alpha is coverage).
struct EnvLight { dir_std: vec4f, irr: vec4f }
struct Env { sh: array<vec4f, 16>, lights: array<EnvLight, 4>, info: vec4u }
@group(0) @binding(2) var<uniform> ENV: Env;
// The background sky when the environment has its own (SPEC 2.9 sky.background_sh_rgb, up to bands 0-8; the lighting
// keeps the 16-coefficient sky), and per light the visible disc's angular std (0 = the lighting's std)
// ring / ground: the horizon's ASG ring (lobe k toward azimuth k * 2pi / n from +x, lying on the horizon; ring_par =
// (lambda along, mu across)) and the ground below the horizon (ground.w = sin of the blend half-width), with info.y lobes;
// glow: the glow round the sun as an SG (rgb irradiance, w = angular std; 0 = none) toward glow_dir; drawn only, its
// light is in the lighting sky
struct SkyBg { sh: array<vec4f, 81>, info: vec4u, disc_std: vec4f, ring: array<vec4f, 8>, ground: vec4f, ring_par: vec4f,
  glow: vec4f, glow_dir: vec4f }
@group(0) @binding(3) var<uniform> BG: SkyBg;

fn sky_radiance(w: vec3f) -> vec3f {
  let x = w.x; let y = w.y; let z = w.z;
  let b = array<f32, 16>(0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x,
    1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y),
    0.590044 * y * (3.0 * x * x - y * y), 2.890611 * x * y * z, 0.457046 * y * (5.0 * z * z - 1.0),
    0.373176 * z * (5.0 * z * z - 3.0), 0.457046 * x * (5.0 * z * z - 1.0), 1.445306 * z * (x * x - y * y),
    0.590044 * x * (x * x - 3.0 * y * y));
  var c = vec3f(0.0);
  for (var k = 0u; k < 16u; k++) { c += b[k] * ENV.sh[k].xyz; }
  return max(c, vec3f(0.0));
}

// the background sky: real SH by recurrence in the order and signs of sky_radiance (index l*l + l + m; m < 0 sin,
// m > 0 cos; no Condon-Shortley phase), as experiments/M49_clear_sky/sh.py; falls back to the lighting sky
fn sky_background(w: vec3f) -> vec3f {
  let n = min(BG.info.x, 81u);
  if (n == 0u) { return sky_radiance(w); }
  let lmax = u32(round(sqrt(f32(n)))) - 1u;
  var c = vec3f(0.0);
  var cm = 1.0; var sm = 0.0;   // r^m cos(m phi), r^m sin(m phi) from (x + i y)^m
  var pmm = 1.0;                // (2m - 1)!!
  for (var m = 0u; m <= lmax; m++) {
    if (m > 0u) {
      let c2 = cm * w.x - sm * w.y; sm = cm * w.y + sm * w.x; cm = c2;
      pmm *= f32(2u * m - 1u);
    }
    var p_prev = 0.0; var p = pmm;
    var kf = 1.0 / (4.0 * 3.14159265);   // (l - m)! / (l + m)! / (4 pi), times (2l + 1) below
    for (var j = 1u; j <= 2u * m; j++) { kf /= f32(j); }
    for (var l = m; l <= lmax; l++) {
      if (l == m + 1u) { p_prev = p; p = w.z * f32(2u * m + 1u) * pmm; }
      else if (l > m + 1u) {
        let pn = (f32(2u * l - 1u) * w.z * p - f32(l + m - 1u) * p_prev) / f32(l - m);
        p_prev = p; p = pn;
        kf *= f32(l - m) / f32(l + m);
      }
      if (l == m + 1u) { kf *= 1.0 / f32(2u * m + 1u); }
      let k = sqrt(f32(2u * l + 1u) * kf);
      let i = l * l + l;
      if (m == 0u) { c += k * p * BG.sh[i].xyz; }
      else { c += 1.41421356 * k * p * (cm * BG.sh[i + m].xyz + sm * BG.sh[i - m].xyz); }
    }
  }
  c = max(c, vec3f(0.0));
  if (BG.glow.w > 0.0) {
    let s2 = BG.glow.w * BG.glow.w;
    c += BG.glow.xyz / (2.0 * 3.14159265 * s2) * exp((dot(w, BG.glow_dir.xyz) - 1.0) / s2);
  }
  let nr = min(BG.info.y, 8u);
  if (nr == 0u) { return c; }
  for (var k = 0u; k < nr; k++) {
    let a = 6.28318531 * f32(k) / f32(nr);
    let z = vec2f(cos(a), sin(a));
    let along = dot(w.xy, vec2f(-z.y, z.x));
    c += BG.ring[k].xyz * max(dot(w.xy, z), 0.0) * exp(-BG.ring_par.x * along * along - BG.ring_par.y * w.z * w.z);
  }
  let t = clamp(w.z / BG.ground.w * 0.5 + 0.5, 0.0, 1.0);
  return mix(BG.ground.xyz, c, t);
}

// the distant lights as seen: an ASG of the light's irradiance spread over its angular size (or over the visible
// disc's, when the environment gives one)
fn lights_radiance(w: vec3f) -> vec3f {
  var c = vec3f(0.0);
  for (var k = 0u; k < min(ENV.info.x, 4u); k++) {
    let L = ENV.lights[k];
    let sd = select(L.dir_std.w, BG.disc_std[k], BG.disc_std[k] > 0.0);
    let s2 = max(sd * sd, 1e-8);
    c += L.irr.rgb / (2.0 * 3.14159265 * s2) * exp((dot(w, L.dir_std.xyz) - 1.0) / s2);
  }
  return c;
}

fn view_dir(pos: vec2f) -> vec3f {
  let v = normalize(vec3f((pos.x - 0.5 * frame.viewport.x) / frame.focal.x, -(pos.y - 0.5 * frame.viewport.y) / frame.focal.y, -1.0));
  let R = mat3x3f(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz);
  return normalize(transpose(R) * v);
}

@vertex
fn vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((v << 1u) & 2u), f32(v & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn srgb_encode(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn heat(t: f32) -> vec3f {
  // black -> blue -> green -> yellow -> red over 0..1
  let x = clamp(t, 0.0, 1.0) * 4.0;
  return clamp(vec3f(x - 2.0, min(x - 1.0, 4.0 - x), min(x, 2.0 - x)), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = textureLoad(hdr, vec2i(pos.xy), 0);
  if (frame.view_mode == 2u) {
    // 0 .. 128 fragments per pixel, log-ish so low counts stay readable.
    return vec4f(heat(log2(1.0 + c.r) / 7.0), 1.0);
  }
  var rgb = c.rgb;
  if (frame.view_mode == 3u && (ENV.info.x > 0u || ENV.info.y != 0u)) {
    let bare = 1.0 - clamp(c.a, 0.0, 1.0);
    if (bare > 1.0 / 1024.0) {   // only where the splats leave the sky showing (the L8 sky is ~250 multiply-adds)
      let w = view_dir(pos.xy);
      rgb += bare * (sky_background(w) * f32(ENV.info.y != 0u) + lights_radiance(w));
    }
  }
  return vec4f(srgb_encode(tone_map(max(rgb * frame.exposure, vec3f(0.0)), frame.tone.x)), 1.0);
}

// Display tone curves (David, 2026-10-31; renderer panel "tone curve", ?tone=clamp|agx|aces|reinhard|neutral|filmic): linear HDR after
// exposure -> linear [0, 1] for the sRGB encode. A per-channel clamp (0, the old behaviour) turns a bright orange
// (1.6, 1.2, 0.7) into a near-white (1, 1, 0.7); the curves roll highlights off instead.
fn tone_map(x: vec3f, mode: u32) -> vec3f {
  if (mode == 1u) { return agx(x); }
  if (mode == 2u) {   // ACES filmic, Narkowicz's fit: punchy, saturated
    return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  }
  if (mode == 3u) {   // Reinhard on luminance: the hue stays, brightness rolls off softly
    let l = dot(x, vec3f(0.2126, 0.7152, 0.0722));
    return clamp(x / (1.0 + l), vec3f(0.0), vec3f(1.0));
  }
  if (mode == 4u) { return pbr_neutral(x); }
  if (mode == 5u) { return clamp(hable(2.0 * x) / hable(vec3f(11.2)), vec3f(0.0), vec3f(1.0)); }   // Filmic (Hable, Uncharted 2): bias 2, white 11.2
  return clamp(x, vec3f(0.0), vec3f(1.0));
}

// Hable's filmic curve (John Hable, Uncharted 2): shoulder 0.15, linear 0.5, angle 0.1, toe 0.2 / 0.02 / 0.3
fn hable(x: vec3f) -> vec3f {
  return (x * (0.15 * x + 0.05) + 0.004) / (x * (0.15 * x + 0.5) + 0.06) - 0.02 / 0.3;
}

// Khronos PBR Neutral (Khronos 3D Commerce reference): identity below ~0.76 apart from a small toe offset, so base
// colours stay true; above that the peak channel rolls off to 1 and the colour desaturates toward white
fn pbr_neutral(c: vec3f) -> vec3f {
  let start = 0.8 - 0.04; let desat = 0.15;
  let lo = min(c.r, min(c.g, c.b));
  var x = c - select(0.04, lo - 6.25 * lo * lo, lo < 0.08);
  let peak = max(x.r, max(x.g, x.b));
  if (peak < start) { return clamp(x, vec3f(0.0), vec3f(1.0)); }
  let d = 1.0 - start;
  let new_peak = 1.0 - d * d / (peak + d - start);
  x *= new_peak / peak;
  let g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0);
  return clamp(mix(x, vec3f(new_peak), g), vec3f(0.0), vec3f(1.0));
}

// AgX (Troy Sobotka's, in the common minimal form with the default look): inset into a wider gamut, log2 encode over
// [-12.47, 4.03] EV, a sigmoid contrast curve, outset and back to linear; hues hold as highlights brighten to white
fn agx(v: vec3f) -> vec3f {
  let m = mat3x3f(vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
                  vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
                  vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
  let mi = mat3x3f(vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
                   vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
                   vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
  let lo = -12.47393; let hi = 4.026069;
  var x = clamp(log2(max(m * v, vec3f(1e-10))), vec3f(lo), vec3f(hi));
  x = (x - lo) / (hi - lo);
  let x2 = x * x; let x4 = x2 * x2;
  let y = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
  return clamp(pow(max(mi * y, vec3f(0.0)), vec3f(2.2)), vec3f(0.0), vec3f(1.0));
}
