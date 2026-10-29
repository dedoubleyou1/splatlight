// Pass 5: resolve the linear HDR target to the swapchain (exposure, sRGB
// encode), or map the overdraw count to a heat ramp.

@group(0) @binding(0) var<uniform> frame: RenderFrame;
@group(0) @binding(1) var hdr: texture_2d<f32>;

// The environment (runtime/SPEC.md 2.9; runtime/wgsl/environment.wgsl): in the lit view the sky and the distant lights'
// discs fill what the splats leave uncovered (the target's alpha is coverage).
struct EnvLight { dir_std: vec4f, irr: vec4f }
struct Env { sh: array<vec4f, 16>, lights: array<EnvLight, 4>, info: vec4u }
@group(0) @binding(2) var<uniform> ENV: Env;

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

// the distant lights as seen: an ASG of the light's irradiance spread over its angular size
fn lights_radiance(w: vec3f) -> vec3f {
  var c = vec3f(0.0);
  for (var k = 0u; k < min(ENV.info.x, 4u); k++) {
    let L = ENV.lights[k];
    let s2 = max(L.dir_std.w * L.dir_std.w, 1e-8);
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
    let w = view_dir(pos.xy);
    rgb += (1.0 - clamp(c.a, 0.0, 1.0)) * (sky_radiance(w) * f32(ENV.info.y != 0u) + lights_radiance(w));
  }
  return vec4f(srgb_encode(clamp(rgb * frame.exposure, vec3f(0.0), vec3f(1.0))), 1.0);
}
