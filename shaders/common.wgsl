// Shared declarations, prepended to every renderer shader (WGSL has no
// #include; the browser and the native shell both concatenate this file).
// Buffer layouts follow runtime/SPEC.md v0.1.

// SPEC §2.1, 64 B, uploaded straight from gaussians.bin.
struct Gaussian {
  mu: vec3f,           // centre, part rest frame
  kappa: f32,          // peak extinction scale 1/m
  quat: vec4f,         // x, y, z, w; C = R diag(scale^2) R^T
  scale: vec3f,        // std devs, ascending (x = short axis)
  part_surface: u32,   // part | surface << 16
  albedo: vec3f,       // linear single-scatter albedo
  flags: u32,
}

// SPEC §2.4, 128 B, from parts.bin; the host rewrites xf / xf_prev / flags bit 1 each frame.
struct Part {
  xf: array<vec4f, 3>,       // world_from_rest, rows [R row | t]
  xf_prev: array<vec4f, 3>,
  bound: vec4f,
  root: u32,
  g_lo: u32,
  g_hi: u32,
  flags: u32,
}

// Renderer-only: one projected splat, 48 B. Quad axes are the 2D covariance
// eigenvectors scaled to the opacity-aware cutoff radius, in NDC units.
struct Splat {
  center_axis0: vec4f,  // centre.xy (NDC), axis0.xy (NDC)
  axis1_r_tau: vec4f,   // axis1.xy (NDC), r (Mahalanobis cutoff), tau0 (peak optical depth)
  color_depth: vec4u,   // bitcast linear rgb, bitcast view depth (m). Coverage split (lit, REFL_SPLIT): pack2x16float
                        // (diffuse rg), (diffuse b, specular r), (specular gb), depth, and axis1_r_tau.z = tau_c, the
                        // coverage's peak optical depth (the vertex shader rederives r)
}

const NUM_BUCKETS: u32 = 65536u;   // one 16-bit depth key
const INVALID_KEY: u32 = 0xffffffffu;
const MAX_PARTS: u32 = 32u;
const PART_HIDDEN: u32 = 4u;   // Part.flags bit 2 (SPEC 2.4): hidden, not drawn
const EMITTER_MATERIAL: u32 = 254u;  // emissive Gaussians (runtime/wgsl/emissive.wgsl)

// Renderer-only per-frame uniform (the lighting kernels' FrameUniforms is SPEC §3.1).
struct RenderFrame {
  view: mat4x4f,
  proj: mat4x4f,
  palette: array<vec4f, 4>,         // material albedo override (white, red, green); [3] = (bounce gain, emitter count, blur shutter, blur coverage cap)
  part_material: array<vec4u, 4>,   // palette slot per part (parts 0-15; 16-31 in part_material_ext)
  lamp: vec4f,                      // xyz lamp position (unused by the shaders now), w = 1: SGGX flakes (camera)
  lamp_color: vec4f,                // linear rgb, w = reflection flags: 1 coverage split, 2 Fresnel (+ probe sheen)
  viewport: vec2f,                  // pixels
  focal: vec2f,                     // pixels
  near: f32,
  far: f32,
  count: u32,                       // Gaussians in the asset (index `count` is an unused slot)
  view_mode: u32,                   // 0 flat albedo, 1 depth key, 2 overdraw, 3 lit
  albedo_scale: f32,
  flake_roughness: f32,             // carried for the shading pass; flat mode ignores it
  exposure: f32,
  lowpass_px2: f32,                 // screen-space dilation added to the 2D covariance
  palette_ext: array<vec4f, 4>,     // palette slots 4-7 (same use as palette[0..2]; slot 3 is not a colour)
  mat_view: array<vec4f, 16>,       // per material (Gaussian flags low 16 bits): coverage peak, F0, specular weight k,
                                    // 1 = coverage set (else coverage = alpha); experiments/reflection_coverage
  part_material_ext: array<vec4u, 4>,   // palette slot per part, parts 16-31
  lod: vec4u,                       // screen-space LOD: threshold px (f32 bits, 0 = off), node splat base, internal nodes
                                    // (u32: f32 bits of small ints are denormals, which the GPU may flush)
  explode: vec4f,                   // Explode button (project.wgsl): flight time s, reassembly 0-1, and the previous
                                    // frame's two (motion blur)
  explode_c: vec4f,                 // burst centre xyz, floor z
  explode_k: vec4f,                 // speed m/s, gravity m/s^2, on (> 0.5; 0 = off), unused
  mat_view_ext: array<vec4f, 16>,   // mat_view for materials 16-31 (cornell_sdf_fx's ice is 16, the jellyfish bell 19)
  tone: vec4u,                      // x: display tone curve (display.wgsl: 0 clamp, 1 AgX, 2 ACES fit, 3 Reinhard on luminance)
  part_fade: array<vec4f, 8>,       // per part (0-31), draw time only: 0 shown (the default), 1 gone; project.wgsl scales the
                                    // splat's optical depth by 1 - fade (drifting clouds wrapping round, main.js cloudPose)
  particles: vec4u,                 // Gaussian particles (src/particles.js): x = first splat slot (0 = none), y = count,
                                    // z = their first index in the gaussians buffer
}

const REFL_SPLIT: u32 = 1u;     // RenderFrame.lamp_color.w bits (u32(w + 0.5))
const REFL_FRESNEL: u32 = 2u;


fn quat_to_mat(q: vec4f) -> mat3x3f {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}

fn mat3_of(m: mat4x4f) -> mat3x3f {
  return mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
}

// Rows [R row | t] -> column-major rotation.
fn part_rotation(p: Part) -> mat3x3f {
  return transpose(mat3x3f(p.xf[0].xyz, p.xf[1].xyz, p.xf[2].xyz));
}

fn part_apply(p: Part, x: vec3f) -> vec3f {
  let h = vec4f(x, 1.0);
  return vec3f(dot(p.xf[0], h), dot(p.xf[1], h), dot(p.xf[2], h));
}
