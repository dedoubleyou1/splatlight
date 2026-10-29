// Pass 4: instanced quads, one per visible splat, blended back to front into
// an rgba16float target with premultiplied "over". In overdraw mode the same
// quads add 1 per fragment above alpha 1/255 (M29's fragment count).

@group(0) @binding(0) var<uniform> frame: RenderFrame;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> order: array<u32>;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) local: vec2f,                      // Mahalanobis offset from the centre
  @location(1) @interpolate(flat) tau0: f32,
  @location(2) @interpolate(flat) color: vec4f,  // rgb, depth key in [0, 1]
  @location(3) @interpolate(flat) spec: vec4f,   // coverage split: specular rgb, tau_c (coverage peak optical depth)
}

const ALPHA_MIN: f32 = 1.0 / 255.0;

// Splat -> the varyings both vertex shaders share; `corner` in [-1, 1]^2.
fn splat_out(s: Splat, corner: vec2f) -> VsOut {
  let ndc = s.center_axis0.xy + corner.x * s.center_axis0.zw + corner.y * s.axis1_r_tau.xy;
  var out: VsOut;
  out.pos = vec4f(ndc, 0.5, 1.0);
  out.tau0 = s.axis1_r_tau.w;
  let depth = bitcast<f32>(s.color_depth.w);
  let t = log(depth / frame.near) / log(frame.far / frame.near);
  var r = s.axis1_r_tau.z;
  out.spec = vec4f(0.0);
  if ((u32(frame.lamp_color.w + 0.5) & REFL_SPLIT) != 0u && frame.view_mode == 3u) {
    // coverage split (project.wgsl): half-float diffuse and specular; r follows the further of alpha and coverage
    let a = unpack2x16float(s.color_depth.x);
    let b = unpack2x16float(s.color_depth.y);
    let c = unpack2x16float(s.color_depth.z);
    let tau_c = s.axis1_r_tau.z;
    r = sqrt(2.0 * log(max(out.tau0, tau_c) / -log(1.0 - ALPHA_MIN)));
    out.color = vec4f(a, b.x, clamp(t, 0.0, 1.0));
    out.spec = vec4f(b.y, c, tau_c);
  } else {
    out.color = vec4f(bitcast<vec3f>(s.color_depth.xyz), clamp(t, 0.0, 1.0));
  }
  out.local = corner * r;
  return out;
}

@vertex
fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) inst: u32) -> VsOut {
  let corner = vec2f(f32(v & 1u), f32(v >> 1u)) * 2.0 - 1.0;
  return splat_out(splats[order[inst]], corner);
}

// Non-instanced variant: 6 vertices per splat in one triangle list
// (draw args vertexCount = 6 x visible, instanceCount = 1).
const LIST_CORNERS = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1));

@vertex
fn vs_list(@builtin(vertex_index) v: u32) -> VsOut {
  return splat_out(splats[order[v / 6u]], LIST_CORNERS[v % 6u]);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  let e = exp(-0.5 * dot(in.local, in.local));
  let alpha = 1.0 - exp(-in.tau0 * e);
  // coverage split: rgb = diffuse alpha + specular coverage, alpha out = alpha only, so a Gaussian with alpha 0 and
  // coverage 1 adds its reflection and hides nothing (spec.w = tau_c is 0 otherwise)
  let cov = 1.0 - exp(-in.spec.w * e);
  if (max(alpha, cov) < ALPHA_MIN) { discard; }
  if (frame.view_mode == 2u) { return vec4f(1.0, 0.0, 0.0, 0.0); }
  var rgb = in.color.rgb;
  if (frame.view_mode == 1u) { rgb = vec3f(1.0 - in.color.a); }
  return vec4f(rgb * alpha + in.spec.rgb * cov, alpha);
}
