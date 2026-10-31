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
  @location(4) @interpolate(flat) grad: vec4f,   // faked refraction B: d(local)/d(pixel) rows for the x and y Mahalanobis axes
  @location(5) @interpolate(flat) flags: u32,    // SPLAT_GLASS / SPLAT_WARPED (lens.wgsl), the depth word's low bits
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
  let a0 = s.center_axis0.zw * frame.viewport * 0.5;
  let a1 = s.axis1_r_tau.xy * frame.viewport * 0.5;
  out.grad = vec4f(a0 * r / max(dot(a0, a0), 1e-12), a1 * r / max(dot(a1, a1), 1e-12));
  out.flags = s.color_depth.w & 3u;
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

// Faked refraction (?refract=1, src/refract.js): the same "over", sorted into three layers per pixel against the
// nearest lens on the eye ray: in front of it (with the near side of the glass skin, which carries the reflection),
// inside it, behind it. Two r16float targets keep each back layer's alpha-weighted view depth for refract.wgsl.
@group(0) @binding(3) var<uniform> LENSES: Lenses;
@group(0) @binding(4) var lensmap: texture_2d<f32>;   // refract.wgsl fs_lensmap: (t_near, t_far, lens + 1, 0) per pixel

struct LayersOut {
  @location(0) front: vec4f,
  @location(1) inside: vec4f,
  @location(2) behind: vec4f,
  @location(3) z_inside: vec4f,
  @location(4) z_behind: vec4f,
}

@fragment
fn fs_layers(in: VsOut) -> LayersOut {
  let e = exp(-0.5 * dot(in.local, in.local));
  let alpha = 1.0 - exp(-in.tau0 * e);
  let cov = 1.0 - exp(-in.spec.w * e);
  if (max(alpha, cov) < ALPHA_MIN) { discard; }
  var rgb = in.color.rgb;
  if (frame.view_mode == 1u) { rgb = vec3f(1.0 - in.color.a); }
  let c = vec4f(rgb * alpha + in.spec.rgb * cov, alpha);
  let depth = frame.near * pow(frame.far / frame.near, in.color.a);   // the depth key's log depth, inverted
  let d = eye_ray(in.pos.xy);
  var layer = 0u;
  var best = 1e30;
  var out: LayersOut;
  if ((LENSES.info.z & REFR_LENS) == 0u) {
    // B: glass (and with C the warped Gaussians) in front, everything else behind; glass adds its optical-depth
    // gradient (per pixel) into the inside target, blended additively by this pipeline
    let tau = in.tau0 * e;
    let gp = -tau * (in.local.x * in.grad.xy + in.local.y * in.grad.zw);
    let fr = in.flags != 0u;
    out.front = select(vec4f(0.0), c, fr);
    out.behind = select(c, vec4f(0.0), fr);
    out.inside = select(vec4f(0.0), vec4f(gp, 0.0, 0.0), (in.flags & SPLAT_GLASS) != 0u);
    out.z_inside = vec4f(0.0);
    out.z_behind = vec4f(depth * alpha * f32(!fr), 0.0, 0.0, select(alpha, 0.0, fr));
    return out;
  }
  let lm = textureLoad(lensmap, vec2i(in.pos.xy), 0);
  if (lm.z > 0.5) {
    best = lm.x;
    let margin = LENSES.params.w * (lm.y - lm.x);
    layer = select(select(2u, 1u, depth < lm.y), 0u, depth < lm.x + margin && (in.flags & SPLAT_WARPED) == 0u);
  }
  // the glass skin carries the reflection; glass well beyond this lens's far side (another marble behind it) is behind
  if ((in.flags & SPLAT_GLASS) != 0u && best < 1e30) { layer = select(0u, 2u, depth > lm.y + LENSES.params.w * (lm.y - lm.x)); }
  out.front = select(vec4f(0.0), c, layer == 0u);
  out.inside = select(vec4f(0.0), c, layer == 1u);
  out.behind = select(vec4f(0.0), c, layer == 2u);
  out.z_inside = vec4f(depth * alpha, 0.0, 0.0, select(0.0, alpha, layer == 1u));
  out.z_behind = vec4f(depth * alpha, 0.0, 0.0, select(0.0, alpha, layer == 2u));
  out.z_inside.x *= f32(layer == 1u);
  out.z_behind.x *= f32(layer == 2u);
  return out;
}

// Compact layers (the default for A + C, B and B + C): two colour targets and one aux target. A + C: everything up to the
// lens's far side in front (the warped inside Gaussians sit there at their images), the rest behind with its depth in
// aux (r16float). B / B + C: glass and warped Gaussians in front, the rest behind, the glass gradient summed in aux
// (rg16float, additive).
struct Layers2Out {
  @location(0) front: vec4f,
  @location(1) behind: vec4f,
  @location(2) aux: vec4f,
}

@fragment
fn fs_layers2(in: VsOut) -> Layers2Out {
  let e = exp(-0.5 * dot(in.local, in.local));
  let alpha = 1.0 - exp(-in.tau0 * e);
  let cov = 1.0 - exp(-in.spec.w * e);
  if (max(alpha, cov) < ALPHA_MIN) { discard; }
  var rgb = in.color.rgb;
  if (frame.view_mode == 1u) { rgb = vec3f(1.0 - in.color.a); }
  let c = vec4f(rgb * alpha + in.spec.rgb * cov, alpha);
  var out: Layers2Out;
  if ((LENSES.info.z & REFR_LENS) == 0u) {
    let tau = in.tau0 * e;
    let gp = -tau * (in.local.x * in.grad.xy + in.local.y * in.grad.zw);
    let fr = in.flags != 0u;
    out.front = select(vec4f(0.0), c, fr);
    out.behind = select(c, vec4f(0.0), fr);
    out.aux = select(vec4f(0.0), vec4f(gp, 0.0, 0.0), (in.flags & SPLAT_GLASS) != 0u);
    return out;
  }
  let depth = frame.near * pow(frame.far / frame.near, in.color.a);
  let lm = textureLoad(lensmap, vec2i(in.pos.xy), 0);
  var back = lm.z > 0.5 && depth >= lm.y + select(0.0, LENSES.params.w * (lm.y - lm.x), (in.flags & SPLAT_GLASS) != 0u);
  // G: the inside is gathered between the glass's near and far sides, so the far half of the skin goes behind (the
  // composite sees it bent, through the exit map, like the rest of what is behind)
  if ((LENSES.info.z & REFR_GATHER) != 0u && lm.z > 0.5 && depth > 0.5 * (lm.x + lm.y)) { back = true; }
  out.front = select(c, vec4f(0.0), back);
  out.behind = select(vec4f(0.0), c, back);
  out.aux = vec4f(depth * alpha * f32(back), 0.0, 0.0, select(0.0, alpha, back));
  return out;
}

// Faked refraction E: the exit map. Each front-facing glass Gaussian (exitmap.wgsl trace) adds its exit point,
// direction and transmission, weighted by its footprint, into two additive targets; refract.wgsl divides by the
// weight, so values interpolate between neighbouring Gaussians instead of the nearest one winning.
@group(0) @binding(5) var<storage, read> exits: array<vec4f>;

struct ExitVs {
  @builtin(position) pos: vec4f,
  @location(0) local: vec2f,
  @location(1) @interpolate(flat) slot: u32,
}

@vertex
fn vs_exit(@builtin(vertex_index) v: u32, @builtin(instance_index) inst: u32) -> ExitVs {
  let i = order[inst];
  let s = splats[i];
  var out: ExitVs;
  out.slot = i;
  // not on the map; LOD node splats (index >= frame.lod.y, the node base, when LOD is on) have no exit entry
  if ((frame.lod.x != 0u && i >= frame.lod.y) || (s.color_depth.w & SPLAT_GLASS) == 0u || exits[2u * i].w <= 0.0) { out.pos = vec4f(2.0, 2.0, 2.0, 1.0); return out; }
  let corner = vec2f(f32(v & 1u), f32(v >> 1u)) * 2.0 - 1.0;
  let ndc = s.center_axis0.xy + corner.x * s.center_axis0.zw + corner.y * s.axis1_r_tau.xy;
  out.pos = vec4f(ndc, 0.5, 1.0);
  var r = s.axis1_r_tau.z;   // as splat_out: with the coverage split, z holds tau_c and r follows the further of the two
  if ((u32(frame.lamp_color.w + 0.5) & REFL_SPLIT) != 0u && frame.view_mode == 3u) {
    r = sqrt(2.0 * log(max(s.axis1_r_tau.w, s.axis1_r_tau.z) / -log(1.0 - ALPHA_MIN)));
  }
  out.local = corner * r;
  return out;
}

struct ExitOut {
  @location(0) dir: vec4f,   // exit direction * w, w
  @location(1) pnt: vec4f,   // exit point * w, transmission * w
  @location(2) conf: vec4f,  // miss confidence * w, 0, 0, w
}

@fragment
fn fs_exit(in: ExitVs) -> ExitOut {
  let w = exp(-0.5 * dot(in.local, in.local));
  if (w < 0.01) { discard; }
  let a = exits[2u * in.slot];
  let b = exits[2u * in.slot + 1u];
  var out: ExitOut;
  out.dir = vec4f(b.xyz * w, w);
  out.pnt = vec4f(a.xyz * w, b.w * w);
  out.conf = vec4f(a.w * w, 0.0, 0.0, w);
  return out;
}

// G: the inside ray map, splatted like the exit map with the same weights: (entry point - lens centre) * w, leg 0
// length * w; leg 0 direction * w, w; leg 1 direction * w, leg 1 length * w.
@group(0) @binding(6) var<storage, read> inray: array<vec4f>;

struct InOut {
  @location(0) pnt: vec4f,
  @location(1) dir0: vec4f,
  @location(2) dir1: vec4f,
}

@fragment
fn fs_exitin(in: ExitVs) -> InOut {
  let w = exp(-0.5 * dot(in.local, in.local));
  if (w < 0.01) { discard; }
  let a = inray[3u * in.slot];
  let b = inray[3u * in.slot + 1u];
  let c = inray[3u * in.slot + 2u];
  var out: InOut;
  out.pnt = a * w;
  out.dir0 = vec4f(b.xyz * w, w);
  out.dir1 = c * w;
  return out;
}
