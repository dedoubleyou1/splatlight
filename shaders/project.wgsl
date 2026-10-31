// Pass 1: pose, cull and project every Gaussian; write its splat, its 16-bit
// back-to-front depth key and the key histogram for the counting sort.

@group(0) @binding(0) var<uniform> frame: RenderFrame;
// a material's mat_view entry (materials 0-31; higher ones read 31)
fn mat_view_of(m: u32) -> vec4f {
  if (m < 16u) { return frame.mat_view[m]; }
  return frame.mat_view_ext[min(m, 31u) - 16u];
}

@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> parts: array<Part>;
@group(0) @binding(3) var<storage, read_write> splats: array<Splat>;
@group(0) @binding(4) var<storage, read_write> keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> colour: array<vec4f>;   // lit radiance per Gaussian (runtime shade.wgsl)
@group(0) @binding(7) var<storage, read> lo_b: array<vec4f>;     // bounce, 3 per Gaussian: [0] toward the eye, [2] the specular lanes (runtime common.wgsl LOB)
@group(0) @binding(8) var<storage, read> emitters: array<Gaussian>;   // emissive Gaussians, world space, albedo = Le
@group(0) @binding(9) var<storage, read> lodflag: array<u32>;
@group(0) @binding(10) var<uniform> LENSES: Lenses;   // faked refraction (lens.wgsl): C warps Gaussians inside a lens   // screen-space LOD (runtime/wgsl/lod.wgsl): 2 = draw this node
@group(0) @binding(11) var<storage, read> inner_mask: array<u32>;   // G: a bit per Gaussian inside a lens part (renderer.js setGlass)

const ALPHA_MIN: f32 = 1.0 / 255.0;

// Palette slots 0-2 and 4-7 (slot 3 carries bounce gain and emitter count). Lit view: rgb is chosen / asset albedo;
// other views: rgb replaces the albedo when w > 0.5.
fn palette_slot(m: u32) -> vec4f {
  if (m < 3u) { return frame.palette[m]; }
  if (m >= 4u && m < 8u) { return frame.palette_ext[m - 4u]; }
  return vec4f(1.0, 1.0, 1.0, 0.0);
}
const TWO_PI: f32 = 6.283185307;

// Explode button (main.js): every Gaussian keeps its lit colour (the lighting never sees the move) and flies off on its
// own ballistic path from frame.explode_c, tumbling and bouncing on the floor plane, then eases back home as
// frame.explode.y goes 0 -> 1. Per Gaussian randoms from its index.
fn explode_rand(i: u32) -> vec4f {
  var h = i * 747796405u + 2891336453u;
  var r: vec4f;
  for (var k = 0; k < 4; k++) {
    h = h * 747796405u + 2891336453u;
    let w = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    r[k] = f32((w >> 22u) ^ w) / 4294967296.0;
  }
  return r;
}

// Offset from p0 (world) at flight time t, reassembly a; w = tumble angle (rad)
fn explode_offset(i: u32, p0: vec3f, t: f32, a: f32) -> vec4f {
  let h = explode_rand(i);
  let speed = frame.explode_k.x;
  let grav = frame.explode_k.y;
  let fz = min(frame.explode_c.w, p0.z);
  let dir = normalize(p0 - frame.explode_c.xyz + (h.xyz - 0.5) * 0.6 + vec3f(0.0, 0.0, 0.2));
  var v = dir * speed * (0.4 + 0.9 * h.x);
  v.z += speed * 0.5 * h.y;
  var p = p0;
  var tl = t;
  var landed = true;
  for (var k = 0; k < 6; k++) {   // bounces: restitution 0.45, floor friction 0.6
    let tau = (v.z + sqrt(v.z * v.z + 2.0 * grav * max(p.z - fz, 0.0))) / grav;
    if (tl < tau) { p += v * tl; p.z -= 0.5 * grav * tl * tl; landed = false; break; }
    p += v * tau;
    p.z = fz;
    v = vec3f(v.xy * 0.6, -(v.z - grav * tau) * 0.45);
    tl -= tau;
  }
  if (landed) { p.z = fz; }
  let spin = (h.w - 0.5) * 30.0 * (1.0 - exp(-t / 0.7));
  // reassembly: staggered per Gaussian, eased, with a lift on the way home
  let e = smoothstep(0.0, 1.0, clamp((a - 0.45 * h.w) / 0.55, 0.0, 1.0));
  let home = (p - p0) * (1.0 - e) + vec3f(0.0, 0.0, 0.5 * sin(3.14159265 * e) * min(length(p - p0), 1.0));
  return vec4f(home, spin * (1.0 - e));
}

fn axis_angle(axis: vec3f, ang: f32) -> mat3x3f {
  let c = cos(ang); let s = sin(ang); let u = axis; let k = 1.0 - c;
  return mat3x3f(
    vec3f(c + u.x * u.x * k, u.y * u.x * k + u.z * s, u.z * u.x * k - u.y * s),
    vec3f(u.x * u.y * k - u.z * s, c + u.y * u.y * k, u.z * u.y * k + u.x * s),
    vec3f(u.x * u.z * k + u.y * s, u.y * u.z * k - u.x * s, c + u.z * u.z * k));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  // Indices: scene Gaussians, one unused slot, then the host-built emissive Gaussians (palette[3].y of them). The
  // lamp is always shown by emissive Gaussians, its asset part or the host panel, in both light models.
  let i = gid.x;
  let n_emit = u32(frame.palette[3].y);
  // Screen-space LOD (?lod=<px>, frame.lod = (threshold px, node splat base, internal node count, 0)): slots from the
  // base on are the tree's internal nodes (gaussians[count + k] holds node count + k as a Gaussian, colour[2 count + k]
  // its lit colour); lod.wgsl's lod_select flags the cut, and only flagged leaves and nodes are drawn.
  let lod_on = frame.lod.x != 0u;
  let node_base = frame.lod.y;
  var node = 0xffffffffu;
  if (i > frame.count + n_emit && i < node_base) { keys[i] = INVALID_KEY; return; }   // unused emitter slots: sorted with LOD on
  keys[i] = INVALID_KEY;
  if (i == frame.count) { return; }
  // Gaussian particles: the last splat slots (frame.particles = first slot, count, first Gaussian in `gaussians`),
  // world space, drawn as emitters (their colour is final radiance from particles.wgsl), no part, no LOD
  let pbase = frame.particles.x;
  let is_particle = pbase != 0u && i >= pbase;
  if (is_particle) {
    if (i - pbase >= frame.particles.y) { return; }
  } else if (i >= node_base) {
    if (!lod_on || i - node_base >= frame.lod.z) { return; }
    node = frame.count + (i - node_base);
    if (lodflag[node] != 2u) { return; }
  } else if (lod_on && i < frame.count && lodflag[i] != 2u) { return; }

  var g: Gaussian;
  var mu: vec3f;
  var rot: mat3x3f;
  var material = EMITTER_MATERIAL;
  var vel = vec3f(0.0);   // motion blur: world-space displacement over the exposure (zero when off or still)
  var keep = 1.0;         // 1 - frame.part_fade: draw-time fade per part (drifting clouds)
  if (is_particle) {
    g = gaussians[frame.particles.z + (i - pbase)];
    if (g.kappa <= 0.0) { return; }
    mu = g.mu;
    rot = quat_to_mat(g.quat);
    material = EMITTER_MATERIAL;
  } else if (i > frame.count && node == 0xffffffffu) {
    g = emitters[i - frame.count - 1u];
    mu = g.mu;
    rot = quat_to_mat(g.quat);
    material = EMITTER_MATERIAL;
  } else {
    // Pose: the part's rigid transform on the centre and the rotation.
    g = gaussians[select(i, node, node != 0xffffffffu)];
    let part = min(g.part_surface & 0xffffu, MAX_PARTS - 1u);
    let p = parts[part];
    if ((p.flags & PART_HIDDEN) != 0u) { return; }   // a parked spare (runtime/add_spares.py): not drawn until shown
    keep = 1.0 - clamp(frame.part_fade[part / 4u][part % 4u], 0.0, 1.0);
    if (keep <= 0.0) { return; }
    mu = part_apply(p, g.mu);
    rot = part_rotation(p) * quat_to_mat(g.quat);
    // Motion blur (?blur=1, palette[3].z = shutter fraction): a Gaussian moving along d over the exposure, averaged
    // over a box shutter, is to second order a Gaussian with covariance + d d^T / 12 centred mid-exposure. The
    // exposure ends at this frame's pose; d comes from xf_prev (the chord, so rotation is per Gaussian).
    let shutter = frame.palette[3].z;
    if (shutter > 0.0 && (p.flags & 2u) != 0u) {
      let h = vec4f(g.mu, 1.0);
      vel = shutter * (mu - vec3f(dot(p.xf_prev[0], h), dot(p.xf_prev[1], h), dot(p.xf_prev[2], h)));
      mu -= 0.5 * vel;
    }
    if (part < 16u) { material = frame.part_material[part / 4u][part % 4u]; }
    else { material = frame.part_material_ext[(part - 16u) / 4u][part % 4u]; }
  }
  if (frame.explode_k.z > 0.5 && !is_particle) {
    // Explode button: the move since the previous frame joins the motion blur
    let o = explode_offset(i, mu, frame.explode.x, frame.explode.y);
    let o_prev = explode_offset(i, mu, frame.explode.z, frame.explode.w);
    let dv = frame.palette[3].z * (o.xyz - o_prev.xyz);
    mu += o.xyz - 0.5 * dv;
    vel += dv;
    rot = axis_angle(normalize(explode_rand(i ^ 0x9e3779b9u).xyz - 0.5 + vec3f(1e-3)), o.w) * rot;
  }

  var tv = frame.view * vec4f(mu, 1.0);
  // Faked refraction (src/refract.js). Glass (coverage materials) is flagged for the splat pass; with C on, a Gaussian
  // whose centre is inside a lens is mapped to its image through the lens's near surface (lens.wgsl lens_warp) by the
  // unscented transform (3DGUT, docs/references.bib): its six sigma points mu +- sqrt(3) a_j (a_j the scaled view-space
  // axes) go through the map and the image Gaussian is their mean and covariance, so it stretches per axis as the map
  // does, not by one magnification. Optical depth (tau0) is kept, so it is as opaque as before.
  var splat_flags = 0u;
  let mvf = u32(mat_view_of(g.flags & 0xffffu).w + 0.5);
  if ((mvf & 1u) != 0u && material != EMITTER_MATERIAL) { splat_flags = SPLAT_GLASS; }
  var warped_cov = mat3x3f();
  // G: a Gaussian on the inside list keeps its splat (the composite reads its colour) but is not drawn (below); LOD
  // node slots (>= node_base > count) have no bit
  let gathered = (LENSES.info.z & REFR_GATHER) != 0u && i < frame.count && (inner_mask[i >> 5u] & (1u << (i & 31u))) != 0u;
  if ((LENSES.info.z & REFR_WARP) != 0u && splat_flags == 0u && i < frame.count && material != EMITTER_MATERIAL) {
    for (var k = 0u; k < min(LENSES.info.x, 8u); k++) {
      let l = LENSES.lens[k];
      let u = lens_n(l, tv.xyz - l.c.xyz);
      if (dot(u, u) >= 1.0) { continue; }
      let n = LENSES.params.x;
      let minv = inv3(lens_mat(l));
      let wr = mat3_of(frame.view) * rot;
      var ys: array<vec3f, 6>;
      var mean = vec3f(0.0);
      for (var j = 0; j < 3; j++) {
        let a = wr[j] * (g.scale[j] * 1.7320508);
        ys[2 * j] = lens_warp(l, minv, tv.xyz + a, n);
        ys[2 * j + 1] = lens_warp(l, minv, tv.xyz - a, n);
        mean += ys[2 * j] + ys[2 * j + 1];
      }
      mean *= 1.0 / 6.0;
      for (var j = 0; j < 6; j++) {
        let e = ys[j] - mean;
        warped_cov += mat3x3f(e * e.x, e * e.y, e * e.z) * (1.0 / 6.0);
      }
      tv = vec4f(mean, 1.0);
      splat_flags = SPLAT_WARPED;
      break;
    }
  }
  let depth = -tv.z;
  if (depth < frame.near || depth > frame.far) { return; }

  // 3D covariance in view space: W R S^2 R^T W^T.
  let wr = mat3_of(frame.view) * rot;
  let s2 = g.scale * g.scale;
  let m = mat3x3f(wr[0] * g.scale.x, wr[1] * g.scale.y, wr[2] * g.scale.z);
  var cov = m * transpose(m);
  if ((splat_flags & SPLAT_WARPED) != 0u) { cov = warped_cov; }
  let cov_s = cov;   // the sharp footprint, for the coverage cap below (on by default, ?blurcov=0 turns it off)
  let blurred = any(vel != vec3f(0.0));
  if (blurred) { let wv = mat3_of(frame.view) * vel; cov += mat3x3f(wv * wv.x, wv * wv.y, wv * wv.z) * (1.0 / 12.0); }

  // Perspective Jacobian (EWA), pixel units, y up.
  let fx = frame.focal.x; let fy = frame.focal.y;
  let iz = 1.0 / depth;
  let px = fx * tv.x * iz;
  let py = fy * tv.y * iz;
  // Clamp the tangent like 3DGS so the Jacobian stays sane off-screen.
  let lim = 1.3 * frame.viewport * 0.5;
  let cx = clamp(tv.x * iz, -lim.x / fx, lim.x / fx) * depth;
  let cy = clamp(tv.y * iz, -lim.y / fy, lim.y / fy) * depth;
  let j0 = vec3f(fx * iz, 0.0, fx * cx * iz * iz);
  let j1 = vec3f(0.0, fy * iz, fy * cy * iz * iz);
  let cj0 = cov * j0;
  let cj1 = cov * j1;
  var a = dot(j0, cj0);
  let b = dot(j0, cj1);
  var c = dot(j1, cj1);
  let det0 = a * c - b * b;
  a += frame.lowpass_px2;
  c += frame.lowpass_px2;
  let det = a * c - b * b;
  if (det <= 0.0) { return; }

  // Volumetric opacity: peak optical depth through the centre along the view
  // ray, tau0 = kappa sqrt(2 pi / (d^T C^-1 d)); the low-pass dilation keeps
  // the integrated optical depth (3DGS-style compensation).
  let d = normalize(tv.xyz);
  let dl = transpose(wr) * d;
  var q = dot(dl * dl, 1.0 / s2);
  let q_s = q;
  var kappa = g.kappa;
  if (blurred) {
    // Sigma' = Sigma + v v^T / 12 in the Gaussian's own frame (Sigma = diag(s2)): Sherman-Morrison for the view
    // ray's d^T Sigma'^-1 d, and the determinant lemma for the density that keeps the mass, kappa / sqrt(1 + m / 12)
    let vl = transpose(rot) * vel;
    let mv2 = dot(vl * vl, 1.0 / s2);
    let dv = dot(dl * vl, 1.0 / s2);
    q -= dv * dv / (12.0 + mv2);
    kappa /= sqrt(1.0 + mv2 / 12.0);
  }
  // the sharp footprint's determinants (same Jacobian), for the coverage caps below; det0 / det when not blurred
  var det0_s = det0;
  var det_s = det;
  if (blurred) {
    let a_s = dot(j0, cov_s * j0) + frame.lowpass_px2;
    let b_s = dot(j0, cov_s * j1);
    let c_s = dot(j1, cov_s * j1) + frame.lowpass_px2;
    det0_s = (a_s - frame.lowpass_px2) * (c_s - frame.lowpass_px2) - b_s * b_s;
    det_s = a_s * c_s - b_s * b_s;
  }
  var tau0 = kappa * sqrt(TWO_PI / q) * sqrt(max(det0, 0.0) / det);
  if (blurred && frame.palette[3].w > 0.5) {
    // coverage cap, on by default, ?blurcov=0 turns it off (palette[3].w): a box shutter averages coverage, not optical depth, so cap the peak alpha at the
    // sharp splat's (same Jacobian) x sqrt(det_s / det); fades the smear ends
    let tau_s = g.kappa * sqrt(TWO_PI / q_s) * sqrt(max(det0_s, 0.0) / det_s);
    let ap = (1.0 - exp(-tau_s)) * sqrt(det_s / det);
    tau0 = min(tau0, -log(max(1.0 - ap, 1e-6)));
  }
  // Reflection coverage (experiments/reflection_coverage; lit view, scene Gaussians, not the camera-only flakes): the
  // specular layer blends by its own coverage, 1 - exp(-tau_c exp(-r^2/2)) with tau_c from the material's peak
  // coverage (mat_view.x, view-independent; unset = alpha, tau_c = tau0). Alpha alone stays the lighting's opacity.
  let refl = u32(frame.lamp_color.w + 0.5);
  let lit_view = refl != 0u && frame.view_mode == 3u && frame.lamp.w < 0.5;
  let lit_refl = lit_view && i < frame.count;   // the specular layer's inputs exist for leaves only
  // the split splat format is the frame's, not the splat's: LOD node splats (i >= node_base) write it too, with no
  // specular layer (2026-10-29: they wrote the unsplit format, which the splat pass read as packed halves -> NaN
  // outlines on the marbles at the 2 px default LOD)
  let split = lit_view && (refl & REFL_SPLIT) != 0u;
  let mv = mat_view_of(g.flags & 0xffffu);
  let mv_flags = u32(mv.w + 0.5);   // bit 0 coverage set, bit 1 no view reflection (emissive)
  tau0 *= keep;
  var tau_c = tau0;
  if (split && (mv_flags & 1u) != 0u) {
    // the material's peak coverage on the sharp footprint; blurred, spread like alpha's cap (peak x sqrt(det_s / det))
    // whatever ?blurcov says, since the coverage has no mass-keeping of its own: a smear at the sharp peak along its
    // whole length added sqrt(1 + m / 12) times the reflection while alpha stayed put (white rims on moving glass)
    let tau_m = -log(1.0 - min(mv.x, 0.999)) * sqrt(max(det0_s, 0.0) / det_s);
    tau_c = keep * select(tau_m, -log(max(1.0 - (1.0 - exp(-tau_m)) * sqrt(det_s / det), 1e-6)), blurred);
  }
  // alpha = 1 - exp(-tau0 exp(-r^2/2)) >= ALPHA_MIN  =>  r^2 <= 2 ln(tau0 / -ln(1 - ALPHA_MIN)); with coverage the
  // quad covers whichever of alpha and coverage reaches further
  let tau_min = -log(1.0 - ALPHA_MIN);
  let tau_cut = max(tau0, tau_c);
  if (tau_cut <= tau_min) { return; }
  let r = sqrt(2.0 * log(tau_cut / tau_min));

  // Eigen-decomposition of the 2x2 covariance, stable for long thin splats (motion blur): no mid^2 - det
  // cancellation, and the eigenvector from the larger diagonal term
  let mid = 0.5 * (a + c);
  let disc = sqrt(0.25 * (a - c) * (a - c) + b * b);
  let l0 = mid + disc;
  let l1 = max(det / l0, 1e-6);
  var e0 = vec2f(1.0, 0.0);
  if (a >= c) { if (abs(b) > 0.0 || l0 > a) { e0 = normalize(vec2f(l0 - c, b)); } }
  else { e0 = normalize(vec2f(b, l0 - a)); }
  let e1 = vec2f(-e0.y, e0.x);
  let to_ndc = 2.0 / frame.viewport;
  let axis0 = e0 * sqrt(l0) * r * to_ndc;
  let axis1 = e1 * sqrt(l1) * r * to_ndc;
  // Quads below a tenth of a pixel contribute nothing visible.
  if (sqrt(l0) * r < 0.1) { return; }

  let center = vec2f(px, py) * to_ndc;
  let extent = abs(axis0) + abs(axis1);
  if (any(abs(center) - extent > vec2f(1.0))) { return; }

  var color = g.albedo;
  var spec = vec3f(0.0);   // the specular layer (reflection coverage / Fresnel), already scaled like color
  if (material == EMITTER_MATERIAL) {
    color = g.albedo;   // emitted radiance; shown in every view as itself
  } else if (frame.view_mode == 3u) {
    // Lit: shading is linear in albedo, so the palette holds chosen / asset albedo.
    // palette[3].x = bounce gain; lamp.w = 1: colour is the camera-only flake colour (flake.js), bounce included
    let gain = select(frame.palette[3].x, 0.0, frame.lamp.w > 0.5);
    if (node != 0xffffffffu) { color = colour[2u * frame.count + (node - frame.count)].rgb; }   // bounce included (lod.wgsl)
    else { color = colour[i].rgb + gain * lo_b[i * 3u].rgb; }
    var sheen = vec3f(0.0);
    if (lit_refl && (mv_flags & 2u) == 0u) {
      // specular toward the eye: direct (colour's second half, emissive.wgsl) + bounce (lo_b lanes,
      // bounce_dirs); for matte materials those lanes hold the probe sheen instead (bounce_sh, Fresnel only)
      let lanes = gain * lo_b[i * 3u + 2u].rgb;
      let k = mv.z;
      spec = colour[frame.count + i].rgb + select(vec3f(0.0), lanes, k > 0.0);
      if ((refl & REFL_FRESNEL) != 0u) {
        // view-only Schlick, eye against the short axis, two-sided. Coverage materials (glass): F scales the layer
        // (F0 = mat_view.y); other shiny ones: F(k) / k, so head-on keeps today's look and grazing rises to a mirror;
        // matte: the sheen at F(F0)
        let x = pow(1.0 - abs(dot(d, normalize(wr[0]))), 5.0);
        let f0 = mv.y;
        if ((mv_flags & 1u) != 0u) { spec *= f0 + (1.0 - f0) * x; }
        else if (k > 0.0) { spec *= (k + (1.0 - k) * x) / k; }
        if (k <= 0.0) { sheen = lanes * (f0 + (1.0 - f0) * x); }
      }
    }
    if (material < 8u) { color *= palette_slot(material).rgb; spec *= palette_slot(material).rgb; }
    spec = spec * frame.albedo_scale + sheen;   // the sheen is uncoloured: neither palette nor albedo scale
  } else if (material < 8u && palette_slot(material).w > 0.5) {
    color = palette_slot(material).rgb;
  }
  if (material != EMITTER_MATERIAL) { color *= frame.albedo_scale; }

  if (split) {
    let packed = vec4u(pack2x16float(color.rg), pack2x16float(vec2f(color.b, spec.r)), pack2x16float(spec.gb), (bitcast<u32>(depth) & ~3u) | splat_flags);
    splats[i] = Splat(vec4f(center, axis0), vec4f(axis1, tau_c, tau0), packed);
  } else {
    // Fresnel without the coverage split: the specular layer rides on alpha as today
    splats[i] = Splat(vec4f(center, axis0), vec4f(axis1, r, tau0), bitcast<vec4u>(vec4f(color + spec, bitcast<f32>((bitcast<u32>(depth) & ~3u) | splat_flags))));
  }

  // G: its key stays invalid
  if (gathered) { return; }

  // 16-bit log-depth key, inverted so an ascending sort draws far to near. Clamped to 0xFFFE so the stable radix sort
  // (radix.wgsl) can put INVALID_KEY (digit 0xFF in both passes) after every valid key; this merges the nearest
  // bucket into the next one, a sub-millimetre loss at the near plane.
  let t = clamp(log(depth / frame.near) / log(frame.far / frame.near), 0.0, 1.0);
  let key = min(65535u - u32(t * 65535.0 + 0.5), 0xfffeu);
  keys[i] = key;
  atomicAdd(&histogram[key], 1u);
}
