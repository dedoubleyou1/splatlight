// Shiny materials (renderer shell, 2026-10-23; proposed for SPEC 2.6, not merged yet). Prepend after common.wgsl.
//
// A material with flags bit 3 set mixes M36's diffuse flakes with specular (mirror) flakes oriented by the same SGGX
// matrix S = R diag(flake) R^T (x = the Gaussian's short axis): `flake = (1, r^2, r^2)` gives roughness r. The mix
// weight k (0 = matte, 1 = all mirror flakes) is pack2x16float(k, 0) in `pad0`, which only emissive materials use
// otherwise. The specular flake phase is Heitz et al. 2015: p(wi -> wo) = D(wh) / (4 sigma(wi)), wh = the half
// vector, sigma(w) = sqrt(w^T S w), D(w) = 1 / (pi sqrt(det S) (w^T S^-1 w)^2). It is normalised over wo, so the
// mix keeps energy.

const MAT_SPECULAR: u32 = 8u;

fn sym_det(d: vec3f, o: vec3f) -> f32 {
    return d.x * (d.y * d.z - o.z * o.z) - o.x * (o.x * d.z - o.z * o.y) + o.y * (o.x * o.z - d.y * o.y);
}

// w^T S^-1 w det S, via the adjugate (no division)
fn sym_adj_quad(d: vec3f, o: vec3f, w: vec3f) -> f32 {
    let a00 = d.y * d.z - o.z * o.z;
    let a11 = d.x * d.z - o.y * o.y;
    let a22 = d.x * d.y - o.x * o.x;
    let a01 = o.y * o.z - o.x * d.z;
    let a02 = o.x * o.z - o.y * d.y;
    let a12 = o.x * o.y - d.x * o.z;
    return a00 * w.x * w.x + a11 * w.y * w.y + a22 * w.z * w.z + 2.0 * (a01 * w.x * w.y + a02 * w.x * w.z + a12 * w.y * w.z);
}

fn phase_specular(Sd: vec3f, So: vec3f, wi: vec3f, wo: vec3f) -> f32 {
    let h = wi + wo;
    let hl = length(h);
    if (hl < 1e-6) { return 0.0; }
    let wh = h / hl;
    let det = max(sym_det(Sd, So), 1e-20);
    let q = sym_adj_quad(Sd, So, wh) / det;                 // wh^T S^-1 wh
    let D = 1.0 / (PI * sqrt(det) * q * q);
    let sigma = sqrt(max(dot(wi, sym_mul(Sd, So, wi)), 1e-12));
    return D / (4.0 * sigma);
}

fn spec_weight(mat: Material) -> f32 {
    if ((mat.flags & MAT_SPECULAR) == 0u || (mat.flags & 1u) != 0u) { return 0.0; }
    return clamp(unpack2x16float(mat.pad0).x, 0.0, 1.0);
}

// M39 direction-independent transfer with shine: the colour is alb iso_norm sum L V A x iso_lobe (common.wgsl
// iso_norm), iso_lobe = (1 - k) + k 4 pi p_spec damp; 1 for a matte material, so the diffuse part keeps one colour for
// every direction and only the specular flakes add a view-dependent lobe. damp = iso_damp: the lobe's energy is
// displayed over the Gaussian's visible area around the mirror direction w_r, A(w_r) V(w_r), not the mean S / 4
// (that made face-on highlights up to 4x too bright); never brighter than the S / 4 form, and independent of w_o.
fn iso_lobe(mat: Material, Sd: vec3f, So: vec3f, wi: vec3f, wo: vec3f, damp: f32) -> f32 {
    let k = spec_weight(mat);
    if (k <= 0.0) { return 1.0; }
    return iso_lobe_from(k, phase_specular(Sd, So, wi, wo), damp);
}

// the same with the specular phase given (emissive.wgsl asg_phase: a light with a shape)
fn iso_lobe_from(k: f32, p_spec: f32, damp: f32) -> f32 {
    return (1.0 - k) + k * 4.0 * PI * p_spec * damp;
}

// min(1, (S / 4) / (A(w_r) V(w_r))) = min(1, 1 / (4 pi iso_norm A(w_r) V(w_r))), w_r = wi mirrored about the short axis n
fn iso_damp(inorm: f32, area_vis_r: f32) -> f32 {
    return min(1.0, 1.0 / max(4.0 * PI * inorm * area_vis_r, 1e-30));
}

fn mirror(wi: vec3f, n: vec3f) -> vec3f {
    return 2.0 * dot(wi, n) * n - wi;
}

fn phase_ex(mat: Material, n: vec3f, Sd: vec3f, So: vec3f, wi: vec3f, wo: vec3f) -> f32 {
    let k = spec_weight(mat);
    var p = phase(mat, n, Sd, So, wi, wo);
    if (k > 0.0) { p = mix(p, phase_specular(Sd, So, wi, wo), k); }
    return p;
}

// ---- ASG specular (experiments/P3_asg_light_projection) -------------------------------------------------------
// A light with a shape in the specular lobe: emissive.wgsl (cut nodes of the emitter tree) and bounce.wgsl (probe
// direction cells) pass the light's angular covariance; see the emissive.wgsl section for the derivation.

const ASG_W = vec3f(0.61163, 0.35083, 0.0267);    // (1 + q)^-2 ~ sum w_k exp(-a_k q) (P3 fitlobe.py)
const ASG_A = vec3f(2.54696, 0.6893, 0.10727);

// orthonormal tangent pair for unit n (Duff et al. 2017)
fn onb(n: vec3f) -> array<vec3f, 2> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return array<vec3f, 2>(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y));
}

// symmetric 2x2 matrices as (xx, xy, yy)
fn s2_det(s: vec3f) -> f32 { return s.x * s.z - s.y * s.y; }
fn s2_quad(s: vec3f, v: vec2f) -> f32 { return s.x * v.x * v.x + 2.0 * s.y * v.x * v.y + s.z * v.y * v.y; }
fn s2_quad_inv(s: vec3f, v: vec2f) -> f32 {
    return (s.z * v.x * v.x - 2.0 * s.y * v.x * v.y + s.x * v.y * v.y) / max(s2_det(s), 1e-30);
}
// T S T^T for T with rows t0, t1
fn s2_xf(s: vec3f, t0: vec2f, t1: vec2f) -> vec3f {
    let a = vec2f(s.x * t0.x + s.y * t0.y, s.y * t0.x + s.z * t0.y);
    let b = vec2f(s.x * t1.x + s.y * t1.y, s.y * t1.x + s.z * t1.y);
    return vec3f(dot(t0, a), dot(t0, b), dot(t1, b));
}
// (unit major axis, largest eigenvalue)
fn s2_major(s: vec3f) -> vec3f {
    let h = 0.5 * (s.x - s.z);
    let lam = 0.5 * (s.x + s.z) + sqrt(h * h + s.y * s.y);
    let v = select(vec2f(s.y, lam - s.x), vec2f(lam - s.z, s.y), s.x >= s.z);
    let l = length(v);
    return vec3f(select(vec2f(1.0, 0.0), v / l, l > 1e-20), lam);
}
fn s2_minor(s: vec3f) -> f32 {
    let h = 0.5 * (s.x - s.z);
    return 0.5 * (s.x + s.z) - sqrt(h * h + s.y * s.y);
}

// unit a in the tangent plane at c (basis cb): geodesic angle times direction
fn log_map(c: vec3f, cb: array<vec3f, 2>, a: vec3f) -> vec2f {
    let ca = clamp(dot(c, a), -1.0, 1.0);
    let v = a - ca * c;
    let nv = length(v);
    if (nv < 1e-9) { return vec2f(0.0); }
    return acos(ca) / nv * vec2f(dot(cb[0], v), dot(cb[1], v));
}

// int L(w) p(w -> wo) dw / int L(w) dw for a light ASG centred on wi with covariance sl in basis lb (rest frame).
// n0, t1, t2: the receiver's axes (S's eigenvectors, flake = their eigenvalues; the flake normal is n0).
fn asg_phase(wi: vec3f, lb: array<vec3f, 2>, sl: vec3f, n0: vec3f, t1: vec3f, t2: vec3f, flake: vec3f,
             Sd: vec3f, So: vec3f, wo: vec3f) -> f32 {
    let pc = phase_specular(Sd, So, wi, wo);
    if (flake.x <= max(flake.y, flake.z) * 1.0001) { return pc; }   // not a flake lobe (fibre-like S): keep the point
    // the lobe: Sigma_k = J A^-1 J^T / (2 a_k), A = lam_n S_T^-1 - I, J the mirror map's Jacobian at the normal
    let co = dot(n0, wo);
    let n = select(-n0, n0, co >= 0.0);
    let cos_o = abs(co);
    let wr = 2.0 * cos_o * n - wo;
    let rb = onb(wr);
    let dw1 = 2.0 * dot(wo, t1) * n + 2.0 * cos_o * t1;
    let dw2 = 2.0 * dot(wo, t2) * n + 2.0 * cos_o * t2;
    let j0 = vec2f(dot(rb[0], dw1), dot(rb[0], dw2));
    let j1 = vec2f(dot(rb[1], dw1), dot(rb[1], dw2));
    let ia = vec2f(flake.y / (flake.x - flake.y), flake.z / (flake.x - flake.z));
    let s0 = vec3f(dot(j0 * j0, ia), dot(j0 * j1, ia), dot(j1 * j1, ia));
    let pk = flake.x * flake.x / (PI * sqrt(flake.x * flake.y * flake.z)) /
             (4.0 * sqrt(max(dot(wr, sym_mul(Sd, So, wr)), 1e-12)));
    // both into the tangent plane at their midpoint
    let cs = wi + wr;
    let cl = length(cs);
    if (cl < 1e-4) { return pc; }
    let c = cs / cl;
    let cb = onb(c);
    let dm = log_map(c, cb, wr) - log_map(c, cb, wi);
    let sL = s2_xf(sl, vec2f(dot(cb[0], lb[0]), dot(cb[0], lb[1])), vec2f(dot(cb[1], lb[0]), dot(cb[1], lb[1])));
    let sR = s2_xf(s0, vec2f(dot(cb[0], rb[0]), dot(cb[0], rb[1])), vec2f(dot(cb[1], rb[0]), dot(cb[1], rb[1])));
    // hand-off to the point: far out in the tail (and the light small along the offset), or a light small next to the
    // lobe's core
    let ndm = max(length(dm), 1e-9);
    let d = sqrt(s2_quad_inv(sL + sR / (2.0 * ASG_A.z), dm));
    let along = sqrt(max(s2_quad(sL, dm), 0.0)) / ndm;
    let far = smoothstep(1.0, 2.5, d) * (1.0 - smoothstep(0.15, 0.3, along / ndm));
    let maj = s2_major(sl);
    let ratio = sqrt(maj.z / max(s2_minor(s0) / (2.0 * ASG_A.x), 1e-30));
    let t = max(far, 1.0 - smoothstep(0.12, 0.3, ratio));
    var p_asg = 0.0;
    if (t < 1.0) {
        let aw = ASG_W;
        let aa = ASG_A;
        for (var k = 0u; k < 3u; k++) {
            let sk = sR / (2.0 * aa[k]);
            let st = sL + sk;
            p_asg += aw[k] * sqrt(max(s2_det(sk), 0.0) / max(s2_det(st), 1e-30)) * exp(-0.5 * s2_quad_inv(st, dm));
        }
        p_asg *= pk;
    }
    var p_pt = pc;
    if (t > 0.0) {
        let ax = (maj.x * lb[0] + maj.y * lb[1]) * sqrt(3.0 * maj.z);
        p_pt = (2.0 / 3.0) * pc + (1.0 / 6.0) * (phase_specular(Sd, So, normalize(wi + ax), wo) +
                                                  phase_specular(Sd, So, normalize(wi - ax), wo));
    }
    return mix(p_asg, p_pt, t);
}
