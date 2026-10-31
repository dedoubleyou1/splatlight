// GS runtime v0.1 shared declarations (runtime/SPEC.md). Portable WGSL: no extensions.
// Hosts prepend this file to each kernel file (WGSL has no include).

struct Gaussian {            // SPEC 2.1, 64 B
    mu: vec3f, kappa: f32,
    quat: vec4f,             // x y z w
    scale: vec3f,            // ascending: x = short axis
    part_surface: u32,       // part | surface << 16
    albedo: vec3f, flags: u32,
}

struct Node {                // SPEC 2.2, 96 B
    mean: vec3f, radius: f32,
    cinv_diag: vec3f, kappa: f32,
    cinv_off: vec3f, smax: f32,      // xy xz yz
    sn_diag: vec3f, part: u32,
    sn_off: vec3f, child_lo: u32,
    child_hi: u32, depth: u32, pad0: u32, pad1: u32,   // leaves: pad0 = bitcast iso_norm (iso_init.wgsl), 0 before it runs
}

struct VGrid { w: array<u32, 32> }   // SPEC 2.3, 8x16 bytes

struct Part {                // SPEC 2.4, 128 B
    xf: array<vec4f, 3>,     // world_from_rest rows [R row | t]
    xf_prev: array<vec4f, 3>,
    bound: vec4f,            // rest-frame centre, radius
    root: u32, g_lo: u32, g_hi: u32, flags: u32,
}

struct Lamp {                // SPEC 2.5, 96 B
    center: vec3f, kind: u32,
    e1: vec3f, radius: f32,
    e2: vec3f, range: f32,
    normal: vec3f, flags: u32,
    radiance: vec3f, pad0: u32,
    center_prev: vec3f, pad1: u32,
}

struct FrameUniforms {       // SPEC 3.1, 64 B
    frame: u32, gaussian_count: u32, part_count: u32, lamp_count: u32,
    theta: f32, alpha_min: f32, alpha_part: f32, slot_count: u32,
    eye: vec3f, stratum_step: u32,
    light_samples_axis: u32, material_count: u32, history: u32,   // history: 0 EMA (3.3), 1 count rule
    iso: u32,                // bit 0: direction-independent transfer (M39, the renderer's default; see iso_norm);
                             // bit 1 (with bit 0): SH2 probes (bounce.wgsl iso_xfer, bounce_sh, probe_sh2)
                             // bit 2 (with bit 0): P3 ASG, probe cells as shapes toward the eye (bounce_dirs)
}

// Per-Gaussian outgoing light for the bounce (bounce.wgsl; 2026-10-30, GPU-memory cut): lo_dir holds one vec4 per
// Gaussian, its direct light toward the probes (one colour for every direction, M39; a shiny one's mean over the 14
// fixed directions), and lo_b LOB vec4s: the bounce toward the eye (rgb), the bounce toward the probes (rgb, as
// lo_dir), and the eye's specular lanes (rgb: reflection coverage's specular bounce, or Fresnel's probe sheen).
// Was 14 + 15 vec4s per Gaussian, one per direction.
const LOB: u32 = 3u;
const LOB_EYE: u32 = 0u;
const LOB_SEND: u32 = 1u;
const LOB_LANES: u32 = 2u;

struct Material {            // SPEC 2.6, 32 B
    flake: vec3f,            // SGGX S diagonal in the Gaussian's own frame (x = short axis)
    phase_samples: u32,      // 0 = flat-flake closed form; else fixed VNDF samples
    vfloor: f32,             // floor on V(w_o) in the view correction
    flags: u32,              // bit 0 emissive, bit 1 probe-only, bit 2 one-sided (+short axis)
    pad0: u32, pad1: u32,    // emission: pack2x16float(r, g), pack2x16float(b, 0)
}

struct Sample { tau_other: f32, v_own: f32, tests: u32, evals: u32 }   // SPEC 3.2
const SAMPLE_STALE: u32 = 0xFFFFFFFFu;   // Sample.tests: no sample this frame (amortized walk_emit); ema keeps the history
struct LampState { slots: array<vec2u, 4> }                            // SPEC 3.3: (bitcast v, lamp | count << 16)

const PI: f32 = 3.14159265358979;
const STACK: u32 = 64u;
const LAMP_RECT: u32 = 0u;
const LAMP_SPHERE: u32 = 1u;
const SEED: u32 = 0x9E3779B9u;

// ---- transforms ----------------------------------------------------------------------------------------------

fn xf_point(m: array<vec4f, 3>, p: vec3f) -> vec3f {
    return vec3f(dot(m[0].xyz, p) + m[0].w, dot(m[1].xyz, p) + m[1].w, dot(m[2].xyz, p) + m[2].w);
}

fn xf_dir(m: array<vec4f, 3>, d: vec3f) -> vec3f {
    return vec3f(dot(m[0].xyz, d), dot(m[1].xyz, d), dot(m[2].xyz, d));
}

// R^T d: world direction into the rest frame
fn xf_dir_inv(m: array<vec4f, 3>, d: vec3f) -> vec3f {
    return m[0].xyz * d.x + m[1].xyz * d.y + m[2].xyz * d.z;
}

fn xf_point_inv(m: array<vec4f, 3>, p: vec3f) -> vec3f {
    return xf_dir_inv(m, p - vec3f(m[0].w, m[1].w, m[2].w));
}

// ---- hash (SPEC 4.3) -----------------------------------------------------------------------------------------

fn pcg_hash(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn to_unit(x: u32) -> f32 { return f32(x >> 8u) * (1.0 / 16777216.0); }

fn jitter(g: u32, frame: u32, k: u32) -> f32 {
    return to_unit(pcg_hash(g ^ pcg_hash(frame * 2u + k + SEED)));
}

// ---- maths ---------------------------------------------------------------------------------------------------

// erfc for z >= 0 (Numerical Recipes erfcc, relative error < 1.2e-7)
fn erfc_pos(z: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * z);
    let p = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
            + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    return t * exp(p);
}

fn sym_mul(diag: vec3f, off: vec3f, v: vec3f) -> vec3f {
    return vec3f(diag.x * v.x + off.x * v.y + off.y * v.z,
                 off.x * v.x + diag.y * v.y + off.z * v.z,
                 off.y * v.x + off.z * v.y + diag.z * v.z);
}

// sqrt(w^T S_hat w): SGGX projected-area extinction multiplier (es.dirm)
fn dirm(nd: Node, w: vec3f) -> f32 {
    return sqrt(max(dot(w, sym_mul(nd.sn_diag, nd.sn_off, w)), 0.0));
}

// int_t0^t1 exp(-q(t)/2) dt along P + t w through node nd (es.seg_geom). The exponent uses the perpendicular offset
// e = d - m w, e^T C^-1 e = c - b^2/a without the cancellation.
fn seg_geom(nd: Node, P: vec3f, w: vec3f, t0: f32, t1: f32) -> f32 {
    let d = nd.mean - P;
    let cw = sym_mul(nd.cinv_diag, nd.cinv_off, w);
    let a = dot(cw, w);
    let m = dot(cw, d) / a;
    let e = d - m * w;
    let q = max(dot(e, sym_mul(nd.cinv_diag, nd.cinv_off, e)), 0.0);
    let s = sqrt(0.5 * a);
    let x0 = s * (t0 - m);
    let x1 = s * (t1 - m);
    var df: f32;
    if (x0 > 0.0) {
        df = erfc_pos(x0) - erfc_pos(x1);
    } else if (x1 < 0.0) {
        df = erfc_pos(-x1) - erfc_pos(-x0);
    } else {
        df = 2.0 - erfc_pos(x1) - erfc_pos(-x0);
    }
    return exp(-0.5 * q) * sqrt(PI / (2.0 * a)) * df;
}

// ---- own-part visibility grid (SPEC 2.3) ---------------------------------------------------------------------

fn vgrid_cell(v: VGrid, i: u32, j: u32) -> f32 {
    let k = i * 16u + j;
    return f32((v.w[k >> 2u] >> ((k & 3u) * 8u)) & 0xFFu) * (1.0 / 255.0);
}

// bilinear on the 8x16 cell centres; w is a unit direction in the part's rest frame
fn vgrid_lookup(v: VGrid, w: vec3f) -> f32 {
    let el = asin(clamp(w.z, -1.0, 1.0));
    let fi = clamp((0.5 * PI - el) / (PI / 8.0) - 0.5, 0.0, 7.0);
    let i0 = min(u32(fi), 6u);
    let ti = fi - f32(i0);
    // at an exact pole atan2(0, 0) is NaN on some backends (Metal); any azimuth is right there, take 0 as numpy does
    let az = select(atan2(w.y, w.x), 0.0, abs(w.x) + abs(w.y) < 1e-30);
    let fj = (az + PI) / (2.0 * PI) * 16.0 - 0.5;
    let j0f = floor(fj);
    let tj = fj - j0f;
    let j0 = u32(i32(j0f) + 16) % 16u;
    let j1 = (j0 + 1u) % 16u;
    let a = mix(vgrid_cell(v, i0, j0), vgrid_cell(v, i0, j1), tj);
    let b = mix(vgrid_cell(v, i0 + 1u, j0), vgrid_cell(v, i0 + 1u, j1), tj);
    return mix(a, b, ti);
}

// ---- lamps (SPEC 2.5) ----------------------------------------------------------------------------------------

fn basis_k(w: vec3f) -> vec3f {
    var a = vec3f(0.0, 0.0, 1.0);
    if (abs(w.z) >= 0.999) { a = vec3f(1.0, 0.0, 0.0); }
    return normalize(cross(a, w));
}

// point on lamp L for unit-square (u, v), seen from P (M35 ll.lamp_point)
fn lamp_point(L: Lamp, P: vec3f, u: f32, v: f32) -> vec3f {
    if (L.kind == LAMP_RECT) {
        return L.center + (2.0 * u - 1.0) * L.e1 + (2.0 * v - 1.0) * L.e2;
    }
    let w = normalize(L.center - P);
    let k = basis_k(w);
    let j = cross(w, k);
    let a = 2.0 * u - 1.0;
    let b = 2.0 * v - 1.0;
    var r = 0.0;
    var ph = 0.0;
    if (a == 0.0 && b == 0.0) {
        r = 0.0;
    } else if (abs(a) > abs(b)) {
        r = a; ph = PI / 4.0 * (b / a);
    } else {
        r = b; ph = PI / 2.0 - PI / 4.0 * (a / b);
    }
    return L.center + L.radius * r * (cos(ph) * k + sin(ph) * j);
}

fn lamp_extent(L: Lamp) -> f32 {
    if (L.kind == LAMP_RECT) { return sqrt(dot(L.e1, L.e1) + dot(L.e2, L.e2)); }
    return L.radius;
}

// ---- surface-element shading helpers (SPEC 4.4; shared by shade.wgsl and bounce) --------------------------

const EULER_GAMMA: f32 = 0.5772156649;

fn quat_matrix(q: vec4f) -> mat3x3f {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3f(vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
                   vec3f(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
                   vec3f(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)));
}

// Ein(K) = int_0^K (1 - e^-x) / x dx: series below 1, else ln K + gamma + E1(K) (Abramowitz-Stegun 5.1.56)
fn ein(x: f32) -> f32 {
    if (x < 1.0) {
        var term = x;
        var s = x;
        for (var k = 2; k <= 12; k++) {
            term *= -x / f32(k);
            s += term / f32(k);
        }
        return s;
    }
    let p = x * x * x * x + 8.5733287401 * x * x * x + 18.0590169730 * x * x + 8.6347608925 * x + 0.2677737343;
    let q = x * x * x * x + 9.5733223454 * x * x * x + 25.6329561486 * x * x + 21.0996530827 * x + 3.9584969228;
    return log(x) + EULER_GAMMA + p / q / (x * exp(x));
}

// footprint A(w) = 2 pi sqrt(det C w^T C^-1 w) Ein(K(w)), K = kappa sigma_hat(w) sqrt(2 pi / w^T C^-1 w)
fn footprint(nd: Node, detC: f32, w: vec3f) -> f32 {
    let a = dot(w, sym_mul(nd.cinv_diag, nd.cinv_off, w));
    let K = nd.kappa * dirm(nd, w) * sqrt(2.0 * PI / a);
    return 2.0 * PI * sqrt(detC * a) * ein(K);
}

// Direction-independent transfer (M39, experiments/M39_iso_lighting): a Gaussian shows one colour from every
// direction, the light it catches spread over its exposed area,
//   c = alb * iso_norm * sum L V(w_i) A(w_i) dw,   iso_norm = 1 / (pi max(S, 0.02 S0)),
// S = 4 <A V> over the sphere (own part), S0 the same with V = 1. No phase, no 1/V(w_o). Stored per leaf in Node.pad0.
fn iso_norm(nd: Node) -> f32 {
    return bitcast<f32>(nd.pad0);
}

// Transmission (M41, experiments/M41_transmittance; proposed for SPEC 2.6, not merged): a material with flags bit 4
// lets light further into its own part and tints it. The own-part optical depth toward a light is scaled per channel
// by m = k (1 - s alb), so V(w) -> V(w)^m, with s = tint and k = 1 - translucency stored as pack2x16float(s, k) in
// Material.pad1 (non-emissive materials only). The exposure S is taken with V^m as well, so energy still balances:
// iso_init writes the per-channel iso_norm to Node.pad1 (iso_norm3). FrameUniforms.iso bit 3 switches it on (off by
// default; M39 transfer only). Bounce and other-part shadows are unchanged.
const MAT_TRANSMIT: u32 = 16u;

fn transmit_m(mat: Material, alb: vec3f, on: bool) -> vec3f {
    if (!on || (mat.flags & MAT_TRANSMIT) == 0u || (mat.flags & 1u) != 0u) { return vec3f(1.0); }
    let p = unpack2x16float(mat.pad1);
    return clamp(p.y * (vec3f(1.0) - clamp(p.x, 0.0, 1.0) * alb), vec3f(0.0), vec3f(1.0));
}

// V^m per channel
fn vpow3(v: f32, m: vec3f) -> vec3f {
    if (v <= 0.0) { return vec3f(0.0); }
    return exp2(m * log2(v));
}

// iso_norm per channel: Node.pad1 holds q_c = -log2(iso_norm_c / iso_norm) for r, g, b, 10 bits each over [0, 16]
// (0 = the plain iso_norm). iso_norm_c <= iso_norm since V^m >= V.
fn iso_norm3(nd: Node, on: bool) -> vec3f {
    let n = iso_norm(nd);
    if (!on || nd.pad1 == 0u) { return vec3f(n); }
    let q = vec3f(f32(nd.pad1 & 1023u), f32((nd.pad1 >> 10u) & 1023u), f32((nd.pad1 >> 20u) & 1023u));
    return n * exp2(-q * (16.0 / 1023.0));
}

fn iso_norm3_code(n: f32, n3: vec3f) -> u32 {
    let q = vec3u(round(clamp(-log2(max(n3, vec3f(1e-30)) / n), vec3f(0.0), vec3f(16.0)) * (1023.0 / 16.0)));
    return q.x | (q.y << 10u) | (q.z << 20u);
}

// Heitz et al. 2015 visible normal of SGGX S (diag, off) seen from wi (pt.sggx_vndf)
fn sggx_vndf(wi: vec3f, Sd: vec3f, So: vec3f, u1: f32, u2: f32) -> vec3f {
    let wk = basis_k(wi);
    let wj = cross(wi, wk);
    let Skk = dot(wk, sym_mul(Sd, So, wk)); let Sjj = dot(wj, sym_mul(Sd, So, wj)); let Sii = dot(wi, sym_mul(Sd, So, wi));
    let Skj = dot(wk, sym_mul(Sd, So, wj)); let Ski = dot(wk, sym_mul(Sd, So, wi)); let Sji = dot(wj, sym_mul(Sd, So, wi));
    let det = abs(Skk * Sjj * Sii - Skj * Skj * Sii - Ski * Ski * Sjj - Sji * Sji * Skk + 2.0 * Skj * Ski * Sji);
    let inv = 1.0 / sqrt(Sii);
    let tmp = sqrt(max(Sjj * Sii - Sji * Sji, 1e-30));
    let Mk = vec3f(sqrt(det) / tmp, 0.0, 0.0);
    let Mj = vec3f(-inv * (Ski * Sji - Skj * Sii) / tmp, inv * tmp, 0.0);
    let Mi = vec3f(inv * Ski, inv * Sji, inv * Sii);
    let r = sqrt(u1);
    let ph = 2.0 * PI * u2;
    let u = r * cos(ph); let v = r * sin(ph); let ww = sqrt(max(1.0 - u * u - v * v, 0.0));
    let m = normalize(u * Mk + v * Mj + ww * Mi);
    return normalize(m.x * wk + m.y * wj + m.z * wi);
}

// p(wi -> wo) of diffuse flakes: E_{m ~ VNDF(wi)} max(0, wo.m) / pi
fn phase(mat: Material, n: vec3f, Sd: vec3f, So: vec3f, wi: vec3f, wo: vec3f) -> f32 {
    if (mat.phase_samples == 0u) {               // flat flakes: the visible normal is the side of n facing wi
        let ns = select(-n, n, dot(n, wi) >= 0.0);
        return max(0.0, dot(wo, ns)) / PI;
    }
    var acc = 0.0;
    let N = mat.phase_samples;
    for (var k = 0u; k < N; k++) {
        let u1 = (f32(k) + 0.5) / f32(N);
        let u2 = fract(f32(k) * 0.6180339887 + 0.5);
        acc += max(0.0, dot(wo, sggx_vndf(wi, Sd, So, u1, u2)));
    }
    return acc / (f32(N) * PI);
}

fn lamp_geometry(L: Lamp, P: vec3f, y: vec3f) -> f32 {
    let d = y - P;
    let d2 = dot(d, d);
    if (L.kind == LAMP_RECT) {
        let cl = -dot(d, L.normal) / sqrt(d2);
        return max(cl, 0.0) * 4.0 * length(L.e1) * length(L.e2) / d2;
    }
    return PI * L.radius * L.radius / d2;
}

fn window(L: Lamp, d: f32) -> f32 {
    if (L.range <= 0.0) { return 1.0; }
    let x = clamp(1.0 - pow(d / L.range, 4.0), 0.0, 1.0);
    return x * x;
}

// Sender LOD (bounce.wgsl node_cut): a cut node is a two-sided flake about its short axis n. A direction w at
// cos = w.n is shared between the faces, face + taking (1 + sc_face(cos)) / 2: SC_FACE_K 1 a linear blend (a blob),
// large a hard split (a thin sheet: a wall's inner and outer faces).
const SC_FACE_K: f32 = 1.0;
fn sc_face(c: f32) -> f32 { return clamp(c * SC_FACE_K, -1.0, 1.0); }
