// SGGX diffuse flakes for the camera only (experiments/M40_flake_camera; renderer toggle "SGGX flakes (camera)",
// off by default). Everything that reads a Gaussian's light (bounce, probes, tree parents, near-field blockers) keeps
// M39's one colour, alb iso_norm sum L V A; only the colour the camera shows is
//   c(w_o) = alb iso_norm sum L V(w_i) A(w_i) lobe(w_i -> w_o)
// with diffuse flakes oriented by the Gaussian's own shape: S = R diag(1, (sx/sy)^2, (sx/sz)^2) R^T, its projected-
// area matrix (det C C^-1, the footprint's own), not the material's flake.
//
// lobe(w_i -> w_o) = mean_k max(0, w_o . m_k) / (iso_norm mean_k h(m_k)), m_k = visible normals of w_i (VNDF, the
// fixed samples of common.wgsl phase; flat Gaussians take the side of the short axis facing w_i), h(m) =
// int max(0, w . m) A(w) V(w) dw. So for every w_i the lobe's mean over view directions weighted by A(w_o) V(w_o),
// the weight the camera sees a Gaussian with, is exactly 1 (1 / iso_norm = int A V): the same emitted power as M39,
// a different angular spread. h comes from the SH2 of A V (baked once per asset, `expo`) through the clamped-cosine
// convolution (pi, 2 pi / 3, pi / 4), Ramamoorthi & Hanrahan's irradiance approximation.
//
// Prepend common.wgsl, material_ext.wgsl and emissive.wgsl (flake_direct.wgsl) or bounce.wgsl (flake_bounce.wgsl).

struct FlakeParams {
    gain: f32,              // bounce gain (RenderFrame palette[3].x)
    flat_r: f32,            // sx / sy below this: flat flakes (closed form)
    h_floor: f32,           // floor on iso_norm mean h: caps the lobe at 1 / h_floor
    samples: u32,           // VNDF samples for the other Gaussians
    flags: u32,             // bit 0: flake_bounce runs (SH2 probes); else matte Gaussians keep the M39 bounce
    count: u32,             // Gaussians
    material_count: u32,
    pad: u32,
}

@group(0) @binding(30) var<storage, read_write> cam: array<vec4f>;    // per Gaussian: camera colour, w = 1 flaked (matte)
@group(0) @binding(31) var<uniform> FP: FlakeParams;
@group(0) @binding(32) var<storage, read_write> expo: array<vec4f>;   // per Gaussian x 3: SH2 of A V, rest frame

const FL_ZH = vec3f(3.14159265358979, 2.09439510239320, 0.78539816339745);   // clamped-cosine convolution, l = 0, 1, 2

fn fl_sh2(d: vec3f) -> array<f32, 9> {       // bounce.wgsl sh2_basis
    let x = d.x; let y = d.y; let z = d.z;
    return array<f32, 9>(0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z,
                         0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y));
}

// the exposure SH2 premultiplied by the convolution, so h(m) = max(dot with Y(m), 0)
fn fl_expo(g: u32) -> array<f32, 9> {
    let e0 = expo[3u * g]; let e1 = expo[3u * g + 1u]; let e2 = expo[3u * g + 2u];
    return array<f32, 9>(FL_ZH.x * e0.x, FL_ZH.y * e0.y, FL_ZH.y * e0.z, FL_ZH.y * e0.w, FL_ZH.z * e1.x, FL_ZH.z * e1.y,
                         FL_ZH.z * e1.z, FL_ZH.z * e1.w, FL_ZH.z * e2.x);
}

fn fl_h(E: array<f32, 9>, m: vec3f) -> f32 {
    let y = fl_sh2(m);
    var h = 0.0;
    for (var k = 0u; k < 9u; k++) { h += E[k] * y[k]; }
    return max(h, 0.0);
}

struct FlakeShape { n: vec3f, Sd: vec3f, So: vec3f, is_flat: bool, h_pos: f32, h_neg: f32 }

fn fl_shape(G: Gaussian, E: array<f32, 9>) -> FlakeShape {
    let Rq = quat_matrix(G.quat);
    let ry = G.scale.x / max(G.scale.y, 1e-12);
    let rz = G.scale.x / max(G.scale.z, 1e-12);
    let Sloc = Rq * mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, ry * ry, 0.0), vec3f(0.0, 0.0, rz * rz)) * transpose(Rq);
    let n = Rq[0];
    return FlakeShape(n, vec3f(Sloc[0].x, Sloc[1].y, Sloc[2].z), vec3f(Sloc[1].x, Sloc[2].x, Sloc[2].y),
                      ry < FP.flat_r, fl_h(E, n), fl_h(E, -n));
}

// lobe(wi -> wo), rest frame (see the top)
fn fl_lobe(sh: FlakeShape, E: array<f32, 9>, inorm: f32, wi: vec3f, wo: vec3f) -> f32 {
    if (sh.is_flat) {
        let up = dot(sh.n, wi) >= 0.0;
        let ns = select(-sh.n, sh.n, up);
        return max(0.0, dot(wo, ns)) / max(inorm * select(sh.h_neg, sh.h_pos, up), FP.h_floor);
    }
    let N = max(FP.samples, 1u);
    var num = 0.0;
    var den = 0.0;
    for (var k = 0u; k < N; k++) {
        let u1 = (f32(k) + 0.5) / f32(N);
        let u2 = fract(f32(k) * 0.6180339887 + 0.5);
        let m = sggx_vndf(wi, sh.Sd, sh.So, u1, u2);
        num += max(0.0, dot(wo, m));
        den += fl_h(E, m);
    }
    return num / max(inorm * den, FP.h_floor * f32(N));
}
