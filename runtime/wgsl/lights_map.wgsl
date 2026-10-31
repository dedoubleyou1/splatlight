// Opacity maps for every light (?lights=unified, David 2026-10-29). Prepend common.wgsl.
//
// The single lamp's light-side opacity map (light_shadow.wgsl A) made shadows a lookup; unified lights still walked
// the scene's tree per receiver per light sample (walk_emit + ema_light), about 1 s a frame on the M1 for 4 lights.
// Here every light gets map A's per-part layers (layer = light x part_count + part), splatted from the light's root
// centre (full octahedral, world axes) and re-splatted only for a part that moved or a light that moved (lights_map.js).
// A receiver then looks up its T toward every light that is on (skipping its own part, and each light's own part, which
// never shadows its own light) and writes emissive.wgsl's slots: the 3 lights strongest at it (strength / d^2) by id,
// the rest as one LIGHT_MIXED T weighted by strength, so shade_emitters' light_T reads them unchanged.
//   lm_clear / lm_splat / lm_resolve / lm_down  as light_shadow.wgsl lsm_*, over the dirty layers (LM dirty array)
//   lm_lookup  one thread per receiver
// Moving parts splat light_shadow.wgsl's motion-blur cut (nodes ~ k x the smear, coverage kept).

struct LMParams {
    n_lights: u32, part_count: u32, res: u32, levels: u32,
    layer_texels: u32, node_count: u32, shutter: f32, cut_k: f32,
    level_off: array<vec4u, 4>,
}
struct LMLevel { level: u32, res: u32, pad0: u32, pad1: u32 }

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;   // motion-blur copies (stretched while moving)
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(4) var<storage, read> parts: array<Part>;
@group(0) @binding(5) var<storage, read> lamps: array<Lamp>;           // lights.js records: centre, radius, on, strength
@group(0) @binding(6) var<storage, read_write> lamp_state: array<LampState>;
@group(0) @binding(35) var<storage, read> node_parent: array<u32>;
@group(0) @binding(52) var<storage, read> gaussians_rest: array<Gaussian>;
@group(0) @binding(53) var<storage, read> nodes_rest: array<Node>;
@group(0) @binding(70) var<uniform> LM: LMParams;
struct LightInfo { nrm: vec3f, pp: u32 }   // one-sided lights' world emitting axis; pp: own part | one-sided << 16
@group(0) @binding(71) var<storage, read> linfo: array<LightInfo>;
@group(0) @binding(72) var<storage, read> dirty: array<u32>;           // 1 = re-splat this layer this frame
@group(0) @binding(73) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(74) var<storage, read_write> mip: array<vec4f>;
@group(0) @binding(75) var<uniform> LV: LMLevel;

const LM_FIX: f32 = 65536.0;
const LM_MAX_HALF: i32 = 48;
const LM_LOWPASS: f32 = 0.3;
const LM_NONE: u32 = 0xFFFFFFFFu;
const LM_MIXED: u32 = 0xFFFFu;          // emissive.wgsl LIGHT_MIXED

fn lm_texels() -> u32 { return LM.res * LM.res; }
fn lm_level_off(k: u32) -> u32 { return LM.level_off[k / 4u][k % 4u]; }
fn layer_of(l: u32, p: u32) -> u32 { return l * LM.part_count + p; }
fn light_part(l: u32) -> u32 { return linfo[l].pp & 0xFFFFu; }
fn light_on(l: u32) -> bool { return lamps[l].flags != 0u && bitcast<f32>(lamps[l].pad0) > 0.0; }

// ---- full octahedral map over world axes ---------------------------------------------------------------------

fn lm_uv(w: vec3f) -> vec2f {
    var p = w.xy / max(abs(w.x) + abs(w.y) + abs(w.z), 1e-12);
    if (w.z < 0.0) { p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0)); }
    return p;
}
fn lm_basis(w: vec3f) -> mat2x3f { let k = basis_k(w); return mat2x3f(k, cross(w, k)); }
fn lm_jac(w: vec3f, b: mat2x3f) -> mat2x2f {
    let e = 1e-3;
    let u0 = lm_uv(w);
    return mat2x2f((lm_uv(normalize(w + e * b[0])) - u0) / e, (lm_uv(normalize(w + e * b[1])) - u0) / e);
}
fn lm_inv2(m: mat2x2f) -> mat2x2f {
    let d = determinant(m);
    return mat2x2f(vec2f(m[1].y, -m[0].y), vec2f(-m[1].x, m[0].x)) * (1.0 / d);
}
fn lm_sym3(dg: vec3f, off: vec3f) -> mat3x3f {
    return mat3x3f(vec3f(dg.x, off.x, off.y), vec3f(off.x, dg.y, off.z), vec3f(off.y, off.z, dg.z));
}
fn lm_inv3(m: mat3x3f) -> mat3x3f {
    let r0 = cross(m[1], m[2]); let r1 = cross(m[2], m[0]); let r2 = cross(m[0], m[1]);
    return transpose(mat3x3f(r0, r1, r2)) * (1.0 / dot(m[0], r0));
}
fn lm_rot(xf: array<vec4f, 3>) -> mat3x3f { return transpose(mat3x3f(xf[0].xyz, xf[1].xyz, xf[2].xyz)); }

fn lm_cov(i: u32, R: mat3x3f, rest: bool) -> mat3x3f {
    var C: mat3x3f;
    if (i < U.gaussian_count) {
        var G = gaussians[i];
        if (rest) { G = gaussians_rest[i]; }
        let Rq = quat_matrix(G.quat);
        let s2 = G.scale * G.scale;
        C = Rq * mat3x3f(vec3f(s2.x, 0.0, 0.0), vec3f(0.0, s2.y, 0.0), vec3f(0.0, 0.0, s2.z)) * transpose(Rq);
    } else {
        var nd = nodes[i];
        if (rest) { nd = nodes_rest[i]; }
        C = lm_inv3(lm_sym3(nd.cinv_diag, nd.cinv_off));
    }
    return R * C * transpose(R);
}

@compute @workgroup_size(64)
fn lm_clear(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = lm_texels() * 3u;
    let layers = LM.n_lights * LM.part_count;
    if (i >= n * layers || dirty[i / n] == 0u) { return; }
    atomicStore(&acc[i], 0u);
}

// ---- splat ----------------------------------------------------------------------------------------------------

fn lm_smear(p: u32) -> f32 {
    if (LM.cut_k <= 0.0 || LM.shutter <= 0.0 || (parts[p].flags & 2u) == 0u) { return 0.0; }
    let P = parts[p];
    let x = nodes_rest[P.root].mean;
    return LM.shutter * length(xf_point(P.xf, x) - xf_point(P.xf_prev, x));
}

// one thread per record (leaves, and tree nodes for a moving part's motion-blur cut); loops the lights
@compute @workgroup_size(64)
fn lm_splat(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= LM.node_count) { return; }
    let leaf = g < U.gaussian_count;
    var p: u32;
    if (leaf) { p = gaussians[g].part_surface & 0xFFFFu; } else { p = nodes_rest[g].part; }
    if (p >= LM.part_count) { return; }
    var any_dirty = false;
    for (var l = 0u; l < LM.n_lights; l++) { any_dirty = any_dirty || dirty[layer_of(l, p)] != 0u; }
    if (!any_dirty) { return; }
    let L = lm_smear(p);
    var cover = false;
    if (L > 0.0) {
        let thr = LM.cut_k * L;
        if (!leaf && nodes_rest[g].radius > thr) { return; }
        let par = node_parent[g];
        if (g != parts[p].root && par != LM_NONE && nodes_rest[par].radius <= thr) { return; }
        cover = true;
    } else if (!leaf) { return; }
    for (var l = 0u; l < LM.n_lights; l++) {
        if (dirty[layer_of(l, p)] != 0u && light_part(l) != p && light_on(l)) { lm_splat_one(g, p, l, cover); }
    }
}

fn lm_splat_one(g: u32, p: u32, l: u32, cover: bool) {
    let xf = parts[p].xf;
    let R = lm_rot(xf);
    let nd = nodes[g];
    var mu = nd.mean;
    if (g < U.gaussian_count) { mu = gaussians[g].mu; }
    let dv = xf_point(xf, mu) - lamps[l].center;
    let r = length(dv);
    if (r < 1e-4) { return; }
    let w = dv / r;
    let C = lm_cov(g, R, false);
    // behind a one-sided light (the ceiling right above the lamp panel): it never blocks that light (light_shadow.wgsl
    // lsm_splat's hemi test). Without this the ceiling skin, a few cm from the lamp's centre, spreads over a huge solid
    // angle and dims every direction (fx: the lamp alone 7% dark everywhere, 2026-10-29)
    let ln = linfo[l];
    if ((ln.pp >> 16u) != 0u && dot(dv, ln.nrm) < -3.0 * sqrt(max(dot(ln.nrm, C * ln.nrm), 0.0))) { return; }
    let b = lm_basis(w);
    let c0 = C * b[0]; let c1 = C * b[1]; let cw = C * w;
    let Cbb = mat2x2f(vec2f(dot(b[0], c0), dot(b[1], c0)), vec2f(dot(b[0], c1), dot(b[1], c1)));
    let cbz = vec2f(dot(b[0], cw), dot(b[1], cw));
    let Cbb_i = lm_inv2(Cbb);
    let sz2 = max(dot(w, cw) - dot(cbz, Cbb_i * cbz), 1e-12);
    let J = lm_jac(w, b) * (0.5 * f32(LM.res));
    var S = J * (Cbb * (1.0 / (r * r))) * transpose(J);
    let det0 = determinant(S);
    S = S + mat2x2f(vec2f(LM_LOWPASS, 0.0), vec2f(0.0, LM_LOWPASS));
    let det1 = determinant(S);
    var tau0 = nd.kappa * dirm(nd, xf_dir_inv(xf, w)) * sqrt(2.0 * PI * sz2) * sqrt(max(det0, 0.0) / det1);
    if (cover) {   // light_shadow.wgsl splat_one: the rest record's peak opacity over the stretched footprint
        let ndr = nodes_rest[g];
        let Cr = lm_cov(g, R, true);
        let r0 = Cr * b[0]; let r1 = Cr * b[1]; let rw = Cr * w;
        let Cbr = mat2x2f(vec2f(dot(b[0], r0), dot(b[1], r0)), vec2f(dot(b[0], r1), dot(b[1], r1)));
        let cbzr = vec2f(dot(b[0], rw), dot(b[1], rw));
        let szr = max(dot(w, rw) - dot(cbzr, lm_inv2(Cbr) * cbzr), 1e-12);
        var Sr = J * (Cbr * (1.0 / (r * r))) * transpose(J);
        let d0r = determinant(Sr);
        Sr = Sr + mat2x2f(vec2f(LM_LOWPASS, 0.0), vec2f(0.0, LM_LOWPASS));
        let d1r = determinant(Sr);
        let tau_r = ndr.kappa * dirm(ndr, xf_dir_inv(xf, w)) * sqrt(2.0 * PI * szr) * sqrt(max(d0r, 0.0) / d1r);
        let a = min((1.0 - exp(-tau_r)) * sqrt(max(d1r, 1e-30) / max(det1, 1e-30)), 0.999);
        tau0 = min(-log(1.0 - a), tau0);
    }
    let Si = lm_inv2(S);
    let gz = transpose(lm_inv2(J)) * (Cbb_i * cbz * r);
    let ct = (lm_uv(w) * 0.5 + 0.5) * f32(LM.res);
    let hx = min(i32(ceil(3.0 * sqrt(S[0].x))), LM_MAX_HALF);
    let hy = min(i32(ceil(3.0 * sqrt(S[1].y))), LM_MAX_HALF);
    let cx = i32(floor(ct.x));
    let cy = i32(floor(ct.y));
    let base = layer_of(l, p) * lm_texels();
    let res = i32(LM.res);
    for (var y = max(cy - hy, 0); y <= min(cy + hy, res - 1); y++) {
        for (var x = max(cx - hx, 0); x <= min(cx + hx, res - 1); x++) {
            let dt = vec2f(f32(x) + 0.5, f32(y) + 0.5) - ct;
            let q = dot(dt, Si * dt);
            if (q > 9.0) { continue; }
            let t = min(tau0 * exp(-0.5 * q), 64.0);
            let t_fix = t * LM_FIX;
            if (t_fix < 0.5) { continue; }
            let z = max(r + dot(gz, dt), 0.0);
            let k = (base + u32(y) * LM.res + u32(x)) * 3u;
            atomicAdd(&acc[k], u32(t_fix + 0.5));
            atomicAdd(&acc[k + 1u], u32(t_fix * z + 0.5));
            atomicAdd(&acc[k + 2u], u32(t_fix * (z * z + sz2) + 0.5));
        }
    }
}

// ---- resolve and mips -----------------------------------------------------------------------------------------

@compute @workgroup_size(64)
fn lm_resolve(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = lm_texels();
    if (i >= n * LM.n_lights * LM.part_count) { return; }
    let ly = i / n;
    if (dirty[ly] == 0u) { return; }
    let s0 = f32(atomicLoad(&acc[i * 3u])) / LM_FIX;
    var o = vec4f(0.0);
    if (s0 > 1e-5) {
        let a = 1.0 - exp(-s0);
        o = vec4f(a, a * f32(atomicLoad(&acc[i * 3u + 1u])) / (LM_FIX * s0), a * f32(atomicLoad(&acc[i * 3u + 2u])) / (LM_FIX * s0), 0.0);
    }
    mip[ly * LM.layer_texels + (i - ly * n)] = o;
}

@compute @workgroup_size(64)
fn lm_down(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = LV.res * LV.res;
    if (i >= n * LM.n_lights * LM.part_count) { return; }
    let ly = i / n;
    if (dirty[ly] == 0u) { return; }
    let j = i - ly * n;
    let y = j / LV.res;
    let x = j - y * LV.res;
    let pr = LV.res * 2u;
    let src = ly * LM.layer_texels + lm_level_off(LV.level - 1u) + (2u * y) * pr + 2u * x;
    mip[ly * LM.layer_texels + lm_level_off(LV.level) + j] = 0.25 * (mip[src] + mip[src + 1u] + mip[src + pr] + mip[src + pr + 1u]);
}

fn lm_mip_at(ly: u32, k: u32, tc: vec2f) -> vec4f {
    let rk = max(LM.res >> k, 1u);
    let c = clamp(tc / f32(1u << k) - 0.5, vec2f(0.0), vec2f(f32(rk - 1u)));
    let i0 = vec2u(floor(c));
    let i1 = min(i0 + 1u, vec2u(rk - 1u));
    let f = c - vec2f(i0);
    let base = ly * LM.layer_texels + lm_level_off(k);
    let a = mix(mip[base + i0.y * rk + i0.x], mip[base + i0.y * rk + i1.x], f.x);
    let b = mix(mip[base + i1.y * rk + i0.x], mip[base + i1.y * rk + i1.x], f.x);
    return mix(a, b, f.y);
}

fn lm_mip_tri(ly: u32, lv: f32, tc: vec2f) -> vec4f {
    let lc = clamp(lv, 0.0, f32(LM.levels - 1u));
    let k0 = u32(floor(lc));
    let k1 = min(k0 + 1u, LM.levels - 1u);
    return mix(lm_mip_at(ly, k0, tc), lm_mip_at(ly, k1, tc), lc - f32(k0));
}

fn lm_phi(x: f32) -> f32 {
    let a = abs(x) * 0.70710678;
    let h = 0.5 * erfc_pos(a);
    return select(h, 1.0 - h, x >= 0.0);
}

// ---- lookup ---------------------------------------------------------------------------------------------------

// receiver at P (own part `own`), light l: light_shadow.wgsl lsm_lookup_one's T over the other parts' layers
fn lm_T(l: u32, own: u32, P: vec3f) -> f32 {
    let dv = P - lamps[l].center;
    let z = length(dv);
    if (z < 1e-4) { return 1.0; }
    let w = dv / z;
    let J = lm_jac(w, lm_basis(w));
    let tex_ang = (2.0 / f32(LM.res)) / sqrt(max(abs(determinant(J)), 1e-12));
    let tc = (lm_uv(w) * 0.5 + 0.5) * f32(LM.res);
    let sig = max(lamps[l].radius / 3.0, 1e-3);   // the light's own size, as an isotropic std (m)
    var T = 1.0;
    for (var p = 0u; p < LM.part_count; p++) {
        if (p == own || p == light_part(l)) { continue; }
        let bs = parts[p].bound;
        let bc = xf_point(parts[p].xf, bs.xyz) - lamps[l].center;
        let dc = length(bc);
        let zn = max(dc - bs.w, 0.02);
        if (z <= zn) { continue; }
        let search = 2.0 * sig * (z - zn) / (zn * z);
        if (dc > bs.w) {
            let ang = acos(clamp(dot(bc / dc, w), -1.0, 1.0));
            if (ang > asin(bs.w / dc) + search + 2.0 * tex_ang) { continue; }
        }
        let ly = layer_of(l, p);
        let a0 = lm_mip_tri(ly, ceil(log2(max(2.0 * search / tex_ang, 1.0))), tc);
        if (a0.x < 1e-4) { continue; }
        let zb = max(a0.y / a0.x, 0.02);
        let pen = sig * max(z - zb, 0.0) / (zb * z);
        let s = lm_mip_tri(ly, log2(max(2.0 * pen / tex_ang, 1.0)), tc);
        if (s.x < 1e-5) { continue; }
        let mu = s.y / s.x;
        let sd = sqrt(max(s.z / s.x - mu * mu, 1e-8));
        T *= 1.0 - s.x * lm_phi((z - mu) / sd);
    }
    return T;
}

@compute @workgroup_size(64)
fn lm_lookup(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let G = gaussians_rest[g];
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    // the 3 strongest lights here keep their own slot; the rest share LIGHT_MIXED, weighted by strength
    var ids = array<u32, 3>(LM_NONE, LM_NONE, LM_NONE);
    var ws = array<f32, 3>(0.0, 0.0, 0.0);
    var Ts = array<f32, 3>(1.0, 1.0, 1.0);
    var mw = 0.0;
    var mt = 0.0;
    for (var l = 0u; l < LM.n_lights; l++) {
        if (!light_on(l)) { continue; }
        let d = lamps[l].center - P;
        var w = bitcast<f32>(lamps[l].pad0) / max(dot(d, d), 1e-4);
        var T = lm_T(l, own, P);
        // insert into the top 3; the displaced one (or this one) joins the mixed slot
        for (var k = 0u; k < 3u; k++) {
            if (w > ws[k]) {
                let wi = ws[k]; let ti = Ts[k]; let ii = ids[k];
                ws[k] = w; Ts[k] = T; ids[k] = l;
                w = wi; T = ti;
                if (ii == LM_NONE) { w = 0.0; }
            }
        }
        mw += w; mt += w * T;
    }
    var st: LampState;
    for (var k = 0u; k < 3u; k++) {
        st.slots[k] = select(vec2u(0u, 0u), vec2u(bitcast<u32>(Ts[k]), ids[k] | (1u << 16u)), ids[k] != LM_NONE);
    }
    st.slots[3] = select(vec2u(0u, 0u), vec2u(bitcast<u32>(mt / max(mw, 1e-30)), LM_MIXED | (1u << 16u)), mw > 0.0);
    lamp_state[g] = st;
}
