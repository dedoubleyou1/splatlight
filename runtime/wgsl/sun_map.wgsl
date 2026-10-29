// One-frame shadows from distant lights (the sun) cast by movable parts (David, 2026-10-29). Prepend common.wgsl.
//
// The sun's walk (environment.wgsl env_shadow, 16 rays per receiver) costs too much to rerun every frame outdoors
// (hill: 1.17M receivers, ~36 frames a pass). Static occluders don't change unless the sun does, so they keep the walk
// (env_T[count + g], refreshed in slices only when the sun moves). Movable parts get an orthographic opacity map per
// part and light instead, fitted to the part's bounding sphere (so a 10 m cloud gets a few-cm texel at 512), re-splatted
// only when that part moves, the light moves, or on load. A receiver multiplies the walked static T by each other
// movable part's map T, every frame anything moved: env_T[g] = env_T[count + g] x prod_m T_m.
//   sm_frame    one thread per layer: the layer's frame (centre across the light, half size, depth origin)
//   sm_clear    the dirty layers' accumulators
//   sm_splat    one thread per record of a dirty part (ranges): tau and tau-weighted depth moments (as light_shadow.wgsl
//               lsm_splat, orthographic); a moving part splats light_shadow.wgsl's motion-blur cut with coverage kept
//   sm_resolve  (alpha, alpha mu, alpha m2); sm_down the mip pyramid
//   sm_lookup   one thread per receiver: blocker depth at a coarse level, then the level whose texel matches the
//               penumbra sigma (z - z_b), T_m = 1 - alpha Phi((z - mu) / sd); env_T[g] = static T x prod T_m
// Layer l = m x K + k: movable part m (SM.movable[m]), light k (K = SM.n_lights).

struct SunParams {
    n_layers: u32, res: u32, levels: u32, layer_texels: u32,
    n_lights: u32, n_movable: u32, dirty: u32, range_count: u32,   // dirty: bit l = re-splat layer l this frame
    margin: f32, shutter: f32, cut_k: f32, node_count: u32,       // margin (m) for penumbrae past the part; blur cut
    lights: array<vec4f, 4>,          // xyz unit direction toward the light (world), w angular std (rad); w < 0: no shadows
    movable: array<vec4u, 4>,         // movable part index by m (16 max)
    level_off: array<vec4u, 4>,       // first texel of each mip level within a layer
}
struct SunLayer { c: vec2f, half: f32, zref: f32, t1: vec3f, part: u32, t2: vec3f, light: u32 }
struct SunLevel { level: u32, res: u32, pad0: u32, pad1: u32 }

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;   // motion-blur copies (stretched while moving)
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(4) var<storage, read> parts: array<Part>;
@group(0) @binding(35) var<storage, read> node_parent: array<u32>;
@group(0) @binding(41) var<storage, read_write> env_T: array<vec4f>;   // environment.wgsl: [0, G) T, [G, 2G) static occluders' T
@group(0) @binding(52) var<storage, read> gaussians_rest: array<Gaussian>;
@group(0) @binding(53) var<storage, read> nodes_rest: array<Node>;
@group(0) @binding(60) var<uniform> SM: SunParams;
@group(0) @binding(61) var<storage, read_write> layers: array<SunLayer>;
@group(0) @binding(62) var<storage, read_write> acc: array<atomic<u32>>;   // [layer][texel][3] fixed point
@group(0) @binding(63) var<storage, read_write> mip: array<vec4f>;         // [layer][level texels] (a, a mu, a m2, 0)
@group(0) @binding(64) var<storage, read> ranges: array<vec4u>;            // (first, end, thread offset, m)
@group(0) @binding(65) var<uniform> LV: SunLevel;

const SM_FIX: f32 = 4096.0;           // fixed point: tau <= 64 and depth^2 up to ~1e4 m^2 stay in u32
const SM_MAX_HALF: i32 = 32;
const SM_LOWPASS: f32 = 0.3;
const SM_NONE: u32 = 0xFFFFFFFFu;

fn movable_part(m: u32) -> u32 { return SM.movable[m / 4u][m % 4u]; }
fn sm_level_off(k: u32) -> u32 { return SM.level_off[k / 4u][k % 4u]; }
fn sm_texels() -> u32 { return SM.res * SM.res; }
fn layer_dirty(l: u32) -> bool { return (SM.dirty & (1u << l)) != 0u; }

fn sym3s(dg: vec3f, off: vec3f) -> mat3x3f {
    return mat3x3f(vec3f(dg.x, off.x, off.y), vec3f(off.x, dg.y, off.z), vec3f(off.y, off.z, dg.z));
}
fn inv3s(m: mat3x3f) -> mat3x3f {
    let r0 = cross(m[1], m[2]); let r1 = cross(m[2], m[0]); let r2 = cross(m[0], m[1]);
    return transpose(mat3x3f(r0, r1, r2)) * (1.0 / dot(m[0], r0));
}
fn rot_of(xf: array<vec4f, 3>) -> mat3x3f { return transpose(mat3x3f(xf[0].xyz, xf[1].xyz, xf[2].xyz)); }

// world covariance of record i, from the motion-blur copies (rest = false) or the rest records
fn rec_cov(i: u32, R: mat3x3f, rest: bool) -> mat3x3f {
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
        C = inv3s(sym3s(nd.cinv_diag, nd.cinv_off));
    }
    return R * C * transpose(R);
}

// ---- layer frames ---------------------------------------------------------------------------------------------

@compute @workgroup_size(64)
fn sm_frame(@builtin(global_invocation_id) gid: vec3u) {
    let l = gid.x;
    if (l >= SM.n_layers) { return; }
    let m = l / SM.n_lights;
    let k = l - m * SM.n_lights;
    let p = movable_part(m);
    let L = SM.lights[k].xyz;
    let t1 = basis_k(L);
    let t2 = cross(L, t1);
    let b = parts[p].bound;
    let c = xf_point(parts[p].xf, b.xyz);
    layers[l] = SunLayer(vec2f(dot(c, t1), dot(c, t2)), b.w + SM.margin, dot(c, L) + b.w, t1, p, t2, k);
}

@compute @workgroup_size(64)
fn sm_clear(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = sm_texels() * 3u;
    if (i >= n * SM.n_layers || !layer_dirty(i / n)) { return; }
    atomicStore(&acc[i], 0u);
}

// ---- splat ----------------------------------------------------------------------------------------------------

// part p's smear this exposure; 0 = still or the motion-blur cut is off (light_shadow.wgsl smear_len)
fn sm_smear(p: u32) -> f32 {
    if (SM.cut_k <= 0.0 || SM.shutter <= 0.0 || (parts[p].flags & 2u) == 0u) { return 0.0; }
    let P = parts[p];
    let x = nodes_rest[P.root].mean;
    return SM.shutter * length(xf_point(P.xf, x) - xf_point(P.xf_prev, x));
}

@compute @workgroup_size(64)
fn sm_splat(@builtin(global_invocation_id) gid: vec3u) {
    var g = SM_NONE;
    var m = 0u;
    for (var r = 0u; r < SM.range_count; r++) {
        let R = ranges[r];
        if (gid.x >= R.z && gid.x < R.z + (R.y - R.x)) { g = R.x + (gid.x - R.z); m = R.w; break; }
    }
    if (g == SM_NONE) { return; }
    let p = movable_part(m);
    let leaf = g < U.gaussian_count;
    let L = sm_smear(p);
    var cover = false;
    if (L > 0.0) {
        // light_shadow.wgsl's motion-blur cut: nodes about k x the smear, each keeping its coverage
        let thr = SM.cut_k * L;
        if (!leaf && nodes_rest[g].radius > thr) { return; }
        let par = node_parent[g];
        if (g != parts[p].root && par != SM_NONE && nodes_rest[par].radius <= thr) { return; }
        cover = true;
    } else if (!leaf) { return; }
    for (var k = 0u; k < SM.n_lights; k++) {
        let l = m * SM.n_lights + k;
        if (layer_dirty(l) && SM.lights[k].w >= 0.0) { sm_splat_one(g, p, l, cover); }
    }
}

fn sm_splat_one(g: u32, p: u32, l: u32, cover: bool) {
    let ly = layers[l];
    let xf = parts[p].xf;
    let R = rot_of(xf);
    let nd = nodes[g];
    var mu = nd.mean;
    if (g < U.gaussian_count) { mu = gaussians[g].mu; }
    let x = xf_point(xf, mu);
    let w = SM.lights[ly.light].xyz;
    let b = mat2x3f(ly.t1, ly.t2);
    let C = rec_cov(g, R, false);
    let c0 = C * b[0]; let c1 = C * b[1]; let cw = C * w;
    let Cbb = mat2x2f(vec2f(dot(b[0], c0), dot(b[1], c0)), vec2f(dot(b[0], c1), dot(b[1], c1)));
    let cbz = vec2f(dot(b[0], cw), dot(b[1], cw));
    let Cbb_i = inverse2(Cbb);
    let sz2 = max(dot(w, cw) - dot(cbz, Cbb_i * cbz), 1e-12);
    // orthographic: metres -> texels
    let tex = f32(SM.res) / (2.0 * ly.half);
    var S = Cbb * (tex * tex);
    let det0 = determinant(S);
    S = S + mat2x2f(vec2f(SM_LOWPASS, 0.0), vec2f(0.0, SM_LOWPASS));
    let det1 = determinant(S);
    var tau0 = nd.kappa * dirm(nd, xf_dir_inv(xf, w)) * sqrt(2.0 * PI * sz2) * sqrt(max(det0, 0.0) / det1);
    if (cover) {
        let ndr = nodes_rest[g];
        let Cr = rec_cov(g, R, true);
        let r0 = Cr * b[0]; let r1 = Cr * b[1]; let rw = Cr * w;
        let Cbr = mat2x2f(vec2f(dot(b[0], r0), dot(b[1], r0)), vec2f(dot(b[0], r1), dot(b[1], r1)));
        let cbzr = vec2f(dot(b[0], rw), dot(b[1], rw));
        let szr = max(dot(w, rw) - dot(cbzr, inverse2(Cbr) * cbzr), 1e-12);
        var Sr = Cbr * (tex * tex);
        let d0r = determinant(Sr);
        Sr = Sr + mat2x2f(vec2f(SM_LOWPASS, 0.0), vec2f(0.0, SM_LOWPASS));
        let d1r = determinant(Sr);
        let tau_r = ndr.kappa * dirm(ndr, xf_dir_inv(xf, w)) * sqrt(2.0 * PI * szr) * sqrt(max(d0r, 0.0) / d1r);
        let a = min((1.0 - exp(-tau_r)) * sqrt(max(d1r, 1e-30) / max(det1, 1e-30)), 0.999);
        tau0 = min(-log(1.0 - a), tau0);
    }
    let Si = inverse2(S);
    // depth below the layer's top along the light, and its change across the footprint
    let z0 = ly.zref - dot(x, w);
    let gz = (Cbb_i * cbz) / tex;   // E[x.w | across offset] rises along Cbb^-1 cbz, so the depth z falls
    let ct = (vec2f(dot(x, b[0]), dot(x, b[1])) - ly.c + ly.half) * tex;
    let hx = min(i32(ceil(3.0 * sqrt(S[0].x))), SM_MAX_HALF);
    let hy = min(i32(ceil(3.0 * sqrt(S[1].y))), SM_MAX_HALF);
    let cx = i32(floor(ct.x));
    let cy = i32(floor(ct.y));
    let base = l * sm_texels();
    let res = i32(SM.res);
    for (var y = max(cy - hy, 0); y <= min(cy + hy, res - 1); y++) {
        for (var xx = max(cx - hx, 0); xx <= min(cx + hx, res - 1); xx++) {
            let dt = vec2f(f32(xx) + 0.5, f32(y) + 0.5) - ct;
            let q = dot(dt, Si * dt);
            if (q > 9.0) { continue; }
            let t = min(tau0 * exp(-0.5 * q), 64.0);
            let t_fix = t * SM_FIX;
            if (t_fix < 0.5) { continue; }
            let z = max(z0 - dot(gz, dt), 0.0);
            let i = (base + u32(y) * SM.res + u32(xx)) * 3u;
            atomicAdd(&acc[i], u32(t_fix + 0.5));
            atomicAdd(&acc[i + 1u], u32(t_fix * z + 0.5));
            atomicAdd(&acc[i + 2u], u32(t_fix * (z * z + sz2) + 0.5));
        }
    }
}

fn inverse2(m: mat2x2f) -> mat2x2f {
    let d = determinant(m);
    return mat2x2f(vec2f(m[1].y, -m[0].y), vec2f(-m[1].x, m[0].x)) * (1.0 / d);
}

// ---- resolve and mips -----------------------------------------------------------------------------------------

@compute @workgroup_size(64)
fn sm_resolve(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = sm_texels();
    if (i >= n * SM.n_layers) { return; }
    let l = i / n;
    if (!layer_dirty(l)) { return; }
    let s0 = f32(atomicLoad(&acc[i * 3u])) / SM_FIX;
    var o = vec4f(0.0);
    if (s0 > 1e-5) {
        let a = 1.0 - exp(-s0);
        o = vec4f(a, a * f32(atomicLoad(&acc[i * 3u + 1u])) / (SM_FIX * s0), a * f32(atomicLoad(&acc[i * 3u + 2u])) / (SM_FIX * s0), 0.0);
    }
    mip[l * SM.layer_texels + (i - l * n)] = o;
}

@compute @workgroup_size(64)
fn sm_down(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = LV.res * LV.res;
    if (i >= n * SM.n_layers) { return; }
    let l = i / n;
    if (!layer_dirty(l)) { return; }
    let j = i - l * n;
    let y = j / LV.res;
    let x = j - y * LV.res;
    let pr = LV.res * 2u;
    let src = l * SM.layer_texels + sm_level_off(LV.level - 1u) + (2u * y) * pr + 2u * x;
    let s = mip[src] + mip[src + 1u] + mip[src + pr] + mip[src + pr + 1u];
    mip[l * SM.layer_texels + sm_level_off(LV.level) + j] = 0.25 * s;
}

fn sm_mip_at(l: u32, k: u32, tc: vec2f) -> vec4f {
    let rk = max(SM.res >> k, 1u);
    let c = clamp(tc / f32(1u << k) - 0.5, vec2f(0.0), vec2f(f32(rk - 1u)));
    let i0 = vec2u(floor(c));
    let i1 = min(i0 + 1u, vec2u(rk - 1u));
    let f = c - vec2f(i0);
    let base = l * SM.layer_texels + sm_level_off(k);
    let a = mix(mip[base + i0.y * rk + i0.x], mip[base + i0.y * rk + i1.x], f.x);
    let b = mix(mip[base + i1.y * rk + i0.x], mip[base + i1.y * rk + i1.x], f.x);
    return mix(a, b, f.y);
}

fn sm_mip_tri(l: u32, lv: f32, tc: vec2f) -> vec4f {
    let lc = clamp(lv, 0.0, f32(SM.levels - 1u));
    let k0 = u32(floor(lc));
    let k1 = min(k0 + 1u, SM.levels - 1u);
    return mix(sm_mip_at(l, k0, tc), sm_mip_at(l, k1, tc), lc - f32(k0));
}

fn sm_phi(x: f32) -> f32 {
    let a = abs(x) * 0.70710678;
    let h = 0.5 * erfc_pos(a);
    return select(h, 1.0 - h, x >= 0.0);
}

// ---- lookup ---------------------------------------------------------------------------------------------------

@compute @workgroup_size(64)
fn sm_lookup(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    let n = U.gaussian_count;
    if (g >= n) { return; }
    let G = gaussians_rest[g];
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    var T = vec4f(1.0);
    for (var l = 0u; l < SM.n_layers; l++) {
        let ly = layers[l];
        if (ly.part == own || SM.lights[ly.light].w < 0.0) { continue; }
        let w = SM.lights[ly.light].xyz;
        let uv = vec2f(dot(P, ly.t1), dot(P, ly.t2)) - ly.c;
        if (any(abs(uv) >= vec2f(ly.half))) { continue; }
        let z = ly.zref - dot(P, w);
        if (z <= 0.0) { continue; }                        // above the part's top: nothing of it lies sunward
        let texel = 2.0 * ly.half / f32(SM.res);
        let tc = (uv + ly.half) / texel;
        let sig = SM.lights[ly.light].w;
        let search = 2.0 * sig * z;                       // widest penumbra (2 sigma) from this part, metres
        let a0 = sm_mip_tri(l, log2(max(2.0 * search / texel, 1.0)), tc);
        if (a0.x < 1e-4) { continue; }
        let zb = a0.y / a0.x;
        let pen = sig * max(z - zb, 0.0);
        let s = sm_mip_tri(l, log2(max(2.0 * pen / texel, 1.0)), tc);
        if (s.x < 1e-5) { continue; }
        let mu = s.y / s.x;
        let sd = sqrt(max(s.z / s.x - mu * mu, 1e-8));
        T[ly.light] *= 1.0 - s.x * sm_phi((z - mu) / sd);
    }
    env_T[g] = env_T[n + g] * T;
}
