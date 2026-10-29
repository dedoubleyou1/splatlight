// Light-side deterministic shadows (shadow light-map thread, 2026-10-25). Prepend common.wgsl.
//
// Two replacements for walk_emit + ema: each writes the other-part transmittance toward the light straight into
// lamp_state slot 0 (bitcast T, lamp 0, count 1), which shade_emit and direct_dirs_emit already read. Nothing is
// sampled, so there is no history to settle: the same scene gives the same T every frame.
//
// The light is seen from one point, LS.center (the emitter tree root), in an octahedral map around LS.normal
// (hemisphere-octahedral for one-sided lights: the front hemisphere fills the square; full octahedral otherwise).
//
// A. Per-part opacity map (lsm_*): every Gaussian of a dirty part is splatted from the light into its part's layer,
//    adding its optical depth tau and tau-weighted depth moments (fixed point, atomics: no sort, tau just adds).
//    lsm_resolve turns each texel into (alpha, alpha mu, alpha m2) with alpha = 1 - exp(-tau): alpha filters like
//    coverage, so lsm_down's box mips are a valid pre-filter. A receiver skips its own part's layer (its V grid has
//    that), finds the blocker depth at a coarse level, picks the level whose texel matches the penumbra
//    sigma_L (z - z_b) / (z_b z), and takes T_p = 1 - alpha Phi((z - mu) / sd). T = prod T_p.
// B. Gaussian-lobe light map (lobe_*): a cut of each part's tree as seen from the light (a node is used when its
//    angular radius from the light is under LS.cut_theta, or it is a leaf), binned into LS.tiles^2 octahedral tiles
//    widened by the light's own radius. A receiver walks its tile: each lobe of another part, seen from the receiver,
//    is a 2D Gaussian in the tangent plane of the direction to the light, as is the light (P3's ASG, LS.lc). The
//    fraction of the light it blocks is P3's closed-form overlap of the two, with the occluder's flat top
//    1 - exp(-tau) widened by Ein(tau0) / (1 - e^-tau0) into one Gaussian. Within a part the fractions add, divided by
//    the part's mean surface count along a ray (LS.layers); parts multiply: T = prod_p (1 - min(sum b / layers, 1)). A lobe
//    is replaced by its subtree where it is wider than LS.refine x the light's spread at its distance (too_coarse):
//    contact detail near receivers, coarse lobes far from them.

struct LSParams {
    center: vec3f, res: u32,          // light point, map resolution (level 0, texels per side)
    normal: vec3f, levels: u32,       // emitting axis (map centre), mip level count
    t1: vec3f, layer_texels: u32,     // light-frame tangents; texels per layer over all mip levels
    t2: vec3f, dirty: u32,            // bit p: re-splat part p this frame (A)
    sigma_l: f32, skip_part: u32, hemi: u32, part_count: u32,   // light std (isotropic, m); the emitters' own part
    cut_theta: f32, tiles: u32, tile_cap: u32, refine: f32,     // B: cut rule, tiles per side, slots per tile, refine
    lc_diag: vec3f, max_lobes: u32,   // B: the light's ASG covariance (world, P3 widened), lobe list capacity
    lc_off: vec3f, budget: u32,       // B: node expansions per receiver for contact refinement
    slices: u32, d0: f32, dlog: f32, combine: u32,   // combine: within-part rule (lobe_lookup)   // B: receiver-distance slices, slice k ends at d0 exp((k + 1) dlog)
    level_off: array<vec4u, 4>,       // A: first texel of each mip level within a layer
    layers: array<vec4f, 8>,          // B: per part, the mean number of its surfaces a ray crosses (host estimate)
}
struct LSLevel { level: u32, res: u32, pad0: u32, pad1: u32 }

// B: one cut node as seen from the light, world space (64 B)
struct Lobe {
    mean: vec3f, tau_k: f32,          // tau_k = kappa * dirm toward the light
    c_diag: vec3f, detc: f32,
    c_off: vec3f, part: u32,
    radius: f32, node: u32, pad0: u32, pad1: u32,
}

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(4) var<storage, read> parts: array<Part>;
@group(0) @binding(6) var<storage, read_write> lamp_state: array<LampState>;
@group(0) @binding(9) var<storage, read> kids: array<u32>;
@group(0) @binding(35) var<storage, read> node_parent: array<u32>;      // shadow_cull.wgsl hs_box_up (part roots: none)
@group(0) @binding(40) var<uniform> LS: LSParams;
@group(0) @binding(41) var<storage, read_write> acc: array<atomic<u32>>; // A: [layer][texel][3] fixed point
@group(0) @binding(42) var<storage, read_write> mip: array<vec4f>;       // A: [layer][level texels] (a, a mu, a m2, 0)
@group(0) @binding(43) var<storage, read_write> lobes: array<Lobe>;
@group(0) @binding(44) var<storage, read_write> tile_count: array<atomic<u32>>;   // [slices][tiles^2], then the lobe counter
@group(0) @binding(45) var<storage, read_write> tile_items: array<u32>;   // [slices][tiles^2][tile_cap] lobe indices
@group(0) @binding(46) var<uniform> LV: LSLevel;

const FIX: f32 = 65536.0;             // A: fixed-point scale of the atomics
const MAX_HALF: i32 = 48;             // A: splat footprint half-size cap in texels (light-adjacent Gaussians)
const LOWPASS: f32 = 0.3;             // A: texel^2 added to every splat (3DGS low-pass, integral kept)
const NONE: u32 = 0xFFFFFFFFu;

// ---- light frame and octahedral map ---------------------------------------------------------------------------

// world direction (unit) -> map coordinates in [-1, 1]^2
fn oct_uv(w: vec3f) -> vec2f {
    let l = vec3f(dot(w, LS.t1), dot(w, LS.t2), dot(w, LS.normal));
    if (LS.hemi != 0u) {
        let z = max(l.z, 0.0);
        let p = l.xy / max(abs(l.x) + abs(l.y) + z, 1e-12);
        return vec2f(p.x + p.y, p.x - p.y);
    }
    var p = l.xy / max(abs(l.x) + abs(l.y) + abs(l.z), 1e-12);
    if (l.z < 0.0) { p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0)); }
    return p;
}

fn ls_basis(w: vec3f) -> mat2x3f {
    let k = basis_k(w);
    return mat2x3f(k, cross(w, k));
}

// d uv / d angle at w along the basis (columns: the two tangent directions)
fn oct_jac(w: vec3f, b: mat2x3f) -> mat2x2f {
    let e = 1e-3;
    let u0 = oct_uv(w);
    let ua = oct_uv(normalize(w + e * b[0]));
    let ub = oct_uv(normalize(w + e * b[1]));
    return mat2x2f((ua - u0) / e, (ub - u0) / e);
}

fn inv2(m: mat2x2f) -> mat2x2f {
    let d = determinant(m);
    return mat2x2f(vec2f(m[1].y, -m[0].y), vec2f(-m[1].x, m[0].x)) * (1.0 / d);
}

fn phi(x: f32) -> f32 {
    let a = abs(x) * 0.70710678;
    let h = 0.5 * erfc_pos(a);
    return select(h, 1.0 - h, x >= 0.0);
}

// ---- covariances --------------------------------------------------------------------------------------------

fn part_rot(xf: array<vec4f, 3>) -> mat3x3f {
    return transpose(mat3x3f(xf[0].xyz, xf[1].xyz, xf[2].xyz));   // columns of R from its rows
}

fn sym3(dg: vec3f, off: vec3f) -> mat3x3f {
    return mat3x3f(vec3f(dg.x, off.x, off.y), vec3f(off.x, dg.y, off.z), vec3f(off.y, off.z, dg.z));
}

fn inv3(m: mat3x3f) -> mat3x3f {
    let r0 = cross(m[1], m[2]); let r1 = cross(m[2], m[0]); let r2 = cross(m[0], m[1]);
    return transpose(mat3x3f(r0, r1, r2)) * (1.0 / dot(m[0], r0));
}

// world covariance of node i (a leaf from its Gaussian's quat and scale, a parent from C^-1)
fn node_cov(i: u32, R: mat3x3f) -> mat3x3f {
    var C: mat3x3f;
    if (i < U.gaussian_count) {
        let G = gaussians[i];
        let Rq = quat_matrix(G.quat);
        let s2 = G.scale * G.scale;
        C = Rq * mat3x3f(vec3f(s2.x, 0.0, 0.0), vec3f(0.0, s2.y, 0.0), vec3f(0.0, 0.0, s2.z)) * transpose(Rq);
    } else {
        let nd = nodes[i];
        C = inv3(sym3(nd.cinv_diag, nd.cinv_off));
    }
    return R * C * transpose(R);
}

fn proj2(C: mat3x3f, b: mat2x3f) -> vec3f {   // (b0 C b0, b0 C b1, b1 C b1)
    let c0 = C * b[0];
    let c1 = C * b[1];
    return vec3f(dot(b[0], c0), dot(b[0], c1), dot(b[1], c1));
}

// ---- A: per-part opacity map ---------------------------------------------------------------------------------

fn texels() -> u32 { return LS.res * LS.res; }

@compute @workgroup_size(64)
fn lsm_clear(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = texels() * 3u;
    if (i >= n * LS.part_count) { return; }
    if ((LS.dirty & (1u << (i / n))) == 0u) { return; }
    atomicStore(&acc[i], 0u);
}

// Motion-blur shadows from a cut (?blurcut=<k>, David 2026-10-29): stretching every Gaussian of a moving part and
// keeping its mass leaves a stack of them as opaque as before, so the swept strip shadows at full strength instead of
// on average (mean of e^-tau, not e^-mean tau). A moving part instead splats a cut of its tree whose nodes are about
// k x its smear |d| across (bounding radius <= k |d| < the parent's), so few nodes overlap, and each stretched node
// keeps its coverage, not its density: alpha' = alpha_rest x (rest footprint / stretched footprint). Slow parts cut
// down to the leaves; still parts splat their leaves as before.
struct BlurCut { shutter: f32, k: f32, node_count: u32, pad: u32 }
@group(0) @binding(52) var<storage, read> gaussians_rest: array<Gaussian>;
@group(0) @binding(53) var<storage, read> nodes_rest: array<Node>;
@group(0) @binding(54) var<uniform> BC: BlurCut;

// part p's smear this exposure (world length at its root's centre); 0 = still or the cut is off
fn smear_len(p: u32) -> f32 {
    if (BC.k <= 0.0 || BC.shutter <= 0.0 || (parts[p].flags & 2u) == 0u) { return 0.0; }
    let P = parts[p];
    let x = nodes_rest[P.root].mean;
    return BC.shutter * length(xf_point(P.xf, x) - xf_point(P.xf_prev, x));
}

fn node_cov_rest(i: u32, R: mat3x3f) -> mat3x3f {
    var C: mat3x3f;
    if (i < U.gaussian_count) {
        let G = gaussians_rest[i];
        let Rq = quat_matrix(G.quat);
        let s2 = G.scale * G.scale;
        C = Rq * mat3x3f(vec3f(s2.x, 0.0, 0.0), vec3f(0.0, s2.y, 0.0), vec3f(0.0, 0.0, s2.z)) * transpose(Rq);
    } else {
        let nd = nodes_rest[i];
        C = inv3(sym3(nd.cinv_diag, nd.cinv_off));
    }
    return R * C * transpose(R);
}

@compute @workgroup_size(64)
fn lsm_splat(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count && g >= BC.node_count) { return; }
    let leaf = g < U.gaussian_count;
    var p: u32;
    if (leaf) { p = gaussians[g].part_surface & 0xFFFFu; } else { p = nodes_rest[g].part; }
    if (p >= LS.part_count || p == LS.skip_part || (LS.dirty & (1u << p)) == 0u) { return; }
    let L = smear_len(p);
    if (L <= 0.0) {
        if (leaf) { splat_one(g, p, false); }
        return;
    }
    let thr = BC.k * L;
    if (!leaf && nodes_rest[g].radius > thr) { return; }
    let par = node_parent[g];
    if (g != parts[p].root && par != NONE && nodes_rest[par].radius <= thr) { return; }
    splat_one(g, p, true);
}

// splat record g (a leaf or, for a moving part's cut, a tree node) into part p's layer; cover: keep coverage
fn splat_one(g: u32, p: u32, cover: bool) {
    let xf = parts[p].xf;
    let R = part_rot(xf);
    let nd = nodes[g];
    var mu = nd.mean;
    if (g < U.gaussian_count) { mu = gaussians[g].mu; }
    let m = xf_point(xf, mu);
    let dv = m - LS.center;
    let r = length(dv);
    if (r < 1e-4) { return; }
    let w = dv / r;
    let C = node_cov(g, R);
    if (LS.hemi != 0u && dot(dv, LS.normal) < -3.0 * sqrt(dot(LS.normal, C * LS.normal))) { return; }   // behind a one-sided light
    let b = ls_basis(w);
    let cbb = proj2(C, b);
    let cw = C * w;
    let cbz = vec2f(dot(b[0], cw), dot(b[1], cw));
    let czz = dot(w, cw);
    let Cbb = mat2x2f(vec2f(cbb.x, cbb.y), vec2f(cbb.y, cbb.z));
    let Cbb_i = inv2(Cbb);
    let sz2 = max(czz - dot(cbz, Cbb_i * cbz), 1e-12);      // depth variance along the ray (= 1 / w^T C^-1 w)
    var tau0 = nd.kappa * dirm(nd, xf_dir_inv(xf, w)) * sqrt(2.0 * PI * sz2);
    // angle -> texel coordinates
    let half_res = 0.5 * f32(LS.res);
    let J = oct_jac(w, b) * half_res;
    let Sang = Cbb * (1.0 / (r * r));
    var S = J * Sang * transpose(J);
    let det0 = determinant(S);
    S = S + mat2x2f(vec2f(LOWPASS, 0.0), vec2f(0.0, LOWPASS));
    let det1 = determinant(S);
    tau0 *= sqrt(max(det0, 0.0) / det1);
    if (cover) {
        // the rest record's peak opacity, spread evenly over the stretched footprint: alpha' A' = alpha A
        let ndr = nodes_rest[g];
        let Cr = node_cov_rest(g, R);
        let cwr = Cr * w;
        let cbr = proj2(Cr, b);
        let Cbr = mat2x2f(vec2f(cbr.x, cbr.y), vec2f(cbr.y, cbr.z));
        let cbzr = vec2f(dot(b[0], cwr), dot(b[1], cwr));
        let szr = max(dot(w, cwr) - dot(cbzr, inv2(Cbr) * cbzr), 1e-12);
        var Sr = J * (Cbr * (1.0 / (r * r))) * transpose(J);
        let d0r = determinant(Sr);
        Sr = Sr + mat2x2f(vec2f(LOWPASS, 0.0), vec2f(0.0, LOWPASS));
        let d1r = determinant(Sr);
        let tau_r = ndr.kappa * dirm(ndr, xf_dir_inv(xf, w)) * sqrt(2.0 * PI * szr) * sqrt(max(d0r, 0.0) / d1r);
        let a = min((1.0 - exp(-tau_r)) * sqrt(max(d1r, 1e-30) / max(det1, 1e-30)), 0.999);
        tau0 = min(-log(1.0 - a), tau0);
    }
    let Si = inv2(S);
    // depth along the ray at texel offset dt: r + (r Cbb^-1 cbz) . (J^-1 dt)
    let gz = transpose(inv2(J)) * (Cbb_i * cbz * r);
    let uv = oct_uv(w);
    let ct = (uv * 0.5 + 0.5) * f32(LS.res);
    let hx = min(i32(ceil(3.0 * sqrt(S[0].x))), MAX_HALF);
    let hy = min(i32(ceil(3.0 * sqrt(S[1].y))), MAX_HALF);
    let cx = i32(floor(ct.x));
    let cy = i32(floor(ct.y));
    let base = p * texels();
    let res = i32(LS.res);
    for (var y = max(cy - hy, 0); y <= min(cy + hy, res - 1); y++) {
        for (var x = max(cx - hx, 0); x <= min(cx + hx, res - 1); x++) {
            let dt = vec2f(f32(x) + 0.5, f32(y) + 0.5) - ct;
            let q = dot(dt, Si * dt);
            if (q > 9.0) { continue; }
            let t = min(tau0 * exp(-0.5 * q), 64.0);   // alpha is 1 long before; keeps the fixed-point sums in range
            let t_fix = t * FIX;
            if (t_fix < 0.5) { continue; }
            let z = max(r + dot(gz, dt), 0.0);
            let k = (base + u32(y) * LS.res + u32(x)) * 3u;
            atomicAdd(&acc[k], u32(t_fix + 0.5));
            atomicAdd(&acc[k + 1u], u32(t_fix * z + 0.5));
            atomicAdd(&acc[k + 2u], u32(t_fix * (z * z + sz2) + 0.5));
        }
    }
}

fn level_off(k: u32) -> u32 { return LS.level_off[k / 4u][k % 4u]; }

@compute @workgroup_size(64)
fn lsm_resolve(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = texels();
    if (i >= n * LS.part_count) { return; }
    let p = i / n;
    if ((LS.dirty & (1u << p)) == 0u) { return; }
    let s0 = f32(atomicLoad(&acc[i * 3u])) / FIX;
    var o = vec4f(0.0);
    if (s0 > 1e-5) {
        let a = 1.0 - exp(-s0);
        o = vec4f(a, a * f32(atomicLoad(&acc[i * 3u + 1u])) / (FIX * s0), a * f32(atomicLoad(&acc[i * 3u + 2u])) / (FIX * s0), 0.0);
    }
    mip[p * LS.layer_texels + (i - p * n)] = o;
}

// level LV.level (side LV.res) from the level above: the mean of its 4 children
@compute @workgroup_size(64)
fn lsm_down(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = LV.res * LV.res;
    if (i >= n * LS.part_count) { return; }
    let p = i / n;
    if ((LS.dirty & (1u << p)) == 0u) { return; }
    let j = i - p * n;
    let y = j / LV.res;
    let x = j - y * LV.res;
    let pr = LV.res * 2u;
    let src = p * LS.layer_texels + level_off(LV.level - 1u) + (2u * y) * pr + 2u * x;
    let s = mip[src] + mip[src + 1u] + mip[src + pr] + mip[src + pr + 1u];
    mip[p * LS.layer_texels + level_off(LV.level) + j] = 0.25 * s;
}

// bilinear at level k; tc in level-0 texel units
fn mip_at(p: u32, k: u32, tc: vec2f) -> vec4f {
    let rk = max(LS.res >> k, 1u);
    let c = clamp(tc / f32(1u << k) - 0.5, vec2f(0.0), vec2f(f32(rk - 1u)));
    let i0 = vec2u(floor(c));
    let i1 = min(i0 + 1u, vec2u(rk - 1u));
    let f = c - vec2f(i0);
    let base = p * LS.layer_texels + level_off(k);
    let a = mix(mip[base + i0.y * rk + i0.x], mip[base + i0.y * rk + i1.x], f.x);
    let b = mix(mip[base + i1.y * rk + i0.x], mip[base + i1.y * rk + i1.x], f.x);
    return mix(a, b, f.y);
}

fn mip_tri(p: u32, l: f32, tc: vec2f) -> vec4f {
    let lc = clamp(l, 0.0, f32(LS.levels - 1u));
    let k0 = u32(floor(lc));
    let k1 = min(k0 + 1u, LS.levels - 1u);
    return mix(mip_at(p, k0, tc), mip_at(p, k1, tc), lc - f32(k0));
}

fn write_t(g: u32, T: f32) {
    lamp_state[g].slots[0] = vec2u(bitcast<u32>(T), 0u | (1u << 16u));
}

@compute @workgroup_size(64)
fn lsm_lookup(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= U.gaussian_count) { return; }
    lsm_lookup_one(gid.x);
}

fn lsm_lookup_one(g: u32) {
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    let dv = P - LS.center;
    let z = length(dv);
    if (z < 1e-4) { write_t(g, 1.0); return; }
    let w = dv / z;
    let b = ls_basis(w);
    let J = oct_jac(w, b);
    let tex_ang = (2.0 / f32(LS.res)) / sqrt(max(abs(determinant(J)), 1e-12));
    let tc = (oct_uv(w) * 0.5 + 0.5) * f32(LS.res);
    var T = 1.0;
    for (var p = 0u; p < LS.part_count; p++) {
        if (p == own || p == LS.skip_part) { continue; }
        // the part's bounding sphere must lie in front of the receiver, near its direction
        let bs = parts[p].bound;
        let bc = xf_point(parts[p].xf, bs.xyz) - LS.center;
        let dc = length(bc);
        let zn = max(dc - bs.w, 0.02);
        if (z <= zn) { continue; }
        let search = 2.0 * LS.sigma_l * (z - zn) / (zn * z);        // widest penumbra (2 sigma) from this part
        if (dc > bs.w) {
            let ang = acos(clamp(dot(bc / dc, w), -1.0, 1.0));
            if (ang > asin(bs.w / dc) + search + 2.0 * tex_ang) { continue; }
        }
        let ls = log2(max(2.0 * search / tex_ang, 1.0));
        let a0 = mip_tri(p, ceil(ls), tc);
        if (a0.x < 1e-4) { continue; }
        let zb = max(a0.y / a0.x, 0.02);
        let pen = LS.sigma_l * max(z - zb, 0.0) / (zb * z);         // penumbra std, radians from the light
        // a level-k texel read bilinearly is a box of 2^k texels convolved with a tent of 2^k: std ~ 0.5 x 2^k
        let s = mip_tri(p, log2(max(2.0 * pen / tex_ang, 1.0)), tc);
        if (s.x < 1e-5) { continue; }
        let mu = s.y / s.x;
        let sd = sqrt(max(s.z / s.x - mu * mu, 1e-8));
        T *= 1.0 - s.x * phi((z - mu) / sd);
    }
    write_t(g, T);
}

// ---- B: Gaussian-lobe light map -------------------------------------------------------------------------------

fn tile_of(w: vec3f) -> u32 {
    let t = clamp(vec2u((oct_uv(w) * 0.5 + 0.5) * f32(LS.tiles)), vec2u(0u), vec2u(LS.tiles - 1u));
    return t.y * LS.tiles + t.x;
}

@compute @workgroup_size(64)
fn lobe_clear(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i <= LS.slices * LS.tiles * LS.tiles) { atomicStore(&tile_count[i], 0u); }   // the last word is the lobe counter
}

fn accept(i: u32) -> bool {
    if (i < U.gaussian_count) { return true; }
    let nd = nodes[i];
    let m = xf_point(parts[nd.part].xf, nd.mean);
    return nd.radius < LS.cut_theta * length(m - LS.center);
}

@compute @workgroup_size(64)
fn lobe_cut(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= arrayLength(&nodes)) { return; }
    let nd = nodes[i];
    let p = nd.part;
    if (p == NONE || p == LS.skip_part || !accept(i)) { return; }
    // in the cut when no ancestor is accepted
    var a = node_parent[i];
    for (var k = 0u; k < 32u && a != NONE; k++) {
        if (accept(a)) { return; }
        a = node_parent[a];
    }
    let xf = parts[p].xf;
    let m = xf_point(xf, nd.mean);
    let dv = m - LS.center;
    let r = max(length(dv), 1e-4);
    let w = dv / r;
    let C = node_cov(i, part_rot(xf));
    // a one-sided light's rays only leave its front: an occluder more than 3 sigma behind its plane cannot touch them
    if (LS.hemi != 0u && dot(dv, LS.normal) < -3.0 * sqrt(dot(LS.normal, C * LS.normal))) { return; }
    let nbin = LS.tiles * LS.tiles;
    let idx = atomicAdd(&tile_count[LS.slices * nbin], 1u);
    if (idx >= LS.max_lobes) { return; }
    lobes[idx] = Lobe(m, nd.kappa * dirm(nd, xf_dir_inv(xf, w)), vec3f(C[0].x, C[1].y, C[2].z), determinant(C),
                      vec3f(C[1].x, C[2].x, C[2].y), p, nd.radius, i, 0u, 0u);
    // bins: per receiver-distance slice, the tiles whose receivers' rays to any light point can pass within the lobe's
    // radius. A ray from light point c + l to a receiver at distance D through the lobe at r leaves the lobe's direction
    // by at most (radius + |l| (D - r) / D) / r, so near receivers need little widening, far ones the light's radius.
    let J = oct_jac(w, ls_basis(w));
    let jx = length(vec2f(J[0].x, J[1].x));
    let jy = length(vec2f(J[0].y, J[1].y));
    let uv = oct_uv(w);
    let nt = i32(LS.tiles);
    for (var sl = 0u; sl < LS.slices; sl++) {
        let dhi = select(LS.d0 * exp(f32(sl + 1u) * LS.dlog), 1e30, sl + 1u == LS.slices);
        if (dhi < r - nd.radius) { continue; }                       // every receiver of this slice is nearer the light
        let rho = (nd.radius + 3.0 * LS.sigma_l * max(min(dhi, 1e3) - r, 0.0) / min(dhi, 1e3)) / r;
        var lo = vec2i(0);
        var hi = vec2i(nt - 1);
        if (rho < 0.6 && r > nd.radius) {
            let ext = 1.25 * rho * vec2f(jx, jy);
            lo = max(vec2i(floor(((uv - ext) * 0.5 + 0.5) * f32(nt))), vec2i(0));
            hi = min(vec2i(floor(((uv + ext) * 0.5 + 0.5) * f32(nt))), vec2i(nt - 1));
        }
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                let t = sl * nbin + u32(y) * LS.tiles + u32(x);
                let k = atomicAdd(&tile_count[t], 1u);
                if (k < LS.tile_cap) { tile_items[t * LS.tile_cap + k] = idx; }
            }
        }
    }
}

struct Recv2 { P: vec3f, wl: vec3f, D: f32, b: mat2x3f, sl: vec3f }

// the fraction of the light one occluder Gaussian blocks (world mean, covariance) seen from the receiver
fn lobe_b(rv: Recv2, mean: vec3f, C: mat3x3f, tau_k: f32, detc: f32) -> vec3f {
    let e = mean - rv.P;
    let cw = C * rv.wl;
    let st = sqrt(max(dot(rv.wl, cw), 1e-12));
    let t = dot(e, rv.wl);
    let front = phi(t / st) * phi((rv.D - t) / st);
    if (front < 1e-3) { return vec3f(0.0); }
    let te = max(t, 0.5 * st);
    let cbb = proj2(C, rv.b);                         // unscaled marginal across the ray
    let dbb = max(cbb.x * cbb.z - cbb.y * cbb.y, 1e-30);
    let tau0 = tau_k * sqrt(2.0 * PI * max(detc, 0.0) / dbb);
    if (tau0 < 1e-4) { return vec3f(0.0); }
    let a0 = 1.0 - exp(-tau0);
    let m = vec2f(dot(rv.b[0], e), dot(rv.b[1], e)) / te;
    let so1 = cbb * (1.0 / (te * te));
    // x: the flat top 1 - exp(-tau) widened by Ein(tau0) / (1 - e^-tau0) into one Gaussian, overlapped with the light
    // y: the light-averaged optical depth (no saturation); z: the flat top at the plain footprint (no widening)
    var out = vec3f(0.0);
    for (var k = 0u; k < 2u; k++) {
        let kw = select(1.0, ein(tau0) / max(a0, 1e-12), k == 0u);
        let so = so1 * kw;
        let sm = rv.sl + so;
        let ds = sm.x * sm.z - sm.y * sm.y;
        let q = (sm.z * m.x * m.x - 2.0 * sm.y * m.x * m.y + sm.x * m.y * m.y) / ds;
        if (q > 25.0) { continue; }
        let ov = front * sqrt(max(so.x * so.z - so.y * so.y, 0.0) / ds) * exp(-0.5 * q);
        if (k == 0u) { out.x = a0 * ov; } else { out.y = tau0 * ov; out.z = a0 * ov; }
    }
    return out;
}

// A merged node is too coarse for a receiver when it is wider than refine x the light's spread at its distance t from
// the receiver (sigma_L t / D): detail finer than that is blurred out by the penumbra anyway.
fn too_coarse(ni: u32, radius: f32, t: f32, D: f32) -> bool {
    return ni >= U.gaussian_count && LS.refine > 0.0 && radius > LS.refine * LS.sigma_l * max(t, 0.0) / D;
}

// a lobe's subtree, opened where too_coarse (at most LS.budget node expansions per receiver; past that, nodes are
// taken as they are)
fn subtree_b(rv: Recv2, root: u32, budget: ptr<function, u32>) -> vec3f {
    var stack: array<u32, STACK>;
    stack[0] = root;
    var sp = 1u;
    var acc_l = vec3f(0.0);
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        let xf = parts[nd.part].xf;
        let m = xf_point(xf, nd.mean);
        // the ray toward the light must pass within the node's radius + the light's spread there
        let e = m - rv.P;
        let t = clamp(dot(e, rv.wl), 0.0, rv.D);
        let off = e - t * rv.wl;
        let lr = 3.0 * LS.sigma_l * t / rv.D;
        if (dot(off, off) > (nd.radius + lr) * (nd.radius + lr)) { continue; }
        if (too_coarse(ni, nd.radius, t, rv.D) && *budget > 0u && sp + (nd.child_hi - nd.child_lo) <= STACK) {
            *budget -= 1u;
            for (var j = nd.child_lo; j < nd.child_hi; j++) { stack[sp] = kids[j]; sp += 1u; }
            continue;
        }
        let C = node_cov(ni, part_rot(xf));
        acc_l += lobe_b(rv, m, C, nd.kappa * dirm(nd, xf_dir_inv(xf, rv.wl)), determinant(C));
    }
    return acc_l;
}

@compute @workgroup_size(64)
fn lobe_lookup(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    let dv = P - LS.center;
    let D = length(dv);
    if (D < 1e-4) { write_t(g, 1.0); return; }
    let wl = -dv / D;
    let b = ls_basis(wl);
    let sl = proj2(sym3(LS.lc_diag, LS.lc_off), b) * (1.0 / (D * D));
    let rv = Recv2(P, wl, D, b, sl);
    let ds = min(u32(max(log(max(D, 1e-6) / LS.d0) / LS.dlog, 0.0)), LS.slices - 1u);
    let t = ds * LS.tiles * LS.tiles + tile_of(-wl);
    let n = min(atomicLoad(&tile_count[t]), LS.tile_cap);
    var budget = LS.budget;
    var bsum: array<vec3f, 32>;        // per part: summed lobe_b terms
    for (var p = 0u; p < 32u; p++) { bsum[p] = vec3f(0.0); }
    for (var k = 0u; k < n; k++) {
        let L = lobes[tile_items[t * LS.tile_cap + k]];
        if (L.part == own) { continue; }
        if (budget > 0u && too_coarse(L.node, L.radius, dot(L.mean - P, wl), D)) {
            bsum[L.part] += subtree_b(rv, L.node, &budget);
            continue;
        }
        bsum[L.part] += lobe_b(rv, L.mean, sym3(L.c_diag, L.c_off), L.tau_k, L.detc);
    }
    // a part's lobes tile its surfaces, so their blocked fractions add (they do not multiply as independent occluders
    // would); a ray crosses LS.layers[p] of its surfaces on average (2 for a closed box, 1 for a wall)
    var T = 1.0;
    for (var p = 0u; p < LS.part_count; p++) {
        let nu = LS.layers[p / 4u][p % 4u];
        if (LS.combine == 1u) { T *= exp(-bsum[p].y); }
        else if (LS.combine == 2u) { T *= 1.0 - min(bsum[p].z / nu, 1.0); }
        else { T *= 1.0 - min(bsum[p].x / nu, 1.0); }
    }
    write_t(g, T);
}

// ---- C: light-space lobes with a cut per distance slice (B simplified) -------------------------------------------
// B's per-receiver tree walk is replaced by one cut per receiver-distance slice: for receivers at distance D (slice
// start) a node is fine enough when it is no wider than refine x the light's spread at its distance behind it,
// sigma_L (D - r) / D, and under cut_theta of angle from the light. Each lobe is stored in the light's direction space
// (unit direction w, a tangent basis, its angular covariance S and the lamp's covariance in that basis); a receiver at
// distance D sees its penumbra as S + L2 ((D - r) / (D r))^2, the same closed-form overlap as B up to the shared scale,
// so evaluating a lobe is one 2x2 Gaussian. Bins are (slice, tile, part) and compacted (count, scan, scatter), so a
// receiver reads only other parts' lobes (most lobes in a wall's direction are that wall's own). LS.tile_cap is the item
// capacity here; lc_count holds [slices][tiles^2][parts] counts, then the lobe and item totals.

struct LLobe {
    w: vec3f, r: f32,                 // direction from the light, distance
    b0: vec3f, sr: f32,               // tangent (b1 = w x b0), depth std along w
    s: vec3f, a0: f32,                // angular covariance (b0b0, b0b1, b1b1); peak opacity 1 - exp(-tau0) along w
    l2: vec3f, part: u32,             // the lamp's covariance in (b0, b1), unscaled
    bins: vec4u,                      // slice, lo.x | lo.y << 16, hi.x | hi.y << 16, 0
}

@group(0) @binding(47) var<storage, read_write> lc_lobes: array<LLobe>;
@group(0) @binding(48) var<storage, read_write> lc_count: array<atomic<u32>>;
@group(0) @binding(49) var<storage, read_write> lc_start: array<u32>;
@group(0) @binding(50) var<storage, read_write> lc_cursor: array<atomic<u32>>;
@group(0) @binding(51) var<storage, read_write> lc_items: array<u32>;

fn nbins() -> u32 { return LS.slices * LS.tiles * LS.tiles * LS.part_count; }
fn bin_of(k: u32, t: u32, p: u32) -> u32 { return (k * LS.tiles * LS.tiles + t) * LS.part_count + p; }
fn slice_lo(k: u32) -> f32 { return LS.d0 * exp(f32(k) * LS.dlog); }
fn slice_hi(k: u32) -> f32 { return select(LS.d0 * exp(f32(k + 1u) * LS.dlog), 1e3, k + 1u == LS.slices); }

@compute @workgroup_size(64)
fn lc_clear(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x < nbins() + 2u) { atomicStore(&lc_count[gid.x], 0u); }
}

fn accept_c(i: u32, D: f32) -> bool {
    if (i < U.gaussian_count) { return true; }
    let nd = nodes[i];
    let r = length(xf_point(parts[nd.part].xf, nd.mean) - LS.center);
    return nd.radius < LS.cut_theta * r && nd.radius <= LS.refine * LS.sigma_l * max(D - r, 0.0) / D;
}

// one thread per (node, slice)
@compute @workgroup_size(64)
fn lc_cut(@builtin(global_invocation_id) gid: vec3u) {
    let N = arrayLength(&nodes);
    let k = gid.x / N;
    let i = gid.x - k * N;
    if (k >= LS.slices) { return; }
    let nd = nodes[i];
    let p = nd.part;
    if (p == NONE || p == LS.skip_part) { return; }
    let xf = parts[p].xf;
    let m = xf_point(xf, nd.mean);
    let dv = m - LS.center;
    let r = max(length(dv), 1e-4);
    let Dlo = slice_lo(k);
    let Dhi = slice_hi(k);
    if (r - nd.radius > Dhi) { return; }                 // behind every receiver of this slice
    // in the cut when accepted and its parent is not (the rule shrinks monotonically down the tree up to the small
    // change in distance, so one parent test stands in for B's whole ancestor chain)
    if (!accept_c(i, Dlo)) { return; }
    let pa = node_parent[i];
    if (pa != NONE && accept_c(pa, Dlo)) { return; }
    let C = node_cov(i, part_rot(xf));
    if (LS.hemi != 0u && dot(dv, LS.normal) < -3.0 * sqrt(dot(LS.normal, C * LS.normal))) { return; }
    let w = dv / r;
    let b = ls_basis(w);
    let cbb = proj2(C, b);
    let Cbb = mat2x2f(vec2f(cbb.x, cbb.y), vec2f(cbb.y, cbb.z));
    let cw = C * w;
    let cbz = vec2f(dot(b[0], cw), dot(b[1], cw));
    let sz2 = max(dot(w, cw) - dot(cbz, inv2(Cbb) * cbz), 1e-12);
    let tau0 = nd.kappa * dirm(nd, xf_dir_inv(xf, w)) * sqrt(2.0 * PI * sz2);
    if (tau0 < 1e-4) { return; }
    // bins: tiles within the lobe's radius plus the lamp's spread at this slice's farthest receivers
    let rho = (nd.radius + 3.0 * LS.sigma_l * max(min(Dhi, 1e2) - r, 0.0) / min(Dhi, 1e2)) / r;
    let nt = i32(LS.tiles);
    var lo = vec2i(0);
    var hi = vec2i(nt - 1);
    if (rho < 0.6 && r > nd.radius) {
        let J = oct_jac(w, b);
        let ext = 1.25 * rho * vec2f(length(vec2f(J[0].x, J[1].x)), length(vec2f(J[0].y, J[1].y)));
        let uv = oct_uv(w);
        lo = max(vec2i(floor(((uv - ext) * 0.5 + 0.5) * f32(nt))), vec2i(0));
        hi = min(vec2i(floor(((uv + ext) * 0.5 + 0.5) * f32(nt))), vec2i(nt - 1));
    }
    if (any(hi < lo)) { return; }                        // off the map (and keeps the u32 ranges in lc_scatter valid)
    let idx = atomicAdd(&lc_count[nbins()], 1u);
    if (idx >= LS.max_lobes) { return; }
    let l2 = proj2(sym3(LS.lc_diag, LS.lc_off), b);
    lc_lobes[idx] = LLobe(w, r, b[0], sqrt(max(dot(w, cw), 1e-12)), cbb * (1.0 / (r * r)), 1.0 - exp(-tau0), l2, p,
                          vec4u(k, u32(lo.x) | (u32(lo.y) << 16u), u32(hi.x) | (u32(hi.y) << 16u), 1u));
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            atomicAdd(&lc_count[bin_of(k, u32(y) * LS.tiles + u32(x), p)], 1u);
        }
    }
}

// exclusive prefix sum of the bin counts (one workgroup)
var<workgroup> scan_part: array<u32, 256>;
@compute @workgroup_size(256)
fn lc_scan(@builtin(local_invocation_index) li: u32) {
    let n = nbins();
    let per = (n + 255u) / 256u;
    let b0 = li * per;
    var s = 0u;
    for (var j = b0; j < min(b0 + per, n); j++) { s += atomicLoad(&lc_count[j]); }
    scan_part[li] = s;
    workgroupBarrier();
    if (li == 0u) {
        var run = 0u;
        for (var j = 0u; j < 256u; j++) { let v = scan_part[j]; scan_part[j] = run; run += v; }
        atomicStore(&lc_count[n + 1u], run);
    }
    workgroupBarrier();
    var run = scan_part[li];
    for (var j = b0; j < min(b0 + per, n); j++) {
        lc_start[j] = run;
        atomicStore(&lc_cursor[j], run);
        run += atomicLoad(&lc_count[j]);
    }
}

@compute @workgroup_size(64)
fn lc_scatter(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= min(atomicLoad(&lc_count[nbins()]), LS.max_lobes)) { return; }
    let L = lc_lobes[i];
    let lo = vec2u(L.bins.y & 0xFFFFu, L.bins.y >> 16u);
    let hi = min(vec2u(L.bins.z & 0xFFFFu, L.bins.z >> 16u), vec2u(LS.tiles - 1u));
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let pos = atomicAdd(&lc_cursor[bin_of(L.bins.x, y * LS.tiles + x, L.part)], 1u);
            if (pos < LS.tile_cap) { lc_items[pos] = i; }
        }
    }
}

@compute @workgroup_size(64)
fn lc_lookup(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    let dv = P - LS.center;
    let D = length(dv);
    if (D < 1e-4) { write_t(g, 1.0); return; }
    let v = dv / D;
    let k = min(u32(max(log(max(D, 1e-6) / LS.d0) / LS.dlog, 0.0)), LS.slices - 1u);
    let tile = tile_of(v);
    var budget = LS.budget;
    var bsum: array<f32, 32>;
    for (var p = 0u; p < 32u; p++) { bsum[p] = 0.0; }
    for (var p = 0u; p < LS.part_count; p++) {
      if (p == own || p == LS.skip_part) { continue; }
      let bin = bin_of(k, tile, p);
      let s0 = lc_start[bin];
      let s1 = min(min(s0 + atomicLoad(&lc_count[bin]), LS.tile_cap), s0 + budget);   // LS.budget bounds a receiver's work
      budget -= s1 - min(s0, s1);
      for (var j = s0; j < s1; j++) {
        let L = lc_lobes[lc_items[j]];
        let t = D - L.r;                                  // receiver behind the lobe
        if (t < -3.0 * L.sr) { continue; }
        let c = dot(L.w, v);
        if (c < 0.1) { continue; }
        let b1 = cross(L.w, L.b0);
        let m = vec2f(dot(L.b0, v), dot(b1, v)) / c;
        let f = max(t, 0.0) / (D * L.r);
        let sm = L.s + L.l2 * (f * f);
        let ds = sm.x * sm.z - sm.y * sm.y;
        let q = (sm.z * m.x * m.x - 2.0 * sm.y * m.x * m.y + sm.x * m.y * m.y) / ds;
        if (q > 25.0) { continue; }
        let dS = max(L.s.x * L.s.z - L.s.y * L.s.y, 0.0);
        bsum[p] += L.a0 * phi(t / L.sr) * sqrt(dS / ds) * exp(-0.5 * q);
      }
    }
    var T = 1.0;
    for (var p = 0u; p < LS.part_count; p++) {
        T *= 1.0 - min(bsum[p] / LS.layers[p / 4u][p % 4u], 1.0);
    }
    write_t(g, T);
}
