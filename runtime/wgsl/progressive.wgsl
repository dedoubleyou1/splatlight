// Progressive tree lighting (David, 2026-10-23): incoming light is computed for tree nodes from the root down under a
// fixed per-frame budget, refined deeper where nothing changed and coarsened where something did; every leaf then
// shades itself each frame from its deepest lit ancestor. Prepend common.wgsl and material_ext.wgsl.
//
// Per node (persistent): the direct light as one directional light (radiance x solid angle, rgb, and its
// luminance-weighted direction; SH2 smeared a small lamp onto faces that cannot see it), the bounce as SH2 per colour,
// plus a shadow history and the refinement state. Slots per node: [0] direct rgb, [1] direct direction, [2..8] unused,
// [9..17] bounce SH2.
//   lp_schedule        per node: decide which nodes want shading this frame and put a random subset on the worklist
//                      (subset size = the budget, acceptance from last frame's candidate count)
//   lp_shade           per worklist node: direct light from the emitter light cut x the node's shadow history (one
//                      walked sample per shading), bounce from the 8 surrounding probes, both projected to SH2
//   lp_finish          one thread: next frame's acceptance probability, counters reset
//   lp_transfer        per leaf: its deepest lit ancestor's SH (blended in from that node's parent over its first
//                      frames), turned into radiance toward the eye with the leaf's own V grid and flakes
//   lp_transfer_dirs   the same toward the 14 fixed directions for the probes' node emitters (1 / 8 of leaves a frame)
// With U.iso bit 0 the leaf transfer is M39's direction-independent one (transfer_iso).
// The light is the asset's emissive part (light cut over emit_nodes, as emissive.wgsl). Only own-part visibility at the
// leaf; other parts shadow the direct light through the walk and the bounce through the probes.

struct EmitParams {
    count: u32, root: u32, scale: f32, theta: f32,
    mode: u32, node_count: u32, skip_part: u32, one_sided: u32,
    normal: vec3f,
    n5: u32,             // N5 opt-ins: bit 0 golden-ratio pick sequence, bit 1 sample the emitter's own covariance,
                         // bit 2 progressive: 4 samples per re-shade, count history, leaves ease toward it
    edge_soft: f32,      // one-sided fade width in cos (emissive.wgsl EmitParams); 0 = hard cut
    pad0: u32, pad1: u32, pad2: u32,
}
struct EmitNode {
    mean: vec3f, radius: f32,
    m_diag: vec3f, pad: f32,
    m_off: vec3f, child_lo: u32,
    Le: vec3f, child_hi: u32,
    c_diag: vec3f, pad1: f32,      // P3 ASG: the node's covariance with the leaves' flat-top widening folded in
    c_off: vec3f, pad2: f32,       // (xy, xz, yz); 96 B per node (lighting.js packEmitNodes)
}
struct ProbeParams {
    lo: vec3f, n_axis: u32,
    step: vec3f, node_count: u32,
    block_rad: f32, theta: f32, tmax: f32, near_r: f32,
    alpha: f32, force_near: u32, vfloor: f32, g_valid: f32,
    subset: u32, phase: u32, part_probes: u32, r1: u32,   // bounce.wgsl: part probes follow the 8^3 lattice
    dirs: array<vec4f, 14>,
    pinv: array<vec4f, 32>,
}
struct LPParams {
    budget: u32,          // nodes shaded per frame
    min_ang: f32,         // a node's children are refined while its radius / distance exceeds this
    refresh: u32,         // a lit node is re-shaded about every `refresh` frames (shadow history, moving light)
    reset_depth: u32,     // on reset, nodes deeper than this lose their light and refine again
    reset: u32,           // 1 = this frame something changed (light or part moved)
    blend: f32,           // frames over which a newly lit node fades in from its parent
    dir_phase: u32,       // lp_transfer_dirs handles leaves with g % 8 == dir_phase
    r1: u32,
}

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> vgrid: array<VGrid>;
@group(0) @binding(4) var<storage, read> parts: array<Part>;
@group(0) @binding(7) var<storage, read> materials: array<Material>;
@group(0) @binding(8) var<storage, read_write> colour: array<vec4f>;
@group(0) @binding(9) var<storage, read> kids: array<u32>;
@group(0) @binding(10) var<uniform> PP: ProbeParams;
@group(0) @binding(11) var<storage, read> probe_data: array<vec4f>;
@group(0) @binding(13) var<storage, read_write> lo_dir: array<vec4f>;
@group(0) @binding(14) var<storage, read_write> lo_b: array<vec4f>;
@group(0) @binding(18) var<storage, read> emit_nodes: array<EmitNode>;
@group(0) @binding(19) var<uniform> EP: EmitParams;
@group(0) @binding(21) var<uniform> LP: LPParams;
@group(0) @binding(22) var<storage, read_write> node_light: array<vec4f>;    // per node x 18: SH2 direct (9), bounce (9)
@group(0) @binding(23) var<storage, read_write> node_state: array<vec4u>;    // (bitcast T, count, age | lit << 31, parent)
@group(0) @binding(24) var<storage, read_write> work: array<atomic<u32>>;    // [0] listed, [1] candidates, [2] p bits, [3] -, [4..) list

const NCELL: u32 = 128u;
const ESTACK: u32 = 48u;
const NO_CHILD: u32 = 0xFFFFFFFFu;
const LIT: u32 = 0x80000000u;
const NQ: u32 = 32u;                   // quadrature directions for the leaf transfer
const SHN: u32 = 18u;

const DIRS14 = array<vec3f, 14>(
    vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(1e-4, 0.0, 1.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, -1.0, 0.0),
    vec3f(1e-4, 0.0, -1.0),
    vec3f(0.57735, 0.57735, 0.57735), vec3f(0.57735, 0.57735, -0.57735), vec3f(0.57735, -0.57735, 0.57735),
    vec3f(0.57735, -0.57735, -0.57735), vec3f(-0.57735, 0.57735, 0.57735), vec3f(-0.57735, 0.57735, -0.57735),
    vec3f(-0.57735, -0.57735, 0.57735), vec3f(-0.57735, -0.57735, -0.57735));

// ---- small helpers -------------------------------------------------------------------------------------------

fn sh2(d: vec3f) -> array<f32, 9> {
    let x = d.x; let y = d.y; let z = d.z;
    return array<f32, 9>(0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z,
                         0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y));
}

fn fib_dir(k: u32) -> vec3f {
    let z = 1.0 - (2.0 * f32(k) + 1.0) / f32(NQ);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let ph = f32(k) * 2.399963;
    return vec3f(r * cos(ph), r * sin(ph), z);
}

fn cell_dir(c: u32) -> vec3f {
    let i = c / 16u; let j = c % 16u;
    let el = 0.5 * PI - (f32(i) + 0.5) * PI / 8.0;
    let az = -PI + (f32(j) + 0.5) * 2.0 * PI / 16.0;
    return vec3f(cos(el) * cos(az), cos(el) * sin(az), sin(el));
}

fn cell_omega(c: u32) -> f32 {
    let top = 0.5 * PI - f32(c / 16u) * PI / 8.0;
    return (2.0 * PI / 16.0) * (sin(top) - sin(top - PI / 8.0));
}

fn cell_of(w: vec3f) -> u32 {
    let el = asin(clamp(w.z, -1.0, 1.0));
    let i = min(u32(max((0.5 * PI - el) / (PI / 8.0), 0.0)), 7u);
    var az = 0.0;
    if (abs(w.x) + abs(w.y) > 1e-30) { az = atan2(w.y, w.x); }
    let j = u32(floor((az + PI) / (2.0 * PI) * 16.0)) % 16u;
    return i * 16u + j;
}

fn probe_pos(p: u32) -> vec3f {
    let n = PP.n_axis;
    return PP.lo + PP.step * vec3f(f32(p / (n * n)), f32((p / n) % n), f32(p % n));
}

fn is_lit(i: u32) -> bool { return (node_state[i].z & LIT) != 0u; }

// world centre of node i (stored in its part's rest frame)
fn node_world(nd: Node) -> vec3f { return xf_point(parts[nd.part].xf, nd.mean); }

fn intensity(e: EmitNode, w: vec3f) -> vec3f {
    var s = 1.0;
    if (EP.one_sided != 0u) {
        let c = dot(w, EP.normal);
        s = select(select(0.0, 1.0, c > 0.0), smoothstep(-EP.edge_soft, EP.edge_soft, c), EP.edge_soft > 0.0);
    }
    return e.Le * (s * EP.scale * sqrt(max(dot(w, sym_mul(e.m_diag, e.m_off, w)), 0.0)));
}

fn use_emit(e: EmitNode, P: vec3f) -> bool {
    if (e.child_lo == NO_CHILD || EP.mode == 0u) { return e.child_lo == NO_CHILD; }
    let d = P - e.mean;
    return dot(d, d) > EP.theta * EP.theta * e.radius * e.radius;
}

// ---- schedule ------------------------------------------------------------------------------------------------

// the shadow T a lit node shows this frame with easing (n5 bit 2): linear from T_from to T over `refresh` frames
fn shown_T(i: u32) -> f32 {
    let a = node_light[i * SHN + 2u];
    let b = node_light[i * SHN + 3u];
    let age = f32(U.frame - bitcast<u32>(b.y)) + 1.0;
    return mix(a.w, b.x, min(age / f32(max(LP.refresh, 1u)), 1.0));
}

// Pick with probability p. Golden: split u's interval (so one u drives every choice and stays uniform); else hash.
fn choose(u: ptr<function, f32>, rng: ptr<function, u32>, golden: bool, p: f32) -> bool {
    if (golden) {
        if (*u < p) { *u = min(*u / max(p, 1e-30), 0.99999994); return true; }
        *u = min((*u - p) / max(1.0 - p, 1e-30), 0.99999994);
        return false;
    }
    *rng = pcg_hash(*rng);
    return to_unit(*rng) < p;
}

const EIN3: f32 = 1.68888;  // Ein(3) = int_0^3 (1 - e^-x) / x dx

fn sym_mat(dg: vec3f, off: vec3f) -> mat3x3f {
    return mat3x3f(vec3f(dg.x, off.x, off.y), vec3f(off.x, dg.y, off.z), vec3f(off.y, off.z, dg.z));
}

fn inverse3(m: mat3x3f) -> mat3x3f {
    let r0 = cross(m[1], m[2]); let r1 = cross(m[2], m[0]); let r2 = cross(m[0], m[1]);
    return transpose(mat3x3f(r0, r1, r2)) * (1.0 / dot(m[0], r0));
}

// lower Cholesky factor of a symmetric positive definite matrix
fn chol3(m: mat3x3f) -> mat3x3f {
    let l00 = sqrt(max(m[0][0], 1e-20));
    let l10 = m[0][1] / l00;
    let l20 = m[0][2] / l00;
    let l11 = sqrt(max(m[1][1] - l10 * l10, 1e-20));
    let l21 = (m[1][2] - l20 * l10) / l11;
    let l22 = sqrt(max(m[2][2] - l20 * l20 - l21 * l21, 1e-20));
    return mat3x3f(vec3f(l00, l10, l20), vec3f(0.0, l11, l21), vec3f(0.0, 0.0, l22));
}

// three standard normals (Box-Muller) from one hash
fn gauss3(h: u32) -> vec3f {
    let h1 = pcg_hash(h); let h2 = pcg_hash(h1); let h3 = pcg_hash(h2); let h4 = pcg_hash(h3);
    let r1 = sqrt(-2.0 * log(max(to_unit(h1), 1e-7)));
    let r2 = sqrt(-2.0 * log(max(to_unit(h3), 1e-7)));
    return vec3f(r1 * cos(2.0 * PI * to_unit(h2)), r1 * sin(2.0 * PI * to_unit(h2)), r2 * cos(2.0 * PI * to_unit(h4)));
}

@compute @workgroup_size(64)
fn lp_schedule(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= PP.node_count) { return; }
    let nd = nodes[i];
    if (nd.part == 0xFFFFFFFFu) { return; }
    var st = node_state[i];
    // a change coarsens: deep nodes lose their light, coarse ones stay and re-shade first
    if (LP.reset != 0u && nd.depth > LP.reset_depth) { st.z = 0u; }
    var lit = (st.z & LIT) != 0u;
    if (lit) { st.z = (min((st.z & ~LIT) + 1u, 0xFFFFu)) | LIT; }
    node_state[i] = st;

    let is_root = parts[nd.part].root == i;
    var wanted = false;
    if (is_root) {
        wanted = !lit;
    } else {
        let par = st.w;
        if ((node_state[par].z & LIT) != 0u) {
            // the parent is lit: refine into this child while the parent still looks too big from the eye
            let pn = nodes[par];
            let ang = pn.radius / max(length(U.eye - node_world(pn)), 1e-3);
            wanted = !lit && ang > LP.min_ang;
        }
    }
    let h = pcg_hash(i ^ pcg_hash(U.frame + 0x51ED27u));
    if (lit && (LP.reset != 0u || (h % max(LP.refresh, 1u)) == (U.frame % max(LP.refresh, 1u)))) { wanted = true; }
    if (!wanted) { return; }
    atomicAdd(&work[1], 1u);
    let p = bitcast<f32>(atomicLoad(&work[2]));
    if (to_unit(pcg_hash(h)) > p) { return; }
    let k = atomicAdd(&work[0], 1u);
    if (k < LP.budget) { atomicStore(&work[4u + k], i); }
}

@compute @workgroup_size(1)
fn lp_finish() {
    let cand = atomicLoad(&work[1]);
    let p = select(1.0, min(1.0, f32(LP.budget) / f32(cand)), cand > 0u);
    atomicStore(&work[2], bitcast<u32>(p));
    atomicStore(&work[0], 0u);
    atomicStore(&work[1], 0u);
}

// ---- shade nodes ---------------------------------------------------------------------------------------------

fn walk_tau(root: u32, P: vec3f, w: vec3f, dist: f32, mur: vec3f, theta: f32) -> f32 {
    var stack: array<u32, STACK>;
    stack[0] = root;
    var sp = 1u;
    var tau = 0.0;
    let th2 = theta * theta;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        let d = nd.mean - P;
        let tt = clamp(dot(d, w), 0.0, dist);
        let e = d - tt * w;
        if (dot(e, e) > nd.radius * nd.radius) { continue; }
        var use_it = ni < U.gaussian_count;
        if (!use_it && theta > 0.0) {
            let r = mur - nd.mean;
            use_it = dot(r, r) > th2 * nd.smax * nd.smax;
        }
        if (use_it) {
            tau += nd.kappa * dirm(nd, w) * seg_geom(nd, P, w, 0.0, dist);
        } else {
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                if (sp < STACK) { stack[sp] = kids[j]; sp += 1u; }
            }
        }
    }
    return tau;
}

@compute @workgroup_size(64)
fn lp_shade(@builtin(global_invocation_id) gid: vec3u) {
    let k = gid.x;
    if (k >= min(atomicLoad(&work[0]), LP.budget)) { return; }
    let i = atomicLoad(&work[4u + k]);
    let nd = nodes[i];
    let own = nd.part;
    let P = node_world(nd);
    var st = node_state[i];
    let fresh = (st.z & LIT) == 0u;

    // light cut: SH of the direct light, and a weighted reservoir pick for this shading's shadow sample
    var Ed = vec3f(0.0);
    var Dd = vec3f(0.0);
    // N5 opt-ins (EP.n5): bit 2 takes NS = 4 samples per re-shade; bit 0 drives each pick with a per-node
    // golden-ratio sequence instead of fresh hashes
    let smooth4 = (EP.n5 & 4u) != 0u;
    let golden = (EP.n5 & 1u) != 0u;
    let ns = select(1u, 4u, smooth4);
    var picks: array<u32, 4>;
    var us: array<f32, 4>;
    let o_g = to_unit(pcg_hash(i ^ 0x5bd1e995u));
    for (var k = 0u; k < 4u; k++) {
        picks[k] = EP.root;
        us[k] = fract(o_g + f32((U.frame * 4u + k) % 65536u) * 0.6180339887);
    }
    var wsum = 0.0;
    var rng = pcg_hash(i ^ pcg_hash(U.frame + SEED));
    var stack: array<u32, ESTACK>;
    stack[0] = EP.root;
    var sp = 1u;
    while (sp > 0u && EP.count > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let e = emit_nodes[ni];
        if (!use_emit(e, P)) {
            if (sp + 2u <= ESTACK) { stack[sp] = e.child_lo; stack[sp + 1u] = e.child_hi; sp += 2u; }
            continue;
        }
        let d = e.mean - P;
        let d2 = max(dot(d, d), 1e-6);
        let ww = d / sqrt(d2);
        let L = intensity(e, -ww) / d2;                     // radiance x solid angle arriving from ww
        let wgt = dot(L, vec3f(0.2126, 0.7152, 0.0722));
        Ed += L;
        Dd += wgt * ww;
        if (wgt > 0.0) {
            wsum += wgt;
            for (var k = 0u; k < ns; k++) {
                if (choose(&us[k], &rng, golden, wgt / wsum)) { picks[k] = ni; }
            }
        }
    }

    // shadow: one walked sample toward an emitter inside the picked node, into the node's history
    var T = bitcast<f32>(st.x);
    var cnt = st.y;
    if (fresh) {
        let ps = node_state[st.w];
        let parent_lit = (ps.z & LIT) != 0u && parts[own].root != i;
        T = select(1.0, bitcast<f32>(ps.x), parent_lit);
        cnt = select(0u, min(ps.y, 8u), parent_lit);
    }
    if (EP.count > 0u) {
        for (var k = 0u; k < ns; k++) {
            var e = emit_nodes[picks[k]];
            for (var depth = 0u; depth < 32u && e.child_lo != NO_CHILD; depth++) {
                let a = emit_nodes[e.child_lo];
                let b = emit_nodes[e.child_hi];
                let da = a.mean - P;
                let db = b.mean - P;
                let wa = dot(intensity(a, -normalize(da)), vec3f(0.2126, 0.7152, 0.0722)) / max(dot(da, da), 1e-6);
                let wb = dot(intensity(b, -normalize(db)), vec3f(0.2126, 0.7152, 0.0722)) / max(dot(db, db), 1e-6);
                if (choose(&us[k], &rng, golden, wa / max(wa + wb, 1e-30))) { e = a; } else { e = b; }
            }
            var yp: vec3f;
            if ((EP.n5 & 2u) != 0u) {
                // a point drawn from the emitter Gaussian itself (see emissive.wgsl emitter_sample)
                let M = sym_mat(e.m_diag, e.m_off);
                let C = inverse3(M) * (sqrt(max(determinant(M), 0.0)) / (2.0 * PI * EIN3));
                yp = e.mean + chol3(C) * clamp(gauss3(pcg_hash(rng ^ (0x68e31da4u + k))), vec3f(-3.0), vec3f(3.0));
            } else {
                let j3 = vec3f(to_unit(pcg_hash(rng + 1u)), to_unit(pcg_hash(rng + 2u)), to_unit(pcg_hash(rng + 3u))) * 2.0 - 1.0;
                yp = e.mean + j3 * (e.radius / 3.0);
            }
            rng = pcg_hash(rng + 7u);
            let dist = length(yp - P);
            let w = (yp - P) / dist;
            var tau = 0.0;
            for (var p = 0u; p < U.part_count; p++) {
                if (p == own || p == EP.skip_part) { continue; }
                let xf = parts[p].xf;
                let Pl = xf_point_inv(xf, P);
                tau += walk_tau(parts[p].root, Pl, xf_dir_inv(xf, w), dist, Pl, U.theta);
            }
            if (smooth4) {
                // count history (N5): a plain average of up to 256 samples, no floor
                let n = min(cnt + 1u, 256u);
                T += (exp(-tau) - T) / f32(n);
                cnt = n;
            } else {
                let a = max(1.0 / f32(cnt + 1u), 1.0 / 16.0);
                T = mix(T, exp(-tau), a);
                cnt = min(cnt + 1u, 0xFFFFu);
            }
        }
    }

    // bounce: the 8 surrounding probes (trilinear x valid x Chebyshev, as bounce.wgsl), projected to SH2
    var shb: array<vec3f, 9>;
    for (var m = 0u; m < 9u; m++) { shb[m] = vec3f(0.0); }
    let n = PP.n_axis;
    let fq = (clamp(P, PP.lo, PP.lo + PP.step * f32(n - 1u)) - PP.lo) / PP.step;
    let fi = min(vec3u(floor(fq)), vec3u(n - 2u));
    let fr = fq - vec3f(fi);
    var pid: array<u32, 8>;
    var wk: array<f32, 8>;
    var wsum2 = 0.0;
    for (var q = 0u; q < 8u; q++) {
        let o = vec3u(q >> 2u, (q >> 1u) & 1u, q & 1u);
        let ii = fi + o;
        let p = ii.x * n * n + ii.y * n + ii.z;
        pid[q] = p;
        let t = select(1.0 - fr, fr, o == vec3u(1u));
        let cw = t.x * t.y * t.z * probe_data[2u * (p * NCELL) + 1u].z;
        let v = P - probe_pos(p);
        let dist = length(v);
        let mm = probe_data[2u * (p * NCELL + cell_of(v / max(dist, 1e-9))) + 1u];
        let var_ = max(mm.y - mm.x * mm.x, 1e-4);
        let dd = max(dist - 0.03 - mm.x, 0.0);
        var vw = 1.0;
        if (dd > 0.0) { vw = pow(var_ / (var_ + dd * dd), 3.0); }
        wk[q] = cw * max(vw, 1e-3);
        wsum2 += wk[q];
    }
    // a movable part's nodes blend its own 8 part probes instead (bounce.wgsl probe_weights), trilinear in the
    // part's rest frame between probe 0 and probe 7, whose world positions probe_gather keeps in cells 1-3 .z
    let np = n * n * n;
    if (PP.part_probes > 0u && own < PP.part_probes / 8u && (parts[own].flags & 1u) == 0u) {
        let p0 = np + own * 8u;
        let a0 = xf_point_inv(parts[own].xf, vec3f(probe_data[2u * (p0 * NCELL + 1u) + 1u].z,
            probe_data[2u * (p0 * NCELL + 2u) + 1u].z, probe_data[2u * (p0 * NCELL + 3u) + 1u].z));
        let p7 = p0 + 7u;
        let a7 = xf_point_inv(parts[own].xf, vec3f(probe_data[2u * (p7 * NCELL + 1u) + 1u].z,
            probe_data[2u * (p7 * NCELL + 2u) + 1u].z, probe_data[2u * (p7 * NCELL + 3u) + 1u].z));
        let d7 = a7 - a0;
        let f = clamp((xf_point_inv(parts[own].xf, P) - a0) / select(d7, vec3f(1e-6), abs(d7) < vec3f(1e-6)),
                      vec3f(0.0), vec3f(1.0));
        wsum2 = 0.0;
        for (var q = 0u; q < 8u; q++) {
            let o = vec3u(q >> 2u, (q >> 1u) & 1u, q & 1u);
            let t = select(1.0 - f, f, o == vec3u(1u));
            pid[q] = p0 + q;
            wk[q] = t.x * t.y * t.z;
            wsum2 += wk[q];
        }
    }
    for (var c = 0u; c < NCELL; c++) {
        var L = vec3f(0.0);
        for (var q = 0u; q < 8u; q++) {
            if (wk[q] > 0.0) { L += (wk[q] / max(wsum2, 1e-12)) * probe_data[2u * (pid[q] * NCELL + c)].rgb; }
        }
        let y = sh2(cell_dir(c));
        let Lw = L * cell_omega(c);
        for (var m = 0u; m < 9u; m++) { shb[m] += Lw * y[m]; }
    }

    let lum = dot(Ed, vec3f(0.2126, 0.7152, 0.0722));
    // ease (bit 2): leaves show T walking from the value shown now to the new T over `refresh` frames
    // slot 2 = (unshadowed Ed, T shown at this re-shade), slot 3 = (new T, frame of this re-shade)
    let T_from = select(shown_T(i), T, fresh);
    node_light[i * SHN + 2u] = vec4f(Ed, T_from);
    node_light[i * SHN + 3u] = vec4f(T, bitcast<f32>(U.frame), 0.0, 0.0);
    node_light[i * SHN] = vec4f(Ed * T, 0.0);
    node_light[i * SHN + 1u] = vec4f(select(vec3f(0.0, 0.0, 1.0), normalize(Dd), dot(Dd, Dd) > 0.0), length(Dd) / max(lum, 1e-12));
    for (var m = 0u; m < 9u; m++) { node_light[i * SHN + 9u + m] = vec4f(shb[m], 0.0); }
    var age = st.z & ~LIT;
    if (fresh) { age = 0u; }
    node_state[i] = vec4u(bitcast<u32>(T), cnt, age | LIT, st.w);
}

// ---- leaf transfer -------------------------------------------------------------------------------------------

struct Leaf { P: vec3f, xf: array<vec4f, 3>, n: vec3f, Sd: vec3f, So: vec3f, detC: f32, mat: Material, nd: Node }

// The leaf as a Node record for footprint(): built from its Gaussian (isotropic extinction shape).
fn leaf_of(g: u32) -> Leaf {
    let G = gaussians[g];
    let xf = parts[G.part_surface & 0xFFFFu].xf;
    let mat = materials[min(G.flags & 0xFFFFu, max(U.material_count, 1u) - 1u)];
    let Rq = quat_matrix(G.quat);
    let Sloc = Rq * mat3x3f(vec3f(mat.flake.x, 0.0, 0.0), vec3f(0.0, mat.flake.y, 0.0), vec3f(0.0, 0.0, mat.flake.z)) * transpose(Rq);
    let inv = 1.0 / (G.scale * G.scale);
    let Ci = Rq * mat3x3f(vec3f(inv.x, 0.0, 0.0), vec3f(0.0, inv.y, 0.0), vec3f(0.0, 0.0, inv.z)) * transpose(Rq);
    var nd: Node;
    nd.cinv_diag = vec3f(Ci[0].x, Ci[1].y, Ci[2].z);
    nd.cinv_off = vec3f(Ci[1].x, Ci[2].x, Ci[2].y);
    nd.kappa = G.kappa;
    nd.sn_diag = vec3f(1.0);
    nd.sn_off = vec3f(0.0);
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    return Leaf(xf_point(xf, G.mu), xf, Rq[0], vec3f(Sloc[0].x, Sloc[1].y, Sloc[2].z), vec3f(Sloc[1].x, Sloc[2].x, Sloc[2].y),
                s3 * s3, mat, nd);
}

// the leaf's incoming light: its deepest lit ancestor's SH, faded in from that node's parent over LP.blend frames
fn incoming(g: u32) -> array<vec3f, 18> {
    var out: array<vec3f, 18>;
    for (var m = 0u; m < SHN; m++) { out[m] = vec3f(0.0); }
    var a = g;
    for (var s = 0u; s < 16u && !is_lit(a); s++) {
        let par = node_state[a].w;
        if (par == a) { return out; }
        a = par;
    }
    if (!is_lit(a)) { return out; }
    let st = node_state[a];
    let par = st.w;
    let w = clamp(f32(st.z & ~LIT) / max(LP.blend, 1.0), 0.0, 1.0);
    let use_par = par != a && is_lit(par) && w < 1.0;
    for (var m = 0u; m < SHN; m++) {
        let c = node_light[a * SHN + m].rgb;
        out[m] = select(c, mix(node_light[par * SHN + m].rgb, c, w), use_par);
    }
    if ((EP.n5 & 4u) != 0u) {
        let c = node_light[a * SHN + 2u].rgb * shown_T(a);
        out[0] = select(c, mix(node_light[par * SHN + 2u].rgb * shown_T(par), c, w), use_par);
    }
    out[1] = select(vec3f(0.0, 0.0, 1.0), normalize(out[1]), dot(out[1], out[1]) > 0.0);
    return out;
}

// radiance toward world directions wo_w[0..nout): direct part and bounce part
struct TwoParts { d: array<vec3f, 15>, b: array<vec3f, 15> }

fn transfer(g: u32, lf: Leaf, sh: array<vec3f, 18>, wo_w: array<vec3f, 15>, nout: u32) -> TwoParts {
    if ((U.iso & 1u) != 0u) { return transfer_iso(g, lf, sh, wo_w, nout); }
    var wo: array<vec3f, 15>;
    var fsame: array<f32, 15>;
    var res: TwoParts;
    for (var k = 0u; k < nout; k++) {
        wo[k] = xf_dir_inv(lf.xf, wo_w[k]);
        fsame[k] = 1.0 / max(vgrid_lookup(vgrid[g], wo[k]), lf.mat.vfloor);
        res.d[k] = vec3f(0.0);
        res.b[k] = vec3f(0.0);
    }
    // direct: one light from the node's direction, evaluated exactly (as the per-leaf shading does per sample)
    let Ed = sh[0];
    if (Ed.r + Ed.g + Ed.b > 0.0) {
        let wi = xf_dir_inv(lf.xf, sh[1]);
        let vi = vgrid_lookup(vgrid[g], wi);
        if (vi > 1e-7) {
            let catchw = vi * footprint(lf.nd, lf.detC, wi);
            let wn = dot(wi, lf.n);
            for (var k = 0u; k < nout; k++) {
                let f = select(1.0, fsame[k], wn * dot(wo[k], lf.n) > 0.0);
                res.d[k] += Ed * (phase_ex(lf.mat, lf.n, lf.Sd, lf.So, wi, wo[k]) * f * catchw);
            }
        }
    }
    // bounce: SH2 by quadrature
    let dw = 4.0 * PI / f32(NQ);
    for (var q = 0u; q < NQ; q++) {
        let dwld = fib_dir(q);
        let y = sh2(dwld);
        var Lb = vec3f(0.0);
        for (var m = 0u; m < 9u; m++) { Lb += sh[9u + m] * y[m]; }
        Lb = max(Lb, vec3f(0.0));
        if (Lb.r + Lb.g + Lb.b <= 0.0) { continue; }
        let wi = xf_dir_inv(lf.xf, dwld);
        let vi = vgrid_lookup(vgrid[g], wi);
        if (vi <= 1e-7) { continue; }
        let catchw = vi * footprint(lf.nd, lf.detC, wi) * dw;
        let wn = dot(wi, lf.n);
        for (var k = 0u; k < nout; k++) {
            let f = select(1.0, fsame[k], wn * dot(wo[k], lf.n) > 0.0);
            res.b[k] += Lb * (phase_ex(lf.mat, lf.n, lf.Sd, lf.So, wi, wo[k]) * f * catchw);
        }
    }
    let alb = gaussians[g].albedo;
    for (var k = 0u; k < nout; k++) {
        let s = 1.0 / footprint(lf.nd, lf.detC, wo[k]);
        res.d[k] = alb * res.d[k] * s;
        res.b[k] = alb * res.b[k] * s;
    }
    return res;
}

// M39 direction-independent transfer (common.wgsl iso_norm, material_ext.wgsl iso_lobe): alb iso_norm sum L V A x
// iso_lobe, 1 for matte. The catch uses the leaf's own Node (leaf_of, isotropic extinction) as the M36 transfer here
// does, so iso_norm is computed with the same footprint by lp_iso_init and kept in the leaf's unused node_light slot 4
// (lp_transfer has no room for the nodes binding under the 8-storage-buffer limit).
fn transfer_iso(g: u32, lf: Leaf, sh: array<vec3f, 18>, wo_w: array<vec3f, 15>, nout: u32) -> TwoParts {
    let nd = lf.nd;
    let inorm = node_light[g * SHN + 4u].x;
    var wo: array<vec3f, 15>;
    var res: TwoParts;
    for (var k = 0u; k < nout; k++) {
        wo[k] = xf_dir_inv(lf.xf, wo_w[k]);
        res.d[k] = vec3f(0.0);
        res.b[k] = vec3f(0.0);
    }
    let Ed = sh[0];
    if (Ed.r + Ed.g + Ed.b > 0.0) {
        let wi = xf_dir_inv(lf.xf, sh[1]);
        let vi = vgrid_lookup(vgrid[g], wi);
        if (vi > 1e-7) {
            let catchw = vi * footprint(nd, lf.detC, wi);
            let wr = mirror(wi, lf.n);
            let damp = iso_damp(inorm, footprint(nd, lf.detC, wr) * vgrid_lookup(vgrid[g], wr));
            for (var k = 0u; k < nout; k++) {
                res.d[k] += Ed * (catchw * iso_lobe(lf.mat, lf.Sd, lf.So, wi, wo[k], damp));
            }
        }
    }
    let dw = 4.0 * PI / f32(NQ);
    for (var q = 0u; q < NQ; q++) {
        let dwld = fib_dir(q);
        let y = sh2(dwld);
        var Lb = vec3f(0.0);
        for (var m = 0u; m < 9u; m++) { Lb += sh[9u + m] * y[m]; }
        Lb = max(Lb, vec3f(0.0));
        if (Lb.r + Lb.g + Lb.b <= 0.0) { continue; }
        let wi = xf_dir_inv(lf.xf, dwld);
        let vi = vgrid_lookup(vgrid[g], wi);
        if (vi <= 1e-7) { continue; }
        let catchw = vi * footprint(nd, lf.detC, wi) * dw;
        let wr = mirror(wi, lf.n);
        let damp = iso_damp(inorm, footprint(nd, lf.detC, wr) * vgrid_lookup(vgrid[g], wr));
        for (var k = 0u; k < nout; k++) {
            res.b[k] += Lb * (catchw * iso_lobe(lf.mat, lf.Sd, lf.So, wi, wo[k], damp));
        }
    }
    let s = gaussians[g].albedo * inorm;
    for (var k = 0u; k < nout; k++) {
        res.d[k] = s * res.d[k];
        res.b[k] = s * res.b[k];
    }
    return res;
}

// Once per asset: each leaf's iso_norm for transfer_iso, as iso_init.wgsl but with leaf_of's footprint.
@compute @workgroup_size(64)
fn lp_iso_init(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let lf = leaf_of(g);
    var S = 0.0;
    var S0 = 0.0;
    for (var c = 0u; c < NCELL; c++) {
        let a = footprint(lf.nd, lf.detC, cell_dir(c)) * cell_omega(c);
        S += a * f32((vgrid[g].w[c >> 2u] >> ((c & 3u) * 8u)) & 0xFFu) * (1.0 / 255.0);
        S0 += a;
    }
    node_light[g * SHN + 4u] = vec4f(1.0 / max(max(S, 0.02 * S0), 1e-30), 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn lp_transfer(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let lf = leaf_of(g);
    var wo: array<vec3f, 15>;
    wo[0] = normalize(U.eye - lf.P);
    let r = transfer(g, lf, incoming(g), wo, 1u);
    var c = r.d[0];
    // an emitter's own radiance (SPEC 2.6), one-sided ones only toward +short axis
    if ((lf.mat.flags & 1u) != 0u && ((lf.mat.flags & 4u) == 0u || dot(xf_dir_inv(lf.xf, wo[0]), lf.n) > 0.0)) {
        c += vec3f(unpack2x16float(lf.mat.pad0), unpack2x16float(lf.mat.pad1).x);
    }
    colour[g] = vec4f(c, 1.0);
    lo_b[g * 15u + 14u] = vec4f(r.b[0], 0.0);
}

@compute @workgroup_size(64)
fn lp_transfer_dirs(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x * 8u + LP.dir_phase;
    if (g >= U.gaussian_count) { return; }
    let lf = leaf_of(g);
    var wo: array<vec3f, 15>;
    for (var k = 0u; k < 14u; k++) { wo[k] = DIRS14[k]; }
    let r = transfer(g, lf, incoming(g), wo, 14u);
    for (var k = 0u; k < 14u; k++) {
        lo_dir[g * 14u + k] = vec4f(r.d[k], 0.0);
        lo_b[g * 15u + k] = vec4f(r.b[k], 0.0);
    }
}
