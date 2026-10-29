// E9 milestone 1: the one-path gather on the GPU (candidate A), static scene, parts at rest (xf = identity).
// Mirrors experiments/M45_unified/E8_motion/e8.py sampled_gather_walk without the motion terms:
//   per receiver: E2's light cut (budget K, bound e = min(Ib / dmin^2, 2 pi Bmax)), each cut node replaced by one
//   sender leaf picked by a radiance-weighted descent (child probability (1 - mix) lum(N_c(u)) / r^2 + mix cnt share),
//   E1's exact pair geometry, T from ONE bundle ray (endpoints drawn inside both Gaussians, leaf-exact walk over the
//   other parts), weighted by 1 / probability. B_frame = Le + alb iso Phi.
// Prepend runtime/wgsl/common.wgsl.

struct Params {
    G: u32, N: u32, R: u32, K: u32,           // Gaussians, nodes, receivers in this dispatch, cut budget
    frame: u32, part_count: u32, mix: f32, alpha: f32,
    lvl_off: u32, lvl_cnt: u32, nsend: u32, lsh_off: u32,   // lists offsets; leafsh lives in lists as f32 bits
    theta: f32, cl_off: u32, ncl: u32, pad1: u32,            // walk rule (0 = leaves); cluster table (start, count)
    roots: array<vec4u, 4>,                    // up to 16 part roots
    pbound: array<vec4f, 16>,                  // part bounds (world = rest), xyz centre, w radius
    dtot: vec4f,                               // moving part: displacement since the cache (xyz)
    mdelta: vec4f,                             // its displacement this frame (xyz), w: 1 while moving
    // renderer motion (gather_m): part transforms world_from_rest as rows [R | t], now / last frame / at the cache
    xf: array<vec4f, 48>, xfp: array<vec4f, 48>, xfc: array<vec4f, 48>,
    masks: vec4u,                              // x: parts moved since the cache, y: moved this frame, z: cache valid, w: K idle
    knobs: vec4f,                              // x: n_max, y: idle trickle period, z: dirty share threshold, w: 1 = no control variate (debug)
    // environment (M45 E5, SPEC 2.9) as distant senders in gather_m: the sun (direction, ASG std; irradiance) and the
    // SH sky as NB Fibonacci bins (direction, solid angle; L Omega of the bin, importance CDF in w)
    sun: vec4f, sunE: vec4f,
    envu: vec4u,                               // x: bins, y: sky rays per cluster (shared), z: sun rays per receiver (own walks),
                                               // w: bit 0 = emitters send as Lambertian panels (lamp_area; M48, off by default)
    bins: array<vec4f, 64>, brad: array<vec4f, 64>,
    // progressive scheduling (renderer ?budget=N, 2026-10-28): x = clusters per frame that converge at K = y, chosen by
    // sched_cut's priority (on-screen size x how far from converged); 0 = off (every converging cluster at K idle)
    sched: vec4u,
    vp: mat4x4f,                               // the camera's view-projection (world -> clip), for on-screen size
    // soft cluster membership (renderer ?soft=N, 2026-10-28): x = neighbour clusters per receiver (0 = off), y = sb
    // offset of the cluster headers, z = of the per-receiver own terms, w = pick divisor (K per cluster / w)
    soft: vec4u,
}

struct TreeNode { cen: vec3f, rad: f32, Ib: f32, Bmax: f32, sidx: i32, cnt: u32 }   // 32 B
struct GExt { Le_w: vec4f, B: vec4f, Bf: vec4f, mu0: vec4f, Bc: vec4f }
// Le + wsend (overlap / power factor); history B (w: effective count); this frame's B (w: share of light the moving
// part touches); the cache's centre (the placement the cache was solved at); the cache B (control variate reference)

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> kids: array<u32>;
@group(0) @binding(4) var<storage, read> vgrid: array<VGrid>;
@group(0) @binding(5) var<storage, read_write> tree: array<TreeNode>;
@group(0) @binding(6) var<storage, read_write> Nsh: array<f32>;          // node x 3 channels x 9 SH
@group(0) @binding(7) var<storage, read_write> gext: array<GExt>;
@group(0) @binding(8) var<storage, read> lists: array<u32>;   // receivers | senders | node levels | leafsh (f32 bits,
                                                               // sender index x 10: 9 SH of a_j(u), amax)
// soft membership: gather_m's picks per cluster (2 x vec4f each: end point + probability, sender + shared depths + dirty),
// cluster headers (n, mode, near mask) and each receiver's own terms (sky + sun); gather_soft blends the clusters.
// Group 2 (gather_m / gather_soft only): the other pipelines stay within 8 storage buffers + onepath_spec's two, and
// onepath_spec.wgsl (which prepends this file) keeps group 1 for its own
@group(2) @binding(0) var<storage, read_write> sb: array<vec4u>;   // u32 words (f32 as bits: no denormal flush of the ints)
@group(2) @binding(1) var<storage, read> snb: array<vec4u>;   // per Gaussian: neighbour clusters, their weights (f32 bits)

const KMAX: u32 = 64u;
const SHARE_FULL: f32 = 0.25;                 // gather_m: dirty share at which a cluster gets the full K active
const KFRESH: f32 = 8.0;
const SCHED_MIN: f32 = 0.0;                   // progressive scheduling: below this history count a cluster refines every
                                              // frame at K idle as without a budget. 32 was tried (2026-10-28): no better
                                              // in the first frames (run-to-run noise dominates there), worse at 64-128                      // gather_m: K active until a receiver's history count reaches this
const EDGE_SOFT: f32 = 0.2;
// a one-sided emitter's sending area toward its outward cosine c = -n.w (aj = A(w) V(-w) wsend): the Gaussian footprint
// x the EDGE_SOFT smoothstep (default), or with P.envu.w bit 0 a Lambertian panel A(n) V(-w) wsend max(c, 0), as the
// mesh's area light (M48: FLIP vs the mesh path trace 0.158 -> 0.101 with exact lamp leaves; the Gaussian lamp sent
// too much light sideways)
fn lamp_area(j: u32, aj: f32, nrm: vec3f, c: f32, vw: f32) -> f32 {
    if ((P.envu.w & 1u) != 0u) { return footprint(nodes[j], det_c(j), nrm) * vw * max(c, 0.0); }
    let s = clamp((c + EDGE_SOFT) / (2.0 * EDGE_SOFT), 0.0, 1.0);
    return aj * s * s * (3.0 - 2.0 * s);
}
const LUMV: vec3f = vec3f(0.2126, 0.7152, 0.0722);

fn root_of(p: u32) -> u32 { return P.roots[p / 4u][p % 4u]; }

fn quat_mat(q: vec4f) -> mat3x3f {            // columns = the Gaussian's axes (x y z w quaternion)
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3f(vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
                   vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
                   vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}

fn det_c(g: u32) -> f32 {
    let s = gaussians[g].scale;
    let d = s.x * s.y * s.z;
    return d * d;
}

// ---- random ----------------------------------------------------------------------------------------------------

var<private> rng: u32;
fn rnd() -> f32 { rng = pcg_hash(rng); return to_unit(rng); }
fn gauss2() -> vec2f {
    let u1 = max(rnd(), 1e-7); let u2 = rnd();
    let r = sqrt(-2.0 * log(u1));
    return vec2f(r * cos(2.0 * PI * u2), r * sin(2.0 * PI * u2));
}
fn point_in(g: u32) -> vec3f {               // a point drawn from Gaussian g
    let G = gaussians[g];
    let a = gauss2(); let b = gauss2();
    let sc = vec3f(G.scale.x, G.scale.y, G.scale.z);       // explicit vec3f (Metal: no packed_float3 x matrix)
    let mu = vec3f(G.mu.x, G.mu.y, G.mu.z);
    let off = sc * vec3f(a.x, a.y, b.x);
    let R = quat_mat(G.quat);
    return mu + R[0] * off.x + R[1] * off.y + R[2] * off.z;
}

// ---- SH ------------------------------------------------------------------------------------------------------------

fn sh_lum(n: u32, u: vec3f) -> f32 {
    var acc = 0.0;
    let x = u.x; let y = u.y; let z = u.z;
    let b = array<f32, 9>(0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z,
                          0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y));
    for (var c = 0u; c < 3u; c++) {
        var s = 0.0;
        let o = (n * 3u + c) * 9u;
        for (var k = 0u; k < 9u; k++) { s += Nsh[o + k] * b[k]; }
        acc += LUMV[c] * s;
    }
    return acc;
}

// ---- pair geometry (lib.pair_geom) ---------------------------------------------------------------------------------

struct Geom { ai: f32, om: f32 }

fn pair_geom(i: u32, j: u32) -> Geom {
    let a = gaussians[i].mu; let b = gaussians[j].mu;
    return pair_geom_at(i, j, vec3f(a.x, a.y, a.z), vec3f(b.x, b.y, b.z));
}

fn pair_geom_at(i: u32, j: u32, mi: vec3f, mj: vec3f) -> Geom {
    let Gi = gaussians[i]; let Gj = gaussians[j];
    if ((Gi.part_surface & 0xFFFFu) == (Gj.part_surface & 0xFFFFu)) { return Geom(0.0, 0.0); }
    let dv = mj - mi;
    let r2 = dot(dv, dv); let r = sqrt(r2);
    let w = dv / r;
    let vi = vgrid_lookup(vgrid[i], w);
    if (vi <= 1e-4) { return Geom(0.0, 0.0); }
    let vj = vgrid_lookup(vgrid[j], -w);
    if (vj <= 1e-4) { return Geom(0.0, 0.0); }
    var aj = footprint(nodes[j], det_c(j), w) * vj * gext[j].Le_w.w;
    let mat_one_sided = (gext[j].Le_w.x + gext[j].Le_w.y + gext[j].Le_w.z) > 0.0;   // emitters here are one-sided
    if (mat_one_sided) {
        let nrm = quat_mat(Gj.quat)[0];
        aj = lamp_area(j, aj, nrm, -dot(nrm, w), vj * gext[j].Le_w.w);
        if (aj <= 0.0) { return Geom(0.0, 0.0); }
    }
    let ai = footprint(nodes[i], det_c(i), w) * vi;
    let om = 2.0 * PI * (1.0 - r / sqrt(r2 + aj / PI));
    return Geom(ai, om);
}

// ---- walk: leaf-exact optical depth over the other parts (walk_lamp.wgsl walk_part, theta 0) ----------------------

fn walk_tau(A: vec3f, Bp: vec3f, skip_a: u32, skip_b: u32) -> f32 { return walk_tau3(A, Bp, skip_a, skip_b, 0xFFFFFFFFu, 0xFFFFFFFFu); }

// only = a part index: walk that part alone; skip_c: also skip it
fn walk_tau3(A: vec3f, Bp: vec3f, skip_a: u32, skip_b: u32, skip_c: u32, only: u32) -> f32 {
    let dv = Bp - A;
    let L = length(dv);
    if (L < 1e-9) { return 0.0; }
    let w = dv / L;
    var tau = 0.0;
    var stack: array<u32, STACK>;
    for (var p = 0u; p < P.part_count; p++) {
        if (p == skip_a || p == skip_b || p == skip_c) { continue; }
        if (only != 0xFFFFFFFFu && p != only) { continue; }
        let pb = P.pbound[p];
        let c = pb.xyz - A;
        let t = clamp(dot(c, w), 0.0, L);
        let e = c - t * w;
        if (dot(e, e) > pb.w * pb.w) { continue; }
        stack[0] = root_of(p);
        var sp = 1u;
        while (sp > 0u) {
            sp -= 1u;
            let ni = stack[sp];
            let nd = nodes[ni];
            let d = nd.mean - A;
            let tt = clamp(dot(d, w), 0.0, L);
            let ee = d - tt * w;
            if (dot(ee, ee) > nd.radius * nd.radius) { continue; }
            var use_it = ni < P.G;
            if (!use_it && P.theta > 0.0) {               // walk_lamp's distance rule: a parent far enough away
                use_it = dot(d, d) > P.theta * P.theta * nd.smax * nd.smax;
            }
            if (use_it) {
                tau += nd.kappa * dirm(nd, w) * seg_geom(nd, A, w, 0.0, L);
                continue;
            }
            for (var k = nd.child_lo; k < nd.child_hi; k++) {
                if (sp < STACK) { stack[sp] = kids[k]; sp += 1u; }
            }
        }
    }
    return tau;
}

// ---- sender tree aggregates ----------------------------------------------------------------------------------------

// leaves: N_j = a_j(u) SH x B_j, Ib = lum(B) amax, Bmax = lum(B). lists[lvl_off ..] = sender leaf ids, leafsh by index
@compute @workgroup_size(64)
fn nb_leaf(@builtin(global_invocation_id) gid: vec3u) {
    let k = gid.x;
    if (k >= P.nsend) { return; }
    let j = lists[P.lvl_off + k];
    let B = gext[j].B.xyz;
    for (var c = 0u; c < 3u; c++) {
        for (var q = 0u; q < 9u; q++) { Nsh[(j * 3u + c) * 9u + q] = bitcast<f32>(lists[P.lsh_off + k * 10u + q]) * B[c]; }
    }
    let l = dot(B, LUMV);
    tree[j].Ib = l * bitcast<f32>(lists[P.lsh_off + k * 10u + 9u]);
    tree[j].Bmax = l;
}

// internal nodes of one depth (deepest first): sums over children that carry senders
@compute @workgroup_size(64)
fn nb_up(@builtin(global_invocation_id) gid: vec3u) {
    let k = gid.x;
    if (k >= P.lvl_cnt) { return; }
    let n = lists[P.lvl_off + k];
    var acc: array<f32, 27>;
    for (var q = 0u; q < 27u; q++) { acc[q] = 0.0; }
    var Ib = 0.0; var Bm = 0.0;
    let nd = nodes[n];
    for (var kk = nd.child_lo; kk < nd.child_hi; kk++) {
        let c = kids[kk];
        if (tree[c].cnt == 0u) { continue; }
        for (var q = 0u; q < 27u; q++) { acc[q] += Nsh[c * 27u + q]; }
        Ib += tree[c].Ib; Bm = max(Bm, tree[c].Bmax);
    }
    for (var q = 0u; q < 27u; q++) { Nsh[n * 27u + q] = acc[q]; }
    tree[n].Ib = Ib; tree[n].Bmax = Bm;
}

// ---- gather --------------------------------------------------------------------------------------------------------

fn bound(n: u32, x: vec3f) -> f32 {
    if (n < P.G) { return 0.0; }
    let tn = tree[n];
    let d = length(tn.cen - x) - tn.rad;
    let lim = 2.0 * PI * tn.Bmax;
    if (d <= 1e-6) { return lim; }
    return min(tn.Ib / (d * d), lim);
}

// ---- cluster-shared gather: one workgroup per receiver cluster (one part, a few cm), one cut at the members'
// centroid, one bundle ray per pick (from a member's Gaussian), its T shared; each member keeps its own geometry ----

var<workgroup> wcut: array<u32, KMAX>;
var<workgroup> wpick: array<u32, KMAX>;
var<workgroup> wprob: array<f32, KMAX>;
var<workgroup> wT: array<f32, KMAX>;
var<workgroup> wn: u32;
var<workgroup> wx: vec3f;

@compute @workgroup_size(64)
fn gather_cl(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let c = wid.x;
    if (c >= P.ncl) { return; }
    let start = lists[P.cl_off + 2u * c];
    let count = lists[P.cl_off + 2u * c + 1u];
    let own = gaussians[lists[start]].part_surface & 0xFFFFu;
    if (li == 0u) {
        var xs = vec3f(0.0);
        for (var m = 0u; m < count; m++) { let g = gaussians[lists[start + m]]; xs += vec3f(g.mu.x, g.mu.y, g.mu.z); }
        let x = xs / f32(count);
        wx = x;
        var n = 0u;
        let K = min(P.K, KMAX);
        var ce: array<f32, KMAX>;
        for (var p = 0u; p < P.part_count; p++) {
            let r = root_of(p);
            if (p == own || tree[r].cnt == 0u || n >= K) { continue; }
            wcut[n] = r; ce[n] = bound(r, x); n += 1u;
        }
        loop {
            var best = -1.0; var bi = 0u;
            for (var k = 0u; k < n; k++) { if (wcut[k] >= P.G && ce[k] > best) { best = ce[k]; bi = k; } }
            if (best <= 0.0) { break; }
            let nd = nodes[wcut[bi]];
            var nk = 0u;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) { if (tree[kids[kk]].cnt > 0u) { nk += 1u; } }
            if (n - 1u + nk > K) { break; }
            var first = true;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) {
                let cc = kids[kk];
                if (tree[cc].cnt == 0u) { continue; }
                if (first) { wcut[bi] = cc; ce[bi] = bound(cc, x); first = false; }
                else { wcut[n] = cc; ce[n] = bound(cc, x); n += 1u; }
            }
        }
        wn = n;
    }
    workgroupBarrier();
    let n = wn;
    let x = wx;
    if (li < n) {
        rng = pcg_hash(c * 7919u + P.frame * 104729u + li * 31u + 7u);
        var nd = wcut[li];
        var prob = 1.0;
        while (nd >= P.G) {
            let node = nodes[nd];
            var lt = 0.0; var ct = 0.0;
            var pc: array<f32, 8>; var cc: array<u32, 8>; var m = 0u;
            for (var kk = node.child_lo; kk < node.child_hi; kk++) {
                let ch = kids[kk];
                if (tree[ch].cnt == 0u || m >= 8u) { continue; }
                let dv = x - tree[ch].cen;
                let ul = length(dv) + 1e-12;
                let l = max(sh_lum(ch, dv / ul), 0.0) / (ul * ul);
                pc[m] = l; cc[m] = ch; lt += l; ct += f32(tree[ch].cnt); m += 1u;
            }
            let u = rnd();
            var a = 0.0; var pick = m - 1u;
            for (var q = 0u; q < m; q++) {
                var pq = P.mix * f32(tree[cc[q]].cnt) / ct;
                if (lt > 0.0) { pq += (1.0 - P.mix) * pc[q] / lt; } else { pq += (1.0 - P.mix) * f32(tree[cc[q]].cnt) / ct; }
                pc[q] = pq;
            }
            for (var q = 0u; q < m; q++) { a += pc[q]; if (u < a) { pick = q; break; } }
            prob *= pc[pick]; nd = cc[pick];
        }
        wpick[li] = nd; wprob[li] = prob;
        // one shared bundle ray, from a member's Gaussian (round robin) to the picked sender
        let mem = lists[start + li % count];
        let A = point_in(mem); let Bq = point_in(nd);
        wT[li] = exp(-walk_tau(A, Bq, own, gaussians[nd].part_surface & 0xFFFFu));
    }
    workgroupBarrier();
    if (li >= count) { return; }
    let i = lists[start + li];
    var acc = vec3f(0.0);
    for (var k = 0u; k < n; k++) {
        let j = wpick[k];
        let Bj = gext[j].B.xyz;
        if (Bj.x + Bj.y + Bj.z <= 0.0) { continue; }
        let gm = pair_geom(i, j);
        if (gm.ai * gm.om <= 0.0) { continue; }
        acc += (gm.ai * gm.om * wT[k] / wprob[k]) * Bj;
    }
    let iso = bitcast<f32>(nodes[i].pad0);
    gext[i].Bf = vec4f(gext[i].Le_w.xyz + gaussians[i].albedo * iso * acc, 1.0);
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3u) {
    let ri = gid.x;
    if (ri >= P.R) { return; }
    let i = lists[ri];
    let Gi = gaussians[i];
    let x = Gi.mu;
    let own = Gi.part_surface & 0xFFFFu;
    rng = pcg_hash(i * 9781u + P.frame * 6271u + 1u);
    // light cut: leaves and internal nodes in one list; split the largest bound until K entries
    var cn: array<u32, KMAX>;
    var ce: array<f32, KMAX>;
    var n = 0u;
    let K = min(P.K, KMAX);
    for (var p = 0u; p < P.part_count; p++) {
        let r = root_of(p);
        if (p == own || tree[r].cnt == 0u || n >= K) { continue; }
        cn[n] = r; ce[n] = bound(r, x); n += 1u;
    }
    loop {
        var best = -1.0; var bi = 0u;
        for (var k = 0u; k < n; k++) { if (cn[k] >= P.G && ce[k] > best) { best = ce[k]; bi = k; } }
        if (best <= 0.0) { break; }
        let nd = nodes[cn[bi]];
        var nk = 0u;
        for (var kk = nd.child_lo; kk < nd.child_hi; kk++) { if (tree[kids[kk]].cnt > 0u) { nk += 1u; } }
        if (n - 1u + nk > K) { break; }
        // replace bi by its first child, append the rest
        var first = true;
        for (var kk = nd.child_lo; kk < nd.child_hi; kk++) {
            let c = kids[kk];
            if (tree[c].cnt == 0u) { continue; }
            if (first) { cn[bi] = c; ce[bi] = bound(c, x); first = false; }
            else { cn[n] = c; ce[n] = bound(c, x); n += 1u; }
        }
    }
    // one sender leaf per cut entry
    var acc = vec3f(0.0);
    for (var k = 0u; k < n; k++) {
        var nd = cn[k];
        var prob = 1.0;
        while (nd >= P.G) {
            let node = nodes[nd];
            var lt = 0.0; var ct = 0.0;
            var pc: array<f32, 8>; var cc: array<u32, 8>; var m = 0u;
            for (var kk = node.child_lo; kk < node.child_hi; kk++) {
                let c = kids[kk];
                if (tree[c].cnt == 0u || m >= 8u) { continue; }
                let dv = x - tree[c].cen;
                let ul = length(dv) + 1e-12;
                let l = max(sh_lum(c, dv / ul), 0.0) / (ul * ul);
                pc[m] = l; cc[m] = c; lt += l; ct += f32(tree[c].cnt); m += 1u;
            }
            let u = rnd();
            var a = 0.0; var pick = m - 1u;
            for (var q = 0u; q < m; q++) {
                var pq = P.mix * f32(tree[cc[q]].cnt) / ct;
                if (lt > 0.0) { pq += (1.0 - P.mix) * pc[q] / lt; } else { pq += (1.0 - P.mix) * f32(tree[cc[q]].cnt) / ct; }
                pc[q] = pq;
            }
            for (var q = 0u; q < m; q++) { a += pc[q]; if (u < a) { pick = q; break; } }
            prob *= pc[pick]; nd = cc[pick];
        }
        let Bj = gext[nd].B.xyz;
        if (Bj.x + Bj.y + Bj.z <= 0.0) { continue; }
        let gm = pair_geom(i, nd);
        if (gm.ai * gm.om <= 0.0) { continue; }
        let A = point_in(i); let Bq = point_in(nd);
        var tau = 0.0;
        if (P.alpha >= 0.0) { tau = walk_tau(A, Bq, own, gaussians[nd].part_surface & 0xFFFFu); }   // alpha < 0: profiling, no walk
        acc += (gm.ai * gm.om * exp(-tau) / prob) * Bj;
    }
    let iso = bitcast<f32>(nodes[i].pad0);
    gext[i].Bf = vec4f(gext[i].Le_w.xyz + Gi.albedo * iso * acc, 1.0);
}

// history: B <- (1 - a) B + a B_frame for the receivers (emitters keep Le)
@compute @workgroup_size(64)
fn history(@builtin(global_invocation_id) gid: vec3u) {
    let ri = gid.x;
    if (ri >= P.R) { return; }
    let i = lists[ri];
    gext[i].B = vec4f(mix(gext[i].B.xyz, gext[i].Bf.xyz, P.alpha), 1.0);
}


// ---- motion: control variate against the cache, cluster-shared rays (milestone 2) ----------------------------------
// Each pick's shared ray is walked over the static parts once, and over the moving part at its current placement,
// last frame's (dirty test) and the cache's (control variate: the part shifted back by dtot = the ray shifted by +dtot).
// A pick or receiver on the moving part takes its cache-time centre (mu0) for the old geometry and ray.
// B_frame = Bc + alb iso sum_k (c_new - c_old) / p_k. Bf.w = the share of the receiver's light the moving part touches.

var<workgroup> wTn: array<f32, KMAX>;
var<workgroup> wTo: array<f32, KMAX>;
var<workgroup> winv: array<u32, KMAX>;

@compute @workgroup_size(64)
fn gather_cv(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let c = wid.x;
    if (c >= P.ncl) { return; }
    let start = lists[P.cl_off + 2u * c];
    let count = lists[P.cl_off + 2u * c + 1u];
    let own = gaussians[lists[start]].part_surface & 0xFFFFu;
    let mp = P.pad1;
    if (li == 0u) {
        var xs = vec3f(0.0);
        for (var m = 0u; m < count; m++) { let g = gaussians[lists[start + m]]; xs += vec3f(g.mu.x, g.mu.y, g.mu.z); }
        let x = xs / f32(count);
        wx = x;
        var n = 0u;
        let K = min(P.K, KMAX);
        var ce: array<f32, KMAX>;
        for (var p = 0u; p < P.part_count; p++) {
            let r = root_of(p);
            if (p == own || tree[r].cnt == 0u || n >= K) { continue; }
            wcut[n] = r; ce[n] = bound(r, x); n += 1u;
        }
        loop {
            var best = -1.0; var bi = 0u;
            for (var k = 0u; k < n; k++) { if (wcut[k] >= P.G && ce[k] > best) { best = ce[k]; bi = k; } }
            if (best <= 0.0) { break; }
            let nd = nodes[wcut[bi]];
            var nk = 0u;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) { if (tree[kids[kk]].cnt > 0u) { nk += 1u; } }
            if (n - 1u + nk > K) { break; }
            var first = true;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) {
                let cc = kids[kk];
                if (tree[cc].cnt == 0u) { continue; }
                if (first) { wcut[bi] = cc; ce[bi] = bound(cc, x); first = false; }
                else { wcut[n] = cc; ce[n] = bound(cc, x); n += 1u; }
            }
        }
        wn = n;
    }
    workgroupBarrier();
    let n = wn;
    let x = wx;
    if (li < n) {
        rng = pcg_hash(c * 7919u + P.frame * 104729u + li * 31u + 7u);
        var nd = wcut[li];
        var prob = 1.0;
        while (nd >= P.G) {
            let node = nodes[nd];
            var lt = 0.0; var ct = 0.0;
            var pc: array<f32, 8>; var cc: array<u32, 8>; var m = 0u;
            for (var kk = node.child_lo; kk < node.child_hi; kk++) {
                let ch = kids[kk];
                if (tree[ch].cnt == 0u || m >= 8u) { continue; }
                let dv = x - tree[ch].cen;
                let ul = length(dv) + 1e-12;
                let l = max(sh_lum(ch, dv / ul), 0.0) / (ul * ul);
                pc[m] = l; cc[m] = ch; lt += l; ct += f32(tree[ch].cnt); m += 1u;
            }
            let u = rnd();
            var a = 0.0; var pick = m - 1u;
            for (var q = 0u; q < m; q++) {
                var pq = P.mix * f32(tree[cc[q]].cnt) / ct;
                if (lt > 0.0) { pq += (1.0 - P.mix) * pc[q] / lt; } else { pq += (1.0 - P.mix) * f32(tree[cc[q]].cnt) / ct; }
                pc[q] = pq;
            }
            for (var q = 0u; q < m; q++) { a += pc[q]; if (u < a) { pick = q; break; } }
            prob *= pc[pick]; nd = cc[pick];
        }
        wpick[li] = nd; wprob[li] = prob;
        let mem = lists[start + li % count];
        let pj = gaussians[nd].part_surface & 0xFFFFu;
        let A = point_in(mem); let Bq = point_in(nd);
        var inv = 0u;
        if (mp == 0xFFFFFFFFu) {
            let t = exp(-walk_tau(A, Bq, own, pj));
            wTn[li] = t; wTo[li] = t;
        } else if (own == mp || pj == mp) {
            // an endpoint on the moving part: the old ray starts / ends at the cache-time placement; the moving part
            // is an endpoint, so it is skipped, and the other parts are static
            inv = 1u;
            wTn[li] = exp(-walk_tau(A, Bq, own, pj));
            let Ao = select(A, A - P.dtot.xyz, own == mp);
            let Bo = select(Bq, Bq - P.dtot.xyz, pj == mp);
            wTo[li] = exp(-walk_tau(Ao, Bo, own, pj));
        } else {
            let tr = walk_tau3(A, Bq, own, pj, mp, 0xFFFFFFFFu);
            let tn = walk_tau3(A, Bq, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, mp);
            let to = walk_tau3(A + P.dtot.xyz, Bq + P.dtot.xyz, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, mp);
            var tp = 0.0;
            if (P.mdelta.w > 0.0) {
                tp = walk_tau3(A + P.mdelta.xyz, Bq + P.mdelta.xyz, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, mp);
            }
            if (P.mdelta.w > 0.0 && (tn > 0.01 || tp > 0.01)) { inv = 1u; }
            wTn[li] = exp(-(tr + tn)); wTo[li] = exp(-(tr + to));
        }
        winv[li] = inv;
    }
    workgroupBarrier();
    if (li >= count) { return; }
    let i = lists[start + li];
    let mi = vec3f(gaussians[i].mu.x, gaussians[i].mu.y, gaussians[i].mu.z);
    let mi0 = gext[i].mu0.xyz;
    var d = vec3f(0.0);
    var full = vec3f(0.0);
    var wall = 0.0; var wmov = 0.0;
    for (var k = 0u; k < n; k++) {
        let j = wpick[k];
        let Bn = gext[j].B.xyz; let Bo = gext[j].Bc.xyz;
        let mj = vec3f(gaussians[j].mu.x, gaussians[j].mu.y, gaussians[j].mu.z);
        let gn = pair_geom_at(i, j, mi, mj);
        let go = pair_geom_at(i, j, mi0, gext[j].mu0.xyz);
        let pr = wprob[k];
        d += (gn.ai * gn.om * wTn[k] * Bn - go.ai * go.om * wTo[k] * Bo) / pr;
        full += gn.ai * gn.om * wTn[k] * Bn / pr;
        let wu = gn.ai * gn.om / pr * dot(Bn, LUMV);
        wall += wu;
        if (winv[k] != 0u) { wmov += wu; }
    }
    let iso = bitcast<f32>(nodes[i].pad0);
    var share = 0.0;
    if (wall > 0.0) { share = wmov / wall; }
    if (P.mdelta.w > 0.0 && own == mp) { share = 1.0; }
    // a receiver on the moving part: all of its light changes, so the control variate's two noisy terms only add
    // variance; it takes the full estimate (P.dtot.w > 0.5 switches this off)
    var est = gext[i].Bc.xyz + gaussians[i].albedo * iso * d;
    if (own == mp && P.dtot.w < 0.5) { est = gext[i].Le_w.xyz + gaussians[i].albedo * iso * full; }
    gext[i].Bf = vec4f(est, share);
}

// motion-aware history: effective count n (B.w) capped at 4 where the moving part touches > 5% of the light this
// frame, then n + 1 up to n_max (P.alpha holds n_max); B <- B + (B_frame - B) / n
@compute @workgroup_size(64)
fn history_m(@builtin(global_invocation_id) gid: vec3u) {
    let ri = gid.x;
    if (ri >= P.R) { return; }
    let i = lists[ri];
    if (gext[i].Bf.w < 0.0) { gext[i].Bc.w = 0.0; return; }   // not gathered this frame (idle, off the trickle): keep B
    gext[i].Bc.w = gext[i].Bf.w;                    // this frame's dirty share, for the host
    var n = gext[i].B.w;
    if (P.mdelta.w > 0.0 && gext[i].Bf.w > 0.05) { n = min(n, 4.0); }
    n = min(n + 1.0, P.alpha);
    let b = gext[i].B.xyz + (gext[i].Bf.xyz - gext[i].B.xyz) / n;
    gext[i].B = vec4f(max(b, vec3f(0.0)), n);
    gext[i].Bf.w = -1.0;                            // consumed
}


// ---- renderer motion (milestone 5): every part at its own transform ------------------------------------------------
// Static data stays in each part's rest frame (SPEC 2.4). Set 0 = now, 1 = last frame, 2 = the cache's placement.
// A Gaussian's world centre is xf * mu; V and footprints take the direction rotated into its part's rest frame; walks
// carry the segment into each part's rest frame; sender centroids (tree.cen, rest) are moved with their part.

fn rows(xs_id: u32, p: u32) -> array<vec4f, 3> {
    if (xs_id == 0u) { return array<vec4f, 3>(P.xf[3u * p], P.xf[3u * p + 1u], P.xf[3u * p + 2u]); }
    if (xs_id == 1u) { return array<vec4f, 3>(P.xfp[3u * p], P.xfp[3u * p + 1u], P.xfp[3u * p + 2u]); }
    return array<vec4f, 3>(P.xfc[3u * p], P.xfc[3u * p + 1u], P.xfc[3u * p + 2u]);
}
fn gpart(g: u32) -> u32 { return gaussians[g].part_surface & 0xFFFFu; }
fn mu_rest(g: u32) -> vec3f { let m = gaussians[g].mu; return vec3f(m.x, m.y, m.z); }
fn gpos(g: u32, xs_id: u32) -> vec3f { return xf_point(rows(xs_id, gpart(g)), mu_rest(g)); }
fn npos(n: u32, xs_id: u32) -> vec3f { return xf_point(rows(xs_id, nodes[n].part), tree[n].cen); }
fn op_onb(w: vec3f) -> array<vec3f, 2> {       // two unit vectors perpendicular to w (material_ext's onb is the specular pass's)
    var a = vec3f(1.0, 0.0, 0.0);
    if (abs(w.x) > 0.9) { a = vec3f(0.0, 1.0, 0.0); }
    let t = normalize(cross(w, a));
    return array<vec3f, 2>(t, cross(w, t));
}
fn point_rest(g: u32) -> vec3f {               // a point drawn from Gaussian g, rest frame
    let G = gaussians[g];
    let a = gauss2(); let b = gauss2();
    let sc = vec3f(G.scale.x, G.scale.y, G.scale.z);
    let off = sc * vec3f(a.x, a.y, b.x);
    let R = quat_mat(G.quat);
    return mu_rest(g) + R[0] * off.x + R[1] * off.y + R[2] * off.z;
}

fn geom_x(i: u32, j: u32, xs_id: u32) -> Geom {
    let pi = gpart(i); let pj = gpart(j);
    if (pi == pj) { return Geom(0.0, 0.0); }
    let ri = rows(xs_id, pi); let rj = rows(xs_id, pj);
    let dv = xf_point(rj, mu_rest(j)) - xf_point(ri, mu_rest(i));
    let r2 = dot(dv, dv); let r = sqrt(r2);
    let w = dv / r;
    let wi = xf_dir_inv(ri, w); let wj = xf_dir_inv(rj, w);
    let vi = vgrid_lookup(vgrid[i], wi);
    if (vi <= 1e-4) { return Geom(0.0, 0.0); }
    let vj = vgrid_lookup(vgrid[j], -wj);
    if (vj <= 1e-4) { return Geom(0.0, 0.0); }
    var aj = footprint(nodes[j], det_c(j), wj) * vj * gext[j].Le_w.w;
    if ((gext[j].Le_w.x + gext[j].Le_w.y + gext[j].Le_w.z) > 0.0) {   // emitters here are one-sided
        let nrm = quat_mat(gaussians[j].quat)[0];
        aj = lamp_area(j, aj, nrm, -dot(nrm, wj), vj * gext[j].Le_w.w);
        if (aj <= 0.0) { return Geom(0.0, 0.0); }
    }
    let ai = footprint(nodes[i], det_c(i), wi) * vi;
    return Geom(ai, 2.0 * PI * (1.0 - r / sqrt(r2 + aj / PI)));
}

// optical depth on the world segment A -> B over the parts in `mask` (bit p), each at transform set xs_id (0 now, 1 last frame, 2 cache)
fn walk_x(A: vec3f, Bp: vec3f, mask: u32, xs_id: u32) -> f32 {
    var tau = 0.0;
    var stack: array<u32, STACK>;
    for (var p = 0u; p < P.part_count; p++) {
        if ((mask & (1u << p)) == 0u) { continue; }
        let rr = rows(xs_id, p);
        let a = xf_point_inv(rr, A); let b = xf_point_inv(rr, Bp);
        let dv = b - a;
        let L = length(dv);
        if (L < 1e-9) { continue; }
        let w = dv / L;
        let pb = P.pbound[p];
        let c = pb.xyz - a;
        let t = clamp(dot(c, w), 0.0, L);
        let e = c - t * w;
        if (dot(e, e) > pb.w * pb.w) { continue; }
        stack[0] = root_of(p);
        var sp = 1u;
        while (sp > 0u) {
            sp -= 1u;
            let ni = stack[sp];
            let nd = nodes[ni];
            let d = nd.mean - a;
            let tt = clamp(dot(d, w), 0.0, L);
            let ee = d - tt * w;
            if (dot(ee, ee) > nd.radius * nd.radius) { continue; }
            var use_it = ni < P.G;
            if (!use_it && P.theta > 0.0) { use_it = dot(d, d) > P.theta * P.theta * nd.smax * nd.smax; }
            if (use_it) { tau += nd.kappa * dirm(nd, w) * seg_geom(nd, a, w, 0.0, L); continue; }
            for (var k = nd.child_lo; k < nd.child_hi; k++) { if (sp < STACK) { stack[sp] = kids[k]; sp += 1u; } }
        }
    }
    return tau;
}

var<workgroup> wmode: u32;                      // 0 skip, else the cluster's budget K this frame
var<workgroup> wnear: atomic<u32>;              // parts within CONTACT of the cluster (bit p): members walk them themselves
var<workgroup> wnear_u: u32;                    // its value once all threads have voted (naga: no uniform load of an atomic)
var<workgroup> wrc: f32;                        // cluster radius (members' centres around the centroid)
var<workgroup> wpb: array<vec3f, KMAX>;         // each pick's ray end point, the sender's rest frame
var<workgroup> wTsn: array<f32, KMAX>;          // the shared depth over the parts not near the cluster, now
var<workgroup> wTso: array<f32, KMAX>;          // the same with the cache placement
const NDIST: u32 = 16u;                         // distant rays (sky + sun) per cluster and frame, at most
const ENV_BIT: u32 = 0x80000000u;               // masks.x / y: the environment changed (since the cache / this frame)
var<workgroup> wdn: u32;
var<workgroup> wdu: array<vec3f, NDIST>;        // distant ray directions (world)
var<workgroup> wdL: array<vec3f, NDIST>;        // their radiance weight (L Omega / p, or E / rays)
var<workgroup> wdTn: array<f32, NDIST>;         // shared depth over the far parts, now
var<workgroup> wdTo: array<f32, NDIST>;         // the same from the cache placement
var<workgroup> wdinv: array<u32, NDIST>;        // dirty: through a part that moved this frame

const CONTACT: f32 = 0.03;

// is any Gaussian of part p within r of world point x (part p at transform set 0)? tree descent on node spheres
fn part_near(p: u32, x: vec3f, r: f32) -> bool {
    let xl = xf_point_inv(rows(0u, p), x);
    var stack: array<u32, STACK>;
    stack[0] = root_of(p);
    var sp = 1u;
    var visits = 0u;
    while (sp > 0u && visits < 4096u) {
        sp -= 1u; visits += 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        let rr = select(nd.radius, 3.0 * nd.smax, ni < P.G);     // a leaf: its 3 sigma extent
        if (length(nd.mean - xl) > rr + r) { continue; }
        if (ni < P.G) { return true; }
        for (var k = nd.child_lo; k < nd.child_hi; k++) { if (sp < STACK) { stack[sp] = kids[k]; sp += 1u; } }
    }
    return false;
}

// One workgroup per receiver cluster (all clusters every frame; idle ones at K idle while a part moves, else on a
// 1/period trickle; skipped ones exit).
// K active for a cluster with a member dirty last frame (share in Bc.w above the threshold) or early in its history
// (count < KFRESH); K idle while a member is still converging (count < n_max) or while any part moves.
// With a valid cache: control variate (B_frame = Bc + alb iso sum (c_now - c_cache) / p, the cache term with every part
// at its cache transform), except receivers on a part moved since the cache, which take the full estimate.
@compute @workgroup_size(64)
fn gather_m(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let c = wid.x;
    if (c >= P.ncl) { return; }
    let start = lists[P.cl_off + 2u * c];
    let count = lists[P.cl_off + 2u * c + 1u];
    let own = gpart(lists[start]);
    let all = (1u << P.part_count) - 1u;
    let msince = P.masks.x; let mframe = P.masks.y; let cache_ok = P.masks.z != 0u;
    if (li == 0u) {
        // fresh: a member dirty last frame or early in its history (count < KFRESH) -> K active; converging (count below
        // n_max) -> K idle: the history averages over many frames, so the tail needs few samples per frame
        var young = false; var conv = false; var smax = 0.0; var cmin = 1e9;
        for (var m = 0u; m < count; m++) {
            let g = lists[start + m];
            if (gext[g].B.w < KFRESH) { young = true; }
            smax = max(smax, gext[g].Bc.w);
            if (gext[g].B.w < P.knobs.x - 0.5) { conv = true; }
            cmin = min(cmin, gext[g].B.w);
        }
        let fresh = young || smax > P.knobs.z;
        // a dirty cluster whose change is a small share of its light (a moving box's bounce across the room) needs fewer
        // samples than one in the box's shadow: K scales with the dirty share, full K from SHARE_FULL up
        var kf = P.K;
        if (smax > P.knobs.z) { kf = clamp(u32(ceil(f32(P.K) * smax / SHARE_FULL)), P.masks.w, P.K); }
        let period = max(u32(P.knobs.y), 1u);
        // while a part moves every cluster gathers at least K idle: the gather is the dirty test, so a cluster left idle
        // would never notice that its light changed (a moved box's old shadow stayed on the floor)
        let probe = mframe != 0u || conv || (c % period) == (P.frame % period);
        wmode = select(select(0u, P.masks.w, probe), kf, fresh);
        var xs = vec3f(0.0);
        for (var m = 0u; m < count; m++) { xs += gpos(lists[start + m], 0u); }
        let x = xs / f32(count);
        wx = x;
        var rc = 0.0;
        for (var m = 0u; m < count; m++) { rc = max(rc, length(gpos(lists[start + m], 0u) - x)); }
        wrc = rc;
        // progressive scheduling: a converging cluster only gathers when its priority makes this frame's budget, then at
        // K sched (its history counts the sample K / K idle times); while parts move, the rest still probe at K idle
        if (P.sched.x > 0u && !fresh && conv && cmin >= SCHED_MIN) {
            let b = sched_bucket(sched_score(x, rc, start, count));
            let cut = u32(gext[P.G].Le_w.x);
            wmode = select(select(0u, P.masks.w, mframe != 0u), P.sched.y, b >= cut);
        }
        atomicStore(&wnear, 0u);
        var n = 0u;
        var K = min(wmode, KMAX);
        // soft: each receiver blends P.soft.x clusters, so each takes 1 / P.soft.w of the picks (at least one per part).
        // Converging clusters (K idle or less) keep their K: gather_soft then draws one neighbour per receiver instead
        if (P.soft.x > 0u && K > P.masks.w) { K = min(max((K + P.soft.w - 1u) / P.soft.w, P.part_count - 1u), KMAX); }
        var ce: array<f32, KMAX>;
        if (K > 0u) {
            for (var p = 0u; p < P.part_count; p++) {
                let r = root_of(p);
                if (p == own || tree[r].cnt == 0u || n >= K) { continue; }
                wcut[n] = r; ce[n] = bound_x(r, x); n += 1u;
            }
            loop {
                var best = -1.0; var bi = 0u;
                for (var k = 0u; k < n; k++) { if (wcut[k] >= P.G && ce[k] > best) { best = ce[k]; bi = k; } }
                if (best <= 0.0) { break; }
                let nd = nodes[wcut[bi]];
                var nk = 0u;
                for (var kk = nd.child_lo; kk < nd.child_hi; kk++) { if (tree[kids[kk]].cnt > 0u) { nk += 1u; } }
                if (n - 1u + nk > K) { break; }
                var first = true;
                for (var kk = nd.child_lo; kk < nd.child_hi; kk++) {
                    let cc = kids[kk];
                    if (tree[cc].cnt == 0u) { continue; }
                    if (first) { wcut[bi] = cc; ce[bi] = bound_x(cc, x); first = false; }
                    else { wcut[n] = cc; ce[n] = bound_x(cc, x); n += 1u; }
                }
            }
        }
        wn = n;
    }
    let mode = workgroupUniformLoad(&wmode);    // a barrier too; uniform, so the early return keeps barriers uniform
    let n = wn;
    if (mode == 0u) {                           // not gathered this frame: history leaves these members alone
        if (li < count) { gext[lists[start + li]].Bf.w = -1.0; }
        if (li == 0u && P.soft.x > 0u) { sb[P.soft.y + c] = vec4u(0u); }
        return;
    }
    let x = wx;
    // contact-aware sharing (E3's contact rule): parts with a Gaussian within CONTACT of the cluster's extent are walked
    // by each member itself; only the rest of each ray is shared
    if (li < P.part_count && li != own) {
        if (part_near(li, x, wrc + CONTACT)) { atomicOr(&wnear, 1u << li); }
    }
    workgroupBarrier();
    if (li == 0u) { wnear_u = atomicLoad(&wnear); }
    let near = workgroupUniformLoad(&wnear_u);
    if (li < n) {
        rng = pcg_hash(c * 7919u + P.frame * 104729u + li * 31u + 7u);
        var nd = wcut[li];
        var prob = 1.0;
        while (nd >= P.G) {
            let node = nodes[nd];
            var lt = 0.0; var ct = 0.0;
            var pc: array<f32, 8>; var cc: array<u32, 8>; var m = 0u;
            for (var kk = node.child_lo; kk < node.child_hi; kk++) {
                let ch = kids[kk];
                if (tree[ch].cnt == 0u || m >= 8u) { continue; }
                let dv = x - npos(ch, 0u);
                let ul = length(dv) + 1e-12;
                let l = max(sh_lum(ch, dv / ul), 0.0) / (ul * ul);
                pc[m] = l; cc[m] = ch; lt += l; ct += f32(tree[ch].cnt); m += 1u;
            }
            let u = rnd();
            var a = 0.0; var pick = m - 1u;
            for (var q = 0u; q < m; q++) {
                var pq = P.mix * f32(tree[cc[q]].cnt) / ct;
                if (lt > 0.0) { pq += (1.0 - P.mix) * pc[q] / lt; } else { pq += (1.0 - P.mix) * f32(tree[cc[q]].cnt) / ct; }
                pc[q] = pq;
            }
            for (var q = 0u; q < m; q++) { a += pc[q]; if (u < a) { pick = q; break; } }
            prob *= pc[pick]; nd = cc[pick];
        }
        wpick[li] = nd; wprob[li] = prob;
        let mem = lists[start + li % count];
        let pj = gpart(nd);
        let pa = point_rest(mem); let pb = point_rest(nd);
        wpb[li] = pb;
        let A = xf_point(rows(0u, own), pa); let Bq = xf_point(rows(0u, pj), pb);
        let others = all & ~(1u << own) & ~(1u << pj);
        // shared over the cluster: parts that are neither near it nor moved since the cache (static, smooth shadows);
        // near parts and moved parts are walked per member below, so a moving object's shadow stays per-Gaussian
        let far = others & ~near & ~msince;
        var inv = 0u;
        let ends_moved = (msince & ((1u << own) | (1u << pj))) != 0u;
        wTsn[li] = walk_x(A, Bq, far, 0u);
        if (cache_ok && ends_moved) {
            let Ao = xf_point(rows(2u, own), pa); let Bo = xf_point(rows(2u, pj), pb);
            wTso[li] = walk_x(Ao, Bo, far, 2u);
        } else {
            wTso[li] = wTsn[li];
        }
        // dirty: an endpoint on a part that moved this frame, or the ray through such a part now or a frame ago
        if (mframe != 0u) {
            if ((mframe & ((1u << own) | (1u << pj))) != 0u) { inv = 1u; }
            else if ((mframe & near) != 0u) { inv = 1u; }
            else if (walk_x(A, Bq, others & mframe, 0u) > 0.01 || walk_x(A, Bq, others & mframe, 1u) > 0.01) { inv = 1u; }
        }
        winv[li] = inv;
        if (P.soft.x > 0u) {
            sb[2u * (c * KMAX + li)] = bitcast<vec4u>(vec4f(pb, prob));
            sb[2u * (c * KMAX + li) + 1u] = vec4u(nd, bitcast<u32>(wTsn[li]), bitcast<u32>(wTso[li]), inv);
        }
    }
    if (li == 0u && P.soft.x > 0u) { sb[P.soft.y + c] = vec4u(n, mode, near, 0u); }
    // sky bins: rays shared over the cluster like the picks (far parts), near and moved parts walked per member below.
    // The sun's shadow is sharp (ASG std ~1 degree), so a shared ray would show the cluster grid along its edges: each
    // receiver walks its own sun rays below
    let nd = min(P.envu.y, NDIST);
    if (li < nd) {
        rng = pcg_hash(c * 6271u + P.frame * 92821u + li * 17u + 11u);
        let r = rnd();
        var b = 0u;
        for (var q = 0u; q < P.envu.x; q++) { b = q; if (r < P.brad[q].w) { break; } }
        let pb = P.brad[b].w - select(0.0, P.brad[max(b, 1u) - 1u].w, b > 0u);
        let w = P.bins[b].xyz;
        let cosa = max(-1.0, 1.0 - P.bins[b].w / (2.0 * PI));
        let cc = 1.0 - rnd() * (1.0 - cosa); let sn = sqrt(max(0.0, 1.0 - cc * cc)); let ph = 2.0 * PI * rnd();
        let on = op_onb(w);
        let u = cc * w + sn * cos(ph) * on[0] + sn * sin(ph) * on[1];
        let L = P.brad[b].xyz / (max(pb, 1e-12) * f32(nd));
        let mem = lists[start + li % count];
        let pa = point_rest(mem);
        let A = xf_point(rows(0u, own), pa);
        let far = all & ~(1u << own) & ~near & ~msince;
        wdu[li] = u; wdL[li] = L;
        wdTn[li] = walk_x(A, A + 1000.0 * u, far, 0u);
        if (cache_ok && (msince & (1u << own)) != 0u) {
            let Ao = xf_point(rows(2u, own), pa);
            wdTo[li] = walk_x(Ao, Ao + 1000.0 * u, far, 2u);
        } else { wdTo[li] = wdTn[li]; }
        var inv = 0u;
        if ((mframe & ENV_BIT) != 0u || (mframe & ((1u << own) | near)) != 0u) { inv = 1u; }
        else if (mframe != 0u) {
            let mv = all & ~(1u << own) & mframe;
            if (walk_x(A, A + 1000.0 * u, mv, 0u) > 0.01 || walk_x(A, A + 1000.0 * u, mv, 1u) > 0.01) { inv = 1u; }
        }
        wdinv[li] = inv;
    }
    workgroupBarrier();
    if (li >= count) { return; }
    let i = lists[start + li];
    rng = pcg_hash(i * 7907u + P.frame * 6113u + 3u);
    let pi_rest = point_rest(i);
    var d = vec3f(0.0); var full = vec3f(0.0);
    var wall = 0.0; var wmov = 0.0;
    let nk = select(n, 0u, P.soft.x > 0u);       // soft: gather_soft adds the picks of the neighbour clusters
    for (var k = 0u; k < nk; k++) {
        let t = pick_term(i, own, pi_rest, wpick[k], wprob[k], wpb[k], wTsn[k], wTso[k], near, winv[k] != 0u);
        full += t.full; d += t.d; wall += t.w.x; wmov += t.w.y;
    }
    let mine_d = (near | msince) & all & ~(1u << own);
    for (var k = 0u; k < nd; k++) {
        let u = wdu[k];
        let rn = rows(0u, own);
        let wi = xf_dir_inv(rn, u);
        let an = footprint(nodes[i], det_c(i), wi) * vgrid_lookup(vgrid[i], wi);
        var ao = an;
        let own_moved = cache_ok && (msince & (1u << own)) != 0u;
        if (own_moved) {
            let wo = xf_dir_inv(rows(2u, own), u);
            ao = footprint(nodes[i], det_c(i), wo) * vgrid_lookup(vgrid[i], wo);
        }
        if (an <= 1e-6 && ao <= 1e-6) { continue; }
        var tn = wdTn[k]; var to = wdTo[k];
        if (mine_d != 0u) {
            let A = xf_point(rn, pi_rest);
            tn += walk_x(A, A + 1000.0 * u, mine_d, 0u);
            if (cache_ok && msince != 0u) {
                let Ao = xf_point(rows(2u, own), pi_rest);
                to += walk_x(Ao, Ao + 1000.0 * u, mine_d, 2u);
            } else { to = tn; }
        }
        let cn = an * exp(-tn) * wdL[k];
        full += cn;
        if (cache_ok) { d += cn - ao * exp(-to) * wdL[k]; }
        let wu = an * dot(wdL[k], LUMV);
        wall += wu;
        if (wdinv[k] != 0u) { wmov += wu; }
    }
    // the sun: rays per receiver from its own points, directions over the ASG, every other part walked
    let osun = all & ~(1u << own);
    for (var k = 0u; k < P.envu.z; k++) {
        let g = gauss2() * P.sun.w;
        let on = op_onb(P.sun.xyz);
        let u = normalize(sqrt(max(0.0, 1.0 - dot(g, g))) * P.sun.xyz + g.x * on[0] + g.y * on[1]);
        let L = P.sunE.xyz / f32(P.envu.z);
        let rn = rows(0u, own);
        let wi = xf_dir_inv(rn, u);
        let an = footprint(nodes[i], det_c(i), wi) * vgrid_lookup(vgrid[i], wi);
        var ao = an;
        let own_moved = cache_ok && (msince & (1u << own)) != 0u;
        if (own_moved) {
            let wo = xf_dir_inv(rows(2u, own), u);
            ao = footprint(nodes[i], det_c(i), wo) * vgrid_lookup(vgrid[i], wo);
        }
        if (an <= 1e-6 && ao <= 1e-6) { continue; }
        let pp = point_rest(i);
        let A = xf_point(rn, pp);
        let tn = walk_x(A, A + 1000.0 * u, osun, 0u);
        var to = tn;
        if (cache_ok && msince != 0u) {
            let Ao = xf_point(rows(2u, own), pp);
            to = walk_x(Ao, Ao + 1000.0 * u, osun, 2u);
        }
        let cn = an * exp(-tn) * L;
        full += cn;
        if (cache_ok) { d += cn - ao * exp(-to) * L; }
        let wu = an * dot(L, LUMV);
        wall += wu;
        if ((mframe & (ENV_BIT | (1u << own))) != 0u) { wmov += wu; }
        else if ((mframe & osun) != 0u) {
            if (abs(walk_x(A, A + 1000.0 * u, mframe & osun, 0u) - walk_x(A, A + 1000.0 * u, mframe & osun, 1u)) > 0.01) { wmov += wu; }
        }
    }
    gext[i].mu0.w = select(1.0, f32(mode) / f32(max(P.masks.w, 1u)), P.sched.x > 0u);   // history weight
    if (P.soft.x > 0u) {
        sb[P.soft.z + 2u * i] = bitcast<vec4u>(vec4f(full, wall));
        sb[P.soft.z + 2u * i + 1u] = bitcast<vec4u>(vec4f(d, wmov));
        return;
    }
    finish(i, own, full, d, wall, wmov);
}

// B_frame and the dirty share from the summed terms (gather_m, or gather_soft after the blend)
fn finish(i: u32, own: u32, full: vec3f, d: vec3f, wall: f32, wmov: f32) {
    let msince = P.masks.x; let mframe = P.masks.y; let cache_ok = P.masks.z != 0u;
    let iso = bitcast<f32>(nodes[i].pad0);
    var share = 0.0;
    if (wall > 0.0) { share = wmov / wall; }
    if ((mframe & ((1u << own) | ENV_BIT)) != 0u) { share = 1.0; }
    var est = gext[i].Le_w.xyz + gaussians[i].albedo * iso * full;
    if (cache_ok && (msince & ((1u << own) | ENV_BIT)) == 0u && P.knobs.w == 0.0) { est = gext[i].Bc.xyz + gaussians[i].albedo * iso * d; }
    gext[i].Bf = vec4f(est, share);
}

struct PickTerm { full: vec3f, d: vec3f, w: vec2f }   // full estimate, control-variate change, (light weight, dirty weight)

// one pick's contribution to receiver i: sender j (probability pr, ray end point pb in j's rest frame), the cluster's
// shared depth over its far parts (now, cache placement), the near and moved parts walked from i's own point
fn pick_term(i: u32, own: u32, pi_rest: vec3f, j: u32, pr: f32, pb: vec3f, tsn: f32, tso: f32, near: u32, dirty: bool) -> PickTerm {
    var r = PickTerm(vec3f(0.0), vec3f(0.0), vec2f(0.0));
    let msince = P.masks.x; let cache_ok = P.masks.z != 0u;
    let Bn = gext[j].B.xyz; let Bo = gext[j].Bc.xyz;
    let gn = geom_x(i, j, 0u);
    var go = gn;
    if (cache_ok && msince != 0u) { go = geom_x(i, j, 2u); }
    if (gn.ai * gn.om <= 0.0 && go.ai * go.om <= 0.0) { return r; }
    let pj = gpart(j);
    let mine = (near | msince) & ~(1u << own) & ~(1u << pj);
    var tn = tsn; var to = tso;
    if (mine != 0u) {
        let A = xf_point(rows(0u, own), pi_rest); let Bq = xf_point(rows(0u, pj), pb);
        tn += walk_x(A, Bq, mine, 0u);
        if (cache_ok && msince != 0u) {
            let Ao = xf_point(rows(2u, own), pi_rest); let Bo2 = xf_point(rows(2u, pj), pb);
            to += walk_x(Ao, Bo2, mine, 2u);
        } else { to = tn; }
    }
    let Tn = exp(-tn); let To = exp(-to);
    r.full = gn.ai * gn.om * Tn * Bn / pr;
    if (cache_ok) { r.d = (gn.ai * gn.om * Tn * Bn - go.ai * go.om * To * Bo) / pr; }
    let wu = gn.ai * gn.om / pr * dot(Bn, LUMV);
    r.w = vec2f(wu, select(0.0, wu, dirty));
    return r;
}

// soft membership, per receiver: the picks of its P.soft.x nearest clusters on its own part (snb; weights fall to 0 at
// the next cluster out, so the blend is continuous), each cluster's sum a full estimate; their weighted mean plus the
// receiver's own sky and sun terms from gather_m. Clusters not gathered this frame drop out of the mean.
@compute @workgroup_size(64)
fn gather_soft(@builtin(global_invocation_id) gid: vec3u) {
    let ri = gid.x;
    if (ri >= P.R) { return; }
    let i = lists[ri];
    let nb = snb[2u * i]; let wb = bitcast<vec4f>(snb[2u * i + 1u]);
    if (sb[P.soft.y + nb.x].y == 0u) { return; }   // own cluster not gathered: gather_m set Bf.w = -1
    let own = gpart(i);
    rng = pcg_hash(i * 7907u + P.frame * 6113u + 5u);
    let pi_rest = point_rest(i);
    var full = vec3f(0.0); var d = vec3f(0.0); var wall = 0.0; var wmov = 0.0; var wsum = 0.0;
    // converging (own cluster at K idle or less): one neighbour drawn by weight among those gathered this frame, so the
    // cost is today's and the history does the blend; fresh / dirty: every neighbour's picks, weighted mean
    var only = 4u;
    if (sb[P.soft.y + nb.x].y <= P.masks.w) {
        var tot = 0.0;
        for (var q = 0u; q < min(P.soft.x, 4u); q++) { if (wb[q] > 0.0 && sb[P.soft.y + nb[q]].y != 0u) { tot += wb[q]; } }
        var u = rnd() * tot;
        for (var q = 0u; q < min(P.soft.x, 4u); q++) {
            if (wb[q] <= 0.0 || sb[P.soft.y + nb[q]].y == 0u) { continue; }
            only = q; u -= wb[q];
            if (u < 0.0) { break; }
        }
    }
    for (var q = 0u; q < min(P.soft.x, 4u); q++) {
        let c = nb[q]; var wq = wb[q];
        if (only < 4u) { wq = select(0.0, 1.0, q == only); }
        if (wq <= 0.0) { continue; }
        let h = sb[P.soft.y + c];
        if (h.y == 0u) { continue; }
        let n = min(h.x, KMAX); let near = h.z;
        for (var k = 0u; k < n; k++) {
            let a = bitcast<vec4f>(sb[2u * (c * KMAX + k)]); let b = sb[2u * (c * KMAX + k) + 1u];
            let t = pick_term(i, own, pi_rest, b.x, a.w, a.xyz, bitcast<f32>(b.y), bitcast<f32>(b.z), near, b.w != 0u);
            full += wq * t.full; d += wq * t.d; wall += wq * t.w.x; wmov += wq * t.w.y;
        }
        wsum += wq;
    }
    let s = 1.0 / max(wsum, 1e-12);
    let e0 = bitcast<vec4f>(sb[P.soft.z + 2u * i]); let e1 = bitcast<vec4f>(sb[P.soft.z + 2u * i + 1u]);
    finish(i, own, e0.xyz + full * s, e1.xyz + d * s, e0.w + wall * s, e1.w + wmov * s);
}

fn bound_x(n: u32, x: vec3f) -> f32 {
    if (n < P.G) { return 0.0; }
    let tn = tree[n];
    let d = length(npos(n, 0u) - x) - tn.rad;
    let lim = 2.0 * PI * tn.Bmax;
    if (d <= 1e-6) { return lim; }
    return min(tn.Ib / (d * d), lim);
}

// ---- progressive scheduling (?budget=N): which converging clusters gather this frame ---------------------------------
// Priority = on-screen size (cluster radius / view depth; off-screen clusters count 5% of their size at their distance)
// x how far from converged (1 - mean count / n_max). sched_cut bins every converging cluster's priority into 64 log
// buckets and finds the bucket where the budget runs out; gather_m compares against it (gext[G].Le_w.x).

fn sched_score(x: vec3f, rc: f32, start: u32, count: u32) -> f32 {
    var cs = 0.0;
    for (var m = 0u; m < count; m++) { cs += gext[lists[start + m]].B.w; }
    let left = max(1.0 - cs / (f32(count) * P.knobs.x), 0.0);
    let c = P.vp * vec4f(x, 1.0);
    var size = 0.0;
    if (c.w > 1e-3) {
        let ndc = c.xyz / c.w;
        let on = all(abs(ndc.xy) < vec2f(1.0 + 4.0 * rc / c.w)) && ndc.z < 1.0;
        size = (rc + 0.02) / c.w * select(0.05, 1.0, on);
    } else {
        size = 0.05 * (rc + 0.02) / max(-c.w, 0.1);
    }
    return size * left;
}

fn sched_bucket(s: f32) -> u32 {
    if (s <= 0.0) { return 0u; }
    return u32(clamp(floor(log2(s) * 4.0) + 48.0, 1.0, 63.0));
}

var<workgroup> shist: array<atomic<u32>, 64>;

@compute @workgroup_size(256)
fn sched_cut(@builtin(local_invocation_index) li: u32) {
    if (li < 64u) { atomicStore(&shist[li], 0u); }
    workgroupBarrier();
    for (var c = li; c < P.ncl; c += 256u) {
        let start = lists[P.cl_off + 2u * c];
        let count = lists[P.cl_off + 2u * c + 1u];
        var conv = false; var young = false; var smax = 0.0;
        var xs = vec3f(0.0);
        for (var m = 0u; m < count; m++) {
            let g = lists[start + m];
            if (gext[g].B.w < max(KFRESH, SCHED_MIN)) { young = true; }
            if (gext[g].B.w < P.knobs.x - 0.5) { conv = true; }
            smax = max(smax, gext[g].Bc.w);
            xs += gpos(g, 0u);
        }
        if (!conv || young || smax > P.knobs.z) { continue; }
        let x = xs / f32(count);
        var rc = 0.0;
        for (var m = 0u; m < count; m++) { rc = max(rc, length(gpos(lists[start + m], 0u) - x)); }
        atomicAdd(&shist[sched_bucket(sched_score(x, rc, start, count))], 1u);
    }
    workgroupBarrier();
    if (li == 0u) {
        var acc = 0u;
        var cut = 1u;
        for (var b = 63u; b >= 1u; b--) {
            acc += atomicLoad(&shist[b]);
            if (acc >= P.sched.x) { cut = b; break; }
        }
        gext[P.G].Le_w = vec4f(f32(cut), 1.0, 0.0, 0.0);
    }
}

// history for gather_m: count n (B.w) capped at 4 where the moving parts touch more than the threshold of the light,
// then n + 1 up to n_max, B <- B + (B_frame - B) / n; the dirty share goes to Bc.w for the next frame's activity test
@compute @workgroup_size(64)
fn history_x(@builtin(global_invocation_id) gid: vec3u) {
    let ri = gid.x;
    if (ri >= P.R) { return; }
    let i = lists[ri];
    if (gext[i].Bf.w < 0.0) { gext[i].Bc.w = 0.0; return; }
    let share = gext[i].Bf.w;
    var n = gext[i].B.w;
    if (P.masks.y != 0u && share > P.knobs.z) { n = min(n, 4.0); }
    // a scheduled gather at K sched counts K / K idle samples (gather_m writes the weight into mu0.w)
    let wgt = select(1.0, max(gext[i].mu0.w, 1.0), P.sched.x > 0u);
    n = min(n + wgt, P.knobs.x);
    let b = gext[i].B.xyz + (gext[i].Bf.xyz - gext[i].B.xyz) * min(wgt / max(n, 1.0), 1.0);
    gext[i].B = vec4f(max(b, vec3f(0.0)), n);
    gext[i].Bc.w = select(0.0, share, P.masks.y != 0u);
    gext[i].Bf.w = -1.0;
}

// the scene has been still long enough: the history becomes the control variate's cache (B -> Bc for every Gaussian)
@compute @workgroup_size(64)
fn rebase(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= P.G) { return; }
    gext[g].Bc = vec4f(gext[g].B.xyz, 0.0);
}
