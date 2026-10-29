// Hierarchical shadow walk (Renderer shell thread, approved by David 2026-10-25). Prepend common.wgsl.
//
// walk_emit walks every Gaussian's shadow ray through every other part. Most of those rays hit nothing, and whole
// subtrees of the scene tree share that answer, so before it this pass asks it once per scene-tree node, top down:
// does the convex hull of the node (its sphere and its box) and the emitters' sample region (the light box) touch any
// Gaussian of another part? If not, every shadow ray from under the node misses every other part, so walk_emit writes
// tau = 0 for its Gaussians without walking (node_status = HS_CLEAR). Otherwise the children are asked, and the
// Gaussians whose own hull (their centre to the light box) still touches something walk as before.
//
// "Touch" is a separating-axis test between the hull and an occluder node (its bounding sphere and its rest-frame box,
// node_box): the hull's projection onto an axis is the hull of its parts' projections, so each test is exact for that
// axis, and a node is passed only when no tested axis separates. Leaves' boxes are their 3-sigma ellipsoids' boxes (the
// same 3 sigma as Node.radius), so the cull agrees with the leaf walk up to that truncation. It ignores walk_tau's
// theta merge: a merged node far from the receiver can put a little optical depth where none of its leaves are; the
// cull writes 0 there.
//
//   hs_box_leaf   once per asset: each leaf's rest-frame box (3 sigma per axis)
//   hs_box_up     once per asset, per depth (deepest first): an internal node's box = its children's, and the children's
//                 parent index
//   shadow_cull   per frame, per depth from the part roots down, over that depth's nodes (level_nodes): clear when the
//                 parent is, else the hull test
//
// HP.light_lo / light_hi is the world box every emitter_sample point can land in (lighting.js shadowLightBox).

struct HSParams { light_lo: vec3f, skip_part: u32, light_hi: vec3f, budget: u32 }   // budget bit 31: leaves not tested
struct HSLevel { offset: u32, count: u32, depth: u32, skip: u32 }   // skip 1: this depth's nodes are not tested (they walk)

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(4) var<storage, read> parts: array<Part>;
@group(0) @binding(9) var<storage, read> kids: array<u32>;
@group(0) @binding(33) var<storage, read_write> node_status: array<u32>;
@group(0) @binding(34) var<storage, read_write> node_box: array<vec4f>;      // per node: lo.xyz, hi.xyz (rest frame)
@group(0) @binding(35) var<storage, read_write> node_parent: array<u32>;
@group(0) @binding(36) var<storage, read> level_nodes: array<u32>;
@group(0) @binding(37) var<uniform> HP: HSParams;
@group(0) @binding(38) var<uniform> LV: HSLevel;

const HS_WALK: u32 = 0u;
const HS_CLEAR: u32 = 1u;

@compute @workgroup_size(64)
fn hs_box_leaf(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let R = quat_matrix(G.quat);
    let s2 = G.scale * G.scale;
    let rx = vec3f(R[0].x, R[1].x, R[2].x);
    let ry = vec3f(R[0].y, R[1].y, R[2].y);
    let rz = vec3f(R[0].z, R[1].z, R[2].z);
    let h = 3.0 * sqrt(vec3f(dot(rx * rx, s2), dot(ry * ry, s2), dot(rz * rz, s2)));
    node_box[2u * g] = vec4f(G.mu - h, 0.0);
    node_box[2u * g + 1u] = vec4f(G.mu + h, 0.0);
}

@compute @workgroup_size(64)
fn hs_box_up(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= LV.count) { return; }
    let ni = level_nodes[LV.offset + i];
    if (ni < U.gaussian_count) { return; }
    let nd = nodes[ni];
    var lo = vec3f(3.0e38);
    var hi = vec3f(-3.0e38);
    for (var j = nd.child_lo; j < nd.child_hi; j++) {
        let c = kids[j];
        lo = min(lo, node_box[2u * c].xyz);
        hi = max(hi, node_box[2u * c + 1u].xyz);
        node_parent[c] = ni;
    }
    node_box[2u * ni] = vec4f(lo, 0.0);
    node_box[2u * ni + 1u] = vec4f(hi, 0.0);
}

// The hull, in the occluder part's rest frame: the receiver (sphere c, r intersected with the box at bc, axes rax,
// half sizes rh) and the light box (lc, lax, lh).
struct Hull { c: vec3f, r: f32, bc: vec3f, rax: mat3x3f, rh: vec3f, lc: vec3f, lax: mat3x3f, lh: vec3f }

fn box_extent(u: vec3f, ax: mat3x3f, h: vec3f) -> f32 {
    return abs(dot(u, ax[0])) * h.x + abs(dot(u, ax[1])) * h.y + abs(dot(u, ax[2])) * h.z;
}

fn hull_interval(H: Hull, u: vec3f) -> vec2f {
    let s = dot(u, H.c);
    let b = dot(u, H.bc);
    let be = box_extent(u, H.rax, H.rh);
    let l = dot(u, H.lc);
    let le = box_extent(u, H.lax, H.lh);
    return vec2f(min(max(s - H.r, b - be), l - le), max(min(s + H.r, b + be), l + le));
}

// does the axis u separate the hull from the occluder node (sphere m, R intersected with its box at bc, half bh)?
fn separates(H: Hull, u: vec3f, m: vec3f, R: f32, bc: vec3f, bh: vec3f) -> bool {
    let a = hull_interval(H, u);
    let s = dot(u, m);
    let b = dot(u, bc);
    let e = dot(abs(u), bh);
    return a.y < max(s - R, b - e) || min(s + R, b + e) < a.x;
}

fn hull_touches(H: Hull, ni: u32) -> bool {
    let nd = nodes[ni];
    let lo = node_box[2u * ni].xyz;
    let hi = node_box[2u * ni + 1u].xyz;
    let bc = 0.5 * (lo + hi);
    let bh = 0.5 * (hi - lo);
    if (separates(H, vec3f(1.0, 0.0, 0.0), nd.mean, nd.radius, bc, bh)) { return false; }
    if (separates(H, vec3f(0.0, 1.0, 0.0), nd.mean, nd.radius, bc, bh)) { return false; }
    if (separates(H, vec3f(0.0, 0.0, 1.0), nd.mean, nd.radius, bc, bh)) { return false; }
    // along the hull, and across it toward the occluder
    let a = H.lc - H.c;
    let L = length(a);
    if (L > 1e-6) {
        let ua = a / L;
        if (separates(H, ua, nd.mean, nd.radius, bc, bh)) { return false; }
        let q = nd.mean - H.c;
        let p = q - dot(q, ua) * ua;
        let pl = length(p);
        if (pl > 1e-6 && separates(H, p / pl, nd.mean, nd.radius, bc, bh)) { return false; }
    }
    for (var k = 0u; k < 3u; k++) {
        if (separates(H, H.lax[k], nd.mean, nd.radius, bc, bh)) { return false; }
        if (separates(H, H.rax[k], nd.mean, nd.radius, bc, bh)) { return false; }
    }
    return true;
}

// true when no Gaussian of another part (the emitters' own part aside) can touch a shadow ray from under node i
fn hull_clear(i: u32) -> bool {
    let nd = nodes[i];
    let own = nd.part;
    let oxf = parts[own].xf;
    let leaf = i < U.gaussian_count;
    let c = xf_point(oxf, nd.mean);
    let r = select(nd.radius, 0.0, leaf);          // a leaf's rays all start at its centre
    let lo = node_box[2u * i].xyz;
    let hi = node_box[2u * i + 1u].xyz;
    let bc = xf_point(oxf, select(0.5 * (lo + hi), nd.mean, leaf));
    let rh = select(0.5 * (hi - lo), vec3f(0.0), leaf);
    let rax = mat3x3f(xf_dir(oxf, vec3f(1.0, 0.0, 0.0)), xf_dir(oxf, vec3f(0.0, 1.0, 0.0)), xf_dir(oxf, vec3f(0.0, 0.0, 1.0)));
    let lc = 0.5 * (HP.light_lo + HP.light_hi);
    let lh = 0.5 * (HP.light_hi - HP.light_lo);
    var visits = 0u;
    for (var p = 0u; p < U.part_count; p++) {
        if (p == own || p == HP.skip_part) { continue; }
        let xf = parts[p].xf;
        let H = Hull(xf_point_inv(xf, c), r, xf_point_inv(xf, bc),
                     mat3x3f(xf_dir_inv(xf, rax[0]), xf_dir_inv(xf, rax[1]), xf_dir_inv(xf, rax[2])), rh,
                     xf_point_inv(xf, lc),
                     mat3x3f(xf_dir_inv(xf, vec3f(1.0, 0.0, 0.0)), xf_dir_inv(xf, vec3f(0.0, 1.0, 0.0)),
                             xf_dir_inv(xf, vec3f(0.0, 0.0, 1.0))), lh);
        var stack: array<u32, STACK>;
        stack[0] = parts[p].root;
        var sp = 1u;
        while (sp > 0u) {
            sp -= 1u;
            let ni = stack[sp];
            visits += 1u;
            if (visits > (HP.budget & 0x7FFFFFFFu)) { return false; }
            if (!hull_touches(H, ni)) { continue; }
            if (ni < U.gaussian_count) { return false; }   // an occluder Gaussian may cross a shadow ray
            let on = nodes[ni];
            for (var j = on.child_lo; j < on.child_hi; j++) {
                if (sp >= STACK) { return false; }
                stack[sp] = kids[j];
                sp += 1u;
            }
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn shadow_cull(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= LV.count) { return; }
    let i = level_nodes[LV.offset + gid.x];
    let parent = node_parent[i];
    let keep = node_status[i] & ~1u;   // bit 1 is ema's (amortized walk_emit)
    if (parent != 0xFFFFFFFFu && (node_status[parent] & 1u) == HS_CLEAR) {
        node_status[i] = keep | HS_CLEAR;
        return;
    }
    if (LV.skip != 0u || (i < U.gaussian_count && (HP.budget & 0x80000000u) != 0u)) {   // walk untested
        node_status[i] = keep;
        return;
    }
    node_status[i] = keep | select(HS_WALK, HS_CLEAR, hull_clear(i));
}
