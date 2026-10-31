// Screen-space LOD for drawing (renderer ?lod=<px>, 2026-10-28). Prepend common.wgsl. The asset's tree (SPEC 2.2)
// is the LOD: every internal node is a moment-matched Gaussian of its subtree, drawn in place of its leaves when it is
// small on screen. Two passes per frame, after the lighting (and the display fill), before the splat renderer:
//   lod_colour  bottom-up, after lod_select, one dispatch per depth over that depth's nodes (lod_nodes; deepest first),
//               only drawn nodes and their subtrees (LOD_UNDER): a node's lit colour = the mass-weighted mean of its
//               children's (leaves: colour + gain x the bounce toward the eye, as project.wgsl adds them), written to
//               colour[2G + (n - G)]
//   lod_select  top-down, one dispatch per depth: part roots are candidates; a candidate opens when its on-screen size
//               exceeds LU.threshold (its children become candidates) and is drawn otherwise; leaves are always drawn
//               when they are candidates. Size = (2/3) f R / max(z - R, near): R the node's bounding radius (3 sigma
//               for a leaf), so it never grows from parent to child and every leaf is covered exactly once.
// project.wgsl draws nodes whose flag is LOD_DRAW. The emitters' part always opens down to its leaves.
// Lighting on the cut (?lodlight=2, stage 2): lod_select runs before the lighting, then lod_list collects the drawn
// leaves (plus the display fill's sources of drawn leaves, `need`) and the drawn nodes into lodlist, and lod_args turns
// the counts into indirect dispatch sizes, so the eye shading runs over the cut only (emissive.wgsl shade_emit_list,
// shade_emit_nodes). lodlist (u32): [0] leaf count, [1] node count, [4..6] leaf dispatch, [8..10] node dispatch,
// leaves from LODLIST_HDR, nodes from LODLIST_HDR + G (each entry the Gaussian/node index).

struct LodU {
    view: mat4x4f,                 // world -> view (RenderFrame.view)
    G: u32, N: u32, only_parts: u32, emit_part: u32,   // leaves, nodes, lod_select: re-select only these parts (bit per part; ~0 all), the emitters' part (or none)
    threshold: f32, focal: f32, near: f32, gain: f32,   // node footprint (px) to open at, focal (px), near plane, bounce gain
    shutter: f32, blur_lod: f32, pad0: f32, open_parts: u32,   // motion blur variant D: a moving part's cut coarsens to blur_lod x its smear (px);
                                                        // pad0 != 0: lit nodes (node_shadow); open_parts: parts (bit per part < 32)
                                                        // that always open to their leaves, like the emitters' (main.js: glass
                                                        // parts while the refraction exit map, E / EC, is on)
    off: u32, n_int: u32, n_all: u32, size_rule: u32,   // this depth in lod_nodes: [off, off + n_int) internal, then leaves to n_all;
                                                        // size_rule bit 0: 1 projected area (default), 0 bounding radius
                                                        // (?lodsize=radius); bit 1: lod_colour colours every node
}

const LOD_NONE: u32 = 0u;
const LOD_OPEN: u32 = 1u;
const LOD_DRAW: u32 = 2u;
const LOD_CAND: u32 = 3u;
const LOD_UNDER: u32 = 4u;   // inside a drawn node's subtree (lod_colour aggregates only these and the drawn nodes)

@group(0) @binding(0) var<uniform> LU: LodU;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> kids: array<u32>;
@group(0) @binding(3) var<storage, read> parts: array<Part>;
@group(0) @binding(4) var<storage, read_write> colour: array<vec4f>;
@group(0) @binding(5) var<storage, read> lo_b: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> lodflag: array<u32>;
@group(0) @binding(7) var<storage, read_write> lodlist: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> need: array<u32>;
@group(0) @binding(11) var<storage, read> lod_nodes: array<u32>;   // nodes by depth (lod.js levels)
// lit nodes (?lodlight=1/2): lod_colour also leaves each node its leaves' mass-weighted mean shadow T, for every light
// the renderer has, so a node is shaded like a leaf with whatever shadow method produced the leaves' T
@group(0) @binding(9) var<storage, read_write> lamp_state: array<LampState>;   // emitters / unified lights, per key
@group(0) @binding(10) var<storage, read_write> env_T: array<vec4f>;          // distant lights; nodes at 2 G + k

const LODLIST_HDR: u32 = 16u;
const NO_KEY: u32 = 0xFFFFFFFFu;
const LIGHT_MIXED: u32 = 0xFFFFu;   // emissive.wgsl: unified lights' shared slot 3

fn node_mass(n: u32) -> f32 {
    let a = nodes[n].cinv_diag; let o = nodes[n].cinv_off;
    let det_inv = a.x * (a.y * a.z - o.z * o.z) - o.x * (o.x * a.z - o.z * o.y) + o.y * (o.x * o.z - a.y * o.y);
    return nodes[n].kappa / sqrt(max(det_inv, 1e-30));
}

fn shown_colour(n: u32) -> vec3f {
    if (n < LU.G) { return colour[n].rgb + LU.gain * lo_b[n * LOB + LOB_EYE].rgb; }
    return colour[2u * LU.G + (n - LU.G)].rgb;
}

@compute @workgroup_size(64)
fn lod_colour(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= LU.n_int) { return; }
    let n = lod_nodes[LU.off + gid.x];
    // only drawn nodes and their subtrees need a colour (2026-10-31: every node every frame was ~a fifth of hill's lit
    // frame); size_rule bit 1 colours every node (an explosion keeps the lit colours while the cut changes), and lit
    // nodes (pad0) need every node's shadow T
    let fl = lodflag[n];
    if (fl != LOD_DRAW && fl != LOD_UNDER && (LU.size_rule & 2u) == 0u && LU.pad0 == 0.0) { return; }
    var c = vec3f(0.0); var b = vec3f(0.0); var w = 0.0;
    for (var k = nodes[n].child_lo; k < nodes[n].child_hi; k++) {
        let ch = kids[k];
        let m = node_mass(ch);
        c += m * shown_colour(ch); b += m * mean_bounce(ch); w += m;
    }
    colour[2u * LU.G + (n - LU.G)] = vec4f(c / max(w, 1e-30), 1.0);
    // the mean bounce alone, for emissive.wgsl shade_emit_nodes (?lodlight=1: direct light on the node itself)
    colour[2u * LU.G + (LU.N - LU.G) + (n - LU.G)] = vec4f(b / max(w, 1e-30), 0.0);
    if (LU.pad0 != 0.0) { node_shadow(n); }
}

// a child's T toward light key `id` as emissive.wgsl reads it (its slot, else the shared mixed slot 3, else 1)
fn child_T(ch: u32, id: u32) -> f32 {
    let st = lamp_state[ch].slots;
    for (var s = 0u; s < 4u; s++) {
        if ((st[s].y & 0xFFFFu) == id && (st[s].y >> 16u) > 0u) { return bitcast<f32>(st[s].x); }
    }
    if ((st[3].y >> 16u) > 0u && (st[3].y & 0xFFFFu) == LIGHT_MIXED) { return bitcast<f32>(st[3].x); }
    return 1.0;
}

fn env_T_of(ch: u32) -> vec4f {
    if (ch < LU.G) { return env_T[ch]; }
    return env_T[2u * LU.G + (ch - LU.G)];
}

// node n's slots: the first four light keys its children carry, each the mass-weighted mean of the children's T
fn node_shadow(n: u32) {
    var keys = array<u32, 4>(NO_KEY, NO_KEY, NO_KEY, NO_KEY);
    var nk = 0u;
    for (var k = nodes[n].child_lo; k < nodes[n].child_hi; k++) {
        let st = lamp_state[kids[k]].slots;
        for (var s = 0u; s < 4u; s++) {
            let id = st[s].y & 0xFFFFu;
            if ((st[s].y >> 16u) == 0u || nk >= 4u) { continue; }
            var seen = false;
            for (var j = 0u; j < nk; j++) { seen = seen || keys[j] == id; }
            if (!seen) { keys[nk] = id; nk++; }
        }
    }
    var T = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var E = vec4f(0.0);
    var w = 0.0;
    for (var k = nodes[n].child_lo; k < nodes[n].child_hi; k++) {
        let ch = kids[k];
        let m = node_mass(ch);
        for (var j = 0u; j < nk; j++) { T[j] += m * child_T(ch, keys[j]); }
        E += m * env_T_of(ch);
        w += m;
    }
    var out: LampState;
    for (var j = 0u; j < 4u; j++) {
        out.slots[j] = select(vec2u(0u, 0u), vec2u(bitcast<u32>(T[j] / max(w, 1e-30)), keys[j] | (1u << 16u)), j < nk);
    }
    lamp_state[n] = out;
    if (arrayLength(&env_T) >= 2u * LU.G + (LU.N - LU.G)) { env_T[2u * LU.G + (n - LU.G)] = E / max(w, 1e-30); }
}

fn mean_bounce(n: u32) -> vec3f {
    if (n < LU.G) { return LU.gain * lo_b[n * LOB + LOB_EYE].rgb; }
    return colour[2u * LU.G + (LU.N - LU.G) + (n - LU.G)].rgb;
}

// node n's covariance projected to the screen (px^2), at its world mean w (part P)
fn screen_cov(n: u32, P: u32, w: vec3f) -> mat2x2f {
    let a = nodes[n].cinv_diag; let o = nodes[n].cinv_off;
    let Ci = mat3x3f(vec3f(a.x, o.x, o.y), vec3f(o.x, a.y, o.z), vec3f(o.y, o.z, a.z));
    let r0 = cross(Ci[1], Ci[2]); let r1 = cross(Ci[2], Ci[0]); let r2 = cross(Ci[0], Ci[1]);
    let C = transpose(mat3x3f(r0, r1, r2)) * (1.0 / dot(Ci[0], r0));
    let Rp = transpose(mat3x3f(parts[P].xf[0].xyz, parts[P].xf[1].xyz, parts[P].xf[2].xyz));
    let V = mat3x3f(LU.view[0].xyz, LU.view[1].xyz, LU.view[2].xyz);
    let M = V * Rp;
    let Cv = M * C * transpose(M);
    let pv = (LU.view * vec4f(w, 1.0)).xyz;
    let z = -pv.z;
    let f = LU.focal;
    // d(f x / -z) / d(x, y, z_view): rows of the 2x3 Jacobian, as columns of a 3x2
    let j0 = vec3f(f / z, 0.0, f * pv.x / (z * z));
    let j1 = vec3f(0.0, f / z, f * pv.y / (z * z));
    let c0 = Cv * j0; let c1 = Cv * j1;
    return mat2x2f(vec2f(dot(j0, c0), dot(j1, c0)), vec2f(dot(j0, c1), dot(j1, c1)));
}

fn is_part_root(n: u32) -> bool {
    let p = nodes[n].part;
    return p < 32u && parts[p].root == n;
}

@compute @workgroup_size(64)
fn lod_select(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= LU.n_all) { return; }
    let n = lod_nodes[LU.off + gid.x];
    // the camera and settings held still: only the moved parts' subtrees change their cut (a part's tree is its own,
    // so the others keep last frame's flags; 2026-10-31: drifting clouds re-selected the whole hill every frame)
    if (LU.only_parts != 0xFFFFFFFFu) {
        let p = nodes[n].part;
        if (p >= 32u || ((LU.only_parts >> p) & 1u) == 0u) { return; }
    }
    let f = lodflag[n];
    let cand = is_part_root(n) || f == LOD_CAND;
    if (!cand) {
        // every internal node writes its children's flags (CAND, UNDER or NONE), so a flag left from an earlier frame
        // is always overwritten before its node is read; UNDER marks the subtrees lod_colour must aggregate
        let under = f == LOD_UNDER;
        if (!under) { lodflag[n] = LOD_NONE; }
        let kf = select(LOD_NONE, LOD_UNDER, under);
        if (n >= LU.G) { for (var k = nodes[n].child_lo; k < nodes[n].child_hi; k++) { lodflag[kids[k]] = kf; } }
        return;
    }
    if (n < LU.G) { lodflag[n] = LOD_DRAW; return; }
    var open = nodes[n].part == LU.emit_part || (nodes[n].part < 32u && ((LU.open_parts >> nodes[n].part) & 1u) != 0u);
    if (!open) {
        let P = nodes[n].part;
        let h = vec4f(nodes[n].mean, 1.0);
        let w = vec3f(dot(parts[P].xf[0], h), dot(parts[P].xf[1], h), dot(parts[P].xf[2], h));
        let z = -(LU.view * vec4f(w, 1.0)).z;
        let R = nodes[n].radius;
        var size = (2.0 / 3.0) * LU.focal * R / max(z - R, LU.near);
        if ((LU.size_rule & 1u) == 1u && z - R > LU.near) {
            // projected area (default): the radius measure scaled by sqrt(short / long) of the node's on-screen ellipse,
            // i.e. its footprint's geometric mean relative to its longest axis: round or face-on nodes keep the radius
            // rule (and its bounding-radius margin), long thin ones (grass blades) collapse once thin on screen
            let S = screen_cov(n, P, w);
            let tr = S[0].x + S[1].y;
            let dt = max(determinant(S), 0.0);
            let disc = sqrt(max(0.25 * (S[0].x - S[1].y) * (S[0].x - S[1].y) + S[0].y * S[0].y, 0.0));
            let l1 = max(0.5 * tr + disc, 1e-12);          // long axis variance; short = det / long (float32-safe)
            size *= sqrt(sqrt(max(dt / l1, 0.0) / l1));
        }
        var thr = LU.threshold;
        if (LU.blur_lod > 0.0 && LU.shutter > 0.0 && (parts[P].flags & 2u) != 0u) {
            // variant D (motion blur thread): the box shutter smears a moving part over its path this exposure, so
            // detail under a fraction of the smear is not seen; open only nodes larger than blur_lod x the smear
            let wp = vec3f(dot(parts[P].xf_prev[0], h), dot(parts[P].xf_prev[1], h), dot(parts[P].xf_prev[2], h));
            let smear = LU.focal * LU.shutter * length(w - wp) / max(z - R, LU.near);
            thr = max(thr, LU.blur_lod * smear);
        }
        open = size > thr;
    }
    if (!open) {
        lodflag[n] = LOD_DRAW;
        for (var k = nodes[n].child_lo; k < nodes[n].child_hi; k++) { lodflag[kids[k]] = LOD_UNDER; }
        return;
    }
    lodflag[n] = LOD_OPEN;
    for (var k = nodes[n].child_lo; k < nodes[n].child_hi; k++) { lodflag[kids[k]] = LOD_CAND; }
}

// ?lodlight=2: after lod_select (and lighting.js fill_need), one thread per tree node; lodlist[0..1] cleared before
@compute @workgroup_size(64)
fn lod_list(@builtin(global_invocation_id) gid: vec3u) {
    let n = gid.x;
    if (n >= LU.N) { return; }
    let drawn = lodflag[n] == LOD_DRAW;
    if (n < LU.G) {
        let want = drawn || need[n] != 0u;
        need[n] = 0u;
        if (want) { atomicStore(&lodlist[LODLIST_HDR + atomicAdd(&lodlist[0], 1u)], n); }
    } else if (drawn) {
        atomicStore(&lodlist[LODLIST_HDR + LU.G + atomicAdd(&lodlist[1], 1u)], n);
    }
}

@compute @workgroup_size(1)
fn lod_args() {
    let l = atomicLoad(&lodlist[0]); let k = atomicLoad(&lodlist[1]);
    atomicStore(&lodlist[4], (l + 63u) / 64u); atomicStore(&lodlist[5], 1u); atomicStore(&lodlist[6], 1u);
    atomicStore(&lodlist[8], (k + 63u) / 64u); atomicStore(&lodlist[9], 1u); atomicStore(&lodlist[10], 1u);
}
