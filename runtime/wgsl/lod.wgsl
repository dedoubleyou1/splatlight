// Screen-space LOD for drawing (renderer ?lod=<px>, 2026-10-28). Prepend common.wgsl. The asset's tree (SPEC 2.2)
// is the LOD: every internal node is a moment-matched Gaussian of its subtree, drawn in place of its leaves when it is
// small on screen. Two passes per frame, after the lighting (and the display fill), before the splat renderer:
//   lod_colour  bottom-up, one dispatch per depth over that depth's nodes (lod_nodes; deepest first): a node's lit colour = the mass-weighted mean of its
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
    G: u32, N: u32, depth: u32, emit_part: u32,   // leaves, nodes, this dispatch's depth, the emitters' part (or none)
    threshold: f32, focal: f32, near: f32, gain: f32,   // node footprint (px) to open at, focal (px), near plane, bounce gain
    shutter: f32, blur_lod: f32, pad0: f32, pad1: f32,  // motion blur variant D: a moving part's cut coarsens to blur_lod x its smear (px);
                                                        // pad0 != 0: lit nodes (node_shadow)
    off: u32, n_int: u32, n_all: u32, pad2: u32,        // this depth in lod_nodes: [off, off + n_int) internal, then leaves to n_all
}

const LOD_NONE: u32 = 0u;
const LOD_OPEN: u32 = 1u;
const LOD_DRAW: u32 = 2u;
const LOD_CAND: u32 = 3u;

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
    if (n < LU.G) { return colour[n].rgb + LU.gain * lo_b[n * 15u + 14u].rgb; }
    return colour[2u * LU.G + (n - LU.G)].rgb;
}

@compute @workgroup_size(64)
fn lod_colour(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= LU.n_int) { return; }
    let n = lod_nodes[LU.off + gid.x];
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
    if (n < LU.G) { return LU.gain * lo_b[n * 15u + 14u].rgb; }
    return colour[2u * LU.G + (LU.N - LU.G) + (n - LU.G)].rgb;
}

fn is_part_root(n: u32) -> bool {
    let p = nodes[n].part;
    return p < 32u && parts[p].root == n;
}

@compute @workgroup_size(64)
fn lod_select(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= LU.n_all) { return; }
    let n = lod_nodes[LU.off + gid.x];
    let cand = is_part_root(n) || lodflag[n] == LOD_CAND;
    if (!cand) { lodflag[n] = LOD_NONE; return; }
    if (n < LU.G) { lodflag[n] = LOD_DRAW; return; }
    var open = nodes[n].part == LU.emit_part;
    if (!open) {
        let P = nodes[n].part;
        let h = vec4f(nodes[n].mean, 1.0);
        let w = vec3f(dot(parts[P].xf[0], h), dot(parts[P].xf[1], h), dot(parts[P].xf[2], h));
        let z = -(LU.view * vec4f(w, 1.0)).z;
        let R = nodes[n].radius;
        let size = (2.0 / 3.0) * LU.focal * R / max(z - R, LU.near);
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
    if (!open) { lodflag[n] = LOD_DRAW; return; }
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
