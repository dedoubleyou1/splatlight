// Probe-loop bounce (the settled M36 §3 design, experiments/M36_surface_lighting/loop_surf.py), one gather round per
// frame with temporal blending, so the bounce converges over frames instead of rounds.
//
// Prepend common.wgsl and material_ext.wgsl. Bindings 0-7 mirror shade.wgsl's, so the host can bind the same buffers. Per frame, after
// walk_lamp, ema and shade:
//   near_vis      per Gaussian: other-part transmittance over the first near_r metres per world cell (only where a
//                 part moved, or when force_near is set)
//   near_cones    (prototype, instead of near_vis when PP.refl.z is 6 or 8) the same output from a few wide cones
//   direct_dirs   per Gaussian: direct light toward the 14 fixed directions (shade's transfer with w_o = D_k)
//   bounce_dirs   per Gaussian (1 / subset of them per frame): probe light -> incoming per cell -> outgoing toward
//                 the 14 directions and the eye
//   node_leaf     per leaf: view-weighted emitter sums (weight A(D_k) V(D_k), M36 parents' b3 rule)
//   node_up       per tree level, deepest first: parents sum their children
//   node_sh       per node: 14-direction mean -> SH2
//   probe_gather  per (probe, cell), 1 / subset of the rays per frame: composite the theta-4 shared 2x2x2 cut front
//                 to back, blend into history
//   probe_valid   per probe: drop probes inside a moving part or a wall slab
// Cells everywhere use the V grid layout (SPEC 2.3): 8 rows of elevation from the zenith x 16 columns of azimuth.

const NCELL: u32 = 128u;
const HIT_MAX: u32 = 16u;   // probe_gather's hit list; a full list folds the rest into its last entry (2026-10-30)

struct ProbeParams {
    lo: vec3f, n_axis: u32,              // probe lattice origin (world) and probes per axis
    step: vec3f, node_count: u32,        // lattice spacing
    block_rad: f32, theta: f32, tmax: f32, near_r: f32,
    alpha: f32, force_near: u32, vfloor: f32, g_valid: f32,   // history blend weight of this frame; free-space test
    subset: u32, phase: u32, part_probes: u32, sh_full: u32,     // 1 / subset per frame; part probes after the lattice;
                                         // sh_full = 1: bounce_sh refreshes every Gaussian every frame
    dirs: array<vec4f, 14>,              // D14: 6 axes + 8 diagonals (world)
    pinv: array<vec4f, 32>,              // 9 x 14 least-squares SH2 fit of D14 (row-major floats)
    refl: vec4u,                         // x = 1: near-field reflection (near_vis records blockers; near_refl runs);
                                         // y = first Gaussian of this frame's near_vis chunk (0 unless chunked);
                                         // z = near-field cone count for near_cones (6 or 8; 0 = the walk)
                                         // w = emitter part + 1: node_leaf leaves that part's Gaussians out of the probes' light
                                         // (M47: an emitter's reflected light is mostly its neighbours' emission)
    // SPEC 2.7 probe layout (2026-10-27): the lattice is dims.x x dims.y columns at lo.xy + step.xy (i, j), each with
    // dims.z probes at heights ground[column] + levels[k]. The room lattice: ground 0, levels lo.z + k step.z. The
    // terrain layout (dims.w = 1): ground = the terrain's height under the column, levels = heights above it, so the
    // probes hug the ground and thin out upward (open scenes).
    dims: vec4u,                         // x, y, z counts; w = 1 terrain layout
    levels: array<vec4f, 4>,             // up to 16 heights (m), ascending
    ground: array<vec4f, 256>,           // up to 1024 column ground heights (m), index i * dims.y + j
    // Probe volumes on static objects (SPEC 2.7, e.g. a tree): vols[q] = (first probe after the lattice, nx, ny, nz),
    // nx = 0: none. A grid over the part's rest-frame box (part_probe_frame, widened). Like a movable part's 8 probes
    // it leaves its own object out of its rays (the object's self-shadowing is in its baked V; counting the object here
    // too darkened the crown twice and made probe-sized blotches, hill_lite 2026-10-27), is always valid and is used
    // only by the object's own Gaussians, trilinear without Chebyshev. Movable parts keep their 8 probes at 8 q
    // (vol_info.x = 8 x parts of them, before the volumes).
    vols: array<vec4u, 32>,
    vol_info: vec4u,                     // x: part probes before the volumes; y = 1: every gather keeps the block margin (?probemargin=old)
    // node probes (renderer ?nodeprobes=1, NODE_PROBES, 2026-10-28): per movable part PART_PROBES_PER probes at its
    // tree patches, rest frame: xyz = just outside the patch along its facing direction, w = blend radius (m)
    nodep: array<vec4f, 256>,
}

fn vol_dims(q: u32) -> vec3u { return select(vec3u(0u), PP.vols[q].yzw, q < 32u); }
fn has_vol(q: u32) -> bool { return q < 32u && PP.vols[q].y > 0u; }

// the static volume a probe index belongs to (0xFFFFFFFF: none)
fn vol_of_probe(r: u32) -> u32 {
    for (var q = 0u; q < min(U.part_count, 32u); q++) {
        let v = PP.vols[q];
        if (v.y > 0u && r >= v.x && r < v.x + v.y * v.z * v.w) { return q; }
    }
    return 0xFFFFFFFFu;
}

// a static volume's box in its part's rest frame: the part probe frame, widened to about the part's extent
fn vol_frame(q: u32) -> PartProbeFrame {
    let pf = part_probe_frame_raw(q);
    return PartProbeFrame(pf.c, min(1.6 * pf.h, vec3f(parts[q].bound.w)));
}

fn lat_count() -> u32 { return PP.dims.x * PP.dims.y * PP.dims.z; }
fn lat_index(i: u32, j: u32, k: u32) -> u32 { return (i * PP.dims.y + j) * PP.dims.z + k; }
fn lat_level(k: u32) -> f32 { return PP.levels[k / 4u][k % 4u]; }
fn lat_ground(i: u32, j: u32) -> f32 { let q = i * PP.dims.y + j; return PP.ground[q / 4u][q % 4u]; }

struct Level { depth: u32, r0: u32, r1: u32, r2: u32 }

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> vgrid: array<VGrid>;
@group(0) @binding(4) var<storage, read> parts: array<Part>;
@group(0) @binding(5) var<storage, read> lamps: array<Lamp>;
@group(0) @binding(6) var<storage, read_write> lamp_state: array<LampState>;   // read_write: sc_shadow
@group(0) @binding(7) var<storage, read> materials: array<Material>;
@group(0) @binding(9) var<storage, read> kids: array<u32>;
@group(0) @binding(10) var<uniform> PP: ProbeParams;
@group(0) @binding(11) var<storage, read_write> probe_data: array<vec4f>;   // per ray: (rgb, first-hit part), (m1, m2, valid, near)
@group(0) @binding(12) var<storage, read_write> nearvis: array<VGrid>;      // per Gaussian, world cells
@group(0) @binding(13) var<storage, read_write> lo_dir: array<vec4f>;       // per Gaussian: direct toward the probes (common.wgsl LOB)
@group(0) @binding(14) var<storage, read_write> lo_b: array<vec4f>;         // per Gaussian x LOB: bounce toward the eye, the probes; lanes
// Tree nodes' light toward the probes, per TREE SLOT, not per node (2026-10-30, GPU-memory cut): sc_flag[n] >> 2 is
// node n's slot. With sender LOD (sc_select) only the nodes the gather can reach (OPEN, the cut's NODEs and LEAFs)
// get one, handed out when the cut is re-selected; the rest point at the null slot, the arrays' last entry, which
// stays zero (it is never written). Without sender LOD the host fills sc_flag with slot = node index.
@group(0) @binding(15) var<storage, read_write> node_acc: array<vec4f>;     // per slot x 14: (sum w L, sum w)
@group(0) @binding(16) var<storage, read_write> node_sh: array<u32>;        // per slot x 14 words: SH2 rgb as 27 halves
@group(0) @binding(80) var<storage, read_write> sc_flag: array<u32>;        // per node: sender-cut state (bits 0-1) | slot << 2

fn tree_slot(n: u32) -> u32 { return sc_flag[n] >> 2u; }
fn acc_null() -> u32 { return arrayLength(&node_acc) / 14u - 1u; }
fn sh_null() -> u32 { return arrayLength(&node_sh) / 14u - 1u; }

// a tree node's outgoing light as the probes see it (node_sh2's SH2 fit). Readers go through these two, so the
// storage behind them can change without touching them. Valid for nodes the gather can use; with sender LOD, nodes
// under the cut read zero.
fn node_sh_at(ni: u32, m: u32) -> vec3f {
    let s = tree_slot(ni);
    if (s >= sh_null()) { return vec3f(0.0); }
    let o = s * 14u;
    let a = 3u * m;
    let h0 = unpack2x16float(node_sh[o + a / 2u]);
    let h1 = unpack2x16float(node_sh[o + a / 2u + 1u]);
    return select(vec3f(h0.y, h1.x, h1.y), vec3f(h0.x, h0.y, h1.x), (a & 1u) == 0u);
}

// radiance node ni sends along the direction whose SH2 basis is b (sh2_basis of the travel direction reversed), unclamped
fn node_radiance(ni: u32, b: array<f32, 9>) -> vec3f {
    let s = tree_slot(ni);
    if (s >= sh_null()) { return vec3f(0.0); }
    var h: array<f32, 28>;
    for (var j = 0u; j < 14u; j++) { let p = unpack2x16float(node_sh[s * 14u + j]); h[2u * j] = p.x; h[2u * j + 1u] = p.y; }
    var v = vec3f(0.0);
    for (var m = 0u; m < 9u; m++) { v += b[m] * vec3f(h[3u * m], h[3u * m + 1u], h[3u * m + 2u]); }
    return v;
}
@group(0) @binding(17) var<uniform> LV: Level;
@group(0) @binding(29) var<storage, read> up_nodes: array<u32>;   // node_up: internal nodes by depth
// M39 stage 2 (U.iso == 3: direction-independent transfer with SH2 probes)
@group(0) @binding(25) var<storage, read_write> probe_sh: array<vec4f>;     // per probe x 9: SH2 rgb of its incoming light
@group(0) @binding(26) var<storage, read_write> xfer: array<vec4f>;         // per Gaussian x 3: t'_0..8 (world frame)
// near-field reflection (PP.refl.x == 1)
@group(0) @binding(8) var<storage, read> colour: array<vec4f>;             // per Gaussian: direct toward the eye (shade)
@group(0) @binding(27) var<storage, read_write> blk: array<vec2u>;         // per Gaussian x 16: (node, bitcast weight);
                                                                           // 0-7 own part (baked), 8-15 other parts
@group(0) @binding(28) var<storage, read_write> lo_b_prev: array<vec4f>;   // last frame's lo_b

// The environment (SPEC 2.9, environment.wgsl): probe rays that leave the scene see the sky (probe_gather).
struct EnvLight { dir_std: vec4f, irr: vec4f }
struct Env { sh: array<vec4f, 16>, lights: array<EnvLight, 4>, info: vec4u }
@group(0) @binding(40) var<uniform> ENV: Env;

// sky radiance toward unit w (world, z up): real SH bands 0-3, clamped at 0
fn sky_radiance(w: vec3f) -> vec3f {
    let x = w.x; let y = w.y; let z = w.z;
    let b = array<f32, 16>(0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y),
        0.590044 * y * (3.0 * x * x - y * y), 2.890611 * x * y * z, 0.457046 * y * (5.0 * z * z - 1.0),
        0.373176 * z * (5.0 * z * z - 3.0), 0.457046 * x * (5.0 * z * z - 1.0), 1.445306 * z * (x * x - y * y),
        0.590044 * x * (x * x - 3.0 * y * y));
    var c = vec3f(0.0);
    for (var k = 0u; k < 16u; k++) { c += b[k] * ENV.sh[k].xyz; }
    return max(c, vec3f(0.0));
}

// ---- small helpers -------------------------------------------------------------------------------------------

// shade.wgsl's EMA lookup (the other-part transmittance toward lamp l), 1 when the lamp has no slot
fn lamp_transmittance(g: u32, l: u32) -> f32 {
    for (var s = 0u; s < 4u; s++) {
        let st = lamp_state[g].slots[s];
        if ((st.y & 0xFFFFu) == l && (st.y >> 16u) > 0u) { return bitcast<f32>(st.x); }
    }
    return 1.0;
}

fn cell_dir(c: u32) -> vec3f {
    let i = c / 16u; let j = c % 16u;
    let el = 0.5 * PI - (f32(i) + 0.5) * PI / 8.0;
    let az = -PI + (f32(j) + 0.5) * 2.0 * PI / 16.0;
    return vec3f(cos(el) * cos(az), cos(el) * sin(az), sin(el));
}

// P3 ASG: a cell's angular covariance, as a Gaussian CELL_STD of its width across in elevation and azimuth
// (1 / sqrt(12) matches a box's variance; a little wider keeps neighbouring cells from showing as separate dots)
const CELL_STD: f32 = 0.4;
fn cell_asg_phase(c: u32, xf: array<vec4f, 3>, u: vec3f, fl: Flake, mat: Material, wo: vec3f) -> f32 {
    let i = c / 16u; let j = c % 16u;
    let el = 0.5 * PI - (f32(i) + 0.5) * PI / 8.0;
    let az = -PI + (f32(j) + 0.5) * 2.0 * PI / 16.0;
    let e_el = vec3f(-sin(el) * cos(az), -sin(el) * sin(az), cos(el));
    let e_az = vec3f(-sin(az), cos(az), 0.0);
    let s_el = CELL_STD * PI / 8.0;
    let s_az = CELL_STD * 2.0 * PI / 16.0 * max(cos(el), 0.05);
    let lb = array<vec3f, 2>(xf_dir_inv(xf, e_el), xf_dir_inv(xf, e_az));
    return asg_phase(u, lb, vec3f(s_el * s_el, 0.0, s_az * s_az), fl.n, fl.t1, fl.t2, mat.flake, fl.Sd, fl.So, wo);
}

fn cell_omega(c: u32) -> f32 {
    let i = f32(c / 16u);
    let top = 0.5 * PI - i * PI / 8.0;
    return (2.0 * PI / 16.0) * (sin(top) - sin(top - PI / 8.0));
}

fn cell_of(w: vec3f) -> u32 {
    let el = asin(clamp(w.z, -1.0, 1.0));
    let i = min(u32(max((0.5 * PI - el) / (PI / 8.0), 0.0)), 7u);
    let j = u32(floor((atan2(w.y, w.x) + PI) / (2.0 * PI) * 16.0)) % 16u;
    return i * 16u + j;
}

fn sh2_basis(d: vec3f) -> array<f32, 9> {
    let x = d.x; let y = d.y; let z = d.z;
    return array<f32, 9>(0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z,
                         0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y));
}

fn pinv_at(m: u32, k: u32) -> f32 {
    let f = m * 14u + k;
    return PP.pinv[f / 4u][f % 4u];
}

fn probe_pos(p: u32) -> vec3f {
    let nl = lat_count();
    if (p >= nl + PP.vol_info.x) {      // static volume probe: a grid over the object's box
        let r = p - nl;
        let q = vol_of_probe(r);
        if (q == 0xFFFFFFFFu) { return vec3f(0.0); }
        let d = vol_dims(q);
        let l = r - PP.vols[q].x;
        let ijk = vec3f(f32(l / (d.y * d.z)), f32((l / d.z) % d.y), f32(l % d.z));
        let vf = vol_frame(q);
        return xf_point(parts[q].xf, vf.c - vf.h + 2.0 * vf.h * ijk / max(vec3f(d) - vec3f(1.0), vec3f(1.0)));
    }
    if (p >= nl) {                      // part probe: 2x2x2 in the part's rest frame, moving with it
        let q = (p - nl) / PART_PROBES_PER; let k = (p - nl) % PART_PROBES_PER;
        if (NODE_PROBES) { return xf_point(parts[q].xf, PP.nodep[q * PART_PROBES_PER + k].xyz); }
        let sgn = vec3f(f32(k >> 2u), f32((k >> 1u) & 1u), f32(k & 1u)) * 2.0 - 1.0;
        let pf = part_probe_frame(q);
        return xf_point(parts[q].xf, pf.c + sgn * pf.h);
    }
    let i = p / (PP.dims.y * PP.dims.z);
    let j = (p / PP.dims.z) % PP.dims.y;
    let k = p % PP.dims.z;
    return vec3f(PP.lo.x + PP.step.x * f32(i), PP.lo.y + PP.step.y * f32(j), lat_ground(i, j) + lat_level(k));
}

// Part probes (experiments/P1_probe_dark, David's idea): 8 probes per movable part at its rest-frame centroid
// +- 0.95 sigma per axis (the part root's covariance; ~a quarter of the extent for a box), clamped to half the bound.
// They leave their own part out of their rays: the part's self-shadowing is already in each Gaussian's V, so its
// Gaussians only need the rest of the world, and the probes never end up inside the part.
struct PartProbeFrame { c: vec3f, h: vec3f }
// Node probes (David, 2026-10-28; lighting.js nodeProbes): instead of the 2x2x2 cube inside the part, one probe per patch
// of the part's own split into PART_PROBES_PER groups, pushed out of the patch along its facing direction (PP.nodep),
// so a probe sees what its patch sees; Gaussians blend their part's nearest probes by distance. lighting.js patches
// these two constants (16, true) when ?nodeprobes=1.
const PART_PROBES_PER: u32 = 8u;
const NODE_PROBES: bool = false;   // node probes: archived 2026-10-29 (the renderer no longer switches them on)

fn part_probe_frame(q: u32) -> PartProbeFrame { return part_probe_frame_raw(q); }

fn part_probe_frame_raw(q: u32) -> PartProbeFrame {
    let nd = nodes[parts[q].root];
    let a = nd.cinv_diag; let o = nd.cinv_off;     // inverse covariance (xx, yy, zz), (xy, xz, yz)
    let det = a.x * (a.y * a.z - o.z * o.z) - o.x * (o.x * a.z - o.z * o.y) + o.y * (o.x * o.z - a.y * o.y);
    let var_ = vec3f(a.y * a.z - o.z * o.z, a.x * a.z - o.y * o.y, a.x * a.y - o.x * o.x) / max(det, 1e-30);
    return PartProbeFrame(nd.mean, min(0.95 * sqrt(max(var_, vec3f(0.0))), vec3f(0.5 * parts[q].bound.w)));
}

fn is_part_probe_part(q: u32) -> bool {
    return PP.part_probes > 0u && q < PP.vol_info.x / PART_PROBES_PER && (parts[q].flags & 1u) == 0u;
}

// near-field visibility of Gaussian g toward cell c; 1 while near field has never been on (lighting.js allocates nearvis
// on first use and binds a one-entry placeholder until then)
fn near_byte(g: u32, c: u32) -> f32 {
    if (g >= arrayLength(&nearvis)) { return 1.0; }
    return grid_byte(nearvis[g], c);
}

fn grid_byte(v: VGrid, c: u32) -> f32 {
    return f32((v.w[c >> 2u] >> ((c & 3u) * 8u)) & 0xFFu) * (1.0 / 255.0);
}

// Flake frame of Gaussian G in its rest frame (as shade does).
struct Flake { n: vec3f, Sd: vec3f, So: vec3f, detC: f32, t1: vec3f, t2: vec3f }   // t1, t2: Rq[1], Rq[2]

fn flake_of(G: Gaussian, mat: Material) -> Flake {
    let Rq = quat_matrix(G.quat);
    let Sloc = Rq * mat3x3f(vec3f(mat.flake.x, 0.0, 0.0), vec3f(0.0, mat.flake.y, 0.0), vec3f(0.0, 0.0, mat.flake.z)) * transpose(Rq);
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    return Flake(Rq[0], vec3f(Sloc[0].x, Sloc[1].y, Sloc[2].z), vec3f(Sloc[1].x, Sloc[2].x, Sloc[2].y), s3 * s3, Rq[1], Rq[2]);
}

fn material_of(G: Gaussian) -> Material {
    return materials[min(G.flags & 0xFFFFu, max(U.material_count, 1u) - 1u)];
}

// ---- near-field visibility -----------------------------------------------------------------------------------

// optical depth along P + t w, t in [0, dist], through one part's subtree (rest-frame P, w): walk_lamp's walk_part
fn near_walk(root: u32, P: vec3f, w: vec3f, dist: f32, theta: f32) -> f32 {
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
        var use_node = ni < U.gaussian_count;
        if (!use_node) {
            let r = P - nd.mean;
            use_node = dot(r, r) > th2 * nd.smax * nd.smax;
        }
        if (use_node) {
            tau += nd.kappa * dirm(nd, w) * seg_geom(nd, P, w, 0.0, dist);
        } else {
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                if (sp < STACK) { stack[sp] = kids[j]; sp += 1u; }
            }
        }
    }
    return tau;
}

// ---- near-field reflection -----------------------------------------------------------------------------------
// The light near-field visibility blocks is not dropped but replaced by the blockers' own colour: per Gaussian a
// short list (blk) of blockers with weights w_i = sum_c A(u_c) V(u_c) Omega_c share_i,c vb_i, where per cell the
// blockers' shares sum to 1 - T (front-to-back compositing, rescaled) and vb_i = V_i(-w) is how exposed blocker i
// is toward the receiver (parents: 1). Own-part blockers are baked (runtime/bake_blockers.py, same rule with the V
// grid's 1 - V); other parts' come from the near_vis walk below. Hits within KEX thicknesses along the ray are the
// receiver's own layer: they occlude but do not reflect.

const NHIT: u32 = 16u;     // hits kept per near-field ray, nearest first
const NCAND: u32 = 16u;    // candidate blockers per Gaussian before the top 8
const KEX: f32 = 3.0;      // own-layer exclusion (bake_blockers.py KEX)
const NO_BLOCKER: u32 = 0xFFFFFFFFu;

struct NearHits { t: array<f32, NHIT>, tau: array<f32, NHIT>, node: array<u32, NHIT>, vb: array<f32, NHIT>,
                  k: u32, tau_all: f32, tau_ex: f32 }

// near_walk that also keeps the depth-sorted hits (rest-frame P, w of the blocking part; sg = the receiver's
// thickness along w). tau_all is near_walk's optical depth.
fn near_walk_hits(root: u32, P: vec3f, w: vec3f, dist: f32, theta: f32, sg: f32, h: ptr<function, NearHits>) {
    var stack: array<u32, STACK>;
    stack[0] = root;
    var sp = 1u;
    let th2 = theta * theta;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        let d = nd.mean - P;
        let tt = clamp(dot(d, w), 0.0, dist);
        let e = d - tt * w;
        if (dot(e, e) > nd.radius * nd.radius) { continue; }
        var use_node = ni < U.gaussian_count;
        if (!use_node) {
            let r = P - nd.mean;
            use_node = dot(r, r) > th2 * nd.smax * nd.smax;
        }
        if (use_node) {
            let tau = nd.kappa * dirm(nd, w) * seg_geom(nd, P, w, 0.0, dist);
            (*h).tau_all += tau;
            if (tau <= 1e-5) { continue; }
            let cw = sym_mul(nd.cinv_diag, nd.cinv_off, w);
            let a = dot(cw, w);
            let tp = dot(cw, d) / a;
            if (tp < KEX * (sg + inverseSqrt(a))) { (*h).tau_ex += tau; continue; }
            if ((*h).k == NHIT && tp >= (*h).t[NHIT - 1u]) { continue; }
            var vb = 1.0;
            if (ni < U.gaussian_count) { vb = vgrid_lookup(vgrid[ni], -w); }
            var at = min((*h).k, NHIT - 1u);
            while (at > 0u && (*h).t[at - 1u] > tp) {
                (*h).t[at] = (*h).t[at - 1u]; (*h).tau[at] = (*h).tau[at - 1u];
                (*h).node[at] = (*h).node[at - 1u]; (*h).vb[at] = (*h).vb[at - 1u];
                at -= 1u;
            }
            (*h).t[at] = tp; (*h).tau[at] = tau; (*h).node[at] = ni; (*h).vb[at] = vb;
            (*h).k = min((*h).k + 1u, NHIT);
        } else {
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                if (sp < STACK) { stack[sp] = kids[j]; sp += 1u; }
            }
        }
    }
}

// near_vis with PP.refl.x set: the same transmittance into nearvis, plus the other-part blockers into blk[8..16).
fn near_vis_refl(g: u32, G: Gaussian, own: u32, P: vec3f) {
    let nd = nodes[g];
    let xfo = parts[own].xf;
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    var out: VGrid;
    for (var k = 0u; k < 32u; k++) { out.w[k] = 0u; }
    var cid: array<u32, NCAND>;
    var cwt: array<f32, NCAND>;
    for (var q = 0u; q < NCAND; q++) { cid[q] = NO_BLOCKER; cwt[q] = 0.0; }
    var wtot = 0.0;
    for (var c = 0u; c < NCELL; c++) {
        let w = cell_dir(c);
        let u = xf_dir_inv(xfo, w);
        var h: NearHits;
        h.k = 0u; h.tau_all = 0.0; h.tau_ex = 0.0;
        let sg = inverseSqrt(dot(u, sym_mul(nd.cinv_diag, nd.cinv_off, u)));
        for (var p = 0u; p < U.part_count; p++) {
            if (p == own) { continue; }
            let xf = parts[p].xf;
            let b = parts[p].bound;
            if (length(xf_point_inv(xf, P) - b.xyz) > b.w + PP.near_r) { continue; }
            near_walk_hits(parts[p].root, xf_point_inv(xf, P), xf_dir_inv(xf, w), PP.near_r, 8.0, sg, &h);
        }
        let T = clamp(exp(-h.tau_all), 0.0, 1.0);
        out.w[c >> 2u] |= u32(round(T * 255.0)) << ((c & 3u) * 8u);
        if (h.k == 0u || T >= 1.0) { continue; }
        let a = footprint(nd, s3 * s3, u) * vgrid_lookup(vgrid[g], u) * cell_omega(c);
        if (a <= 0.0) { continue; }
        // own-layer hits first, then the kept ones front to back; rescaled so T + sum of shares = 1
        var Tr = exp(-h.tau_ex);
        var ssum = 1.0 - Tr;
        for (var i = 0u; i < h.k; i++) { ssum += Tr * (1.0 - exp(-h.tau[i])); Tr *= exp(-h.tau[i]); }
        if (ssum <= 1e-9) { continue; }
        let sc = (1.0 - T) / ssum;
        Tr = exp(-h.tau_ex);
        for (var i = 0u; i < h.k; i++) {
            let wb = a * Tr * (1.0 - exp(-h.tau[i])) * sc * h.vb[i];
            Tr *= exp(-h.tau[i]);
            if (wb <= 0.0) { continue; }
            wtot += wb;
            // accumulate, else replace the smallest candidate if this one is larger
            var slot = NCAND;
            var mn = 0u;
            for (var q = 0u; q < NCAND; q++) {
                if (cid[q] == h.node[i]) { slot = q; break; }
                if (cwt[q] < cwt[mn]) { mn = q; }
            }
            if (slot < NCAND) { cwt[slot] += wb; } else if (cwt[mn] < wb) { cid[mn] = h.node[i]; cwt[mn] = wb; }
        }
    }
    nearvis[g] = out;
    // the 8 largest, rescaled to the total
    var bi: array<u32, 8>;
    var bw: array<f32, 8>;
    var kept = 0.0;
    for (var r = 0u; r < 8u; r++) {
        var best = NCAND;
        for (var q = 0u; q < NCAND; q++) {
            if (cid[q] != NO_BLOCKER && cwt[q] > 0.0 && (best == NCAND || cwt[q] > cwt[best])) { best = q; }
        }
        bi[r] = NO_BLOCKER; bw[r] = 0.0;
        if (best == NCAND) { continue; }
        bi[r] = cid[best]; bw[r] = cwt[best]; kept += cwt[best];
        cwt[best] = 0.0;
    }
    let s = select(0.0, wtot / kept, kept > 0.0);
    for (var r = 0u; r < 8u; r++) { blk[g * 16u + 8u + r] = vec2u(bi[r], bitcast<u32>(bw[r] * s)); }
}

@compute @workgroup_size(64)
fn near_vis(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x + PP.refl.y;
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    // Recompute only where something moved relative to this Gaussian.
    var stale = PP.force_near != 0u || (parts[own].flags & 2u) != 0u;
    for (var p = 0u; p < U.part_count && !stale; p++) {
        if (p == own || (parts[p].flags & 2u) == 0u) { continue; }
        let b = parts[p].bound;
        let r = b.w + PP.near_r;
        stale = length(P - xf_point(parts[p].xf, b.xyz)) < r || length(P - xf_point(parts[p].xf_prev, b.xyz)) < r;
    }
    if (!stale) { return; }
    if (PP.refl.x != 0u) { near_vis_refl(g, G, own, P); return; }
    var out: VGrid;
    for (var k = 0u; k < 32u; k++) { out.w[k] = 0u; }
    for (var c = 0u; c < NCELL; c++) {
        let w = cell_dir(c);
        var tau = 0.0;
        for (var p = 0u; p < U.part_count; p++) {
            if (p == own) { continue; }
            let xf = parts[p].xf;
            let b = parts[p].bound;
            // skip parts whose bound is out of reach
            if (length(xf_point_inv(xf, P) - b.xyz) > b.w + PP.near_r) { continue; }
            tau += near_walk(parts[p].root, xf_point_inv(xf, P), xf_dir_inv(xf, w), PP.near_r, 8.0);
        }
        let q = u32(round(clamp(exp(-tau), 0.0, 1.0) * 255.0));
        out.w[c >> 2u] |= q << ((c & 3u) * 8u);
    }
    nearvis[g] = out;
}

// ---- near-field cones (prototype, PP.refl.z = cone count 6 or 8; 0 = the near_vis walk) ------------------------
// A cheaper near_vis: per Gaussian K wide cones with fixed world axes (6: the axes, 8: the cube diagonals; half-angle
// cos a = 1 - 2 / K, so together they hold the sphere's solid angle) through the other parts' trees out to near_r.
// A node is used once it is smaller than CONE_LAMBDA x the cone's radius at its depth, so the cut is finer near the
// origin. A used node covers a share c of the cone's cross-section (its projected Gaussian against the cone's disc
// taken as a Gaussian of variance r^2 / 2 per axis) and is opaque by 1 - exp(-K) through its centre:
// alpha = c (1 - exp(-K)). The hits composite front to back per cone; the cone transmittances fill the 128 world
// cells through a smooth kernel over the cone axes (so iso_xfer and bounce_dirs read nearvis unchanged), and with
// PP.refl.x the blockers (per cone T_before alpha, weighted by the cone's share of A V Omega; the own-layer rule of
// near_vis_refl) go to blk[8..16) for near_refl.

const CONE_LAMBDA: f32 = 0.5;   // use a node once smax <= CONE_LAMBDA x the cone radius at its depth
const CONE_T0: f32 = 0.02;      // the cut's cone radius never drops below its value at 2 cm
const CONE_KERNEL: f32 = 4.0;   // cone -> cell weights exp(CONE_KERNEL (cos - 1))

fn cone_axis(k: u32, K: u32) -> vec3f {
    if (K == 6u) {
        let s = select(1.0, -1.0, k >= 3u);
        return select(select(vec3f(0.0, 0.0, s), vec3f(0.0, s, 0.0), k % 3u == 1u), vec3f(s, 0.0, 0.0), k % 3u == 0u);
    }
    return vec3f(select(1.0, -1.0, (k & 1u) != 0u), select(1.0, -1.0, (k & 2u) != 0u),
                 select(1.0, -1.0, (k & 4u) != 0u)) * 0.5773503;
}

// one cone through one part's subtree (rest-frame P, w), hits kept as near_walk_hits does (tau = -ln(1 - alpha))
fn cone_walk_hits(root: u32, P: vec3f, w: vec3f, dist: f32, tan_a: f32, sg: f32, h: ptr<function, NearHits>) {
    var stack: array<u32, STACK>;
    stack[0] = root;
    var sp = 1u;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        let d = nd.mean - P;
        let tt = dot(d, w);
        if (tt < -nd.radius || tt > dist + nd.radius) { continue; }
        let tc = clamp(tt, 0.0, dist);
        let r = tan_a * tc;
        let e = d - tt * w;
        let e2 = dot(e, e);
        if (e2 > (r + nd.radius) * (r + nd.radius)) { continue; }
        if (ni >= U.gaussian_count && nd.smax > CONE_LAMBDA * tan_a * max(tc, CONE_T0)) {
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                if (sp < STACK) { stack[sp] = kids[j]; sp += 1u; }
            }
            continue;
        }
        // optical depth through the node's centre (a ray parallel to the axis), cut to [0, dist]
        let K = nd.kappa * dirm(nd, w) * seg_geom(nd, P + e, w, 0.0, dist);
        if (K <= 1e-5) { continue; }
        let cw = sym_mul(nd.cinv_diag, nd.cinv_off, w);
        let a = dot(cw, w);
        let sig2 = sqrt(a / max(sym_det(nd.cinv_diag, nd.cinv_off), 1e-30));   // sqrt(det) of the projected covariance
        let s2 = 0.5 * r * r;
        let cov = sig2 / (sig2 + s2) * exp(-0.5 * e2 / (sig2 + s2));
        let alpha = min(cov * (1.0 - exp(-K)), 0.999);
        if (alpha <= 1e-5) { continue; }
        let tau = -log(1.0 - alpha);
        (*h).tau_all += tau;
        if (tt < KEX * (sg + inverseSqrt(a))) { (*h).tau_ex += tau; continue; }
        if ((*h).k == NHIT && tt >= (*h).t[NHIT - 1u]) { continue; }
        var vb = 1.0;
        if (ni < U.gaussian_count) { vb = vgrid_lookup(vgrid[ni], -w); }
        var at = min((*h).k, NHIT - 1u);
        while (at > 0u && (*h).t[at - 1u] > tt) {
            (*h).t[at] = (*h).t[at - 1u]; (*h).tau[at] = (*h).tau[at - 1u];
            (*h).node[at] = (*h).node[at - 1u]; (*h).vb[at] = (*h).vb[at - 1u];
            at -= 1u;
        }
        (*h).t[at] = tt; (*h).tau[at] = tau; (*h).node[at] = ni; (*h).vb[at] = vb;
        (*h).k = min((*h).k + 1u, NHIT);
    }
}

@compute @workgroup_size(64)
fn near_cones(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x + PP.refl.y;
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    // near_vis's staleness rule
    var stale = PP.force_near != 0u || (parts[own].flags & 2u) != 0u;
    for (var p = 0u; p < U.part_count && !stale; p++) {
        if (p == own || (parts[p].flags & 2u) == 0u) { continue; }
        let b = parts[p].bound;
        let r = b.w + PP.near_r;
        stale = length(P - xf_point(parts[p].xf, b.xyz)) < r || length(P - xf_point(parts[p].xf_prev, b.xyz)) < r;
    }
    if (!stale) { return; }
    let nk = min(PP.refl.z, 8u);
    let ca = 1.0 - 2.0 / f32(nk);
    let tan_a = sqrt(1.0 - ca * ca) / ca;
    let refl = PP.refl.x != 0u;
    let nd = nodes[g];
    let xfo = parts[own].xf;
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    // per cone, its share of A V Omega over the cells (the receiver's exposure through it), for the blocker weights
    var Wk: array<f32, 8>;
    for (var k = 0u; k < 8u; k++) { Wk[k] = 0.0; }
    if (refl) {
        for (var c = 0u; c < NCELL; c++) {
            let dc = cell_dir(c);
            let u = xf_dir_inv(xfo, dc);
            let a = footprint(nd, s3 * s3, u) * vgrid_lookup(vgrid[g], u) * cell_omega(c);
            if (a <= 0.0) { continue; }
            var kw: array<f32, 8>;
            var ks = 0.0;
            for (var k = 0u; k < nk; k++) { kw[k] = exp(CONE_KERNEL * (dot(dc, cone_axis(k, nk)) - 1.0)); ks += kw[k]; }
            for (var k = 0u; k < nk; k++) { Wk[k] += a * kw[k] / ks; }
        }
    }
    var Tk: array<f32, 8>;
    var cid: array<u32, NCAND>;
    var cwt: array<f32, NCAND>;
    for (var q = 0u; q < NCAND; q++) { cid[q] = NO_BLOCKER; cwt[q] = 0.0; }
    var wtot = 0.0;
    for (var k = 0u; k < nk; k++) {
        let w = cone_axis(k, nk);
        let u = xf_dir_inv(xfo, w);
        var h: NearHits;
        h.k = 0u; h.tau_all = 0.0; h.tau_ex = 0.0;
        let sg = inverseSqrt(dot(u, sym_mul(nd.cinv_diag, nd.cinv_off, u)));
        for (var p = 0u; p < U.part_count; p++) {
            if (p == own) { continue; }
            let xf = parts[p].xf;
            let b = parts[p].bound;
            if (length(xf_point_inv(xf, P) - b.xyz) > b.w + PP.near_r) { continue; }
            cone_walk_hits(parts[p].root, xf_point_inv(xf, P), xf_dir_inv(xf, w), PP.near_r, tan_a, sg, &h);
        }
        let T = clamp(exp(-h.tau_all), 0.0, 1.0);
        Tk[k] = T;
        if (!refl || h.k == 0u || T >= 1.0 || Wk[k] <= 0.0) { continue; }
        // near_vis_refl's shares: own-layer hits first, then the kept ones front to back, rescaled to T + sum = 1
        var Tr = exp(-h.tau_ex);
        var ssum = 1.0 - Tr;
        for (var i = 0u; i < h.k; i++) { ssum += Tr * (1.0 - exp(-h.tau[i])); Tr *= exp(-h.tau[i]); }
        if (ssum <= 1e-9) { continue; }
        let sc = (1.0 - T) / ssum;
        Tr = exp(-h.tau_ex);
        for (var i = 0u; i < h.k; i++) {
            let wb = Wk[k] * Tr * (1.0 - exp(-h.tau[i])) * sc * h.vb[i];
            Tr *= exp(-h.tau[i]);
            if (wb <= 0.0) { continue; }
            wtot += wb;
            var slot = NCAND;
            var mn = 0u;
            for (var q = 0u; q < NCAND; q++) {
                if (cid[q] == h.node[i]) { slot = q; break; }
                if (cwt[q] < cwt[mn]) { mn = q; }
            }
            if (slot < NCAND) { cwt[slot] += wb; } else if (cwt[mn] < wb) { cid[mn] = h.node[i]; cwt[mn] = wb; }
        }
    }
    var out: VGrid;
    for (var j = 0u; j < 32u; j++) { out.w[j] = 0u; }
    for (var c = 0u; c < NCELL; c++) {
        let dc = cell_dir(c);
        var ts = 0.0;
        var ks = 0.0;
        for (var k = 0u; k < nk; k++) {
            let kw = exp(CONE_KERNEL * (dot(dc, cone_axis(k, nk)) - 1.0));
            ts += kw * Tk[k]; ks += kw;
        }
        out.w[c >> 2u] |= u32(round(clamp(ts / ks, 0.0, 1.0) * 255.0)) << ((c & 3u) * 8u);
    }
    nearvis[g] = out;
    if (!refl) { return; }
    // the 8 largest, rescaled to the total (near_vis_refl)
    var bi: array<u32, 8>;
    var bw: array<f32, 8>;
    var kept = 0.0;
    for (var r = 0u; r < 8u; r++) {
        var best = NCAND;
        for (var q = 0u; q < NCAND; q++) {
            if (cid[q] != NO_BLOCKER && cwt[q] > 0.0 && (best == NCAND || cwt[q] > cwt[best])) { best = q; }
        }
        bi[r] = NO_BLOCKER; bw[r] = 0.0;
        if (best == NCAND) { continue; }
        bi[r] = cid[best]; bw[r] = cwt[best]; kept += cwt[best];
        cwt[best] = 0.0;
    }
    let s = select(0.0, wtot / kept, kept > 0.0);
    for (var r = 0u; r < 8u; r++) { blk[g * 16u + 8u + r] = vec2u(bi[r], bitcast<u32>(bw[r] * s)); }
}

// ---- direct light toward the fixed directions ----------------------------------------------------------------

const KMAX: u32 = 64u;   // lamp samples kept per Gaussian (lamps x light_samples_axis^2)

@compute @workgroup_size(64)
fn direct_dirs(@builtin(global_invocation_id) gid: vec3u) {
    // stage 2 (U.iso bit 1): a quarter of the Gaussians per frame; the probes blend over frames anyway
    let g = select(gid.x, gid.x * 4u + U.frame % 4u, (U.iso & 2u) != 0u);
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let nd = nodes[g];
    let own = G.part_surface & 0xFFFFu;
    let xf = parts[own].xf;
    let P = xf_point(xf, G.mu);
    let mat = material_of(G);
    let fl = flake_of(G, mat);
    let na = max(U.light_samples_axis, 1u);
    let K = na * na;

    // light-side terms per sample: wi (rest frame), weight = Le win T geo V(wi) A(wi) / K
    var wis: array<vec3f, KMAX>;
    var ws: array<vec3f, KMAX>;
    var m = 0u;
    for (var l = 0u; l < U.lamp_count; l++) {
        let L = lamps[l];
        if ((L.flags & 1u) == 0u) { continue; }
        let win = window(L, length(L.center - P));
        if (win <= 0.0) { continue; }
        let scale = L.radiance * (win * lamp_transmittance(g, l) / f32(K));
        for (var k = 0u; k < K; k++) {
            if (m >= KMAX) { break; }
            let y = lamp_point(L, P, (f32(k / na) + 0.5) / f32(na), (f32(k % na) + 0.5) / f32(na));
            let geo = lamp_geometry(L, P, y);
            if (geo <= 0.0) { continue; }
            let wi = xf_dir_inv(xf, normalize(y - P));
            let vi = vgrid_lookup(vgrid[g], wi);
            if (vi <= 1e-7) { continue; }
            wis[m] = wi;
            ws[m] = scale * (geo * vi * footprint(nd, fl.detC, wi));
            m += 1u;
        }
    }
    var mean = vec3f(0.0);   // the mean over the 14 directions (common.wgsl LOB: one value toward the probes)
    if ((U.iso & 1u) != 0u) {                     // M39: one colour for every direction, plus specular flakes' lobe
        for (var k = 0u; k < 14u; k++) {
            let wo = xf_dir_inv(xf, PP.dirs[k].xyz);
            var c = vec3f(0.0);
            for (var s = 0u; s < m; s++) {
                let wr = mirror(wis[s], fl.n);
                let damp = iso_damp(iso_norm(nd), footprint(nd, fl.detC, wr) * vgrid_lookup(vgrid[g], wr));
                c += ws[s] * iso_lobe(mat, fl.Sd, fl.So, wis[s], wo, damp);
            }
            mean += c / 14.0;
        }
        lo_dir[g] = vec4f(G.albedo * mean * iso_norm(nd), 0.0);
        return;
    }
    for (var k = 0u; k < 14u; k++) {
        let wo = xf_dir_inv(xf, PP.dirs[k].xyz);
        let fsame = 1.0 / max(vgrid_lookup(vgrid[g], wo), mat.vfloor);
        let on = dot(wo, fl.n);
        var c = vec3f(0.0);
        for (var s = 0u; s < m; s++) {
            let f = select(1.0, fsame, dot(wis[s], fl.n) * on > 0.0);
            c += ws[s] * (phase_ex(mat, fl.n, fl.Sd, fl.So, wis[s], wo) * f);
        }
        mean += G.albedo * c / footprint(nd, fl.detC, wo) / 14.0;
    }
    lo_dir[g] = vec4f(mean, 0.0);
}

// ---- bounce: probes -> Gaussian ------------------------------------------------------------------------------

// The 8 probes around world point P and their normalised weights: trilinear x valid x Chebyshev (DDGI), as
// loop_surf.py. Shared by bounce_dirs and bounce_sh, so the validity and leak rule lives in one place.
struct ProbeWeights { pid: array<u32, 8>, w: array<f32, 8> }

// DDGI Chebyshev visibility of world point P from probe p
fn probe_cheb(p: u32, P: vec3f) -> f32 {
    let v = P - probe_pos(p);
    let dist = length(v);
    let mm = probe_data[2u * (p * NCELL + cell_of(v / max(dist, 1e-9))) + 1u];
    let var_ = max(mm.y - mm.x * mm.x, 1e-4);
    let dd = max(dist - 0.03 - mm.x, 0.0);
    if (dd <= 0.0) { return 1.0; }
    return pow(var_ / (var_ + dd * dd), 3.0);
}

// When none of the 8 is valid (a cell inside or against a solid): the 8 best valid probes of the surrounding 4x4x4
// block by Chebyshev / distance^2 (experiments/P1_probe_dark 'fallback').
fn probe_fallback(P: vec3f, fi: vec3u) -> ProbeWeights {
    let dm = vec3i(PP.dims.xyz);
    var out: ProbeWeights;
    for (var k = 0u; k < 8u; k++) { out.pid[k] = 0u; out.w[k] = 0.0; }
    for (var j = 0u; j < 64u; j++) {
        let ii = vec3i(fi) + vec3i(i32(j >> 4u), i32((j >> 2u) & 3u), i32(j & 3u)) - vec3i(1);
        if (any(ii < vec3i(0)) || any(ii >= dm)) { continue; }
        let p = lat_index(u32(ii.x), u32(ii.y), u32(ii.z));
        if (probe_data[2u * (p * NCELL) + 1u].z <= 0.0) { continue; }
        let d = max(length(P - probe_pos(p)), 0.05);
        let w = max(probe_cheb(p, P), 1e-3) / (d * d);
        if (w <= out.w[7]) { continue; }
        var at = 7u;                            // insertion into the descending top 8
        while (at > 0u && out.w[at - 1u] < w) {
            out.w[at] = out.w[at - 1u]; out.pid[at] = out.pid[at - 1u];
            at -= 1u;
        }
        out.w[at] = w; out.pid[at] = p;
    }
    var wsum = 0.0;
    for (var k = 0u; k < 8u; k++) { wsum += out.w[k]; }
    for (var k = 0u; k < 8u; k++) { out.w[k] /= max(wsum, 1e-12); }
    return out;
}

// the lattice cell around P: columns bilinear in x, y; the vertical coordinate is P's height above the interpolated
// ground, located among the levels (the room lattice: ground 0, levels evenly spaced, i.e. plain trilinear)
fn lat_cell(P: vec3f, fi: ptr<function, vec3u>, fr: ptr<function, vec3f>) {
    let dm = PP.dims.xyz;
    let fxy = clamp((P.xy - PP.lo.xy) / PP.step.xy, vec2f(0.0), vec2f(dm.xy - vec2u(1u)));
    let ij = min(vec2u(floor(fxy)), dm.xy - vec2u(2u));
    let t = fxy - vec2f(ij);
    let g = mix(mix(lat_ground(ij.x, ij.y), lat_ground(ij.x, ij.y + 1u), t.y),
                mix(lat_ground(ij.x + 1u, ij.y), lat_ground(ij.x + 1u, ij.y + 1u), t.y), t.x);
    let h = P.z - g;
    var k = 0u;
    while (k + 2u < dm.z && h > lat_level(k + 1u)) { k += 1u; }
    let tz = clamp((h - lat_level(k)) / max(lat_level(k + 1u) - lat_level(k), 1e-6), 0.0, 1.0);
    *fi = vec3u(ij, k);
    *fr = vec3f(t, tz);
}

fn probe_weights(P: vec3f, own: u32) -> ProbeWeights {
    if (NODE_PROBES && is_part_probe_part(own)) {
        // node probes: the part's PART_PROBES_PER probes weighted exp(-d^2 / 2 r^2) by rest-frame distance, the 8 largest
        let lp = xf_point_inv(parts[own].xf, P);
        var out: ProbeWeights;
        for (var k = 0u; k < 8u; k++) { out.pid[k] = lat_count(); out.w[k] = 0.0; }
        var tot = 0.0;
        for (var k = 0u; k < PART_PROBES_PER; k++) {
            let q = PP.nodep[own * PART_PROBES_PER + k];
            let dv = lp - q.xyz;
            let w = exp(-0.5 * dot(dv, dv) / max(q.w * q.w, 1e-8)) + 1e-12;
            var j = 7u;                                          // insert into the top 8 (smallest at the end)
            if (w <= out.w[7]) { continue; }
            loop {
                if (j == 0u || out.w[j - 1u] >= w) { break; }
                out.w[j] = out.w[j - 1u]; out.pid[j] = out.pid[j - 1u]; j -= 1u;
            }
            out.w[j] = w; out.pid[j] = lat_count() + own * PART_PROBES_PER + k;
        }
        for (var k = 0u; k < 8u; k++) { tot += out.w[k]; }
        for (var k = 0u; k < 8u; k++) { out.w[k] /= max(tot, 1e-12); }
        return out;
    }
    if (is_part_probe_part(own)) {      // a movable part's Gaussians: trilinear over its own 8 probes, no Chebyshev
        let pf = part_probe_frame(own);
        let f = clamp((xf_point_inv(parts[own].xf, P) - pf.c + pf.h) / max(2.0 * pf.h, vec3f(1e-6)), vec3f(0.0), vec3f(1.0));
        var out: ProbeWeights;
        for (var k = 0u; k < 8u; k++) {
            let o = vec3u(k >> 2u, (k >> 1u) & 1u, k & 1u);
            let t = select(1.0 - f, f, o == vec3u(1u));
            out.pid[k] = lat_count() + own * 8u + k;
            out.w[k] = t.x * t.y * t.z;
        }
        return out;
    }
    // a static object's own volume: its Gaussians only, trilinear (no Chebyshev: the probes skip the object itself)
    if (has_vol(own)) {
        let vf = vol_frame(own);
        let lp = xf_point_inv(parts[own].xf, P);
        let d = vol_dims(own);
        let fq = clamp((lp - vf.c + vf.h) / max(2.0 * vf.h, vec3f(1e-6)), vec3f(0.0), vec3f(1.0)) * (vec3f(d) - vec3f(1.0));
        let fi = min(vec3u(floor(fq)), d - vec3u(2u));
        let fr = fq - vec3f(fi);
        var out: ProbeWeights;
        for (var k = 0u; k < 8u; k++) {
            let o = vec3u(k >> 2u, (k >> 1u) & 1u, k & 1u);
            let ii = fi + o;
            out.pid[k] = lat_count() + PP.vols[own].x + (ii.x * d.y + ii.y) * d.z + ii.z;
            let t = select(1.0 - fr, fr, o == vec3u(1u));
            out.w[k] = t.x * t.y * t.z;
        }
        return out;
    }
    var fi: vec3u;
    var fr: vec3f;
    lat_cell(P, &fi, &fr);
    var out: ProbeWeights;
    var wsum = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let o = vec3u(k >> 2u, (k >> 1u) & 1u, k & 1u);
        let ii = fi + o;
        let p = lat_index(ii.x, ii.y, ii.z);
        out.pid[k] = p;
        let t = select(1.0 - fr, fr, o == vec3u(1u));
        let valid = probe_data[2u * (p * NCELL) + 1u].z;
        let cw = t.x * t.y * t.z * valid;
        out.w[k] = cw * max(probe_cheb(p, P), 1e-3);
        wsum += out.w[k];
    }
    if (wsum <= 0.0) { return probe_fallback(P, fi); }
    for (var k = 0u; k < 8u; k++) { out.w[k] /= max(wsum, 1e-12); }
    return out;
}

@compute @workgroup_size(64)
fn bounce_dirs(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x * PP.subset + PP.phase;   // round-robin: this frame's share of the Gaussians
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let nd = nodes[g];
    let own = G.part_surface & 0xFFFFu;
    let xf = parts[own].xf;
    let P = xf_point(xf, G.mu);
    let mat = material_of(G);
    let fl = flake_of(G, mat);

    // stage 2: matte Gaussians take the bounce from bounce_sh; only specular flakes need the per-cell lobe here
    if ((U.iso & 3u) == 3u && (spec_weight(mat) <= 0.0 || gloss_from_sh(mat))) { return; }   // bounce_sh has these
    let pw = probe_weights(P, own);
    var pid = pw.pid;      // var: indexed at runtime below
    var wk = pw.w;

    // outgoing toward D_k and the eye (same transfer as the direct light), accumulated cell by cell so no per-cell
    // arrays are kept: Lin_c = sum_k w_k Lp_k,c x V(u_c) x near(u_c) x A(u_c) x Omega_c. With U.iso bit 0 (M39) one
    // value for every direction: alb iso_norm sum_c Lin_c.
    let iso = (U.iso & 1u) != 0u;
    let ks = spec_weight(mat);
    let asg_cells = iso && (U.iso & 4u) != 0u;
    // cells within 3 (cell std + mirror-lobe std) of the eye's mirror direction take the ASG
    let lobe_w = sqrt(min(mat.flake.y, mat.flake.z) / max(mat.flake.x, 1e-12));
    let cos_near = cos(min(3.0 * (CELL_STD * PI / 8.0 + 2.0 * lobe_w), PI));
    var wo: array<vec3f, 15>;
    var fsame: array<f32, 15>;
    var acc: array<vec3f, 15>;
    for (var k = 0u; k < 15u; k++) {
        var wo_w = PP.dirs[min(k, 13u)].xyz;
        if (k == 14u) { wo_w = normalize(U.eye - P); }
        wo[k] = xf_dir_inv(xf, wo_w);
        fsame[k] = 1.0 / max(vgrid_lookup(vgrid[g], wo[k]), mat.vfloor);
        acc[k] = vec3f(0.0);
    }
    let wr_eye = mirror(wo[14], fl.n);
    // reflection coverage (U.iso bit 4): toward the eye the specular share goes to spec14, the eye slot keeps (1 - k)
    let split = iso && (U.iso & 16u) != 0u;
    var spec14 = vec3f(0.0);
    for (var c = 0u; c < NCELL; c++) {
        var L = vec3f(0.0);
        for (var k = 0u; k < 8u; k++) {
            if (wk[k] > 0.0) { L += wk[k] * probe_data[2u * (pid[k] * NCELL + c)].rgb; }
        }
        if (L.r + L.g + L.b <= 0.0) { continue; }
        let u = xf_dir_inv(xf, cell_dir(c));
        let vis = vgrid_lookup(vgrid[g], u) * near_byte(g, c);
        if (vis <= 0.0) { continue; }
        let lin = L * (vis * footprint(nd, fl.detC, u) * cell_omega(c));
        if (iso) {
            if (ks <= 0.0) { acc[0] += lin; continue; }
            let ur = mirror(u, fl.n);
            let damp = iso_damp(iso_norm(nd), footprint(nd, fl.detC, ur) * vgrid_lookup(vgrid[g], ur));
            for (var k = 0u; k < 14u; k++) { acc[k] += lin * iso_lobe(mat, fl.Sd, fl.So, u, wo[k], damp); }
            // toward the eye with U.iso bit 2 (P3 ASG): the cell as a light with its own angular size, so a sharp
            // floor shows the room smoothly instead of one dot per cell. Only cells near the mirror direction: further
            // out asg_phase hands off to the point phase anyway (about 8 of the 128 cells pay)
            var lobe14 = 0.0;
            if (asg_cells && dot(u, wr_eye) > cos_near) {
                lobe14 = iso_lobe_from(ks, cell_asg_phase(c, xf, u, fl, mat, wo[14]), damp);
            } else {
                lobe14 = iso_lobe(mat, fl.Sd, fl.So, u, wo[14], damp);
            }
            if (split) { acc[14] += lin * (1.0 - ks); spec14 += lin * (lobe14 - (1.0 - ks)); } else { acc[14] += lin * lobe14; }
            continue;
        }
        let un = dot(u, fl.n);
        for (var k = 0u; k < 15u; k++) {
            let f = select(1.0, fsame[k], un * dot(wo[k], fl.n) > 0.0);
            acc[k] += lin * (phase_ex(mat, fl.n, fl.Sd, fl.So, u, wo[k]) * f);
        }
    }
    if (iso) {
        var send = acc[0];
        if (ks > 0.0) {   // shiny: the mean over the 14 directions weighted as node_leaf weights them (footprint x V)
            send = vec3f(0.0);
            var ws = 0.0;
            for (var k = 0u; k < 14u; k++) {
                let w = footprint(nd, fl.detC, wo[k]) * vgrid_lookup(vgrid[g], wo[k]);
                send += acc[k] * w; ws += w;
            }
            send /= max(ws, 1e-20);
        }
        let s = G.albedo * iso_norm(nd);
        lo_b[g * LOB + LOB_EYE] = vec4f(s * select(acc[14], acc[0], ks <= 0.0), 0.0);
        lo_b[g * LOB + LOB_SEND] = vec4f(s * send, 0.0);
        // the eye's specular bounce (project.wgsl reads the lanes with coverage on)
        lo_b[g * LOB + LOB_LANES] = vec4f(s * spec14, 0.0);
        return;
    }
    var send = vec3f(0.0);
    for (var k = 0u; k < 14u; k++) { send += G.albedo * acc[k] / footprint(nd, fl.detC, wo[k]) / 14.0; }
    lo_b[g * LOB + LOB_EYE] = vec4f(G.albedo * acc[14] / footprint(nd, fl.detC, wo[14]), 0.0);
    lo_b[g * LOB + LOB_SEND] = vec4f(send, 0.0);
    lo_b[g * LOB + LOB_LANES] = vec4f(0.0);
}

// ---- M39 stage 2: SH2 probes ----------------------------------------------------------------------------------
// The bounce a matte Gaussian shows is alb max(t' . L, 0): L = the 8 surrounding probes' SH2 (probe_weights), t' =
// SH2 of its transfer grid u(w) = iso_norm A(w) V(w) near(w) over the 128 world cells (experiments/M39_iso_lighting
// bounce_sh.md; matches the per-cell gather within ~2% per part on the CPU).

// Per Gaussian, only where something moved relative to it (near_vis's rule) or on a forced refresh: t' in world frame.
@compute @workgroup_size(64)
fn iso_xfer(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count || (U.iso & 3u) != 3u) { return; }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let xf = parts[own].xf;
    let P = xf_point(xf, G.mu);
    var stale = PP.force_near != 0u || (parts[own].flags & 2u) != 0u;
    for (var p = 0u; p < U.part_count && !stale; p++) {
        if (p == own || (parts[p].flags & 2u) == 0u) { continue; }
        let b = parts[p].bound;
        let r = b.w + PP.near_r;
        stale = length(P - xf_point(parts[p].xf, b.xyz)) < r || length(P - xf_point(parts[p].xf_prev, b.xyz)) < r;
    }
    if (!stale) { return; }
    let nd = nodes[g];
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    let inorm = iso_norm(nd);
    var t: array<f32, 9>;
    for (var m = 0u; m < 9u; m++) { t[m] = 0.0; }
    for (var c = 0u; c < NCELL; c++) {
        let w = cell_dir(c);
        let u = xf_dir_inv(xf, w);
        let v = vgrid_lookup(vgrid[g], u) * near_byte(g, c);
        if (v <= 0.0) { continue; }
        let a = inorm * footprint(nd, s3 * s3, u) * v * cell_omega(c);
        let y = sh2_basis(w);
        for (var m = 0u; m < 9u; m++) { t[m] += a * y[m]; }
    }
    xfer[3u * g] = vec4f(t[0], t[1], t[2], t[3]);
    xfer[3u * g + 1u] = vec4f(t[4], t[5], t[6], t[7]);
    xfer[3u * g + 2u] = vec4f(t[8], 0.0, 0.0, 0.0);
}

// Per probe, after probe_gather: SH2 of the cells' radiance (radiance arriving from cell direction w).
@compute @workgroup_size(64)
fn probe_sh2(@builtin(global_invocation_id) gid: vec3u) {
    let p = gid.x;
    if (p >= lat_count() + PP.part_probes) { return; }
    var c9: array<vec3f, 9>;
    for (var m = 0u; m < 9u; m++) { c9[m] = vec3f(0.0); }
    for (var c = 0u; c < NCELL; c++) {
        let L = probe_data[2u * (p * NCELL + c)].rgb * cell_omega(c);
        let y = sh2_basis(cell_dir(c));
        for (var m = 0u; m < 9u; m++) { c9[m] += L * y[m]; }
    }
    for (var m = 0u; m < 9u; m++) { probe_sh[p * 9u + m] = vec4f(c9[m], 0.0); }
}

// Per Gaussian (every frame, or round-robin as bounce_dirs when sh_full is 0): the matte bounce from the probes' SH2, the same toward every direction.
@compute @workgroup_size(64)
fn bounce_sh(@builtin(global_invocation_id) gid: vec3u) {
    // every Gaussian each frame unless sh_full is 0: a round-robin quarter sparkles while the probes settle, since
    // Gaussian order is shuffled and neighbours then show probe light from different frames
    let g = select(gid.x * PP.subset + PP.phase, gid.x, PP.sh_full != 0u);
    if (g >= U.gaussian_count || (U.iso & 3u) != 3u) { return; }
    let G = gaussians[g];
    let mat = material_of(G);
    let ks = spec_weight(mat);
    if (ks > 0.0 && !gloss_from_sh(mat)) { return; }        // bounce_dirs handles near-mirror materials
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    var pw = probe_weights(P, own);
    var Lsh: array<vec3f, 9>;
    for (var m = 0u; m < 9u; m++) { Lsh[m] = vec3f(0.0); }
    for (var k = 0u; k < 8u; k++) {
        let w = pw.w[k];
        if (w <= 0.0) { continue; }
        for (var m = 0u; m < 9u; m++) { Lsh[m] += w * probe_sh[pw.pid[k] * 9u + m].rgb; }
    }
    let t0 = xfer[3u * g]; let t1 = xfer[3u * g + 1u]; let t2 = xfer[3u * g + 2u];
    let c = t0.x * Lsh[0] + t0.y * Lsh[1] + t0.z * Lsh[2] + t0.w * Lsh[3] + t1.x * Lsh[4] + t1.y * Lsh[5]
          + t1.z * Lsh[6] + t1.w * Lsh[7] + t2.x * Lsh[8];
    let cb = vec4f(G.albedo * max(c, vec3f(0.0)), 0.0);
    lo_b[g * LOB + LOB_EYE] = cb;
    lo_b[g * LOB + LOB_SEND] = cb;
    lo_b[g * LOB + LOB_LANES] = vec4f(0.0);
    if (ks > 0.0) {   // glossy: the diffuse share as above, the specular share from the probes' SH2 (gloss_sh)
        let sp = gloss_sh(g, G, mat, ks, P, own, Lsh);
        let split = (U.iso & 16u) != 0u;   // reflection coverage: the highlight in the lanes (project.wgsl)
        lo_b[g * LOB + LOB_EYE] = vec4f(cb.rgb * (1.0 - ks) + select(sp, vec3f(0.0), split), 0.0);
        if (split) { lo_b[g * LOB + LOB_LANES] = vec4f(sp, 0.0); }
        return;
    }
    // view-only sheen (U.iso bit 5, Fresnel in project.wgsl): the probes' SH2 radiance from the eye's mirror direction
    // about the short axis (two-sided), uncoloured, into the lanes. Evaluated here because project.wgsl has no binding
    // left for the probes; it never feeds back into lighting.
    if ((U.iso & 32u) != 0u) {
        let n = normalize(xf_dir(parts[own].xf, quat_matrix(G.quat)[0]));
        let y = sh2_basis(mirror(normalize(U.eye - P), n));
        var sh = vec3f(0.0);
        for (var m = 0u; m < 9u; m++) { sh += y[m] * Lsh[m]; }
        sh = max(sh, vec3f(0.0));
        lo_b[g * LOB + LOB_LANES] = vec4f(sh, 0.0);
    }
}

// Glossy bounce from the probes' SH2 (David, 2026-10-30: shiny grass made bounce_dirs' 128-cell walk crawl on the
// hill). bounce_dirs gives the eye alb iso_norm sum_c Lin_c iso_lobe; its specular share
// k 4 pi p_spec(u_c, w_o) damp, with the phase normalised and peaked at the mirror direction w_m of w_o about the
// flake normal, is about k 4 pi damp L~(w_m) V(w_m) A(w_m): L~ the probes' SH2 at w_m blurred by the reflected lobe
// (bands scaled exp(-l (l + 1) s^2 / 2), s = 2 x the flake lobe's angular width). Same lobe energy, no per-cell
// walk; SH2 can't hold a sharp reflection, so near-mirror materials (lobe narrower than GLOSS_SH_MIN) keep the walk.
// Toward the probes a glossy Gaussian sends its matte value (the lobe averages to about 1 over directions).
const GLOSS_SH_MIN: f32 = 0.1;   // lobe width (rad) from which the SH2 path is used

fn gloss_lobe_w(mat: Material) -> f32 { return sqrt(min(mat.flake.y, mat.flake.z) / max(mat.flake.x, 1e-12)); }
fn gloss_from_sh(mat: Material) -> bool { return (U.iso & 3u) == 3u && gloss_lobe_w(mat) >= GLOSS_SH_MIN; }

fn gloss_sh(g: u32, G: Gaussian, mat: Material, ks: f32, P: vec3f, own: u32, Lsh: array<vec3f, 9>) -> vec3f {
    let xf = parts[own].xf;
    let nd = nodes[g];
    let n = quat_matrix(G.quat)[0];                              // rest frame
    let wo = xf_dir_inv(xf, normalize(U.eye - P));
    let wm = mirror(wo, n);                                     // where the light comes from (rest frame)
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    let detC = s3 * s3;
    let vis = vgrid_lookup(vgrid[g], wm) * footprint(nd, detC, wm);
    if (vis <= 0.0) { return vec3f(0.0); }
    let inorm = iso_norm(nd);
    let damp = iso_damp(inorm, footprint(nd, detC, wo) * vgrid_lookup(vgrid[g], wo));   // w_r = mirror of w_m = w_o
    let sg = 2.0 * gloss_lobe_w(mat);
    let a1 = exp(-sg * sg);           // l = 1: exp(-2 s^2 / 2)
    let a2 = exp(-3.0 * sg * sg);     // l = 2: exp(-6 s^2 / 2)
    let y = sh2_basis(xf_dir(xf, wm));
    var L = y[0] * Lsh[0];
    for (var m = 1u; m < 4u; m++) { L += a1 * y[m] * Lsh[m]; }
    for (var m = 4u; m < 9u; m++) { L += a2 * y[m] * Lsh[m]; }
    return G.albedo * inorm * ks * 4.0 * PI * damp * max(L, vec3f(0.0)) * vis;
}

// ---- near-field reflection: the blockers' colour into the bounce ----------------------------------------------

// blocker i's outgoing colour: a leaf's direct + last frame's bounce toward the probes, a parent's emitter SH2 at DC
// (its mean over directions)
fn blocker_colour(i: u32) -> vec3f {
    if (i < U.gaussian_count) { return colour[i].rgb + lo_b_prev[i * LOB + LOB_SEND].rgb; }
    return max(node_sh_at(i, 0u) * 0.282095, vec3f(0.0));
}

// Per Gaussian whose lo_b bounce_sh / bounce_dirs rewrote this frame (M39 only), after them: R = alb iso_norm
// sum_i w_i C_i. Other parts' blockers (blk 8-15) fill light near_vis took away, so they add. Own-part blockers
// (0-7) sit where the V grid is dark, which iso_norm already normalises away, so they join as an average: the
// Gaussian's total (direct + bounce + R) is divided by 1 + r, r = iso_norm W_own; colour[] is the direct, so lo_b
// takes (lo_b + R - r colour) / (1 + r). Loop gain stays below alb, so the one-frame-lag multi-bounce converges.
@compute @workgroup_size(64)
fn near_refl(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count || (U.iso & 1u) == 0u) { return; }
    let G = gaussians[g];
    let matte = spec_weight(material_of(G)) <= 0.0;
    if (!(((U.iso & 3u) == 3u && matte && PP.sh_full != 0u) || g % PP.subset == PP.phase)) { return; }
    var R = vec3f(0.0);
    var wown = 0.0;
    for (var k = 0u; k < 16u; k++) {
        let e = blk[g * 16u + k];
        let w = bitcast<f32>(e.y);
        if (e.x == NO_BLOCKER || w <= 0.0) { continue; }
        // emitters (the lamp's Gaussians) are skipped: their light already arrives as direct light
        if (e.x < U.gaussian_count && (material_of(gaussians[e.x]).flags & 1u) != 0u) { continue; }
        R += w * blocker_colour(e.x);
        if (k < 8u) { wown += w; }
    }
    let inorm = iso_norm(nodes[g]);
    let r = inorm * wown;
    let add = G.albedo * inorm * R - r * colour[g].rgb;
    for (var k = 0u; k < 2u; k++) {   // the eye and the probes' bounce (LOB_EYE, LOB_SEND); the lanes stay
        lo_b[g * LOB + k] = vec4f((lo_b[g * LOB + k].rgb + add) / (1.0 + r), 0.0);
    }
}

// ---- node emitters -------------------------------------------------------------------------------------------

@compute @workgroup_size(64)
fn node_leaf(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= U.gaussian_count) { return; }
    node_leaf_one(gid.x);
}

fn node_leaf_one(g: u32) {
    let G = gaussians[g];
    let nd = nodes[g];
    let xf = parts[G.part_surface & 0xFFFFu].xf;
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    // emitters lit by the emitter cut catch their neighbours' emission at point-blank range (M47: the lamp's
    // Gaussians reflect ~3 in lo_dir vs ~0.03 on the floor); with PP.refl.w they send nothing into the probes, and
    // their light reaches the room only as direct light (near_refl skips them the same way). Weight kept: the lamp
    // still occludes what is behind it. (By part, not material: node_leaf has no materials binding.)
    let dark = PP.refl.w != 0u && (G.part_surface & 0xFFFFu) == PP.refl.w - 1u;
    let t = tree_slot(g);
    if (t >= acc_null()) { return; }
    for (var k = 0u; k < 14u; k++) {
        let wo = xf_dir_inv(xf, PP.dirs[k].xyz);
        let w = footprint(nd, s3 * s3, wo) * vgrid_lookup(vgrid[g], wo);
        let L = select(lo_dir[g].rgb + lo_b[g * LOB + LOB_SEND].rgb, vec3f(0.0), dark);
        node_acc[t * 14u + k] = vec4f(L * w, w);
    }
}

@compute @workgroup_size(64)
fn node_up(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= LV.r1) { return; }
    node_up_one(up_nodes[LV.r0 + gid.x]);   // this depth's internal nodes (lighting.js levels: r0 offset, r1 count)
}

fn node_up_one(i: u32) {
    if (i < U.gaussian_count || i >= PP.node_count) { return; }
    let nd = nodes[i];
    if (nd.depth != LV.depth || nd.part == 0xFFFFFFFFu) { return; }
    let t = tree_slot(i);
    if (t >= acc_null()) { return; }
    for (var k = 0u; k < 14u; k++) {
        var s = vec4f(0.0);
        for (var j = nd.child_lo; j < nd.child_hi; j++) { s += node_acc[tree_slot(kids[j]) * 14u + k]; }
        node_acc[t * 14u + k] = s;
    }
}

@compute @workgroup_size(64)
fn node_sh2(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= PP.node_count) { return; }
    node_sh2_one(gid.x);
}

fn node_sh2_one(i: u32) {
    let t = tree_slot(i);
    if (t >= sh_null() || t >= acc_null()) { return; }
    var L: array<vec3f, 14>;
    for (var k = 0u; k < 14u; k++) {
        let a = node_acc[t * 14u + k];
        L[k] = a.rgb / max(a.w, 1e-12);
    }
    var h: array<f32, 28>;
    h[27] = 0.0;
    for (var m = 0u; m < 9u; m++) {
        var c = vec3f(0.0);
        for (var k = 0u; k < 14u; k++) { c += pinv_at(m, k) * L[k]; }
        h[3u * m] = c.x; h[3u * m + 1u] = c.y; h[3u * m + 2u] = c.z;
    }
    for (var j = 0u; j < 14u; j++) { node_sh[t * 14u + j] = pack2x16float(vec2f(h[2u * j], h[2u * j + 1u])); }
}

// ---- probe gather --------------------------------------------------------------------------------------------

// distance to the first enabled rectangle lamp along o + t w (the lamp ends the ray; it is not a bounce emitter)
fn hit_lamps(o: vec3f, w: vec3f, tmax: f32) -> f32 {
    var t1 = tmax;
    for (var l = 0u; l < U.lamp_count; l++) {
        let L = lamps[l];
        if ((L.flags & 1u) == 0u || L.kind != LAMP_RECT) { continue; }
        let den = dot(w, L.normal);
        if (abs(den) < 1e-6) { continue; }
        let t = dot(L.center - o, L.normal) / den;
        if (t <= 0.0 || t >= t1) { continue; }
        let h = o + t * w - L.center;
        if (abs(dot(h, L.e1)) <= dot(L.e1, L.e1) && abs(dot(h, L.e2)) <= dot(L.e2, L.e2)) { t1 = t; }
    }
    return t1;
}

const FIR_TAPS: vec3f = vec3f(0.5, 0.3, 0.2);   // probe_gather alpha < 0: weights of this gather and the two before

@compute @workgroup_size(64)
fn probe_gather(@builtin(global_invocation_id) gid: vec3u) {
    let np = lat_count();
    let id = gid.x * PP.subset + PP.phase;  // round-robin over (probe, cell) rays
    if (id >= (np + PP.part_probes) * NCELL) { return; }
    let p = id / NCELL;
    let c = id % NCELL;
    // a part probe skips its own part (and does nothing for static parts, which stay on the lattice)
    // (a static volume's probes skip their object too)
    var skip = 0xFFFFFFFFu;
    if (p >= np && p < np + PP.vol_info.x) {
        skip = (p - np) / PART_PROBES_PER;
        if (!is_part_probe_part(skip)) { return; }
    } else if (p >= np) {
        skip = vol_of_probe(p - np);
        if (skip == 0xFFFFFFFFu) { return; }
    }
    let o = probe_pos(p);
    let w = cell_dir(c);
    // the shared conservative cut of this probe's 2x2x2 block (M34 c); a part probe cuts around itself
    // (the room lattice only; terrain columns differ in height, so each terrain probe cuts around itself)
    let dz = PP.dims.z;
    let b = vec3f(f32((p / (PP.dims.y * dz)) / 2u), f32(((p / dz) % PP.dims.y) / 2u), f32((p % dz) / 2u));
    let cen = select(PP.lo + PP.step * (2.0 * b + vec3f(0.5)), o, p >= np || PP.dims.w == 1u);
    let t1 = hit_lamps(o, w, PP.tmax);

    var ht: array<f32, HIT_MAX>;
    var hv: array<f32, HIT_MAX>;
    var hn: array<u32, HIT_MAX>;
    var k = 0u;
    let th = PP.theta;
    // the block's radius keeps a shared 2x2x2 cut conservative for all its probes; a probe that cuts around itself
    // (terrain, part and volume probes) needs no margin (2026-10-29: on the hill the 3.8 m margin opened every
    // node near a probe down to its leaves)
    let brad = select(PP.block_rad, 0.0, (p >= np || PP.dims.w == 1u) && PP.vol_info.y == 0u);   // vol_info.y = 1: ?probemargin=old
    for (var q = 0u; q < U.part_count; q++) {
        if (q == skip) { continue; }
        let xf = parts[q].xf;
        let Po = xf_point_inv(xf, o);
        let wr = xf_dir_inv(xf, w);
        let cr = xf_point_inv(xf, cen);
        var stack: array<u32, STACK>;
        stack[0] = parts[q].root;
        var sp = 1u;
        while (sp > 0u) {
            sp -= 1u;
            let ni = stack[sp];
            let nd = nodes[ni];
            let d = nd.mean - Po;
            let tt = clamp(dot(d, wr), 0.0, t1);
            let e = d - tt * wr;
            if (dot(e, e) > nd.radius * nd.radius) { continue; }
            var use_node = ni < U.gaussian_count;
            if (!use_node) { use_node = length(cr - nd.mean) - brad > th * nd.smax; }
            if (use_node) {
                let tau = nd.kappa * dirm(nd, wr) * seg_geom(nd, Po, wr, 0.0, t1);
                if (tau > 1e-6) {
                    let cw = sym_mul(nd.cinv_diag, nd.cinv_off, wr);
                    let den = dot(cw, wr);
                    let tp = select(t1, clamp(dot(cw, d) / den, 0.0, t1), abs(den) > 1e-30);   // the depth the composite uses
                    // insertion into the depth-sorted hit list. When full, the farthest hit folds into the last entry
                    // instead of being dropped: its optical depth adds, the entry's depth becomes the farthest folded in
                    // (probe_cheb's moments) and its node (light, first_part, backface) stays the nearest one. Dropping
                    // let a wall behind HIT_MAX faint hits go missing (about 1 probe ray in 20 came out 13% or more too
                    // open on the marbles); with the fold 16 entries look the same as 24 and the gather is ~1.9 ms
                    // faster on the M4 (ray-intersect thread, experiments/ray_intersect/probe_proto/m4_fold).
                    var at = min(k, HIT_MAX - 1u);
                    let l = HIT_MAX - 1u;
                    if (k == HIT_MAX && tp >= ht[l]) {
                        ht[l] = max(ht[l], tp);
                        hv[l] = min(hv[l] + tau, 64.0);   // alpha is 1 by then
                        continue;
                    }
                    let full = k == HIT_MAX;
                    let sv = hv[l];   // the entry about to be evicted
                    let st = ht[l];
                    while (at > 0u && ht[at - 1u] > tp) {
                        ht[at] = ht[at - 1u]; hv[at] = hv[at - 1u]; hn[at] = hn[at - 1u];
                        at -= 1u;
                    }
                    ht[at] = tp; hv[at] = tau; hn[at] = ni;
                    if (full) {
                        ht[l] = max(ht[l], st);
                        hv[l] = min(hv[l] + sv, 64.0);
                    }
                    k = min(k + 1u, HIT_MAX);
                }
            } else {
                for (var j = nd.child_lo; j < nd.child_hi; j++) {
                    if (sp < STACK) { stack[sp] = kids[j]; sp += 1u; }
                }
            }
        }
    }

    var T = 1.0;
    var acc = vec3f(0.0);
    var m1 = 0.0;
    var m2 = 0.0;
    var first_part = -1.0;
    var near = 0.0;
    let bw = sh2_basis(-w);
    for (var i = 0u; i < k; i++) {
        if (T <= 0.0) { break; }   // opaque: later hits add nothing
        let al = 1.0 - exp(-hv[i]);
        acc += T * al * max(node_radiance(hn[i], bw), vec3f(0.0));
        let dep = clamp(ht[i], 0.0, t1);
        m1 += T * al * dep;
        m2 += T * al * dep * dep;
        let Tn = T * (1.0 - al);
        if (T >= 0.5 && Tn < 0.5) {
            let hp = nodes[hn[i]].part;
            first_part = f32(hp);
            near = select(0.0, 1.0, dep < 0.05);
            // backface (experiments/P1_probe_dark): the hit leaf is buried toward the probe, V(-w) < 0.1, so the probe
            // looks at a solid from inside. Parents carry no V; an inside probe's crossing is near, where the cut
            // keeps leaves.
            if (hn[i] < U.gaussian_count && vgrid_lookup(vgrid[hn[i]], xf_dir_inv(parts[hp].xf, -w)) < 0.1) {
                near += 2.0;
            }
        }
        T = Tn;
    }
    // what the ray leaves unblocked out to tmax (no lamp hit) is sky (SPEC 2.9; zero without an environment)
    if (ENV.info.y != 0u && t1 >= PP.tmax * 0.999) { acc += T * sky_radiance(w); }
    m1 += T * t1;
    m2 += T * t1 * t1;

    let a = PP.alpha;
    let old0 = probe_data[2u * id];
    let old1 = probe_data[2u * id + 1u];
    var o0 = mix(old0.rgb, acc, a); var om = vec2f(mix(old1.x, m1, a), mix(old1.y, m2, a));
    // alpha < 0: a tapered blend of this gather and the two before it (FIR_TAPS) instead of the EMA, so a change is fully
    // in after 3 gathers (David, 2026-10-28). The raw gathers live after the records (the buffer is 3x: 2 records + 4
    // history vec4f per ray: (rgb, m1), (m2) for the last and the one before); the first gather (alpha 1) fills both.
    if (a < 0.0) {
        let h = arrayLength(&probe_data) / 3u + 4u * id;
        let g1a = probe_data[h]; let g1b = probe_data[h + 1u];
        let g2a = probe_data[h + 2u]; let g2b = probe_data[h + 3u];
        o0 = FIR_TAPS.x * acc + FIR_TAPS.y * g1a.rgb + FIR_TAPS.z * g2a.rgb;
        om = FIR_TAPS.x * vec2f(m1, m2) + FIR_TAPS.y * vec2f(g1a.w, g1b.x) + FIR_TAPS.z * vec2f(g2a.w, g2b.x);
        probe_data[h + 2u] = g1a; probe_data[h + 3u] = g1b;
        probe_data[h] = vec4f(acc, m1); probe_data[h + 1u] = vec4f(m2, 0.0, 0.0, 0.0);
    } else if (a >= 1.0) {
        let h = arrayLength(&probe_data) / 3u + 4u * id;
        if (h + 3u < arrayLength(&probe_data)) {
            probe_data[h] = vec4f(acc, m1); probe_data[h + 1u] = vec4f(m2, 0.0, 0.0, 0.0);
            probe_data[h + 2u] = vec4f(acc, m1); probe_data[h + 3u] = vec4f(m2, 0.0, 0.0, 0.0);
        }
    }
    probe_data[2u * id] = vec4f(o0, first_part);
    // a part probe keeps its world position in the unused .z of cells 1-3, for passes that can't bind nodes
    // (progressive.wgsl); cell 0's .z is the valid flag
    var z1 = old1.z;
    if (p >= np && c >= 1u && c <= 3u) { z1 = o[c - 1u]; }
    probe_data[2u * id + 1u] = vec4f(om, z1, near);
}

// A probe is dropped when half its rays start inside a slab, when nearly all of them first hit one moving part
// (it sits inside a box shell), or when more than half first hit a Gaussian from its buried side (inside a solid;
// experiments/P1_probe_dark). The flag lives in the probe's cell 0 record (.z of the moments); per ray, .w of the
// moments holds near (bit 0) and backface (bit 1).
@compute @workgroup_size(64)
fn probe_valid(@builtin(global_invocation_id) gid: vec3u) {
    let p = gid.x;
    if (p >= lat_count() + PP.part_probes) { return; }
    let at = 2u * (p * NCELL) + 1u;
    if (p >= lat_count()) {   // part and volume probes skip their own object, so never sit inside it: always valid
        let r = probe_data[at];
        probe_data[at] = vec4f(r.x, r.y, 1.0, r.w);
        return;
    }
    var cnt: array<u32, 17>;
    for (var i = 0u; i < 17u; i++) { cnt[i] = 0u; }
    var near = 0u;
    var back = 0u;
    for (var c = 0u; c < NCELL; c++) {
        let r0 = probe_data[2u * (p * NCELL + c)];
        let r1 = probe_data[2u * (p * NCELL + c) + 1u];
        let fl = u32(max(r1.w, 0.0) + 0.5);
        near += fl & 1u;
        back += (fl >> 1u) & 1u;
        if (r0.w >= 0.0) { cnt[min(u32(r0.w), 16u)] += 1u; }
    }
    var enclosed = false;
    for (var q = 0u; q < min(U.part_count, 16u); q++) {
        if ((parts[q].flags & 1u) == 0u && f32(cnt[q]) > PP.g_valid * f32(NCELL)) { enclosed = true; }
    }
    let valid = select(1.0, 0.0, enclosed || near * 2u > NCELL || back * 2u > NCELL);
    let r = probe_data[at];
    probe_data[at] = vec4f(r.x, r.y, valid, r.w);
}

// ---- sender LOD (default since 2026-10-29, ?sendercut=0 off) ---------------------------------------------------
// probe_gather uses a tree node whole once (distance from its gather centre - block_rad) > theta smax. A node every
// probe uses whole never needs its leaves' outgoing light: only the nodes the gather can reach do. Per frame, top
// down by depth, sc_select marks each node of a static part: OPEN (some probe opens it), NODE (the sender cut: every
// probe uses it whole), BELOW (under the cut or under a cut leaf: never read); leaves under an OPEN parent are LEAF.
// Movable parts too: the cut follows them, since it is re-selected every frame with the probes where they are now
// (a part's own probes skip it). Then the cut leaves take the per-leaf
// path (direct_dirs_emit_sc, env_dirs_sc, node_leaf_list), the cut nodes are lit as one Gaussian each (their record,
// V grid and iso_norm from lod.js nodeGaussians; sc_shadow's sampled-leaf shadow T; direct_dirs_nodes_sc and
// env_dirs_nodes_sc into sc_L; node_cut adds the bounce from the probes at the node and writes node_acc), and
// node_up / node_sh2 run on the OPEN nodes only.
// sc_list: [0] cut leaves, [1] cut nodes (atomic counts), [2..4] / [5..7] their indirect dispatch sizes, [8] node
// slots (host, the size of sc_X / 8), [9] (host) 1 in a frame that re-selected the cut, [10] tree slots handed out
// and [11] tree slots that did not fit (sc_mark), then the leaves at 16 and the
// nodes at 16 + G. The cut is re-selected only when a part moved (lighting.js); between, the nodes keep their slots,
// so like the leaves (direct_dirs_emit) their direct light and shadow T refresh a quarter of them a frame.

const SC_BELOW: u32 = 0u;
const SC_OPEN: u32 = 1u;
const SC_NODE: u32 = 2u;
const SC_LEAF: u32 = 3u;
const SC_SAMPLES: u32 = 8u;   // leaves sc_shadow averages per cut node

@group(0) @binding(35) var<storage, read> node_parent: array<u32>;
@group(0) @binding(41) var<storage, read_write> env_T: array<vec4f>;
@group(0) @binding(81) var<storage, read_write> sc_list: array<atomic<u32>>;
@group(0) @binding(82) var<storage, read> sc_nodes: array<u32>;   // every node by depth (LV: r0 offset, r1 count)
@group(0) @binding(84) var<storage, read> sc_ro: array<u32>;          // sc_list again, read-only: the passes dispatched from it
@group(0) @binding(85) var<storage, read_write> sc_X: array<vec4f>;   // per cut node slot x 8: 0-5 each face's SH2 transfer and exposure (sc_xfer)

// the smallest (distance - margin) from world point P to the gather centres of the probes that see part `own`
// (probe_gather's cen and brad: a room probe's 2x2x2 block centre less the block radius, else the probe itself)
fn sc_probe_dist(P: vec3f, own: u32) -> f32 {
    var best = 1e30;
    let m = select(0.0, PP.block_rad, PP.vol_info.y != 0u);   // probe_gather's margin for probes that cut around themselves
    if (PP.dims.w == 0u) {
        let nb = vec3f((PP.dims.xyz + vec3u(1u)) / 2u);
        let b = clamp(round(((P - PP.lo) / max(PP.step, vec3f(1e-6)) - 0.5) * 0.5), vec3f(0.0), nb - 1.0);
        best = length(P - (PP.lo + PP.step * (2.0 * b + 0.5))) - PP.block_rad;
    } else {
        let ci = vec2i(round((P.xy - PP.lo.xy) / max(PP.step.xy, vec2f(1e-6))));
        for (var di = -1; di <= 1; di++) {
            for (var dj = -1; dj <= 1; dj++) {
                let i = u32(clamp(ci.x + di, 0, i32(PP.dims.x) - 1));
                let j = u32(clamp(ci.y + dj, 0, i32(PP.dims.y) - 1));
                let xy = PP.lo.xy + PP.step.xy * vec2f(f32(i), f32(j));
                let gz = lat_ground(i, j);
                for (var k = 0u; k < PP.dims.z; k++) { best = min(best, length(P - vec3f(xy, gz + lat_level(k))) - m); }
            }
        }
    }
    let np = lat_count();
    for (var q = 0u; q < min(U.part_count, 32u); q++) {
        if (q == own) { continue; }
        if (is_part_probe_part(q)) {
            let pf = part_probe_frame(q);
            if (length(P - xf_point(parts[q].xf, pf.c)) - length(pf.h) - m >= best) { continue; }
            for (var k = 0u; k < PART_PROBES_PER; k++) { best = min(best, length(P - probe_pos(np + q * PART_PROBES_PER + k)) - m); }
        } else if (has_vol(q)) {
            let vf = vol_frame(q);
            let d = vec3f(vol_dims(q));
            let lp = xf_point_inv(parts[q].xf, P);
            let span = max(d - vec3f(1.0), vec3f(1.0));
            let ijk = clamp(round((lp - vf.c + vf.h) / max(2.0 * vf.h, vec3f(1e-6)) * span), vec3f(0.0), d - 1.0);
            best = min(best, length(lp - (vf.c - vf.h + 2.0 * vf.h * ijk / span)) - m);
        }
    }
    return best;
}

// node n's state, and for every state but BELOW a tree slot (sc_list[10] hands them out; past the arrays' size,
// counted in sc_list[11] for the host to grow them, the null slot)
fn sc_mark(n: u32, state: u32) {
    var t = acc_null();
    if (state != SC_BELOW) {
        let s = atomicAdd(&sc_list[10], 1u);
        if (s < t) { t = s; } else { atomicAdd(&sc_list[11], 1u); }
    }
    sc_flag[n] = state | (t << 2u);
}

@compute @workgroup_size(64)
fn sc_select(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= LV.r1) { return; }
    let n = sc_nodes[LV.r0 + gid.x];
    let nd = nodes[n];
    let q = nd.part;
    if (q >= U.part_count) { sc_mark(n, SC_BELOW); return; }   // scene nodes above the parts: the gather starts at part roots
    if (parts[q].root != n && (sc_flag[node_parent[n]] & 3u) != SC_OPEN) { sc_mark(n, SC_BELOW); return; }
    if (n < U.gaussian_count) {
        sc_mark(n, SC_LEAF);
        atomicStore(&sc_list[16u + atomicAdd(&sc_list[0], 1u)], n);
        return;
    }
    let P = xf_point(parts[q].xf, nd.mean);
    if (sc_probe_dist(P, q) <= PP.theta * nd.smax) { sc_mark(n, SC_OPEN); return; }
    let s = atomicAdd(&sc_list[1], 1u);
    if (s >= atomicLoad(&sc_list[8])) { sc_mark(n, SC_OPEN); return; }   // out of node slots: its children decide
    sc_mark(n, SC_NODE);
    atomicStore(&sc_list[16u + U.gaussian_count + s], n);
}

@compute @workgroup_size(1)
fn sc_args() {
    let nl = atomicLoad(&sc_list[0]);
    let nn = min(atomicLoad(&sc_list[1]), atomicLoad(&sc_list[8]));
    atomicStore(&sc_list[1], nn);
    atomicStore(&sc_list[2], (nl + 63u) / 64u); atomicStore(&sc_list[3], 1u); atomicStore(&sc_list[4], 1u);
    atomicStore(&sc_list[5], (nn + 63u) / 64u); atomicStore(&sc_list[6], 1u); atomicStore(&sc_list[7], 1u);
}

fn sc_mass(n: u32) -> f32 {
    let nd = nodes[n];
    return nd.kappa / sqrt(max(sym_det(nd.cinv_diag, nd.cinv_off), 1e-30));
}

// a cut node's shadow T (lamp_state[n], and its distant lights' T at env_T[2 G + n - G]): the mean over
// SC_SAMPLES leaves under it, each reached by a mass-weighted descent (fixed per node, so it holds still)
@compute @workgroup_size(64)
fn sc_shadow(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= sc_ro[1] || !sc_now(gid.x)) { return; }
    let n = sc_ro[16u + U.gaussian_count + gid.x];
    var leaf: array<u32, SC_SAMPLES>;
    for (var s = 0u; s < SC_SAMPLES; s++) {
        var m = n;
        var h = pcg_hash(n * SC_SAMPLES + s);
        for (var it = 0u; it < 64u && m >= U.gaussian_count; it++) {
            let nd = nodes[m];
            var tot = 0.0;
            for (var j = nd.child_lo; j < nd.child_hi; j++) { tot += sc_mass(kids[j]); }
            var u = to_unit(h) * tot;
            h = pcg_hash(h);
            var pick = kids[nd.child_lo];
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                pick = kids[j];
                u -= sc_mass(kids[j]);
                if (u <= 0.0) { break; }
            }
            m = pick;
        }
        leaf[s] = m;
    }
    var keys = array<u32, 4>(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    var nk = 0u;
    for (var s = 0u; s < SC_SAMPLES; s++) {
        let st = lamp_state[leaf[s]].slots;
        for (var k = 0u; k < 4u; k++) {
            let id = st[k].y & 0xFFFFu;
            if ((st[k].y >> 16u) == 0u || nk >= 4u) { continue; }
            var seen = false;
            for (var j = 0u; j < nk; j++) { seen = seen || keys[j] == id; }
            if (!seen) { keys[nk] = id; nk++; }
        }
    }
    var T = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var E = vec4f(0.0);
    for (var s = 0u; s < SC_SAMPLES; s++) {
        let st = lamp_state[leaf[s]].slots;
        for (var j = 0u; j < nk; j++) {
            var t = 1.0;
            var found = false;
            for (var k = 0u; k < 4u; k++) {
                if (!found && (st[k].y & 0xFFFFu) == keys[j] && (st[k].y >> 16u) > 0u) { t = bitcast<f32>(st[k].x); found = true; }
            }
            if (!found && (st[3].y >> 16u) > 0u && (st[3].y & 0xFFFFu) == 0xFFFFu) { t = bitcast<f32>(st[3].x); }
            T[j] += t;
        }
        E += env_T[leaf[s]];
    }
    var out: LampState;
    for (var j = 0u; j < 4u; j++) {
        out.slots[j] = select(vec2u(0u, 0u), vec2u(bitcast<u32>(T[j] / f32(SC_SAMPLES)), keys[j] | (1u << 16u)), j < nk);
    }
    lamp_state[n] = out;
    env_T[2u * U.gaussian_count + (n - U.gaussian_count)] = E / f32(SC_SAMPLES);
}

@compute @workgroup_size(64)
fn node_leaf_list(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= sc_ro[0]) { return; }
    node_leaf_one(sc_ro[16u + gid.x]);
}

// per cut node slot, when the cut was re-selected (sc_list[9] = 1; the slots keep their nodes until then): each
// face's share of iso_xfer's SH2 transfer from the node's own V grid (no near field) and its exposure S (iso_init's,
// split by face); x 6 vec4: face 0 t0-3, t4-7, (t8, S), then face 1
@compute @workgroup_size(64)
fn sc_xfer(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= sc_ro[1] || sc_ro[9] == 0u) { return; }
    let n = sc_ro[16u + U.gaussian_count + i];
    let G = gaussians[n];
    let nd = nodes[n];
    let xf = parts[G.part_surface & 0xFFFFu].xf;
    let nrm = quat_matrix(G.quat)[0];
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    var t: array<f32, 20>;   // face 0: t 0-8, S at 9; face 1 at 10
    for (var m = 0u; m < 20u; m++) { t[m] = 0.0; }
    var S0 = 0.0;
    for (var cc = 0u; cc < NCELL; cc++) {
        let w = cell_dir(cc);
        let u = xf_dir_inv(xf, w);
        let a = footprint(nd, s3 * s3, u) * cell_omega(cc);
        S0 += a;
        let v = vgrid_lookup(vgrid[n], u);
        if (v <= 0.0) { continue; }
        let h = 0.5 + 0.5 * sc_face(dot(u, nrm));
        let y = sh2_basis(w);
        for (var m = 0u; m < 9u; m++) { t[m] += a * v * h * y[m]; t[10u + m] += a * v * (1.0 - h) * y[m]; }
        t[9] += a * v * h;
        t[19] += a * v * (1.0 - h);
    }
    let Smin = max(0.5 * 0.02 * S0, 1e-30);   // iso_init.wgsl ISO_EPS, per face
    for (var f = 0u; f < 2u; f++) {
        let o = 10u * f;
        sc_X[i * 8u + 3u * f] = vec4f(t[o], t[o + 1u], t[o + 2u], t[o + 3u]);
        sc_X[i * 8u + 3u * f + 1u] = vec4f(t[o + 4u], t[o + 5u], t[o + 6u], t[o + 7u]);
        sc_X[i * 8u + 3u * f + 2u] = vec4f(t[o + 8u], max(t[o + 9u], Smin), 0.0, 0.0);
    }
}

// a cut node as one sender, a two-sided flake (common.wgsl sc_face): each face of its short axis (its record's first
// axis) takes the light arriving on its side (direct_dirs_nodes_sc / env_dirs_nodes_sc: sc_X 6 and 7; the
// probes' SH2 at its centre through sc_xfer's transfer), spread over that face's exposure, and shows it toward each
// D_k by its share there, weighted like a leaf (node_leaf). One colour for the whole node averaged a wall's lit inner
// face with its unlit outer one (2026-10-29: cut nodes sent ~4% less on the room, the left wall 9%); a hard split
// (SC_FACE_K large) overshot by about as much.
@compute @workgroup_size(64)
fn node_cut(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= sc_ro[1]) { return; }
    let n = sc_ro[16u + U.gaussian_count + i];
    let G = gaussians[n];
    let nd = nodes[n];
    let own = G.part_surface & 0xFFFFu;
    let xf = parts[own].xf;
    let P = xf_point(xf, G.mu);
    let nrm = quat_matrix(G.quat)[0];
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    let pw = probe_weights(P, own);
    var Lsh: array<vec3f, 9>;
    for (var m = 0u; m < 9u; m++) { Lsh[m] = vec3f(0.0); }
    for (var k = 0u; k < 8u; k++) {
        let w = pw.w[k];
        if (w <= 0.0) { continue; }
        for (var m = 0u; m < 9u; m++) { Lsh[m] += w * probe_sh[pw.pid[k] * 9u + m].rgb; }
    }
    var col: array<vec3f, 2>;
    for (var f = 0u; f < 2u; f++) {
        let a = sc_X[i * 8u + 3u * f]; let b = sc_X[i * 8u + 3u * f + 1u]; let c = sc_X[i * 8u + 3u * f + 2u];
        let B = a.x * Lsh[0] + a.y * Lsh[1] + a.z * Lsh[2] + a.w * Lsh[3] + b.x * Lsh[4] + b.y * Lsh[5]
              + b.z * Lsh[6] + b.w * Lsh[7] + c.x * Lsh[8];
        col[f] = G.albedo * max(sc_X[i * 8u + 6u + f].rgb + max(B, vec3f(0.0)), vec3f(0.0)) / c.y;   // 6-7: the direct light (sc_L)
    }
    let dark = PP.refl.w != 0u && own == PP.refl.w - 1u;
    let t = tree_slot(n);
    if (t >= acc_null()) { return; }
    for (var k = 0u; k < 14u; k++) {
        let wo = xf_dir_inv(xf, PP.dirs[k].xyz);
        let w = footprint(nd, s3 * s3, wo) * vgrid_lookup(vgrid[n], wo);
        let h = 0.5 + 0.5 * sc_face(dot(wo, nrm));
        let L = select(col[0] * h + col[1] * (1.0 - h), vec3f(0.0), dark);
        node_acc[t * 14u + k] = vec4f(L * w, w);
    }
}

@compute @workgroup_size(64)
fn node_up_sc(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= LV.r1) { return; }
    let i = up_nodes[LV.r0 + gid.x];
    if ((sc_flag[i] & 3u) != SC_OPEN) { return; }
    node_up_one(i);
}

@compute @workgroup_size(64)
fn node_sh2_sc(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= PP.node_count || (sc_flag[gid.x] & 3u) == SC_BELOW) { return; }
    node_sh2_one(gid.x);
}

// a cut node slot refreshes its direct light and shadow T this frame (sc_shadow, direct_dirs_nodes_sc,
// env_dirs_nodes_sc): every slot when the cut was re-selected, else a quarter a frame as direct_dirs_emit does
fn sc_now(i: u32) -> bool { return sc_ro[9] != 0u || (U.iso & 2u) == 0u || i % 4u == U.frame % 4u; }
