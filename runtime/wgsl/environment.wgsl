// Environment light (runtime/SPEC.md 2.9): a spherical-harmonic sky plus distant lights (the sun) as anisotropic
// spherical Gaussians (ASG) at infinity. Prepend common.wgsl and material_ext.wgsl.
//
//   env_shadow   per Gaussian, only when a distant light or a movable part moved: one deterministic walk toward each
//                shadow-casting distant light over the other parts -> env_T (no sampling, so nothing to settle).
//                Bounded (hill: 1.17M Gaussians in one dispatch hung the Mac): environment.js runs it in slices of
//                ESH.count Gaussians from ESH.first, one slice a frame; each walk visits at most ENV_WALK_STEPS nodes
//                and stops once the light is fully blocked. ESH.mode 1 (a movable part moved, the sun did not): static
//                receivers keep their static-occluder T (env_T[count + g]) and walk the movable parts only. ESH.mode 2
//                (with sun_map.wgsl, the default): static occluders only, into both halves; the sun maps multiply in
//                the movable parts every frame.
//   env_direct   per Gaussian, every frame, after shade_emit: the distant lights' direct light added to colour
//   env_dirs     the same toward the probes: added to lo_dir for the Gaussians direct_dirs(_emit) refreshed this frame
//
// The sky reaches Gaussians through the probes: bounce.wgsl probe_gather returns sky_radiance() for what a ray
// leaves unblocked. The display pass draws sky_radiance() + distant_radiance() behind the splats.
// M39 transfer only (FrameUniforms.iso bit 0): a distant light adds alb iso_norm E T V(w) A(w), a light cut node with
// no 1/d^2. Toward the eye, shiny materials add the light's highlight with its own shape (material_ext.wgsl asg_phase,
// the light an isotropic ASG of angular std sigma), split into colour's second half with reflection coverage (bit 4).

struct EnvLight {
    dir_std: vec4f,      // xyz unit direction toward the light (world), w angular standard deviation (rad)
    irr: vec4f,          // rgb irradiance on a plane facing the light, w: 1 = casts shadows
}

struct Env {
    sh: array<vec4f, 16>,            // sky radiance, real SH bands 0-3 (SPEC 2.9 basis), rgb in xyz
    lights: array<EnvLight, 4>,
    info: vec4u,                     // x distant lights, y 1 = sky on, z -, w -
}

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> vgrid: array<VGrid>;
@group(0) @binding(4) var<storage, read> parts: array<Part>;
@group(0) @binding(7) var<storage, read> materials: array<Material>;
@group(0) @binding(8) var<storage, read_write> colour: array<vec4f>;
@group(0) @binding(9) var<storage, read> kids: array<u32>;
@group(0) @binding(13) var<storage, read_write> lo_dir: array<vec4f>;
@group(0) @binding(40) var<uniform> ENV: Env;
@group(0) @binding(41) var<storage, read_write> env_T: array<vec4f>;   // [0, count) T; [count, 2 count) static occluders' T;
                                                                       // ?lod: [2 count, + nodes) drawn nodes' mean T (lod.wgsl)

struct EnvShadow {
    first: u32,                      // the slice's first Gaussian
    count: u32,                      // its length
    mode: u32,                       // 0 every occluder; 1 static receivers walk movable occluders only
    rays: u32,                       // bundle rays per receiver and light (0 or 1: the centre ray)
    static_mask: array<vec4u, 4>,    // bit p: part p is static (SPEC 2.4 mobility); parts >= 512 count as movable
}
@group(0) @binding(42) var<uniform> ESH: EnvShadow;

const ENV_WALK_STEPS: u32 = 2048u;   // node visits per receiver and light, over all parts
const ENV_TAU_MAX: f32 = 12.0;       // T < 6e-6: blocked, stop walking

fn part_static(p: u32) -> bool {
    if (p >= 512u) { return false; }
    return ((ESH.static_mask[p / 128u][(p / 32u) % 4u] >> (p % 32u)) & 1u) != 0u;
}

// optical depth along w over one part's subtree; `steps` is the node-visit budget left, shared across parts
// cut > 0 (motion blur, a moving part): nodes of bounding radius <= cut are used whole, each keeping its rest record's
// coverage across its stretched footprint (light_shadow.wgsl lsm_splat's ?blurcut rule) instead of its density
fn walk_tau(root: u32, P: vec3f, w: vec3f, dist: f32, theta: f32, cut: f32, steps: ptr<function, u32>) -> f32 {
    var stack: array<u32, STACK>;
    stack[0] = root;
    var sp = 1u;
    var tau = 0.0;
    let th2 = theta * theta;
    while (sp > 0u && *steps > 0u && tau < ENV_TAU_MAX) {
        *steps -= 1u;
        sp -= 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        let d = nd.mean - P;
        let tt = clamp(dot(d, w), 0.0, dist);
        let e = d - tt * w;
        if (dot(e, e) > nd.radius * nd.radius) { continue; }
        var use_it = ni < U.gaussian_count;
        if (!use_it && theta > 0.0) { use_it = dot(d, d) > th2 * nd.smax * nd.smax; }
        if (!use_it && cut > 0.0) { use_it = nodes_rest[ni].radius <= cut; }
        if (use_it) {
            var f = 1.0;
            if (cut > 0.0) { f = coverage_scale(nd, nodes_rest[ni], w); }
            tau += f * nd.kappa * dirm(nd, w) * seg_geom(nd, P, w, 0.0, dist);
        } else {
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                if (sp < STACK) { stack[sp] = kids[j]; sp += 1u; }
            }
        }
    }
    return tau;
}

// low-discrepancy standard normals for bundle ray r of receiver g (fixed per receiver: shadows stay deterministic)
fn bundle_normals(g: u32, r: u32) -> vec4f {
    let h = pcg_hash((g * 0x9E3779B9u) ^ ((r + 1u) * 0x85EBCA6Bu));
    let u = vec4f(f32(h & 0xFFu), f32((h >> 8u) & 0xFFu), f32((h >> 16u) & 0xFFu), f32(h >> 24u)) / 256.0 + 1.0 / 512.0;
    let r1 = sqrt(-2.0 * log(u.x)); let r2 = sqrt(-2.0 * log(u.z));
    return vec4f(r1 * cos(6.2831853 * u.y), r1 * sin(6.2831853 * u.y), r2 * cos(6.2831853 * u.w), r2 * sin(6.2831853 * u.w));
}

// motion-blur cut (lighting.js ?blurcut, env_shadow only): rest records and (shutter, k)
@group(0) @binding(55) var<uniform> BCE: vec4f;
@group(0) @binding(56) var<storage, read> nodes_rest: array<Node>;

// a moving part's cut size k |d| (its smear at the root's centre); 0 = still or off
fn blur_cut(p: u32) -> f32 {
    if (BCE.y <= 0.0 || BCE.x <= 0.0 || (parts[p].flags & 2u) == 0u) { return 0.0; }
    let P = parts[p];
    let x = nodes_rest[P.root].mean;
    return BCE.y * BCE.x * length(xf_point(P.xf, x) - xf_point(P.xf_prev, x));
}

// the factor that turns the stretched record's (mass-kept) optical depth along w into the coverage-kept one: peak
// opacity alpha' = alpha_rest x (rest / stretched footprint across w), as a ratio of peak optical depths
fn coverage_scale(ns: Node, nr: Node, w: vec3f) -> f32 {
    let qs = max(dot(w, sym_mul(ns.cinv_diag, ns.cinv_off, w)), 1e-30);   // 1 / depth variance
    let qr = max(dot(w, sym_mul(nr.cinv_diag, nr.cinv_off, w)), 1e-30);
    let ts = ns.kappa * dirm(ns, w) * sqrt(2.0 * PI / qs);                 // peak column optical depth
    let tr = nr.kappa * dirm(nr, w) * sqrt(2.0 * PI / qr);
    // footprint across w ~ sqrt(det C (w^T C^-1 w)), det C = 1 / det C^-1
    let ds = determinant(sym3m(ns.cinv_diag, ns.cinv_off));
    let dr = determinant(sym3m(nr.cinv_diag, nr.cinv_off));
    let area = sqrt(max(ds * qr, 0.0) / max(dr * qs, 1e-30));             // rest / stretched
    let a = min((1.0 - exp(-tr)) * min(area, 1.0), 0.999);
    return min(-log(1.0 - a) / max(ts, 1e-30), 1.0);
}

fn sym3m(dg: vec3f, off: vec3f) -> mat3x3f {
    return mat3x3f(vec3f(dg.x, off.x, off.y), vec3f(off.x, dg.y, off.z), vec3f(off.y, off.z, dg.z));
}

// optical depth from world point P toward w over the other parts (which: 0 all, 1 movable only, 2 static only)
fn env_tau(P: vec3f, w: vec3f, own: u32, which: u32, tau_static: ptr<function, f32>) -> f32 {
    var tau = 0.0;
    var steps = ENV_WALK_STEPS;
    for (var p = 0u; p < U.part_count; p++) {
        if (p == own) { continue; }
        let ps = part_static(p);
        if (which == 1u && ps) { continue; }
        if (which == 2u && !ps) { continue; }   // static occluders only (sun_map.wgsl draws the movable ones)
        let xf = parts[p].xf;
        let b = parts[p].bound;
        // the ray only meets this part within its bound: walk up to the far side of it
        let c = xf_point(xf, b.xyz) - P;
        let tc = dot(c, w);
        if (tc + b.w < 0.0 || dot(c, c) - tc * tc > b.w * b.w) { continue; }
        let t = walk_tau(parts[p].root, xf_point_inv(xf, P), xf_dir_inv(xf, w), tc + b.w, U.theta, blur_cut(p), &steps);
        tau += t;
        if (ps) { *tau_static += t; }
        if (tau >= ENV_TAU_MAX || steps == 0u) { break; }
    }
    return tau;
}

@compute @workgroup_size(64)
fn env_shadow(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= ESH.count) { return; }
    let g = ESH.first + gid.x;
    let n = U.gaussian_count;
    if (g >= n) { return; }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let xf = parts[own].xf;
    let P = xf_point(xf, G.mu);
    let Rq = quat_matrix(G.quat);
    let nr = max(ESH.rays, 1u);
    // a static receiver's shadow from static occluders only changes with the light: mode 1 reuses it
    let keep = ESH.mode == 1u && part_static(own);
    var T = vec4f(1.0);
    var Ts = select(vec4f(1.0), env_T[n + g], keep);
    for (var k = 0u; k < min(ENV.info.x, 4u); k++) {
        let L = ENV.lights[k];
        if (L.irr.w < 0.5) { continue; }
        let w0 = L.dir_std.xyz;
        let bw = onb(w0);
        var tsum = 0.0;
        var tssum = 0.0;
        for (var r = 0u; r < nr; r++) {
            var Pr = P;
            var w = w0;
            if (nr > 1u) {
                // a point drawn from the receiver Gaussian and a direction within the light's angular spread
                let z = bundle_normals(g, r);
                let z3 = bundle_normals(g, r + 97u).xy;
                let sc = vec3f(G.scale.x, G.scale.y, G.scale.z) * vec3f(z.x, z.y, z.z) * 0.8;   // a local copy (Metal: no matrix x packed vector)
                Pr = xf_point(xf, G.mu + Rq * sc);
                w = normalize(w0 + L.dir_std.w * (z.w * bw[0] + z3.x * bw[1]));
            }
            var ts = 0.0;
            let tau = env_tau(Pr, w, own, select(select(0u, 1u, keep), 2u, ESH.mode == 2u), &ts);
            tsum += exp(-tau);
            tssum += exp(-ts);
        }
        if (!keep) { Ts[k] = tssum / f32(nr); }
        T[k] = select(tsum / f32(nr), Ts[k] * tsum / f32(nr), keep);
    }
    env_T[g] = T;
    env_T[n + g] = Ts;
}

// direct light from the distant lights on Gaussian g (before albedo and iso_norm): sum E T V(w) A(w)
fn distant_catch(g: u32) -> vec3f { return distant_catch_T(g, env_T[g]); }

fn distant_catch_T(g: u32, T: vec4f) -> vec3f {
    let G = gaussians[g];
    let nd = nodes[g];
    let xf = parts[G.part_surface & 0xFFFFu].xf;
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    var acc = vec3f(0.0);
    for (var k = 0u; k < min(ENV.info.x, 4u); k++) {
        let L = ENV.lights[k];
        let wi = xf_dir_inv(xf, L.dir_std.xyz);
        let v = vgrid_lookup(vgrid[g], wi);
        if (v <= 1e-7) { continue; }
        acc += L.irr.rgb * (T[k] * v * footprint(nd, s3 * s3, wi));
    }
    return acc;
}

fn env_active() -> bool { return ENV.info.x > 0u && (U.iso & 1u) != 0u; }

@compute @workgroup_size(64)
fn env_direct(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= U.gaussian_count || !env_active()) { return; }
    env_add(gid.x, env_T[gid.x], gid.x, U.gaussian_count + gid.x);
}

// screen-space LOD on the cut (?lodlight=2, lod.wgsl lod_list): the drawn leaves (and the display fill's sources)
@group(0) @binding(61) var<storage, read> lodflag: array<u32>;
@group(0) @binding(62) var<storage, read> lodlist: array<u32>;
@compute @workgroup_size(64)
fn env_direct_list(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= lodlist[0] || !env_active()) { return; }
    let g = lodlist[16u + gid.x];
    env_add(g, env_T[g], g, U.gaussian_count + g);
}

// drawn internal tree nodes (?lodlight=1: by flag; 2: the node list), after emissive.wgsl shade_emit_nodes: their
// own geometry with the mass-weighted mean of their leaves' T (lod.wgsl lod_colour, env_T[2 G + k]), into the shown
// colour at 2 G + k (highlights folded in: nodes have no coverage half)
@compute @workgroup_size(64)
fn env_direct_nodes(@builtin(global_invocation_id) gid: vec3u) {
    let g = U.gaussian_count + gid.x;
    if (g >= arrayLength(&lodflag) || lodflag[g] != 2u || !env_active()) { return; }
    env_node(g);
}

@compute @workgroup_size(64)
fn env_direct_nodes_list(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= lodlist[1] || !env_active()) { return; }
    env_node(lodlist[16u + U.gaussian_count + gid.x]);
}

fn env_node(g: u32) {
    let k = g - U.gaussian_count;
    env_add(g, env_T[2u * U.gaussian_count + k], 2u * U.gaussian_count + k, NO_SPEC_SLOT);
}

const NO_SPEC_SLOT: u32 = 0xFFFFFFFFu;

// the distant lights' direct light on Gaussian (or tree node) g with shadow T, added to colour[ci]; highlights to
// colour[si] with reflection coverage on (NO_SPEC_SLOT: into colour[ci])
fn env_add(g: u32, T: vec4f, ci: u32, si: u32) {
    let G = gaussians[g];
    let mat = materials[min(G.flags & 0xFFFFu, max(U.material_count, 1u) - 1u)];
    if ((mat.flags & 1u) != 0u) { return; }   // emitters show their own emission
    let ks = spec_weight(mat);
    let nd = nodes[g];
    let inorm = iso_norm(nd);
    if (ks <= 0.0) {
        colour[ci] = vec4f(colour[ci].rgb + G.albedo * distant_catch_T(g, T) * inorm, colour[ci].w);
        return;
    }
    // shiny: the eye sees (1 - k) of the matte light plus each light's highlight, its shape the light's ASG
    let xf = parts[G.part_surface & 0xFFFFu].xf;
    let P = xf_point(xf, G.mu);
    let wo = xf_dir_inv(xf, normalize(U.eye - P));
    let Rq = quat_matrix(G.quat);
    let Sl = Rq * mat3x3f(vec3f(mat.flake.x, 0.0, 0.0), vec3f(0.0, mat.flake.y, 0.0), vec3f(0.0, 0.0, mat.flake.z)) * transpose(Rq);
    let Sd = vec3f(Sl[0].x, Sl[1].y, Sl[2].z);
    let So = vec3f(Sl[1].x, Sl[2].x, Sl[2].y);
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    var diff = vec3f(0.0);
    var spec = vec3f(0.0);
    for (var k = 0u; k < min(ENV.info.x, 4u); k++) {
        let L = ENV.lights[k];
        let wi = xf_dir_inv(xf, L.dir_std.xyz);
        let v = vgrid_lookup(vgrid[g], wi);
        if (v <= 1e-7) { continue; }
        let light = L.irr.rgb * (T[k] * v * footprint(nd, s3 * s3, wi));
        let wr = mirror(wi, Rq[0]);
        let damp = iso_damp(inorm, footprint(nd, s3 * s3, wr) * vgrid_lookup(vgrid[g], wr));
        let s2 = L.dir_std.w * L.dir_std.w;
        let p = asg_phase(wi, onb(wi), vec3f(s2, 0.0, s2), Rq[0], Rq[1], Rq[2], mat.flake, Sd, So, wo);
        diff += light * (1.0 - ks);
        spec += light * (iso_lobe_from(ks, p, damp) - (1.0 - ks));
    }
    let a = G.albedo * inorm;
    if ((U.iso & 16u) != 0u && si != NO_SPEC_SLOT) {   // reflection coverage: the highlight in colour's second half (project.wgsl)
        colour[ci] = vec4f(colour[ci].rgb + a * diff, colour[ci].w);
        colour[si] = vec4f(colour[si].rgb + a * spec, colour[si].w);
    } else {
        colour[ci] = vec4f(colour[ci].rgb + a * (diff + spec), colour[ci].w);
    }
}

@compute @workgroup_size(64)
fn env_dirs(@builtin(global_invocation_id) gid: vec3u) {
    let g = select(gid.x, gid.x * 4u + U.frame % 4u, (U.iso & 2u) != 0u);   // the Gaussians direct_dirs refreshed
    if (g >= U.gaussian_count || !env_active()) { return; }
    env_dirs_one(g);
}

// sender LOD (default, ?sendercut=0 off; bounce.wgsl sc_select): the cut's leaves as direct_dirs_emit_sc picks them, and its
// nodes with sc_shadow's T (env_T[2 G + n - G]) into sc_L after direct_dirs_nodes_sc
@group(0) @binding(81) var<storage, read> sc_list: array<u32>;
@group(0) @binding(83) var<storage, read_write> sc_L: array<vec4f>;   // bounce.wgsl sc_X: per slot x 8, the faces' direct light at 6-7
@compute @workgroup_size(64)
fn env_dirs_sc(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= sc_list[0] || !env_active()) { return; }
    let g = sc_list[16u + gid.x];
    if ((U.iso & 2u) != 0u && g % 4u != U.frame % 4u) { return; }
    env_dirs_one(g);
}

@compute @workgroup_size(64)
fn env_dirs_nodes_sc(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= sc_list[1] || !env_active() || !(sc_list[9] != 0u || (U.iso & 2u) == 0u || gid.x % 4u == U.frame % 4u)) { return; }   // bounce.wgsl sc_now
    let n = sc_list[16u + U.gaussian_count + gid.x];
    let G = gaussians[n];
    let mat = materials[min(G.flags & 0xFFFFu, max(U.material_count, 1u) - 1u)];
    if ((mat.flags & 1u) != 0u) { return; }
    // per face of the node's short axis, before albedo and exposure (direct_dirs_nodes_sc)
    let T = env_T[2u * U.gaussian_count + (n - U.gaussian_count)];
    let nd = nodes[n];
    let xf = parts[G.part_surface & 0xFFFFu].xf;
    let nrm = quat_matrix(G.quat)[0];
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    for (var k = 0u; k < min(ENV.info.x, 4u); k++) {
        let L = ENV.lights[k];
        let wi = xf_dir_inv(xf, L.dir_std.xyz);
        let v = vgrid_lookup(vgrid[n], wi);
        if (v <= 1e-7) { continue; }
        let c = L.irr.rgb * (T[k] * v * footprint(nd, s3 * s3, wi));
        let h = 0.5 + 0.5 * sc_face(dot(wi, nrm));
        sc_L[gid.x * 8u + 6u] = vec4f(sc_L[gid.x * 8u + 6u].rgb + c * h, 0.0);
        sc_L[gid.x * 8u + 7u] = vec4f(sc_L[gid.x * 8u + 7u].rgb + c * (1.0 - h), 0.0);
    }
}

fn env_dirs_one(g: u32) {
    let G = gaussians[g];
    let mat = materials[min(G.flags & 0xFFFFu, max(U.material_count, 1u) - 1u)];
    if ((mat.flags & 1u) != 0u) { return; }
    let c = G.albedo * distant_catch(g) * iso_norm(nodes[g]);
    lo_dir[g] = vec4f(lo_dir[g].rgb + c, 0.0);   // common.wgsl LOB
}
