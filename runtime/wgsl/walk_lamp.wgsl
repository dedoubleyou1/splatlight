// Lamp segment walk (SPEC 4.1): one stratified lamp sample per Gaussian per frame, other-part optical depth by the
// finite-segment walk on each part's subtree (M33 engine stack es.walk1, theta distance rule), own part from the
// 8x16 V grid. Prepend common.wgsl.

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> kids: array<u32>;
@group(0) @binding(4) var<storage, read> vgrid: array<VGrid>;
@group(0) @binding(5) var<storage, read> parts: array<Part>;
@group(0) @binding(6) var<storage, read> lamps: array<Lamp>;
@group(0) @binding(7) var<storage, read_write> samples: array<Sample>;

struct WalkOut { tau: f32, tests: u32, evals: u32, overflow: bool }

// optical depth along P + t w, t in [0, dist], through one part's subtree (rest-frame P, w); mur = receiver centre
fn walk_part(root: u32, P: vec3f, w: vec3f, dist: f32, mur: vec3f, theta: f32, acc: ptr<function, WalkOut>) {
    var stack: array<u32, STACK>;
    stack[0] = root;
    var sp = 1u;
    let th2 = theta * theta;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        (*acc).tests += 1u;
        let d = nd.mean - P;
        let tt = clamp(dot(d, w), 0.0, dist);
        let e = d - tt * w;
        if (dot(e, e) > nd.radius * nd.radius) { continue; }
        var use_node = ni < U.gaussian_count;           // leaves are nodes 0..G-1
        if (!use_node && theta > 0.0) {
            let r = mur - nd.mean;
            use_node = dot(r, r) > th2 * nd.smax * nd.smax;
        }
        if (use_node) {
            (*acc).tau += nd.kappa * dirm(nd, w) * seg_geom(nd, P, w, 0.0, dist);
            (*acc).evals += 1u;
        } else {
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                if (sp < STACK) { stack[sp] = kids[j]; sp += 1u; } else { (*acc).overflow = true; }
            }
        }
    }
}

@compute @workgroup_size(64)
fn walk_lamp(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let oxf = parts[own].xf;
    let P = xf_point(oxf, G.mu);
    let lamp = U.frame % max(U.slot_count, 1u);          // one lamp per slot in this stub (SPEC 3.3)
    if (lamp >= U.lamp_count || (lamps[lamp].flags & 1u) == 0u) {
        samples[g] = Sample(0.0, 1.0, 0u, 0u);
        return;
    }
    let L = lamps[lamp];
    let s = (U.frame + U.stratum_step * g) % 16u;
    let u = (f32(s / 4u) + jitter(g, U.frame, 0u)) * 0.25;
    let v = (f32(s % 4u) + jitter(g, U.frame, 1u)) * 0.25;
    let y = lamp_point(L, P, u, v);
    let dist = length(y - P);
    let w = (y - P) / dist;

    var acc = WalkOut(0.0, 0u, 0u, false);
    for (var p = 0u; p < U.part_count; p++) {
        if (p == own) { continue; }
        let xf = parts[p].xf;
        let Pl = xf_point_inv(xf, P);
        walk_part(parts[p].root, Pl, xf_dir_inv(xf, w), dist, Pl, U.theta, &acc);
    }
    let v_own = vgrid_lookup(vgrid[g], xf_dir_inv(oxf, w));
    var tests = acc.tests;
    if (acc.overflow) { tests |= 0x80000000u; }
    samples[g] = Sample(acc.tau, v_own, tests, acc.evals);
}
