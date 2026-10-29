// One-path specular (M45 E9 milestone 7, E6 on the GPU): shiny Gaussians take every node of a light cut as an ASG lobe
// toward the eye. Prepend common.wgsl, material_ext.wgsl and onepath.wgsl (without its own onb, which material_ext
// defines the same way); group 0 is onepath's (Params, buffers), group 1 this pass's.
//
// One workgroup and one record in `sbuf` per receiver cluster; clusters without a shiny member store an empty cut and
// return at once. spec_cut, when the lighting, a part or a material changed,
// stores E2's light cut (budget SP.K, up to SKMAX) at the members' centroid from the history B (the same sender tree
// node_B builds) and T per cut node from one walk centroid -> node centroid (theta walk, shared over the cluster);
// spec_eval, every frame (it depends on the eye), takes per member and stored node: W = A_i V_i(w) T N(-w) / r^2 (the node's
// radiance x area SH, capped at 2 pi B_max), its shape the node's moment-matched covariance projected across w
// (onepath_spec.bin), and the specular share k 4 pi p_asg damp toward the eye (material_ext asg_phase, iso_damp).
// The display gets B + k (alb iso S - (B - Le)): the matte part keeps the gather's direction-independent B, only the
// shiny share k is replaced by its view-dependent lobe; B itself (the sender radiance for bounce) stays unchanged.

const SKMAX: u32 = 512u;
const SPEC_TAIL: f32 = 4.0;

const SREC: u32 = 2u + 2u * SKMAX;              // a cluster's record: unused, cut size, nodes, T

struct SpecU {
    eye: vec4f,                  // xyz: camera position (world)
    K: u32, mcount: u32, ncl: u32, rec_off: u32,      // rec_off: first record word in sbuf
    c6_off: u32, p0: u32, p1: u32, p2: u32,           // c6_off: first covariance word (after the materials); p0: 1 =
                                                      // coverage materials' specular into colour[G + g]
}

// group 1 has two storage buffers, so the pass fits WebGPU's common limit of 10 per stage (onepath's 8 + these)
@group(1) @binding(0) var<uniform> SP: SpecU;
@group(1) @binding(1) var<storage, read_write> sbuf: array<u32>;   // materials (SPEC 2.6, 8 words each, copied in by
                                                                    // the host) | node covariances (6 f32 per node,
                                                                    // rest frame) | per cluster: unused, n, nodes, T
@group(1) @binding(2) var<storage, read_write> colour_out: array<vec4f>;   // the display colour: += the shiny change

var<workgroup> scut: array<u32, SKMAX>;
var<workgroup> sce: array<f32, SKMAX>;
var<workgroup> sn: u32;
var<workgroup> sx: vec3f;

fn mat_of(g: u32) -> Material {
    let o = 8u * min(gaussians[g].flags & 0xFFFFu, max(SP.mcount, 1u) - 1u);
    return Material(vec3f(bitcast<f32>(sbuf[o]), bitcast<f32>(sbuf[o + 1u]), bitcast<f32>(sbuf[o + 2u])), sbuf[o + 3u],
                    bitcast<f32>(sbuf[o + 4u]), sbuf[o + 5u], sbuf[o + 6u], sbuf[o + 7u]);
}

fn sh_rgb(n: u32, u: vec3f) -> vec3f {
    let x = u.x; let y = u.y; let z = u.z;
    let b = array<f32, 9>(0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z,
                          0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y));
    var out = vec3f(0.0);
    for (var c = 0u; c < 3u; c++) {
        var s = 0.0;
        let o = (n * 3u + c) * 9u;
        for (var k = 0u; k < 9u; k++) { s += Nsh[o + k] * b[k]; }
        out[c] = s;
    }
    return out;
}

// the node's covariance across w (basis lb, world) over r^2: its ASG shape seen from the receiver
fn node_sl(n: u32, lb: array<vec3f, 2>, d2: f32) -> vec3f {
    let r = rows(0u, nodes[n].part);
    let a = xf_dir_inv(r, lb[0]); let b = xf_dir_inv(r, lb[1]);
    let o = SP.c6_off + n * 6u;
    let d = vec3f(bitcast<f32>(sbuf[o]), bitcast<f32>(sbuf[o + 1u]), bitcast<f32>(sbuf[o + 2u]));
    let f = vec3f(bitcast<f32>(sbuf[o + 3u]), bitcast<f32>(sbuf[o + 4u]), bitcast<f32>(sbuf[o + 5u]));   // xy xz yz
    let Ca = vec3f(d.x * a.x + f.x * a.y + f.y * a.z, f.x * a.x + d.y * a.y + f.z * a.z, f.y * a.x + f.z * a.y + d.z * a.z);
    let Cb = vec3f(d.x * b.x + f.x * b.y + f.y * b.z, f.x * b.x + d.y * b.y + f.z * b.z, f.y * b.x + f.z * b.y + d.z * b.z);
    return vec3f(dot(a, Ca), dot(a, Cb), dot(b, Cb)) / d2;
}

@compute @workgroup_size(64)
fn spec_cut(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    if (wid.x >= SP.ncl) { return; }
    let rec = SP.rec_off + wid.x * SREC;
    let c = wid.x;
    let start = lists[P.cl_off + 2u * c];
    let count = lists[P.cl_off + 2u * c + 1u];
    let own = gpart(lists[start]);
    let all = (1u << P.part_count) - 1u;
    if (li == 0u) {
        var shiny = false;
        for (var m = 0u; m < count; m++) { if (spec_weight(mat_of(lists[start + m])) > 0.0) { shiny = true; } }
        var xs = vec3f(0.0);
        for (var m = 0u; m < count; m++) { xs += gpos(lists[start + m], 0u); }
        let x = xs / f32(count);
        sx = x;
        var n = 0u;
        let K = select(0u, min(SP.K, SKMAX), shiny);
        for (var p = 0u; p < P.part_count; p++) {
            let r = root_of(p);
            if (p == own || tree[r].cnt == 0u || n >= K) { continue; }
            scut[n] = r; sce[n] = bound_x(r, x); n += 1u;
        }
        loop {
            if (n == 0u) { break; }
            var best = -1.0; var bi = 0u;
            for (var k = 0u; k < n; k++) { if (scut[k] >= P.G && sce[k] > best) { best = sce[k]; bi = k; } }
            if (best <= 0.0) { break; }
            let nd = nodes[scut[bi]];
            var nk = 0u;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) { if (tree[kids[kk]].cnt > 0u) { nk += 1u; } }
            if (n - 1u + nk > K) { break; }
            var first = true;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) {
                let cc = kids[kk];
                if (tree[cc].cnt == 0u) { continue; }
                if (first) { scut[bi] = cc; sce[bi] = bound_x(cc, x); first = false; }
                else { scut[n] = cc; sce[n] = bound_x(cc, x); n += 1u; }
            }
        }
        sn = n;
        sbuf[rec + 1u] = n;
    }
    let n = workgroupUniformLoad(&sn);
    let x = sx;
    for (var k = li; k < n; k += 64u) {
        let nd = scut[k];
        let pn = nodes[nd].part;
        sbuf[rec + 2u + k] = nd;
        sbuf[rec + 2u + SKMAX + k] = bitcast<u32>(exp(-walk_x(x, npos(nd, 0u), all & ~(1u << own) & ~(1u << pn), 0u)));
    }
}

// every frame: the members' specular toward the eye from the stored cut
@compute @workgroup_size(64)
fn spec_eval(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    if (wid.x >= SP.ncl) { return; }
    let rec = SP.rec_off + wid.x * SREC;
    let c = wid.x;
    let n = sbuf[rec + 1u];
    if (n == 0u) { return; }
    let start = lists[P.cl_off + 2u * c];
    let count = lists[P.cl_off + 2u * c + 1u];
    if (li >= count) { return; }
    let own = gpart(lists[start]);
    let i = lists[start + li];
    let mat = mat_of(i);
    let ks = spec_weight(mat);
    if (ks <= 0.0) { return; }
    let G = gaussians[i];
    let ri = rows(0u, own);
    let Pi = xf_point(ri, mu_rest(i));
    let Rq = quat_mat(G.quat);
    let fl = mat.flake;
    let Sl = Rq * mat3x3f(vec3f(fl.x, 0.0, 0.0), vec3f(0.0, fl.y, 0.0), vec3f(0.0, 0.0, fl.z)) * transpose(Rq);
    let Sd = vec3f(Sl[0].x, Sl[1].y, Sl[2].z); let So = vec3f(Sl[1].x, Sl[2].x, Sl[2].y);
    let wo = xf_dir_inv(ri, normalize(SP.eye.xyz - Pi));
    let iso = bitcast<f32>(nodes[i].pad0);
    let detc = det_c(i);
    var S = vec3f(0.0);
    // the eye's mirror direction and the lobe's angular width (roughness, x2 for the half-vector), for the tail test
    let co = dot(Rq[0], wo);
    let wr_eye = 2.0 * abs(co) * select(-Rq[0], Rq[0], co >= 0.0) - wo;
    let lobe = 2.0 * sqrt(max(fl.y, fl.z) / max(fl.x, 1e-12));
    for (var k = 0u; k < n; k++) {
        let nd = sbuf[rec + 2u + k];
        let T = bitcast<f32>(sbuf[rec + 2u + SKMAX + k]);
        if (T < 1e-4) { continue; }
        let dv = npos(nd, 0u) - Pi;
        let d2 = max(dot(dv, dv), 1e-8);
        let ww = dv / sqrt(d2);
        let wi = xf_dir_inv(ri, ww);
        let lbw = onb(ww);
        let sl = node_sl(nd, lbw, d2);
        // tail test (E6's open item): a node more than SPEC_TAIL widths (lobe + its own angular size) from the mirror
        // direction adds nothing visible; skip its asg_phase
        let width = lobe + sqrt(max(sl.x + sl.z, 0.0));
        if (acos(clamp(dot(wi, wr_eye), -1.0, 1.0)) > SPEC_TAIL * width + 0.02) { continue; }
        let v = vgrid_lookup(vgrid[i], wi);
        if (v <= 1e-4) { continue; }
        let ai = footprint(nodes[i], detc, wi) * v;
        var L = max(sh_rgb(nd, -ww), vec3f(0.0)) / d2;
        let ll = dot(L, LUMV);
        let cap = 2.0 * PI * tree[nd].Bmax;
        if (ll > cap) { L *= cap / ll; }
        let W = ai * T * L;
        if (dot(W, LUMV) <= 0.0) { continue; }
        let lb = array<vec3f, 2>(xf_dir_inv(ri, lbw[0]), xf_dir_inv(ri, lbw[1]));
        let p = asg_phase(wi, lb, sl, Rq[0], Rq[1], Rq[2], fl, Sd, So, wo);
        let wr = mirror(wi, Rq[0]);
        let damp = iso_damp(iso, footprint(nodes[i], detc, wr) * vgrid_lookup(vgrid[i], wr));
        S += W * (4.0 * PI * p * damp);
    }
    let B = gext[i].B.xyz;
    let spec = ks * G.albedo * iso * S;
    let dif = colour_out[i].xyz - ks * (B - gext[i].Le_w.xyz);
    if (SP.p0 != 0u && (mat.flags & 32u) != 0u) {
        // reflection coverage (renderer specSplit, material flags bit 5, e.g. glass): the specular goes to colour's
        // second half, which project.wgsl splats by coverage instead of alpha
        colour_out[i] = vec4f(max(dif, vec3f(0.0)), colour_out[i].w);
        colour_out[P.G + i] = vec4f(spec, 0.0);
    } else {
        colour_out[i] = vec4f(max(dif + spec, vec3f(0.0)), colour_out[i].w);
    }
}
