// One-frame one path (M45 E9 section 10; David 2026-10-28: converge in one render frame): a deterministic gather.
// Prepend common.wgsl and onepath.wgsl (group 0 is onepath's); group 1 = DetU.
//
// One workgroup per receiver cluster: E2's light cut at budget DU.K (up to DKMAX) at the members' centroid, nothing
// sampled. T per cut node from one walk centroid -> node centroid over the parts not near the cluster (shared); the
// near parts (E3's contact rule, as gather_m) walked per member. Per member and node: a sender leaf takes E1's exact
// pair geometry (geom_x: A_i V_i x disk solid angle) times its B; an internal node takes A_i V_i(w) N(-w) / r^2 (its
// radiance x area SH from node_B), capped at 2 pi B_max x A_i V_i. B_frame = Le + alb iso sum, into Bf (w = 0), so
// history_x with n_max 1 copies it to B. A frame = DU.iters x (node_B, gather_det, history_x): each pass is one bounce.

const DKMAX: u32 = 512u;

struct DetU {
    K: u32, shared_only: u32, mode: u32, epart: u32,   // shared_only: 1 = no contact rule (near parts on the shared walk too);
                                                      // epart: bit mask of the emitter parts whose nodes always open (exact lamp)
    dc_off: u32, wc_off: u32, av_off: u32, pad: u32,  // word offsets in dbuf: cut cache, weight cache, av table / cluster flags
    gain: f32, p5: f32, p6: f32, p7: f32,             // det_tail: per-bounce gain g of the geometric tail
    ds_off: u32, p9: u32, p10: u32, p11: u32,         // word offset of the distant light per receiver slot (3 f32)
}
// mode bits: 1 = take the cut and T from dcache (a later bounce of the same frame), 2 = store them there, 4 = profile:
// skip the members' sums, 8 = profile: no walks (T = 1), 16 = store each member's per-entry weights in wcache,
// 32 = sum from them (a later bounce: only B / the node SH change, the geometry and T do not)
@group(1) @binding(0) var<uniform> DU: DetU;
// one storage buffer for the pass (WebGPU's common limit is 10 per stage: onepath's 8 + this): per cluster the cut
// cache (n, nodes[DKMAX], T[DKMAX]) | per receiver slot and cut entry the weights | the av table or cluster flags
@group(1) @binding(1) var<storage, read_write> dbuf: array<u32>;
const DREC: u32 = 1u + 2u * DKMAX;

fn oct_enc(w: vec3f) -> u32 {
    var p = w.xy / (abs(w.x) + abs(w.y) + abs(w.z));
    if (w.z < 0.0) { p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0)); }
    return pack2x16unorm(p * 0.5 + 0.5);
}
fn oct_dec(u: u32) -> vec3f {
    let p = unpack2x16unorm(u) * 2.0 - 1.0;
    var w = vec3f(p, 1.0 - abs(p.x) - abs(p.y));
    if (w.z < 0.0) { w = vec3f((1.0 - abs(w.yx)) * select(vec2f(-1.0), vec2f(1.0), w.xy >= vec2f(0.0)), w.z); }
    return normalize(w);
}

var<workgroup> dcut: array<u32, DKMAX>;
var<workgroup> dce: array<f32, DKMAX>;
var<workgroup> dT: array<f32, DKMAX>;
var<workgroup> dn: u32;
var<workgroup> dskyT: array<f32, 64>;           // the sky bins' optical depth from the cluster centroid
var<workgroup> dx: vec3f;
var<workgroup> dnear: atomic<u32>;
var<workgroup> dnear_u: u32;
var<workgroup> drc: f32;

fn det_sh_rgb(n: u32, u: vec3f) -> vec3f {
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

// A cut node large for its distance from this member (radius > NEAR_OPEN x distance, 1.0 by a sweep: the floor node next to a box's
// base) is not one aggregate: descend it here, per member, until the pieces are small for their distance or leaves
// (leaves take E1's exact pair geometry, small nodes their SH / r^2), all with the node's shared T. The aggregate's one
// direction and SH from far away made a pale, streaky patch where the box meets the floor (cut 128).
// Nodes of the emitter parts in DU.epart always open to their leaves (M48; 0 = off, the renderer default, &detlamp=exact
// on): the lamp as one SH L2 lobe draws a ring on the ceiling round it (FLIP vs leaf-exact 0.089 -> 0.064 open), but
// the lobe lands closer to the mesh (0.158 vs 0.186): the Gaussian lamp sends too much light sideways.
const NEAR_OPEN: f32 = 1.0;
fn det_force(nd: u32) -> bool { return ((DU.epart >> nodes[nd].part) & 1u) != 0u; }
fn near_node_term(i: u32, Pi: vec3f, ri: array<vec4f, 3>, detc: f32, root: u32, T: f32) -> vec3f {
    var stack: array<u32, 24>;
    var sp = 1u; stack[0] = root;
    var acc = vec3f(0.0);
    loop {
        if (sp == 0u) { break; }
        sp -= 1u;
        let nd = stack[sp];
        if (nd < P.G) {
            let g = geom_x(i, nd, 0u);
            acc += g.ai * g.om * T * gext[nd].B.xyz;
            continue;
        }
        let dv = npos(nd, 0u) - Pi;
        let d2 = max(dot(dv, dv), 1e-8);
        let d = sqrt(d2);
        if (tree[nd].rad > NEAR_OPEN * d || det_force(nd)) {
            let nn = nodes[nd];
            var room = true;
            for (var kk = nn.child_lo; kk < nn.child_hi; kk++) {
                let cc = kids[kk];
                if (tree[cc].cnt == 0u) { continue; }
                if (sp < 24u) { stack[sp] = cc; sp += 1u; } else { room = false; }
            }
            if (room) { continue; }
        }
        let ww = dv / d;
        let wi = xf_dir_inv(ri, ww);
        let v = vgrid_lookup(vgrid[i], wi);
        if (v <= 1e-4) { continue; }
        var L = max(det_sh_rgb(nd, -ww), vec3f(0.0)) / d2;
        let ll = dot(L, LUMV);
        let cap = 2.0 * PI * tree[nd].Bmax;
        if (ll > cap) { L *= cap / ll; }
        acc += footprint(nodes[i], detc, wi) * v * T * L;
    }
    return acc;
}

// a later bounce of the frame: the stored weights times this bounce's sender radiance
fn member_phi_cached(slot: u32, n: u32) -> vec3f {
    var phi = vec3f(0.0);
    for (var k = 0u; k < n; k++) {
        let o = (slot * min(DU.K, DKMAX) + k) * 3u;
        let w = bitcast<f32>(dbuf[DU.wc_off + o]);
        if (w < 0.0) {                                // a near node: descend it again with this bounce's B
            let i = lists[slot];
            let own = gpart(i);
            let ri = rows(0u, own);
            phi += near_node_term(i, xf_point(ri, mu_rest(i)), ri, det_c(i), dcut[k], exp(-dT[k]));
            continue;
        }
        if (w <= 0.0) { continue; }
        let nd = dcut[k];
        if (nd < P.G) { phi += w * gext[nd].B.xyz; continue; }
        let id2 = bitcast<f32>(dbuf[DU.wc_off + o + 1u]);
        var L = max(det_sh_rgb(nd, oct_dec(dbuf[DU.wc_off + o + 2u])), vec3f(0.0)) * id2;
        let ll = dot(L, LUMV);
        let cap = 2.0 * PI * tree[nd].Bmax;
        if (ll > cap) { L *= cap / ll; }
        phi += w * L;
    }
    return phi;
}

fn member_phi(i: u32, n: u32, near: u32, own: u32, slot: u32) -> vec3f {
    let store = (DU.mode & 16u) != 0u;
    let ri = rows(0u, own);
    let Pi = xf_point(ri, mu_rest(i));
    let detc = det_c(i);
    var phi = vec3f(0.0);
    for (var k = 0u; k < n; k++) {
        let nd = dcut[k];
        let pn = nodes[nd].part;
        var tau = dT[k];
        let mine = near & ~(1u << own) & ~(1u << pn);
        if (mine != 0u) { tau += walk_x(Pi, npos(nd, 0u), mine, 0u); }
        let T = exp(-tau);
        let o = (slot * min(DU.K, DKMAX) + k) * 3u;
        if (store) { dbuf[DU.wc_off + o] = 0u; }
        if (T < 1e-4) { continue; }
        if (nd < P.G) {
            let g = geom_x(i, nd, 0u);
            phi += g.ai * g.om * T * gext[nd].B.xyz;
            if (store) { dbuf[DU.wc_off + o] = bitcast<u32>(g.ai * g.om * T); }
            continue;
        }
        let dv = npos(nd, 0u) - Pi;
        let d2 = max(dot(dv, dv), 1e-8);
        if (tree[nd].rad > NEAR_OPEN * sqrt(d2) || det_force(nd)) {
            phi += near_node_term(i, Pi, ri, detc, nd, T);
            if (store) { dbuf[DU.wc_off + o] = bitcast<u32>(-1.0); }   // marked: later bounces descend again
            continue;
        }
        let ww = dv / sqrt(d2);
        let wi = xf_dir_inv(ri, ww);
        let v = vgrid_lookup(vgrid[i], wi);
        if (v <= 1e-4) { continue; }
        let ai = footprint(nodes[i], detc, wi) * v;
        var L = max(det_sh_rgb(nd, -ww), vec3f(0.0)) / d2;
        let ll = dot(L, LUMV);
        let cap = 2.0 * PI * tree[nd].Bmax;
        if (ll > cap) { L *= cap / ll; }
        phi += ai * T * L;
        if (store) { dbuf[DU.wc_off + o] = bitcast<u32>(ai * T); dbuf[DU.wc_off + o + 1u] = bitcast<u32>(1.0 / d2); dbuf[DU.wc_off + o + 2u] = oct_enc(-ww); }
    }
    return phi;
}

@compute @workgroup_size(64)
fn gather_det(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let c = wid.x;
    if (c >= P.ncl) { return; }
    let start = lists[P.cl_off + 2u * c];
    let count = lists[P.cl_off + 2u * c + 1u];
    let own = gpart(lists[start]);
    let all = (1u << P.part_count) - 1u;
    let rec = c * DREC;
    let cached = (DU.mode & 1u) != 0u;
    if (li == 0u) {
        var xs = vec3f(0.0);
        for (var m = 0u; m < count; m++) { xs += gpos(lists[start + m], 0u); }
        let x = xs / f32(count);
        dx = x;
        var rc = 0.0;
        for (var m = 0u; m < count; m++) { rc = max(rc, length(gpos(lists[start + m], 0u) - x)); }
        drc = rc;
        atomicStore(&dnear, 0u);
        var n = 0u;
        let K = select(min(DU.K, DKMAX), 0u, cached);
        for (var p = 0u; p < P.part_count; p++) {
            let r = root_of(p);
            if (p == own || tree[r].cnt == 0u || n >= K) { continue; }
            dcut[n] = r; dce[n] = bound_x(r, x); n += 1u;
        }
        loop {
            var best = -1.0; var bi = 0u;
            for (var k = 0u; k < n; k++) { if (dcut[k] >= P.G && dce[k] > best) { best = dce[k]; bi = k; } }
            if (best <= 0.0) { break; }
            let nd = nodes[dcut[bi]];
            var nk = 0u;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) { if (tree[kids[kk]].cnt > 0u) { nk += 1u; } }
            if (n - 1u + nk > K) { break; }
            var first = true;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) {
                let cc = kids[kk];
                if (tree[cc].cnt == 0u) { continue; }
                if (first) { dcut[bi] = cc; dce[bi] = bound_x(cc, x); first = false; }
                else { dcut[n] = cc; dce[n] = bound_x(cc, x); n += 1u; }
            }
        }
        if (cached) { n = dbuf[DU.dc_off + rec]; }
        dn = n;
    }
    workgroupBarrier();
    let x = dx;
    if (li < P.part_count && li != own && DU.shared_only == 0u) {
        if (part_near(li, x, drc + CONTACT)) { atomicOr(&dnear, 1u << li); }
    }
    workgroupBarrier();
    if (li == 0u) { dnear_u = atomicLoad(&dnear); }
    let near = workgroupUniformLoad(&dnear_u);
    let n = workgroupUniformLoad(&dn);
    for (var k = li; k < n; k += 64u) {
        if (cached) {
            dcut[k] = dbuf[DU.dc_off + rec + 1u + k]; dT[k] = bitcast<f32>(dbuf[DU.dc_off + rec + 1u + DKMAX + k]);
            continue;
        }
        let nd = dcut[k];
        let pn = nodes[nd].part;
        var t = 0.0;
        if ((DU.mode & 8u) == 0u) { t = walk_x(x, npos(nd, 0u), all & ~(1u << own) & ~(1u << pn) & ~near, 0u); }
        dT[k] = t;
        if ((DU.mode & 2u) != 0u) { dbuf[DU.dc_off + rec + 1u + k] = nd; dbuf[DU.dc_off + rec + 1u + DKMAX + k] = bitcast<u32>(t); }
    }
    if (li == 0u && (DU.mode & 2u) != 0u) { dbuf[DU.dc_off + rec] = n; }
    // sky and sun (distant senders, M45 E5): they do not depend on B, so the frame's first bounce (no cached cut)
    // gathers them once per receiver into dbuf and the later bounces add the stored value. Sky: P.envu.x fixed bins,
    // one walk per bin from the centroid shared by the cluster; sun: one centre ray per receiver
    let distant = (P.envu.y + P.envu.z) > 0u;
    if (distant && !cached && li < min(P.envu.x, 64u) && P.envu.y > 0u) {
        let w = P.bins[li].xyz;
        dskyT[li] = walk_x(x, x + 1000.0 * w, all & ~(1u << own), 0u);
    }
    workgroupBarrier();
    if (li >= count || (DU.mode & 4u) != 0u) { return; }
    let i = lists[start + li];
    var phi: vec3f;
    if ((DU.mode & 32u) != 0u) { phi = member_phi_cached(start + li, n); }
    else { phi = member_phi(i, n, near, own, start + li); }
    if (distant) {
        let o = DU.ds_off + (start + li) * 3u;
        var dl = vec3f(0.0);
        if (cached) {
            dl = vec3f(bitcast<f32>(dbuf[o]), bitcast<f32>(dbuf[o + 1u]), bitcast<f32>(dbuf[o + 2u]));
        } else {
            let ri = rows(0u, own);
            let detc = det_c(i);
            if (P.envu.y > 0u) {
                for (var b = 0u; b < min(P.envu.x, 64u); b++) {
                    let wi = xf_dir_inv(ri, P.bins[b].xyz);
                    let v = vgrid_lookup(vgrid[i], wi);
                    if (v <= 1e-4) { continue; }
                    dl += footprint(nodes[i], detc, wi) * v * exp(-dskyT[b]) * P.brad[b].xyz;
                }
            }
            if (P.envu.z > 0u) {
                let wi = xf_dir_inv(ri, P.sun.xyz);
                let v = vgrid_lookup(vgrid[i], wi);
                if (v > 1e-4) {
                    let A = xf_point(ri, mu_rest(i));
                    dl += footprint(nodes[i], detc, wi) * v * exp(-walk_x(A, A + 1000.0 * P.sun.xyz, all & ~(1u << own), 0u)) * P.sunE.xyz;
                }
            }
            dbuf[o] = bitcast<u32>(dl.x); dbuf[o + 1u] = bitcast<u32>(dl.y); dbuf[o + 2u] = bitcast<u32>(dl.z);
        }
        phi += dl;
    }
    let iso = bitcast<f32>(nodes[i].pad0);
    gext[i].Bf = vec4f(gext[i].Le_w.xyz + gaussians[i].albedo * iso * phi, 0.0);
}

// ---- binned variant (gather_hdet): per receiver cost independent of the cut size --------------------------------------
// The cut entries far from the cluster (beyond NEAR_F x its radius + the node's radius) are summed per cluster into the
// 8 x 16 V-grid cells of the cluster's part (rest frame) as radiance x solid angle; each member then takes
// sum_c L_c A_i V_i(w_c) from a baked table (onepath_av.bin: A V at the cell centres, u8 x a per-Gaussian scale). Near
// entries keep the exact per-member term. One cluster-level pass over the cut, 128 MACs per member.


const NEAR_F: f32 = 2.0;
var<workgroup> hcell: array<u32, DKMAX>;                       // cut entry -> cell (128 = near: per member)
var<workgroup> hL: array<vec3f, DKMAX>;                        // its radiance x solid angle seen from the centroid
var<workgroup> hcellL: array<vec3f, 128>;

fn cell_of(w: vec3f) -> u32 {
    let el = asin(clamp(w.z, -1.0, 1.0));
    let r = u32(clamp(floor((0.5 * PI - el) / (PI / 8.0)), 0.0, 7.0));
    let az = atan2(w.y, w.x);
    let c = u32(clamp(floor((az + PI) / (2.0 * PI) * 16.0), 0.0, 15.0));
    return r * 16u + c;
}

fn av_at(g: u32, c: u32) -> f32 {
    let wd = dbuf[DU.av_off + P.G + g * 32u + c / 4u];
    return f32((wd >> (8u * (c % 4u))) & 255u) / 255.0 * bitcast<f32>(dbuf[DU.av_off + g]);
}

@compute @workgroup_size(64)
fn gather_hdet(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let c = wid.x;
    if (c >= P.ncl) { return; }
    let start = lists[P.cl_off + 2u * c];
    let count = lists[P.cl_off + 2u * c + 1u];
    let own = gpart(lists[start]);
    let all = (1u << P.part_count) - 1u;
    if (li == 0u) {
        var xs = vec3f(0.0);
        for (var m = 0u; m < count; m++) { xs += gpos(lists[start + m], 0u); }
        let x = xs / f32(count);
        dx = x;
        var rc = 0.0;
        for (var m = 0u; m < count; m++) { rc = max(rc, length(gpos(lists[start + m], 0u) - x)); }
        drc = rc;
        var n = 0u;
        let K = min(DU.K, DKMAX);
        for (var p = 0u; p < P.part_count; p++) {
            let r = root_of(p);
            if (p == own || tree[r].cnt == 0u || n >= K) { continue; }
            dcut[n] = r; dce[n] = bound_x(r, x); n += 1u;
        }
        loop {
            var best = -1.0; var bi = 0u;
            for (var k = 0u; k < n; k++) { if (dcut[k] >= P.G && dce[k] > best) { best = dce[k]; bi = k; } }
            if (best <= 0.0) { break; }
            let nd = nodes[dcut[bi]];
            var nk = 0u;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) { if (tree[kids[kk]].cnt > 0u) { nk += 1u; } }
            if (n - 1u + nk > K) { break; }
            var first = true;
            for (var kk = nd.child_lo; kk < nd.child_hi; kk++) {
                let cc = kids[kk];
                if (tree[cc].cnt == 0u) { continue; }
                if (first) { dcut[bi] = cc; dce[bi] = bound_x(cc, x); first = false; }
                else { dcut[n] = cc; dce[n] = bound_x(cc, x); n += 1u; }
            }
        }
        dn = n;
    }
    let n = workgroupUniformLoad(&dn);
    let x = dx; let rc = drc;
    let ri = rows(0u, own);
    // per cut entry: T (shared), cell, radiance x solid angle from the centroid
    for (var k = li; k < n; k += 64u) {
        let nd = dcut[k];
        let pn = nodes[nd].part;
        let cen = npos(nd, 0u);
        let T = exp(-walk_x(x, cen, all & ~(1u << own) & ~(1u << pn), 0u));
        dT[k] = T;
        let dv = cen - x;
        let d2 = max(dot(dv, dv), 1e-8); let d = sqrt(d2);
        let ww = dv / d;
        var L = vec3f(0.0);
        if (nd < P.G) {
            let rj = rows(0u, pn);
            let wj = xf_dir_inv(rj, ww);
            let vj = vgrid_lookup(vgrid[nd], -wj);
            var aj = footprint(nodes[nd], det_c(nd), wj) * vj * gext[nd].Le_w.w;
            if ((gext[nd].Le_w.x + gext[nd].Le_w.y + gext[nd].Le_w.z) > 0.0) {
                let nrm = quat_mat(gaussians[nd].quat)[0];
                aj = lamp_area(nd, aj, nrm, -dot(nrm, wj), vj * gext[nd].Le_w.w);
            }
            L = 2.0 * PI * (1.0 - d / sqrt(d2 + max(aj, 0.0) / PI)) * gext[nd].B.xyz;
        } else {
            L = max(det_sh_rgb(nd, -ww), vec3f(0.0)) / d2;
            let ll = dot(L, LUMV);
            let cap = 2.0 * PI * tree[nd].Bmax;
            if (ll > cap) { L *= cap / ll; }
        }
        hL[k] = L * T;
        let near = d < NEAR_F * rc + 0.5 * tree[nd].rad;
        hcell[k] = select(cell_of(xf_dir_inv(ri, ww)), 128u, near);
    }
    workgroupBarrier();
    for (var cl = li; cl < 128u; cl += 64u) {
        var s = vec3f(0.0);
        for (var k = 0u; k < n; k++) { if (hcell[k] == cl) { s += hL[k]; } }
        hcellL[cl] = s;
    }
    workgroupBarrier();
    if (li >= count) { return; }
    let i = lists[start + li];
    let Pi = xf_point(ri, mu_rest(i));
    let detc = det_c(i);
    var phi = vec3f(0.0);
    for (var cl = 0u; cl < 128u; cl++) {
        let L = hcellL[cl];
        if (L.x + L.y + L.z <= 0.0) { continue; }
        phi += L * av_at(i, cl);
    }
    for (var k = 0u; k < n; k++) {                // near entries: the exact per-member term (as gather_det)
        if (hcell[k] != 128u) { continue; }
        let nd = dcut[k];
        let T = dT[k];
        if (T < 1e-4) { continue; }
        if (nd < P.G) {
            let g = geom_x(i, nd, 0u);
            phi += g.ai * g.om * T * gext[nd].B.xyz;
            continue;
        }
        let dv = npos(nd, 0u) - Pi;
        let d2 = max(dot(dv, dv), 1e-8);
        let ww = dv / sqrt(d2);
        let wi = xf_dir_inv(ri, ww);
        let v = vgrid_lookup(vgrid[i], wi);
        if (v <= 1e-4) { continue; }
        var L = max(det_sh_rgb(nd, -ww), vec3f(0.0)) / d2;
        let ll = dot(L, LUMV);
        let cap = 2.0 * PI * tree[nd].Bmax;
        if (ll > cap) { L *= cap / ll; }
        phi += footprint(nodes[i], detc, wi) * v * T * L;
    }
    let iso = bitcast<f32>(nodes[i].pad0);
    gext[i].Bf = vec4f(gext[i].Le_w.xyz + gaussians[i].albedo * iso * phi, 0.0);
}

// ---- the geometric tail (one frame: a few bounces, then the rest of the series) -------------------------------------
// det_save before the frame's last bounce keeps B in Bc (free in this mode); det_tail after it adds
// (B - B_prev) g / (1 - g): each further bounce would add ~g times the last one (E1: g ~ 0.45)
@compute @workgroup_size(64)
fn det_save(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.R) { return; }
    let i = lists[gid.x];
    gext[i].Bc = vec4f(gext[i].B.xyz, gext[i].Bc.w);
}

@compute @workgroup_size(64)
fn det_tail(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.R) { return; }
    let i = lists[gid.x];
    let d = max(gext[i].B.xyz - gext[i].Bc.xyz, vec3f(0.0));
    gext[i].B = vec4f(gext[i].B.xyz + d * DU.gain / (1.0 - DU.gain), gext[i].B.w);
}
