// World-space Gaussian motion blur for the lighting (motion blur thread, variant C; renderer/shaders/project.wgsl
// does the display). A Gaussian moving along d over the exposure, averaged over a box shutter, is to second order a
// Gaussian with covariance C + d d^T / 12 centred mid-exposure, its density scaled by sqrt(det C / det C') so its mass
// is kept. Every frame with blur on, this kernel writes that stretched copy of each record of a moving part (the
// exposure ends at this frame's pose, d from xf_prev, in the part's rest frame so the walks' transforms stay as they
// are) and an unchanged copy of every other record; the lighting kernels bind the copies. Only the records of parts
// that moved this frame or last frame are rewritten (ranges); nothing runs while the scene is still. Ŝ (extinction SGGX) keeps
// the original orientation: the smear is of where the flakes are, not of how they face. Receivers skip their own
// part (its V grid), so a part never shadows itself with its own smear.

struct BlurParams { shutter: f32, gaussian_count: u32, node_count: u32, range_count: u32 }

@group(0) @binding(0) var<uniform> BP: BlurParams;
@group(0) @binding(1) var<storage, read> parts: array<Part>;
@group(0) @binding(2) var<storage, read> g_src: array<Gaussian>;
@group(0) @binding(3) var<storage, read> n_src: array<Node>;
@group(0) @binding(4) var<storage, read_write> g_dst: array<Gaussian>;
@group(0) @binding(5) var<storage, read_write> n_dst: array<Node>;
// the record ranges to rewrite this frame (lighting.js encodeBlur): the moving parts' runs of leaves and nodes, plus
// last frame's movers (rewritten unstretched); (first, end, thread offset, -). Other records keep their copies.
@group(0) @binding(6) var<storage, read> ranges: array<vec4u>;

// rest-frame displacement of rest point x over the exposure (zero when off or still)
fn blur_disp(p: u32, x: vec3f) -> vec3f {
    if (BP.shutter <= 0.0 || p >= arrayLength(&parts) || (parts[p].flags & 2u) == 0u) { return vec3f(0.0); }
    let P = parts[p];
    let d = BP.shutter * (xf_point(P.xf, x) - xf_point(P.xf_prev, x));
    return xf_dir_inv(P.xf, d);
}

fn smat(dg: vec3f, off: vec3f) -> mat3x3f {   // off: xy xz yz
    return mat3x3f(vec3f(dg.x, off.x, off.y), vec3f(off.x, dg.y, off.z), vec3f(off.y, off.z, dg.z));
}

// Cyclic Jacobi eigen-decomposition of a symmetric 3x3: columns of V are eigenvectors, the result's values the
// eigenvalues.
fn eig3(A0: mat3x3f, V: ptr<function, mat3x3f>) -> vec3f {
    var A = A0;
    var Q = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    for (var sweep = 0; sweep < 8; sweep++) {
        for (var k = 0; k < 3; k++) {
            let pp = select(0, 1, k == 2);   // pairs (0,1) (0,2) (1,2)
            let q = select(k + 1, 2, k == 2);
            let apq = A[q][pp];
            if (abs(apq) < 1e-30) { continue; }
            let theta = 0.5 * (A[q][q] - A[pp][pp]) / apq;
            let t = sign(theta + select(0.0, 1e-30, theta == 0.0)) / (abs(theta) + sqrt(theta * theta + 1.0));
            let c = inverseSqrt(t * t + 1.0);
            let s = t * c;
            var J = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
            J[pp][pp] = c; J[q][q] = c; J[q][pp] = s; J[pp][q] = -s;
            A = transpose(J) * A * J;
            Q = Q * J;
        }
    }
    *V = Q;
    return vec3f(A[0][0], A[1][1], A[2][2]);
}

fn mat_to_quat(m: mat3x3f) -> vec4f {   // column-major rotation -> x y z w (quat_matrix's inverse)
    let tr = m[0][0] + m[1][1] + m[2][2];
    var q: vec4f;
    if (tr > 0.0) {
        let s = sqrt(tr + 1.0) * 2.0;
        q = vec4f((m[1][2] - m[2][1]) / s, (m[2][0] - m[0][2]) / s, (m[0][1] - m[1][0]) / s, 0.25 * s);
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        let s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q = vec4f(0.25 * s, (m[1][0] + m[0][1]) / s, (m[2][0] + m[0][2]) / s, (m[1][2] - m[2][1]) / s);
    } else if (m[1][1] > m[2][2]) {
        let s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q = vec4f((m[1][0] + m[0][1]) / s, 0.25 * s, (m[2][1] + m[1][2]) / s, (m[2][0] - m[0][2]) / s);
    } else {
        let s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q = vec4f((m[2][0] + m[0][2]) / s, (m[2][1] + m[1][2]) / s, 0.25 * s, (m[0][1] - m[1][0]) / s);
    }
    return normalize(q);
}

@compute @workgroup_size(64)
fn blur_records(@builtin(global_invocation_id) gid: vec3u) {
    var i = 0xFFFFFFFFu;
    for (var r = 0u; r < BP.range_count; r++) {
        let R = ranges[r];
        if (gid.x >= R.z && gid.x < R.z + (R.y - R.x)) { i = R.x + (gid.x - R.z); break; }
    }
    if (i >= BP.node_count) { return; }
    var nd = n_src[i];
    let iso = n_dst[i].pad0;   // iso_init's per-leaf norm lives in the copy the lighting binds
    let leaf = i < BP.gaussian_count;
    var G: Gaussian;
    if (leaf) { G = g_src[i]; }
    let d = blur_disp(nd.part, nd.mean);
    let dd = dot(d, d);
    if (dd > 0.0) {
        let Ci = smat(nd.cinv_diag, nd.cinv_off);
        let cd = Ci * d;
        let m = dot(d, cd);
        let Cn = Ci - mat3x3f(cd * cd.x, cd * cd.y, cd * cd.z) * (1.0 / (12.0 + m));   // Sherman-Morrison
        nd.mean -= 0.5 * d;
        nd.cinv_diag = vec3f(Cn[0][0], Cn[1][1], Cn[2][2]);
        nd.cinv_off = vec3f(Cn[0][1], Cn[0][2], Cn[1][2]);
        nd.kappa /= sqrt(1.0 + m / 12.0);
        nd.smax = sqrt(nd.smax * nd.smax + dd / 12.0);
        nd.radius = sqrt(nd.radius * nd.radius + 0.75 * dd);   // 3 sigma along d: sqrt(9 s^2 + 9 dd / 12)
        if (leaf) {
            let Rq = quat_matrix(G.quat);
            let s2 = G.scale * G.scale;
            let C = Rq * mat3x3f(vec3f(s2.x, 0.0, 0.0), vec3f(0.0, s2.y, 0.0), vec3f(0.0, 0.0, s2.z)) * transpose(Rq)
                + mat3x3f(d * d.x, d * d.y, d * d.z) * (1.0 / 12.0);
            var V: mat3x3f;
            var ev = eig3(C, &V);
            // ascending (x = short axis), right-handed
            var c0 = V[0]; var c1 = V[1]; var c2 = V[2];
            if (ev.x > ev.y) { ev = ev.yxz; let t = c0; c0 = c1; c1 = t; }
            if (ev.y > ev.z) { ev = ev.xzy; let t = c1; c1 = c2; c2 = t; }
            if (ev.x > ev.y) { ev = ev.yxz; let t = c0; c0 = c1; c1 = t; }
            c2 = cross(c0, c1);
            G.mu -= 0.5 * d;
            G.kappa = nd.kappa;
            G.scale = sqrt(max(ev, vec3f(1e-12)));
            G.quat = mat_to_quat(mat3x3f(c0, c1, c2));
        }
    }
    nd.pad0 = iso;
    n_dst[i] = nd;
    if (leaf) { g_dst[i] = G; }
}
