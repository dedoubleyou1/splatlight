// M39 direction-independent transfer: per leaf, iso_norm = 1 / (pi max(S, 0.02 S0)) into Node.pad0 (common.wgsl
// iso_norm). S = 4 <A V> over the sphere = (1 / pi) sum_c A(w_c) V(w_c) Omega_c on the 8 x 16 V-grid cell centres,
// S0 the same with V = 1; rest frame, own part only, so it is fixed for a rigid part and runs once per asset.
// Prepend common.wgsl.

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> vgrid: array<VGrid>;
@group(0) @binding(4) var<storage, read> materials: array<Material>;

const ISO_EPS: f32 = 0.02;

@compute @workgroup_size(64)
fn iso_init(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= U.gaussian_count) { return; }
    iso_one(gid.x);
}

// screen-space LOD with lit nodes (?lodlight=1, lod.js): the internal tree nodes as Gaussians (their records follow the
// leaves in `gaussians`, their V grids in `vgrid`), so a drawn node is shaded like a leaf
@compute @workgroup_size(64)
fn iso_init_nodes(@builtin(global_invocation_id) gid: vec3u) {
    let g = U.gaussian_count + gid.x;
    if (g >= arrayLength(&nodes) || g >= arrayLength(&vgrid)) { return; }
    iso_one(g);
}

fn iso_one(g: u32) {
    let G = gaussians[g];
    let nd = nodes[g];
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    // M41 transmission: the same exposure with V^m per channel (common.wgsl transmit_m), for iso_norm3
    let m = transmit_m(materials[min(G.flags & 0xFFFFu, max(U.material_count, 1u) - 1u)], G.albedo, true);
    let tr = any(m < vec3f(1.0));
    var S3 = vec3f(0.0);
    var S = 0.0;
    var S0 = 0.0;
    for (var c = 0u; c < 128u; c++) {
        let i = c / 16u; let j = c % 16u;
        let el = 0.5 * PI - (f32(i) + 0.5) * PI / 8.0;
        let az = -PI + (f32(j) + 0.5) * 2.0 * PI / 16.0;
        let w = vec3f(cos(el) * cos(az), cos(el) * sin(az), sin(el));
        let top = 0.5 * PI - f32(i) * PI / 8.0;
        let om = (2.0 * PI / 16.0) * (sin(top) - sin(top - PI / 8.0));
        let k = i * 16u + j;
        let v = f32((vgrid[g].w[k >> 2u] >> ((k & 3u) * 8u)) & 0xFFu) * (1.0 / 255.0);
        let a = footprint(nd, s3 * s3, w) * om;
        S += a * v;
        S0 += a;
        if (tr) { S3 += a * vpow3(v, m); }
    }
    let n = 1.0 / max(max(S, ISO_EPS * S0), 1e-30);   // (1/pi)(..) and the pi of iso_norm cancel
    nodes[g].pad0 = bitcast<u32>(n);
    nodes[g].pad1 = select(0u, iso_norm3_code(n, 1.0 / max(max(S3, vec3f(ISO_EPS * S0)), vec3f(1e-30))), tr);
}
