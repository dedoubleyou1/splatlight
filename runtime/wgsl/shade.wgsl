// Surface-element shading (SPEC 4.4; the settled M36 recipe). Per Gaussian, the radiance it shows toward the eye:
//   c = alb / A(w_o) * sum_lamps Le win T_lamp (1/K) sum_k G_k V(w_k) A(w_k) p(w_k -> w_o) f_k
// G_k = lamp geometry for stratified sample k, V = own-part 8x16 grid, A = footprint, p = SGGX diffuse-flake phase
// with the material's flake shape oriented by the Gaussian, f_k = 1 / max(V(w_o), vfloor) when w_k and w_o are on
// the same side of the short axis, else 1. T_lamp = the per-Gaussian EMA of the other-part transmittance
// (ema.wgsl), 1 for lamps without a slot. All directions are taken in the Gaussian's part rest frame.
// With U.iso bit 0 (M39): c = alb * iso_norm * sum_lamps Le win T_lamp (1/K) sum_k G_k V(w_k) A(w_k), the same
// from every direction (common.wgsl iso_norm).
// Prepend common.wgsl.

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> vgrid: array<VGrid>;
@group(0) @binding(4) var<storage, read> parts: array<Part>;
@group(0) @binding(5) var<storage, read> lamps: array<Lamp>;
@group(0) @binding(6) var<storage, read> lamp_state: array<LampState>;
@group(0) @binding(7) var<storage, read> materials: array<Material>;
@group(0) @binding(8) var<storage, read_write> colour: array<vec4f>;

fn lamp_transmittance(g: u32, l: u32) -> f32 {
    for (var s = 0u; s < 4u; s++) {
        let st = lamp_state[g].slots[s];
        if ((st.y & 0xFFFFu) == l && (st.y >> 16u) > 0u) { return bitcast<f32>(st.x); }
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn shade(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let nd = nodes[g];
    let own = G.part_surface & 0xFFFFu;
    let xf = parts[own].xf;
    let P = xf_point(xf, G.mu);
    let mat = materials[min(G.flags & 0xFFFFu, max(U.material_count, 1u) - 1u)];
    let Rq = quat_matrix(G.quat);
    let n = Rq[0];                                          // short axis, rest frame
    let Sloc = Rq * mat3x3f(vec3f(mat.flake.x, 0.0, 0.0), vec3f(0.0, mat.flake.y, 0.0), vec3f(0.0, 0.0, mat.flake.z)) * transpose(Rq);
    let Sd = vec3f(Sloc[0].x, Sloc[1].y, Sloc[2].z);
    let So = vec3f(Sloc[1].x, Sloc[2].x, Sloc[2].y);
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    let detC = s3 * s3;

    let wo = xf_dir_inv(xf, normalize(U.eye - P));
    let Ao = footprint(nd, detC, wo);
    let fsame = 1.0 / max(vgrid_lookup(vgrid[g], wo), mat.vfloor);
    let on = dot(wo, n);

    let iso = (U.iso & 1u) != 0u;
    let tr = (U.iso & 9u) == 9u;                            // M41 transmission (common.wgsl transmit_m)
    let tm = transmit_m(mat, G.albedo, tr);
    let na = max(U.light_samples_axis, 1u);
    let K = na * na;
    var c = vec3f(0.0);
    for (var l = 0u; l < U.lamp_count; l++) {
        let L = lamps[l];
        if ((L.flags & 1u) == 0u) { continue; }
        let win = window(L, length(L.center - P));
        if (win <= 0.0) { continue; }
        var acc = 0.0;
        var acc3 = vec3f(0.0);                              // iso: per channel, V^m
        for (var k = 0u; k < K; k++) {
            let y = lamp_point(L, P, (f32(k / na) + 0.5) / f32(na), (f32(k % na) + 0.5) / f32(na));
            let geo = lamp_geometry(L, P, y);
            if (geo <= 0.0) { continue; }
            let wi = xf_dir_inv(xf, normalize(y - P));
            let vi = vgrid_lookup(vgrid[g], wi);
            if (vi <= 1e-7) { continue; }
            let catchw = geo * vi * footprint(nd, detC, wi);
            if (iso) { acc3 += (geo * footprint(nd, detC, wi)) * vpow3(vi, tm); continue; }
            let f = select(1.0, fsame, dot(wi, n) * on > 0.0);
            acc += catchw * phase(mat, n, Sd, So, wi, wo) * f;
        }
        c += L.radiance * (win * lamp_transmittance(g, l) / f32(K)) * select(vec3f(acc), acc3, iso);
    }
    var col = G.albedo * c / Ao;
    if (iso) { col = G.albedo * c * iso_norm3(nd, tr); }
    // emitters (SPEC 2.6): radiance Le where opaque, i.e. a splat of colour Le; one-sided ones emit along +n only
    if ((mat.flags & 1u) != 0u && ((mat.flags & 4u) == 0u || on > 0.0)) {
        col += vec3f(unpack2x16float(mat.pad0), unpack2x16float(mat.pad1).x);
    }
    colour[g] = vec4f(col, 1.0);
}
