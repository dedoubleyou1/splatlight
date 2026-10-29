// Camera-only SGGX diffuse flakes (flake_common.wgsl), the probe bounce. Prepend common.wgsl, material_ext.wgsl,
// bounce.wgsl and flake_common.wgsl. After bounce_sh and flake_direct, per matte Gaussian (M39 stage 2 only):
//   c_b(w_o) = alb sum_c iso_norm A(u_c) V(u_c) Omega_c lobe(u_c -> w_o) L(w_c),  L = the 8 probes' SH2 (bounce_sh's
//   blend), as the flake transfer t_f(w_o) = SH2 of the cell weights, dotted with L; added x gain to cam.
// Not in it: near-field T and near-field reflection (both off by default), which bounce_sh / near_refl apply.

@compute @workgroup_size(64)
fn flake_bounce(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count || (U.iso & 3u) != 3u) { return; }
    let prev = cam[g];
    if (prev.w < 0.5) { return; }
    let G = gaussians[g];
    let nd = nodes[g];
    let own = G.part_surface & 0xFFFFu;
    let xf = parts[own].xf;
    let P = xf_point(xf, G.mu);
    let pw = probe_weights(P, own);
    var Lsh: array<vec3f, 9>;
    for (var m = 0u; m < 9u; m++) { Lsh[m] = vec3f(0.0); }
    for (var k = 0u; k < 8u; k++) {
        let w = pw.w[k];
        if (w <= 0.0) { continue; }
        for (var m = 0u; m < 9u; m++) { Lsh[m] += w * probe_sh[pw.pid[k] * 9u + m].rgb; }
    }
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    let E = fl_expo(g);
    let sh = fl_shape(G, E);
    let inorm = iso_norm(nd);
    let wo = xf_dir_inv(xf, normalize(U.eye - P));
    var t: array<f32, 9>;
    for (var m = 0u; m < 9u; m++) { t[m] = 0.0; }
    for (var c = 0u; c < NCELL; c++) {
        let w = cell_dir(c);
        let u = xf_dir_inv(xf, w);
        let v = vgrid_lookup(vgrid[g], u);
        if (v <= 0.0) { continue; }
        let a = inorm * footprint(nd, s3 * s3, u) * v * cell_omega(c) * fl_lobe(sh, E, inorm, u, wo);
        let y = sh2_basis(w);
        for (var m = 0u; m < 9u; m++) { t[m] += a * y[m]; }
    }
    var cb = vec3f(0.0);
    for (var m = 0u; m < 9u; m++) { cb += t[m] * Lsh[m]; }
    cam[g] = vec4f(prev.rgb + FP.gain * G.albedo * max(cb, vec3f(0.0)), 1.0);
}
