// Camera-only SGGX diffuse flakes (flake_common.wgsl), the direct light and the bookkeeping. Prepend common.wgsl,
// material_ext.wgsl, emissive.wgsl and flake_common.wgsl. Per frame, after the lighting and before project:
//   flake_base    per Gaussian: shiny, emissive and probe-only materials keep their colour + gain x eye bounce (w = 0);
//                 matte ones start from the M39 bounce when flake_bounce does not run, else 0 (w = 1)
//   flake_direct  per matte Gaussian: shade_emit's emitter cut with the flake lobe toward the eye, added
// and once per asset flake_expo (the SH2 of A V that normalises the lobe).

@group(0) @binding(14) var<storage, read> lo_b: array<vec4f>;   // bounce.wgsl's, [14] = toward the eye

// Per Gaussian, once: SH2 of A(w) V(w) over the 8 x 16 V-grid cells, rest frame (iso_init.wgsl's quadrature).
@compute @workgroup_size(64)
fn flake_expo(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= FP.count) { return; }
    let G = gaussians[g];
    let nd = nodes[g];
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    var e: array<f32, 9>;
    for (var m = 0u; m < 9u; m++) { e[m] = 0.0; }
    for (var c = 0u; c < 128u; c++) {
        let i = c / 16u; let j = c % 16u;
        let el = 0.5 * PI - (f32(i) + 0.5) * PI / 8.0;
        let az = -PI + (f32(j) + 0.5) * 2.0 * PI / 16.0;
        let w = vec3f(cos(el) * cos(az), cos(el) * sin(az), sin(el));
        let top = 0.5 * PI - f32(i) * PI / 8.0;
        let om = (2.0 * PI / 16.0) * (sin(top) - sin(top - PI / 8.0));
        let v = f32((vgrid[g].w[c >> 2u] >> ((c & 3u) * 8u)) & 0xFFu) * (1.0 / 255.0);
        let a = footprint(nd, s3 * s3, w) * v * om;
        let y = fl_sh2(w);
        for (var m = 0u; m < 9u; m++) { e[m] += a * y[m]; }
    }
    expo[3u * g] = vec4f(e[0], e[1], e[2], e[3]);
    expo[3u * g + 1u] = vec4f(e[4], e[5], e[6], e[7]);
    expo[3u * g + 2u] = vec4f(e[8], 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn flake_base(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= FP.count) { return; }
    let mat = materials[min(gaussians[g].flags & 0xFFFFu, max(FP.material_count, 1u) - 1u)];
    let b = FP.gain * lo_b[g * 15u + 14u].rgb;
    if (spec_weight(mat) > 0.0 || (mat.flags & 3u) != 0u) {
        cam[g] = vec4f(colour[g].rgb + b, 0.0);
        return;
    }
    cam[g] = vec4f(select(b, vec3f(0.0), (FP.flags & 1u) != 0u), 1.0);
}

@compute @workgroup_size(64)
fn flake_direct(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count || EP.count == 0u) { return; }
    let prev = cam[g];
    if (prev.w < 0.5) { return; }
    let G = gaussians[g];
    let nd = nodes[g];
    let xf = parts[G.part_surface & 0xFFFFu].xf;
    let P = xf_point(xf, G.mu);
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    let detC = s3 * s3;
    let E = fl_expo(g);
    let sh = fl_shape(G, E);
    let inorm = iso_norm(nd);
    let wo = xf_dir_inv(xf, normalize(U.eye - P));
    // shade_emitters' cut walk (emissive.wgsl), one output direction
    var acc = vec3f(0.0);
    var stack: array<u32, ESTACK>;
    stack[0] = EP.root;
    var sp = 1u;
    while (sp > 0u) {
        sp -= 1u;
        let e = emit_nodes[stack[sp]];
        if (!use_node(e, P)) {
            if (sp + 2u <= ESTACK) { stack[sp] = e.child_lo; stack[sp + 1u] = e.child_hi; sp += 2u; }
            continue;
        }
        let d = e.mean - P;
        let d2 = dot(d, d);
        if (d2 < 1e-8) { continue; }
        let ww = d / sqrt(d2);
        let wi = xf_dir_inv(xf, ww);
        let vi = vgrid_lookup(vgrid[g], wi);
        if (vi <= 1e-7) { continue; }
        let Tn = select(1.0, light_T(g, e.tag >> 16u), EP.lights != 0u);   // unified lights: each its own shadow
        let light = intensity(e, -ww) * (vi * footprint(nd, detC, wi) / d2 * Tn);
        acc += light * fl_lobe(sh, E, inorm, wi, wo);
    }
    let col = G.albedo * acc * (select(lamp_transmittance(g, 0u), 1.0, EP.lights != 0u) * inorm);
    cam[g] = vec4f(prev.rgb + col, 1.0);
}
