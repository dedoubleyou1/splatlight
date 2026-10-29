// One-path lighting (M45 E9, renderer ?lighting=onepath): copy each Gaussian's direction-independent B into the
// renderer's colour buffer (SPEC 3.4: f32x4 radiance per Gaussian; the coverage half colour[G + g] is cleared here), with
// E2's display fill: a Gaussian that doesn't face the room shows the exposure-weighted mean B of the room-facing
// Gaussians of its part within 3 cm (baked table, onepath.bin `fill`), so the dark deeper skin layers no longer show
// through the gaps of the lit skin (the felt texture). C.p0 = 0 turns the fill off. Prepend common.wgsl.

struct OpColour { G: u32, p0: u32, p1: u32, p2: u32 }
struct GExtC { Le_w: vec4f, B: vec4f, Bf: vec4f, mu0: vec4f, Bc: vec4f }

@group(0) @binding(0) var<uniform> C: OpColour;
@group(0) @binding(1) var<storage, read> gext: array<GExtC>;
@group(0) @binding(2) var<storage, read_write> colour: array<vec4f>;
@group(0) @binding(3) var<storage, read> fill: array<u32>;     // offsets [G + 1], then (index, f32 weight bits) pairs

@compute @workgroup_size(64)
fn op_colour(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= C.G) { return; }
    var c = gext[g].B.xyz;
    let lo = fill[g]; let hi = fill[g + 1u];
    if (C.p0 != 0u && hi > lo) {
        c = vec3f(0.0);
        for (var k = lo; k < hi; k++) {
            let o = C.G + 1u + 2u * k;
            c += bitcast<f32>(fill[o + 1u]) * gext[fill[o]].B.xyz;
        }
    }
    colour[g] = vec4f(c, 1.0);
    colour[C.G + g] = vec4f(0.0);   // the coverage half: the specular pass (onepath_spec.wgsl) fills it for coverage materials
}
