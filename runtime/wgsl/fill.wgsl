// Display fill (M45 E2, from the one path; renderer 2026-10-28): a Gaussian that doesn't face the room shows the
// exposure-weighted mean lighting of the room-facing Gaussians of its part within 3 cm (baked table, the asset's
// onepath.bin `fill`), so the dark deeper layers of a true-density asset don't show through the gaps of the lit skin
// (the felt texture). Runs after the lighting, in place on colour (direct) and lo_b lane 14 (the bounce the display
// adds); targets and sources are disjoint, and both buffers are rewritten for every Gaussian each frame before
// anything else reads them, so the fill only reaches the display. Prepend common.wgsl.

struct FillU { G: u32, p0: u32, p1: u32, p2: u32 }

@group(0) @binding(0) var<uniform> F: FillU;
@group(0) @binding(1) var<storage, read> fill: array<u32>;     // offsets [G + 1], then (index, f32 weight bits) pairs
@group(0) @binding(2) var<storage, read_write> colour: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> lo_b: array<vec4f>;

@compute @workgroup_size(64)
fn display_fill(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= F.G) { return; }
    let lo = fill[g]; let hi = fill[g + 1u];
    if (hi <= lo) { return; }
    var c = vec3f(0.0); var b = vec3f(0.0);
    for (var k = lo; k < hi; k++) {
        let o = F.G + 1u + 2u * k;
        let j = fill[o]; let w = bitcast<f32>(fill[o + 1u]);
        c += w * colour[j].rgb;
        b += w * lo_b[j * 15u + 14u].rgb;
    }
    colour[g] = vec4f(c, colour[g].w);
    lo_b[g * 15u + 14u] = vec4f(b, lo_b[g * 15u + 14u].w);
}

// ?lodlight=2 (lod.wgsl lod_list): a drawn leaf's fill sources must be shaded this frame too, though they may sit
// under a drawn node; mark them (lod_list shades and clears them)
@group(0) @binding(4) var<storage, read> lodflag: array<u32>;
@group(0) @binding(5) var<storage, read_write> need: array<u32>;

@compute @workgroup_size(64)
fn fill_need(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= F.G || lodflag[g] != 2u) { return; }
    for (var k = fill[g]; k < fill[g + 1u]; k++) { need[fill[F.G + 1u + 2u * k]] = 1u; }
}
