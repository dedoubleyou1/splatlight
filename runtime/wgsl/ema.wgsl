// Per-Gaussian lamp EMA (SPEC 3.3, M35 step 3 'adapt'): blend this frame's sample into the slot's running mean,
// with the weight raised when the lamp moved relative to its apparent size or a moving part crossed the shadow path.
// Prepend common.wgsl. Runs after walk_lamp in the same frame.

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> parts: array<Part>;
@group(0) @binding(3) var<storage, read> lamps: array<Lamp>;
@group(0) @binding(4) var<storage, read> samples: array<Sample>;
@group(0) @binding(5) var<storage, read_write> lamp_state: array<LampState>;
// bit 1 "walk next frame" for the amortized walk_emit (emissive.wgsl EP.hier bit 1): set while this Gaussian moves
// or has no history; bit 0 belongs to shadow_cull.wgsl
@group(0) @binding(6) var<storage, read_write> node_status: array<u32>;

// does the capsule from P to the lamp centre c (radius growing from 0 to R) come within rad of the sphere centre q?
fn capsule_hits(P: vec3f, c: vec3f, R: f32, q: vec3f, rad: f32) -> bool {
    let d = c - P;
    let len = length(d);
    let w = d / len;
    let t = clamp(dot(q - P, w), 0.0, len);
    return length(P + t * w - q) < rad + R * t / len;
}

fn angle_between(a: vec3f, b: vec3f) -> f32 {
    return atan2(length(cross(a, b)), dot(a, b));
}

@compute @workgroup_size(64)
fn ema(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let slot = U.frame % max(U.slot_count, 1u);
    let lamp = slot;
    if (lamp >= U.lamp_count || (lamps[lamp].flags & 1u) == 0u) { return; }
    let L = lamps[lamp];
    let P = xf_point(parts[own].xf, G.mu);
    let Pp = xf_point(parts[own].xf_prev, G.mu);

    var st = lamp_state[g].slots[slot];
    var count = st.y >> 16u;
    var v_old = bitcast<f32>(st.x);
    if ((st.y & 0xFFFFu) != lamp) { count = 0u; v_old = 0.0; }

    let smp = samples[g];

    // lamp (or receiver) motion relative to the lamp's apparent size
    let dist = length(L.center - P);
    let dth = angle_between(L.center - P, L.center_prev - Pp);
    let R = lamp_extent(L);
    let motion = dth * dist / max(R, 1e-6);
    var crossed = false;
    // a moving part's bounds across the shadow path, this frame or the last
    for (var p = 0u; p < U.part_count; p++) {
        if (p == own || (parts[p].flags & 2u) == 0u) { continue; }
        let b = parts[p].bound;
        let qc = xf_point(parts[p].xf, b.xyz);
        let qp = xf_point(parts[p].xf_prev, b.xyz);
        if (capsule_hits(P, L.center, R, qc, b.w) || capsule_hits(P, L.center, R, qp, b.w)) {
            crossed = true;
            break;
        }
    }
    // amortized walk: ask for a sample every frame while moving or without history
    let urgent = crossed || motion > 0.01 || count == 0u;
    node_status[g] = (node_status[g] & ~2u) | select(0u, 2u, urgent);
    if (smp.tests == SAMPLE_STALE) { return; }
    // other-part transmittance only: the own-part grid is in the shading's unshadowed term (SPEC 3.3)
    let V = exp(-smp.tau_other);
    var v_new: f32;
    if (U.history == 1u) {
        // effective count (N5): count holds round(n * 255); no floor, so a still scene averages up to 256 samples
        var d = min(1.0, 4.0 * motion);
        if (crossed) { d = max(d, U.alpha_part); }
        let n = min((1.0 - d) * f32(count) / 255.0 + 1.0, 256.0);
        v_new = v_old + (V - v_old) / n;
        count = u32(round(n * 255.0));
    } else {
        var a = max(1.0 / f32(count + 1u), U.alpha_min);
        a = max(a, clamp(motion, U.alpha_min, 1.0));
        if (crossed) { a = max(a, U.alpha_part); }
        v_new = (1.0 - a) * v_old + a * V;
        count = min(count + 1u, 0xFFFFu);
    }
    st = vec2u(bitcast<u32>(v_new), lamp | (count << 16u));
    lamp_state[g].slots[slot] = st;
}

// Unified lights (renderer/src/lights.js, emissive.wgsl EP.lights): the sample went to light smp.evals (lamps[id] is
// that light's sphere record: centre, radius, last centre). Slots 0-2 hold the receiver's strongest lights, slot 3 the
// rest together (id 0xFFFF). Samples are picked in proportion to each light's unshadowed share, so strong lights
// come back often. A fully clear receiver (walk_emit's hierarchical clear) sets every slot to 1. Same history rule
// and motion tests as ema, with the sample's light.
const LIGHT_REST: u32 = 0xFFFFu;
const HYST: f32 = 1.5;
// crossed caps the slot's history at NCROSS: strength-keyed slots keep long histories, and a jump is crossed for one frame
const NCROSS: f32 = 4.0;

fn light_strength(L: Lamp, P: vec3f) -> f32 {
    let d = L.center - P;
    return select(0.0, bitcast<f32>(L.pad0), (L.flags & 1u) != 0u) / max(dot(d, d), L.radius * L.radius);
}

// Shadow pooling (M42): up to 3 neighbours per Gaussian, siblings under its parent node in the same part (lighting.js
// poolNeighbours; 0xFFFFFFFF pads, all pads with ?shadowpool=0). ema_light averages V over its own sample and the
// neighbours' samples that went to the same light this frame, and applies the mean as one update of weight k (the
// number pooled): about 4 samples a frame per light for one walk each (M35 follow-up 2, clusters of 4).
@group(0) @binding(7) var<storage, read> pool: array<vec4u>;

@compute @workgroup_size(64)
fn ema_light(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count) { return; }
    let smp = samples[g];
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    let Pp = xf_point(parts[own].xf_prev, G.mu);
    let id = smp.evals;   // >= lamp_count: no particular light (walk_emit's hierarchical clear)
    let L = lamps[min(id, U.lamp_count - 1u)];

    var sts = lamp_state[g].slots;
    // walk_emit's hierarchical clear: no other part can touch any shadow ray from here, so every light is unshadowed.
    // Every slot takes V = 1 (it is exact, not a sample), so a slot whose light an occluder blocked earlier clears too.
    if (id == LIGHT_REST && smp.tests != SAMPLE_STALE) {
        let one = bitcast<u32>(1.0);
        for (var s = 0u; s < 4u; s++) {
            if ((sts[s].y >> 16u) > 0u) { sts[s].x = one; }
        }
        if ((sts[3].y >> 16u) == 0u) { sts[3] = vec2u(one, LIGHT_REST | (select(1u, 255u, U.history == 1u) << 16u)); }
        lamp_state[g].slots = sts;
        node_status[g] = node_status[g] & ~2u;
        return;
    }
    // Slots 0-2 go to the receiver's strongest lights (strength = lamps[id].pad0 as f32 / d^2, lights.js records): a light
    // without a slot takes an empty one, or the weakest one's when it is HYST times stronger (so lights of about equal
    // strength don't keep evicting each other and resetting their history), else it feeds the shared slot 3.
    var slot = 3u;
    var empty = 3u;
    var weak = 3u;
    var weak_s = 3.4e38;
    for (var s = 0u; s < 3u; s++) {
        let n = sts[s].y >> 16u;
        let sid = sts[s].y & 0xFFFFu;
        if (sid == id && n > 0u) { slot = s; }
        if (n == 0u || sid >= U.lamp_count) { if (empty == 3u) { empty = s; } continue; }
        let ss = light_strength(lamps[sid], P);
        if (ss < weak_s) { weak = s; weak_s = ss; }
    }
    if (slot == 3u && id < U.lamp_count) {
        if (empty < 3u) { slot = empty; }
        else if (light_strength(L, P) > HYST * weak_s) { slot = weak; }
    }
    let key = select(id, LIGHT_REST, slot == 3u);
    var st = sts[slot];
    var count = st.y >> 16u;
    var v_old = bitcast<f32>(st.x);
    if ((st.y & 0xFFFFu) != key) { count = 0u; v_old = 0.0; }

    let dist = length(L.center - P);
    let dth = angle_between(L.center - P, L.center_prev - Pp);
    let R = lamp_extent(L);
    let motion = dth * dist / max(R, 1e-6);
    var crossed = false;
    for (var p = 0u; p < U.part_count; p++) {
        if (p == own || (parts[p].flags & 2u) == 0u) { continue; }
        let b = parts[p].bound;
        let qc = xf_point(parts[p].xf, b.xyz);
        let qp = xf_point(parts[p].xf_prev, b.xyz);
        if (capsule_hits(P, L.center, R, qc, b.w) || capsule_hits(P, L.center, R, qp, b.w)) {
            crossed = true;
            break;
        }
    }
    let urgent = crossed || motion > 0.01 || count == 0u;
    node_status[g] = (node_status[g] & ~2u) | select(0u, 2u, urgent);
    if (smp.tests == SAMPLE_STALE) { return; }
    var vsum = exp(-smp.tau_other);
    var k = 1.0;
    let nb = pool[g];
    for (var j = 0u; j < 3u; j++) {
        if (nb[j] == 0xFFFFFFFFu) { break; }
        let s = samples[nb[j]];
        if (s.tests == SAMPLE_STALE || s.evals != id) { continue; }
        vsum += exp(-s.tau_other);
        k += 1.0;
    }
    let V = vsum / k;
    var v_new: f32;
    if (U.history == 1u) {
        var d = min(1.0, 4.0 * motion);
        if (crossed) { d = max(d, U.alpha_part); }
        var h = (1.0 - d) * f32(count) / 255.0;
        if (crossed) { h = min(h, NCROSS); }
        let n = min(h + k, 256.0);
        v_new = v_old + k * (V - v_old) / n;
        count = u32(round(n * 255.0));
    } else {
        var a = max(1.0 / f32(count + 1u), U.alpha_min);
        a = max(a, clamp(motion, U.alpha_min, 1.0));
        if (crossed) { a = max(a, max(U.alpha_part, 1.0 / NCROSS)); }
        v_new = (1.0 - a) * v_old + a * V;
        count = min(count + 1u, 0xFFFFu);
    }
    lamp_state[g].slots[slot] = vec2u(bitcast<u32>(v_new), key | (count << 16u));
}
