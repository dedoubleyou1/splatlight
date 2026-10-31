// Emissive Gaussian lights (David, 2026-10-23: lights are Gaussians too). Prepend common.wgsl and material_ext.wgsl.
//
// An emitter is a Gaussian record (SPEC 2.1) in world space whose `albedo` field holds its emitted radiance Le: along
// a ray it shows Le (1 - exp(-tau)), so drawing it is ordinary splatting. As a light it is volumetric and isotropic,
// with no normal: seen along w it has radiant intensity I(w) = Le sqrt(w^T M w), with M = (2 pi Ein(3))^2 det C C^-1
// so that I is M36's footprint at optical depth 3.
//
// Emitters are grouped in a binary emitter tree (built on the host, emit_nodes): a parent's M is fitted so its
// intensity along the principal axes of the children's moment-matched Gaussian equals their sum, and its Le is
// their intensity-weighted mean. Each receiver takes a light cut of that tree (a node is used when it is a leaf or farther than
// theta x its radius), so the cost per receiver follows the cut size, not the emitter count.
//
//   (shadow_cull.wgsl, with EP.hier bit 0: per scene-tree node, whether any shadow ray from under it can touch
//                     another part; walk_emit skips the Gaussians it clears)
//   walk_emit         per receiver: one node of the cut per frame, picked in proportion to its unshadowed
//                     contribution, then one emitter inside it the same way, a point jittered inside that, the walk_lamp segment walk toward it -> samples (the
//                     shadow history, ema.wgsl, is unchanged: lamps[0] is the group's bounding rectangle, used only for
//                     its motion test)
//   shade_emit        per receiver: shade.wgsl's transfer with the cut as the light set -> colour
//   direct_dirs_emit  the same toward the 14 fixed directions -> lo_dir (bounce.wgsl's emitter input)
// With U.iso bit 0 the transfer is M39's direction-independent one (common.wgsl iso_norm).
//
// mode 0 sums every emitter (brute force, for checking); mode 1 uses the cut.
// Emitters come either from the asset (an emissive part, SPEC 2.6: in the scene tree, so they occlude and the probes
// see them, but carry only their reflected light into the probes; shadow rays toward them skip that part) or, for
// assets without one, from a panel the host builds (not in the scene tree: no shadows, unseen by the probes).
//
// Unified lights (EP.lights = 1, renderer/src/lights.js): every emissive object is a light and none is special. The
// tree spans them all; each node carries its light id and, if one-sided, its own normal (EmitNode.tag / nrm), each
// leaf its part. A shadow sample records the light it went to (Sample.evals); ema_light keeps a receiver's shadow
// history per light in lamp_state's slots (0-2: its strongest lights, 3: the rest shared), and shading takes each
// node's T from its light's slot. EP.scale / one_sided / normal / skip_part are then unused (1, 0, -, ~0).

struct EmitParams {
    count: u32, root: u32,
    scale: f32,          // overlap correction: summing overlapping emitters counts the overlap several times
    theta: f32,          // cut rule: use a node when |P - mean| > theta * radius
    mode: u32, node_count: u32,
    skip_part: u32,      // the emitters' own part when they are in the asset (shadow rays to them skip it), else ~0
    one_sided: u32,      // 1: the emitters emit only toward +normal (SPEC 2.6 flag bit 2)
    normal: vec3f,
    n5: u32,             // N5 opt-ins: bit 0 golden-ratio pick sequence, bit 1 sample the emitter's own covariance,
                         // bit 2 progressive: 4 samples per re-shade, count history, leaves ease toward it
                         // (not N5) bit 3 ASG specular toward the eye (asg_phase), bit 4 with it: open cut nodes
                         // wider than a shiny receiver's lobe (spec_open)
    edge_soft: f32,      // one-sided emitters fade over |cos| < edge_soft (smoothstep in cos(w, normal)) instead of
                         // cutting off at the emitter's plane; 0 = the old hard cut
    hier: u32,           // bit 0: hierarchical shadow walk (shadow_cull.wgsl): skip the walk where node_status says clear;
                         // bit 1: amortized walk, a Gaussian walks once every `stride` frames unless ema flagged it
    stride: u32,
    lights: u32,         // bit 0: unified lights (above); bits 1-2 DEBUG ONLY (?lightdbg, unified only): bit 1 own-part
                         // V = 1 toward the lights, bit 2 footprint at the normal instead of toward the light;
                         // bit 3 self-shadow (?selfshadow=1): a light's own part occludes it (walk_tau_light)
}

// 64 B. Leaves: child_lo = 0xFFFFFFFF, child_hi = the emitter index.
// Unified lights only: tag bit 1 one-sided toward nrm (octahedral, pack2x16snorm), bits 16-31 the light id (0xFFFF:
// several); part = a leaf's part (else ~0). The single-lamp path leaves all three 0.
struct EmitNode {
    mean: vec3f, radius: f32,
    m_diag: vec3f, tag: u32,
    m_off: vec3f, child_lo: u32,
    Le: vec3f, child_hi: u32,
    c_diag: vec3f, nrm: u32,       // P3 ASG: the node's covariance with the leaves' flat-top widening folded in
    c_off: vec3f, part: u32,       // (xy, xz, yz); 96 B per node (lighting.js packEmitNodes)
}

const TAG_ONE_SIDED: u32 = 2u;
const LIGHT_MIXED: u32 = 0xFFFFu;

fn oct_decode(p: u32) -> vec3f {
    let e = unpack2x16snorm(p);
    var n = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) { n = vec3f((1.0 - abs(n.y)) * sign_nz(n.x), (1.0 - abs(n.x)) * sign_nz(n.y), n.z); }
    return normalize(n);
}
fn sign_nz(x: f32) -> f32 { return select(-1.0, 1.0, x >= 0.0); }

// unified lights: receiver g's shadow T toward light `id` (its slot, else the shared rest slot 3, else unshadowed)
fn light_T(g: u32, id: u32) -> f32 {
    let st = lamp_state[g].slots;
    for (var s = 0u; s < 3u; s++) {
        if ((st[s].y & 0xFFFFu) == id && (st[s].y >> 16u) > 0u) { return bitcast<f32>(st[s].x); }
    }
    if ((st[3].y >> 16u) > 0u && (st[3].y & 0xFFFFu) == LIGHT_MIXED) { return bitcast<f32>(st[3].x); }
    return 1.0;
}

@group(0) @binding(0) var<uniform> U: FrameUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
@group(0) @binding(3) var<storage, read> vgrid: array<VGrid>;
@group(0) @binding(4) var<storage, read> parts: array<Part>;
@group(0) @binding(6) var<storage, read> lamp_state: array<LampState>;
@group(0) @binding(7) var<storage, read> materials: array<Material>;
@group(0) @binding(8) var<storage, read_write> colour: array<vec4f>;
@group(0) @binding(9) var<storage, read> kids: array<u32>;
@group(0) @binding(13) var<storage, read_write> lo_dir: array<vec4f>;   // per Gaussian (common.wgsl LOB)
@group(0) @binding(18) var<storage, read> emit_nodes: array<EmitNode>;
@group(0) @binding(19) var<uniform> EP: EmitParams;
@group(0) @binding(20) var<storage, read_write> samples: array<Sample>;
@group(0) @binding(33) var<storage, read> node_status: array<u32>;   // bit 0 cull clear (shadow_cull.wgsl), bit 1 walk now (ema)

const ESTACK: u32 = 48u;
const NO_CHILD: u32 = 0xFFFFFFFFu;

const DIRS14 = array<vec3f, 14>(
    vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(1e-4, 0.0, 1.0), vec3f(-1.0, 0.0, 0.0), vec3f(0.0, -1.0, 0.0),
    vec3f(1e-4, 0.0, -1.0),
    vec3f(0.57735, 0.57735, 0.57735), vec3f(0.57735, 0.57735, -0.57735), vec3f(0.57735, -0.57735, 0.57735),
    vec3f(0.57735, -0.57735, -0.57735), vec3f(-0.57735, 0.57735, 0.57735), vec3f(-0.57735, 0.57735, -0.57735),
    vec3f(-0.57735, -0.57735, 0.57735), vec3f(-0.57735, -0.57735, -0.57735));

fn lamp_transmittance(g: u32, l: u32) -> f32 {
    for (var s = 0u; s < 4u; s++) {
        let st = lamp_state[g].slots[s];
        if ((st.y & 0xFFFFu) == l && (st.y >> 16u) > 0u) { return bitcast<f32>(st.x); }
    }
    return 1.0;
}

// radiant intensity of an emitter node toward direction w (unit, from the node to the receiver; s is symmetric)
fn intensity(e: EmitNode, w: vec3f) -> vec3f {
    var s = 1.0;
    if (EP.one_sided != 0u || (e.tag & TAG_ONE_SIDED) != 0u) {
        let c = dot(w, select(EP.normal, oct_decode(e.nrm), (e.tag & TAG_ONE_SIDED) != 0u));
        s = select(select(0.0, 1.0, c > 0.0), smoothstep(-EP.edge_soft, EP.edge_soft, c), EP.edge_soft > 0.0);
    }
    return e.Le * (s * EP.scale * sqrt(max(dot(w, sym_mul(e.m_diag, e.m_off, w)), 0.0)));
}

// the cut rule: leaves always; a parent when the receiver is farther than theta x its radius (mode 1 only)
fn use_node(e: EmitNode, P: vec3f) -> bool {
    if (e.child_lo == NO_CHILD || EP.mode == 0u) { return e.child_lo == NO_CHILD; }
    let d = P - e.mean;
    return dot(d, d) > EP.theta * EP.theta * e.radius * e.radius;
}

struct Recv { P: vec3f, xf: array<vec4f, 3>, n: vec3f, Sd: vec3f, So: vec3f, detC: f32, mat: Material, t1: vec3f, t2: vec3f }

fn receiver(g: u32) -> Recv {
    let G = gaussians[g];
    let xf = parts[G.part_surface & 0xFFFFu].xf;
    let mat = materials[min(G.flags & 0xFFFFu, max(U.material_count, 1u) - 1u)];
    let Rq = quat_matrix(G.quat);
    let Sloc = Rq * mat3x3f(vec3f(mat.flake.x, 0.0, 0.0), vec3f(0.0, mat.flake.y, 0.0), vec3f(0.0, 0.0, mat.flake.z)) * transpose(Rq);
    let s3 = G.scale.x * G.scale.y * G.scale.z;
    return Recv(xf_point(xf, G.mu), xf, Rq[0], vec3f(Sloc[0].x, Sloc[1].y, Sloc[2].z), vec3f(Sloc[1].x, Sloc[2].x, Sloc[2].y),
                s3 * s3, mat, Rq[1], Rq[2]);
}

// ---- shadows: one node of the cut per receiver per frame -------------------------------------------------------

fn walk_tau(root: u32, P: vec3f, w: vec3f, dist: f32, mur: vec3f, theta: f32) -> f32 {
    var stack: array<u32, STACK>;
    stack[0] = root;
    var sp = 1u;
    var tau = 0.0;
    let th2 = theta * theta;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        let d = nd.mean - P;
        let tt = clamp(dot(d, w), 0.0, dist);
        let e = d - tt * w;
        if (dot(e, e) > nd.radius * nd.radius) { continue; }
        var use_it = ni < U.gaussian_count;
        if (!use_it && theta > 0.0) {
            let r = mur - nd.mean;
            use_it = dot(r, r) > th2 * nd.smax * nd.smax;
        }
        if (use_it) {
            tau += nd.kappa * dirm(nd, w) * seg_geom(nd, P, w, 0.0, dist);
        } else {
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                if (sp < STACK) { stack[sp] = kids[j]; sp += 1u; }
            }
        }
    }
    return tau;
}

// walk_tau through the light's own part (unified lights with self-shadow, EP.lights bit 3): its emissive Gaussians
// (Gaussian flags bit 16, set by lighting.js) don't occlude (the sample point sits inside one), the rest do, so a lantern's paper softens its core and a candle's wax hides its flame from below.
// Nodes near either end (receiver mur, sample point yl) open down to leaves, since an aggregate there would hold
// the emitter's own density.
fn walk_tau_light(root: u32, P: vec3f, w: vec3f, dist: f32, mur: vec3f, yl: vec3f, theta: f32) -> f32 {
    var stack: array<u32, STACK>;
    stack[0] = root;
    var sp = 1u;
    var tau = 0.0;
    let th2 = theta * theta;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        let d = nd.mean - P;
        let tt = clamp(dot(d, w), 0.0, dist);
        let e = d - tt * w;
        if (dot(e, e) > nd.radius * nd.radius) { continue; }
        let leaf = ni < U.gaussian_count;
        var use_it = leaf;
        if (!use_it && theta > 0.0) {
            let r = mur - nd.mean;
            let q = yl - nd.mean;
            use_it = min(dot(r, r), dot(q, q)) > th2 * nd.smax * nd.smax;
        }
        if (leaf) {
            if (((gaussians[ni].flags >> 16u) & 1u) != 0u) { continue; }   // emissive (lighting.js sets bit 16, unified)
        }
        if (use_it) {
            tau += nd.kappa * dirm(nd, w) * seg_geom(nd, P, w, 0.0, dist);
        } else {
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                if (sp < STACK) { stack[sp] = kids[j]; sp += 1u; }
            }
        }
    }
    return tau;
}

// One sample point on the emitters as seen from P: a node of the cut picked in proportion to its unshadowed
// contribution, then one emitter inside it the same way, then a point inside that emitter. `k` separates several
// samples in one frame. With n5 bit 0 one number u = frac(o_g + (frame * 4 + k) phi), o_g hashed per receiver,
// drives the whole pick (a per-receiver rank-1 sequence, N5); otherwise each choice takes a fresh hash.
fn emitter_sample(g: u32, P: vec3f, frame: u32, k: u32, picked: ptr<function, vec2u>) -> vec3f {
    let golden = (EP.n5 & 1u) != 0u;
    var u = fract(to_unit(pcg_hash(g ^ 0x5bd1e995u)) + f32((frame * 4u + k) % 65536u) * 0.6180339887);
    var rng = pcg_hash(g ^ pcg_hash(frame * 4u + k + SEED));
    // weighted reservoir pick over the cut, weight = unshadowed luminance reaching P
    var pick = EP.root;
    var wsum = 0.0;
    var stack: array<u32, ESTACK>;
    stack[0] = EP.root;
    var sp = 1u;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let e = emit_nodes[ni];
        if (use_node(e, P)) {
            let d = e.mean - P;
            let d2 = max(dot(d, d), 1e-6);
            let I = intensity(e, -d / sqrt(d2));
            let wgt = dot(I, vec3f(0.2126, 0.7152, 0.0722)) / d2;
            if (wgt <= 0.0) { continue; }
            wsum += wgt;
            if (choose(&u, &rng, golden, wgt / wsum)) { pick = ni; }
        } else if (sp + 2u <= ESTACK) {
            stack[sp] = e.child_lo; stack[sp + 1u] = e.child_hi; sp += 2u;
        }
    }
    // descend from the picked node to one emitter, each step in proportion to the children's contribution, so the
    // sample point lies inside an emitter (a merged node's centre or spread can sit outside the light)
    var e = emit_nodes[pick];
    for (var depth = 0u; depth < 32u && e.child_lo != NO_CHILD; depth++) {
        let a = emit_nodes[e.child_lo];
        let b = emit_nodes[e.child_hi];
        let da = a.mean - P;
        let db = b.mean - P;
        let wa = dot(intensity(a, -normalize(da)), vec3f(0.2126, 0.7152, 0.0722)) / max(dot(da, da), 1e-6);
        let wb = dot(intensity(b, -normalize(db)), vec3f(0.2126, 0.7152, 0.0722)) / max(dot(db, db), 1e-6);
        if (choose(&u, &rng, golden, wa / max(wa + wb, 1e-30))) { e = a; } else { e = b; }
    }
    *picked = vec2u(e.tag >> 16u, e.part);   // unified lights: the leaf's light id and part
    if ((EP.n5 & 2u) != 0u) {
        // a point drawn from the emitter Gaussian itself: C = (sqrt(det M) / (2 pi Ein(3))) M^-1 (M's definition)
        let M = sym_mat(e.m_diag, e.m_off);
        let det = determinant(M);
        let C = inverse3(M) * (sqrt(max(det, 0.0)) / (2.0 * PI * EIN3));
        let z = gauss3(pcg_hash(rng ^ 0x68e31da4u));
        return e.mean + chol3(C) * clamp(z, vec3f(-3.0), vec3f(3.0));
    }
    // a point within one sigma of the emitter's centre
    rng = pcg_hash(rng);
    let j = vec3f(jitter(g, frame * 4u + k, 0u), jitter(g, frame * 4u + k, 1u), to_unit(pcg_hash(rng))) * 2.0 - 1.0;
    return e.mean + j * (e.radius / 3.0);
}

// Pick with probability p. Golden: split u's interval (so one u drives every choice and stays uniform); else hash.
fn choose(u: ptr<function, f32>, rng: ptr<function, u32>, golden: bool, p: f32) -> bool {
    if (golden) {
        if (*u < p) { *u = min(*u / max(p, 1e-30), 0.99999994); return true; }
        *u = min((*u - p) / max(1.0 - p, 1e-30), 0.99999994);
        return false;
    }
    *rng = pcg_hash(*rng);
    return to_unit(*rng) * 1.0 < p;
}

const EIN3: f32 = 1.68888;  // Ein(3) = int_0^3 (1 - e^-x) / x dx

fn sym_mat(dg: vec3f, off: vec3f) -> mat3x3f {
    return mat3x3f(vec3f(dg.x, off.x, off.y), vec3f(off.x, dg.y, off.z), vec3f(off.y, off.z, dg.z));
}

fn inverse3(m: mat3x3f) -> mat3x3f {
    let a = m[0]; let b = m[1]; let c = m[2];
    let r0 = cross(b, c); let r1 = cross(c, a); let r2 = cross(a, b);
    let inv_det = 1.0 / dot(a, r0);
    return transpose(mat3x3f(r0, r1, r2)) * inv_det;
}

// lower Cholesky factor of a symmetric positive definite matrix
fn chol3(m: mat3x3f) -> mat3x3f {
    let l00 = sqrt(max(m[0][0], 1e-20));
    let l10 = m[0][1] / l00;
    let l20 = m[0][2] / l00;
    let l11 = sqrt(max(m[1][1] - l10 * l10, 1e-20));
    let l21 = (m[1][2] - l20 * l10) / l11;
    let l22 = sqrt(max(m[2][2] - l20 * l20 - l21 * l21, 1e-20));
    return mat3x3f(vec3f(l00, l10, l20), vec3f(0.0, l11, l21), vec3f(0.0, 0.0, l22));
}

// three standard normals (Box-Muller) from one hash
fn gauss3(h: u32) -> vec3f {
    let h1 = pcg_hash(h); let h2 = pcg_hash(h1); let h3 = pcg_hash(h2); let h4 = pcg_hash(h3);
    let r1 = sqrt(-2.0 * log(max(to_unit(h1), 1e-7)));
    let r2 = sqrt(-2.0 * log(max(to_unit(h3), 1e-7)));
    return vec3f(r1 * cos(2.0 * PI * to_unit(h2)), r1 * sin(2.0 * PI * to_unit(h2)), r2 * cos(2.0 * PI * to_unit(h4)));
}

@compute @workgroup_size(64)
fn walk_emit(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count || EP.count == 0u) { return; }
    // hierarchical walk: no Gaussian of another part can touch any shadow ray from here (v_own is not read downstream)
    let hs = node_status[g];
    if ((EP.hier & 1u) != 0u && (hs & 1u) != 0u) {
        samples[g] = Sample(0.0, 1.0, 0u, select(0u, LIGHT_MIXED, EP.lights != 0u));   // clear toward every light
        return;
    }
    // amortized: a still Gaussian walks on its block's frame of every `stride` (blocks of 32 neighbours in index order,
    // so a whole SIMD group skips together); one that ema saw moving, or without history, walks every frame. The
    // others keep their history (ema skips a sample marked SAMPLE_STALE).
    if ((EP.hier & 2u) != 0u && (hs & 2u) == 0u && (pcg_hash(g / 32u) + U.frame) % max(EP.stride, 1u) != 0u) {
        samples[g] = Sample(0.0, 1.0, SAMPLE_STALE, 0u);
        return;
    }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let oxf = parts[own].xf;
    let P = xf_point(oxf, G.mu);

    var picked = vec2u(0u);
    let y = emitter_sample(g, P, U.frame, 0u, &picked);
    let dist = length(y - P);
    let w = (y - P) / dist;
    // shadow rays skip the lamp's part, or (unified) the picked leaf's part; with self-shadow (EP.lights bit 3) they
    // cross the picked leaf's part but not its emissive Gaussians
    let unified = EP.lights != 0u;
    let self_shadow = (EP.lights & 8u) != 0u;
    let skip = select(EP.skip_part, select(picked.y, 0xFFFFFFFFu, self_shadow), unified);
    var tau = 0.0;
    for (var p = 0u; p < U.part_count; p++) {
        if (p == own || p == skip) { continue; }
        let xf = parts[p].xf;
        let Pl = xf_point_inv(xf, P);
        if (self_shadow && p == picked.y) {
            tau += walk_tau_light(parts[p].root, Pl, xf_dir_inv(xf, w), dist, Pl, xf_point_inv(xf, y), U.theta);
        } else {
            tau += walk_tau(parts[p].root, Pl, xf_dir_inv(xf, w), dist, Pl, U.theta);
        }
    }
    samples[g] = Sample(tau, vgrid_lookup(vgrid[g], xf_dir_inv(oxf, w)), 0u, picked.x);   // evals: the light (ema_light)
}

// ---- projection-field shadows (prototype, ?lights=unified&shadow=pf; experiments/M42_many_lights) --------------
// Occluder-side M38 projection fields, baked per part in its rest frame (pfield.bin, layout in
// experiments/M42_many_lights/pf_export.py): per axis map, the optical-depth moments along the axis and the first hit
// from either side. A part with a field answers a shadow ray's optical depth with one bilinear lookup on the map of
// the axis nearest the ray, along the axis line through the ray's closest point to the part (the "hybrid" line of
// pf_stage1.py; receivers inside the part's bound keep their own line). Parts without one (the room shell, an
// all-emissive lamp) keep walk_tau.
@group(0) @binding(50) var<storage, read_write> pfield: array<u32>;   // read_write: shade_emit_pfd writes the T cache

const PF_TAB: u32 = 32u;   // axis table cells per side (pf_export.py TAB)
fn pf_f(i: u32) -> f32 { return bitcast<f32>(pfield[i]); }
fn pf_v3(i: u32) -> vec3f { return vec3f(pf_f(i), pf_f(i + 1u), pf_f(i + 2u)); }
fn erf_s(x: f32) -> f32 { return sign(x) * (1.0 - erfc_pos(abs(x))); }

// the 9 channels of one pixel (5 words: fp16 pairs, the 10th half unused)
fn pf_pixel(base: u32, ch: ptr<function, array<f32, 9>>, wgt: f32) {
    for (var j = 0u; j < 5u; j++) {
        let v = unpack2x16float(pfield[base + j]);
        (*ch)[2u * j] += wgt * v.x;
        if (j < 4u) { (*ch)[2u * j + 1u] += wgt * v.y; }
    }
}

// optical depth through part p's field along P + t w, t in [0, dist] (P, w in the part's rest frame); -1 = no field
fn pf_tau(p: u32, P: vec3f, w: vec3f, dist: f32) -> f32 {
    let K = pfield[2];
    let h = pf_f(4u);
    let d = 8u + 12u * K + 8u * p;
    let maps = pfield[d + 4u];
    if (maps == 0u) { return -1.0; }
    let c = pf_v3(d);
    let r = pf_f(d + 3u);
    let nx = pfield[d + 5u];
    let q = P - c;
    let tt = clamp(-dot(q, w), 0.0, dist);
    let e = q + tt * w;
    if (dot(e, e) > r * r) { return 0.0; }
    var best = -1.0;
    var k = 0u;
    var dk = 0.0;
    let tab = pfield[5];
    if (tab != 0u) {
        // the axis table: the direction folded to z >= 0, its octahedral cell, up to 4 candidate axes (pf_export.py)
        let df = w * select(1.0, -1.0, w.z < 0.0);
        let pp = df.xy / (abs(df.x) + abs(df.y) + abs(df.z));
        let c = clamp(vec2i((pp * 0.5 + 0.5) * f32(PF_TAB)), vec2i(0), vec2i(i32(PF_TAB) - 1));
        let cand = pfield[tab + u32(c.y) * PF_TAB + u32(c.x)];
        for (var j = 0u; j < 4u; j++) {
            let i = (cand >> (8u * j)) & 255u;
            if (i == 255u) { break; }
            let dd = dot(w, pf_v3(8u + 12u * i));
            if (abs(dd) > best) { best = abs(dd); k = i; dk = dd; }
        }
    } else {
        for (var i = 0u; i < K; i++) {
            let dd = dot(w, pf_v3(8u + 12u * i));
            if (abs(dd) > best) { best = abs(dd); k = i; dk = dd; }
        }
    }
    let Wk = pf_v3(8u + 12u * k);
    let Uk = pf_v3(8u + 12u * k + 4u);
    let Vk = pf_v3(8u + 12u * k + 8u);
    let x = select(e + Wk * dot(q - e, Wk), q, dot(q, q) <= r * r);
    let lo = pfield[d + 6u] + 2u * k;
    var fx = (dot(x, Uk) - pf_f(lo)) / h;
    var fy = (dot(x, Vk) - pf_f(lo + 1u)) / h;
    let nxf = f32(nx);
    if (fx < 0.0 || fy < 0.0 || fx > nxf || fy > nxf) { return 0.0; }
    fx = clamp(fx - 0.5, 0.0, nxf - 1.001);
    fy = clamp(fy - 0.5, 0.0, nxf - 1.001);
    let x0 = u32(fx);
    let y0 = u32(fy);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    var ch: array<f32, 9>;
    let row = (k * nx + y0) * nx + x0;
    pf_pixel(maps + row * 5u, &ch, (1.0 - tx) * (1.0 - ty));
    pf_pixel(maps + (row + 1u) * 5u, &ch, tx * (1.0 - ty));
    pf_pixel(maps + (row + nx) * 5u, &ch, (1.0 - tx) * ty);
    pf_pixel(maps + (row + nx + 1u) * 5u, &ch, tx * ty);
    let s0 = ch[0];
    if (s0 <= 1e-6) { return 0.0; }
    let m = ch[1] / s0;
    let sd = sqrt(max(ch[2] / s0 - m * m, 1e-8)) * 1.41421356;
    let Ap = ch[3];
    let Am = ch[6];
    if (Ap <= 1e-5 || Am <= 1e-5) { return 0.0; }
    let mp = ch[4] / Ap;
    let sp = sqrt(max(ch[5] / Ap, 1e-8));
    let mm = ch[7] / Am;
    let sm = sqrt(max(ch[8] / Am, 1e-8));
    var e0 = mp - 2.0 * sp;
    var e1 = mm + 2.0 * sm;
    if (e1 <= e0) { e0 = min(mp, mm) - 2.0 * max(sp, sm); e1 = max(mp, mm) + 2.0 * max(sp, sm); }
    // the window: the ray's depth range along the axis, from the receiver toward the light
    let s = dot(x, Wk);
    let rr = dist * best;
    let a0 = select(s - rr, s, dk >= 0.0);
    let a1 = select(s, s + rr, dk >= 0.0);
    let b0 = max(a0, e0);
    let b1 = min(a1, e1);
    if (b1 <= b0) { return 0.0; }
    let fe = erf_s((e1 - m) / sd) - erf_s((e0 - m) / sd);
    if (fe <= 1e-9) { return 0.0; }
    return s0 * (erf_s((b1 - m) / sd) - erf_s((b0 - m) / sd)) / fe;
}

// Direct field shadows (option B, ?lights=unified&shadow=pfd): shade_emitters' T toward each visited cut node,
// deterministic, with no samples and no history (walk_emit and ema_light do not run). lighting.js builds that variant
// of this file by replacing the light_T and lamp_transmittance calls in shade_emitters with pfd_T and 1.0, so its
// entry points never touch lamp_state and bind the field in its place. The receiver's own part is left to the own-part
// V grid, and with self-shadow off the node's own light part is skipped (its leaf's part; a LIGHT_MIXED node skips none).
// Limitation: parts without a field (the room shell, an all-emissive lamp) count as clear (T = 1) here.
// Strongest-lights split: pfd_select (called at the top of shade_emitters in the pfd variant) walks the same cut once
// and keeps the nodes that make up the top `share` of the receiver's unshadowed luminance (at most K, strongest first);
// only those get field lookups, the faint rest take T = 1. EP.lights bits 8-11 K (0 = 8), bits 16-23 share x 255
// (0, the default, or 255 = no split: every node gets a lookup; the split did not pay on cornell_sdf_candles);
// ?pfk=, ?pfshare=. Nodes the ASG spec cut (EP.n5 bit 4) opens are not in the list (T = 1).
var<private> pfd_sel: array<u32, 8>;
var<private> pfd_nsel: u32;
var<private> pfd_mode: u32;     // 1 write the T cache (shade_emit), 2 read it (direct_dirs_emit)
var<private> pfd_visit: u32;    // visited cut nodes so far (both passes visit the same nodes in the same order)
var<private> pfd_prevT: f32;
var<private> pfd_nlook: u32;    // inspection: field T evaluations and parts that passed the world-space cull
var<private> pfd_npass: u32;
var<private> pfd_mask: u32 = 0xFFFFFFFFu;   // field parts whose world bound meets the receiver's cone to the lights
const PFD_ALL: u32 = 0xFFFFFFFFu;

fn pfd_select(g: u32, r: Recv, nout: u32) {
    pfd_mode = select(2u, 1u, nout == 1u);   // shade_emit (one output) writes the T cache, direct_dirs_emit reads it
    pfd_visit = 0u;
    // per-receiver part cull, once before the cut walk: the parts (first 32) whose world bound meets the cone from
    // P tangent to the light tree's root sphere (cross-section radius t R / sqrt(len^2 - R^2) at distance t along the
    // axis, clamped at the centre); every shadow segment P -> node lies inside it. P inside the sphere: no cull.
    let wb = pfield[6];
    let root = emit_nodes[EP.root];
    let ax = root.mean - r.P;
    let len = length(ax);
    let R = root.radius;
    if (wb != 0u && U.part_count <= 32u && len > R * 1.0001) {
        let u = ax / len;
        let slope = R / sqrt(len * len - R * R);
        pfd_mask = 0u;
        for (var p = 0u; p < U.part_count; p++) {
            let rb = pf_f(wb + 4u * p + 3u);
            if (rb < 0.0) { continue; }
            let c = pf_v3(wb + 4u * p) - r.P;
            let t = clamp(dot(c, u), 0.0, len);
            let q = c - t * u;
            let rr = rb + t * slope;
            if (dot(q, q) <= rr * rr) { pfd_mask |= 1u << p; }
        }
    }
    let sb0 = (EP.lights >> 16u) & 255u;
    if (sb0 == 0u || sb0 == 255u) { pfd_nsel = PFD_ALL; return; }   // no ?pfshare= (default) or 1: every node gets a lookup
    // one walk of the cut keeping the 8 strongest nodes (insertion, strongest first) and the total; then the shortest
    // prefix of them that reaches share x total (at most K)
    var wt: array<f32, 8>;
    var total = 0.0;
    let nd = nodes[g];
    var stack: array<u32, ESTACK>;
    stack[0] = EP.root;
    var sp = 1u;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let e = emit_nodes[ni];
        if (!use_node(e, r.P)) {
            if (sp + 2u <= ESTACK) { stack[sp] = e.child_lo; stack[sp + 1u] = e.child_hi; sp += 2u; }
            continue;
        }
        let d = e.mean - r.P;
        let d2 = dot(d, d);
        if (d2 < 1e-8) { continue; }
        let ww = d / sqrt(d2);
        let wi = xf_dir_inv(r.xf, ww);
        var x = dot(intensity(e, -ww), vec3f(0.2126, 0.7152, 0.0722)) * vgrid_lookup(vgrid[g], wi) * footprint(nd, r.detC, wi) / d2;
        total += x;
        var id = ni;
        for (var j = 0u; j < 8u; j++) {          // insert, carrying the displaced entry down
            if (x > wt[j]) {
                let tw = wt[j]; let ti = pfd_sel[j];
                wt[j] = x; pfd_sel[j] = id;
                x = tw; id = ti;
            }
        }
    }
    let kb = (EP.lights >> 8u) & 15u;
    let kmax = select(min(kb, 8u), 8u, kb == 0u);
    let share = f32(sb0) / 255.0;
    pfd_nsel = 0u;
    var cum = 0.0;
    while (pfd_nsel < kmax && cum < share * total && wt[pfd_nsel] > 0.0) {
        cum += wt[pfd_nsel];
        pfd_nsel++;
    }
}

// T cache shared by shade_emit_pfd and direct_dirs_emit_pfd (pfield words from header word 7, 4 u32 per Gaussian):
// the T of the first 8 visited cut nodes as f16 pairs, written by shade_emit and read back by direct_dirs in the same
// frame (same receiver, same cut, same visit order; not with the ASG spec cut, EP.n5 bit 4, which opens nodes only
// for shade_emit).
fn pfd_T(g: u32, P: vec3f, e: EmitNode, ni: u32) -> f32 {
    let v = pfd_visit;
    pfd_visit++;
    let cache = pfield[7];
    let use_cache = cache != 0u && v < 8u && (EP.n5 & 16u) == 0u;
    let o = cache + 4u * g + v / 2u;
    if (use_cache && pfd_mode == 2u) {
        let t2 = unpack2x16float(pfield[o]);
        return select(t2.x, t2.y, (v & 1u) == 1u);
    }
    let T = pfd_T_calc(g, P, e, ni);
    if (use_cache && pfd_mode == 1u) {
        if ((v & 1u) == 0u) { pfd_prevT = T; pfield[o] = pack2x16float(vec2f(T, 1.0)); }
        else { pfield[o] = pack2x16float(vec2f(pfd_prevT, T)); }
    }
    return T;
}

fn pfd_T_calc(g: u32, P: vec3f, e: EmitNode, ni: u32) -> f32 {
    var chosen = pfd_nsel == PFD_ALL;
    for (var j = 0u; j < select(pfd_nsel, 0u, chosen); j++) { if (pfd_sel[j] == ni) { chosen = true; } }
    if (!chosen) { return 1.0; }
    return pfd_T_core(gaussians[g].part_surface & 0xFFFFu, P, e);
}

// T from P toward cut node e, skipping part `own` (the receiver's) and, without self-shadow, the node's light part
fn pfd_T_core(own: u32, P: vec3f, e: EmitNode) -> f32 {
    var skip = 0xFFFFFFFFu;
    if ((EP.lights & 8u) == 0u && (e.tag >> 16u) != LIGHT_MIXED) {
        var n = e;
        for (var i = 0u; i < 32u && n.part == NO_CHILD && n.child_lo != NO_CHILD; i++) { n = emit_nodes[n.child_lo]; }
        skip = n.part;
    }
    let v = e.mean - P;
    let dist = length(v);
    if (dist < 1e-6) { return 1.0; }
    let w = v / dist;
    var tau = 0.0;
    // world-space cull first: each field part's bound (centre, radius; radius < 0 = no field) as the host wrote it
    // this frame at pfield word 6, so only parts the segment touches get transformed and looked up
    let wb = pfield[6];
    pfd_nlook++;
    for (var p = 0u; p < U.part_count; p++) {
        if (p == own || p == skip) { continue; }
        if (p < 32u && (pfd_mask & (1u << p)) == 0u) { continue; }
        if (wb != 0u) {
            let c = pf_v3(wb + 4u * p);
            let rb = pf_f(wb + 4u * p + 3u);
            if (rb < 0.0) { continue; }
            let tt = clamp(dot(c - P, w), 0.0, dist);
            let q = c - P - tt * w;
            if (dot(q, q) > rb * rb) { continue; }
        }
        pfd_npass++;
        let xf = parts[p].xf;
        tau += max(pf_tau(p, xf_point_inv(xf, P), xf_dir_inv(xf, w), dist), 0.0);   // -1: no field, clear
    }
    return exp(-tau);
}

// Occluder-proxy variants of pfd_T_calc (experiments/M42_many_lights/proxy_stage1.py): lighting.js pfdSource swaps
// pf_tau for one of these, and binds a small host-built buffer at 50 in place of the field, same header words 3
// (part count), 6 (world bounds) and 7 (T cache) so pfd_select, the cache and the cull run unchanged.
// ?shadow=rcd: walk_tau over the part's own tree, stopping at nodes with radius < rmin (header word 4, f32) or at
// leaves, no theta test. Emissive leaves (gaussians flags bit 16) don't occlude. The kids array is copied into the
// buffer at header word 1 so the entry points stay within 8 storage buffers.
fn rcd_tau(p: u32, P: vec3f, w: vec3f, dist: f32) -> f32 {
    let kb = pfield[1];
    let rmin = pf_f(4u);
    var stack: array<u32, STACK>;
    stack[0] = parts[p].root;
    var sp = 1u;
    var tau = 0.0;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let nd = nodes[ni];
        let d = nd.mean - P;
        let tt = clamp(dot(d, w), 0.0, dist);
        let e = d - tt * w;
        if (dot(e, e) > nd.radius * nd.radius) { continue; }
        let leaf = ni < U.gaussian_count;
        if (leaf && ((gaussians[ni].flags >> 16u) & 1u) != 0u) { continue; }
        if (leaf || nd.radius < rmin) {
            tau += nd.kappa * dirm(nd, w) * seg_geom(nd, P, w, 0.0, dist);
        } else {
            for (var j = nd.child_lo; j < nd.child_hi; j++) {
                if (sp < STACK) { stack[sp] = pfield[kb + j]; sp += 1u; }
            }
        }
    }
    return tau;
}

// ?shadow=pxd: a fixed cut of the part's tree (proxy.bin, experiments/M42_many_lights/proxy_export.py): directory at
// word 8, 2 words per part (record offset, count); a record is 8 words: node index, then the footprint-fit kappa scale
// as a symmetric matrix M (xx yy zz, xy xz yz as f32; tau scales by wᵀ M w), pad. Each proxy is bound-tested first.
fn pxd_tau(p: u32, P: vec3f, w: vec3f, dist: f32) -> f32 {
    let o = pfield[8u + 2u * p];
    let n = pfield[9u + 2u * p];
    var tau = 0.0;
    for (var j = 0u; j < n; j++) {
        let rec = o + 8u * j;
        let ni = pfield[rec];
        let nd = nodes[ni];
        let d = nd.mean - P;
        let tt = clamp(dot(d, w), 0.0, dist);
        let e = d - tt * w;
        if (dot(e, e) > nd.radius * nd.radius) { continue; }
        if (ni < U.gaussian_count && ((gaussians[ni].flags >> 16u) & 1u) != 0u) { continue; }
        let s = dot(w, sym_mul(pf_v3(rec + 1u), pf_v3(rec + 4u), w));
        tau += s * nd.kappa * dirm(nd, w) * seg_geom(nd, P, w, 0.0, dist);
    }
    return tau;
}

// Per-cluster shadows (?shadow=pxc / rcc): the T toward the emitter cut computed once per receiver cluster (the part's
// Gaussians in one ?rcl= grid cell of its rest frame; the scene trees group too loosely for this) at the cluster's
// mean by cluster_T, and read back by pfd_T for the cluster's receivers when the cluster's list holds the node (else
// that receiver computes it: a fallback). Block at binding-50 word pfield[5] = C: C nclu, C+1 per-Gaussian cluster
// index (word offset), C+2 cluster list (rest-frame centre xyz f32, part: 4 words each), C+3 table, 24 words per
// cluster: 16 emitter-node ids (0xFFFFFFFF = end), then 8 words of f16 T pairs.
// PXC_NEAR (lighting.js sets it unless ?pxcmiss=walk): a miss (the receiver's cut differs from the cluster centre's)
// takes the T of a listed node that contains it (an ancestor: the listed sphere holds its centre, smallest first),
// else the luminance-weighted mean T of the listed nodes it contains (descendants), else the listed node nearest in
// direction; no fallbacks, counted as hits.
const PXC_N: u32 = 16u;
const PXC_NEAR: bool = false;
var<private> pxc_hit: u32;     // inspection: cut nodes answered from the cluster table

fn pxc_t(tb: u32, j: u32) -> f32 {
    let t2 = unpack2x16float(pfield[tb + PXC_N + j / 2u]);
    return select(t2.x, t2.y, (j & 1u) == 1u);
}

fn pxc_T(g: u32, P: vec3f, e: EmitNode, ni: u32) -> f32 {
    let C = pfield[5];
    let tb = pfield[C + 3u] + 24u * pfield[pfield[C + 1u] + g];
    let w = normalize(e.mean - P);
    var best = -2.0;
    var bj = 0u;
    var anc = 0xFFFFFFFFu;
    var ar = 1e30;
    var dsum = 0.0;
    var dw = 0.0;
    for (var j = 0u; j < PXC_N; j++) {
        let id = pfield[tb + j];
        if (id == 0xFFFFFFFFu) { break; }
        if (id == ni) { pxc_hit++; return pxc_t(tb, j); }
        if (PXC_NEAR) {
            let m = emit_nodes[id];
            let dm = m.mean - e.mean;
            let d2 = dot(dm, dm);
            if (m.radius > e.radius && d2 <= m.radius * m.radius && m.radius < ar) { anc = j; ar = m.radius; }
            if (m.radius < e.radius && d2 <= e.radius * e.radius) {
                let lw = max(dot(m.Le, vec3f(0.2126, 0.7152, 0.0722)), 1e-6);
                dsum += lw * pxc_t(tb, j); dw += lw;
            }
            let c = dot(w, normalize(m.mean - P));
            if (c > best) { best = c; bj = j; }
        }
    }
    if (PXC_NEAR && best > -2.0) {
        pxc_hit++;
        if (anc != 0xFFFFFFFFu) { return pxc_t(tb, anc); }
        if (dw > 0.0) { return dsum / dw; }
        return pxc_t(tb, bj);
    }
    return pfd_T_calc(g, P, e, ni);
}

@compute @workgroup_size(64)
fn cluster_T(@builtin(global_invocation_id) gid: vec3u) {
    let C = pfield[5];
    let c = gid.x;
    if (C == 0u || c >= pfield[C]) { return; }
    let cl = pfield[C + 2u] + 4u * c;
    let own = pfield[cl + 3u];
    let P = xf_point(parts[own].xf, pf_v3(cl));
    let tb = pfield[C + 3u] + 24u * c;
    var tv: array<f32, PXC_N>;
    var n = 0u;
    var stack: array<u32, ESTACK>;
    stack[0] = EP.root;
    var sp = 1u;
    while (sp > 0u && n < PXC_N) {
        sp -= 1u;
        let ni = stack[sp];
        let e = emit_nodes[ni];
        if (!use_node(e, P)) {
            if (sp + 2u <= ESTACK) { stack[sp] = e.child_lo; stack[sp + 1u] = e.child_hi; sp += 2u; }
            continue;
        }
        let d = e.mean - P;
        if (dot(d, d) < 1e-8) { continue; }
        pfield[tb + n] = ni;
        tv[n] = pfd_T_core(own, P, e);
        n++;
    }
    for (var j = n; j < PXC_N; j++) { pfield[tb + j] = 0xFFFFFFFFu; }
    for (var j = 0u; j < PXC_N / 2u; j++) { pfield[tb + PXC_N + j] = pack2x16float(vec2f(tv[2u * j], tv[2u * j + 1u])); }
}

// walk_emit with projection fields for the parts that have one. No own-part V (Sample.v_own is not read downstream),
// so it binds the field in place of vgrid and stays within 8 storage buffers.
@compute @workgroup_size(64)
fn walk_emit_pf(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    if (g >= U.gaussian_count || EP.count == 0u) { return; }
    let hs = node_status[g];
    if ((EP.hier & 1u) != 0u && (hs & 1u) != 0u) {
        samples[g] = Sample(0.0, 1.0, 0u, select(0u, LIGHT_MIXED, EP.lights != 0u));
        return;
    }
    if ((EP.hier & 2u) != 0u && (hs & 2u) == 0u && (pcg_hash(g / 32u) + U.frame) % max(EP.stride, 1u) != 0u) {
        samples[g] = Sample(0.0, 1.0, SAMPLE_STALE, 0u);
        return;
    }
    let G = gaussians[g];
    let own = G.part_surface & 0xFFFFu;
    let P = xf_point(parts[own].xf, G.mu);
    var picked = vec2u(0u);
    let y = emitter_sample(g, P, U.frame, 0u, &picked);
    let dist = length(y - P);
    let w = (y - P) / dist;
    let unified = EP.lights != 0u;
    let self_shadow = (EP.lights & 8u) != 0u;
    let skip = select(EP.skip_part, select(picked.y, 0xFFFFFFFFu, self_shadow), unified);
    var tau = 0.0;
    for (var p = 0u; p < U.part_count; p++) {
        if (p == own || p == skip) { continue; }
        let xf = parts[p].xf;
        let Pl = xf_point_inv(xf, P);
        let wl = xf_dir_inv(xf, w);
        if (self_shadow && p == picked.y) {
            tau += walk_tau_light(parts[p].root, Pl, wl, dist, Pl, xf_point_inv(xf, y), U.theta);
            continue;
        }
        let tf = pf_tau(p, Pl, wl, dist);
        if (tf >= 0.0) { tau += tf; } else { tau += walk_tau(parts[p].root, Pl, wl, dist, Pl, U.theta); }
    }
    samples[g] = Sample(tau, 1.0, 0u, picked.x);
}

// ---- ASG specular (experiments/P3_asg_light_projection; EP.n5 bit 3) -----------------------------------------
// A cut node seen from a receiver is, to first order in the angle, an anisotropic spherical Gaussian (ASG): the same
// Jacobian the camera splatter uses, Sigma_L = B C_s B^T / d^2 (B a tangent basis at the direction to the node, C_s
// the node's covariance with the leaves' flat-top widening Ein(3) / (1 - e^-3) folded in, EmitNode c_diag / c_off). Its integral is
// the node's intensity / d^2, so it only adds the light's shape. The SGGX specular flake lobe toward w_o is a 3-term
// ASG mixture ((1 + q)^-2 ~ sum w_k exp(-a_k q) around the flake normal, warped through the mirror map), and
// light x lobe is the Gaussian product integral. asg_phase returns it per unit light: it stands in for
// phase_specular(w_i, w_o) when the light is a shape rather than a point. Where the fit is weakest (a light much
// smaller than the lobe, or far out in the lobe's GGX tail) it hands off by smoothstep to the point phase, taken at
// 3 Gauss-Hermite points along the light's long axis.

const SPEC_CUT: f32 = 1.0;                         // spec_open: node angular std over the lobe's width
// ASG_W, ASG_A, asg_phase and its 2x2 helpers live in material_ext.wgsl (bounce.wgsl uses them for probe cells too)

// spec cut: open a parent whose angular size (radius / 3 ~ one std) is wider than the receiver's lobe, so a sharp
// reflection shows the lights a merged node would smear. Only near the mirror direction wr (rest frame): out in the
// lobe's tail a merged node reads the same, and a mirror-like receiver would otherwise open every leaf. The lobe width
// is floored at SPEC_LOBE_MIN for the same reason.
const SPEC_LOBE_MIN: f32 = 0.01;
fn spec_open(e: EmitNode, r: Recv, wr: vec3f, lobe_w: f32) -> bool {
    if (e.child_lo == NO_CHILD) { return false; }
    let dv = e.mean - r.P;
    let d = length(dv);
    let ang = e.radius / (3.0 * d);
    let lw = max(lobe_w, SPEC_LOBE_MIN);
    if (ang <= SPEC_CUT * lw) { return false; }
    let off = acos(clamp(dot(xf_dir_inv(r.xf, dv / d), wr), -1.0, 1.0));
    return off < 3.0 * ang + 6.0 * lw;
}

// ---- shading ------------------------------------------------------------------------------------------------

// Reflection coverage (U.iso bit 4, renderer shell 2026-10-25; experiments/reflection_coverage): toward the eye only
// (nout == 1, M39), the specular flakes' share k 4 pi p damp goes to spec_eye instead of the colour, which keeps
// (1 - k); shade_emit stores it in colour's second half, colour[G + g], for project.wgsl to blend by coverage.
var<private> spec_eye: vec3f;

// Radiance toward each of `nout` world directions; each light node's light-side term is shared by all of them.
// asg: shiny receivers take each cut node as a shape (asg_phase), not a point; only shade_emit (the eye) sets it.
fn shade_emitters(g: u32, r: Recv, wo_w: array<vec3f, 15>, nout: u32, asg: bool) -> array<vec3f, 15> {
    let nd = nodes[g];
    let iso = (U.iso & 1u) != 0u;
    let tr = (U.iso & 9u) == 9u;                  // M41 transmission: own-part V^m per channel (common.wgsl transmit_m)
    let tm = transmit_m(r.mat, gaussians[g].albedo, tr);
    let ks = spec_weight(r.mat);
    var wo: array<vec3f, 15>;
    var fsame: array<f32, 15>;
    var acc: array<vec3f, 15>;
    for (var k = 0u; k < nout; k++) {
        wo[k] = xf_dir_inv(r.xf, wo_w[k]);
        fsame[k] = 1.0 / max(vgrid_lookup(vgrid[g], wo[k]), r.mat.vfloor);
        acc[k] = vec3f(0.0);
    }
    let shape = asg && iso && ks > 0.0;
    let spec_cut = shape && (EP.n5 & 16u) != 0u;
    let lobe_w = sqrt(min(r.mat.flake.y, r.mat.flake.z) / max(r.mat.flake.x, 1e-12));
    let wr_eye = mirror(wo[0], r.n);   // spec cut: shade_emit's one direction, the eye
    let split = iso && nout == 1u && (U.iso & 16u) != 0u;
    var spec = vec3f(0.0);
    var stack: array<u32, ESTACK>;
    stack[0] = EP.root;
    var sp = 1u;
    while (sp > 0u) {
        sp -= 1u;
        let ni = stack[sp];
        let e = emit_nodes[ni];
        if (!use_node(e, r.P) || (spec_cut && spec_open(e, r, wr_eye, lobe_w))) {
            if (sp + 2u <= ESTACK) { stack[sp] = e.child_lo; stack[sp + 1u] = e.child_hi; sp += 2u; }
            continue;
        }
        let d = e.mean - r.P;
        let d2 = dot(d, d);
        if (d2 < 1e-8) { continue; }
        let ww = d / sqrt(d2);
        let wi = xf_dir_inv(r.xf, ww);
        var vi = vgrid_lookup(vgrid[g], wi);
        if ((EP.lights & 2u) != 0u) { vi = 1.0; }   // DEBUG (?lightdbg bit 1)
        if (vi <= 1e-7) { continue; }
        // emitter intensity / d^2 (radiance x solid angle), then the receiver's catch: V(wi) A(wi); unified lights
        // take their own shadow T here (the single lamp's multiplies the sum below)
        let Tn = select(1.0, light_T(g, e.tag >> 16u), EP.lights != 0u);
        let fp = footprint(nd, r.detC, select(wi, r.n, (EP.lights & 4u) != 0u));   // DEBUG (?lightdbg bit 2): at the normal
        let light = intensity(e, -ww) * select(vec3f(vi), vpow3(vi, tm), tr) * (fp / d2 * Tn);
        if (iso) {
            if (ks <= 0.0) { acc[0] += light; continue; }
            let wr = mirror(wi, r.n);
            let damp = iso_damp(iso_norm(nd), footprint(nd, r.detC, wr) * vgrid_lookup(vgrid[g], wr));
            if (shape) {
                // the node's footprint as seen from the receiver: B C_s B^T / d^2 in a world tangent basis at ww,
                // the basis carried into the rest frame with wi
                let bw = onb(ww);
                let c0 = sym_mul(e.c_diag, e.c_off, bw[0]);
                let c1 = sym_mul(e.c_diag, e.c_off, bw[1]);
                let sl = vec3f(dot(bw[0], c0), dot(bw[0], c1), dot(bw[1], c1)) / d2;
                let lb = array<vec3f, 2>(xf_dir_inv(r.xf, bw[0]), xf_dir_inv(r.xf, bw[1]));
                for (var k = 0u; k < nout; k++) {
                    let p = asg_phase(wi, lb, sl, r.n, r.t1, r.t2, r.mat.flake, r.Sd, r.So, wo[k]);
                    if (split) { acc[0] += light * (1.0 - ks); spec += light * (iso_lobe_from(ks, p, damp) - (1.0 - ks)); continue; }
                    acc[k] += light * iso_lobe_from(ks, p, damp);
                }
                continue;
            }
            if (split) {
                acc[0] += light * (1.0 - ks);
                spec += light * (iso_lobe(r.mat, r.Sd, r.So, wi, wo[0], damp) - (1.0 - ks));
                continue;
            }
            for (var k = 0u; k < nout; k++) { acc[k] += light * iso_lobe(r.mat, r.Sd, r.So, wi, wo[k], damp); }
            continue;
        }
        let wn = dot(wi, r.n);
        for (var k = 0u; k < nout; k++) {
            let f = select(1.0, fsame[k], wn * dot(wo[k], r.n) > 0.0);
            acc[k] += light * (phase_ex(r.mat, r.n, r.Sd, r.So, wi, wo[k]) * f);
        }
    }
    let alb = gaussians[g].albedo;
    let T = select(lamp_transmittance(g, 0u), 1.0, EP.lights != 0u);
    if (iso) {                                    // M39: one colour for every direction (common.wgsl iso_norm)
        let a0 = acc[0];                          // matte: every direction takes the one sum (read before acc[0] is scaled)
        for (var k = 0u; k < nout; k++) {
            acc[k] = alb * select(acc[k], a0, ks <= 0.0) * (T * iso_norm3(nd, tr));
        }
        spec_eye = alb * spec * (T * iso_norm3(nd, tr));
        return acc;
    }
    for (var k = 0u; k < nout; k++) { acc[k] = alb * acc[k] * (T / footprint(nd, r.detC, wo[k])); }
    return acc;
}

@compute @workgroup_size(64)
fn shade_emit(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= U.gaussian_count) { return; }
    shade_leaf(gid.x);
}

// ?lodlight=2 (lod.wgsl lod_list): only the leaves the LOD cut draws (and the display fill's sources), indirect
@group(0) @binding(62) var<storage, read> lodlist: array<u32>;
@compute @workgroup_size(64)
fn shade_emit_list(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= lodlist[0]) { return; }
    shade_leaf(lodlist[16u + gid.x]);
}

fn shade_leaf(g: u32) {
    if (EP.count == 0u) {
        // no emitter or light on (hill, sun only): nothing to gather, only an emissive material's own radiance below.
        // Skipping the receiver set-up and the walk over the one dark leaf saved ~13 ms a frame on hill (2026-10-31)
        let mat = materials[min(gaussians[g].flags & 0xFFFFu, max(U.material_count, 1u) - 1u)];
        if ((mat.flags & 1u) == 0u) {
            colour[g] = vec4f(0.0, 0.0, 0.0, 1.0);
            if ((U.iso & 16u) != 0u) { colour[U.gaussian_count + g] = vec4f(0.0); }
            return;
        }
    }
    let r = receiver(g);
    var wo: array<vec3f, 15>;
    wo[0] = normalize(U.eye - r.P);
    spec_eye = vec3f(0.0);
    var c = shade_emitters(g, r, wo, 1u, (EP.n5 & 8u) != 0u);
    // an emitter's own radiance (SPEC 2.6), as shade.wgsl: one-sided ones only toward +short axis
    if ((r.mat.flags & 1u) != 0u && ((r.mat.flags & 4u) == 0u || dot(xf_dir_inv(r.xf, wo[0]), r.n) > 0.0)) {
        c[0] += vec3f(unpack2x16float(r.mat.pad0), unpack2x16float(r.mat.pad1).x);
    }
    colour[g] = vec4f(c[0], 1.0);
    if ((U.iso & 16u) != 0u) { colour[U.gaussian_count + g] = vec4f(spec_eye, 0.0); }   // reflection coverage
}

// screen-space LOD with lit nodes (?lodlight=1, lod.js): direct light for each drawn internal tree node, shaded as a
// Gaussian like a leaf (its record, V grid, iso_norm and map T follow the leaves'), plus its leaves' mean bounce
// (lod_colour, at 2 G + (N - G) + k) into the node's shown colour at 2 G + k
@group(0) @binding(61) var<storage, read> lodflag: array<u32>;
@compute @workgroup_size(64)
fn shade_emit_nodes(@builtin(global_invocation_id) gid: vec3u) {
    let g = U.gaussian_count + gid.x;
    if (g >= arrayLength(&lodflag) || lodflag[g] != 2u) { return; }
    shade_node(g, arrayLength(&lodflag));
}

// ?lodlight=2: the drawn nodes from lod_list, indirect
@compute @workgroup_size(64)
fn shade_emit_nodes_list(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= lodlist[1]) { return; }
    shade_node(lodlist[16u + U.gaussian_count + gid.x], arrayLength(&lodflag));
}

fn shade_node(g: u32, N: u32) {
    let r = receiver(g);
    var wo: array<vec3f, 15>;
    wo[0] = normalize(U.eye - r.P);
    spec_eye = vec3f(0.0);
    let c = shade_emitters(g, r, wo, 1u, (EP.n5 & 8u) != 0u);
    let k = g - U.gaussian_count;
    let bounce = colour[2u * U.gaussian_count + (N - U.gaussian_count) + k].rgb;
    colour[2u * U.gaussian_count + k] = vec4f(c[0] + bounce, 1.0);
}

@compute @workgroup_size(64)
fn direct_dirs_emit(@builtin(global_invocation_id) gid: vec3u) {
    let g = select(gid.x, gid.x * 4u + U.frame % 4u, (U.iso & 2u) != 0u);   // stage 2: a quarter per frame
    if (g >= U.gaussian_count) { return; }
    direct_dirs_one(g);
}

// sender LOD (default, ?sendercut=0 off; bounce.wgsl sc_select): the cut's leaves (the same quarter a frame), then its nodes
// (bounce.wgsl sc_now: all of them in a frame that re-selected the cut, else a quarter) into sc_L
@group(0) @binding(81) var<storage, read> sc_list: array<u32>;
@group(0) @binding(83) var<storage, read_write> sc_L: array<vec4f>;   // bounce.wgsl sc_X: per slot x 8, the faces' direct light at 6-7
@compute @workgroup_size(64)
fn direct_dirs_emit_sc(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= sc_list[0]) { return; }
    let g = sc_list[16u + gid.x];
    if ((U.iso & 2u) != 0u && g % 4u != U.frame % 4u) { return; }
    direct_dirs_one(g);
}

@compute @workgroup_size(64)
fn direct_dirs_nodes_sc(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= sc_list[1] || !(sc_list[9] != 0u || (U.iso & 2u) == 0u || gid.x % 4u == U.frame % 4u)) { return; }   // bounce.wgsl sc_now
    let n = sc_list[16u + U.gaussian_count + gid.x];
    let r = receiver(n);
    let nd = nodes[n];
    // the light each face catches (bounce.wgsl node_cut: a cut node is a two-sided flake, one colour per face of its
    // short axis), before albedo and exposure: sum I V A T / d^2, matte (shininess left out of the senders)
    var c = array<vec3f, 2>(vec3f(0.0), vec3f(0.0));
    var stack: array<u32, ESTACK>;
    stack[0] = EP.root;
    var sp = 1u;
    while (sp > 0u) {
        sp -= 1u;
        let e = emit_nodes[stack[sp]];
        if (!use_node(e, r.P)) {
            if (sp + 2u <= ESTACK) { stack[sp] = e.child_lo; stack[sp + 1u] = e.child_hi; sp += 2u; }
            continue;
        }
        let d = e.mean - r.P;
        let d2 = dot(d, d);
        if (d2 < 1e-8) { continue; }
        let ww = d / sqrt(d2);
        let wi = xf_dir_inv(r.xf, ww);
        let vi = vgrid_lookup(vgrid[n], wi);
        if (vi <= 1e-7) { continue; }
        let Tn = select(1.0, light_T(n, e.tag >> 16u), EP.lights != 0u);
        let light = intensity(e, -ww) * (vi * footprint(nd, r.detC, wi) / d2 * Tn);
        let h = 0.5 + 0.5 * sc_face(dot(wi, r.n));
        c[0] += light * h;
        c[1] += light * (1.0 - h);
    }
    let T = select(lamp_transmittance(n, 0u), 1.0, EP.lights != 0u);
    sc_L[gid.x * 8u + 6u] = vec4f(c[0] * T, 0.0);
    sc_L[gid.x * 8u + 7u] = vec4f(c[1] * T, 0.0);
}

fn direct_dirs_one(g: u32) {
    let r = receiver(g);
    var wo: array<vec3f, 15>;
    for (var k = 0u; k < 14u; k++) { wo[k] = DIRS14[k]; }
    let c = shade_emitters(g, r, wo, 14u, false);
    // one value toward the probes (common.wgsl LOB): matte ones have the same in every slot; a shiny one's mean
    // weighted as bounce.wgsl node_leaf weights the directions (footprint x V), so its total into the tree is kept
    var mean = vec3f(0.0);
    var ws = 0.0;
    let nd = nodes[g];
    for (var k = 0u; k < 14u; k++) {
        let u = xf_dir_inv(r.xf, DIRS14[k]);
        let w = footprint(nd, r.detC, u) * vgrid_lookup(vgrid[g], u);
        mean += c[k] * w; ws += w;
    }
    lo_dir[g] = vec4f(select(c[0], mean / max(ws, 1e-20), ws > 0.0), 0.0);
}
