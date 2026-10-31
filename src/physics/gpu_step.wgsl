// The whole physics step on the GPU (gpu_world.py), appended to kernels.wgsl: world.js's step in its GPU form
// (world.js gpuForm, the CPU replica): pjacobi sweeps, no pair cache, Jacobi damping. Per step:
//   bp_build (wake pass, body pairs, root seeds) -> descent (kernels.wgsl detect_prep / detect_ind2, qcut, level caps)
//   -> patches (pair_args .. body_fill, patch_bp) -> snap normals (overlap_xf, patch_step reduce only, snap_bp)
//   -> energy_mark -> nsub x (integrate2, k x [overlap_xf, patch_step, body_apply_pj], velocity, damp_patch,
//   damp_body) -> energy_limit -> sleep_step (island sleep, records the poses)
// Small loops over bodies or body pairs (wake pass, island sleep) run in one thread: bodies are few and the order
// then matches world.js exactly.

const BX: u32 = 53u;   // bext per body: baxes rows 0-8 | loff[level][axis] 9-20 | Iinv (row-major, substep) 21-29 |
                       // E0 30 (-1e30: not limited) | in contact 31 | impact 32 | still 33 | sleepL 34 | speed 35 | root 36
                       // | still window start 37-39 (sleepDisp) | pre-solve velocity 40-42, angular 43-45 (restitution)
                       // | step-start pos 46-48, quat 49-52 (sleepNet)
const MAXB: u32 = 256u;

struct StepW {
  dt: f32, mu: f32, margin: f32, margin_cap: f32,
  wake_v: f32, sleep_v: f32, sleep_t: f32, swept_q: f32,
  swept_min: f32, cos_snap: f32, gx: f32, gy: f32,
  gz: f32, energy_on: f32, island_on: f32, snap_on: f32,
  nbody: u32, cap: u32, nbp_cap: u32, has_lev: u32,
  damp_form: u32,   // world.js dampForm: 0 sum of the pairs' scaled means, 1 their mean, 2 sum of plain means, 3 colour
                    // (one colour per dispatch pair, damp_colors)
  stick_v: f32,     // world.js stickV: velocity-level static friction below this sliding speed (0 off)
  sleep_depth: f32, // world.js sleepDepth: not still while a contact's error exceeds this (0 off)
  anchors_on: u32,  // world.js frictionAnchors
  mu_s: f32,        // Stribeck (world.js muAt): static and kinetic coefficients, blend speed; mu_s < 0: mu everywhere
  mu_k: f32,
  stribeck_v: f32,
  sleep_disp: f32,  // world.js sleepDisp: still needs the body within this distance of its still window's start (0 off)
  surface_normals: u32,   // world.js surfaceNormals: thin nodes' outward normals for the patch normal
  materials: u32,   // world.js materials: per patch friction from the nodes' physics materials (slab slots 6-9)
  restitution: u32, // world.js restitution: the velocity pass sets each contact's normal velocity (replaces damp)
  sleep_net: u32,   // world.js sleepNet: the sleep speed from the step's net pose change, not the end velocity
}

@group(0) @binding(43) var<storage, read_write> pairN: array<vec4f>;      // per body pair: snap normal
@group(0) @binding(44) var<storage, read_write> bext: array<f32>;
@group(0) @binding(46) var<storage, read_write> patch_bp: array<u32>;     // per patch: its body pair
@group(0) @binding(47) var<uniform> SW: StepW;
@group(0) @binding(48) var<storage, read_write> gstate: array<atomic<u32>>;   // [0] body pairs, [1] -, [2] steps done
@group(0) @binding(49) var<storage, read_write> hist: array<f32>;          // per step per body: pos, quat, asleep (9)
@group(0) @binding(51) var<storage, read_write> bp_info_w: array<vec4f>;  // kernels.wgsl bp_info, written here
@group(0) @binding(52) var<storage, read_write> bpflag: array<u32>;       // per body pair: has pairs (sleep_step scratch)
@group(0) @binding(53) var<storage, read_write> shist: array<u32>;        // per step: pairs, patches, overflow flags (4)
@group(0) @binding(54) var<storage, read_write> bpcolor: array<u32>;      // per body pair: damping colour
@group(0) @binding(55) var<storage, read_write> cargs: array<u32>;        // per colour: indirect args, patches then bodies (6)
const NCOL: u32 = 8u;
// friction anchors (world.js frictionAnchors): a persistent table per frame, key (body pair i B + j) << 18 | rest-frame
// cell (+1; 0 empty), two vec4 per slot (la, lb); this frame reads atab_old and writes atab_new (ping-pong, so contacts
// unseen for a frame expire); within the frame per patch: panc 4 vec4 (la valid | lb reanchor | touched | its body
// pair's snap normal, written by snap_patch so patch_step needs neither pairN nor patch_bp)
@group(0) @binding(58) var<storage, read_write> patch_cell: array<u32>;
@group(0) @binding(59) var<storage, read_write> akey_old: array<atomic<u32>>;
@group(0) @binding(60) var<storage, read_write> adat_old: array<vec4f>;
@group(0) @binding(61) var<storage, read_write> akey_new: array<atomic<u32>>;
@group(0) @binding(62) var<storage, read_write> adat_new: array<vec4f>;
@group(0) @binding(63) var<storage, read_write> panc: array<vec4f>;
@group(0) @binding(56) var<storage, read_write> stick_corr: array<vec4f>;  // per patch: static friction, patch_corr layout


fn asleep(b: u32) -> bool { return bodies[b * BODY + 19u] != 0.0; }
fn set_asleep(b: u32, v: bool) { bodies[b * BODY + 19u] = select(0.0, 1.0, v); }
fn vel_of(b: u32) -> vec3f { let o = b * BODY + 8u; return vec3f(bodies[o], bodies[o + 1u], bodies[o + 2u]); }
fn om_of(b: u32) -> vec3f { let o = b * BODY + 12u; return vec3f(bodies[o], bodies[o + 1u], bodies[o + 2u]); }
fn set_vel(b: u32, v: vec3f) { let o = b * BODY + 8u; bodies[o] = v.x; bodies[o + 1u] = v.y; bodies[o + 2u] = v.z; }
fn set_om(b: u32, v: vec3f) { let o = b * BODY + 12u; bodies[o] = v.x; bodies[o + 1u] = v.y; bodies[o + 2u] = v.z; }
fn prev_pos(b: u32) -> vec3f { let o = b * BODY + 20u; return vec3f(bodies[o], bodies[o + 1u], bodies[o + 2u]); }
fn prev_quat(b: u32) -> vec4f { let o = b * BODY + 24u; return vec4f(bodies[o], bodies[o + 1u], bodies[o + 2u], bodies[o + 3u]); }
fn root_of(b: u32) -> u32 { return u32(bext[b * BX + 36u]); }
// the substep's world inverse inertia (row-major, as world.js Iinv): Iinv u
fn iinv(b: u32, u: vec3f) -> vec3f {
  let o = b * BX + 21u;
  return vec3f(bext[o] * u.x + bext[o + 1u] * u.y + bext[o + 2u] * u.z, bext[o + 3u] * u.x + bext[o + 4u] * u.y + bext[o + 5u] * u.z,
               bext[o + 6u] * u.x + bext[o + 7u] * u.y + bext[o + 8u] * u.z);
}
// a body pair's relative speed bound (world.detect)
fn pair_speed(i: u32, j: u32) -> f32 {
  return length(vel_of(i) - vel_of(j)) + length(om_of(i)) * nodes[root_of(i) * NS + 10u] * body_scale(i)
    + length(om_of(j)) * nodes[root_of(j) * NS + 10u] * body_scale(j);
}
fn pair_margin(i: u32, j: u32) -> f32 {
  let sw = 1.5 * pair_speed(i, j) * SW.dt;
  return SW.margin + select(min(sw, SW.margin_cap), sw, body_inv_mass_raw(i) == 0.0 || body_inv_mass_raw(j) == 0.0);
}
fn roots_apart(i: u32, j: u32, margin: f32) -> bool {
  let ri = root_of(i); let rj = root_of(j);
  let a = quat_mat(body_quat(i)) * (node_mean(ri) * body_scale(i)) + body_pos(i);
  let b = quat_mat(body_quat(j)) * (node_mean(rj) * body_scale(j)) + body_pos(j);
  return length(a - b) > nodes[ri * NS + 10u] * body_scale(i) + nodes[rj * NS + 10u] * body_scale(j) + margin;
}

// world.js directedMargin: the root pair apart over its motion segment (root centres' velocities over 1.5 dt; each
// root's reach grown by its spin); the pair's isotropic margin is then SW.margin alone
fn roots_apart_dir(i: u32, j: u32) -> bool {
  let ri = root_of(i); let rj = root_of(j);
  let a = quat_mat(body_quat(i)) * (node_mean(ri) * body_scale(i)) + body_pos(i);
  let b = quat_mat(body_quat(j)) * (node_mean(rj) * body_scale(j)) + body_pos(j);
  let T = 1.5 * SW.dt;
  let M = ((vel_of(i) + cross(om_of(i), a - body_pos(i))) - (vel_of(j) + cross(om_of(j), b - body_pos(j)))) * T;
  var d = a - b;
  let MM = dot(M, M);
  if (MM > 0.0) { d += clamp(-dot(d, M) / MM, 0.0, 1.0) * M; }
  return length(d) > nodes[ri * NS + 10u] * body_scale(i) * (1.0 + length(om_of(i)) * T)
    + nodes[rj * NS + 10u] * body_scale(j) * (1.0 + length(om_of(j)) * T) + SW.margin;
}

// ---- step start (one thread): world.detect's wake pass and body-pair list; root pairs seed the descent; the
// descent's and the solve's counters reset
@compute @workgroup_size(1)
fn bp_build() {
  let B = SW.nbody;
  for (var i = 0u; i < B; i++) {
    for (var j = i + 1u; j < B; j++) {
      if (!(asleep(i) || asleep(j)) || (body_inv_mass_raw(i) == 0.0 && body_inv_mass_raw(j) == 0.0)) { continue; }
      if ((asleep(i) || body_inv_mass_raw(i) == 0.0) && (asleep(j) || body_inv_mass_raw(j) == 0.0)) { continue; }
      if (roots_apart(i, j, pair_margin(i, j))) { continue; }
      if (asleep(i) && length(vel_of(j)) > SW.wake_v) { set_asleep(i, false); }
      if (asleep(j) && length(vel_of(i)) > SW.wake_v) { set_asleep(j, false); }
    }
  }
  var n = 0u;
  for (var i = 0u; i < B; i++) {
    for (var j = i + 1u; j < B; j++) {
      if (body_inv_mass_raw(i) == 0.0 && body_inv_mass_raw(j) == 0.0) { continue; }
      if ((asleep(i) || body_inv_mass_raw(i) == 0.0) && (asleep(j) || body_inv_mass_raw(j) == 0.0)) { continue; }
      let dirm = (P.flags & 16u) != 0u;
      let margin = select(pair_margin(i, j), SW.margin, dirm);
      if (select(roots_apart(i, j, margin), roots_apart_dir(i, j), dirm)) { continue; }
      if (asleep(i) && length(vel_of(j)) > SW.wake_v) { set_asleep(i, false); }
      if (asleep(j) && length(vel_of(i)) > SW.wake_v) { set_asleep(j, false); }
      if (n >= SW.nbp_cap) { atomicStore(&gstate[1], 1u); continue; }
      bp_info_w[n] = vec4f(f32(i), f32(j), margin, 0.0);
      front2[n] = vec4u(n, root_of(i), root_of(j), 0u);
      n++;
    }
  }
  atomicStore(&gstate[0], n);
  for (var k = 0u; k < 8u + 64u; k++) { atomicStore(&dstate[k], 0u); }
  atomicStore(&dstate[0], n);
  atomicStore(&dstate[3], SW.cap);   // the first detect_prep reads half 0
  for (var k = 0u; k < 4u; k++) { atomicStore(&sstate[k], 0u); }
}

// ---- per patch its body pair (from the hash table's key)
@compute @workgroup_size(64)
fn patch_bp_fill(@builtin(global_invocation_id) id: vec3u) {
  let h = id.x;
  if (h >= arrayLength(&ptab)) { return; }
  let key = atomicLoad(&ptab[h]);
  if (key == 0u) { return; }
  patch_bp[pslot_patch[h]] = (key - 1u) >> 18u;
  patch_cell[pslot_patch[h]] = (key - 1u) & 0x3ffffu;
}

fn anchor_key(p: u32) -> u32 {
  let ij = patch_bodies[p];
  return (((ij.x * SW.nbody + ij.y) << 18u) | patch_cell[p]) + 1u;
}
fn anchor_hash(key: u32, T: u32) -> u32 { return ((key * 0x9E3779B1u) >> 9u) & (T - 1u); }

// per patch at the step start: its anchor from last frame's table, if any
@compute @workgroup_size(64)
fn anchor_load(@builtin(global_invocation_id) id: vec3u) {
  let p = id.x;
  if (p >= atomicLoad(&sstate[1])) { return; }
  panc[4u * p] = vec4f(0.0); panc[4u * p + 1u] = vec4f(0.0); panc[4u * p + 2u] = vec4f(0.0);
  let key = anchor_key(p);
  let T = arrayLength(&akey_old);
  var h = anchor_hash(key, T);
  for (var n = 0u; n < T; n++) {
    let k = atomicLoad(&akey_old[h]);
    if (k == 0u) { return; }
    if (k == key) { panc[4u * p] = vec4f(adat_old[2u * h].xyz, 1.0); panc[4u * p + 1u] = vec4f(adat_old[2u * h + 1u].xyz, 0.0); return; }
    h = (h + 1u) & (T - 1u);
  }
}

// per patch at a substep's end (world.js anchorUpdate): contacts that are new or slid anchor at their patch point, in
// both bodies' frames, from the final poses
@compute @workgroup_size(64)
fn anchor_update(@builtin(global_invocation_id) id: vec3u) {
  let p = id.x;
  if (p >= atomicLoad(&sstate[1])) { return; }
  let a0 = panc[4u * p]; let a1 = panc[4u * p + 1u];
  // no friction call this substep: the contact broke and releases its anchor (world.js: valid only through consecutive
  // substeps of contact)
  if (panc[4u * p + 2u].y == 0.0) { panc[4u * p].w = 0.0; return; }
  panc[4u * p + 2u].y = 0.0;
  if (a0.w != 0.0 && a1.w == 0.0) { return; }   // anchored and held
  let ij = patch_bodies[p];
  let c = patch_red[4u * p + 1u].xyz;
  let la = transpose(quat_mat(body_quat(ij.x))) * (c - body_pos(ij.x));
  let lb = transpose(quat_mat(body_quat(ij.y))) * (c - body_pos(ij.y));
  panc[4u * p] = vec4f(la, 1.0); panc[4u * p + 1u] = vec4f(lb, 0.0);
}

// per patch at the step end: anchors touched this frame into this frame's table (next frame reads it)
@compute @workgroup_size(64)
fn anchor_store(@builtin(global_invocation_id) id: vec3u) {
  let p = id.x;
  if (p >= atomicLoad(&sstate[1])) { return; }
  if (panc[4u * p].w == 0.0 || panc[4u * p + 2u].x == 0.0) { return; }
  let key = anchor_key(p);
  let T = arrayLength(&akey_new);
  var h = anchor_hash(key, T);
  for (var n = 0u; n < T; n++) {
    let prev = atomicCompareExchangeWeak(&akey_new[h], 0u, key);
    if (prev.exchanged || prev.old_value == key) {
      adat_new[2u * h] = vec4f(panc[4u * p].xyz, 0.0); adat_new[2u * h + 1u] = vec4f(panc[4u * p + 1u].xyz, 0.0);
      return;
    }
    if (prev.old_value != 0u) { h = (h + 1u) & (T - 1u); }
  }
}

// friction coefficient at slip speed v (world.js muAt)
fn mu_at(v: f32) -> f32 {
  if (SW.mu_s < 0.0) { return SW.mu; }
  let r = v / SW.stribeck_v;
  return SW.mu_k + (SW.mu_s - SW.mu_k) * exp(-r * r);
}

// patch p's coefficient at slip speed v (world.js muAtP): its combined material (panc slots 2.zw, 3.w), else mu_at
fn mu_at_p(p: u32, v: f32) -> f32 {
  if (SW.materials == 0u) { return mu_at(v); }
  let ms = panc[4u * p + 2u].z; let mk = panc[4u * p + 2u].w; let vs = max(panc[4u * p + 3u].w, 1e-6);
  let r = v / vs;
  return mk + (ms - mk) * exp(-r * r);
}

// ---- snap normals (world.js pairN): per body pair the negated, normalised sum of its patches' unit-gradient sums at
// the step-start poses (patch_step in reduce-only mode wrote them to patch_red)
@compute @workgroup_size(64)
fn snap_bp(@builtin(global_invocation_id) id: vec3u) {
  let bp = id.x;
  if (bp >= atomicLoad(&gstate[0])) { return; }
  var g = vec3f(0.0);
  let np = atomicLoad(&sstate[1]);
  for (var p = 0u; p < np; p++) { if (patch_bp[p] == bp) { g -= patch_red[4u * p].yzw; } }
  let l = length(g);
  pairN[bp] = vec4f(select(vec3f(0.0), g / l, l > 0.0), 0.0);
}

// per patch: its body pair's snap normal, beside its anchor (patch_step reads it there)
@compute @workgroup_size(64)
fn snap_patch(@builtin(global_invocation_id) id: vec3u) {
  let p = id.x;
  if (p >= atomicLoad(&sstate[1])) { return; }
  panc[4u * p + 3u] = vec4f(pairN[patch_bp[p]].xyz, 0.0);
}

// mechanical energy of body b: kinetic, kinetic + potential (world.js energy)
fn energy(b: u32) -> vec2f {
  let m = 1.0 / body_inv_mass_raw(b);
  let wb = transpose(quat_mat(body_quat(b))) * om_of(b);
  let ke = 0.5 * m * dot(vel_of(b), vel_of(b)) + 0.5 * dot(body_ib(b), wb * wb);
  return vec2f(ke, ke - m * dot(vec3f(SW.gx, SW.gy, SW.gz), body_pos(b)));
}

// ---- energy limiter, before the substeps (world.js markEnergy): in contact, in an impact, starting energy
@compute @workgroup_size(64)
fn energy_mark(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= SW.nbody || body_inv_mass_raw(b) == 0.0) { return; }
  let o = b * BX;
  if (SW.sleep_net != 0u) {   // the step-start pose (sleep_step)
    let p = body_pos(b); let q = body_quat(b);
    bext[o + 46u] = p.x; bext[o + 47u] = p.y; bext[o + 48u] = p.z;
    bext[o + 49u] = q.x; bext[o + 50u] = q.y; bext[o + 51u] = q.z; bext[o + 52u] = q.w;
  }
  bext[o + 31u] = select(0.0, 1.0, body_patch_start[b + 1u] > body_patch_start[b]);
  var impact = 0.0;
  let nbp = atomicLoad(&gstate[0]);
  for (var bp = 0u; bp < nbp; bp++) {
    let info = bp_info[bp];
    let i = u32(info.x); let j = u32(info.y);
    if (i != b && j != b) { continue; }
    let d = body_pos(j) - body_pos(i);
    let dl = max(length(d), 1e-30);
    if (dot(vel_of(i) - vel_of(j), d) / select(dl, 1.0, length(d) == 0.0) > 0.3) { impact = 1.0; }
  }
  bext[o + 32u] = impact;
  let e = energy(b);
  bext[o + 30u] = select(-1e30, e.y, e.x < 0.5 * 0.09 / body_inv_mass_raw(b));
}

// ---- energy limiter, after the substeps (world.js limitEnergy)
@compute @workgroup_size(64)
fn energy_limit(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (SW.energy_on == 0.0 || b >= SW.nbody || body_inv_mass_raw(b) == 0.0 || asleep(b)) { return; }
  let o = b * BX;
  if (bext[o + 31u] == 0.0 || bext[o + 32u] != 0.0 || bext[o + 30u] == -1e30) { return; }
  let e = energy(b);
  let excess = e.y - bext[o + 30u] - 1e-4;
  if (excess <= 0.0 || e.x <= 0.0) { return; }
  let f = sqrt(max(0.0, e.x - excess) / e.x);
  set_vel(b, vel_of(b) * f); set_om(b, om_of(b) * f);
}

// ---- substep start: previous poses, gravity and gyroscopic integration (world.js substeps), the substep's Iinv
@compute @workgroup_size(64)
fn integrate2(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= SW.nbody) { return; }
  let o = b * BODY;
  bodies[o + 20u] = bodies[o]; bodies[o + 21u] = bodies[o + 1u]; bodies[o + 22u] = bodies[o + 2u];
  bodies[o + 24u] = bodies[o + 4u]; bodies[o + 25u] = bodies[o + 5u]; bodies[o + 26u] = bodies[o + 6u]; bodies[o + 27u] = bodies[o + 7u];
  let h = P.h;
  if (body_inv_mass(b) != 0.0) {
    let v = vel_of(b) + vec3f(SW.gx, SW.gy, SW.gz) * h;
    set_vel(b, v);
    bodies[o] += v.x * h; bodies[o + 1u] += v.y * h; bodies[o + 2u] += v.z * h;
    let R = quat_mat(body_quat(b));
    let Ib = body_ib(b);
    var w = transpose(R) * om_of(b);
    let tq = -cross(w, Ib * w);
    w += h * tq / Ib;
    let om = R * w;
    set_om(b, om);
    var q = body_quat(b);
    q = normalize(q + 0.5 * h * vec4f(-dot(om, q.yzw), om * q.x + cross(om, q.yzw)));
    bodies[o + 4u] = q.x; bodies[o + 5u] = q.y; bodies[o + 6u] = q.z; bodies[o + 7u] = q.w;
  }
  // the pre-solve velocities (world.js restitution: v0, w0)
  let v0 = vel_of(b); let w0 = om_of(b);
  bext[b * BX + 40u] = v0.x; bext[b * BX + 41u] = v0.y; bext[b * BX + 42u] = v0.z;
  bext[b * BX + 43u] = w0.x; bext[b * BX + 44u] = w0.y; bext[b * BX + 45u] = w0.z;
  // Iinv for the substep (zero for static and sleeping bodies)
  let x = b * BX + 21u;
  let R = quat_mat(body_quat(b));
  let Ib = body_ib(b);
  var d = vec3f(0.0);
  if (body_inv_mass(b) != 0.0) { d = vec3f(select(0.0, 1.0 / Ib.x, Ib.x != 0.0), select(0.0, 1.0 / Ib.y, Ib.y != 0.0), select(0.0, 1.0 / Ib.z, Ib.z != 0.0)); }
  let I = R * mat3x3f(vec3f(d.x, 0.0, 0.0), vec3f(0.0, d.y, 0.0), vec3f(0.0, 0.0, d.z)) * transpose(R);
  for (var r = 0u; r < 3u; r++) { for (var c = 0u; c < 3u; c++) { bext[x + 3u * r + c] = I[c][r]; } }
}

// a body point's move over the substep (world.js slip): (p + R l) - (p_prev + R_prev l), l = R^T r
fn slip(b: u32, r: vec3f) -> vec3f {
  let R = quat_mat(body_quat(b));
  let l = transpose(R) * r;
  return (body_pos(b) + R * l) - (prev_pos(b) + quat_mat(prev_quat(b)) * l);
}

// a pair's node ids (a in .y, b in .z), from its record (kernels.wgsl overlap_xf writes them)
fn pair_nodes(k: u32) -> vec4u { return vec4u(0u, bitcast<u32>(pair_out4[PO * k + 4u].w), bitcast<u32>(pair_out4[PO * k + 1u].w), 0u); }

var<workgroup> red2: array<vec4f, 384>;   // 6 more vec4 per thread: EO and level weights | surface normals per side (2) |
                                          // materials per side (2)
var<workgroup> sw_min: array<f32, 64>;
var<workgroup> wflag: u32;
var<workgroup> wtb: f32;

// ---- (c + d1, full) one workgroup per patch (dispatched for exactly the patch count): the reduction as patch_rs2
// plus world.js's node and level offsets, normal snap, swept time-of-impact contact for fast patches, positional
// Coulomb friction; corrections per side into patch_corr (pjacobi: body_apply_pj); per patch its normal, point and
// the largest |dl| of the substep's sweeps into patch_red (damping). Reduce-only mode (flags bit 2): the plain
// reduction into patch_red (snap normals at the step start).
@compute @workgroup_size(64)
fn patch_step(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let p = wid.y * nwg.x + wid.x;
  let ij = patch_bodies[p];
  let i = ij.x; let j = ij.y;
  let first = (P.flags & 8u) != 0u;
  let reduce_only = (P.flags & 4u) != 0u;
  var v0 = vec4f(0.0); var v1 = vec4f(0.0); var v2 = vec4f(0.0); var v3 = vec4f(0.0);
  var v4 = vec4f(0.0); var v5 = vec4f(0.0);   // EO, WA1-3 | WB1-3, -
  var v6 = vec4f(0.0); var v7 = vec4f(0.0);   // surfaceNormals per side (i, j): sum O n (as n: pushing i from j), sum O
  var v8 = vec4f(0.0); var v9 = vec4f(0.0);   // materials per side: sum O (mu_s, mu_k, v_s, restitution)
  for (var t = patch_starts[p] + li; t < patch_starts[p + 1u]; t += 64u) {
    let k = patch_of[t];
    let o = pair_out4[PO * k]; let pc = pair_out4[PO * k + 1u]; let e3 = pair_out4[PO * k + 3u];
    let O = o.x;
    if (O == 0.0) { continue; }
    let g = o.yzw;   // the raw gradient sum (world.js gA)
    let gu = pair_out4[PO * k + 4u].xyz;   // the O-weighted push direction (world.js gu; oneSided: along the sheet)
    let s0 = O * pair_out4[PO * k + 2u];
    let s1 = O * e3.xy;
    let xc = O * pc.xyz;
    v0 += vec4f(O, gu); v1 += vec4f(g, xc.x); v2 += vec4f(xc.yz, s0.xy); v3 += vec4f(s0.zw, s1);
    let la = u32(e3.w) / 128u; let lb = u32(e3.w) % 128u;
    v4.x += e3.z;
    if (la == 1u) { v4.y += O; } else if (la == 2u) { v4.z += O; } else if (la == 3u) { v4.w += O; }
    if (lb == 1u) { v5.x += O; } else if (lb == 2u) { v5.y += O; } else if (lb == 3u) { v5.z += O; }
    if (SW.materials != 0u) {
      let fm = pair_nodes(k);
      v8 += O * vec4f(slab_at(fm.y * P.sls + 6u), slab_at(fm.y * P.sls + 7u), slab_at(fm.y * P.sls + 8u), slab_at(fm.y * P.sls + 9u));
      v9 += O * vec4f(slab_at(fm.z * P.sls + 6u), slab_at(fm.z * P.sls + 7u), slab_at(fm.z * P.sls + 8u), slab_at(fm.z * P.sls + 9u));
    }
    if (SW.surface_normals != 0u) {
      // world.js surfaceNormals: a thin node's thin axis signed outward (slab slot 5), or for two-sided nodes facing the
      // partner body; j's count as they are, i's negated (n pushes i away from j)
      let f = pair_nodes(k);
      for (var side = 0u; side < 2u; side++) {
        let nk = select(f.z, f.y, side == 0u);
        let b = select(j, i, side == 0u); let o = select(i, j, side == 0u);
        if (!(slab_at(nk * P.sls + 3u) <= 0.25 * nodes[nk * NS + 11u])) { continue; }
        let R = quat_mat(body_quat(b));
        let w = R * vec3f(slab_at(nk * P.sls), slab_at(nk * P.sls + 1u), slab_at(nk * P.sls + 2u));
        var sg = sign(slab_at(nk * P.sls + 5u));   // (slot 5 = sign (2 + D) with solid behind)
        if (sg == 0.0) {
          let m = R * (node_mean(nk) * body_scale(b)) + body_pos(b);
          sg = select(-1.0, 1.0, dot(body_pos(o) - m, w) >= 0.0);
        }
        let add = vec4f(O * sg * select(1.0, -1.0, side == 0u) * w, O);
        if (side == 0u) { v6 += add; } else { v7 += add; }
      }
    }
  }
  red[4u * li] = v0; red[4u * li + 1u] = v1; red[4u * li + 2u] = v2; red[4u * li + 3u] = v3;
  red2[6u * li] = v4; red2[6u * li + 1u] = v5; red2[6u * li + 2u] = v6; red2[6u * li + 3u] = v7;
  red2[6u * li + 4u] = v8; red2[6u * li + 5u] = v9;
  workgroupBarrier();
  for (var hh = 32u; hh > 0u; hh >>= 1u) {
    if (li < hh) {
      for (var c = 0u; c < 4u; c++) { red[4u * li + c] += red[4u * (li + hh) + c]; }
      for (var c = 0u; c < 6u; c++) { red2[6u * li + c] += red2[6u * (li + hh) + c]; }
    }
    workgroupBarrier();
  }
  if (reduce_only) {
    // snap_bp reads slot 0; slot 2 then holds the step's largest contact error (sleepDepth), reset here
    if (li == 0u) { patch_red[4u * p] = red[0]; patch_red[4u * p + 1u] = red[1]; patch_red[4u * p + 2u] = vec4f(0.0); patch_red[4u * p + 3u] = red[3]; }
    return;
  }
  // the totals, kept before the swept pass reuses the shared slots
  let R0 = red[0]; let R1 = red[1]; let R2 = red[2]; let R3 = red[3]; let E0 = red2[0]; let E1 = red2[1]; let SNi = red2[2]; let SNj = red2[3];
  let MA = red2[4]; let MB = red2[5];
  // swept test only for fast pairs (world.js: relative move plus a rotation term over the substep): one flag per
  // patch, made uniform for the barriers below through workgroup memory
  let skip = body_inv_mass(i) == 0.0 && body_inv_mass(j) == 0.0;
  if (li == 0u) {
    let rd = (body_pos(i) - prev_pos(i)) - (body_pos(j) - prev_pos(j));
    let rot = (length(om_of(i)) + length(om_of(j))) * P.h * 0.1;
    wflag = select(0u, 1u, !skip && SW.swept_q > 0.0 && length(rd) + rot > SW.swept_min);
  }
  let do_sw = workgroupUniformLoad(&wflag) != 0u;   // includes the barrier before the slots are reused
  var tbest = 2.0;
  if (do_sw) {
    // first time each pair's q falls through swept_q on straight-line means: min time; penetration-weighted sums
    let si = body_scale(i); let sj = body_scale(j);
    let Ri = quat_mat(body_quat(i)); let Rj = quat_mat(body_quat(j));
    let Pi = quat_mat(prev_quat(i)); let Pj = quat_mat(prev_quat(j));
    var tmin = 2.0; var spen = 0.0; var sn = vec3f(0.0); var sx = vec3f(0.0); var swt = 0.0;
    for (var t = patch_starts[p] + li; t < patch_starts[p + 1u]; t += 64u) {
      let k = patch_of[t];
      let f = pair_nodes(k);
      let a = Ri * (node_mean(f.y) * si) + body_pos(i); let b = Rj * (node_mean(f.z) * sj) + body_pos(j);
      let pa = Pi * (node_mean(f.y) * si) + prev_pos(i); let pb = Pj * (node_mean(f.z) * sj) + prev_pos(j);
      let d0 = pa - pb; let dd = (a - b) - d0;
      let s2 = pair_out4[PO * k + 2u]; let s3 = pair_out4[PO * k + 3u];
      let S = mat3x3f(vec3f(s2.x, s2.y, s2.z), vec3f(s2.y, s2.w, s3.x), vec3f(s2.z, s3.x, s3.y));
      let c0 = cross(S[1], S[2]); let c1 = cross(S[2], S[0]); let c2 = cross(S[0], S[1]);
      let idet = 1.0 / dot(S[0], c0);
      let y0 = (c0 * d0.x + c1 * d0.y + c2 * d0.z) * idet;
      let yd = (c0 * dd.x + c1 * dd.y + c2 * dd.z) * idet;
      let qa = dot(dd, yd); let qb = dot(dd, y0); let qc = dot(d0, y0) - SW.swept_q;
      if (qb < 0.0 && qa > 0.0) {
        let disc = qb * qb - qa * qc;
        var tc = 2.0;
        if (qc <= 0.0) { tc = 0.0; } else if (disc > 0.0) { tc = (-qb - sqrt(disc)) / qa; }
        if (tc < 1.0) {
          let n = y0 + tc * yd; let nl = length(n);
          let pen = -(1.0 - tc) * dot(dd, n) / nl;
          if (pen > 0.0) {
            tmin = min(tmin, tc); spen = max(spen, pen);
            sn += pen * n / nl; sx += pen * 0.5 * (a + b); swt += pen;
          }
        }
      }
    }
    sw_min[li] = tmin;
    red[4u * li] = vec4f(sn, spen); red[4u * li + 1u] = vec4f(sx, swt);
    workgroupBarrier();
    for (var hh = 32u; hh > 0u; hh >>= 1u) {
      if (li < hh) {
        sw_min[li] = min(sw_min[li], sw_min[li + hh]);
        let a0 = red[4u * li]; let b0 = red[4u * (li + hh)];
        red[4u * li] = vec4f(a0.xyz + b0.xyz, max(a0.w, b0.w));
        red[4u * li + 1u] += red[4u * (li + hh) + 1u];
      }
      workgroupBarrier();
    }
    if (li == 0u) { wtb = sw_min[0]; }
    tbest = workgroupUniformLoad(&wtb);
    if (tbest < 1.0) {
      // the better normal: the overlap-weighted push direction of the patch's pairs at the time of impact
      var c2s = 0.0; var g2 = vec3f(0.0); var x2 = vec3f(0.0);
      for (var t = patch_starts[p] + li; t < patch_starts[p + 1u]; t += 64u) {
        let k = patch_of[t];
        let f = pair_nodes(k);
        let ca = Ri * (node_mean(f.y) * si) + body_pos(i); let cb = Rj * (node_mean(f.z) * sj) + body_pos(j);
        let pa = Pi * (node_mean(f.y) * si) + prev_pos(i); let pb = Pj * (node_mean(f.z) * sj) + prev_pos(j);
        let a = pa + tbest * (ca - pa); let b = pb + tbest * (cb - pb);
        let Ca = Ri * node_cov(f.y) * transpose(Ri) * (si * si); let Cb = Rj * node_cov(f.z) * transpose(Rj) * (sj * sj);
        let S = Ca + Cb;
        let r0 = cross(S[1], S[2]); let r1 = cross(S[2], S[0]); let r2 = cross(S[0], S[1]);
        let det = dot(S[0], r0);
        let d = a - b;
        let y = (r0 * d.x + r1 * d.y + r2 * d.z) / det;
        let q = dot(d, y);
        if (q > 40.0) { continue; }
        let O = nodes[f.y * NS + 9u] * si * si * si * nodes[f.z * NS + 9u] * sj * sj * sj * exp(-0.5 * q) / (PI32 * sqrt(det));
        let g = -O * y; let gl = length(g);
        if (O > 0.0 && gl > 0.0) {
          c2s += O; g2 -= O * g / gl;
          x2 += O * ((b + Cb * y) + (1.0 - tbest) * (body_pos(i) - prev_pos(i)));
        }
      }
      red[4u * li + 2u] = vec4f(g2, c2s); red[4u * li + 3u] = vec4f(x2, 0.0);   // slots 2-3: unused by the first pass
      workgroupBarrier();
      for (var hh = 32u; hh > 0u; hh >>= 1u) {
        if (li < hh) { red[4u * li + 2u] += red[4u * (li + hh) + 2u]; red[4u * li + 3u] += red[4u * (li + hh) + 3u]; }
        workgroupBarrier();
      }
    }
  }
  if (li != 0u) { return; }
  // ---- thread 0: the patch's contact and friction
  // the patch's combined material (world.js materials: geometric means of the sides' O-weighted mu_s, mu_k, v_s)
  var erest = 0.0;   // the patch's restitution (world.js patchMat: the larger side's)
  if (SW.materials != 0u && R0.x > 0.0) {
    let a = MA / R0.x; let b = MB / R0.x;
    panc[4u * p + 2u] = vec4f(panc[4u * p + 2u].xy, sqrt(a.x * b.x), sqrt(a.y * b.y));
    panc[4u * p + 3u].w = sqrt(a.z * b.z);
    erest = max(a.w, b.w);
  }
  for (var c = 0u; c < 4u; c++) { patch_corr[4u * p + c] = vec4f(0.0); }
  var pl = 0.0;
  if (!first) { pl = patch_red[4u * p].w; }
  if (skip) { patch_red[4u * p] = vec4f(patch_red[4u * p].xyz, pl); return; }
  var n = vec3f(0.0); var cpt = vec3f(0.0); var Cneg = 0.0; var ok = false;
  if (tbest < 1.0) {
    let a0 = red[0]; let a1 = red[1];
    n = a0.xyz / length(a0.xyz); cpt = a1.xyz / a1.w;
    let g2 = red[2]; let x2 = red[3];
    if (g2.w > 0.0 && length(g2.xyz) > 0.0) { n = g2.xyz / length(g2.xyz); cpt = x2.xyz / g2.w; }
    Cneg = -a0.w;
    ok = true;
  }
  if (!ok) {
    let r0 = R0; let r1 = R1; let r2 = R2; let r3 = R3;
    let C = r0.x;
    if (C > 1e-30 && length(r0.yzw) > 0.0) {
      n = -r0.yzw / length(r0.yzw);
      cpt = vec3f(r1.w, r2.x, r2.y) / C;
      let pn = panc[4u * p + 3u].xyz;
      if (SW.surface_normals != 0u && SNi.w + SNj.w >= C && length(SNi.xyz + SNj.xyz) > 0.0) {
        n = (SNi.xyz + SNj.xyz) / length(SNi.xyz + SNj.xyz);   // mostly thin nodes: the O-weighted surface normal
      }
      else if (SW.snap_on != 0.0 && dot(n, pn) > SW.cos_snap) { n = pn; }
      let OS = array<f32, 6>(r2.z, r2.w, r3.x, r3.y, r3.z, r3.w);
      let s2 = (n.x * n.x * OS[0] + n.y * n.y * OS[3] + n.z * n.z * OS[5] + 2.0 * (n.x * n.y * OS[1] + n.x * n.z * OS[2] + n.y * n.z * OS[4])) / C;
      let dsep = s2 * max(-dot(r1.xyz, n), 0.0) / C;
      // level offsets along the bodies' box axes, weighted by the normal's components there (world.js lo)
      var lo = 0.0;
      let W1 = E1; let W0 = E0;
      for (var side = 0u; side < 2u; side++) {
        let b = select(j, i, side == 0u);
        let w = select(vec3f(W1.x, W1.y, W1.z), vec3f(W0.y, W0.z, W0.w), side == 0u);
        if (w.x + w.y + w.z == 0.0) { continue; }
        let nb = transpose(quat_mat(body_quat(b))) * n;
        for (var l = 1u; l < 4u; l++) {
          let wl = w[l - 1u];
          if (wl == 0.0) { continue; }
          for (var k2 = 0u; k2 < 3u; k2++) {
            let ax = vec3f(bext[b * BX + 3u * k2], bext[b * BX + 3u * k2 + 1u], bext[b * BX + 3u * k2 + 2u]);
            let a = dot(ax, nb);
            lo += wl * a * a * bext[b * BX + 9u + 3u * l + k2];
          }
        }
      }
      var Cv = P.kappa * sqrt(s2) + (W0.x + lo) / C - dsep;
      if (Cv > 0.0) {
        patch_red[4u * p + 2u].x = max(patch_red[4u * p + 2u].x, Cv);   // the step's largest error (sleepDepth)
        let ra = cpt - body_pos(i); let rb = cpt - body_pos(j);
        let vapp = max(-dot((vel_of(i) + cross(om_of(i), ra)) - (vel_of(j) + cross(om_of(j), rb)), n), 0.0);
        Cv = min(Cv, (P.maxdep + vapp) * P.h) * P.relax;
        Cneg = -Cv;
        ok = true;
      }
    }
  }
  if (!ok) { patch_red[4u * p] = vec4f(patch_red[4u * p].xyz, pl); return; }
  // world.js contact(): one normal correction along n at the patch point
  let mi = body_inv_mass(i); let mj = body_inv_mass(j);
  let ra = cpt - body_pos(i); let rb = cpt - body_pos(j);
  let ta = cross(ra, -n); let tb = cross(rb, n);
  let wsum = mi + dot(ta, iinv(i, ta)) + mj + dot(tb, iinv(j, tb));
  let dl = Cneg / (wsum + P.alpha_h2);
  var li_ = -n * mi * dl; var ai_ = iinv(i, ta) * dl;
  var lj_ = n * mj * dl; var aj_ = iinv(j, tb) * dl;
  let ln = abs(dl);
  // positional Coulomb friction at the patch point (slip since the substep start, before this sweep's corrections);
  // Stribeck coefficient at this substep's slip speed; with anchors, the anchored points' separation once anchored,
  // held while within mu_s |dl|, else sliding (re-anchored at the substep's end)
  if (SW.mu > 0.0 || SW.mu_s > 0.0 || SW.materials != 0u) {
    var dp = slip(i, ra) - slip(j, rb);
    var tv = dp - dot(dp, n) * n;
    let vslip = length(tv) / P.h;
    var anchored = false;
    if (SW.anchors_on != 0u) {
      panc[4u * p + 2u] = vec4f(1.0, 1.0, panc[4u * p + 2u].zw);   // touched this frame, and this substep
      let a0 = panc[4u * p]; let a1 = panc[4u * p + 1u];
      if (a0.w != 0.0) {
        dp = (body_pos(i) + quat_mat(body_quat(i)) * a0.xyz) - (body_pos(j) + quat_mat(body_quat(j)) * a1.xyz);
        tv = dp - dot(dp, n) * n;
        anchored = true;
      } else {
        panc[4u * p + 1u].w = 1.0;   // new: anchor at the substep's end
      }
    }
    let m = length(tv);
    if (m > 1e-12) {
      let u = tv / m;
      let ca = cross(ra, u); let cb = cross(rb, u);
      let wt = mi + dot(ca, iinv(i, ca)) + mj + dot(cb, iinv(j, cb));
      let need = m / wt;
      var dlt = -min(need, mu_at_p(p, vslip) * ln);
      if (anchored) {
        if (need <= mu_at_p(p, 0.0) * ln) { dlt = -need; } else { panc[4u * p + 1u].w = 1.0; }   // held, or slid
      }
      li_ += u * mi * dlt; ai_ += iinv(i, ca) * dlt;
      lj_ += -u * mj * dlt; aj_ += iinv(j, -cb) * dlt;
    }
  }
  patch_corr[4u * p] = vec4f(li_, select(0.0, 1.0, mi != 0.0));
  patch_corr[4u * p + 1u] = vec4f(ai_, 0.0);
  patch_corr[4u * p + 2u] = vec4f(lj_, select(0.0, 1.0, mj != 0.0));
  patch_corr[4u * p + 3u] = vec4f(aj_, 0.0);
  patch_red[4u * p] = vec4f(n, max(pl, ln));
  patch_red[4u * p + 1u] = vec4f(cpt, erest);
}

// ---- damping colours (world.js colorOrder, one thread, once per step): body pairs by patch count, largest first,
// each the smallest colour not used by its dynamic (awake) bodies; per colour the indirect dispatch arguments (zero
// workgroups for colours not used). More than NCOL colours share the last one (flagged: gstate[1] bit 1).
var<private> bcol: array<u32, MAXB>;   // per body: colours used (bits)
@compute @workgroup_size(1)
fn damp_colors() {
  let nbp = atomicLoad(&gstate[0]);
  let np = atomicLoad(&sstate[1]);
  for (var bp = 0u; bp < nbp; bp++) { bpflag[bp] = 0u; bpcolor[bp] = 0xffffffffu; }
  for (var p = 0u; p < np; p++) { bpflag[patch_bp[p]] += 1u; }
  for (var b = 0u; b < min(SW.nbody, MAXB); b++) { bcol[b] = 0u; }
  var ncol = 0u;
  loop {
    var best = 0xffffffffu; var bc = 0u;
    for (var bp = 0u; bp < nbp; bp++) {
      if (bpcolor[bp] == 0xffffffffu && bpflag[bp] > 0u && (best == 0xffffffffu || bpflag[bp] > bc)) { best = bp; bc = bpflag[bp]; }
    }
    if (best == 0xffffffffu) { break; }
    let info = bp_info[best];
    let i = u32(info.x); let j = u32(info.y);
    var mask = 0u;
    if (body_inv_mass(i) > 0.0) { mask |= bcol[i]; }
    if (body_inv_mass(j) > 0.0) { mask |= bcol[j]; }
    var c = 0u;
    loop { if (c >= NCOL - 1u || (mask & (1u << c)) == 0u) { break; } c++; }
    if ((mask & (1u << c)) != 0u) { atomicOr(&gstate[1], 2u); }
    bpcolor[best] = c; bcol[i] |= 1u << c; bcol[j] |= 1u << c;
    ncol = max(ncol, c + 1u);
  }
  let pg = (np + 63u) / 64u; let bg = (SW.nbody + 63u) / 64u;
  for (var c = 0u; c < NCOL; c++) {
    let on = c < ncol;
    cargs[6u * c] = select(0u, pg, on); cargs[6u * c + 1u] = 1u; cargs[6u * c + 2u] = 1u;
    cargs[6u * c + 3u] = select(0u, bg, on); cargs[6u * c + 4u] = 1u; cargs[6u * c + 5u] = 1u;
  }
}

// ---- damping (world.js, gpuForm): per patch with a contact this substep, the normal impulse that removes damp of
// the relative normal speed, per side into patch_corr (velocity change, torque arm x impulse)
@compute @workgroup_size(64)
fn damp_patch(@builtin(global_invocation_id) id: vec3u) {
  let p = id.x;
  if (p >= atomicLoad(&sstate[1])) { return; }
  for (var c = 0u; c < 4u; c++) { patch_corr[4u * p + c] = vec4f(0.0); }
  if (SW.damp_form == 3u && bpcolor[patch_bp[p]] != P.count) { return; }   // colour damping: this dispatch's colour only
  for (var c = 0u; c < 4u; c++) { stick_corr[4u * p + c] = vec4f(0.0); }   // every patch has one colour: written once
  let pr = patch_red[4u * p];
  if (pr.w <= 0.0) { return; }
  let ij = patch_bodies[p];
  let i = ij.x; let j = ij.y;
  let n = pr.xyz; let c = patch_red[4u * p + 1u].xyz;
  let ra = c - body_pos(i); let rb = c - body_pos(j);
  let vn = dot((vel_of(i) + cross(om_of(i), ra)) - (vel_of(j) + cross(om_of(j), rb)), n);
  let ca = cross(ra, n); let cb = cross(rb, n);
  let mi = body_inv_mass(i); let mj = body_inv_mass(j);
  let wn = mi + dot(ca, iinv(i, ca)) + mj + dot(cb, iinv(j, cb));
  if (wn <= 0.0) { return; }
  var Pm = -P.damp * vn / wn;
  if (SW.restitution != 0u) {
    // world.js restitution: the normal velocity to -e times the pre-solve approach (0 below 2 |g| h: resting)
    let w0i = vec3f(bext[i * BX + 43u], bext[i * BX + 44u], bext[i * BX + 45u]); let w0j = vec3f(bext[j * BX + 43u], bext[j * BX + 44u], bext[j * BX + 45u]);
    let v0i = vec3f(bext[i * BX + 40u], bext[i * BX + 41u], bext[i * BX + 42u]); let v0j = vec3f(bext[j * BX + 40u], bext[j * BX + 41u], bext[j * BX + 42u]);
    let vpre = dot((v0i + cross(w0i, ra)) - (v0j + cross(w0j, rb)), n);
    let thr = 2.0 * length(vec3f(SW.gx, SW.gy, SW.gz)) * P.h;
    Pm = (select(0.0, -patch_red[4u * p + 1u].w * vpre, vpre < -thr) - vn) / wn;
  }
  if (SW.stick_v > 0.0) {
    // velocity-level static friction (world.js stickV): a slow tangential slip at the patch point is removed when mu
    // times the substep's normal impulse (|dl| / h) allows; per body the mean over its sticking patches (stick_body)
    let vr = (vel_of(i) + cross(om_of(i), ra)) - (vel_of(j) + cross(om_of(j), rb));
    let vtv = vr - dot(vr, n) * n;
    let vt = length(vtv);
    if (vt > 1e-9 && vt < SW.stick_v) {
      let u = vtv / vt;
      let sa = cross(ra, u); let sb = cross(rb, u);
      let wt = mi + dot(sa, iinv(i, sa)) + mj + dot(sb, iinv(j, sb));
      if (wt > 0.0 && vt / wt <= mu_at_p(p, vt) * pr.w / P.h) {
        let J = -vt / wt;
        stick_corr[4u * p] = vec4f(mi * J * u, select(0.0, 1.0, mi != 0.0));
        stick_corr[4u * p + 1u] = vec4f(sa * J, 0.0);
        stick_corr[4u * p + 2u] = vec4f(-mj * J * u, select(0.0, 1.0, mj != 0.0));
        stick_corr[4u * p + 3u] = vec4f(-sb * J, 0.0);
      }
    }
  }
  patch_corr[4u * p] = vec4f(mi * Pm * n, select(0.0, 1.0, mi != 0.0));
  patch_corr[4u * p + 1u] = vec4f(ca * Pm, 0.0);
  patch_corr[4u * p + 2u] = vec4f(-mj * Pm * n, select(0.0, 1.0, mj != 0.0));
  patch_corr[4u * p + 3u] = vec4f(-cb * Pm, 0.0);
}

// per body (world.js gpuForm dampApply): per body pair the mean damping impulse, scaled by (1 - (1 - damp)^n) / damp
// for the pair's n patches (Gauss-Seidel damps in turn) unless the form is plain; summed over the body pairs, or
// their mean (form 1): velocity, and Iinv times the arms for the angular velocity
@compute @workgroup_size(64)
fn damp_body(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= P.nbody || body_inv_mass(b) == 0.0) { return; }
  var other: array<u32, 8>; var lin: array<vec3f, 8>; var arm: array<vec3f, 8>; var cnt: array<f32, 8>;
  var used = 0u;
  for (var t = body_patch_start[b]; t < body_patch_start[b + 1u]; t++) {
    let e = body_patch[t]; let p = e >> 1u; let side = e & 1u;
    let l = patch_corr[4u * p + 2u * side];
    if (l.w == 0.0) { continue; }
    let ij = patch_bodies[p];
    let o = select(ij.y, ij.x, side == 1u);
    var s = 0u;
    loop {
      if (s == used) { if (used < 8u) { other[s] = o; lin[s] = vec3f(0.0); arm[s] = vec3f(0.0); cnt[s] = 0.0; used++; } else { s = 7u; } break; }
      if (other[s] == o) { break; }
      s++;
    }
    lin[s] += l.xyz; arm[s] += patch_corr[4u * p + 2u * side + 1u].xyz; cnt[s] += 1.0;
  }
  var dv = vec3f(0.0); var da = vec3f(0.0);
  let d = select(P.damp, 1.0, SW.restitution != 0u);   // restitution: a full target, the pair mean
  for (var s = 0u; s < used; s++) {
    var f = (1.0 - pow(1.0 - d, cnt[s])) / (cnt[s] * d);
    if (SW.damp_form == 2u) { f = 1.0 / cnt[s]; }   // 3 (colour): one body pair per body in a colour, as 0
    if (SW.damp_form == 1u) { f /= f32(used); }
    dv += lin[s] * f; da += arm[s] * f;
  }
  set_vel(b, vel_of(b) + dv);
  set_om(b, om_of(b) + iinv(b, da));
}

// per body after the damping colours: the mean over its sticking patches of their static-friction impulses
@compute @workgroup_size(64)
fn stick_body(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= P.nbody || body_inv_mass(b) == 0.0) { return; }
  var dv = vec3f(0.0); var da = vec3f(0.0); var n = 0.0;
  for (var t = body_patch_start[b]; t < body_patch_start[b + 1u]; t++) {
    let e = body_patch[t]; let p = e >> 1u; let side = e & 1u;
    let l = stick_corr[4u * p + 2u * side];
    if (l.w == 0.0) { continue; }
    dv += l.xyz; da += stick_corr[4u * p + 2u * side + 1u].xyz; n += 1.0;
  }
  if (n == 0.0) { return; }
  set_vel(b, vel_of(b) + dv / n);
  set_om(b, om_of(b) + iinv(b, da / n));
}

var<private> par: array<u32, MAXB>;
var<private> dep: array<f32, MAXB>;
fn find(a0: u32) -> u32 {
  var a = a0;
  loop { if (par[a] == a) { break; } par[a] = par[par[a]]; a = par[a]; }
  return a;
}

// ---- step end (one thread): world.js sleep (still timers, island sleep) and the step's record
@compute @workgroup_size(1)
fn sleep_step() {
  let B = min(SW.nbody, MAXB);
  // per body the largest contact error of the step (world.js depthB, sleepDepth)
  for (var b = 0u; b < B; b++) { dep[b] = 0.0; }
  let npd = atomicLoad(&sstate[1]);
  for (var p = 0u; p < npd; p++) { let ij = patch_bodies[p]; let e = patch_red[4u * p + 2u].x; dep[ij.x] = max(dep[ij.x], e); dep[ij.y] = max(dep[ij.y], e); }
  for (var b = 0u; b < B; b++) {
    if (body_inv_mass_raw(b) == 0.0 || asleep(b)) { continue; }
    var sp = length(vel_of(b)) + length(om_of(b)) * bext[b * BX + 34u];
    if (SW.sleep_net != 0u) {   // net motion over the step: |dp| / dt + angle / dt x sleepL (angle ~ 2 |q - q0|, the closer sign)
      let o = b * BX; let q = body_quat(b);
      let q0 = vec4f(bext[o + 49u], bext[o + 50u], bext[o + 51u], bext[o + 52u]);
      let dp = body_pos(b) - vec3f(bext[o + 46u], bext[o + 47u], bext[o + 48u]);
      sp = (length(dp) + 2.0 * min(length(q - q0), length(q + q0)) * bext[o + 34u]) / SW.dt;
    }
    bext[b * BX + 35u] = sp;
    let deep = SW.sleep_depth > 0.0 && dep[b] > SW.sleep_depth;
    // net displacement since the still window began (world.js sleepDisp)
    var moved = false;
    if (SW.sleep_disp > 0.0) {
      let o = b * BX + 37u;
      if (bext[b * BX + 33u] == 0.0) { bext[o] = body_pos(b).x; bext[o + 1u] = body_pos(b).y; bext[o + 2u] = body_pos(b).z; }
      moved = length(body_pos(b) - vec3f(bext[o], bext[o + 1u], bext[o + 2u])) > SW.sleep_disp;
      if (moved) { bext[o] = body_pos(b).x; bext[o + 1u] = body_pos(b).y; bext[o + 2u] = body_pos(b).z; }
    }
    bext[b * BX + 33u] = select(0.0, bext[b * BX + 33u] + SW.dt, sp < SW.sleep_v && !deep && !moved);
    if (SW.island_on == 0.0 && bext[b * BX + 33u] > SW.sleep_t) { set_asleep(b, true); set_vel(b, vec3f(0.0)); set_om(b, vec3f(0.0)); }
  }
  if (SW.island_on != 0.0) {
    for (var b = 0u; b < B; b++) { par[b] = b; }
    let nbp = atomicLoad(&gstate[0]);
    for (var bp = 0u; bp < nbp; bp++) { bpflag[bp] = 0u; }
    let np = atomicLoad(&sstate[1]);
    for (var p = 0u; p < np; p++) { bpflag[patch_bp[p]] = 1u; }
    for (var bp = 0u; bp < nbp; bp++) {
      let info = bp_info[bp];
      let i = u32(info.x); let j = u32(info.y);
      if (bpflag[bp] != 0u && body_inv_mass_raw(i) > 0.0 && body_inv_mass_raw(j) > 0.0) {
        let ri = find(i); let rj = find(j);
        if (ri != rj) { par[max(ri, rj)] = min(ri, rj); }
      }
    }
    for (var r = 0u; r < B; r++) {
      if (body_inv_mass_raw(r) == 0.0 || find(r) != r) { continue; }
      var any_awake = false; var all_still = true;
      for (var b = 0u; b < B; b++) {
        if (body_inv_mass_raw(b) > 0.0 && find(b) == r && !asleep(b)) { any_awake = true; if (!(bext[b * BX + 33u] > SW.sleep_t)) { all_still = false; } }
      }
      if (!any_awake) { continue; }
      for (var b = 0u; b < B; b++) {
        if (body_inv_mass_raw(b) == 0.0 || find(b) != r) { continue; }
        if (all_still) { if (!asleep(b)) { set_asleep(b, true); set_vel(b, vec3f(0.0)); set_om(b, vec3f(0.0)); } }
        else if (asleep(b)) { set_asleep(b, false); bext[b * BX + 33u] = 0.0; }
      }
    }
  }
}

// ---- step end (one thread, after sleep_step): the step's record (poses and sleep flags, counts and overflow flags;
// gpu_world.py reads them) and the step counter; its own kernel so sleep_step stays within 10 storage buffers
@compute @workgroup_size(1)
fn step_record() {
  let B = SW.nbody;
  let st = atomicLoad(&gstate[2]);
  let o = st * B * 9u;
  if (o + B * 9u <= arrayLength(&hist)) {
    for (var b = 0u; b < B; b++) {
      let q = body_quat(b); let x = body_pos(b);
      hist[o + 9u * b] = x.x; hist[o + 9u * b + 1u] = x.y; hist[o + 9u * b + 2u] = x.z;
      hist[o + 9u * b + 3u] = q.x; hist[o + 9u * b + 4u] = q.y; hist[o + 9u * b + 5u] = q.z; hist[o + 9u * b + 6u] = q.w;
      hist[o + 9u * b + 7u] = select(0.0, 1.0, asleep(b)); hist[o + 9u * b + 8u] = bext[b * BX + 35u];
    }
  }
  if (4u * st + 3u < arrayLength(&shist)) {
    shist[4u * st] = atomicLoad(&dstate[1]); shist[4u * st + 1u] = atomicLoad(&sstate[1]);   // raw detected pairs (past the list: overflow)
    shist[4u * st + 2u] = atomicLoad(&gstate[0]);
    shist[4u * st + 3u] = atomicLoad(&dstate[5]) | (atomicLoad(&sstate[2]) << 1u) | (atomicLoad(&gstate[1]) << 2u);
  }
  atomicStore(&gstate[2], st + 1u);
}
