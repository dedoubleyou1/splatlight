// GPU physics benchmark kernels (bench.py): the maths of renderer/src/physics/world.js (mode-3 overlap contact) in
// f32, laid out for a GPU. Benchmark only: the solve is a Jacobi variant (all patches against the same body state,
// corrections averaged per body), not world.js's Gauss-Seidel, so its trajectories are not compared, only its cost.
//
// Buffers (f32 unless noted):
//   nodes   16 per node (world.js NS layout: mean xyz, cov xx xy xz yy yz zz, mass, reach, size, thick, left, right, off)
//   slab    SLS per node (thin axis xyz, half-thickness, level)
//   bodies  BODY per body: pos xyz, scale | quat wxyz | vel xyz, invMass | om xyz, pad | Ib xyz, pad | prev pos xyz, pad
//           | prev quat wxyz. bodies[19]: asleep (step.wgsl; a sleeping body is static for the solver, world.js invS)

const BODY: u32 = 28u;
const NS: u32 = 16u;
const PI32: f32 = 15.749609945722419;   // (2 pi)^1.5
const PO: u32 = 5u;   // pair_out4 vec4 per pair

struct Params {
  count: u32,      // threads that do work in this dispatch (nodes, pairs, patches, bodies, frontier entries)
  nbody: u32,
  sls: u32,
  flags: u32,      // bit 0: descend() keeps a leaf pair only below qcut (world.detect's test); bit 1: per body pair level
                   // caps (bcap: a node tagged at that level or finer is a leaf); step.wgsl: bit 2 reduce only, bit 3 first sweep;
                   // bit 4: world.js directedMargin (descent along each node pair's motion, pad1 = 1.5 dt); bit 5: world.js
                   // oneSided 'soft' (solid-backed sheets: slab slot 5 = sign (2 + D)); bit 6: world.js refineNeed (slab slot
                   // 11 = every node's extent error)
  h: f32,          // substep
  kappa: f32,
  relax: f32,
  maxdep: f32,
  alpha_h2: f32,   // alpha / h^2
  gz: f32,
  damp: f32,
  qcut: f32,
  patch_size: f32, // world.makePatches' cell size
  pad1: f32,       // directedMargin: the motion horizon 1.5 dt
  slab_off: f32,   // the slab block's offset in nodes (f32 index; exact below 2^24)
  need_tol: f32,   // world.js refineNeed (flags bit 6): a thin node pair with extent errors under this stops descending
}

fn quat_mat(q: vec4f) -> mat3x3f {
  let w = q.x; let x = q.y; let y = q.z; let z = q.w;
  // columns of R (WGSL matrices are column-major)
  return mat3x3f(
    vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
    vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
    vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}

fn body_pos(b: u32) -> vec3f { let o = b * BODY; return vec3f(bodies[o], bodies[o + 1u], bodies[o + 2u]); }
fn body_scale(b: u32) -> f32 { return bodies[b * BODY + 3u]; }
fn body_quat(b: u32) -> vec4f { let o = b * BODY + 4u; return vec4f(bodies[o], bodies[o + 1u], bodies[o + 2u], bodies[o + 3u]); }
fn body_inv_mass(b: u32) -> f32 { return select(bodies[b * BODY + 11u], 0.0, bodies[b * BODY + 19u] != 0.0); }   // the solver's (invS)
fn body_inv_mass_raw(b: u32) -> f32 { return bodies[b * BODY + 11u]; }
fn body_ib(b: u32) -> vec3f { let o = b * BODY + 16u; return vec3f(bodies[o], bodies[o + 1u], bodies[o + 2u]); }

fn node_mean(k: u32) -> vec3f { let o = k * NS; return vec3f(nodes[o], nodes[o + 1u], nodes[o + 2u]); }
fn node_cov(k: u32) -> mat3x3f {
  let o = k * NS;
  let xx = nodes[o + 3u]; let xy = nodes[o + 4u]; let xz = nodes[o + 5u];
  let yy = nodes[o + 6u]; let yz = nodes[o + 7u]; let zz = nodes[o + 8u];
  return mat3x3f(vec3f(xx, xy, xz), vec3f(xy, yy, yz), vec3f(xz, yz, zz));
}

// world-frame inverse inertia of body b
fn inv_inertia(b: u32) -> mat3x3f {
  let R = quat_mat(body_quat(b));
  let ib = body_ib(b);
  var d = vec3f(0.0);
  if (ib.x > 0.0) { d = vec3f(1.0 / ib.x, 1.0 / ib.y, 1.0 / ib.z); }
  if (body_inv_mass(b) == 0.0) { d = vec3f(0.0); }
  return R * mat3x3f(vec3f(d.x, 0.0, 0.0), vec3f(0.0, d.y, 0.0), vec3f(0.0, 0.0, d.z)) * transpose(R);
}

@group(0) @binding(0) var<storage, read> nodes: array<f32>;
@group(0) @binding(1) var<storage, read_write> bodies: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

// ---- (a) node transform: per active (node, body) instance, world mean (xyz, mass * s^3) and covariance (6, x s^2)
@group(0) @binding(3) var<storage, read> act_inst: array<vec2u>;      // node, body
@group(0) @binding(4) var<storage, read_write> wmean: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> wcov: array<vec4f>;  // 2 per instance: xx xy xz yy | yz zz - -

@compute @workgroup_size(64)
fn node_xf(@builtin(global_invocation_id) id: vec3u) {
  let t = id.x;
  if (t >= P.count) { return; }
  let k = act_inst[t].x; let b = act_inst[t].y;
  let s = body_scale(b);
  let R = quat_mat(body_quat(b));
  let m = R * (node_mean(k) * s) + body_pos(b);
  wmean[t] = vec4f(m, nodes[k * NS + 9u] * s * s * s);
  let C = R * node_cov(k) * transpose(R) * (s * s);
  wcov[2u * t] = vec4f(C[0].x, C[1].x, C[2].x, C[1].y);
  wcov[2u * t + 1u] = vec4f(C[2].y, C[2].z, 0.0, 0.0);
}

// ---- (b) pair overlap: one thread per node pair (instances ia, ib): O, dO/dmu_a, contact point, r
@group(0) @binding(6) var<storage, read> pair_inst: array<vec2u>;
@group(0) @binding(7) var<storage, read_write> pair_out: array<vec4f>;  // 2 per pair: O gx gy gz | px py pz r

@compute @workgroup_size(64)
fn overlap(@builtin(global_invocation_id) id: vec3u) {
  let t = id.x;
  if (t >= P.count) { return; }
  let ia = pair_inst[t].x; let ib = pair_inst[t].y;
  let A = wmean[ia]; let Bm = wmean[ib];
  let a0 = wcov[2u * ia]; let a1 = wcov[2u * ia + 1u];
  let b0 = wcov[2u * ib]; let b1 = wcov[2u * ib + 1u];
  let sxx = a0.x + b0.x; let sxy = a0.y + b0.y; let sxz = a0.z + b0.z;
  let syy = a0.w + b0.w; let syz = a1.x + b1.x; let szz = a1.y + b1.y;
  let c00 = syy * szz - syz * syz; let c01 = sxz * syz - sxy * szz; let c02 = sxy * syz - sxz * syy;
  let det = sxx * c00 + sxy * c01 + sxz * c02;
  let c11 = sxx * szz - sxz * sxz; let c12 = sxy * sxz - sxx * syz; let c22 = sxx * syy - sxy * sxy;
  let idet = 1.0 / det;
  let d = A.xyz - Bm.xyz;
  let y = vec3f(c00 * d.x + c01 * d.y + c02 * d.z, c01 * d.x + c11 * d.y + c12 * d.z, c02 * d.x + c12 * d.y + c22 * d.z) * idet;
  let q = dot(d, y);
  if (q > 40.0) { pair_out[2u * t] = vec4f(0.0); pair_out[2u * t + 1u] = vec4f(0.0); return; }
  let r = exp(-0.5 * q);
  let O = A.w * Bm.w * r / (PI32 * sqrt(det));
  pair_out[2u * t] = vec4f(O, -O * y);
  let p = Bm.xyz + vec3f(b0.x * y.x + b0.y * y.y + b0.z * y.z, b0.y * y.x + b0.w * y.y + b1.x * y.z, b0.z * y.x + b1.x * y.y + b1.y * y.z);
  pair_out[2u * t + 1u] = vec4f(p, r);
}

// ---- (c) per-patch segmented reduction: C, unit-gradient sum gu, gradient sum gA, contact point, O-weighted cov sum
@group(0) @binding(8) var<storage, read> patch_starts: array<u32>;
@group(0) @binding(9) var<storage, read> patch_of: array<u32>;       // pair index per slot, patch order
@group(0) @binding(10) var<storage, read_write> patch_red: array<vec4f>;  // 4 per patch: C gu | gA xc.x | xc.yz OS0 OS1 | OS2..OS5
@group(0) @binding(11) var<storage, read> pair_cov_inst: array<vec2u>;   // instance ids again (for the cov sums)

@compute @workgroup_size(64)
fn patch_reduce(@builtin(global_invocation_id) id: vec3u) {
  let p = id.x;
  if (p >= P.count) { return; }
  var C = 0.0; var gu = vec3f(0.0); var gA = vec3f(0.0); var xc = vec3f(0.0);
  var os0 = vec4f(0.0); var os1 = vec2f(0.0);
  for (var t = patch_starts[p]; t < patch_starts[p + 1u]; t++) {
    let k = patch_of[t];
    let o = pair_out[2u * k]; let pc = pair_out[2u * k + 1u];
    let O = o.x;
    if (O == 0.0) { continue; }
    C += O;
    let g = o.yzw; let gl = length(g);
    if (gl > 0.0) { gu += O * g / gl; }
    gA += g;
    xc += O * pc.xyz;
    let ia = pair_cov_inst[k].x; let ib = pair_cov_inst[k].y;
    os0 += O * (wcov[2u * ia] + wcov[2u * ib]);
    os1 += O * (wcov[2u * ia + 1u].xy + wcov[2u * ib + 1u].xy);
  }
  patch_red[4u * p] = vec4f(C, gu);
  patch_red[4u * p + 1u] = vec4f(gA, xc.x);
  patch_red[4u * p + 2u] = vec4f(xc.yz, os0.xy);
  patch_red[4u * p + 3u] = vec4f(os0.zw, os1);
}

// ---- (d1) patch solve (mode 3, Jacobi): per patch the correction for each of its two bodies (linear, angular)
@group(0) @binding(12) var<storage, read> patch_bodies: array<vec2u>;
@group(0) @binding(13) var<storage, read_write> patch_corr: array<vec4f>;  // 4 per patch: lin_i dl | ang_i - | lin_j - | ang_j -

@compute @workgroup_size(64)
fn patch_solve(@builtin(global_invocation_id) id: vec3u) {
  let p = id.x;
  if (p >= P.count) { return; }
  solve_patch(p, patch_red[4u * p], patch_red[4u * p + 1u], patch_red[4u * p + 2u], patch_red[4u * p + 3u]);
}

// patch p's correction from its reduction r0..r3 (patch_red layout)
fn solve_patch(p: u32, r0: vec4f, r1: vec4f, r2: vec4f, r3: vec4f) {
  let C = r0.x;
  patch_corr[4u * p] = vec4f(0.0); patch_corr[4u * p + 1u] = vec4f(0.0); patch_corr[4u * p + 2u] = vec4f(0.0); patch_corr[4u * p + 3u] = vec4f(0.0);
  if (C <= 1e-30) { return; }
  let gn = length(r0.yzw);
  if (gn <= 0.0) { return; }
  let n = -r0.yzw / gn;
  let xc = vec3f(r1.w, r2.x, r2.y) / C;
  let OS = array<f32, 6>(r2.z, r2.w, r3.x, r3.y, r3.z, r3.w);
  let s2 = (n.x * n.x * OS[0] + n.y * n.y * OS[3] + n.z * n.z * OS[5] + 2.0 * (n.x * n.y * OS[1] + n.x * n.z * OS[2] + n.y * n.z * OS[4])) / C;
  let dsep = s2 * max(-dot(r1.xyz, n), 0.0) / C;
  var Cv = P.kappa * sqrt(s2) - dsep;
  if (Cv <= 0.0) { return; }
  let i = patch_bodies[p].x; let j = patch_bodies[p].y;
  let ra = xc - body_pos(i); let rb = xc - body_pos(j);
  let vi = vec3f(bodies[i * BODY + 8u], bodies[i * BODY + 9u], bodies[i * BODY + 10u]);
  let wi = vec3f(bodies[i * BODY + 12u], bodies[i * BODY + 13u], bodies[i * BODY + 14u]);
  let vj = vec3f(bodies[j * BODY + 8u], bodies[j * BODY + 9u], bodies[j * BODY + 10u]);
  let wj = vec3f(bodies[j * BODY + 12u], bodies[j * BODY + 13u], bodies[j * BODY + 14u]);
  let vapp = max(-dot((vi + cross(wi, ra)) - (vj + cross(wj, rb)), n), 0.0);
  Cv = min(Cv, (P.maxdep + vapp) * P.h) * P.relax;
  let ta = cross(ra, -n); let tb = cross(rb, n);
  let Ii = inv_inertia(i); let Ij = inv_inertia(j);
  let w = body_inv_mass(i) + dot(ta, Ii * ta) + body_inv_mass(j) + dot(tb, Ij * tb);
  let dl = -Cv / (w + P.alpha_h2);
  patch_corr[4u * p] = vec4f(-n * body_inv_mass(i) * dl, 1.0);
  patch_corr[4u * p + 1u] = vec4f(Ii * ta * dl, 0.0);
  patch_corr[4u * p + 2u] = vec4f(n * body_inv_mass(j) * dl, 1.0);
  patch_corr[4u * p + 3u] = vec4f(Ij * tb * dl, 0.0);
}


// ---- (c + d1 fused) one workgroup per patch: its 64 threads stride over the patch's pairs, a shared-memory tree
// reduction sums the 16 values, thread 0 solves the patch. The sums' order differs from patch_reduce (f32 rounding).
var<workgroup> red: array<vec4f, 256>;

@compute @workgroup_size(64)
fn patch_rs(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let p = wid.y * nwg.x + wid.x;
  if (p >= P.count) { return; }   // uniform: the whole workgroup returns
  var v0 = vec4f(0.0); var v1 = vec4f(0.0); var v2 = vec4f(0.0); var v3 = vec4f(0.0);
  for (var t = patch_starts[p] + li; t < patch_starts[p + 1u]; t += 64u) {
    let k = patch_of[t];
    let o = pair_out[2u * k]; let pc = pair_out[2u * k + 1u];
    let O = o.x;
    if (O == 0.0) { continue; }
    let g = o.yzw; let gl = length(g);
    var gu = vec3f(0.0);
    if (gl > 0.0) { gu = O * g / gl; }
    let ia = pair_cov_inst[k].x; let ib = pair_cov_inst[k].y;
    let s0 = O * (wcov[2u * ia] + wcov[2u * ib]);
    let s1 = O * (wcov[2u * ia + 1u].xy + wcov[2u * ib + 1u].xy);
    let xc = O * pc.xyz;
    v0 += vec4f(O, gu); v1 += vec4f(g, xc.x); v2 += vec4f(xc.yz, s0.xy); v3 += vec4f(s0.zw, s1);
  }
  red[4u * li] = v0; red[4u * li + 1u] = v1; red[4u * li + 2u] = v2; red[4u * li + 3u] = v3;
  workgroupBarrier();
  for (var h = 32u; h > 0u; h >>= 1u) {
    if (li < h) {
      for (var c = 0u; c < 4u; c++) { red[4u * li + c] += red[4u * (li + h) + c]; }
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    patch_red[4u * p] = red[0]; patch_red[4u * p + 1u] = red[1]; patch_red[4u * p + 2u] = red[2]; patch_red[4u * p + 3u] = red[3];
    solve_patch(p, red[0], red[1], red[2], red[3]);
  }
}

// ---- (d2) per body: average its patches' corrections (Jacobi, mass splitting by count) and apply them
@group(0) @binding(14) var<storage, read> body_patch_start: array<u32>;
@group(0) @binding(15) var<storage, read> body_patch: array<u32>;   // patch * 2 + side (0: body is i, 1: body is j)

@compute @workgroup_size(64)
fn body_apply(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= P.nbody || body_inv_mass(b) == 0.0) { return; }
  var lin = vec3f(0.0); var ang = vec3f(0.0); var cnt = 0.0;
  for (var t = body_patch_start[b]; t < body_patch_start[b + 1u]; t++) {
    let e = body_patch[t]; let p = e >> 1u; let side = e & 1u;
    let l = patch_corr[4u * p + 2u * side];
    lin += l.xyz; ang += patch_corr[4u * p + 2u * side + 1u].xyz; cnt += l.w;
  }
  if (cnt == 0.0) { return; }
  lin /= cnt; ang /= cnt;
  let o = b * BODY;
  bodies[o] += lin.x; bodies[o + 1u] += lin.y; bodies[o + 2u] += lin.z;
  var q = body_quat(b);
  let dq = 0.5 * vec4f(-dot(ang, q.yzw), ang * q.x + cross(ang, q.yzw));
  q = normalize(q + dq);
  bodies[o + 4u] = q.x; bodies[o + 5u] = q.y; bodies[o + 6u] = q.z; bodies[o + 7u] = q.w;
}

// ---- integrate (substep start) and velocity update (substep end), one thread per body
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= P.nbody) { return; }
  let o = b * BODY;
  bodies[o + 20u] = bodies[o]; bodies[o + 21u] = bodies[o + 1u]; bodies[o + 22u] = bodies[o + 2u];
  bodies[o + 24u] = bodies[o + 4u]; bodies[o + 25u] = bodies[o + 5u]; bodies[o + 26u] = bodies[o + 6u]; bodies[o + 27u] = bodies[o + 7u];
  if (body_inv_mass(b) == 0.0) { return; }
  bodies[o + 10u] += P.gz * P.h;
  bodies[o] += bodies[o + 8u] * P.h; bodies[o + 1u] += bodies[o + 9u] * P.h; bodies[o + 2u] += bodies[o + 10u] * P.h;
  let w = vec3f(bodies[o + 12u], bodies[o + 13u], bodies[o + 14u]);
  var q = body_quat(b);
  q = normalize(q + 0.5 * P.h * vec4f(-dot(w, q.yzw), w * q.x + cross(w, q.yzw)));
  bodies[o + 4u] = q.x; bodies[o + 5u] = q.y; bodies[o + 6u] = q.z; bodies[o + 7u] = q.w;
}

@compute @workgroup_size(64)
fn velocity(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= P.nbody || body_inv_mass(b) == 0.0) { return; }
  let o = b * BODY;
  for (var c = 0u; c < 3u; c++) { bodies[o + 8u + c] = (bodies[o + c] - bodies[o + 20u + c]) / P.h; }
  let q = body_quat(b);
  let p = vec4f(bodies[o + 24u], bodies[o + 25u], bodies[o + 26u], bodies[o + 27u]);
  // dq = q * conj(p): its vector part is sin(theta/2) * axis
  let w0 = dot(q, p);
  let v = -q.x * p.yzw + p.x * q.yzw - cross(q.yzw, p.yzw);
  let sgn = select(-1.0, 1.0, w0 >= 0.0);
  bodies[o + 12u] = 2.0 * v.x / P.h * sgn; bodies[o + 13u] = 2.0 * v.y / P.h * sgn; bodies[o + 14u] = 2.0 * v.z / P.h * sgn;
}

// ---- (e) detection, one tree level per dispatch: test each frontier entry (body pair bp, node a on i, node b on j);
// keep leaf pairs, push the children of the larger node to the next frontier. Atomic counters: [0] next, [1] leaves.
// the per-node block (slab: thin axis, half-thickness, level tag, outward sign, materials), stored after the node
// records in the nodes buffer at P.slab_off (one storage binding fewer: browsers allow 10 per stage)
fn slab_at(i: u32) -> f32 { return nodes[u32(P.slab_off) + i]; }
@group(0) @binding(17) var<storage, read> bp_info: array<vec4f>;     // i, j (as f32), margin, -
@group(0) @binding(18) var<storage, read> frontier: array<vec4u>;    // bp, a, b, -
@group(0) @binding(19) var<storage, read_write> next_frontier: array<vec4u>;
@group(0) @binding(20) var<storage, read_write> leaves: array<vec4u>;
@group(0) @binding(21) var<storage, read_write> counters: array<atomic<u32>>;

// one frontier entry f = (bp, node a on body i, node b on body j): kind 0 rejected, 1 leaf pair (leaf), 2 split into
// c0, c1 (the larger node's children)
struct Desc { kind: u32, leaf: vec4u, c0: vec4u, c1: vec4u, }

fn descend(f: vec4u) -> Desc {
  var r: Desc;
  r.kind = 0u;
  let info = bp_info[f.x];
  let i = u32(info.x); let j = u32(info.y); let margin = info.z;
  let ka = f.y; let kb = f.z;
  let si = body_scale(i); let sj = body_scale(j);
  let Ri = quat_mat(body_quat(i)); let Rj = quat_mat(body_quat(j));
  let a = Ri * (node_mean(ka) * si) + body_pos(i);
  let b = Rj * (node_mean(kb) * sj) + body_pos(j);
  var d = a - b;
  let ra = nodes[ka * NS + 10u] * si; let rb = nodes[kb * NS + 10u] * sj;
  let ua = Ri * vec3f(slab_at(ka * P.sls), slab_at(ka * P.sls + 1u), slab_at(ka * P.sls + 2u));
  let ub = Rj * vec3f(slab_at(kb * P.sls), slab_at(kb * P.sls + 1u), slab_at(kb * P.sls + 2u));
  if ((P.flags & 32u) != 0u) {
    // world.js oneSided: behind a solid-backed thin node is in contact range; drop that part of the offset (within the
    // solid held behind it, its footprint, and for surfaces that face it)
    for (var side = 0u; side < 2u; side++) {
      let k = select(kb, ka, side == 0u); let ko = select(ka, kb, side == 0u);
      let s5 = slab_at(k * P.sls + 5u);
      if (abs(s5) < 2.0 || !(slab_at(k * P.sls + 3u) <= 0.25 * nodes[k * NS + 11u])) { continue; }
      let sk = select(sj, si, side == 0u); let sko = select(si, sj, side == 0u);
      let n = select(ub, ua, side == 0u);
      let sg = sign(s5) * select(-1.0, 1.0, side == 0u);
      var fw = 1.0;
      let so5 = slab_at(ko * P.sls + 5u);
      if (so5 != 0.0 && slab_at(ko * P.sls + 3u) <= 0.25 * nodes[ko * NS + 11u]) {
        let uo = select(ua, ub, side == 0u);
        fw = -sign(so5) * sign(s5) * dot(uo, n);
      }
      let c = sg * dot(n, d);
      let reach = nodes[k * NS + 10u] * sk + nodes[ko * NS + 10u] * sko;
      if (fw > 0.0 && c > 0.0 && c <= (abs(s5) - 2.0) * sk && dot(d, d) - c * c <= reach * reach) { d -= sg * c * n; }
    }
  }
  let d0 = d;
  var M = vec3f(0.0); var MM = 0.0; var mg = margin;
  if ((P.flags & 16u) != 0u) {
    // world.js directedMargin: the closest offset on the segment this node pair's offset sweeps (each node centre's own
    // velocity v + w x r over P.pad1); spin adds only each node's own reach
    let vi = vec3f(bodies[i * BODY + 8u], bodies[i * BODY + 9u], bodies[i * BODY + 10u]);
    let wi = vec3f(bodies[i * BODY + 12u], bodies[i * BODY + 13u], bodies[i * BODY + 14u]);
    let vj = vec3f(bodies[j * BODY + 8u], bodies[j * BODY + 9u], bodies[j * BODY + 10u]);
    let wj = vec3f(bodies[j * BODY + 12u], bodies[j * BODY + 13u], bodies[j * BODY + 14u]);
    M = ((vi + cross(wi, a - body_pos(i))) - (vj + cross(wj, b - body_pos(j)))) * P.pad1;
    MM = dot(M, M);
    if (MM > 0.0) { d += clamp(-dot(d, M) / MM, 0.0, 1.0) * M; }
    mg = margin + ra * length(wi) * P.pad1 + rb * length(wj) * P.pad1;
  }
  if (length(d) > ra + rb + mg) { return r; }
  if (abs(dot(ua, d)) > slab_at(ka * P.sls + 3u) * si + rb + mg) { return r; }
  if (abs(dot(ub, d)) > slab_at(kb * P.sls + 3u) * sj + ra + mg) { return r; }
  var la = nodes[ka * NS + 13u] < 0.0; var lb = nodes[kb * NS + 13u] < 0.0;
  if ((P.flags & 2u) != 0u) {   // budget: world.detect's level cap (slab level tag <= the pair's cap)
    let cap = f32(bcap[f.x]);
    la = la || slab_at(ka * P.sls + 4u) <= cap; lb = lb || slab_at(kb * P.sls + 4u) <= cap;
  }
  if ((P.flags & 64u) != 0u && !(la && lb) && bcap[f.x] == 0u) {
    // world.js refineNeed: both thin slabs whose moments already describe their surfaces: stop here (flat faces stay
    // coarse; edges and corners refine)
    let ea = slab_at(ka * P.sls + 11u); let eb = slab_at(kb * P.sls + 11u);
    if (slab_at(ka * P.sls + 3u) <= 0.25 * nodes[ka * NS + 11u] && slab_at(kb * P.sls + 3u) <= 0.25 * nodes[kb * NS + 11u]
        && ea >= 0.0 && ea * si <= P.need_tol && eb >= 0.0 && eb * sj <= P.need_tol) { la = true; lb = true; }
  }
  if (la && lb) {
    if ((P.flags & 1u) != 0u) {
      // world.detect: Mahalanobis distance of the means pulled together by the margin, under the summed covariances
      let S = Ri * node_cov(ka) * transpose(Ri) * (si * si) + Rj * node_cov(kb) * transpose(Rj) * (sj * sj);
      var e = d;
      if (MM > 0.0) {
        // directedMargin: the Mahalanobis-closest offset on the segment (q is quadratic in t)
        let qD = maha(S, d0); let qM = maha(S, M); let qb = 0.5 * (maha(S, d0 + M) - qD - qM);
        e = d0 + select(0.0, clamp(-qb / qM, 0.0, 1.0), qM > 0.0) * M;
      }
      let dist = length(e);
      if (maha(S, e * (max(dist - mg, 0.0) / max(dist, 1e-12))) >= P.qcut) { return r; }
    }
    r.kind = 1u; r.leaf = vec4u(f.x, ka, kb, 0u); return r;
  }
  r.kind = 2u;
  if (!la && (lb || nodes[ka * NS + 11u] * si >= nodes[kb * NS + 11u] * sj)) {
    r.c0 = vec4u(f.x, u32(nodes[ka * NS + 13u]), kb, 0u);
    r.c1 = vec4u(f.x, u32(nodes[ka * NS + 14u]), kb, 0u);
  } else {
    r.c0 = vec4u(f.x, ka, u32(nodes[kb * NS + 13u]), 0u);
    r.c1 = vec4u(f.x, ka, u32(nodes[kb * NS + 14u]), 0u);
  }
  return r;
}

@group(0) @binding(50) var<storage, read> bcap: array<u32>;   // per body pair: detail level cap (0: full detail)

// d^T S^-1 d (cofactors)
fn maha(S: mat3x3f, d: vec3f) -> f32 {
  let r0 = cross(S[1], S[2]); let r1 = cross(S[2], S[0]); let r2 = cross(S[0], S[1]);
  let y = (r0 * d.x + r1 * d.y + r2 * d.z) / dot(S[0], r0);   // S symmetric: S^-1 d = (r0 d.x + r1 d.y + r2 d.z) / det
  return dot(d, y);
}

@compute @workgroup_size(64)
fn detect_level(@builtin(global_invocation_id) id: vec3u) {
  let t = id.x;
  if (t >= P.count) { return; }
  let r = descend(frontier[t]);
  if (r.kind == 1u) {
    let slot = atomicAdd(&counters[1], 1u);
    if (slot < arrayLength(&leaves)) { leaves[slot] = r.leaf; }
  } else if (r.kind == 2u) {
    let slot = atomicAdd(&counters[0], 2u);
    if (slot + 1u >= arrayLength(&next_frontier)) { return; }
    next_frontier[slot] = r.c0;
    next_frontier[slot + 1u] = r.c1;
  }
}

// ---- (e') the same descent in one submission: per level detect_prep (one thread: frontier count -> indirect dispatch
// arguments) then detect_ind (indirect), for a fixed number of levels (a bound from the trees' depths); counts stay on
// the GPU. front2 holds two halves of P.count (= cap) entries; a level reads one and appends to the other.
// dstate: [0] next count, [1] leaves, [2] this level's count, [3] read half base, [4] level, [5] overflow,
//         [6] pair-cache hits, [7] -, [8 + l] frontier size of level l (64 levels)
@group(0) @binding(22) var<storage, read_write> front2: array<vec4u>;
@group(0) @binding(23) var<storage, read_write> dstate: array<atomic<u32>>;
@group(0) @binding(24) var<storage, read_write> dargs: array<u32>;

@compute @workgroup_size(1)
fn detect_prep() {
  let cap = P.count;
  let raw = atomicLoad(&dstate[0]);
  if (raw > cap) { atomicStore(&dstate[5], 1u); }
  let n = min(raw, cap);
  atomicStore(&dstate[0], 0u);
  atomicStore(&dstate[2], n);
  atomicStore(&dstate[3], cap - atomicLoad(&dstate[3]));
  let l = atomicLoad(&dstate[4]);
  if (l < 64u) { atomicStore(&dstate[8u + l], n); }
  atomicStore(&dstate[4], l + 1u);
  let g = (n + 63u) / 64u;
  dargs[0] = min(g, 65535u); dargs[1] = (g + 65534u) / 65535u; dargs[2] = 1u;
}

@compute @workgroup_size(64)
fn detect_ind(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let t = (wid.y * nwg.x + wid.x) * 64u + li;
  if (t >= atomicLoad(&dstate[2])) { return; }
  let cap = P.count;
  let rb = atomicLoad(&dstate[3]);
  let r = descend(front2[rb + t]);
  if (r.kind == 1u) {
    let slot = atomicAdd(&dstate[1], 1u);
    if (slot < arrayLength(&leaves)) { leaves[slot] = r.leaf; }
  } else if (r.kind == 2u) {
    let slot = atomicAdd(&dstate[0], 2u);
    if (slot + 1u < cap) { front2[cap - rb + slot] = r.c0; front2[cap - rb + slot + 1u] = r.c1; }
  }
}

// ---- pair cache (world.js pair_tol, frame-to-frame reuse; detection must also work without it: cold start, misses,
// deforming bodies, LOD cut changes): per body pair the
// relative pose against the one at its last build; within tolerance its last pairs are reused (the host copies them),
// otherwise its root pair seeds the descent (bp_info's margin then includes the tolerance). P.count = body pairs.
// cache_rel: 4 per body pair: cached Rj^T Ri (column-major) 0..8, cached Rj^T (pi - pj) 9..11, build margin 12,
// this frame's margin without tolerance 13, i's radius 14 (root centre distance + reach, scaled)
@group(0) @binding(25) var<storage, read> bp_seed: array<vec4u>;   // bp, root i, root j, -
@group(0) @binding(26) var<storage, read> cache_rel: array<vec4f>;

@compute @workgroup_size(64)
fn cache_check(@builtin(global_invocation_id) id: vec3u) {
  let bp = id.x;
  if (bp >= P.count) { return; }
  let info = bp_info[bp];
  let i = u32(info.x); let j = u32(info.y);
  let Ri = quat_mat(body_quat(i)); let Rj = quat_mat(body_quat(j));
  let RR = transpose(Rj) * Ri;
  let tr = transpose(Rj) * (body_pos(i) - body_pos(j));
  let c0 = cache_rel[4u * bp]; let c1 = cache_rel[4u * bp + 1u]; let c2 = cache_rel[4u * bp + 2u]; let c3 = cache_rel[4u * bp + 3u];
  // trace(R0^T RR): elementwise product sum, both column-major
  let tc = dot(c0.xyz, RR[0]) + dot(vec3f(c0.w, c1.x, c1.y), RR[1]) + dot(vec3f(c1.z, c1.w, c2.x), RR[2]);
  let ang = acos(clamp((tc - 1.0) * 0.5, -1.0, 1.0));
  let drift = length(tr - vec3f(c2.y, c2.z, c2.w)) + ang * c3.z;
  if (drift + c3.y <= c3.x) { atomicAdd(&dstate[6], 1u); return; }
  let slot = atomicAdd(&dstate[0], 1u);
  front2[slot] = bp_seed[bp];
}

// ---- (e'') two tree levels per dispatch: an entry that splits tests its two children in the same thread and appends
// theirs (up to 4 entries), so a descent takes half the dispatches. detect_prep logs every other level's frontier.
fn emit_leaf(l: vec4u) {
  let slot = atomicAdd(&dstate[1], 1u);
  if (slot < arrayLength(&leaves)) { leaves[slot] = l; }
}

fn push_pair(c0: vec4u, c1: vec4u, wb: u32, cap: u32) {
  let slot = atomicAdd(&dstate[0], 2u);
  if (slot + 1u < cap) { front2[wb + slot] = c0; front2[wb + slot + 1u] = c1; }
}

@compute @workgroup_size(64)
fn detect_ind2(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let t = (wid.y * nwg.x + wid.x) * 64u + li;
  if (t >= atomicLoad(&dstate[2])) { return; }
  let cap = P.count;
  let rb = atomicLoad(&dstate[3]);
  let wb = cap - rb;
  let r = descend(front2[rb + t]);
  if (r.kind == 1u) { emit_leaf(r.leaf); return; }
  if (r.kind != 2u) { return; }
  let a = descend(r.c0);
  if (a.kind == 1u) { emit_leaf(a.leaf); } else if (a.kind == 2u) { push_pair(a.c0, a.c1, wb, cap); }
  let b = descend(r.c1);
  if (b.kind == 1u) { emit_leaf(b.leaf); } else if (b.kind == 2u) { push_pair(b.c0, b.c1, wb, cap); }
}

// ---- end-to-end step on the GPU: the detected pairs (plist: bp, a, b) become patches and are solved without the host.
// sstate: [0] pairs, [1] patches, [2] hash table overflow, [3] -. sargs: indirect arguments at u32 0 (one thread per
// pair), 4 (one workgroup per patch), 8 (one thread per patch).
@group(0) @binding(27) var<storage, read> plist: array<vec4u>;
@group(0) @binding(28) var<storage, read_write> sstate: array<atomic<u32>>;
@group(0) @binding(29) var<storage, read_write> sargs: array<u32>;
@group(0) @binding(30) var<storage, read_write> pair_out4: array<vec4f>;   // PO per pair: O g | p r | S xx xy xz yy | S yz zz EO lev | gu
                                                                            // (g: the raw gradient sum, gu: the O-weighted push direction)
@group(0) @binding(31) var<storage, read_write> ptab: array<atomic<u32>>;   // patch key + 1 per slot (0: empty)
@group(0) @binding(32) var<storage, read_write> pcnt: array<atomic<u32>>;   // pairs per slot, then the slot's fill cursor
@group(0) @binding(33) var<storage, read_write> pair_slot: array<u32>;
@group(0) @binding(34) var<storage, read_write> bcnt: array<atomic<u32>>;   // patches per body, then its fill cursor
@group(0) @binding(35) var<storage, read_write> patch_starts_w: array<u32>;
@group(0) @binding(36) var<storage, read_write> patch_of_w: array<u32>;
@group(0) @binding(37) var<storage, read_write> patch_bodies_w: array<vec2u>;
@group(0) @binding(38) var<storage, read_write> body_patch_start_w: array<u32>;
@group(0) @binding(39) var<storage, read_write> body_patch_w: array<u32>;
@group(0) @binding(40) var<storage, read_write> pslot_patch: array<u32>;   // per occupied slot its patch id

fn args_for(n: u32, o: u32) {
  let g = (n + 63u) / 64u;
  sargs[o] = min(g, 65535u); sargs[o + 1u] = (g + 65534u) / 65535u; sargs[o + 2u] = 1u;
}

// one thread: the pair count (clamped to the list) and the per-pair dispatch arguments
@compute @workgroup_size(1)
fn pair_args() {
  let n = min(atomicLoad(&sstate[0]), arrayLength(&plist));
  atomicStore(&sstate[0], n);
  args_for(n, 0u);
}

fn tid(wid: vec3u, nwg: vec3u, li: u32) -> u32 { return (wid.y * nwg.x + wid.x) * 64u + li; }

// (a + b fused) per pair: both nodes' world mean and covariance from the body poses, then the overlap; the pair's
// summed covariance goes along for the patch sums (no per-instance pass)
@compute @workgroup_size(64)
fn overlap_xf(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let t = tid(wid, nwg, li);
  if (t >= atomicLoad(&sstate[0])) { return; }
  let f = plist[t];
  let info = bp_info[f.x];
  let i = u32(info.x); let j = u32(info.y);
  let si = body_scale(i); let sj = body_scale(j);
  let Ri = quat_mat(body_quat(i)); let Rj = quat_mat(body_quat(j));
  let ma = Ri * (node_mean(f.y) * si) + body_pos(i);
  let mb = Rj * (node_mean(f.z) * sj) + body_pos(j);
  let Ca = Ri * node_cov(f.y) * transpose(Ri) * (si * si);
  let Cb = Rj * node_cov(f.z) * transpose(Rj) * (sj * sj);
  let S = Ca + Cb;
  let o = PO * t;
  pair_out4[o + 2u] = vec4f(S[0].x, S[1].x, S[2].x, S[1].y);
  let r0 = cross(S[1], S[2]); let r1 = cross(S[2], S[0]); let r2 = cross(S[0], S[1]);
  let det = dot(S[0], r0);
  // [3].zw: the pair's node-offset term O (off_a s_i + off_b s_j) and both nodes' detail level tags (a 128 + b)
  let lev = f32(min(u32(slab_at(f.y * P.sls + 4u)), 127u) * 128u + min(u32(slab_at(f.z * P.sls + 4u)), 127u));
  let off = nodes[f.y * NS + 15u] * si + nodes[f.z * NS + 15u] * sj;
  // world.js oneSided 'soft': a partner behind a solid-backed thin sheet is evaluated on the sheet (its distance behind
  // as depth, pushing along the sheet normal) with weight w = the facing weight, and as itself with 1 - w
  var behind = 0.0; var bw = 0.0; var shift = vec3f(0.0); var bg = vec3f(0.0);
  if ((P.flags & 32u) != 0u) {
    let ua = Ri * vec3f(slab_at(f.y * P.sls), slab_at(f.y * P.sls + 1u), slab_at(f.y * P.sls + 2u));
    let ub = Rj * vec3f(slab_at(f.z * P.sls), slab_at(f.z * P.sls + 1u), slab_at(f.z * P.sls + 2u));
    let s5a = slab_at(f.y * P.sls + 5u); let s5b = slab_at(f.z * P.sls + 5u);
    let tha = slab_at(f.y * P.sls + 3u) <= 0.25 * nodes[f.y * NS + 11u]; let thb = slab_at(f.z * P.sls + 3u) <= 0.25 * nodes[f.z * NS + 11u];
    if (abs(s5a) >= 2.0 && tha) {
      let n = sign(s5a) * ua;
      let sd = dot(n, mb - ma); let lat2 = dot(mb - ma, mb - ma) - sd * sd;
      var fw = 1.0;
      if (s5b != 0.0 && thb) { fw = max(0.0, -sign(s5b) * dot(ub, n)); }
      if (sd < 0.0 && fw > 0.0 && -sd <= (abs(s5a) - 2.0) * si && lat2 <= (nodes[f.y * NS + 10u] * si) * (nodes[f.y * NS + 10u] * si)) {
        bw = min(1.0, fw); shift = sd * n; behind = -sd; bg = n;
      }
    }
    if (behind == 0.0 && abs(s5b) >= 2.0 && thb) {
      let n = sign(s5b) * ub;
      let sd = dot(n, ma - mb); let lat2 = dot(ma - mb, ma - mb) - sd * sd;
      var fw = 1.0;
      if (s5a != 0.0 && tha) { fw = max(0.0, -sign(s5a) * dot(ua, n)); }
      if (sd < 0.0 && fw > 0.0 && -sd <= (abs(s5b) - 2.0) * sj && lat2 <= (nodes[f.z * NS + 10u] * sj) * (nodes[f.z * NS + 10u] * sj)) {
        bw = min(1.0, fw); shift = -sd * n; behind = -sd; bg = -n;
      }
    }
  }
  var Ot = 0.0; var gA = vec3f(0.0); var gu = vec3f(0.0); var xc = vec3f(0.0); var EOt = 0.0;
  let nev = select(1u, 2u, behind > 0.0 && bw < 1.0);
  for (var ev = 0u; ev < nev; ev++) {
    let one = behind > 0.0 && ev == 0u;
    let wt = select(1.0, select(1.0 - bw, bw, one), behind > 0.0);
    let d = ma + select(vec3f(0.0), shift, one) - mb;
    let y = (r0 * d.x + r1 * d.y + r2 * d.z) / det;
    let q = dot(d, y);
    if (q > 40.0) { continue; }
    let r = exp(-0.5 * q);
    let O = nodes[f.y * NS + 9u] * si * si * si * nodes[f.z * NS + 9u] * sj * sj * sj * r / (PI32 * sqrt(det)) * wt;
    if (O == 0.0) { continue; }
    Ot += O;
    xc += O * (mb + Cb * y);
    EOt += O * (off + select(0.0, behind, one));
    if (one) { gu += O * bg; }
    else {
      let g = -O * y;   // (O already carries wt: the raw gradient scaled as world.js)
      gA += g;
      let gl = length(g);
      if (gl > 0.0) { gu += O * g / gl; }
    }
  }
  // [1].w and [4].w: the pair's node ids b and a (bitcast; gpu_step.wgsl pair_nodes: patch_step binds no pair list)
  let ida = bitcast<f32>(f.y); let idb = bitcast<f32>(f.z);
  if (Ot == 0.0) { pair_out4[o] = vec4f(0.0); pair_out4[o + 1u] = vec4f(0.0, 0.0, 0.0, idb); pair_out4[o + 3u] = vec4f(S[2].y, S[2].z, 0.0, lev); pair_out4[o + 4u] = vec4f(0.0, 0.0, 0.0, ida); return; }
  pair_out4[o] = vec4f(Ot, gA);
  pair_out4[o + 1u] = vec4f(xc / Ot, idb);
  pair_out4[o + 3u] = vec4f(S[2].y, S[2].z, EOt, lev);
  pair_out4[o + 4u] = vec4f(gu, ida);
}

// world.makePatches' key: the body pair and the rounded rest-frame cell (P.patch_size) of the node on the body with the
// larger inverse mass (j on ties), 6 bits per axis (unique while a body spans < 64 cells)
fn patch_key(t: u32) -> u32 {
  let f = plist[t];
  let info = bp_info[f.x];
  let i = u32(info.x); let j = u32(info.y);
  let useB = body_inv_mass(j) >= body_inv_mass(i);
  let k = select(f.y, f.z, useB);
  let sc = select(body_scale(i), body_scale(j), useB);
  let c = vec3u(bitcast<vec3u>(vec3i(floor(node_mean(k) * sc / P.patch_size + 0.5))) & vec3u(63u));
  return (f.x << 18u) | (c.x << 12u) | (c.y << 6u) | c.z;
}

// per pair: insert its patch key (open addressing), count the slot's pairs
@compute @workgroup_size(64)
fn patch_insert(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let t = tid(wid, nwg, li);
  if (t >= atomicLoad(&sstate[0])) { return; }
  let key = patch_key(t) + 1u;
  let T = arrayLength(&ptab);
  var h = ((key * 0x9E3779B1u) >> 7u) & (T - 1u);
  for (var n = 0u; n < 4u * T; n++) {
    let prev = atomicCompareExchangeWeak(&ptab[h], 0u, key);
    if (prev.exchanged || prev.old_value == key) {
      atomicAdd(&pcnt[h], 1u);
      pair_slot[t] = h;
      return;
    }
    if (prev.old_value != 0u) { h = (h + 1u) & (T - 1u); }   // occupied by another key; 0: a spurious failure, retry
  }
  atomicStore(&sstate[2], 1u);
  pair_slot[t] = 0xffffffffu;
}

var<workgroup> scan2: array<vec2u, 256>;

// one workgroup: exclusive scan over the table's slots of (occupied, pairs): patch ids and starts; per patch its bodies
// and per body its patch count; the patch count and the per-patch dispatch arguments
@compute @workgroup_size(256)
fn patch_scan(@builtin(local_invocation_index) li: u32) {
  let T = arrayLength(&ptab);
  let per = (T + 255u) / 256u;
  let lo = li * per; let hi = min(lo + per, T);
  var acc = vec2u(0u);
  for (var h = lo; h < hi; h++) {
    let c = atomicLoad(&pcnt[h]);
    if (c > 0u) { acc += vec2u(1u, c); }
  }
  scan2[li] = acc;
  workgroupBarrier();
  for (var d = 1u; d < 256u; d <<= 1u) {
    var add = vec2u(0u);
    if (li >= d) { add = scan2[li - d]; }
    workgroupBarrier();
    scan2[li] += add;
    workgroupBarrier();
  }
  var base = scan2[li] - acc;   // exclusive
  for (var h = lo; h < hi; h++) {
    let c = atomicLoad(&pcnt[h]);
    if (c == 0u) { continue; }
    let pid = base.x;
    pslot_patch[h] = pid;
    patch_starts_w[pid] = base.y;
    atomicStore(&pcnt[h], base.y);   // the slot's fill cursor
    let info = bp_info[(atomicLoad(&ptab[h]) - 1u) >> 18u];
    let i = u32(info.x); let j = u32(info.y);
    patch_bodies_w[pid] = vec2u(i, j);
    atomicAdd(&bcnt[i], 1u); atomicAdd(&bcnt[j], 1u);
    base += vec2u(1u, c);
  }
  if (li == 255u) {
    let tot = scan2[255];
    patch_starts_w[tot.x] = tot.y;
    atomicStore(&sstate[1], tot.x);
    sargs[4] = min(tot.x, 65535u); sargs[5] = (tot.x + 65534u) / 65535u; sargs[6] = 1u;   // one workgroup per patch
    args_for(tot.x, 8u);
  }
}

// per pair: its place in its patch's run
@compute @workgroup_size(64)
fn patch_fill(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let t = tid(wid, nwg, li);
  if (t >= atomicLoad(&sstate[0])) { return; }
  let h = pair_slot[t];
  if (h == 0xffffffffu) { return; }
  patch_of_w[atomicAdd(&pcnt[h], 1u)] = t;
}

// one workgroup: exclusive scan of the patches per body (P.nbody bodies) into body_patch_start and fill cursors
@compute @workgroup_size(256)
fn body_scan(@builtin(local_invocation_index) li: u32) {
  let B = P.nbody;
  let per = (B + 255u) / 256u;
  let lo = li * per; let hi = min(lo + per, B);
  var acc = 0u;
  for (var b = lo; b < hi; b++) { acc += atomicLoad(&bcnt[b]); }
  scan2[li] = vec2u(acc, 0u);
  workgroupBarrier();
  for (var d = 1u; d < 256u; d <<= 1u) {
    var add = vec2u(0u);
    if (li >= d) { add = scan2[li - d]; }
    workgroupBarrier();
    scan2[li] += add;
    workgroupBarrier();
  }
  var base = scan2[li].x - acc;
  for (var b = lo; b < hi; b++) {
    let c = atomicLoad(&bcnt[b]);
    body_patch_start_w[b] = base;
    atomicStore(&bcnt[b], base);
    base += c;
  }
  if (li == 255u) { body_patch_start_w[B] = scan2[255].x; }
}

// per patch: append (patch, side) to both bodies' lists
@compute @workgroup_size(64)
fn body_fill(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let p = tid(wid, nwg, li);
  if (p >= atomicLoad(&sstate[1])) { return; }
  let ij = patch_bodies_w[p];
  body_patch_w[atomicAdd(&bcnt[ij.x], 1u)] = 2u * p;
  body_patch_w[atomicAdd(&bcnt[ij.y], 1u)] = 2u * p + 1u;
}

var<workgroup> wcount: u32;

// (c + d1 fused, end to end) as patch_rs, with the patch count on the GPU and the pair's summed covariance from
// overlap_xf
@compute @workgroup_size(64)
fn patch_rs2(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) li: u32) {
  let p = wid.y * nwg.x + wid.x;
  if (li == 0u) { wcount = atomicLoad(&sstate[1]); }
  if (p >= workgroupUniformLoad(&wcount)) { return; }   // uniform: one patch per workgroup
  var v0 = vec4f(0.0); var v1 = vec4f(0.0); var v2 = vec4f(0.0); var v3 = vec4f(0.0);
  for (var t = patch_starts[p] + li; t < patch_starts[p + 1u]; t += 64u) {
    let k = patch_of[t];
    let o = pair_out4[PO * k]; let pc = pair_out4[PO * k + 1u];
    let O = o.x;
    if (O == 0.0) { continue; }
    let g = o.yzw; let gl = length(g);
    var gu = vec3f(0.0);
    if (gl > 0.0) { gu = O * g / gl; }
    let s0 = O * pair_out4[PO * k + 2u];
    let s1 = O * pair_out4[PO * k + 3u].xy;
    let xc = O * pc.xyz;
    v0 += vec4f(O, gu); v1 += vec4f(g, xc.x); v2 += vec4f(xc.yz, s0.xy); v3 += vec4f(s0.zw, s1);
  }
  red[4u * li] = v0; red[4u * li + 1u] = v1; red[4u * li + 2u] = v2; red[4u * li + 3u] = v3;
  workgroupBarrier();
  for (var h = 32u; h > 0u; h >>= 1u) {
    if (li < h) {
      for (var c = 0u; c < 4u; c++) { red[4u * li + c] += red[4u * (li + h) + c]; }
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    patch_red[4u * p] = red[0]; patch_red[4u * p + 1u] = red[1]; patch_red[4u * p + 2u] = red[2]; patch_red[4u * p + 3u] = red[3];
    solve_patch(p, red[0], red[1], red[2], red[3]);
  }
}

// (d2, pjacobi) per body: per body pair (the other body of each patch) the mean of its patches' corrections, summed
// over the body pairs (world.js solver 'pjacobi'). Up to 8 body pairs per body tracked; more share the last slot.
@compute @workgroup_size(64)
fn body_apply_pj(@builtin(global_invocation_id) id: vec3u) {
  let b = id.x;
  if (b >= P.nbody || body_inv_mass(b) == 0.0) { return; }
  var other: array<u32, 8>; var lin: array<vec3f, 8>; var ang: array<vec3f, 8>; var cnt: array<f32, 8>;
  var used = 0u;
  for (var t = body_patch_start[b]; t < body_patch_start[b + 1u]; t++) {
    let e = body_patch[t]; let p = e >> 1u; let side = e & 1u;
    let l = patch_corr[4u * p + 2u * side];
    if (l.w == 0.0) { continue; }   // no correction from this patch
    let ij = patch_bodies[p];
    let o = select(ij.y, ij.x, side == 1u);
    var s = 0u;
    loop {
      if (s == used) { if (used < 8u) { other[s] = o; lin[s] = vec3f(0.0); ang[s] = vec3f(0.0); cnt[s] = 0.0; used++; } else { s = 7u; } break; }
      if (other[s] == o) { break; }
      s++;
    }
    lin[s] += l.xyz; ang[s] += patch_corr[4u * p + 2u * side + 1u].xyz; cnt[s] += 1.0;
  }
  if (used == 0u) { return; }
  var dl = vec3f(0.0); var da = vec3f(0.0);
  for (var s = 0u; s < used; s++) { dl += lin[s] / cnt[s]; da += ang[s] / cnt[s]; }
  let o = b * BODY;
  bodies[o] += dl.x; bodies[o + 1u] += dl.y; bodies[o + 2u] += dl.z;
  var q = body_quat(b);
  q = normalize(q + 0.5 * vec4f(-dot(da, q.yzw), da * q.x + cross(da, q.yzw)));
  bodies[o + 4u] = q.x; bodies[o + 5u] = q.y; bodies[o + 6u] = q.z; bodies[o + 7u] = q.w;
}
